#!/usr/bin/env python3
"""[pi05-keyframe] Phase 8 Step 3：Tier B 关键帧"歧义度"打分（数据驱动）。

定义：某帧的当前观测（图像+state）若能在**其它 episode** 找到相似近邻，而近邻的**动作 chunk 明显不同**，
则该帧无法仅凭当前观测决定动作 → 必须靠历史判别 → 关键帧（w>1）。

流程：
  1. 读 mem6 的 1 Hz 侧车帧（当前帧 = sidecar 第 k 帧 ↔ local frame 25k）+ 数据集 state/action；
  2. 图像编码（ResNet-18 ImageNet，CPU，512-d）→ 跨 episode、同任务 kNN（图像距离 + state 距离）；
  3. 近邻动作 chunk 分歧度 sᵢ（前 K 步动作，按维度标准化后的平均 L1）；
  4. wᵢ = clip(1 + α·sᵢ/s_ref, 1, W_MAX) → 输出与 v1 同格式的 keyframe_weights_v2.json + 诊断报告。

用法：
  .venv/bin/python scripts/pi05/build_keyframe_weights_tier_b.py \
      --out robodojo/data/history_sidecar/keyframe_weights_v2.json \
      --diag logs/pi05/keyframe_tierb_diag.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SIDECAR = ROOT / "robodojo/data/history_sidecar"
DATASET = ROOT / "robodojo/data/hf_lerobot_home/RoboDojo_sim_arx-x5_v30_history"


def load_index() -> dict:
    return json.loads((SIDECAR / "index.json").read_text(encoding="utf-8"))


def load_mem6() -> list[int]:
    return [int(e) for e in json.loads((SIDECAR / "cotrain_mix_episodes.json").read_text())["mem6_episodes"]]


def load_dataset_arrays(mem6: set[int]) -> dict[int, dict]:
    """读 parquet（只保留 mem6 行）：episode → {frame_index, state(14), action(14)}。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out: dict[int, dict] = {}
    for f in sorted(glob.glob(str(DATASET / "data/chunk-000/*.parquet"))):
        t0 = time.time()
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state", "action"])
        ep = np.asarray(t["episode_index"].combine_chunks())
        keep = np.isin(ep, np.fromiter(mem6, dtype=np.int64, count=len(mem6)))
        idx = np.nonzero(keep)[0]
        frame = np.asarray(t["frame_index"].combine_chunks())[idx]

        def _to_np(col, dim):
            c = col.combine_chunks()
            if pa.types.is_fixed_size_list(c.type):
                return np.asarray(c.flatten()).reshape(-1, dim)[idx]
            vals = c.to_pylist()
            return np.stack([np.asarray(vals[i], dtype=np.float64) for i in idx])

        state = _to_np(t["observation.state"], 14)
        action = _to_np(t["action"], 14)
        for e in np.unique(ep[idx]):
            m = ep[idx] == e
            rec = out.get(int(e))
            if rec is None:
                out[int(e)] = {"frame_index": frame[m], "state": state[m], "action": action[m]}
            else:  # 跨 parquet 分片的 episode：拼接
                out[int(e)] = {
                    "frame_index": np.concatenate([rec["frame_index"], frame[m]]),
                    "state": np.concatenate([rec["state"], state[m]]),
                    "action": np.concatenate([rec["action"], action[m]]),
                }
        print(f"[tierb] {pathlib.Path(f).name}: kept {idx.size} rows / {np.unique(ep[idx]).size} eps "
              f"({time.time() - t0:.0f}s)", flush=True)
    return out


def encode_images(images: np.ndarray, batch: int = 256) -> np.ndarray:
    """(N,3,128,128) uint8 → (N,512) float32（ResNet-18 ImageNet，L2 归一化）。"""
    import torch
    import torchvision
    from torchvision.models import ResNet18_Weights

    try:
        weights = ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=weights)
    except Exception as exc:  # 断网兜底：随机权重（仅相对相似度，效果弱）
        print(f"[tierb] 警告：ResNet18 权重下载失败（{exc}），回退随机初始化", flush=True)
        net = torchvision.models.resnet18(weights=None)
    net.fc = torch.nn.Identity()
    net.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    emb = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch):
            x = torch.from_numpy(images[i : i + batch].astype(np.float32) / 255.0)
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - mean) / std
            emb.append(net(x).numpy())
    e = np.concatenate(emb).astype(np.float32)
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
    return e


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--knn", type=int, default=8, help="每个 slot 的近邻数")
    p.add_argument("--act-steps", type=int, default=8, help="动作 chunk 比较使用的前 K 步")
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--w-max", type=float, default=8.0)
    p.add_argument("--state-weight", type=float, default=0.5, help="state 距离在联合距离中的权重")
    p.add_argument("--emb-cache", type=pathlib.Path, default=ROOT / "logs/pi05/keyframe_tierb_emb.npz")
    p.add_argument("--out", type=pathlib.Path, default=SIDECAR / "keyframe_weights_v2.json")
    p.add_argument("--diag", type=pathlib.Path, default=ROOT / "logs/pi05/keyframe_tierb_diag.json")
    args = p.parse_args()

    index = load_index()
    mem6 = load_mem6()
    tasks: dict[str, list[int]] = collections.defaultdict(list)
    for e in mem6:
        tasks[index[str(e)]["task"]].append(e)

    # ---- 1) 载入侧车帧 + 数据集 state/action ----
    t0 = time.time()
    data = load_dataset_arrays(set(mem6))
    print(f"[tierb] 数据集载入完成：{len(data)} 集（{time.time() - t0:.0f}s）", flush=True)

    per_task: dict[str, dict] = {}
    for task, eps in sorted(tasks.items()):
        imgs, states, acts, ep_ids, slots = [], [], [], [], []
        for e in eps:
            entry = index[str(e)]
            npz = np.load(SIDECAR / entry["path"])
            n = int(npz.shape[0])
            rec = data.get(e)
            if rec is None:
                continue
            for k in range(n):
                f = k * 25
                j = int(np.searchsorted(rec["frame_index"], f))
                if j >= rec["frame_index"].size or int(rec["frame_index"][j]) != f:
                    continue
                imgs.append(npz[k])
                states.append(rec["state"][j])
                acts.append(rec["action"][j : min(j + args.act_steps, rec["action"].shape[0])])
                ep_ids.append(e)
                slots.append(k)
        imgs_a = np.asarray(imgs)  # (N,3,128,128) uint8
        states_a = np.asarray(states, dtype=np.float64)  # (N,14)
        ep_a = np.asarray(ep_ids)
        slots_a = np.asarray(slots)
        acts_a = np.stack([np.pad(a, ((0, args.act_steps - a.shape[0]), (0, 0))) for a in acts])
        print(f"[tierb] {task}: slots={len(ep_a)} eps={len(np.unique(ep_a))}", flush=True)
        per_task[task] = {"imgs": imgs_a, "states": states_a, "acts": acts_a, "ep": ep_a, "slots": slots_a}

    # ---- 2) 图像编码（带缓存）----
    if args.emb_cache.exists():
        cache = np.load(args.emb_cache)
        print(f"[tierb] 复用编码缓存 {args.emb_cache}")
    else:
        cache = {}
    for task, d in per_task.items():
        key = f"{task}__emb"
        if key in cache:
            d["emb"] = cache[key]
        else:
            d["emb"] = encode_images(d["imgs"])
            print(f"[tierb] {task}: 图像编码完成 {d['emb'].shape}", flush=True)
    np.savez_compressed(args.emb_cache, **{f"{t}__emb": d["emb"] for t, d in per_task.items()})

    # ---- 3) 每任务：kNN + 动作分歧度 ----
    diag: dict[str, dict] = {}
    weights: dict[str, list[float]] = {}
    for task, d in sorted(per_task.items()):
        emb, states, acts, ep = d["emb"], d["states"], d["acts"].astype(np.float64), d["ep"]
        n = emb.shape[0]
        s_std = states.std(axis=0)
        s_std[s_std < 1e-6] = 1.0
        st = (states - states.mean(axis=0)) / s_std
        a_flat = acts.reshape(n, -1)
        a_std = a_flat.std(axis=0)
        a_std[a_std < 1e-6] = 1.0
        af = (a_flat / a_std).astype(np.float32)
        # 距离矩阵（全部 float32，O(N²) 内存；动作距离用 L2 以便走 BLAS）
        s_norm = np.linalg.norm(st, axis=1)
        img_dist = (1.0 - emb @ emb.T).astype(np.float32)
        state_dist = (
            np.sqrt(np.maximum(s_norm[:, None] ** 2 + s_norm[None, :] ** 2 - 2 * st @ st.T, 0.0))
            / np.sqrt(st.shape[1])
        ).astype(np.float32)
        state_ref = float(np.median(state_dist))
        joint = img_dist + np.float32(args.state_weight) * (state_dist / max(state_ref, 1e-6))
        np.fill_diagonal(joint, np.inf)
        # 排除同 episode
        joint[ep[:, None] == ep[None, :]] = np.inf
        if n <= args.knn:
            s_score = np.zeros(n)
        else:
            nn = np.argpartition(joint, args.knn, axis=1)[:, : args.knn]
            af_norm = (af**2).sum(axis=1)
            act_d2 = np.maximum(af_norm[:, None] + af_norm[None, :] - 2.0 * (af @ af.T), 0.0)
            act_dist = np.sqrt(act_d2 / af.shape[1])  # 标准化动作空间的 RMSE
            s_score = act_dist[np.arange(n)[:, None], nn].mean(axis=1)
        diag[task] = {
            "n_slots": int(n),
            "s_quantiles": {q: float(np.quantile(s_score, float(q))) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
            "s_mean": float(s_score.mean()),
        }
        s_ref = float(np.quantile(s_score, 0.9))
        w = np.clip(1.0 + args.alpha * s_score / max(s_ref, 1e-9), 1.0, args.w_max)
        diag[task]["w_gt1_frac"] = float(np.mean(w > 1.0))
        diag[task]["w_mean"] = float(w.mean())
        # 回填到 episode → slot 数组
        for e in np.unique(ep):
            m = ep == e
            entry = index[str(int(e))]
            arr = [1.0] * int(entry["num_samples"])
            for sl, wv in zip(d["slots"][m], w[m]):
                if 0 <= int(sl) < len(arr):
                    arr[int(sl)] = float(wv)
            weights[str(int(e))] = arr
        print(f"[tierb] {task}: s p50={diag[task]['s_quantiles'][0.5]:.4f} p90={s_ref:.4f} "
              f"w>1 占比={diag[task]['w_gt1_frac']:.1%} 平均权重={diag[task]['w_mean']:.2f}", flush=True)

    # ---- 4) 与 v1 的对照 + 写出 ----
    v1_path = SIDECAR / "keyframe_weights_v1.json"
    overlap = None
    if v1_path.exists():
        v1 = json.loads(v1_path.read_text())
        hit = tot = 0
        for e, arr in weights.items():
            v1_arr = v1["weights"].get(e)
            if v1_arr is None:
                continue
            a2 = np.asarray(arr)
            thresh = np.quantile(a2, 0.8)
            top = a2 >= thresh
            v1_key = np.asarray(v1_arr) > 1.0
            hit += int(np.sum(top & v1_key))
            tot += int(np.sum(top))
        overlap = hit / max(tot, 1)
        print(f"[tierb] v2 top-20% 帧落在 v1 关键段内的比例 = {overlap:.1%}")

    all_w = np.concatenate([np.asarray(v) for v in weights.values()])
    payload = {
        "meta": {
            "tier": "B",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "knn": args.knn, "act_steps": args.act_steps, "alpha": args.alpha, "w_max": args.w_max,
            "state_weight": args.state_weight,
            "note": "歧义度打分：同任务跨集 kNN 相似观测 + 动作 chunk 分歧度。",
        },
        "interval": 25,
        "weights": weights,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False))
    diag["_global"] = {
        "episodes": len(weights), "slots": int(all_w.size),
        "w_gt1_frac": float(np.mean(all_w > 1.0)), "w_mean": float(all_w.mean()),
        "v1_overlap_top20": overlap,
    }
    args.diag.parent.mkdir(parents=True, exist_ok=True)
    args.diag.write_text(json.dumps(diag, ensure_ascii=False, indent=2))
    print(f"\n=== Tier B 汇总 ===")
    print(f"  集数={len(weights)} 槽数={all_w.size} w>1 占比={np.mean(all_w > 1.0):.1%} 平均权重={all_w.mean():.2f}"
          f" v1 重合度={overlap if overlap is None else f'{overlap:.1%}'}")
    print(f"[tierb] 已写出 {args.out} 与 {args.diag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
