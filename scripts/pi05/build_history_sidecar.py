#!/usr/bin/env python3
"""Phase 4：构建 RoboDojo 历史帧 sidecar（cam_high，1 Hz，128×128，uint8 CHW）。

用法（venv 已含 pyav/jax/pandas）:
  PY=robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
  $PY scripts/pi05/build_history_sidecar.py --dry-run --tasks mem6      # 只打印计划
  $PY scripts/pi05/build_history_sidecar.py --tasks mem6 --limit-files 2  # 试点
  $PY scripts/pi05/build_history_sidecar.py --tasks mem6 --workers 8      # 全量

输出（默认根：robodojo/data/history_sidecar/）:
  <task>/ep_<episode_index:07d>.npy   # (N,3,128,128) uint8 RGB，element k ↔ episode 内 local frame 25k
  index.json                          # episode_index(str) → {path, task, num_samples, length}
  meta.json                           # 采样约定（fps/interval/size/cam/resize/口径说明）

与评测侧口径对齐（XPolicyLab model.py `_history_entries_for`）:
  * 历史帧 = episode 内每 25 帧（可配）取 1 帧，即 1 Hz@25fps；
  * 训练 transform 取 k ≤ f//interval − 1（严格早于当前帧、排除“过新一帧”）；
  * 帧预处理 resize_with_pad（黑边），与 openpi `preprocess_observation` 保持一致；
  * 帧只在同一 episode 内取，天然不跨集。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OPENPI_SRC = ROOT / "robodojo/XPolicyLab/policy/Pi_05/openpi/src"
DEFAULT_REPO = ROOT / "robodojo/data/RoboDojo_lerobot_v30_video"
DEFAULT_OUT = ROOT / "robodojo/data/history_sidecar"

# memory 6 任务：task_index → 规范短名（已按本地数据逐任务核对：各 100 集）
MEM6 = {
    3: ("cover_blocks", "Cover the blocks"),
    25: ("press_by_number", "Press the two red buttons"),
    16: ("match_and_pick_from_conveyor", "Remember the first object on the conveyor"),
    32: ("swap_T", "swap their positions"),
    11: ("imitate_sorting_sequence", "Observe the object placement order"),
    33: ("swap_blocks", "Swap the two blocks using the empty mat"),
}


def _slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return slug[:max_len] or "task"


def _resize_with_pad_cpu(image: np.ndarray, size: int) -> np.ndarray:
    """在 worker 内调用 openpi 的 resize_with_pad（CPU jax），保证与预训练管线一致。"""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if str(OPENPI_SRC) not in sys.path:
        sys.path.insert(0, str(OPENPI_SRC))
    from openpi.shared.image_tools import resize_with_pad  # noqa: PLC0415

    return np.asarray(resize_with_pad(image, size, size))


def _process_file(job: dict) -> dict:
    """解码一个 cam_high mp4，按 from_ts 邻接窗口归属帧，抽取 target 集的 1 Hz 帧并保存 npy。"""
    import av  # noqa: PLC0415

    path = job["path"]
    fps = job["fps"]
    interval = job["interval"]
    size = job["size"]
    tol = 0.5 / fps
    episodes = job["episodes"]  # 该文件全部集（含非 target），按 from_ts 升序

    targets = []
    for ep in episodes:
        if ep["target"]:
            ep["n_expected"] = (ep["length"] - 1) // interval + 1
            ep["frames"] = {}
            targets.append(ep)
    last_ts_needed = max(
        (ep["from_ts"] + (((ep["length"] - 1) // interval) * interval) / fps for ep in targets),
        default=None,
    )

    t0 = time.time()
    container = av.open(path)
    stream = container.streams.video[0]
    ep_ptr = 0
    decoded = 0
    try:
        for frame in container.decode(stream):
            decoded += 1
            ts = float(frame.pts * frame.time_base) if frame.pts is not None else None
            if ts is None:
                continue
            if last_ts_needed is not None and ts > last_ts_needed + tol:
                break
            # 归属窗口 = [ep.from_ts, 下一集.from_ts)；与 LeRobot 数据行口径一致
            # （frame_index=f ↔ ts=from_ts+f/fps），不依赖 length/to_ts 的 ±1 帧误差。
            while ep_ptr + 1 < len(episodes) and ts >= episodes[ep_ptr + 1]["from_ts"] - tol:
                ep_ptr += 1
            ep = episodes[ep_ptr]
            if not ep["target"]:
                continue
            local = int(round((ts - ep["from_ts"]) * fps))
            if 0 <= local < ep["length"] and local % interval == 0:
                img = frame.to_ndarray(format="rgb24")
                ep["frames"][local // interval] = _resize_with_pad_cpu(img, size)
    finally:
        container.close()

    results = []
    for ep in targets:
        n_expected = ep["n_expected"]
        arr = np.zeros((n_expected, 3, size, size), dtype=np.uint8)
        missing = []
        for k in range(n_expected):
            frame = ep["frames"].get(k)
            if frame is None:
                missing.append(k)
            else:
                arr[k] = np.transpose(frame, (2, 0, 1))
        out_path = Path(ep["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr)
        results.append(
            {
                "episode_index": int(ep["episode_index"]),
                "task": ep["task"],
                "path": str(out_path.relative_to(job["out_root"])),
                "num_samples": int(n_expected),
                "length": int(ep["length"]),
                "missing_samples": missing,
                "mean": round(float(arr.mean()), 3) if n_expected else None,
            }
        )
    return {
        "file": path,
        "decoded_frames": decoded,
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
    }


def _load_episodes(repo: Path) -> tuple[dict, pd.DataFrame, dict[int, str]]:
    info = json.loads((repo / "meta/info.json").read_text())
    tasks = pd.read_parquet(repo / "meta/tasks.parquet")
    names = {i: str(n) for i, n in enumerate(tasks.index)}
    eps_files = sorted((repo / "meta/episodes").glob("**/*.parquet"))
    if not eps_files:
        raise FileNotFoundError(f"no episodes metadata under {repo / 'meta/episodes'}")
    frames = []
    for f in eps_files:
        df = pd.read_parquet(f)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return info, df, names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="LeRobot 数据集根目录")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT, help="sidecar 输出根目录")
    parser.add_argument("--tasks", default="mem6", help="mem6 | all | 逗号分隔的任务名或 task_index")
    parser.add_argument("--cam", default="cam_high", help="相机 key（默认 cam_high）")
    parser.add_argument("--interval", type=int, default=25, help="历史抽帧间隔（帧，默认 25 → 1 Hz@25fps）")
    parser.add_argument("--size", type=int, default=128, help="sidecar 帧边长（默认 128）")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="并行 worker 数（按视频文件）")
    parser.add_argument("--limit-files", type=int, default=0, help="只处理前 N 个文件（试点用）")
    parser.add_argument("--limit-eps", type=int, default=0, help="只处理前 N 集（试点用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写文件")
    args = parser.parse_args()

    info, df, names = _load_episodes(args.repo)
    fps = float(info["fps"])
    cam = args.cam
    vchunk = f"videos/observation.images.{cam}/chunk_index"
    vfile = f"videos/observation.images.{cam}/file_index"
    vfrom = f"videos/observation.images.{cam}/from_timestamp"
    vto = f"videos/observation.images.{cam}/to_timestamp"
    need = {"episode_index", "tasks", "length", vchunk, vfile, vfrom, vto}
    missing_cols = need - set(df.columns)
    if missing_cols:
        raise KeyError(f"episodes metadata missing columns: {sorted(missing_cols)}")

    df["task_name"] = df["tasks"].apply(lambda v: str(np.asarray(v).reshape(-1)[0]))
    df["task_index"] = df["task_name"].map({v: k for k, v in names.items()})

    # 组任务选择
    if args.tasks == "mem6":
        selected_tasks = {k: v[0] for k, v in MEM6.items()}
        for k, (_, kw) in MEM6.items():
            name = names.get(k, "")
            if kw.lower() not in name.lower():
                raise ValueError(f"task_index={k} 文本与预期不符：{name!r}（期望包含 {kw!r}）")
    elif args.tasks == "all":
        selected_tasks = {k: n for k, n in names.items()}
    else:
        selected_tasks = {}
        wanted = [w.strip() for w in args.tasks.split(",") if w.strip()]
        for w in wanted:
            if w in {v[0] for v in MEM6.values()}:
                ti = next(k for k, v in MEM6.items() if v[0] == w)
                selected_tasks[ti] = w
            elif w.isdigit() and int(w) in names:
                selected_tasks[int(w)] = _slugify(names[int(w)])
            else:
                matches = [k for k, n in names.items() if w.lower() in n.lower()]
                if not matches:
                    raise ValueError(f"未找到任务: {w!r}")
                for k in matches[:1]:
                    selected_tasks[k] = _slugify(names[k])

    sel = df[df["task_index"].isin(selected_tasks)].copy()
    sel["task_slug"] = sel["task_index"].map(selected_tasks)
    sel = sel.sort_values(["task_slug", "episode_index"]).reset_index(drop=True)
    if args.limit_eps:
        sel = sel.head(args.limit_eps)

    jobs_by_file: dict[tuple[int, int], list[dict]] = {}
    for row in sel.to_dict("records"):
        key = (int(row[vchunk]), int(row[vfile]))
        jobs_by_file.setdefault(key, []).append(
            {
                "episode_index": int(row["episode_index"]),
                "length": int(row["length"]),
                "from_ts": float(row[vfrom]),
                "to_ts": float(row[vto]),
                "task": row["task_slug"],
                "out_path": str(args.out_root / row["task_slug"] / f"ep_{int(row['episode_index']):07d}.npy"),
            }
        )
    for eps in jobs_by_file.values():
        eps.sort(key=lambda e: e["from_ts"])

    file_keys = sorted(jobs_by_file)
    if args.limit_files:
        file_keys = file_keys[: args.limit_files]
    n_eps = sum(len(jobs_by_file[k]) for k in file_keys)

    print(f"[sidecar] repo={args.repo}")
    print(f"[sidecar] fps={fps} interval={args.interval} size={args.size} cam={cam}")
    print(f"[sidecar] 任务数={len(selected_tasks)} 集数={n_eps} 视频文件数={len(file_keys)} workers={args.workers}")
    for k in file_keys:
        eps = jobs_by_file[k]
        print(f"[sidecar]   chunk={k[0]} file={k[1]:03d}: eps={len(eps)}")

    if args.dry_run:
        print("[sidecar] dry-run 结束（未写任何文件）")
        return 0

    jobs = []
    for key in file_keys:
        chunk, f = key
        targets = {e["episode_index"]: e for e in jobs_by_file[key]}
        sub = df[(df[vchunk] == chunk) & (df[vfile] == f)].sort_values(vfrom)
        eps = []
        for row in sub.to_dict("records"):
            eid = int(row["episode_index"])
            tgt = targets.get(eid)
            eps.append(
                {
                    "episode_index": eid,
                    "length": int(row["length"]),
                    "from_ts": float(row[vfrom]),
                    "to_ts": float(row[vto]),
                    "target": tgt is not None,
                    "task": tgt["task"] if tgt else None,
                    "out_path": tgt["out_path"] if tgt else None,
                }
            )
        jobs.append(
            {
                "path": str(args.repo / "videos" / f"observation.images.{cam}" / f"chunk-{chunk:03d}" / f"file-{f:03d}.mp4"),
                "fps": fps,
                "interval": args.interval,
                "size": args.size,
                "out_root": str(args.out_root),
                "episodes": eps,
            }
        )

    t_start = time.time()
    index: dict[str, dict] = {}
    n_missing_total = 0
    done_files = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_process_file, job) for job in jobs]
        for fut in as_completed(futs):
            res = fut.result()
            done_files += 1
            n_missing = sum(len(r["missing_samples"]) for r in res["results"])
            n_missing_total += n_missing
            for r in res["results"]:
                index[str(r["episode_index"])] = {
                    "path": r["path"],
                    "task": r["task"],
                    "num_samples": r["num_samples"],
                    "length": r["length"],
                    "missing_samples": r["missing_samples"],
                }
            extra = f" missing={n_missing}" if n_missing else ""
            print(
                f"[sidecar] ({done_files}/{len(jobs)}) {res['file'].split('/')[-1]} "
                f"decoded={res['decoded_frames']} eps={len(res['results'])} {res['elapsed_s']}s{extra}",
                flush=True,
            )

    # 合并已有 index（支持增量：先 mem6 再 all）
    index_path = args.out_root / "index.json"
    if index_path.exists():
        old = json.loads(index_path.read_text())
        old.update(index)
        index = old
    args.out_root.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True))
    os.replace(tmp, index_path)

    meta = {
        "cam": cam,
        "fps": fps,
        "interval_frames": args.interval,
        "history_hz": round(fps / args.interval, 4),
        "size": [args.size, args.size],
        "dtype": "uint8",
        "layout": "NCHW RGB",
        "resize": "resize_with_pad(black)",
        "episode_key": "episode_index",
        "sample_k_to_frame": f"local_frame = k * {args.interval}",
        "eval_rule": "k <= frame_index // interval - 1（严格早于当前帧；最近 20 槽，最旧在左，不足左补 pad）",
        "source_repo": str(args.repo),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = args.out_root / "meta.json"
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    os.replace(tmp, meta_path)

    elapsed = time.time() - t_start
    print(
        f"[sidecar] 完成：{len(index)} 集，缺失采样={n_missing_total}，"
        f"用时 {elapsed/60:.1f} min，index={index_path}"
    )
    return 1 if n_missing_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
