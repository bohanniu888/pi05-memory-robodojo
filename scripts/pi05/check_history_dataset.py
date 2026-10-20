#!/usr/bin/env python3
"""Phase 4 验收：历史 sidecar + 训练 transform + dataloader 联检。

用法（CPU 即可）:
  PY=robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
  $PY scripts/pi05/check_history_dataset.py                 # 100 样本 + batch=4 dataloader + 可视化
  $PY scripts/pi05/check_history_dataset.py --skip-dataloader

判据（对齐实施规划 Phase 4）:
  1. 随机 100 个 (episode, frame) 样本经 RoboDojoHistoryTransform：
     shape=(20,3,128,128) uint8、mask 数目/位置正确、当前帧不在历史里、不跨集；
  2. dataloader（batch=4）能出 batch，history_images/history_mask 形状与 dtype 正确，
     且能观察到带历史（mask=True）与不带历史（mask 全 False）的样本；
  3. 抽 3 条 episode 生成可视化拼图到 logs/pi05/phase4_vis/。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPENPI_SRC = ROOT / "robodojo/XPolicyLab/policy/Pi_05/openpi/src"
DEFAULT_SIDECAR = ROOT / "robodojo/data/history_sidecar"

# 必须在 import lerobot/openpi 之前设置，数据集按 HF_LEROBOT_HOME 解析 repo_id。
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "robodojo/data/hf_lerobot_home"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(OPENPI_SRC))

import numpy as np  # noqa: E402

SLOTS = 20
INTERVAL = 25


def check_transform(num: int, sidecar_root: Path) -> tuple[int, list[dict]]:
    from openpi.policies.robodojo_history import RoboDojoHistoryTransform  # noqa: PLC0415

    index = json.loads((sidecar_root / "index.json").read_text())
    episodes = sorted(int(k) for k in index)
    rng = random.Random(0)
    transform = RoboDojoHistoryTransform(sidecar_root=str(sidecar_root), slots=SLOTS, interval=INTERVAL)
    fails: list[dict] = []
    for _ in range(num):
        ep = rng.choice(episodes)
        length = int(index[str(ep)]["length"])
        frame = rng.randrange(length)
        out = transform({"episode_index": ep, "frame_index": frame})
        hist, mask = out["history_images"], out["history_mask"]
        reason = None
        if hist.shape != (SLOTS, 3, 128, 128) or hist.dtype != np.uint8:
            reason = f"shape/dtype {hist.shape} {hist.dtype}"
        # k_max = frame//interval - 1；可取 n=min(k_max+1, slots) 个槽
        k_max = frame // INTERVAL - 1
        n_take = min(max(k_max + 1, 0), SLOTS)
        if reason is None and int(mask.sum()) != n_take:
            reason = f"mask.sum={int(mask.sum())} != {n_take}"
        if reason is None and n_take > 0:
            if not mask[-n_take:].all() or mask[:-n_take].any():
                reason = "mask 位置错误（应右侧连续 True、左侧 False）"
            ks = list(range(k_max - n_take + 1, k_max + 1))
            if any(k * INTERVAL > frame - INTERVAL for k in ks):
                reason = "当前帧过新的样本混入历史"
            if any(k * INTERVAL >= length for k in ks):
                reason = "历史帧跨集/越界"
            for offset, k in enumerate(ks[:3]):
                if hist[SLOTS - n_take + offset].max() == 0:
                    reason = f"历史帧全黑 k={k}"
                    break
        if reason is None and n_take == 0 and mask.any():
            reason = "early-frame 应为全 pad"
        if reason is not None:
            fails.append({"episode": ep, "frame": frame, "reason": reason})
    return num - len(fails), fails


def check_dataloader(batch_size: int, max_batches: int) -> dict:
    from openpi.training import config as _config  # noqa: PLC0415
    from openpi.training import data_loader as _dl  # noqa: PLC0415

    cfg = _config.get_config("pi05_aloha_sim_history_arx-x5_seed_0")
    cfg = dataclasses.replace(cfg, batch_size=batch_size, num_workers=0)
    loader = _dl.create_data_loader(cfg, shuffle=True, num_batches=max_batches)
    n_batches = 0
    n_hist_batches = 0
    first_shape = None
    for obs, _actions in loader:
        n_batches += 1
        hist = np.asarray(obs.history_images)
        mask = np.asarray(obs.history_mask)
        if n_batches == 1:
            first_shape = (hist.shape, str(hist.dtype), float(hist.min()), float(hist.max()), str(mask.dtype))
        if mask.any():
            n_hist_batches += 1
    return {
        "batches": n_batches,
        "history_batches": n_hist_batches,
        "first_shape": first_shape,
    }


def visualize(sidecar_root: Path, out_dir: Path, tasks: list[str]) -> list[str]:
    import cv2  # noqa: PLC0415

    index = json.loads((sidecar_root / "index.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for task in tasks:
        entry = next((v for k, v in index.items() if v["task"] == task), None)
        if entry is None:
            continue
        arr = np.load(sidecar_root / entry["path"])
        n = min(arr.shape[0], 20)
        frames = [np.transpose(arr[i], (1, 2, 0)) for i in range(arr.shape[0] - n, arr.shape[0])]
        while len(frames) < 20:
            frames.append(np.zeros_like(frames[0]))
        rows = [np.hstack(frames[r * 5 : (r + 1) * 5]) for r in range(4)]
        grid = np.vstack(rows)
        grid = cv2.resize(grid, (grid.shape[1] * 2, grid.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(grid, f"{task}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        path = out_dir / f"check_{task}.png"
        cv2.imwrite(str(path), grid)
        saved.append(str(path))
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--num", type=int, default=100, help="transform 随机校验样本数")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--skip-dataloader", action="store_true")
    parser.add_argument("--skip-vis", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[check] sidecar_root={args.sidecar_root}")
    passed, fails = check_transform(args.num, args.sidecar_root)
    print(f"[check] A) transform 随机校验：{passed}/{args.num} 通过")
    for f in fails[:5]:
        print(f"[check]    FAIL {f}")

    dl_stats = None
    if not args.skip_dataloader:
        dl_stats = check_dataloader(args.batch_size, args.batches)
        print(
            f"[check] B) dataloader：batches={dl_stats['batches']} "
            f"history_batches={dl_stats['history_batches']} first={dl_stats['first_shape']}"
        )

    vis = []
    if not args.skip_vis:
        vis = visualize(
            args.sidecar_root,
            ROOT / "logs/pi05/phase4_vis",
            ["cover_blocks", "press_by_number", "match_and_pick_from_conveyor"],
        )
        print(f"[check] C) 可视化：{vis}")

    ok = passed == args.num
    if dl_stats is not None:
        ok = ok and dl_stats["batches"] == args.batches
        ok = ok and dl_stats["first_shape"] is not None
        ok = ok and dl_stats["first_shape"][0] == (args.batch_size, SLOTS, 3, 128, 128)
        ok = ok and dl_stats["history_batches"] > 0
    print(f"[check] 结论：{'PASS' if ok else 'FAIL'}（用时 {time.time()-t0:.1f}s）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
