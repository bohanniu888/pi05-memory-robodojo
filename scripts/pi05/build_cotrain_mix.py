#!/usr/bin/env python3
"""生成 Phase 5 训练用的 cotrain 混采清单（mem6 : 通用 ≈ 1:1，帧级口径）。

输入：
  - sidecar index.json（mem6 600 集）
  - LeRobot meta/episodes parquet（全 3500 集 + task_index/length）
输出：
  - robodojo/data/history_sidecar/cotrain_mix_episodes.json

采样规则：
  - mem6 六任务全部 600 集；
  - 其余 29 任务分层抽样 600 集（每任务 20~21 集，seed 固定），保持任务多样性；
  - 以帧数校验 mem6 : 通用 比例，期望 ≈50:50。
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

MEM6_TASKS = {3, 11, 16, 25, 32, 33}
ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", default=str(ROOT / "robodojo/data/history_sidecar"))
    ap.add_argument("--meta-episodes", default=str(
        ROOT / "robodojo/data/RoboDojo_lerobot_v30_video/meta/episodes/chunk-000/file-000.parquet"))
    ap.add_argument("--general-count", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sidecar = Path(args.sidecar)
    index = json.loads((sidecar / "index.json").read_text())
    mem6_eps = sorted(int(k) for k in index)

    table = pq.read_table(args.meta_episodes, columns=["episode_index", "length", "stats/task_index/mean"])
    rows = table.to_pylist()
    # 该 LeRobot 版本的 stats 列按"每视频一段"存成 list（此处长度恒为 1），取 [0]。
    meta = {
        int(r["episode_index"]): (int(round(r["stats/task_index/mean"][0])), int(r["length"])) for r in rows
    }
    assert len(meta) == 3500, len(meta)

    gen_by_task: dict[int, list[int]] = {}
    for ep, (task, _) in meta.items():
        if task not in MEM6_TASKS:
            gen_by_task.setdefault(task, []).append(ep)
    for eps in gen_by_task.values():
        eps.sort()

    rng = random.Random(args.seed)
    tasks = sorted(gen_by_task)
    base, extra = divmod(args.general_count, len(tasks))
    general_eps: list[int] = []
    for i, task in enumerate(tasks):
        take = base + (1 if i < extra else 0)
        general_eps.extend(rng.sample(gen_by_task[task], take))
    general_eps.sort()

    mem_frames = sum(meta[ep][1] for ep in mem6_eps)
    gen_frames = sum(meta[ep][1] for ep in general_eps)
    out = {
        "seed": args.seed,
        "mem6_tasks": sorted(MEM6_TASKS),
        "mem6_episodes": mem6_eps,
        "general_episodes": general_eps,
        "episodes": sorted(mem6_eps + general_eps),
        "stats": {
            "num_mem6": len(mem6_eps),
            "num_general": len(general_eps),
            "frames_mem6": mem_frames,
            "frames_general": gen_frames,
            "frame_ratio_mem6": round(mem_frames / (mem_frames + gen_frames), 4),
        },
    }
    out_path = Path(args.out) if args.out else sidecar / "cotrain_mix_episodes.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[mix] wrote {out_path}")
    print(f"[mix] episodes={len(out['episodes'])} mem6={len(mem6_eps)} general={len(general_eps)}")
    print(f"[mix] frames mem6={mem_frames} general={gen_frames} ratio_mem6={out['stats']['frame_ratio_mem6']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
