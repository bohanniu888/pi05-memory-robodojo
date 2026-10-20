#!/usr/bin/env python3
"""诊断 lerobot 视频帧容差：指定 episode 的最后一帧在给定 tolerance_s 下能否读取。

用法：python scripts/pi05/check_video_tolerance.py --episode 807 --tolerances 0.0001,0.05
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "robodojo/data/hf_lerobot_home"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="RoboDojo_sim_arx-x5_v30_history")
    ap.add_argument("--episode", type=int, default=807)
    ap.add_argument("--tolerances", default="0.0001,0.05")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    for tol in [float(t) for t in args.tolerances.split(",")]:
        try:
            ds = LeRobotDataset(
                args.repo_id,
                episodes=[args.episode],
                tolerance_s=tol,
                video_backend="pyav",
                delta_timestamps={"action": [t / 25 for t in range(50)]},
            )
            item = ds[len(ds) - 1]
            print(f"tol={tol}: OK len={len(ds)} ep={item['episode_index']} frame={item['frame_index']}")
        except Exception as e:  # noqa: BLE001
            print(f"tol={tol}: FAIL {type(e).__name__}: {str(e).splitlines()[0][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
