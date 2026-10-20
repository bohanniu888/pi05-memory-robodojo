#!/usr/bin/env python3
"""Phase 5 数据加载吞吐基准：不同 num_workers 下的 samples/s（用于判断训练步时是否被 I/O 限住）。"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "robodojo/data/hf_lerobot_home"))
os.environ.setdefault("ROBODOJO_HISTORY_SIDECAR_DIR", str(ROOT / "robodojo/data/history_sidecar"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(ROOT / "robodojo/XPolicyLab/policy/Pi_05/openpi/src"))

from openpi.training import config as c  # noqa: E402
from openpi.training import data_loader as dl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=4)
    args = ap.parse_args()

    cfg = dataclasses.replace(
        c.get_config("pi05_aloha_sim_history_arx-x5_seed_0"),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    loader = dl.create_data_loader(cfg, sharding=None, shuffle=True)

    t0 = time.time()
    t_first = None
    n_samples = 0
    for i, batch in enumerate(loader):
        if i == 0:
            t_first = time.time()
        if i >= args.batches:
            break
        n_samples += len(next(iter(batch[0].images.values())))
    t_end = time.time()
    steady = t_end - (t_first or t0)
    n_steady = args.batches * args.batch_size - args.batch_size
    print(
        f"[bench] workers={args.workers} bs={args.batch_size} batches={args.batches} "
        f"first_batch={ (t_first or t0) - t0:.1f}s steady={steady:.1f}s "
        f"samples/s={n_steady / steady:.1f} (含预取)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
