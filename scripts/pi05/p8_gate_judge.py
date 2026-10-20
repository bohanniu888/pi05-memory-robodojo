#!/usr/bin/env python3
"""Phase 8 门禁判据自动裁判：读 logs/pi05/gate_kf_<step>.npz，逐 step 输出
Δ(croll−on) / Δ(zero−on) 的均值、t 统计与通过判定。

判据（2026-10-20 符号校正）：Δ = bad_history − on；任一 Δ ≥ +1e-4 且 t > +5 → 通过。
用错/清零历史应使 loss 变大，所以通过方向是正，不是负。
用法：python3 scripts/pi05/p8_gate_judge.py [--log-dir logs/pi05] [--delta 1e-4] [--t 5]
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def stats(on: np.ndarray, arm: np.ndarray) -> tuple[float, float, int]:
    d = arm - on
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    t = d.mean() / se if se else float("nan")
    return float(d.mean()), float(t), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=pathlib.Path, default=ROOT / "logs/pi05")
    ap.add_argument("--delta", type=float, default=1e-4, help="Δ 阈值（≥ 视为显著正）")
    ap.add_argument("--t", type=float, default=5.0, help="t 阈值（> 视为显著）")
    ap.add_argument("--min-usable", type=int, default=10)
    ap.add_argument(
        "--patterns",
        default="gate_kf_*.npz,ab6_kf_*.npz",
        help="逗号分隔的 npz glob（相对 --log-dir）；前一个匹配优先。Phase 9 v2 用 gate_v2_*.npz",
    )
    ap.add_argument("--require-pass", action="store_true", help="无任一通过时以非零码退出（供自动接力判据用）")
    args = ap.parse_args()

    # gate_kf_*.npz（3 臂门禁）与 ab6_kf_*.npz（6 臂，含同 3 臂）都收；同 step 时 gate 优先
    _by_step: dict[int, str] = {}
    for pat in [p.strip() for p in args.patterns.split(",") if p.strip()]:
        for f in glob.glob(str(args.log_dir / pat)):
            step = int(pathlib.Path(f).stem.split("_")[-1])
            _by_step.setdefault(step, f)
    files = [_by_step[k] for k in sorted(_by_step)]
    if not files:
        print(f"没有匹配 {args.patterns} 的 npz")
        return 0

    print(f"判据：Δ ≥ {args.delta:+.0e} 且 t > {args.t:+.1f}（zero 或 croll 任一满足即通过）")
    header = f"{'step':>7} | {'arm':>6} | {'Δ mean':>10} | {'t':>7} | {'n':>5} | {'Δ(usable≥10)':>12} | t | 判定"
    print(header)
    print("-" * len(header))
    any_pass = False
    for f in files:
        step = int(pathlib.Path(f).stem.split("_")[-1])
        z = np.load(f)
        on, zero, croll = z["loss_on"], z["loss_zero"], z["loss_croll"]
        usable = z["usable"] if "usable" in z else None
        for name, arm in (("zero", zero), ("croll", croll)):
            m, t, n = stats(on, arm)
            line = f"{step:>7} | {name:>6} | {m:+.3e} | {t:+7.2f} | {n:>5} |"
            ok = m >= args.delta and t > args.t
            if usable is not None:
                sel = usable >= args.min_usable
                if sel.sum() > 1:
                    mu, tu, nu = stats(on[sel], arm[sel])
                    line += f" {mu:+.3e} | {tu:+6.2f} |"
                else:
                    line += f" {'n/a':>12} | {'n/a':>6} |"
            else:
                line += f" {'n/a':>12} | {'n/a':>6} |"
            line += " ✅ 通过" if ok else " ✗"
            if ok:
                any_pass = True
            print(line)
    print("-" * len(header))
    print("门禁结论：", "✅ 通过（可以进 Step 4 评测）" if any_pass else "✗ 未通过（继续等后续 ckpt）")
    if args.require_pass and not any_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
