#!/usr/bin/env python3
"""[pi05-keyframe] Phase 8 进度汇总：Tier B 诊断 + v2/v1 对照 + 门禁(Δ) 序列。"""
from __future__ import annotations
import glob, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SIDECAR = ROOT / "robodojo/data/history_sidecar"


def weights_stats(path: pathlib.Path) -> dict:
    blob = json.loads(path.read_text())
    arrs = [np.asarray(v) for v in blob["weights"].values()]
    allw = np.concatenate(arrs)
    return {
        "episodes": len(arrs),
        "slots": int(allw.size),
        "w_gt1_frac": float((allw > 1.0).mean()),
        "w_mean": float(allw.mean()),
        "w_max": float(allw.max()),
    }


def main() -> int:
    v1 = SIDECAR / "keyframe_weights_v1.json"
    v2 = SIDECAR / "keyframe_weights_v2.json"
    print("=== 权重表 ===")
    for p in (v1, v2):
        if p.exists():
            s = weights_stats(p)
            print(f"  {p.name}: 集数={s['episodes']} 槽数={s['slots']} w>1占比={s['w_gt1_frac']:.1%} 平均={s['w_mean']:.2f} max={s['w_max']:.0f}")
    diag = ROOT / "logs/pi05/keyframe_tierb_diag.json"
    if diag.exists():
        d = json.loads(diag.read_text())
        print("=== Tier B 诊断 ===")
        for k, v in d.items():
            if k.startswith("_"):
                continue
            q = v["s_quantiles"]
            print(f"  {k:30s} n={v['n_slots']:5d} s(p50={q['0.5']:.4f} p90={q['0.9']:.4f}) w>1={v['w_gt1_frac']:.1%} w̄={v['w_mean']:.2f}")
        g = d.get("_global", {})
        if g:
            print(f"  全局: 槽数={g['slots']} w>1={g['w_gt1_frac']:.1%} w̄={g['w_mean']:.2f} 与v1 top20%重合={g['v1_overlap_top20']}")
    print("=== 门禁（离线 ON/OFF 内容对照）===")
    rows = []
    for f in sorted(glob.glob(str(ROOT / "logs/pi05/gate_kf_*.json")), key=lambda s: int(s.split("_")[-1].split(".")[0])):
        j = json.loads(pathlib.Path(f).read_text())
        o = j["overall"]
        rows.append((int(pathlib.Path(f).stem.split("_")[-1]), o))
    for step, o in rows:
        print(f"  step {step:>6}: on={o['on']:.6f} Δ(zero-on)={o.get('Δ(zero-on)', float('nan')):+.2e} Δ(croll-on)={o.get('Δ(croll-on)', float('nan')):+.2e}")
    if not rows:
        print("  （暂无 —— 首个 checkpoint 落地后由 p8_gate 作业自动产出）")

    print("=== 6 臂分解（ab6_kf_*.json；内容效应看 croll，结构效应看 maskroll）===")
    ab6 = sorted(glob.glob(str(ROOT / "logs/pi05/ab6_kf_*.json")), key=lambda s: int(pathlib.Path(s).stem.split("_")[-1]))
    if not ab6:
        print("  （暂无 —— p8_ab6 作业运行后产出）")
    for f in ab6:
        j = json.loads(pathlib.Path(f).read_text())
        o = j["overall"]
        step = pathlib.Path(f).stem.split("_")[-1]
        print(
            f"  step {step:>6}: on={o['on']:.6f} off={o.get('off', float('nan')):.6f} shuf={o.get('shuf', float('nan')):.6f} "
            f"zero={o.get('zero', float('nan')):.6f} maskroll={o.get('maskroll', float('nan')):.6f} croll={o.get('croll', float('nan')):.6f}"
        )
        print(
            f"           Δ(off-on)={o.get('Δ(off-on)', float('nan')):+.2e} Δ(shuf-on)={o.get('Δ(shuf-on)', float('nan')):+.2e} "
            f"Δ(zero-on)={o.get('Δ(zero-on)', float('nan')):+.2e} Δ(maskroll-on)={o.get('Δ(maskroll-on)', float('nan')):+.2e} "
            f"Δ(croll-on)={o.get('Δ(croll-on)', float('nan')):+.2e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
