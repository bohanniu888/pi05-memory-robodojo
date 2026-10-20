#!/usr/bin/env python3
"""[pi05-keyframe] Phase 8 Step 1：关键帧权重表生成 + 视觉活动剖面（Tier A）。

两种模式（可先后执行）：
  --profile  ：用 sidecar 的 1 Hz 帧算"相邻帧视觉变化"剖面（按 episode 相对时间 10 分箱、跨集聚合），
               输出每任务 ASCII 表 + JSON，用于数据化地校准各任务的"关键段"边界（替代拍脑袋）。
  --tier-a   ：按任务规则（相对时间窗 + 权重）生成 keyframe_weights_v1.json。

权重表格式（1 Hz 分辨率，与历史槽同频；slot k ↔ frame_index // 25 ∈ [25k, 25k+25)）：
  {"meta": {...}, "interval": 25, "weights": {"<episode_index>": [w0, w1, ...]}}
未列出的 episode / 超出数组长度的 slot → transform 侧默认 1.0。

用法：
  python scripts/pi05/build_keyframe_weights.py --profile
  python scripts/pi05/build_keyframe_weights.py --tier-a \
    --out robodojo/data/history_sidecar/keyframe_weights_v1.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SIDECAR = ROOT / "robodojo/data/history_sidecar"

# ---- Tier A 规则表（相对时间窗，按 --profile 结果校准）----
# 语义：cue/线索进入视野 → 信息消失（遮挡/交换/移出）之后的部分才算"必须靠回忆"。
# key = 任务名；value = (start_frac, end_frac, weight)，闭区间，按 slot 相对位置判定。
# 校准依据（2026-10-10 profile，n=600 集，10 分箱）：
#   imitate_sorting_sequence: 前 40% 平静(0.011~0.015)、0.4 后跃升(~0.04) → 示范段在前、执行段在后；
#   match_and_pick_from_conveyor: 活动度持续爬升到结尾(0.9-1.0 → 0.0277)，峰值中位 0.89 → 末段最关键；
#   cover_blocks: 0.3-0.4 局部峰(0.0616)、峰值中位 0.70 → 覆盖动作后进入记忆段；
#   press_by_number: 双峰 0.4-0.5(0.0458)/0.8-0.9(0.0539) → 首按之后即依赖已按顺序；
#   swap_T: 峰值中位 0.22、0.2-0.4 高活动 → 交换早段完成，之后靠记忆；
#   swap_blocks: 0.3-0.4 峰(0.0696) → 交换后进入记忆段。
TIER_A_RULES: dict[str, tuple[float, float, float]] = {
    "cover_blocks": (0.35, 1.0, 4.0),
    "imitate_sorting_sequence": (0.40, 1.0, 4.0),
    "match_and_pick_from_conveyor": (0.35, 1.0, 4.0),
    "press_by_number": (0.30, 1.0, 3.0),
    "swap_T": (0.30, 1.0, 4.0),
    "swap_blocks": (0.40, 1.0, 4.0),
}


def load_index() -> dict:
    with open(SIDECAR / "index.json", encoding="utf-8") as f:
        return json.load(f)


def load_mem6() -> list[int]:
    with open(SIDECAR / "cotrain_mix_episodes.json", encoding="utf-8") as f:
        return [int(e) for e in json.load(f)["mem6_episodes"]]


def cmd_profile(n_bins: int, out_json: pathlib.Path | None) -> None:
    index = load_index()
    mem6 = load_mem6()
    acc: dict[str, np.ndarray] = collections.defaultdict(lambda: np.zeros(n_bins))
    cnt: dict[str, np.ndarray] = collections.defaultdict(lambda: np.zeros(n_bins))
    peak: dict[str, list[float]] = collections.defaultdict(list)
    missing = 0
    for ep in mem6:
        entry = index.get(str(ep))
        if entry is None:
            missing += 1
            continue
        task = entry["task"]
        path = SIDECAR / entry["path"]
        if not path.exists():
            missing += 1
            continue
        frames = np.load(path, mmap_mode="r")
        n = int(frames.shape[0])
        if n < 3:
            continue
        x = np.asarray(frames, dtype=np.float32)  # (n,3,128,128)
        d = np.abs(np.diff(x, axis=0)).mean(axis=(1, 2, 3)) / 255.0  # (n-1,)
        rel = (np.arange(1, n) / (n - 1)).clip(0, 0.9999)
        bins = (rel * n_bins).astype(int)
        for b in range(n_bins):
            m = bins == b
            if m.any():
                acc[task][b] += float(d[m].mean())
                cnt[task][b] += 1.0
        peak[task].append(float(np.argmax(d)) / (n - 2) if n > 2 else 0.0)
        del x, d
    if missing:
        print(f"[profile] 跳过 {missing} 集（index 缺失/文件不存在）")
    tasks = sorted(acc)
    print(f"\n=== 1 Hz 相邻帧视觉变化剖面（相对时间 {n_bins} 分箱，行=任务，列=episode 相对位置）===")
    header = "".join(f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}".rjust(12) for i in range(n_bins))
    print(f"{'task':32s}{header}   峰值位置(中位)")
    profile_out: dict[str, dict] = {}
    for t in tasks:
        vals = acc[t] / np.maximum(cnt[t], 1.0)
        line = "".join(f"{v:12.4f}" for v in vals)
        pk = float(np.median(peak[t]))
        print(f"{t:32s}{line}   {pk:.2f}")
        profile_out[t] = {"activity": vals.tolist(), "peak_median": pk, "n_eps": len(peak[t])}
    if out_json is not None:
        out_json.write_text(json.dumps({"n_bins": n_bins, "tasks": profile_out}, ensure_ascii=False, indent=2))
        print(f"[profile] 已写出 {out_json}")


def cmd_tier_a(out_json: pathlib.Path, rules: dict[str, tuple[float, float, float]], interval: int) -> None:
    index = load_index()
    mem6 = load_mem6()
    weights: dict[str, list[float]] = {}
    stats = collections.defaultdict(lambda: {"n_eps": 0, "slots": 0, "key_slots": 0, "sum_w": 0.0})
    unknown_tasks = set()
    for ep in mem6:
        entry = index.get(str(ep))
        if entry is None:
            continue
        task = entry["task"]
        n_slots = int(entry.get("num_samples") or 0)
        if n_slots <= 0:
            n_slots = max(1, int(entry.get("length", 0)) // interval)  # 兜底
        rule = rules.get(task)
        if rule is None:
            unknown_tasks.add(task)
        w = np.ones(n_slots, dtype=np.float64)
        if rule is not None:
            start_frac, end_frac, weight = rule
            if n_slots > 1:
                rel = np.arange(n_slots) / (n_slots - 1)
            else:
                rel = np.zeros(n_slots)
            m = (rel >= start_frac) & (rel <= end_frac)
            w[m] = weight
        weights[str(ep)] = [float(x) for x in w]
        s = stats[task]
        s["n_eps"] += 1
        s["slots"] += n_slots
        s["key_slots"] += int((w > 1.0).sum())
        s["sum_w"] += float(w.sum())
    if unknown_tasks:
        print(f"[tier-a] 警告：以下任务没有规则，整集 w=1 → {sorted(unknown_tasks)}")
    payload = {
        "meta": {
            "tier": "A",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rules": {k: {"start_frac": v[0], "end_frac": v[1], "weight": v[2]} for k, v in rules.items()},
            "note": "Tier A 相对时间窗规则；Tier B（歧义度打分）将替代/修正本表。",
        },
        "interval": interval,
        "weights": weights,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False))
    total_slots = sum(s["slots"] for s in stats.values())
    total_key = sum(s["key_slots"] for s in stats.values())
    total_w = sum(s["sum_w"] for s in stats.values())
    print(f"\n=== Tier A v1 统计（{len(weights)} 集）===")
    print(f"{'task':32s}{'eps':>5}{'slots':>8}{'关键帧占比':>12}{'平均权重':>10}")
    for t in sorted(stats):
        s = stats[t]
        print(f"{t:32s}{s['n_eps']:>5}{s['slots']:>8}{s['key_slots'] / max(s['slots'], 1):>11.1%}{s['sum_w'] / max(s['slots'], 1):>10.2f}")
    print(f"{'ALL':32s}{sum(s['n_eps'] for s in stats.values()):>5}{total_slots:>8}{total_key / max(total_slots, 1):>11.1%}{total_w / max(total_slots, 1):>10.2f}")
    print(f"[tier-a] 已写出 {out_json}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true", help="只跑视觉活动剖面（校准用）")
    p.add_argument("--tier-a", action="store_true", help="生成 Tier A 权重表")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--interval", type=int, default=25, help="抽帧间隔（需与 sidecar/history_interval 一致）")
    p.add_argument("--out", type=pathlib.Path, default=SIDECAR / "keyframe_weights_v1.json")
    p.add_argument("--profile-out", type=pathlib.Path, default=None)
    args = p.parse_args()
    if not args.profile and not args.tier_a:
        p.error("至少指定 --profile 或 --tier-a")
    if args.profile:
        cmd_profile(args.n_bins, args.profile_out)
    if args.tier_a:
        if not TIER_A_RULES:
            p.error("TIER_A_RULES 为空：先用 --profile 校准并填写规则")
        cmd_tier_a(args.out, TIER_A_RULES, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
