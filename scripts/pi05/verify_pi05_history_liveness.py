#!/usr/bin/env python3
"""Phase 2：π0.5 历史（记忆）通路离线活性验证 —— 无需仿真器/websocket。

对着真实 RoboDojo π0.5 checkpoint 直接跑 Policy.infer（固定 flow-matching noise，输出完全可复现）：

  A1 / A2  空历史（20 槽全 pad、mask 全 0）      → 两次完全一致（差 = 0）
  B        20 槽全黑（mask 全 1）                → 与 A 有显著差异（>1e-3）
  C        20 槽全白（mask 全 1）                → 与 B 可观测差异（证明读的是内容）
  D1       15 pad(全零) + 5 真槽(黑)             → 与 D2 差 < 1e-4（证明 mask 屏蔽 pad）
  D2       15 pad(随机) + 5 真槽(黑)             → 同上

另测：history_slots=0（基线）vs =20（满历史）的单次前向延迟与显存占用。

环境变量：PI05_CKPT / PI05_TRAIN_CONFIG / PI05_TEST_SEED / PI05_TEST_HW
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / "robodojo/XPolicyLab/policy/Pi_05"
XPL_ROOT = ROOT / "robodojo/XPolicyLab"
DEFAULT_CKPT = POLICY_DIR / "checkpoints/RoboDojo-sim-arx_x5-joint-2/59999"
TRAIN_CONFIG = os.environ.get("PI05_TRAIN_CONFIG", "pi05_base_aloha_full_sim_arx-x5_seed_0")
SEED = int(os.environ.get("PI05_TEST_SEED", "1234"))
HW = int(os.environ.get("PI05_TEST_HW", "224"))
SLOTS = 20


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ckpt = Path(os.environ.get("PI05_CKPT", str(DEFAULT_CKPT)))
    if not (ckpt / "params").is_dir():
        log(f"[H] ✗ 找不到 checkpoint params: {ckpt}")
        return 2

    sys.path.insert(0, str(XPL_ROOT))
    sys.path.insert(0, str(POLICY_DIR))

    import numpy as np
    import jax
    import flax.nnx as nnx  # noqa: F401
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import normalize as _normalize

    log("=" * 78)
    log(f"[H] host={os.uname().nodename} ckpt={ckpt}")
    log(f"[H] train_config={TRAIN_CONFIG} seed={SEED} hw={HW} slots={SLOTS}")

    train_config = _config.get_config(TRAIN_CONFIG)
    model_cfg0 = train_config.model
    model_cfg20 = dataclasses.replace(model_cfg0, history_slots=SLOTS)
    norm_stats = _normalize.load(ckpt / "assets" / "arx_x5_sim")

    log(f"[H] action_horizon={model_cfg0.action_horizon} action_dim={model_cfg0.action_dim}")

    def build(name: str, model_cfg):
        t0 = time.time()
        tc = dataclasses.replace(train_config, model=model_cfg)
        policy = _policy_config.create_trained_policy(tc, str(ckpt), norm_stats=norm_stats)
        stats = jax.local_devices()[0].memory_stats() or {}
        log(
            f"[H] 构建 {name}: {time.time() - t0:.1f}s  "
            f"jax bytes_in_use={stats.get('bytes_in_use', 0) / 2**30:.2f}GiB "
            f"peak={stats.get('peak_bytes_in_use', 0) / 2**30:.2f}GiB"
        )
        return policy

    policy_hist = build("history(slots=20)", model_cfg20)

    rng = np.random.default_rng(SEED)
    images = {
        name: (rng.integers(0, 256, (3, HW, HW), dtype=np.uint8))
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    }
    state = np.zeros(14, dtype=np.float32)
    obs = {"state": state, "images": images, "prompt": "cover the blocks"}
    noise = rng.standard_normal(
        (int(model_cfg0.action_horizon), int(model_cfg0.action_dim))
    ).astype(np.float32)

    hist_hw = HW
    black = np.zeros((SLOTS, 3, hist_hw, hist_hw), dtype=np.uint8)
    white = np.full((SLOTS, 3, hist_hw, hist_hw), 255, dtype=np.uint8)
    rand = rng.integers(0, 256, (SLOTS, 3, hist_hw, hist_hw), dtype=np.uint8)

    def run(tag: str, history_images=None, history_mask=None, policy=None) -> np.ndarray:
        pol = policy or policy_hist
        this = dict(obs)
        if history_images is not None:
            this["history_images"] = history_images
            this["history_mask"] = history_mask
        pol._rng = jax.random.key(SEED)
        t0 = time.time()
        out = pol.infer(this, noise=noise)
        dt = time.time() - t0
        flat = np.asarray(out["actions"]).astype(np.float32).ravel()
        log(f"[H] {tag:<30} 动作维度={flat.shape[0]}  首4值={np.round(flat[:4], 4)}  用时={dt:.2f}s")
        return flat

    def diff(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).max())

    log("--- 1) A1/A2：空历史（20 槽全 pad、mask=0）跑两次 ---")
    mask_pad_all = np.zeros((SLOTS,), dtype=bool)
    a1 = run("A1 空历史(0/20)", black, mask_pad_all)
    a2 = run("A2 空历史(0/20) 重跑", black, mask_pad_all)

    log("--- 2) B/C：20 槽全黑 / 全白 ---")
    mask_full = np.ones((SLOTS,), dtype=bool)
    b = run("B 全黑(20/20)", black, mask_full)
    c = run("C 全白(20/20)", white, mask_full)

    log("--- 3) D1/D2：15 pad + 5 真槽，pad 内容 全零 vs 随机 ---")
    mask_mix = np.zeros((SLOTS,), dtype=bool)
    mask_mix[SLOTS - 5 :] = True
    d1 = run("D1 pad=全零", black, mask_mix)
    d2 = run("D2 pad=随机", np.concatenate([rand[: SLOTS - 5], black[SLOTS - 5 :]], axis=0), mask_mix)

    log("--- 4) 延迟/显存：基线 slots=0 vs 满历史 slots=20 ---")
    policy_base = build("baseline(slots=0)", model_cfg0)

    def bench(tag: str, pol, hist_on: bool) -> float:
        for _ in range(2):  # warmup / 编译
            if hist_on:
                run(f"{tag} warmup", black, mask_full, policy=pol)
            else:
                run(f"{tag} warmup", None, None, policy=pol)
        times = []
        for _ in range(5):
            t0 = time.time()
            pol._rng = jax.random.key(SEED)
            pol.infer(dict(obs), noise=noise)
            times.append(time.time() - t0)
        med = float(np.median(times))
        log(f"[H] {tag}: 中位前向 {med * 1000:.0f} ms（{['%.0f' % (t * 1000) for t in times]}）")
        return med

    t_base = bench("基线 slots=0", policy_base, hist_on=False)
    t_hist = bench("满历史 slots=20", policy_hist, hist_on=True)
    stats = jax.local_devices()[0].memory_stats() or {}

    noise_floor = diff(a1, a2)
    effect_b = diff(a1, b)
    effect_c = diff(b, c)
    pad_effect = diff(d1, d2)
    lat_ratio = t_hist / max(t_base, 1e-9)

    log("=" * 78)
    log("[H] 结论")
    log(f"[H] 噪声底线   A1 vs A2 = {noise_floor:.9f}（复核固定 noise 的完全可复现性）")
    log(f"[H] 有无历史   A vs B  = {effect_b:.6f}   （判据 > 1e-3）")
    log(f"[H] 历史内容   B vs C  = {effect_c:.6f}   （判据：可观测 > 底线）")
    log(f"[H] pad 屏蔽   D1 vs D2= {pad_effect:.9f}（判据 < 1e-4）")
    log(f"[H] 前向延迟   基线 {t_base * 1000:.0f} ms → 满历史 {t_hist * 1000:.0f} ms，+{(lat_ratio - 1) * 100:.1f}%（判据 < 30%）")
    log(f"[H] JAX 显存   bytes_in_use {stats.get('bytes_in_use', 0) / 2**30:.2f} GiB，peak {stats.get('peak_bytes_in_use', 0) / 2**30:.2f} GiB（两模型同进程）")
    ok_repro = noise_floor < 1e-6
    ok_effect = effect_b > 1e-3
    ok_content = effect_c > max(noise_floor, 1e-6) * 2
    ok_pad = pad_effect < 1e-4
    log(
        f"[H] ★ 记忆通路："
        f"{'已接通' if (ok_repro and ok_effect and ok_content and ok_pad) else '未接通 / 不满足判据'}"
        f"（可复现={ok_repro} 有效应={ok_effect} 读内容={ok_content} pad屏蔽={ok_pad}）"
    )
    log(f"[H] end={time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
