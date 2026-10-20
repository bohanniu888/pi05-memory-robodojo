#!/usr/bin/env python3
"""Phase 6 追加诊断：同一 checkpoint 在训练数据上的"历史 ON/OFF" action-loss 对照。

背景：训练后权重在评测里 history ON≈OFF（无显著差异）。需要在训练分布上判定：
"模型在训练域就不会用历史"（训练侧问题），还是"学到了但评测域/任务体现不出"。

方法：在训练数据上直接比较 flow-matching action loss 的四个臂（同 batch、同随机种子）：
  on   = 真实历史（20 槽 + 真 mask，与训练同口径）
  off  = 全零历史 + mask 全 False（等价"无历史"）
  shuf = 同 batch 内 roll=1 的错位历史（注意力结构同 on，内容不属于本样本）
  zero = 全零历史 + 真 mask（结构对照：插槽数与 on 相同，内容为空）
  maskroll = 全零历史 + roll=1 的错位 mask（只动 mask 结构，内容为空）
  croll    = roll=1 的错位内容 + 真 mask（只动内容，mask 结构与本样本一致）

分解用途：shuf 同时改了 mask 和内容，用 maskroll/croll 可把 Δ(shuf-on) 拆成
"mask 结构效应"与"内容效应"两部分。

判定：
  * on ≈ off ≈ shuf ≈ zero  → 模型在训练域也不用历史 → 训练侧问题（最强结论）
  * on 明显 < off，且 on ≈ shuf → 只是"多一段前缀 binder"效应，不是内容使用
  * on 明显 < off/shuf/zero → 确实按内容使用了历史 → 评测不涨则问题在评测域/任务

注意：sidecar 的 600 个 mem6 episodes（各任务 100 集）全部在 cotrain 训练清单内，
因此这是**训练集内**对照；"是否使用历史内容"依赖 shuf/zero 对照臂判定。

用法（在 GPU 作业内、openpi venv 下）：
  CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME=... ROBODOJO_HISTORY_SIDECAR_DIR=... \\
    robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \\
    scripts/pi05/offline_history_loss_ab.py --num-batches 200 --batch-size 64 \\
    --out-json logs/pi05/offline_history_loss_ab_<ts>.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OPENPI_SRC = ROOT / "robodojo/XPolicyLab/policy/Pi_05/openpi/src"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import data_loader as _data_loader  # noqa: E402

DEFAULT_CONFIG = "pi05_aloha_sim_history_arx-x5_seed_0"
DEFAULT_CKPT = (
    ROOT / "robodojo/XPolicyLab/policy/Pi_05/checkpoints/"
    "RoboDojo-sim-arx_x5-history-lora-seed0/19999"
)
DEFAULT_COTRAIN_JSON = ROOT / "robodojo/data/history_sidecar/cotrain_mix_episodes.json"
ALL_ARMS = ("on", "off", "shuf", "zero", "maskroll", "croll")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default=DEFAULT_CONFIG)
    p.add_argument("--ckpt-dir", type=pathlib.Path, default=DEFAULT_CKPT)
    p.add_argument("--cotrain-json", type=pathlib.Path, default=DEFAULT_COTRAIN_JSON)
    p.add_argument("--num-batches", type=int, default=200, help="评测 batch 数（每 batch = batch-size 个样本）")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234, help="loss 噪声种子基准（每 batch fold_in）")
    p.add_argument("--min-usable", type=int, default=10, help="条件统计的最小历史槽位阈值")
    p.add_argument("--arms", default="on,off,shuf,zero", help=f"要评的臂，逗号分隔，可选 {ALL_ARMS}")
    p.add_argument("--out-json", type=pathlib.Path, default=None)
    p.add_argument("--out-npz", type=pathlib.Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_json = args.out_json or (ROOT / f"logs/pi05/offline_history_loss_ab_{ts}.json")
    out_npz = args.out_npz or out_json.with_suffix(".npz")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    mem6_episodes = json.loads(args.cotrain_json.read_text())["mem6_episodes"]
    print(f"[ab] mem6 episodes: {len(mem6_episodes)}（全部在 cotrain 训练清单内 = 训练集内对照）", flush=True)

    cfg = _config.get_config(args.config_name)
    assert int(getattr(cfg.model, "history_slots", 0) or 0) > 0, "config 未开启 history_slots"
    cfg = dataclasses.replace(cfg, batch_size=args.batch_size, num_workers=args.num_workers)
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            base_config=dataclasses.replace(cfg.data.base_config, episodes=tuple(int(e) for e in mem6_episodes)),
        ),
    )

    print(f"[ab] 加载 checkpoint: {args.ckpt_dir}", flush=True)
    params = _model.restore_params(args.ckpt_dir / "params", dtype=jnp.bfloat16)
    model = cfg.model.load(params)
    del params

    loader = _data_loader.create_data_loader(cfg, sharding=None, shuffle=True, num_batches=args.num_batches)

    @nnx.jit
    def loss_fn(model, observation, actions, rng):
        return model.compute_loss(rng, observation, actions, train=False)

    keys = tuple(k.strip() for k in args.arms.split(",") if k.strip())
    unknown = [k for k in keys if k not in ALL_ARMS]
    if unknown or "on" not in keys:
        raise SystemExit(f"arms 非法：{unknown or '缺少 on'}，可选 {ALL_ARMS}")
    acc = {k: [] for k in keys}
    usable_all: list[np.ndarray] = []
    batch_delta_on_off: list[float] = []
    base_rng = jax.random.key(args.seed)
    t0 = time.time()
    n = 0
    for i, (obs, actions) in enumerate(loader):
        hi, hm = obs.history_images, obs.history_mask
        if i == 0:
            _hi = np.asarray(hi, dtype=np.float32)
            _hm = np.asarray(hm)
            print(
                f"[diag] history_images dtype={hi.dtype} shape={hi.shape} "
                f"min={float(_hi.min()):.3f} max={float(_hi.max()):.3f} mean={float(_hi.mean()):.4f} "
                f"| 每样本可用槽均值={float(_hm.sum(-1).mean()):.2f}",
                flush=True,
            )
        _all_arms = {
            "on": obs,
            "off": obs.replace(history_images=jnp.zeros_like(hi), history_mask=jnp.zeros_like(hm)),
            "shuf": obs.replace(history_images=jnp.roll(hi, 1, axis=0), history_mask=jnp.roll(hm, 1, axis=0)),
            "zero": obs.replace(history_images=jnp.zeros_like(hi), history_mask=hm),
            "maskroll": obs.replace(history_images=jnp.zeros_like(hi), history_mask=jnp.roll(hm, 1, axis=0)),
            "croll": obs.replace(history_images=jnp.roll(hi, 1, axis=0), history_mask=hm),
        }
        arms = {k: _all_arms[k] for k in keys}
        rng = jax.random.fold_in(base_rng, i)
        losses = {k: np.asarray(loss_fn(model, arms[k], actions, rng)) for k in keys}
        for k in keys:
            acc[k].append(losses[k])
        usable = np.asarray(hm.sum(-1))
        usable_all.append(usable)
        if "off" in keys:
            batch_delta_on_off.append(float(losses["off"].mean() - losses["on"].mean()))
        n += 1
        if n % 10 == 0:
            means = {k: float(np.concatenate(acc[k]).mean()) for k in keys}
            extra = " ".join(f"Δ({k}-on)={means[k]-means['on']:+.6f}" for k in keys if k != "on")
            print(
                f"[ab] batch {n}/{args.num_batches} loss(on)={means['on']:.6f} {extra} 用时 {time.time()-t0:.0f}s",
                flush=True,
            )

    losses = {k: np.concatenate(v) for k, v in acc.items()}
    usable = np.concatenate(usable_all)
    deltas = batch_delta_on_off

    def stats(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {"n": 0}
        out = {"n": int(mask.sum())}
        for k in keys:
            out[k] = float(losses[k][mask].mean())
        for k in keys:
            if k != "on":
                out[f"Δ({k}-on)"] = float(losses[k][mask].mean() - losses["on"][mask].mean())
        return out

    summary = {
        "config": args.config_name,
        "ckpt": str(args.ckpt_dir),
        "episodes": len(mem6_episodes),
        "batches": n,
        "batch_size": args.batch_size,
        "samples": int(len(losses["on"])),
        "seed": args.seed,
        "wall_sec": round(time.time() - t0, 1),
        "usableslot_mean": float(usable.mean()),
        "samples_usable_eq0_frac": float((usable == 0).mean()),
        "overall": stats(np.ones_like(usable, dtype=bool)),
        "usable_ge_min": stats(usable >= args.min_usable),
        "usable_lt_min": stats(usable < args.min_usable),
        "batch_delta_off_minus_on": (
            {
                "mean": float(np.mean(deltas)),
                "std": float(np.std(deltas)),
                "se": float(np.std(deltas) / max(np.sqrt(len(deltas)), 1.0)),
            }
            if deltas
            else None
        ),
    }
    np.savez_compressed(
        out_npz,
        usable=usable,
        **{f"loss_{k}": losses[k] for k in keys},
        batch_delta_on_off=np.asarray(deltas),
    )
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[ab] 已写出 {out_json} 与 {out_npz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
