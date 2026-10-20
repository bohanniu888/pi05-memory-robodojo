#!/usr/bin/env python
"""Phase 0D：π0.5 运行期标定（加载权重 → 合成观测 → 计时/显存/动作块形状）。

用法（在作业内，先用 uv 环境）：
    srun --jobid=<jid> --overlap bash -c '
      cd "$PI05_ROOT/robodojo/XPolicyLab/policy/Pi_05/openpi" &&
      uv run python "$PI05_ROOT/scripts/pi05/probe_pi05_runtime.py"'

输出：logs/pi05/probe_<ts>.json + stdout 摘要。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / "robodojo/XPolicyLab/policy/Pi_05"
OPENPI_DIR = POLICY_DIR / "openpi"
XPL_ROOT = POLICY_DIR.parent.parent  # XPolicyLab root
for _p in (str(OPENPI_DIR), str(XPL_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt-path",
        default=str(POLICY_DIR / "checkpoints" / "RoboDojo-sim-arx_x5-joint-2" / "59999"),
    )
    ap.add_argument("--config-name", default="pi05_base_aloha_full_sim_arx-x5_seed_0")
    ap.add_argument("--asset-id", default="arx_x5_sim")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out-dir", default=str(ROOT / "logs/pi05"))
    args = ap.parse_args()

    ckpt = Path(args.ckpt_path)
    if not (ckpt / "params").exists():
        print(f"[probe] 找不到 params/：{ckpt}", file=sys.stderr)
        return 2

    # 与 eval.sh 的评测口径保持一致（server 端 0.3）
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

    from openpi.policies import policy_config as _policy_config
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config

    print(f"[probe] ckpt={ckpt}")
    cfg = _config.get_config(args.config_name)
    model_cfg = {
        "action_horizon": int(getattr(cfg.model, "action_horizon", -1)),
        "action_dim": int(getattr(cfg.model, "action_dim", -1)),
        "max_token_len": int(getattr(cfg.model, "max_token_len", -1)),
        "pi05": bool(getattr(cfg.model, "pi05", False)),
        "paligemma_variant": str(getattr(cfg.model, "paligemma_variant", "")),
        "action_expert_variant": str(getattr(cfg.model, "action_expert_variant", "")),
    }
    print(f"[probe] model_cfg={model_cfg}")
    t0 = time.time()
    norm_stats = _normalize.load(ckpt / "assets" / args.asset_id)
    policy = _policy_config.create_trained_policy(cfg, str(ckpt), norm_stats=norm_stats)
    load_s = time.time() - t0

    rng = np.random.default_rng(0)
    obs = {
        "state": np.zeros(14, dtype=np.float32),
        "images": {
            name: rng.integers(0, 255, (3, args.height, args.width), dtype=np.uint8) for name in CAMERAS
        },
        "prompt": (
            "Pick up the two T-shaped blocks, swap their positions, and place them back "
            "with the correct orientations."
        ),
    }

    t0 = time.time()
    out = policy.infer(obs)
    warmup_s = time.time() - t0
    actions = np.asarray(out["actions"])
    latencies = []
    for _ in range(args.iters):
        t0 = time.time()
        out = policy.infer(obs)
        latencies.append(time.time() - t0)

    gpu = {}
    try:
        import jax

        stats = jax.local_devices()[0].memory_stats() or {}
        gpu = {k: stats.get(k) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")}
    except Exception as exc:  # pragma: no cover - 诊断信息，失败不致命
        gpu = {"error": str(exc)}

    params_bytes = sum(f.stat().st_size for f in (ckpt / "params").rglob("*") if f.is_file())
    report = {
        "env": {
            "host": __import__("socket").gethostname(),
            "checkpoint": str(ckpt),
            "config_name": args.config_name,
            "params_bytes": params_bytes,
        },
        "model_cfg": model_cfg,
        # deploy.py 会把 infer 返回的整段 action chunk 全部执行完后才重新推理，
        # 因此"每个 chunk 实际执行的底层动作数" == action_horizon。
        "executed_actions_per_chunk": model_cfg["action_horizon"],
        "load_seconds": round(load_s, 3),
        "warmup_infer_seconds": round(warmup_s, 3),
        "infer_seconds": [round(x, 3) for x in latencies],
        "infer_mean_seconds": round(float(np.mean(latencies)), 3),
        "actions_shape": list(actions.shape),
        "actions_dtype": str(actions.dtype),
        "action_horizon": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "gpu_memory": gpu,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"probe_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[probe] saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
