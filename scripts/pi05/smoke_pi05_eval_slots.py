#!/usr/bin/env python3
"""Phase 6 前置：π0.5+memory 评测链路冒烟（slots 历史模式）。

两种档位：
  * 默认（CPU + fake policy）：不加载 2B 权重，验证 deploy.yml → Model 装配、
    启动一致性断言（含负例）、slots 组装语义（20×3×128×128 uint8 / mask、25 步 1 帧、
    最旧在左、pad=False、resize_with_pad 同 sidecar 口径）与 history_* 顶层透传；
  * --real-policy（GPU + 真 checkpoint）：完整 Model+Policy 端到端活性
    （有历史 vs 无历史动作差异、满 20 槽、重复前向一致、单次前向延迟）。

入口：bash scripts/pi05/smoke_pi05_eval_slots.sh [PI05_SMOKE_REAL=1 POLICY_GPU=0]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / "robodojo/XPolicyLab/policy/Pi_05"
XPL_ROOT = ROOT / "robodojo/XPolicyLab"
DEPLOY_YML = Path(os.environ.get("PI05_DEPLOY_YML", str(POLICY_DIR / "deploy.yml")))
DEFAULT_CKPT_NAME = "RoboDojo-sim-arx_x5-history-lora-seed0"
SLOTS = 20
INTERVAL = 25
SIZE = 128
SEED = 1234
FAILURES: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def check(name: str, ok: bool, detail: str = "") -> bool:
    log(f"[smoke] {'PASS' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-policy", action="store_true", help="GPU + 真 checkpoint 端到端活性")
    parser.add_argument("--updates-short", type=int, default=52, help="短序列 update 次数（默认 52 → 2 个历史槽）")
    parser.add_argument("--updates-full", type=int, default=501, help="满槽序列 update 次数（默认 501 → 20 个历史槽）")
    return parser.parse_args()


def load_deploy_cfg() -> dict:
    import yaml

    with open(DEPLOY_YML, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    # 评测时 ckpt_name 由 `robodojo.sh server --ckpt` 注入；冒烟脚本手动补齐。
    cfg.setdefault("ckpt_name", os.environ.get("PI05_SMOKE_CKPT", DEFAULT_CKPT_NAME))
    return cfg


def import_pi05_model():
    sys.path.insert(0, str(XPL_ROOT))
    sys.path.insert(0, str(POLICY_DIR))
    spec = importlib.util.spec_from_file_location("xpl_pi05_model", POLICY_DIR / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_frame(value: int, height: int = 160, width: int = 120):
    import numpy as np

    frame = np.full((height, width, 3), value, dtype=np.uint8)
    frame[::4, ::4] = (value + 17) % 256  # 一点空间纹理，避免常数图退化
    return frame


def make_obs(frame_hwc, env_idx: int = 0) -> dict:
    import numpy as np

    return {
        "images": {"cam_high": frame_hwc, "cam_left_wrist": frame_hwc, "cam_right_wrist": frame_hwc},
        "state": np.zeros(14, dtype=np.float32),
        "instruction": "cover the blocks",
        "env_idx": env_idx,
    }


class RecordingPolicy:
    """fake policy：记录收到的 history_*，返回由历史像素均值决定的伪动作（用于透传/活性断言）。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._rng = None

    def infer(self, obs: dict, **kwargs) -> dict:
        import numpy as np

        history = obs.get("history_images")
        mask = obs.get("history_mask")
        self.calls.append(
            {
                "history_images": None if history is None else np.array(history, copy=True),
                "history_mask": None if mask is None else np.array(mask, copy=True),
            }
        )
        if history is None or mask is None or not np.asarray(mask).any():
            actions = np.zeros((50, 14), dtype=np.float32)
        else:
            frames = np.asarray(history, dtype=np.float32)[np.asarray(mask, dtype=bool)]
            actions = np.full((50, 14), float(frames.mean()) / 255.0, dtype=np.float32)
        return {"actions": actions}


def install_fake_policy(pi05_model, recorders: list[RecordingPolicy]) -> None:
    def fake_get_model(self, model_cfg):
        self._train_config_name = str(model_cfg.get("train_config_name") or "")
        self._train_config = pi05_model._config.get_config(self._train_config_name)
        recorder = RecordingPolicy()
        recorders.append(recorder)
        return recorder

    pi05_model.Model.get_model = fake_get_model


def drive(model, frames: list) -> None:
    for frame in frames:
        model.update_obs_batch([make_obs(frame)])


def short_sequence(args, height=160, width=120) -> list:
    frames = [make_frame(37, height, width)] * 25 + [make_frame(201, height, width)] * 25
    return frames + [make_frame(120, height, width)] * max(0, args.updates_short - len(frames))


def report_ckpt_resolution(pi05_model, cfg: dict) -> None:
    try:
        root = pi05_model._resolve_pi05_model_root(dict(cfg))
    except Exception as exc:  # noqa: BLE001
        check("ckpt 目录解析", False, f"{type(exc).__name__}: {exc}")
        return
    steps = sorted(int(path.name) for path in root.parent.iterdir() if path.is_dir() and path.name.isdigit())
    expected = cfg.get("checkpoint_num")
    detail = f"checkpoint_num={expected} → {root.name}；同级 steps={steps[-4:]}"
    if expected is not None and int(expected) in steps:
        check("ckpt 目录解析", root.name == str(int(expected)), detail)
    else:
        check("ckpt 目录解析（预期 step 未保存，回退最大数字目录）", root.name == str(steps[-1]), detail)


def run_fake(pi05_model, cfg: dict, args: argparse.Namespace) -> None:
    import numpy as np

    log("=== 档位 1：CPU + fake policy（组装语义 / 透传 / 一致性断言）===")
    report_ckpt_resolution(pi05_model, cfg)

    recorders: list[RecordingPolicy] = []
    install_fake_policy(pi05_model, recorders)

    model = pi05_model.Model(copy.deepcopy(cfg))
    check("deploy.yml 装配 + 启动一致性断言通过", True, f"train_config={cfg['train_config_name']}")

    frame_a_chw = np.transpose(make_frame(37), (2, 0, 1))
    frame_b_chw = np.transpose(make_frame(201), (2, 0, 1))
    drive(model, short_sequence(args))

    attached = model._attach_history({}, 0)
    history = attached["history_images"]
    mask = attached["history_mask"]
    check("history_images 形状/dtype", history.shape == (SLOTS, 3, SIZE, SIZE) and history.dtype == np.uint8, str(history.shape))
    check("history_mask 形状/dtype", mask.shape == (SLOTS,) and mask.dtype == np.bool_, str(mask.shape))
    check("25 步 1 帧：52 步后 usable=2", int(mask.sum()) == 2, f"usable={int(mask.sum())}, mask[-4:]={mask[-4:].tolist()}")
    check("pad 槽 mask=False", not mask[:-2].any())
    check("槽位最旧在左（[18]=t0 帧，resize_with_pad 同口径）", np.array_equal(history[18], pi05_model.resize_chw_uint8(frame_a_chw, SIZE)))
    check("槽位最新在右（[19]=t25 帧）", np.array_equal(history[19], pi05_model.resize_chw_uint8(frame_b_chw, SIZE)))
    check("pad 槽内容全 0", not history[:-2].any())
    buffer_frames = [entry[2] for entry in model._history_by_env[0]]
    check("缓冲抽帧已 resize 到 128（非方形输入）", all(frame.shape == (3, SIZE, SIZE) for frame in buffer_frames), f"shapes={[frame.shape for frame in buffer_frames]}")

    actions = model.get_action_batch([0])
    passed = recorders[0].calls[-1]
    check("history_* 顶层透传到 policy.infer", np.array_equal(passed["history_images"], history) and np.array_equal(passed["history_mask"], mask))
    check("有历史 → 伪动作非零（活性）", float(np.abs(actions[0]).max()) > 0, f"mean={float(np.asarray(actions[0]).mean()):.4f}")

    model.reset()
    drive(model, [make_frame(120)])
    empty = model._attach_history({}, 0)
    empty_actions = model.get_action_batch([0])
    check("无历史 → mask 全 False、内容全 0", not empty["history_mask"].any() and not empty["history_images"].any())
    check("无历史 → 伪动作全 0", float(np.abs(empty_actions[0]).max()) == 0.0)

    model2 = pi05_model.Model(copy.deepcopy(cfg))
    drive(model2, short_sequence(args))
    attached2 = model2._attach_history({}, 0)
    check("重复组装的 history_images/mask 完全一致", np.array_equal(attached2["history_images"], history) and np.array_equal(attached2["history_mask"], mask))

    bad_slots = copy.deepcopy(cfg)
    bad_slots["history_slots"] = SLOTS - 1
    try:
        pi05_model.Model(bad_slots)
        check("负例：history_slots 不一致应报错", False)
    except ValueError as exc:
        check("负例：history_slots 不一致应报错", True, str(exc).split("。")[0][:80])

    bad_config = copy.deepcopy(cfg)
    bad_config["train_config_name"] = "pi05_base_aloha_full_sim_arx-x5_seed_0"
    try:
        pi05_model.Model(bad_config)
        check("负例：非 history 训练配置应报错", False)
    except ValueError as exc:
        check("负例：非 history 训练配置应报错", True, str(exc).split("。")[0][:80])


def run_real(pi05_model, cfg: dict, args: argparse.Namespace) -> None:
    import numpy as np
    import jax

    log("=== 档位 2：GPU + 真 checkpoint（端到端活性 / 满槽 / 延迟）===")
    report_ckpt_resolution(pi05_model, cfg)
    model = pi05_model.Model(copy.deepcopy(cfg))
    check("真模型加载 + 启动一致性断言通过", True)

    def infer():
        model.policy._rng = jax.random.key(SEED)
        started = time.time()
        out = model.get_action_batch([0])
        return np.asarray(out[0], dtype=np.float32).ravel(), time.time() - started

    frames_short = [make_frame(37, 224, 224)] * 25 + [make_frame(201, 224, 224)] * 25
    frames_short += [make_frame(120, 224, 224)] * max(0, args.updates_short - len(frames_short))

    model.reset()
    drive(model, frames_short)
    attached = model._attach_history({}, 0)
    check("真模型 2 槽组装", int(attached["history_mask"].sum()) == 2, f"usable={int(attached['history_mask'].sum())}")
    actions_hist, dt_hist = infer()

    model.reset()
    drive(model, [make_frame(120, 224, 224)])
    actions_none, dt_none = infer()
    diff = float(np.abs(actions_hist - actions_none).max())
    check("有历史 vs 无历史动作差异 > 1e-3（mem 通路活性）", diff > 1e-3, f"max|Δ|={diff:.3e}")

    model.reset()
    drive(model, frames_short)
    actions_repro, _ = infer()
    repro = float(np.abs(actions_hist - actions_repro).max())
    check("同输入重复前向一致（flow noise 固定）", repro < 1e-6, f"max|Δ|={repro:.3e}")

    frames_full = [make_frame(37, 224, 224)] * 25 + [make_frame(201, 224, 224)] * 25
    frames_full += [make_frame(120, 224, 224)] * max(0, args.updates_full - len(frames_full))
    model.reset()
    drive(model, frames_full)
    attached_full = model._attach_history({}, 0)
    usable = int(attached_full["history_mask"].sum())
    check("满槽 20/20 且最旧在左", usable == SLOTS and not attached_full["history_mask"][: SLOTS - usable].any(), f"usable={usable}")
    oldest = model._history_by_env[0][0][2]
    check("满槽槽位内容与缓冲一致（[0]=最旧）", np.array_equal(attached_full["history_images"][0], oldest))
    actions_full, dt_full = infer()
    diff_full = float(np.abs(actions_full - actions_hist).max())
    check("20 槽 vs 2 槽动作差异 > 1e-3", diff_full > 1e-3, f"max|Δ|={diff_full:.3e}")

    latencies = sorted([dt_hist, dt_none, dt_full])
    log(f"[smoke] 单次 get_action_batch 延迟：hist={dt_hist:.2f}s none={dt_none:.2f}s full20={dt_full:.2f}s median={latencies[1]:.2f}s")
    stats = jax.local_devices()[0].memory_stats() or {}
    log(f"[smoke] jax bytes_in_use={stats.get('bytes_in_use', 0) / 2**30:.2f}GiB peak={stats.get('peak_bytes_in_use', 0) / 2**30:.2f}GiB")


def main() -> int:
    args = parse_args()
    if not args.real_policy:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    cfg = load_deploy_cfg()
    log(f"[smoke] deploy={DEPLOY_YML}")
    log(
        f"[smoke] history_enabled={cfg.get('history_enabled')} mode={cfg.get('history_mode')} "
        f"slots={cfg.get('history_slots')} interval={cfg.get('history_action_interval')} "
        f"slot_size={cfg.get('history_slot_size')} ckpt={cfg.get('ckpt_name')}@{cfg.get('checkpoint_num')}"
    )
    if not cfg.get("history_enabled") or cfg.get("history_mode") != "slots":
        check("deploy.yml 已切到 slots 记忆评测配置", False, "history_enabled/history_mode 不是 slots")
        return 1

    pi05_model = import_pi05_model()
    if args.real_policy:
        run_real(pi05_model, cfg, args)
    else:
        run_fake(pi05_model, cfg, args)

    if FAILURES:
        log(f"[smoke] ✗ 失败 {len(FAILURES)} 项：{FAILURES}")
        return 1
    log("[smoke] ✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
