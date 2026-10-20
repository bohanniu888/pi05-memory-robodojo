#!/usr/bin/env python3
"""Phase 9 CPU 单测：断时钟（pad_mode）/ 写侧直注（history_inject）/ 内容对比（contrast）三件套的
"数据与语义"部分（不碰 GPU / 不加载权重）。

覆盖：
  T1 训练侧 transform pad_mode="zero"（旧行为）：可见槽位数 = min(frame_index//25, 20)，右对齐；
  T2 训练侧 transform pad_mode="repeat_oldest"：可见槽位数恒为 slots（断"剧集时钟"），左侧用最旧帧回填；
  T3 pad_mode="repeat_oldest" 在窗口已满时与旧口径逐帧一致（不改变真实历史内容）；
  T4 frame_index < interval / 无 sidecar：两种 pad_mode 都返回全 pad（"无历史"语义务必保留）；
  T5 评测侧 _attach_history(pad_mode="repeat_oldest") 与训练侧同口径；
  T6 评测侧一致性断言：pad_mode 不一致时拒绝启动；
  T7 写侧 _identity_kernel_init：kernel·x == x（恒等直注语义）。

运行：
  robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python scripts/pi05/test_pi05_history_v2.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
XPL_ROOT = ROOT / "robodojo/XPolicyLab"
OPENPI_SRC = ROOT / "robodojo/XPolicyLab/policy/Pi_05/openpi/src"
for path in (str(OPENPI_SRC), str(XPL_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from openpi.models.pi0 import _identity_kernel_init  # noqa: E402
from openpi.policies.robodojo_history import RoboDojoHistoryTransform  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from XPolicyLab.policy.Pi_05 import model as pi05_model  # noqa: E402

SIDECAR = ROOT / "robodojo/data/history_sidecar"
EPISODE = 300  # cover_blocks/ep_0000300.npy，length=555，num_samples=23
INTERVAL = 25


def _transform(pad_mode: str) -> RoboDojoHistoryTransform:
    return RoboDojoHistoryTransform(sidecar_root=str(SIDECAR), slots=20, interval=INTERVAL, pad_mode=pad_mode)


def _call(pad_mode: str, episode_index=EPISODE, frame_index=60):
    return _transform(pad_mode)({"episode_index": episode_index, "frame_index": frame_index})


def _expected_frames(ks):
    frames = np.load(SIDECAR / f"cover_blocks/ep_0000300.npy", mmap_mode="r")
    return np.asarray(frames[list(ks)])


def test_zero_pad_is_legacy_clock():
    out = _call("zero", frame_index=60)  # k_max = 60//25 - 1 = 1 → 2 个可用槽
    mask = np.asarray(out["history_mask"])
    assert mask.sum() == 2 and not mask[:-2].any(), mask
    frames = np.asarray(out["history_images"])
    assert np.array_equal(frames[-2:], _expected_frames([0, 1]))
    assert np.array_equal(frames[:-2], np.zeros_like(frames[:-2]))
    # 单调"剧集时钟"：可用槽位数随 frame_index 单调增（这正是 Phase 9 要切断的捷径）
    counts = [int(np.asarray(_call("zero", frame_index=f)["history_mask"]).sum()) for f in (0, 10, 25, 60, 200, 550)]
    assert counts == [0, 0, 1, 2, 8, 20], counts
    print(f"[T1] pad_mode=zero 旧口径：可见槽位数随时钟单调增 {counts} ✅")


def test_repeat_oldest_kills_clock():
    out = _call("repeat_oldest", frame_index=60)
    mask = np.asarray(out["history_mask"])
    frames = np.asarray(out["history_images"])
    assert mask.all() and mask.shape == (20,), mask
    # 右侧 2 槽 = 真实历史（k=0,1），左侧 18 槽 = 最旧有效帧（k=0）回填
    assert np.array_equal(frames[-2:], _expected_frames([0, 1]))
    assert all(np.array_equal(frames[i], frames[-2]) for i in range(18))
    counts = [
        int(np.asarray(_call("repeat_oldest", frame_index=f)["history_mask"]).sum())
        for f in (0, 10, 25, 60, 200, 550)
    ]
    assert counts == [0, 0, 20, 20, 20, 20], counts
    print(f"[T2] pad_mode=repeat_oldest：可见槽位数恒定（0/0/20/20/20/20），时钟通道切断 ✅")


def test_repeat_oldest_full_window_unchanged():
    old = _call("zero", frame_index=550)
    new = _call("repeat_oldest", frame_index=550)
    assert np.array_equal(np.asarray(old["history_images"]), np.asarray(new["history_images"]))
    assert np.array_equal(np.asarray(old["history_mask"]), np.asarray(new["history_mask"]))
    assert np.asarray(new["history_mask"]).all()
    assert np.array_equal(np.asarray(new["history_images"])[-1], _expected_frames([21])[0])
    print("[T3] 窗口已满时 repeat_oldest 与旧口径逐位一致（不改变真实历史）✅")


def test_no_history_still_all_pad():
    for pad_mode in ("zero", "repeat_oldest"):
        early = _call(pad_mode, frame_index=10)  # k_max = -1：还没有"够旧"的帧
        assert not np.asarray(early["history_mask"]).any()
        assert not np.asarray(early["history_images"]).any()
        missing = _call(pad_mode, episode_index=999999, frame_index=500)  # 无 sidecar
        assert not np.asarray(missing["history_mask"]).any()
        assert not np.asarray(missing["history_images"]).any()
    print("[T4] 无历史（首秒 / 无 sidecar）仍为全 pad：'无历史'语义保留 ✅")


def _make_eval_model(interval=25, slots=20, pad_mode="zero"):
    parent = pi05_model.Model.__new__(pi05_model.Model)
    parent._configure_history(
        {
            "history_enabled": True,
            "history_mode": "slots",
            "history_image_key": "cam_high",
            "history_slots": slots,
            "history_fps": 1.0,
            "runtime_fps": interval,
            "history_action_interval": interval,
            "history_pad_mode": pad_mode,
        }
    )
    parent._history_by_env = {}
    parent._history_sequence_by_env = {}
    parent._history_logged_envs = set()
    return parent


def _obs(value):
    return {"vision": {"cam_high": np.full((3, 224, 224), value, dtype=np.uint8)}, "env_idx": 0}


def test_eval_side_matches_training_pad_mode():
    parent = _make_eval_model(pad_mode="repeat_oldest")
    for step in range(30):  # 30 步只产生 1 个"够旧"的帧（与训练侧 frame_index=60 同口径）
        parent._append_history_observation(_obs(step % 256), 0)
    attached = parent._attach_history({"images": {"cam_high": np.zeros((3, 224, 224), np.uint8)}}, 0)
    mask = attached["history_mask"]
    history = attached["history_images"]
    assert mask.all() and mask.shape == (20,), mask
    assert int(history[-1][0, 0, 0]) == 0 and int(history[0][0, 0, 0]) == 0
    legacy = _make_eval_model(pad_mode="zero")
    for step in range(30):
        legacy._append_history_observation(_obs(step % 256), 0)
    old = legacy._attach_history({"images": {"cam_high": np.zeros((3, 224, 224), np.uint8)}}, 0)
    assert int(old["history_mask"].sum()) == 1
    print("[T5] 评测侧 pad_mode=repeat_oldest 与训练侧同口径（20 槽全可见）✅")


def test_eval_consistency_assertion():
    parent = _make_eval_model()  # deploy 侧 pad_mode=zero
    parent._train_config_name = "pi05_aloha_sim_history_v2_arx-x5_seed_0"
    parent._train_config = _config.get_config(parent._train_config_name)
    spec = parent._history_train_spec()
    assert spec["pad_mode"] == "repeat_oldest", spec
    assert spec["interval"] == 25, spec
    try:
        parent._assert_history_eval_consistency()
    except ValueError as exc:
        assert "pad_mode" in str(exc), exc
    else:
        raise AssertionError("pad_mode 不一致时应当拒绝启动")
    parent.history_pad_mode = "repeat_oldest"  # 对齐后应通过（其余口径 v2 配置与 deploy 一致）
    parent._assert_history_eval_consistency()
    print("[T6] 评测/训练 pad_mode 一致性断言生效 ✅")


def test_identity_kernel_init():
    import jax

    kernel = np.asarray(_identity_kernel_init(jax.random.key(0), (8, 8), np.float32))
    assert np.array_equal(kernel, np.eye(8))
    x = np.random.default_rng(0).normal(size=(3, 8))
    assert np.allclose(x @ kernel, x)
    kernel_rect = np.asarray(_identity_kernel_init(jax.random.key(0), (5, 8), np.float32))
    assert kernel_rect.shape == (5, 8) and kernel_rect.sum() == 5
    print("[T7] _identity_kernel_init：恒等直注语义（kernel·x == x）✅")


def main() -> int:
    test_zero_pad_is_legacy_clock()
    test_repeat_oldest_kills_clock()
    test_repeat_oldest_full_window_unchanged()
    test_no_history_still_all_pad()
    test_eval_side_matches_training_pad_mode()
    test_eval_consistency_assertion()
    test_identity_kernel_init()
    print("\n[Phase 9] 全部 CPU 单测通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
