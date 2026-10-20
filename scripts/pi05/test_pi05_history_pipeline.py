#!/usr/bin/env python3
"""Phase 3 公共前置 CPU 单测：B1 拼贴图历史管线（不碰 GPU / 模型权重）。

覆盖：
  1. AlohaInputs：无 cam_history 时输出恰好 3 路图像（基线前缀长度不变）；
  2. AlohaInputs：带 cam_history 时输出 4 路图像（history_0_rgb，mask=True）；
  3. ResizeImages(224) 不丢第 4 路图像；
  4. 历史 deque 抽帧语义（DM0.5 口径：每 interval 个动作步一帧、消费时排除过新帧、左侧补 None）；
  5. build_history_collage：网格拼贴、缺帧补黑、象限内容正确；
  6. history_enabled=False / reset 行为。

运行：
  robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python scripts/pi05/test_pi05_history_pipeline.py
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

from openpi import transforms as _transforms  # noqa: E402
from openpi.policies.aloha_policy import AlohaInputs  # noqa: E402
from XPolicyLab.policy.Pi_05 import model as pi05_model  # noqa: E402


def _images(value=0, with_history=False):
    images = {
        name: np.full((3, 224, 224), value, dtype=np.uint8)
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    }
    if with_history:
        images["cam_history"] = np.full((3, 224, 224), 99, dtype=np.uint8)
    return images


def test_aloha_inputs_baseline_unchanged():
    transform = AlohaInputs(adapt_to_pi=False)
    out = transform({"state": np.zeros(14, np.float32), "images": _images(), "prompt": "cover the blocks"})
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}, tuple(out["image"])
    assert "history_0_rgb" not in out["image_mask"]
    print("[T1] 基线 3 路图像：无第 4 路注入 ✅")


def test_aloha_inputs_with_history():
    transform = AlohaInputs(adapt_to_pi=False)
    out = transform(
        {"state": np.zeros(14, np.float32), "images": _images(with_history=True), "prompt": "cover the blocks"}
    )
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "history_0_rgb"}
    assert bool(out["image_mask"]["history_0_rgb"]) is True
    assert out["image"]["history_0_rgb"].shape == (224, 224, 3)
    print("[T2] 带 cam_history：注入 history_0_rgb（mask=True）✅")


def test_resize_keeps_history_image():
    # 真实 transform 顺序：AlohaInputs（CHW→HWC + 命名）在前，ResizeImages 在后。
    aloha = AlohaInputs(adapt_to_pi=False)
    out = aloha({"state": np.zeros(14, np.float32), "images": _images(with_history=True), "prompt": "x"})
    out = _transforms.ResizeImages(224, 224)(out)
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "history_0_rgb"}
    assert all(image.shape[:2] == (224, 224) for image in out["image"].values())
    print("[T3] AlohaInputs→ResizeImages 保留第 4 路图像 ✅")


def _make_model(interval=25, slots=20, mode="collage", multi_frames=3):
    parent = pi05_model.Model.__new__(pi05_model.Model)
    parent._configure_history(
        {
            "history_enabled": True,
            "history_mode": mode,
            "history_image_key": "cam_high",
            "history_slots": slots,
            "history_fps": 1.0,
            "runtime_fps": interval,
            "history_action_interval": interval,
            "history_collage_frames": 4,
            "history_multi_frames": multi_frames,
        }
    )
    parent._history_by_env = {}
    parent._history_sequence_by_env = {}
    parent._history_logged_envs = set()
    parent._captioner = None
    parent._caption_cache = {}
    parent._caption_failed_envs = set()
    parent.observation_window = None
    parent._latest_env_idx_list = [0]
    return parent


def _obs(value):
    return {"vision": {"cam_high": np.full((3, 224, 224), value, dtype=np.uint8)}, "env_idx": 0}


def test_history_deque_semantics():
    parent = _make_model()
    for step in range(30):
        parent._append_history_observation(_obs(step % 256), 0)
    frames = parent._history_frames_for(0)
    assert len(frames) == 20 and sum(frame is not None for frame in frames) == 1, frames
    assert int(frames[-1][0, 0, 0]) == 0  # 30 步时只有第 0 步的帧可用（过滤掉过新的第 25 步）

    for step in range(30, 125):
        parent._append_history_observation(_obs(step % 256), 0)
    frames = parent._history_frames_for(0)
    usable = [frame for frame in frames if frame is not None]
    assert len(frames) == 20 and len(usable) == 4, len(usable)
    assert [int(frame[0, 0, 0]) for frame in usable] == [0, 25, 50, 75]

    # 不同 env 互不串扰：env1 只走 1 步时帧"太新"被过滤（0 帧可用），补足步数后只用自己的帧。
    parent._append_history_observation(_obs(200), 1)
    assert sum(frame is not None for frame in parent._history_frames_for(1)) == 0
    for _ in range(30):
        parent._append_history_observation(_obs(201), 1)
    env1_usable = [frame for frame in parent._history_frames_for(1) if frame is not None]
    assert len(env1_usable) == 1 and int(env1_usable[0][0, 0, 0]) == 200
    print("[T4] deque 抽帧/过滤/补 None/env 隔离 ✅")


def test_collage_layout_and_attach():
    parent = _make_model()
    for step in range(125):
        parent._append_history_observation(_obs(step % 256), 0)
    usable = [frame for frame in parent._history_frames_for(0) if frame is not None][-4:]
    collage = pi05_model.build_history_collage(usable, 4, 224)
    assert collage.shape == (3, 224, 224) and collage.dtype == np.uint8
    quadrants = [collage[0, 0, 0], collage[0, 0, -1], collage[0, -1, 0], collage[0, -1, -1]]
    assert [int(value) for value in quadrants] == [0, 25, 50, 75], quadrants

    # 只有 1 帧时补黑（占右下角）
    single = pi05_model.build_history_collage([np.full((3, 224, 224), 7, dtype=np.uint8)], 4, 224)
    assert int(single[0, 0, 0]) == 0 and int(single[0, -1, -1]) == 7

    observation = {"images": {"cam_high": np.zeros((3, 224, 224), np.uint8)}}
    attached = parent._attach_history(observation, 0)
    assert attached["images"]["cam_history"].shape == (3, 224, 224)
    print("[T5] 拼贴图网格/缺帧补黑/attach ✅")


def test_disabled_and_reset():
    parent = _make_model()
    parent.history_enabled = False
    observation = {"images": {"cam_high": np.zeros((3, 224, 224), np.uint8)}}
    assert "cam_history" not in parent._attach_history(observation, 0)["images"]

    parent.history_enabled = True
    for step in range(10):
        parent._append_history_observation(_obs(step), 0)
    parent.reset()
    assert parent._history_by_env == {} and parent._history_sequence_by_env == {}
    print("[T6] 关闭开关不注入 / reset 清空 ✅")


def test_multi_frame_attach_and_aloha_mapping():
    parent = _make_model(mode="multi", multi_frames=3)
    for step in range(30):
        parent._append_history_observation(_obs(step % 256), 0)
    observation = {"images": {"cam_high": np.zeros((3, 224, 224), np.uint8)}}
    attached = parent._attach_history(observation, 0)
    # 30 步时只有 1 帧可用 → 只注入 cam_history_0
    assert {k for k in attached["images"] if k.startswith("cam_history")} == {"cam_history_0"}

    for step in range(30, 125):
        parent._append_history_observation(_obs(step % 256), 0)
    observation = {"images": {"cam_high": np.zeros((3, 224, 224), np.uint8)}}
    attached = parent._attach_history(observation, 0)
    history_keys = {k for k in attached["images"] if k.startswith("cam_history")}
    assert history_keys == {"cam_history_0", "cam_history_1", "cam_history_2"}, history_keys
    values = [int(attached["images"][f"cam_history_{i}"][0, 0, 0]) for i in range(3)]
    assert values == [25, 50, 75], values  # 最旧在前、最新在最后

    images = dict(attached["images"])
    for name in ("cam_left_wrist", "cam_right_wrist"):
        images[name] = np.zeros((3, 224, 224), np.uint8)
    out = AlohaInputs(adapt_to_pi=False)(
        {"state": np.zeros(14, np.float32), "images": images, "prompt": "cover the blocks"}
    )
    expected = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "history_0_rgb", "history_1_rgb", "history_2_rgb"}
    assert set(out["image"]) == expected, tuple(out["image"])
    assert all(bool(out["image_mask"][f"history_{i}_rgb"]) for i in range(3))
    print("[T7] B2 多帧多图 attach + AlohaInputs 映射 ✅")


def test_text_mode_prompt_injection():
    parent = _make_model(mode="text", multi_frames=3)
    calls: list[int] = []

    def fake_caption(frame):
        calls.append(int(frame[0, 0, 0]))
        return f"a block at x={int(frame[0, 0, 0])}"

    parent._generate_caption = fake_caption
    for step in range(125):
        parent._append_history_observation(_obs(step % 256), 0)
    observation = {"images": {"cam_high": np.zeros((3, 224, 224), np.uint8)}, "prompt": "cover the blocks"}
    attached = parent._attach_history(observation, 0)
    prompt = attached["prompt"]
    assert prompt == "cover the blocks Memory: earlier a block at x=0; recently a block at x=75.", prompt
    before = len(calls)
    parent._attach_history({"images": {}, "prompt": "cover the blocks"}, 0)
    assert len(calls) == before, (calls, before)  # 同帧命中缓存，不重复调用 captioner
    parent.reset()
    assert parent._caption_cache == {}
    print("[T8] B3 text 模式 prompt 注入 + caption 缓存 ✅")


def main() -> int:
    test_aloha_inputs_baseline_unchanged()
    test_aloha_inputs_with_history()
    test_resize_keeps_history_image()
    test_history_deque_semantics()
    test_collage_layout_and_attach()
    test_disabled_and_reset()
    test_multi_frame_attach_and_aloha_mapping()
    test_text_mode_prompt_injection()
    print("★ Phase 3 公共前置 CPU 单测全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
