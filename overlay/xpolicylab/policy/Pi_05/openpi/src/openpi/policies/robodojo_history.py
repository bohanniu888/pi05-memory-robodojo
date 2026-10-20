"""RoboDojo 历史帧（memory）训练侧 transform —— Phase 4。

读取 ``build_history_sidecar.py`` 生成的 sidecar（``index.json`` + ``<task>/ep_XXX.npy``），
为当前训练样本组装 ``slots`` 个历史槽位（口径与 XPolicyLab 评测侧 ``_history_entries_for`` 一致）：

* 历史采样为 1 Hz（每 ``interval`` 帧取 1 帧，k ↔ episode 内 local frame ``k * interval``）；
* 只取严格早于当前帧的样本：``k <= frame_index // interval - 1``（排除“过新一帧”）；
* 最近 ``slots`` 个、最旧在左；不足左侧补 pad（mask=False）；
* 无 sidecar 的样本返回全 pad（供通用数据 cotrain 使用，模型学习“无历史时正常动作”）。

输出顶层 ``history_images`` (slots,3,H,W) uint8 与 ``history_mask`` (slots,) bool，
由 ``Observation.from_dict`` 转成 [-1,1] float32；图像经 ``preprocess_observation``
resize_with_pad 到模型分辨率。默认关闭：只有配置了 ``history_sidecar_root`` 的 TrainConfig 才会挂载。

[pi05-keyframe] 关键帧加权（Phase 8）：可选 ``keyframe_weights_path`` 指向权重表
``{"interval": 25, "weights": {"<episode_index>": [w0, w1, ...]}}``（1 Hz 分辨率，按
``frame_index // interval`` 取第 k 个权重）。配置后额外输出 ``loss_weight``（float32 标量，
缺省/未列出 episode 为 1.0），由 ``train.py`` 做归一化加权均值；不配置时完全不输出该键。
"""

from __future__ import annotations

import dataclasses
import json
import os
from functools import lru_cache
from typing import Any

import numpy as np

from openpi import transforms


@lru_cache(maxsize=8)
def _load_index(sidecar_root: str) -> dict[str, dict[str, Any]]:
    with open(os.path.join(sidecar_root, "index.json"), encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=4)
def _load_keyframe_weights(path: str) -> tuple[int, dict[str, list[float]]]:
    """[pi05-keyframe] 读取关键帧权重表（1 Hz 分辨率，episode → 权重数组）。"""
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    weights = blob.get("weights", blob)
    interval = int(blob.get("interval", 25))
    return interval, weights


def _keyframe_weight(path: str, episode_index: Any, frame_index: Any) -> float:
    """[pi05-keyframe] 查当前样本的 1 Hz 权重；未列出/越界/缺元信息时返回 1.0。"""
    if episode_index is None or frame_index is None:
        return 1.0
    interval, weights = _load_keyframe_weights(path)
    slots = weights.get(str(int(episode_index)))
    if not slots:
        return 1.0
    k = int(frame_index) // max(1, interval)
    if 0 <= k < len(slots):
        return float(slots[k])
    return 1.0


@dataclasses.dataclass(frozen=True)
class RoboDojoHistoryTransform(transforms.DataTransformFn):
    """组 20 槽历史帧 + mask；sidecar 缺失/无历史时返回全 pad。"""

    sidecar_root: str
    slots: int = 20
    interval: int = 25
    height: int = 128
    width: int = 128
    # [pi05-keyframe] Phase 8：可选关键帧权重表；None = 不加权（默认，输出与旧版逐位一致）。
    keyframe_weights_path: str | None = None
    # [pi05-history] Phase 9：pad 槽处理方式。
    #   "zero"（默认，旧行为）= 空槽补 0 + mask=False（可见槽位数 = min(frame_index//25, 20)，
    #       是单调的"剧集时钟"，模型可据此走捷径 —— Phase 8 已证实该捷径确实存在）；
    #   "repeat_oldest" = 用最旧的有效帧回填左侧空槽、mask 全 True（可见槽位数恒为 slots，
    #       注入前缀的 token 数恒定），切断"数槽位/数 token"的时钟通道。
    pad_mode: str = "zero"

    def __post_init__(self):
        if self.pad_mode not in ("zero", "repeat_oldest"):
            raise ValueError(f"unsupported pad_mode: {self.pad_mode}")

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        history = np.zeros((self.slots, 3, self.height, self.width), dtype=np.uint8)
        mask = np.zeros((self.slots,), dtype=np.bool_)

        episode_index = data.get("episode_index")
        frame_index = data.get("frame_index")
        if episode_index is not None and frame_index is not None:
            entry = _load_index(self.sidecar_root).get(str(int(episode_index)))
            if entry is not None:
                frames = np.load(os.path.join(self.sidecar_root, entry["path"]), mmap_mode="r")
                k_max = int(frame_index) // self.interval - 1
                if k_max >= 0:
                    n_take = min(k_max + 1, self.slots, frames.shape[0])
                    ks = range(k_max - n_take + 1, k_max + 1)
                    for offset, k in enumerate(ks):
                        if 0 <= k < frames.shape[0]:
                            history[self.slots - n_take + offset] = frames[k]
                            mask[self.slots - n_take + offset] = True
                    # [pi05-history] Phase 9：断时钟捷径 —— 左侧空槽用最旧有效帧回填，mask 全 True，
                    # 可见槽位数恒为 slots（不再随时间从 0 单调增到 20）。
                    if self.pad_mode == "repeat_oldest" and n_take < self.slots:
                        history[: self.slots - n_take] = history[self.slots - n_take]
                        mask[: self.slots - n_take] = True

        data["history_images"] = history
        data["history_mask"] = mask
        if self.keyframe_weights_path is not None:
            data["loss_weight"] = np.float32(
                _keyframe_weight(self.keyframe_weights_path, episode_index, frame_index)
            )
        return data
