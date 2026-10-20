#!/usr/bin/env python3
"""B3 captioner 实测：从真实 RoboDojo 评测录像抽帧 → Qwen2.5-VL-3B 生成描述。

验证：模型能否在 GPU0 加载、生成描述是否合理、单帧延迟与显存占用。

用法：
  robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python scripts/pi05/test_b3_captioner.py
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/Qwen2.5-VL-3B-Instruct"
VIDEO_GLOB = "robodojo/eval_result/RoboDojo/cover_blocks/Pi_05/arx_x5/*/2026-10-04_19-26-57_cover_blocks_s0/episode_0000000_cam_head_fail.mp4"
PROMPT = (
    "In under 20 words, list the visible objects with their colors and left/right positions. "
    "Start directly with the objects, like: red cube left, blue cube center, green cylinder right."
)


def main() -> int:
    if not (MODEL_DIR / "config.json").is_file():
        print(f"[B3] ✗ 模型未就绪: {MODEL_DIR}")
        return 2

    import cv2
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    video_path = next(iter(sorted(ROOT.glob(VIDEO_GLOB))), None)
    if video_path is None:
        print("[B3] ✗ 找不到录像")
        return 2
    print(f"[B3] video={video_path}")
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for ratio in (0.05, 0.5, 0.95):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * ratio) - 1))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(frame[:, :, ::-1]))
    cap.release()
    print(f"[B3] 抽帧 {len(frames)} 张（video 共 {total} 帧）")
    out_dir = ROOT / "logs/pi05/b3_captioner_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(out_dir / f"frame_{index}.png")

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    print(f"[B3] 模型加载 {time.time() - t0:.1f}s  torch.cuda 显存 {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    for index, frame in enumerate(frames):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[frame], return_tensors="pt").to("cuda:0")
        t0 = time.time()
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        dt = time.time() - t0
        caption = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        print(f"[B3] frame{index}: {dt:.2f}s  {caption.strip()}")
    print(f"[B3] 峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print("[B3] ★ captioner 实测完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
