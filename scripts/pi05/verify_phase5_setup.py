#!/usr/bin/env python3
"""Phase 5 公共前置验收：训练侧代码就位检查（不训练）。

检查项：
  A. 冻结口径：VLM 主干冻结（LoRA 可训）、SigLIP 冻结、action expert 全参可训、history_proj 可训；
  B. 零初始化：config.model.create() 的 history_proj kernel/bias 全 0（Phase 5 关键技巧 1）；
  C. 权重加载：CheckpointWeightLoader 结构完整（含 history 键）、history 键以 init 值占位（后续被丢弃=保留零初始化）；
  D. 数据管线：cotrain 混采 episode 过滤 + RoboDojoHistoryTransform 出样本。

用法（GPU 作业内）：
  srun --jobid=<job> --overlap bash -c "CUDA_VISIBLE_DEVICES=3 python scripts/pi05/verify_phase5_setup.py"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPENPI_SRC = ROOT / "robodojo/XPolicyLab/policy/Pi_05/openpi/src"
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "robodojo/data/hf_lerobot_home"))
os.environ.setdefault("ROBODOJO_HISTORY_SIDECAR_DIR", str(ROOT / "robodojo/data/history_sidecar"))
sys.path.insert(0, str(OPENPI_SRC))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx, traverse_util  # noqa: E402
from flax.nnx import filterlib  # noqa: E402

import openpi.shared.array_typing as at  # noqa: E402
from openpi.training import config as training_config  # noqa: E402

CFG_NAME = "pi05_aloha_sim_history_arx-x5_seed_0"


def check_freeze(cfg) -> None:
    shape = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    _, state = nnx.split(shape)
    flat = traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
    frozen = filterlib.to_predicate(cfg.freeze_filter)

    groups: dict[str, list[int]] = {}
    for path, leaf in flat.items():
        key = tuple(path.split("/"))
        if path.startswith("PaliGemma/img"):
            tag = "img(SigLIP)"
        elif path.startswith("PaliGemma/llm") and any(p.endswith("_1") for p in key):
            tag = "llm_1(action expert)"
        elif path.startswith("PaliGemma/llm"):
            tag = "llm(VLM)"
        else:
            tag = "other(history/action io)"
        g = groups.setdefault(tag, [0, 0])
        g[0 if frozen(key, leaf) else 1] += 1
    for tag, (n_frozen, n_train) in sorted(groups.items()):
        print(f"[A] {tag:24s} frozen={n_frozen:3d} trainable={n_train:3d}")
    assert groups["img(SigLIP)"][1] == 0, "SigLIP 未完全冻结"
    assert groups["llm_1(action expert)"][0] == 0, "action expert 不应被冻结"
    for probe in ("history_proj/kernel", "history_proj/bias"):
        assert not frozen(tuple(probe.split("/")), flat[probe]), f"{probe} 被冻结了"
    assert frozen(("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"), None), "VLM 主干未冻结"
    print("[A] PASS")


def check_zero_init(cfg) -> None:
    model = cfg.model.create(jax.random.key(0))
    _, state = nnx.split(model)
    for key in ("history_proj/kernel", "history_proj/bias"):
        leaf = state.to_pure_dict()
        for part in key.split("/"):
            leaf = leaf[part]
        arr = np.asarray(leaf)
        print(f"[B] {key}: shape={arr.shape} absmax={np.abs(arr).max():.3e}")
        assert np.abs(arr).max() == 0.0, f"{key} 非零初始化"
    del model, state
    print("[B] PASS")


def check_weight_loader(cfg) -> None:
    shape = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    _, state = nnx.split(shape)
    pure = state.to_pure_dict()
    partial = cfg.weight_loader.load(pure)
    at.check_pytree_equality(expected=pure, got=partial, check_shapes=True, check_dtypes=True)
    flat = traverse_util.flatten_dict(partial, sep="/")
    n_sd = sum(1 for v in flat.values() if isinstance(v, jax.ShapeDtypeStruct))
    hist_sd = [k for k, v in flat.items() if "history" in k and isinstance(v, jax.ShapeDtypeStruct)]
    print(f"[C] merged leaves={len(flat)} placeholder(init-kept)={n_sd} history_placeholders={sorted(hist_sd)}")
    assert set(hist_sd) == {"history_proj/bias", "history_proj/kernel"}, hist_sd
    real = next(v for v in flat.values() if isinstance(v, np.ndarray))
    assert real.dtype is not None
    print("[C] PASS")


def check_data_pipeline(cfg, num_samples: int = 4) -> None:
    from openpi.training import data_loader as _data_loader

    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    print(f"[D] episodes filter: {None if data_config.episodes is None else len(data_config.episodes)}")
    assert data_config.episodes is not None, "cotrain 混采清单未生效"
    dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    transformed = _data_loader.transform_dataset(dataset, data_config)
    idxs = [0, len(transformed) // 3, len(transformed) // 2, len(transformed) - 1]
    for i in idxs[:num_samples]:
        sample = transformed[i]
        hist = np.asarray(sample["history_images"])
        mask = np.asarray(sample["history_mask"])
        print(f"[D] sample {i}: ep={int(sample.get('episode_index', -1))} "
              f"history={hist.shape}/{hist.dtype} mask_true={int(mask.sum())} "
              f"images={ {k: np.asarray(v).shape for k, v in sample['image'].items()} }")
        assert hist.shape == (20, 3, 128, 128) and hist.dtype == np.uint8
        assert mask.shape == (20,)
    print("[D] PASS")


def main() -> int:
    cfg = training_config.get_config(CFG_NAME)
    print(f"[init] config={cfg.name} devices={jax.device_count()} backend={jax.default_backend()}")
    check_freeze(cfg)
    if jax.default_backend() == "gpu":
        check_zero_init(cfg)
    else:
        print("[B] SKIP (非 GPU 后端)")
    check_weight_loader(cfg)
    check_data_pipeline(cfg)
    print("[verify] ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
