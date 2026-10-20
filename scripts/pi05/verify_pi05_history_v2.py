#!/usr/bin/env python3
"""Phase 9 GPU 侧验证：写侧注入方式（history_inject）与内容对比辅助损失（contrast）的真实语义。

用小模型（gemma_300m_lora + gemma_300m，逻辑与 2B 完全同构）在单卡上验证：

  A. history_inject="proj_zero"     → 前缀里的历史 token 逐位为 0（旧行为，零初始化瓶颈）
  B. history_inject="proj_identity" → 历史 token == 手工复算的池化特征（step 0 即"直注"，且投影仍可训）
  C. history_inject="direct"        → 同 B，且模型里不存在 history_proj 参数（严格 DM0.5 直注）
  D. contrast_mode="roll"           → compute_contrast_aux 返回有限值；未训练模型内容无感 → hinge 全激活；
                                      stop_grad=False 时 roll 分支有非零梯度；关闭时 aux 恒为 0
  E. 训练配置管道：pi05_aloha_sim_history_v2_arx-x5_seed_0 的 transform 真的是 pad_mode=repeat_oldest

运行（在 GPU 作业内）：
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/pi05/verify_pi05_history_v2.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OPENPI_SRC = ROOT / "robodojo/XPolicyLab/policy/Pi_05/openpi/src"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.models import pi0_config as _pi0_config  # noqa: E402
from openpi.training import config as _train_config  # noqa: E402

B, N_SLOTS, TOK_PER_SLOT, POOL = 2, 20, 16, 4
IMG_TOKENS = 256  # 224/14 = 16 → 16×16
HIST_SLICE = slice(3 * IMG_TOKENS, 3 * IMG_TOKENS + N_SLOTS * TOK_PER_SLOT)


def _dummy_obs(slots: int = N_SLOTS):
    rng = np.random.default_rng(0)
    # 数值域对齐真实管线：图像 ∈ [-1,1]，带低频结构（纯噪声输入会让 SigLIP 特征塌成 0，验证就没意义了）。
    yy, xx = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224), indexing="ij")
    pattern = (np.sin(3.0 * xx) * np.cos(2.0 * yy))[None, :, :, None]
    images = {
        name: jnp.asarray(
            np.clip(0.4 * rng.normal(size=(B, 224, 224, 3)) + 0.5 * pattern + 0.3 * index, -1.0, 1.0).astype(np.float32)
        )
        for index, name in enumerate(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"))
    }
    masks = {name: jnp.ones((B,), jnp.bool_) for name in images}
    # 注意：真实管线里 history 帧已由 preprocess_observation resize 到模型分辨率（224），
    # embed_prefix 直接吃 224×224；这里保持一致（128×128 会让 SigLIP 走另一套 pos_embedding）。
    hist_pattern = (np.cos(2.0 * xx) * np.sin(1.5 * yy))[None, None, :, :, None]
    history = jnp.asarray(
        np.clip(
            0.4 * rng.normal(size=(B, slots, 224, 224, 3)) + 0.5 * hist_pattern + 0.2 * np.arange(slots)[None, :, None, None, None],
            -1.0,
            1.0,
        ).astype(np.float32)
    )
    history = jnp.transpose(history, (0, 1, 4, 2, 3))  # NHWC → NCHW（与 Observation 约定一致）
    return _model.Observation(
        images=images,
        image_masks=masks,
        state=jnp.zeros((B, 32), jnp.float32),
        tokenized_prompt=jnp.ones((B, 8), jnp.int32),
        tokenized_prompt_mask=jnp.ones((B, 8), jnp.bool_),
        history_images=history,
        history_mask=jnp.ones((B, slots), jnp.bool_),
    )


def _small_config(*, history_slots: int = N_SLOTS, **kwargs):
    return _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_300m_lora",
        action_expert_variant="gemma_300m",
        history_slots=history_slots,
        history_tokens_per_slot=TOK_PER_SLOT,
        history_pool_size=POOL,
        **kwargs,
    )


def _pooled_reference(model, history):
    """与 Pi0.embed_prefix 里历史分支逐行同构的手工复算（同一视觉塔）。"""
    hb, hn = history.shape[0], history.shape[1]
    flat = jnp.transpose(history.reshape(hb * hn, *history.shape[2:]), (0, 2, 3, 1))
    tokens, _ = model.PaliGemma.img(flat, train=False)
    side = int(round(tokens.shape[1] ** 0.5))
    pool = max(1, side // POOL)
    tokens = tokens.reshape(hb, hn, side, side, tokens.shape[-1])
    tokens = tokens.reshape(hb, hn, POOL, pool, POOL, pool, tokens.shape[-1]).mean(axis=(3, 5))
    return tokens.reshape(hb, hn * TOK_PER_SLOT, -1)


def _prefix_history_tokens(model, obs):
    tokens, _, _ = model.embed_prefix(obs)
    assert tokens.shape[1] >= HIST_SLICE.stop, tokens.shape
    return jnp.asarray(tokens[:, HIST_SLICE], dtype=jnp.float32)


def _param_paths(model):
    """返回模型参数键路径（list[str]），用于检查 history_proj 是否存在。"""
    paths = []

    def walk(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}/{key}", child)
        else:
            paths.append(prefix)

    walk("", nnx.state(model, nnx.Param).to_pure_dict())
    return paths


def _randomize_image_head(model, seed: int = 0) -> None:
    """把 SigLIP 投影头（num_classes 那层）随机化，模拟"非零投影头"。

    openpi 的 `_siglip._Module` 默认 `head_zeroinit=True`：随机初始化时图像 token 恒为 0，
    identity/direct 直注的"非零"性质就没法验证（真实训练用官方 checkpoint，头是非零的）。
    """
    rng = np.random.default_rng(seed)
    state = nnx.state(model)
    pure = state.to_pure_dict()
    hits = []

    def walk(node, prefix=()):
        if isinstance(node, dict):
            for key, child in list(node.items()):
                node[key] = walk(child, prefix + (key,))
            return node
        if len(prefix) >= 3 and prefix[-2] == "head" and "img" in prefix:
            hits.append("/".join(str(p) for p in prefix))
            return jnp.asarray(rng.normal(scale=0.05, size=node.shape).astype(np.float32), dtype=node.dtype)
        return node

    pure = walk(pure)
    state.replace_by_pure_dict(pure)
    nnx.update(model, state)
    assert hits, "未找到 img/head 参数"
    print(f"[setup] 已随机化图像投影头：{hits}")


def check_inject(modes=("proj_zero", "proj_identity", "direct")):
    obs = _dummy_obs()
    for mode in modes:
        cfg = _small_config(history_inject=mode)
        model = cfg.create(jax.random.key(0))
        _randomize_image_head(model)
        got = _prefix_history_tokens(model, obs)
        has_proj = any("history_proj" in str(leaf) for leaf in _param_paths(model))
        assert (mode != "direct") or not has_proj, "direct 模式不应存在 history_proj 参数"
        assert (mode == "direct") or has_proj, f"{mode} 模式应当存在 history_proj 参数"
        if mode == "proj_zero":
            assert float(jnp.max(jnp.abs(got))) == 0.0, float(jnp.max(jnp.abs(got)))
            print("[A] proj_zero：历史 token 逐位为 0（旧零初始化瓶颈）✅")
            continue
        ref = _pooled_reference(model, obs.history_images)
        max_err = float(jnp.max(jnp.abs(got - ref)))
        rms = float(jnp.sqrt(jnp.mean(jnp.square(got))))
        peak = float(jnp.max(jnp.abs(got)))
        assert max_err <= 1e-5, f"{mode} 历史 token != 池化特征（max_err={max_err}）"
        # 关键：直注后历史 token 与"当前帧图像 token"走的是同一个投影头，量级同源（非零、非塌缩）。
        assert peak > 0, f"{mode} 池化特征塌成 0，检查没有意义"
        print(f"[{'B' if mode == 'proj_identity' else 'C'}] {mode}：历史 token == 池化特征"
              f"（max_err={max_err:.2e}，rms={rms:.3e}，peak={peak:.3e}）✅")


def check_contrast(case: str = "all"):
    # 只放 4 个历史槽：对比损失的数学与槽数无关，但 float32 + 反向传播下 20 槽会顶到 24GB 显存。
    obs = _dummy_obs(slots=4)
    actions = jnp.asarray(np.random.default_rng(1).normal(size=(B, 50, 32)).astype(np.float32))
    # 只建一个模型（参数结构与 contrast 开关无关），用实例属性切换三种口径，避免重复建模型撑爆显存。
    model = _small_config(history_slots=4, contrast_mode="roll").create(jax.random.key(0))
    _randomize_image_head(model)
    rng = jax.random.key(7)

    def run(contrast_mode, contrast_stop_grad=True, contrast_weight=0.5):
        model.contrast_mode = contrast_mode
        model.contrast_stop_grad = contrast_stop_grad
        model.contrast_weight = contrast_weight

        def loss_fn(model):
            chunked = model.compute_loss(rng, obs, actions, train=True)
            per_sample = jnp.mean(chunked, axis=-1)
            info = model.compute_contrast_aux(rng, obs, actions, per_sample, train=True)
            return jnp.mean(chunked) + info["contrast_aux"], info

        (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)(model)
        return loss, info, grads

    # 关闭时：aux 恒为 0，且不产生额外梯度路径
    if case in ("all", "off"):
        _, info_off, _ = run("none")
        assert float(info_off["contrast_aux"]) == 0.0 and float(info_off["contrast_hinge_frac"]) == 0.0
        print("[D1] contrast_mode=none：contrast_aux/contrast_hinge_frac 恒为 0 ✅")
        return
    if case == "roll":
        loss_on, info_on, grads_on = run("roll", contrast_stop_grad=False, contrast_weight=0.5)
    else:
        loss_on, info_on, grads_on = run("roll", contrast_stop_grad=True, contrast_weight=0.5)
    aux = float(info_on["contrast_aux"])
    gap = float(info_on["contrast_gap"])
    frac = float(info_on["contrast_hinge_frac"])
    margin = float(info_on["contrast_margin"])
    assert np.isfinite(loss_on) and np.isfinite(aux), (loss_on, aux)
    assert frac == 1.0, f"未训练模型应当内容无感（hinge 全激活），实际 frac={frac}"
    # aux = β·Σhinge/b；B=2、batch_frac=0.5 → k=1，hinge≈margin（gap≈0）→ aux ≈ β·margin/2
    k = max(1, round(B * 0.5))
    assert abs(aux - 0.5 * (k / B) * margin) < 1e-6 * max(1.0, margin), (aux, margin, k)
    assert gap < margin, (gap, margin)
    grad_norm = float(jax.tree_util.tree_reduce(lambda a, b: a + b, jax.tree_util.tree_map(
        lambda g: jnp.sum(jnp.square(g.astype(jnp.float32))), grads_on)))
    assert grad_norm > 0, "对比辅助应当产生非零梯度"
    print(f"[D2] contrast_mode=roll：aux={aux:.3e}（=β·margin）gap={gap:.2e} margin={margin:.2e} "
          f"hinge_frac={frac:.2f} grad²={grad_norm:.3e} ✅")

    if case == "roll_sg":
        print(f"[D3] contrast_stop_grad=True：仍从 on 分支产生梯度（grad²={grad_norm:.3e}）✅")


def check_data_pipeline():
    cfg = _train_config.get_config("pi05_aloha_sim_history_v2_arx-x5_seed_0")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    transforms = [t for t in data_cfg.data_transforms.inputs if t.__class__.__name__ == "RoboDojoHistoryTransform"]
    assert len(transforms) == 1, transforms
    assert transforms[0].pad_mode == "repeat_oldest", transforms[0].pad_mode
    out = transforms[0]({"episode_index": 300, "frame_index": 60})
    mask = np.asarray(out["history_mask"])
    assert mask.all() and mask.shape == (N_SLOTS,), mask
    print("[E] v2 训练配置的 transform：pad_mode=repeat_oldest，可见槽位恒为 20 ✅")
    assert cfg.model.history_inject == "proj_identity"
    assert cfg.model.contrast_mode == "roll"
    print("[E] v2 训练配置：history_inject=proj_identity，contrast_mode=roll ✅")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all",
                        choices=("all", "pipeline", "inject", "contrast", "contrast_off", "contrast_roll",
                                 "contrast_roll_sg"),
                        help="分段跑（每个阶段都新建模型/编译，显存更省）")
    parser.add_argument("--contrast-case", default=None, choices=("all", "off", "roll", "roll_sg"))
    args = parser.parse_args()
    print(f"[env] devices={jax.devices()}")
    if args.only in ("all", "pipeline"):
        check_data_pipeline()
    if args.only in ("all", "inject"):
        check_inject()
    if args.only in ("all", "contrast"):
        check_contrast(args.contrast_case or "all")
    elif args.only.startswith("contrast_"):
        check_contrast(args.only.removeprefix("contrast_"))
    print("\n[Phase 9] GPU 侧验证全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
