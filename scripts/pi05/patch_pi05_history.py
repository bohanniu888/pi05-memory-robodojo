#!/usr/bin/env python3
"""Phase 2（含 Phase 5/6/8 增量）：为 π0.5（openpi）幂等打上"历史/记忆通路"补丁。

改动点（对应规划文档附录 B 第 1~4 条 + 数据通路 carry-through + 附录 E 增量）：
  1. Observation 增加 history_images (b,n,c,hh,ww) / history_mask (b,n) /
     loss_weight (b,)（Phase 8 关键帧加权），默认 None；
  2. Pi0.embed_prefix 增加历史分支：SigLIP → 16×16 → 自适应池化 4×4=16 token → Linear（零初始化），
     拼在当前相机图像之后、tokenized prompt 之前；pad 槽 input_mask=0（与图像 token 同 ar 规则）；
  3. Pi0Config 增加 history_slots=0（默认关闭，向后兼容）/ history_tokens_per_slot=16 /
     history_pool_size=4 / freeze_vision_tower=False（Phase 5 可选冻结视觉塔）；
  4. 权重加载：history_slots>0 时，检查点缺失的 history 参数（projector）就近零初始化，与 create() 一致；
  9. AlohaInputs：可选 cam_history / cam_history_<i> 键 → history_<i>_rgb；透传 episode/frame/task_index 供 sidecar 反查；
  1b. Policy.infer 把输入 dict 里的 history_* 字段带过 transform 管线（以评测侧 _attach_history 为准，覆盖 transform 产物）。
 10. Phase 9（v2 配方）：Pi0Config 增加 history_inject（proj_zero/proj_identity/direct）与
     contrast_*（内容对比辅助损失 max(0, margin − (loss_roll − loss_on))）；Pi0.__init__ 支持恒等
     初始化投影/直注，compute_loss 拆分为 _prepare_loss_inputs + _action_loss 并新增
     compute_contrast_aux；权重加载在 proj_identity 时用恒等核补齐缺失的 history_proj.kernel。

用法：
  python3 scripts/pi05/patch_pi05_history.py            # 打补丁（幂等；首次自动备份 .orig.pi05history）
  python3 scripts/pi05/patch_pi05_history.py --check    # 只报告状态，不写文件
  python3 scripts/pi05/patch_pi05_history.py --revert   # 从备份还原
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / "robodojo/XPolicyLab/policy/Pi_05/openpi/src/openpi"
BACKUP_SUFFIX = ".orig.pi05history"

MODEL_PY = OPENPI / "models/model.py"
PI0_PY = OPENPI / "models/pi0.py"
PI0_CONFIG_PY = OPENPI / "models/pi0_config.py"
POLICY_PY = OPENPI / "policies/policy.py"
ALOHA_POLICY_PY = OPENPI / "policies/aloha_policy.py"

# (文件, 名称, 旧代码锚点, 新代码, 幂等标记)
PATCHES = [
    (
        MODEL_PY,
        "1. Observation 增加 history 字段",
        """    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod""",
        """    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    # [pi05-history] 记忆（历史）字段：历史帧堆叠 (b, n, c, h, w) ∈ [-1,1]，history_mask (b, n)。
    # 注意：轴名用 hh/ww（而非 h/w），避免与 images 的 h/w=224 在 jaxtyping 跨字段检查时冲突（history 为 128×128）。
    history_images: at.Float[ArrayT, "*b n c hh ww"] | None = None
    history_mask: at.Bool[ArrayT, "*b n"] | None = None

    @classmethod""",
        "[pi05-history] 记忆（历史）字段",
    ),
    (
        MODEL_PY,
        "1c. Observation 增加 loss_weight（Phase 8 关键帧加权）",
        """    history_mask: at.Bool[ArrayT, "*b n"] | None = None

    @classmethod""",
        """    history_mask: at.Bool[ArrayT, "*b n"] | None = None

    # [pi05-keyframe] Phase 8：训练样本权重 (b,)（关键帧加权；None = 不加权，等价旧行为）。
    loss_weight: at.Float[ArrayT, "*b"] | None = None

    @classmethod""",
        "[pi05-keyframe] Phase 8",
    ),
    (
        MODEL_PY,
        "1. Observation.from_dict 读取 history + uint8 转换",
        """        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )""",
        """        # [pi05-history] history 帧与图像同规则：uint8 → [-1,1] float32。
        if data.get("history_images") is not None:
            history = data["history_images"]
            if hasattr(history, "dtype") and history.dtype == np.uint8:
                history = history.astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(history, "dtype") and history.dtype == torch.uint8:
                history = history.to(torch.float32) / 255.0 * 2.0 - 1.0
            data["history_images"] = history
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            history_images=data.get("history_images"),
            history_mask=data.get("history_mask"),
        )""",
        "[pi05-history] history 帧与图像同规则",
    ),
    (
        MODEL_PY,
        "1d. Observation.from_dict 读取 loss_weight（Phase 8）",
        """            history_mask=data.get("history_mask"),
        )""",
        """            history_mask=data.get("history_mask"),
            loss_weight=data.get("loss_weight"),
        )""",
        'loss_weight=data.get("loss_weight"),',
    ),
    (
        MODEL_PY,
        "1. preprocess_observation 透传 history（必要时 resize）",
        """    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )""",
        """    # [pi05-history] 透传 history 帧；分辨率不一致时按帧 resize（数据侧允许 CHW/HWC 之前先转 HWC）。
    out_history = observation.history_images
    if out_history is not None and out_history.shape[-2:] != tuple(image_resolution):
        b, n, c = out_history.shape[:3]
        flat = jnp.transpose(out_history.reshape(b * n, c, *out_history.shape[3:]), (0, 2, 3, 1))
        flat = image_tools.resize_with_pad(flat, *image_resolution)
        flat = jnp.transpose(flat, (0, 3, 1, 2))
        out_history = flat.reshape(b, n, c, *image_resolution)
    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        history_images=out_history,
        history_mask=observation.history_mask,
    )""",
        "[pi05-history] 透传 history 帧",
    ),
    (
        MODEL_PY,
        "4. 权重加载：history 参数缺失时零初始化（仅 history_slots>0）",
        """        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)""",
        """        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        # [pi05-history] 检查点缺失的 history 参数（history projector）就地随机初始化。
        # 仅当模型开启历史通路（history_slots>0）时启用；默认 history_slots=0 时走原路径，
        # 行为与改动前完全一致。其余缺失参数不在白名单内，仍由 check_pytree_equality 报错。
        if getattr(self, "history_slots", 0) > 0:
            flat_state = traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
            flat_params = dict(traverse_util.flatten_dict(params, sep="/"))
            for _key, _leaf in flat_state.items():
                if _key in flat_params or "history" not in _key:
                    continue
                # [pi05-history] 补齐缺失的 history 参数（投影层等），保持与 create() 一致的零初始化
                # （Phase 5 关键技巧 1：step 0 历史通道输出为 0）。
                flat_params[_key] = jnp.zeros(_leaf.shape, _leaf.dtype)
            params = traverse_util.unflatten_dict(flat_params, sep="/")
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)""",
        "[pi05-history] 检查点缺失的 history 参数（history projector）就地随机初始化。",
    ),
    (
        PI0_CONFIG_PY,
        "3. Pi0Config 增加 history_* / freeze_vision_tower 字段",
        """    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore""",
        """    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # [pi05-history] 记忆/历史通路（0 = 关闭；官方 checkpoint 缺 history projector 时随机初始化）。
    history_slots: int = 0
    history_tokens_per_slot: int = 16
    history_pool_size: int = 4
    # [pi05-history] 是否额外冻结 SigLIP 视觉塔（保持当前帧特征不动；history 通路不受影响）。
    freeze_vision_tower: bool = False""",
        "[pi05-history] 记忆/历史通路",
    ),
    (
        PI0_CONFIG_PY,
        "3b. Pi0Config.create 支持冻结视觉塔（freeze_vision_tower）",
        """        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)""",
        """        base_filter = nnx.All(*filters) if filters else nnx.Nothing
        if not self.freeze_vision_tower:
            return base_filter
        # [pi05-history] 追加冻结视觉塔（.*img.*）。注意必须用 Any 并集，不能塞进 All。
        vision_filter = nnx_utils.PathRegex(".*img.*")
        return vision_filter if not filters else nnx.Any(base_filter, vision_filter)""",
        "[pi05-history] 追加冻结视觉塔",
    ),
    (
        PI0_PY,
        "2. Pi0.__init__ 创建 history projector",
        """        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True""",
        """        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # [pi05-history] 历史（记忆）通路：SigLIP 输出 → 池化 → 线性投影 → 前缀 token。
        self.history_slots = config.history_slots
        self.history_tokens_per_slot = config.history_tokens_per_slot
        self.history_pool_size = config.history_pool_size
        if config.history_slots > 0:
            # [pi05-history] 输出零初始化（kernel/bias 全 0）：step 0 历史 token 输出为 0，等价于
            # "可学习 scale 初值 0"（Phase 5 关键技巧 1），训练中由梯度逐步"打开"历史通道，
            # 避免随机初始化的历史 token 一开始就污染已收敛的前缀分布。
            self.history_proj = nnx.Linear(
                paligemma_config.width,
                paligemma_config.width,
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True""",
        "[pi05-history] 历史（记忆）通路",
    ),
    (
        PI0_PY,
        "2. Pi0.embed_prefix 历史分支",
        """            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)""",
        """            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # [pi05-history] history 帧：SigLIP → 16×16 → 自适应池化 → Linear；pad 槽 input_mask=0。
        if self.history_slots > 0 and obs.history_images is not None:
            _history = obs.history_images  # (b, n, c, h, w)
            _hb, _hn = _history.shape[0], _history.shape[1]
            _flat = jnp.transpose(_history.reshape(_hb * _hn, *_history.shape[2:]), (0, 2, 3, 1))
            _hist_tokens, _ = self.PaliGemma.img(_flat, train=False)  # (b*n, s, width)
            _side = int(round(_hist_tokens.shape[1] ** 0.5))
            _pool = max(1, _side // self.history_pool_size)
            _hist_tokens = _hist_tokens.reshape(_hb, _hn, _side, _side, _hist_tokens.shape[-1])
            _hist_tokens = _hist_tokens.reshape(
                _hb, _hn, self.history_pool_size, _pool, self.history_pool_size, _pool, _hist_tokens.shape[-1]
            ).mean(axis=(3, 5))
            _hist_tokens = _hist_tokens.reshape(_hb, _hn * self.history_tokens_per_slot, -1)
            tokens.append(self.history_proj(_hist_tokens))
            _hist_mask = obs.history_mask
            if _hist_mask is None:
                _hist_mask = jnp.ones((_hb, _hn), dtype=jnp.bool_)
            input_mask.append(einops.repeat(_hist_mask, "b n -> b (n k)", k=self.history_tokens_per_slot))
            # history tokens attend to each other (same rule as image tokens)
            ar_mask += [False] * (_hn * self.history_tokens_per_slot)

        # add language (aka tokenized inputs)""",
        "[pi05-history] history 帧：SigLIP",
    ),
    (
        POLICY_PY,
        "1b. Policy.infer 带过 history_* 字段",
        """        inputs = self._input_transform(inputs)

        is_batched = False""",
        """        inputs = self._input_transform(inputs)

        # [pi05-history] 以评测侧（服务器 _attach_history）的 history_* 为准：history 训练配置自带的
        # RoboDojoHistoryTransform 在无 episode_index/frame_index 时会写入全 pad + mask=False，
        # 必须用 obs 里的真实历史覆盖，否则模型读不到记忆（Phase 6 冒烟实测 Δ=0.0 的根因）。
        if isinstance(obs, dict):
            for _history_key in ("history_images", "history_mask"):
                if _history_key in obs:
                    inputs[_history_key] = obs[_history_key]

        is_batched = False""",
        "[pi05-history] 以评测侧（服务器 _attach_history）的 history_* 为准",
    ),
    (
        ALOHA_POLICY_PY,
        "9. AlohaInputs 接受 cam_history（B1 拼贴图）",
        """    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")""",
        """    # [pi05-history] cam_history = B1 拼贴图（可选第 4 路图像；缺省不注入以保持基线前缀不变）
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
        "cam_history",
    )""",
        "[pi05-history] cam_history = B1 拼贴图",
    ),
    (
        ALOHA_POLICY_PY,
        "9b. AlohaInputs 注入 history_0_rgb（仅存在 cam_history 时）",
        """        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = _make_image_mask(in_images[source], True)
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = _make_image_mask(base_image, False)

        inputs = {""",
        """        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = _make_image_mask(in_images[source], True)
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = _make_image_mask(base_image, False)

        # [pi05-history] B1 拼贴图：仅在调用方提供 cam_history 时注入第 4 路图像；
        # 缺省不注入，保证无历史时的前缀长度与官方基线完全一致。
        if "cam_history" in in_images:
            images["history_0_rgb"] = in_images["cam_history"]
            image_masks["history_0_rgb"] = _make_image_mask(in_images["cam_history"], True)

        inputs = {""",
        "[pi05-history] B1 拼贴图：仅在调用方提供 cam_history 时注入第 4 路图像",
    ),
    (
        ALOHA_POLICY_PY,
        "9c. AlohaInputs 放行 cam_history_* 键（B2 多帧多图）",
        """        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")""",
        """        in_images = data["images"]
        # [pi05-history] B2 多帧多图：cam_history_<i> 为可选历史图键，按前缀放行。
        unexpected = {
            name
            for name in in_images
            if name not in self.EXPECTED_CAMERAS and not name.startswith("cam_history")
        }
        if unexpected:
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")""",
        "[pi05-history] B2 多帧多图：cam_history_<i> 为可选历史图键",
    ),
    (
        ALOHA_POLICY_PY,
        "9d. AlohaInputs 注入 history_<i>_rgb（仅存在 cam_history_<i> 时）",
        """        if "cam_history" in in_images:
            images["history_0_rgb"] = in_images["cam_history"]
            image_masks["history_0_rgb"] = _make_image_mask(in_images["cam_history"], True)

        inputs = {""",
        """        if "cam_history" in in_images:
            images["history_0_rgb"] = in_images["cam_history"]
            image_masks["history_0_rgb"] = _make_image_mask(in_images["cam_history"], True)

        # [pi05-history] B2 多帧多图：cam_history_<i> → history_<i>_rgb（按索引升序，最旧在前）。
        history_keys = sorted(
            (name for name in in_images if name.startswith("cam_history_")),
            key=lambda name: int(name.rsplit("_", 1)[-1]),
        )
        for name in history_keys:
            index = int(name.rsplit("_", 1)[-1])
            images[f"history_{index}_rgb"] = in_images[name]
            image_masks[f"history_{index}_rgb"] = _make_image_mask(in_images[name], True)

        inputs = {""",
        "[pi05-history] B2 多帧多图：cam_history_<i> → history_<i>_rgb",
    ),
    (
        ALOHA_POLICY_PY,
        "9e. AlohaInputs 透传 episode/frame/task_index（sidecar 反查）",
        """        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs""",
        """        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # [pi05-history] 透传数据集索引，供 RoboDojoHistoryTransform 反查 sidecar（键缺省时行为不变）。
        for _key in ("episode_index", "frame_index", "task_index"):
            if _key in data:
                inputs[_key] = data[_key]

        return inputs""",
        "[pi05-history] 透传数据集索引",
    ),

    # ---------------- Phase 9（v2 配方：断时钟 / 去零初始化瓶颈 / 内容对比损失） ----------------
    (
        PI0_PY,
        """10. [P9] 恒等初始化 kernel initializer""",
        """logger = logging.getLogger("openpi")

""",
        '''logger = logging.getLogger("openpi")


def _identity_kernel_init(rng, shape, dtype=jnp.float32):
    """[pi05-history] Phase 9：恒等初始化（nnx.Linear kernel 形状 = (in, out)）。"""
    del rng  # 恒等初始化与随机数无关（签名需与 flax initializer 一致：key, shape, dtype）
    in_features, out_features = shape[0], shape[1]
    kernel = jnp.zeros((in_features, out_features), dtype=dtype)
    n = min(in_features, out_features)
    return kernel.at[jnp.arange(n), jnp.arange(n)].set(1.0)

''',
        """def _identity_kernel_init(rng, shape, dtype=jnp.float32):""",
    ),
    (
        PI0_PY,
        """10. [P9] Pi0.__init__ history_inject 分支 + contrast 开关""",
        """        self.history_pool_size = config.history_pool_size
        if config.history_slots > 0:
            # [pi05-history] 输出零初始化（kernel/bias 全 0）：step 0 历史 token 输出为 0，等价于
            # "可学习 scale 初值 0"（Phase 5 关键技巧 1），训练中由梯度逐步"打开"历史通道，
            # 避免随机初始化的历史 token 一开始就污染已收敛的前缀分布。
            self.history_proj = nnx.Linear(
                paligemma_config.width,
                paligemma_config.width,
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

""",
        """        self.history_pool_size = config.history_pool_size
        # [pi05-contrast] Phase 9 内容对比辅助损失开关（默认 none = 完全关闭，行为与旧版一致）。
        self.contrast_mode = getattr(config, "contrast_mode", "none")
        self.contrast_margin_rel = float(getattr(config, "contrast_margin_rel", 0.1))
        self.contrast_weight = float(getattr(config, "contrast_weight", 1.0))
        self.contrast_batch_frac = float(getattr(config, "contrast_batch_frac", 0.5))
        self.contrast_stop_grad = bool(getattr(config, "contrast_stop_grad", True))
        # [pi05-history] Phase 9 写侧注入方式（history_inject）：
        #   proj_zero     = 旧行为（可训 Linear，零初始化；step 0 输出 0）；
        #   proj_identity = 可训 Linear，恒等初始化（step 0 输出 = 池化特征，直注但保留可塑性）；
        #   direct        = 不建投影，池化特征直接当前缀 embedding（对齐 DM0.5 的冻结特征直注）。
        self.history_inject = getattr(config, "history_inject", "proj_zero")
        if config.history_slots > 0 and self.history_inject != "direct":
            if self.history_inject == "proj_identity":
                self.history_proj = nnx.Linear(
                    paligemma_config.width,
                    paligemma_config.width,
                    kernel_init=_identity_kernel_init,
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
            else:
                # 输出零初始化（kernel/bias 全 0）：step 0 历史 token 输出为 0，等价于
                # "可学习 scale 初值 0"（Phase 5 关键技巧 1），训练中由梯度逐步"打开"历史通道，
                # 避免随机初始化的历史 token 一开始就污染已收敛的前缀分布。
                self.history_proj = nnx.Linear(
                    paligemma_config.width,
                    paligemma_config.width,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )

""",
        """self.contrast_mode = getattr(config, "contrast_mode", "none")""",
    ),
    (
        PI0_PY,
        """10. [P9] embed_prefix 直注分支""",
        """            _hist_tokens = _hist_tokens.reshape(_hb, _hn * self.history_tokens_per_slot, -1)
            tokens.append(self.history_proj(_hist_tokens))
            _hist_mask = obs.history_mask
""",
        """            _hist_tokens = _hist_tokens.reshape(_hb, _hn * self.history_tokens_per_slot, -1)
            if self.history_inject == "direct":
                # [pi05-history] Phase 9：冻结特征直注（无投影、无瓶颈）。
                tokens.append(_hist_tokens)
            else:
                tokens.append(self.history_proj(_hist_tokens))
            _hist_mask = obs.history_mask
""",
        """if self.history_inject == "direct":""",
    ),
    (
        PI0_PY,
        """10. [P9] compute_loss 拆分为 _prepare_loss_inputs""",
        """    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
""",
        '''    ) -> at.Float[at.Array, "*b ah"]:
        observation, x_t, u_t, time = self._prepare_loss_inputs(rng, observation, actions, train=train)
        return self._action_loss(observation, x_t, u_t, time)

    def _prepare_loss_inputs(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool,
    ) -> tuple[
        _model.Observation,
        at.Float[at.Array, "*b ah ad"],
        at.Float[at.Array, "*b ah ad"],
        at.Float[at.Array, "*b"],
    ]:
        """[pi05-contrast] Phase 9：预处理 + 流匹配加噪（同一 rng 下可与 compute_loss 逐位复现）。

        compute_loss 与 compute_contrast_aux 都以各自收到的 rng 为种子，因此只要传入同一个 rng，
        两条路径的预处理/噪声/时间完全一致——对比损失才是在"同一批噪声"上比较。
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
''',
        """def _prepare_loss_inputs(""",
    ),
    (
        PI0_PY,
        """10. [P9] _action_loss 单次前向本体""",
        """        u_t = noise - actions

""",
        '''        u_t = noise - actions
        return observation, x_t, u_t, time

    @at.typecheck
    def _action_loss(
        self,
        observation: _model.Observation,
        x_t: at.Float[at.Array, "*b ah ad"],
        u_t: at.Float[at.Array, "*b ah ad"],
        time: at.Float[at.Array, "*b"],
    ) -> at.Float[at.Array, "*b ah"]:
        """单次 prefix+suffix 前向的 flow-matching loss（与改动前 compute_loss 主体逐行一致）。"""

''',
        """def _action_loss(""",
    ),
    (
        PI0_PY,
        """10. [P9] contrast_enabled + compute_contrast_aux""",
        """        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

""",
        '''        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @property
    def contrast_enabled(self) -> bool:
        """[pi05-contrast] Phase 9：内容对比辅助损失是否开启（默认 False = 与旧版逐位一致）。"""
        return self.contrast_mode != "none" and self.history_slots > 0

    def compute_contrast_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        per_sample_loss: at.Float[at.Array, "*b"],
        *,
        train: bool = False,
    ) -> dict[str, at.Float[at.Array, ""]]:
        """[pi05-contrast] Phase 9：内容对比辅助损失 max(0, margin − (loss_roll − loss_on))。

        * ``loss_on``   = 主 loss（由调用方传入 per-sample 均值，形状 (b,)，与训练主 loss 同一批噪声/时间）；
        * ``loss_roll`` = 同 batch 内 roll=1 的错位历史（等价离线门禁的 croll 臂）重算的 flow-matching loss；
        * margin = ``contrast_margin_rel`` × 当日 batch 的 on-loss 均值（stop_gradient，避免尺度漂移）；
        * ``contrast_stop_grad=True`` 时 roll 分支不反传：辅助项的作用是"把用错内容也不变差的样本权重加大"，
          即自动版的关键帧加权；False 时同时把"错内容必须更差"也写进目标（对抗式）。
        * 只对前 k=round(b×contrast_batch_frac) 个样本计算（省算力），其余样本 aux=0。

        返回的 dict 全部是标量（用于训练日志）；关闭时返回全 0，梯度贡献为零。
        """
        zero = jnp.zeros((), jnp.float32)
        info: dict[str, at.Float[at.Array, ""]] = {
            "contrast_aux": zero,
            "contrast_hinge_frac": zero,
            "contrast_gap": zero,
            "contrast_margin": zero,
        }
        if not self.contrast_enabled or observation.history_images is None:
            return info

        obs, x_t, u_t, time = self._prepare_loss_inputs(rng, observation, actions, train=train)
        batch_size = x_t.shape[0]
        k = max(1, min(batch_size, int(round(batch_size * self.contrast_batch_frac))))
        rolled = obs.replace(history_images=jnp.roll(obs.history_images, 1, axis=0))
        if k < batch_size:
            rolled = jax.tree.map(lambda leaf: leaf[:k], rolled)
        loss_roll = self._action_loss(rolled, x_t[:k], u_t[:k], time[:k])

        # 逐样本比较（batch 级均值会被"本来就无关内容"的样本稀释掉，失去选择性）。
        on_per = per_sample_loss[:k]
        roll_per = jnp.mean(loss_roll, axis=-1)
        if self.contrast_stop_grad:
            roll_per = jax.lax.stop_gradient(roll_per)
        margin = jax.lax.stop_gradient(jnp.mean(on_per)) * self.contrast_margin_rel
        hinge = jnp.maximum(0.0, margin - (roll_per - on_per))
        info = {
            # 归一化到"全 batch 平均"：逐样本梯度为 on 侧 (1−β)/b、roll 侧 β/b，
            # 因此 β<1 时 on 侧永远是下降方向（不会出现"为了拉大 gap 而抬高 on-loss"的反向）。
            "contrast_aux": self.contrast_weight * jnp.sum(hinge) / batch_size,
            "contrast_hinge_frac": jnp.mean((hinge > 0).astype(jnp.float32)),
            "contrast_gap": jax.lax.stop_gradient(jnp.mean(roll_per - on_per)),
            "contrast_margin": margin,
        }
        return info

''',
        """def contrast_enabled(self) -> bool:""",
    ),
    (
        PI0_CONFIG_PY,
        """10. [P9] Pi0Config history_inject/contrast_* 字段""",
        """    freeze_vision_tower: bool = False

""",
        """    freeze_vision_tower: bool = False
    # [pi05-history] Phase 9 写侧注入方式（去掉"零初始化瓶颈"）：
    #   proj_zero     = 旧行为：可训 Linear + 零初始化（step 0 输出 0，训练中逐步打开）；
    #   proj_identity = 可训 Linear + 恒等初始化（step 0 输出 = 池化特征，等价直注但保留可塑性）；
    #   direct        = 不建投影，池化后的冻结 SigLIP 特征直接当前缀 embedding（对齐 DM0.5 直注）。
    history_inject: str = "proj_zero"

    # [pi05-contrast] Phase 9 内容对比辅助损失：max(0, margin − (loss_roll − loss_on))。
    # roll = 同 batch 内 roll=1 的错位历史（与离线门禁 croll 臂同口径）。
    #   contrast_mode: none（默认关闭）| roll
    #   contrast_margin_rel: margin = 相对量 × 当日 batch 的 on-loss 均值
    #   contrast_weight: 辅助项权重（加到主 loss 上）
    #   contrast_batch_frac: 只用前 k=round(b*frac) 个样本算 roll 分支（省算力；其余样本 aux=0）
    #   contrast_stop_grad: True = roll 分支不反传（只把"用错内容时更好"的样本权重加大）；
    #                       False = 同时把"用错内容 loss 抬高"也写进目标（对抗式）。
    contrast_mode: str = "none"
    contrast_margin_rel: float = 0.1
    contrast_weight: float = 0.5
    contrast_batch_frac: float = 0.5
    contrast_stop_grad: bool = True

""",
        """# [pi05-history] Phase 9 写侧注入方式（去掉"零初始化瓶颈"）：""",
    ),
    (
        PI0_CONFIG_PY,
        """10. [P9] Pi0Config.__post_init__ 开关校验""",
        """            ]

""",
        """            ]
        # [pi05-history] Phase 9 开关校验（默认值与原行为完全一致）。
        assert self.history_inject in ("proj_zero", "proj_identity", "direct"), self.history_inject
        assert self.contrast_mode in ("none", "roll"), self.contrast_mode
        if self.contrast_mode != "none":
            assert self.history_slots > 0, "contrast 需要 history_slots > 0"
            assert 0.0 < self.contrast_batch_frac <= 1.0, self.contrast_batch_frac
            assert self.contrast_margin_rel >= 0.0, self.contrast_margin_rel
            # β<1：保证 on 分支始终是下降方向（β≥1 会把"当前内容"的 loss 反向抬高）。
            assert 0.0 <= self.contrast_weight < 1.0, self.contrast_weight

""",
        """assert self.history_inject in ("proj_zero", "proj_identity", "direct"), self.history_inject""",
    ),
    (
        MODEL_PY,
        """10. [P9] 权重加载 proj_identity 恒等补齐""",
        """                # （Phase 5 关键技巧 1：step 0 历史通道输出为 0）。
                flat_params[_key] = jnp.zeros(_leaf.shape, _leaf.dtype)
            params = traverse_util.unflatten_dict(flat_params, sep="/")
""",
        """                # （Phase 5 关键技巧 1：step 0 历史通道输出为 0）。
                # [pi05-history] Phase 9：history_inject=proj_identity 时投影核用恒等初始化补齐，
                # 与 Pi0.__init__ 的 _identity_kernel_init 一致（否则会把直注路径退化成零初始化瓶颈）。
                if (
                    getattr(self, "history_inject", "proj_zero") == "proj_identity"
                    and _key.endswith("history_proj/kernel")
                    and len(_leaf.shape) == 2
                ):
                    _eye = jnp.zeros(_leaf.shape, _leaf.dtype)
                    _n = min(_leaf.shape)
                    flat_params[_key] = _eye.at[jnp.arange(_n), jnp.arange(_n)].set(1.0)
                else:
                    flat_params[_key] = jnp.zeros(_leaf.shape, _leaf.dtype)
            params = traverse_util.unflatten_dict(flat_params, sep="/")
""",
        """# 与 Pi0.__init__ 的 _identity_kernel_init 一致（否则会把直注路径退化成零初始化瓶颈）。""",
    ),
]


def apply_all(check_only: bool) -> int:
    files = {}
    status = []
    for path, name, old, new, marker in PATCHES:
        text = files.get(path)
        if text is None:
            text = path.read_text(encoding="utf-8")
        if marker in text:
            status.append((name, "已应用"))
            files[path] = text
            continue
        if old not in text:
            status.append((name, "锚点未命中 ✗"))
            files[path] = text
            continue
        status.append((name, "打了补丁"))
        files[path] = text.replace(old, new, 1)

    ok = all(state != "锚点未命中 ✗" for _, state in status)
    for path, text in files.items():
        original = path.read_text(encoding="utf-8")
        if text == original:
            continue
        if check_only:
            continue
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(text, encoding="utf-8")

    for name, state in status:
        print(f"  [{state:>8}] {name}")
    return 0 if ok else 1


def revert() -> int:
    restored = 0
    for path in dict.fromkeys(patch[0] for patch in PATCHES):
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if backup.exists():
            shutil.copy2(backup, path)
            restored += 1
            print(f"  [还原] {path}")
    print(f"共还原 {restored} 个文件" if restored else "没有可还原的备份")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="只报告状态，不写文件")
    group.add_argument("--revert", action="store_true", help="从 .orig.pi05history 备份还原")
    args = parser.parse_args()

    if not MODEL_PY.exists():
        print(f"找不到 openpi 源码: {MODEL_PY}", file=sys.stderr)
        return 1
    print(f"目标: {OPENPI}")
    if args.revert:
        return revert()
    return apply_all(check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
