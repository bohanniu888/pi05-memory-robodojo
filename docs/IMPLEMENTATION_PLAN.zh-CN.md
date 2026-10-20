# π0.5 × RoboDojo Memory 维度提升实施规划
> **目标**：把 DM0.5 的记忆机制（历史帧 → soft token → VLM 前缀）移植到 π0.5，使 π0.5 在 RoboDojo
> **Memory 维度**达到 DM0.5 量级（目标 ≥ 40，对齐 DM0.5 榜单 47.74 / 本集群复现 44.03）。
> **撰写时间**：2026-09-30 ｜ **归档更新**：2026-10-20。
> **状态**：Phase 9 v2 训练、离线门禁与 Memory 快速评测均已完成；**当前没有训练作业在跑**。
> 评测 **6.20**（6 任务，swap_blocks 记 0）/ **7.44**（5 任务）vs 基线 4.86，仍远低于 40 分目标。
> 下一步：① 继续排查 `swap_blocks` 的仿真侧 `PhysX CUDA 700`（根因尚未闭环）；
> ② 攻 `press_by_number`（自运动计数）与 `cover_blocks` 整集成功率；或 ③ 转 DM0.5 式显式 memory 读写架构。
> **前置文档**：`DM0.5-RoboDojo-复现规划.md`（DM0.5 复现全流程；本文件复用其环境方案、评测链路与结论）
> **一句话结论**：机制可移植，记忆能力必须**训练进权重**；免训练嫁接只当筛查手段，不作为终态。

---

## 0. TL;DR（结论速览）

1. **DM0.5 的记忆不是外挂，是训练出来的条件通路**：head 相机 20 槽 @1 Hz，每帧压成 4×4=16 个 soft token
   （共 320 token）注入 VLM 前缀；checkpoint 强制要求 `is_history=True` 的 experiment 才能加载
   （`policy/OpenDM/model.py:300`）。π0.5 权重里没有这条通路，推理期直接注入属分布外输入。
2. **π0.5 的插入点现成**：`pi0.py` 的 `embed_prefix` 本来就循环处理任意数量的 `obs.images`，
   多一路"历史帧"在结构上天然成立；且 openpi 自带 LoRA（`gemma_2b_lora`）与 `freeze_filter`，
   "冻结主干 + 只训记忆通路"是一条低成本先手路线。
3. **分数结构对我们非常有利**：DM0.5 的 47.74 里，`cover_blocks`(100) + `press_by_number`(95.33) +
   `match_and_pick_from_conveyor`(70.67) 三项就贡献 **44.33**；剩下三项（swap_T 0 / imitate 1.80 /
   swap_blocks 18.67）只贡献 3.41。**把"看线索 → 记住 → 稍后执行"这三类做到 DM0.5 水平，就等于拿到 ~45 分**。
4. **路线分三档**：
   - **B 免训练嫁接**（拼贴图 / 多图 / 文字化记忆）：3~5 天出结论，追平 47.74 概率 10~15%，但**必须先做**——
     它验证的前向管线与后续训练完全同源。
   - **A 训练式移植**（DM0.5 同款历史前缀 + LoRA/全量微调）：达到 ≥40 概率 40~55%，是唯一有希望追平的路线。
   - **C 折中**（冻结主干，只训 memory projector + LoRA）：风险收益比最好，作为第一次训练的默认方案。
5. **两个硬前置已解决（2026-10-03）**：① π0.5 RoboDojo 权重已从 ModelScope 下载并校验；
   ② prepared LeRobot v3.0 数据已在本地：`robodojo/data/RoboDojo_lerobot_v30_video`，
   3500 集 / 1,856,102 帧 / 25 FPS / 14 维 joint。原始来源是 HF/ModelScope 的
   `RoboDojo-Benchmark/RoboDojo`；当前发布与复现仍须遵守上游许可，不随仓库分发数据与权重。
6. 验收按 DM0.5 文档同款口径：Memory 6 任务 × 3 seeds × 50 trials，**≥40 算接近，≥42.74（DM0.5 −5）算达标**，
   且其余维度不得掉超过 3 分。

---

## 1. 目标与验收标准

分五层递进，每层可判定：

| 层级 | 目标 | 验收标准 | 预估资源 |
| --- | --- | --- | --- |
| **L0 环境跑通** | π0.5 离线 wiring check | `EVAL_ENV_TYPE=debug` 下能产出 `_result.json`（不启 Isaac Sim） | 1 GPU，分钟级 |
| **L1 基线复现** | 现成 π0.5 权重的 Memory 基线 | 6 任务 × 3 seeds × 6 集（108 episodes）产出基线分，记录方差 | 2 GPU，小时级 |
| **L2 记忆通路活性** | 历史注入管线离线验证 | ① 有/无历史动作输出最大差 > 1e-3；② pad 槽不参与（置零 vs 随机无差异）；③ 显存/延迟可接受 | 1 GPU，小时级 |
| **L3 免训练 A/B** | 拼贴/多图/文字记忆筛查 | `cover_blocks` + `press_by_number` 各 20 trials，合计提升 ≥ 15 分 → go 训练；否则下调预期 | 2 GPU，天级 |
| **L4 Memory 维度达标** | LoRA/全量训练后复测 | 6 任务 × 3 seeds × **native 50 trials**：得分 **≥ 40**（接近）/ **≥ 42.74**（达标） | 2~4 GPU，天级 |
| **L5 全维回归（可选）** | 五维全量 | 其余维度相对基线 Δ ≤ 3 分；产出与榜单逐项对照表 | 4~8 GPU，数天级 |

**对齐判定原则**（沿用 DM0.5 文档）：

- 单任务分数允许 ±10 分绝对偏差（score），SR 允许 ±8 个百分点；
- 维度均分要求 ±5 分以内（对照 DM0.5 的 47.74 / 47.44，即 **≥ 42.74**）；
- 若单任务偏差超阈值，先怀疑环境/版本漂移，再怀疑权重或数据。

**相对口径（本次的核心指标）**：π0.5 榜单/本集群基线分 → 提升到 ≥40。因为 DM0.5 自己也做不好
swap_T(0) 与 imitate(1.8)，**我们不要求解决全部记忆任务，只要求拿下三个"线索型"任务**。

> **范围边界（重要，别误读目标）**：本文档实现的是「**记忆机制的可移植改造 + 在 RoboDojo 上的能力特化**」。
> - **通用的是机制**：历史通路、采集契约、注入代码与 benchmark 无关，换到 RoboTwin / LIBERO / 真机等，
>   只需替换数据管线与输入配置（相机名、动作空间、任务指令格式）；
> - **特化的是能力**：权重里学到的记忆行为来自 RoboDojo 的 42 个 sim 任务，离开 RoboDojo 不会自动迁移，
>   需要在目标 benchmark 数据上**重做第二段微调**；
> - **想要"任意 benchmark 通用的原生记忆"**：需多 benchmark/多机器人混训、甚至把历史条件注入预训练或
>   大规模后训练（DM0.5 走的正是这条路线），并参考仓库中 RISE / SimpleMemVLA / Mem_0 等记忆型策略的
>   表示设计——那是另一个量级的项目，不在本文档范围内。

---

## 2. 事实基础（已核实信息）

### 2.1 DM0.5 的记忆机制（代码取证）

| 环节 | 实现 | 位置 |
| --- | --- | --- |
| 采集契约 | `images_1`（头部相机）、20 槽、1 Hz、每 25 个完成动作采一帧 | `policy/OpenDM/model.py:220`（`_configure_history`）、`:379`（`_append_history_observation`） |
| 可见性规则 | 最新一帧要等下一个 interval 才可见；不足 20 槽时前面补 None | `policy/OpenDM/model.py:403`（`_history_images_for`） |
| 压缩 | 每帧 4×4=16 个 soft token（`HISTORY_TOKENS_PER_IMAGE=16`） | `policy/OpenDM/opendm/opendm/constants/robot.py:5` |
| 注入 | 历史特征替换 prompt 中 `<unused1>` 占位 token，写入 VLM 前缀/KV cache | `dm05_arch.py:61`（`_project_history_image_features`）、`:996`（prefix 填充） |
| 状态隔离 | 按 `(evaluation_id, env_idx)` 维护 deque，`reset` 时清空 | `policy/OpenDM/model.py:509` |

**离线活性验证（DM0.5，已通过）**：A1 vs A2 = 0.000000（固定 seed 可复现）；空 vs 有历史 = 0.023940；
图案 A vs B = 0.001568。→ 历史内容真实进入模型并被消费（详见 DM0.5 文档 Phase 2B-2）。

**关键结论**：记忆是**预训练/SFT 阶段学进权重**的能力，模块本身不能"即插即用"。

### 2.2 π0.5 现状与插入点（代码取证）

| 项目 | 现状 | 位置 |
| --- | --- | --- |
| 观测输入 | 单帧三相机：`cam_high` / `cam_left_wrist` / `cam_right_wrist` | `policy/Pi_05/model.py`（`encode_obs` / `stack_obs`） |
| 前缀构造 | 循环遍历 `obs.images`，每路图独立编码后拼接，**天然支持任意数量图像** | `openpi/src/openpi/models/pi0.py:98`（`embed_prefix`） |
| 文本预算 | `max_token_len = 200`（pi05） | `openpi/src/openpi/models/pi0_config.py:39` |
| 动作头 | flow matching + adaRMS（pi05），对前缀分布敏感 | `openpi/src/openpi/models/pi0.py:130+` |
| LoRA 支持 | 原生支持：`gemma_2b_lora`（rank 16 / alpha 16）+ `freeze_filter` | `openpi/src/openpi/models/gemma.py:88`、`training/config.py:562` |
| 训练 recipe | `pi05_base_aloha_full_sim_arx-x5_seed_0`，LeRobot repo id `RoboDojo_sim_arx-x5_v30`，batch 256，60k steps，FSDP 2 | `openpi/src/openpi/training/config.py:636` |
| reset 语义 | `reset()` 只清 `observation_window`，无任何历史状态 | `policy/Pi_05/model.py` |

**token 预算核算**：3 相机 × 256 token（So400m/14, 224×224）= 768；加 20 槽历史 × 16 = 320，
前缀长度从 ~968（含 prompt）涨到 ~1288，**+33%**。这是可接受的；且每 25 步只有 1 帧新图，
历史特征可增量编码 + 缓存，无需每步重跑 20 帧 SigLIP。

### 2.3 分数拆解：靶心在哪里

| 任务 | DM0.5 榜单 | 本集群复现 | 性质 | 是否需要"记忆" |
| --- | --- | --- | --- | --- |
| `cover_blocks` | 100.00 | 100.00 | 看初始布局 → 盖住后仍要记得哪个在最左 | **是（视觉线索记忆）** |
| `press_by_number` | 95.33 | 83.33 | 先读数字卡 → 按正确次数按钮并确认 | **是（线索 + 自运动计数）** |
| `match_and_pick_from_conveyor` | 70.67 | 77.78 | 看线索 → 传送带上抓匹配物 | **是（视觉线索记忆）** |
| `swap_T` | 0.00 | 0.00 | 两个 T 块交换位置与朝向 | 否（纯操作难度） |
| `imitate_sorting_sequence` | 1.80 | 3.06 | 模仿辅助臂演示的分拣序列 | 部分是（时序模仿），DM0.5 也不会 |
| `swap_blocks` | 18.67 | 0.00 | 交换两块积木 | 否（纯操作难度） |
| **均值** | **47.74** | **44.03** | | 前三项贡献 **44.33/47.74** |

> 本集群复现值来自 DM0.5 文档 Phase 3Q（3 seeds × 6 集）。前三项目前占 44.33 分——**这就是靶心**。

### 2.4 榜单量级参考

- DM0.5 Memory = 47.74（榜单快照 2026-09-21），公开快照里该维度第二名约 13.37，
  即 **DM0.5 是断档第一（≈3.5 倍）**。
- 推论：π0.5 大概率处在个位数到低十几分区间；**做到 40+ 就是 3 倍级别的提升**，
  但"追平 47.74"需要在前三任务上接近满分。

### 2.5 数据与权重来源（2026-10-01 核实）

官方已把训练数据与策略权重都发布在同一个 HF/ModelScope 仓库 `RoboDojo-Benchmark/RoboDojo`
（HF API 实测可达，仓库 license=apache-2.0），用 RoboDojo 仓库自带下载器获取：

| 资源 | 命令（在 `robodojo/` 根目录） | 体积 | 落点 |
| --- | --- | --- | --- |
| π0.5 权重 | `bash scripts/RoboDojo/download_ckpt.sh <source> pi05` | — | `XPolicyLab/policy/Pi_05/checkpoints/` |
| 小样本 demo | `bash scripts/RoboDojo/download_data.sh <source> demo` | 1.5 GB | `robodojo/data/demo/` |
| **训练数据（首选）** | `bash scripts/RoboDojo/download_data.sh <source> lerobot_v3.0` | 120 GB | `robodojo/data/RoboDojo_lerobot_v30_video/`（joint-only，14 维，与官方 joint ckpt 对齐） |
| 训练数据（**不要用于当前 joint ckpt**） | `bash scripts/RoboDojo/download_data.sh <source> lerobot_v3.0_ee` | 120 GB | 16 维 EE 空间，仅当明确改训 EE 模型时使用 |
| 原始全量 | `bash scripts/RoboDojo/download_data.sh <source> hdf5` | 523 GB | 仅自定义转换/子集时需要 |

`<source>` 取 `huggingface` 或 `modelscope`。**已核实存在**的远端目录：
`ckpt/RoboDojo/Pi_05`、`data/RoboDojo_ee_lerobot_v30_video`、`data/RoboDojo_lerobot_v30_video`、
`data/RoboDojo_lerobot_v21_video`、`data/RoboDojo`（hdf5）、`data/demo`、`data/RoboDojo_real`、`data/RoboDojo_depth`。

注意两点：① 两个下载器的网络机制不同——`download_data.sh` 走 `huggingface_hub`（认 `HF_ENDPOINT`），
`download_ckpt.sh` 走 `git clone --sparse`（要用镜像得覆盖 `HF_REPO_URL`）；② π0.5 的 joint 控制是 14 维
（每臂 6 关节 + 1 夹爪），`lerobot_v3.0` 标注 "joint-only, no ee values"，而 `lerobot_v3.0_ee` 为 16 维
EE 空间（`l_x..l_g, r_x..r_g`）。已发布 ckpt `RoboDojo-sim-arx_x5-joint-*` 的 norm stats 为 14 维
（2026-10-03 实测），因此**训练数据必须选 `lerobot_v3.0`（joint）**；下载后用 `meta/info.json` 校验维度 = 14。

③ 该仓库仍在更新（2026-09-28 有提交），**下载时用 `HF_REVISION` 锁定快照并把 revision 记进环境指纹**，
避免"阶段 0 与阶段 1 的数据版本不一致"；阶段 1（记忆微调）与阶段 0（RoboDojo 特化）原则上用同一份
sim demos，差异只在样本构造（多了 history sidecar）与任务配比（可只取子集 + 通用混采）。

**本项目实际数据规模（归档核对）**：

| 口径 | 集数 | 帧数 | 说明 |
| --- | ---: | ---: | --- |
| 官方 prepared LeRobot v3.0 全量 | 3,500 | 1,856,102 | 35 任务，25 FPS，14 维 joint |
| v2 训练清单 | **1,200** | **644,703** | mem6 600 集 / 328,375 帧 + 通用 600 集 / 316,328 帧；**不是 2047 集** |
| Memory 6 任务子集 | 600 | 328,375 | 每任务 100 集 |
| 通用 cotrain 子集 | 600 | 316,328 | 其余任务分层抽样 |
| history sidecar 元数据 | 600 | 13,407 个历史采样 | 1 Hz、128×128；`.npy` 约 631 MB，不入 Git |

数据与 sidecar 的 `.npy` 均不进入代码仓库；仓库只提交 `index.json`、`meta.json`、
`cotrain_mix_episodes.json` 和关键帧权重表等小型元数据。

---

## 3. 技术方案总览

### 3.1 三条路线

| 路线 | 做法 | 成本 | 达到 ≥40 的概率 | 定位 |
| --- | --- | --- | --- | --- |
| **B 免训练嫁接** | 不动权重：历史帧拼成时间拼贴图作为第 4 路图像；或用 captioner 把历史转成文字 prompt | 3~5 天 | 10~15% | **先做**，验证管线 + 摸底 |
| **C 冻结主干 + 小模块** | 冻结 VLM/动作头主干，只训 history projector + LoRA | 1~2 GPU 天训练 + 1 周工程 | 30~45% | **第一次训练默认方案** |
| **A 训练式移植** | 全量/大范围微调，历史条件与主干一起训 | 5~10 GPU 天 + 2~3 周工程 | 40~55% | C 不达标时启动 |

**三条路线不是并列全做，而是"漏斗式串联"**（B 筛查 → C 默认训练 → A 兜底），与 Phase 的对应关系：

| 路线 | 对应 Phase | 触发条件 | 出口 |
| --- | --- | --- | --- |
| **B 免训练** | Phase 2（共用注入管线）+ Phase 3 | 无条件先做（3~5 天） | 提升 ≥15 分：变体作为训练辅助候选；5~15：保留为辅助；<5：关闭 B，直接进训练 |
| **C 冻结主干 + LoRA** | Phase 4（数据管线）+ Phase 5C + Phase 6 | Phase 3 出结论后启动；**Phase 4/5 不依赖 B 的结果** | ≥35 分（快速门槛）→ Phase 6 正式复测；25~35 → 调超参/数据配比重试一轮；<25 → 启动 A |
| **A 全量微调** | Phase 5A + Phase 6（复用） | 仅当 C 不达标 | 直接进 Phase 6 正式评测 |

- 公共前置：Phase 0（权重/数据/环境）与 Phase 1（基线）三条路线共用；Phase 2 产出的
  `patch_pi05_history.py` 注入管线是 B/C/A 共用的同一套代码，不存在重复投入。
- 唯一的串行关系：**B → C →（必要时）A**；只有 Phase 5 阶段存在 C/A 二选一。

> **路线 B 为什么"不训练也可能涨分"**：记忆 =「时间上的保持」+「需要时的取用」。π0.5 缺的是**保持**，
> 而**取用**能力（从当前输入里找到线索并据此行动）在预训练中已经具备。B 把"保持"这一步交给外部代码
> （deque / 拼贴 / captioner），再把结果重新塞回当前输入——模型本身仍然无记忆，只是拿到了一张"便签"。
> 能否奏效取决于注入通道是否在训练分布内：**文字 > 单张拼贴图 > 多帧序列**；且模型没有被告知"这些 token 重要"，
> 所以预期上限只有 10~15%（详见 §3.1 与 Phase 3）。

### 3.2 记忆契约（三条路线共用，必须逐条对齐）

> **"（记忆）注入管线"指什么**：从仿真环境的一帧观测开始，到该帧变成 π0.5 前缀里的 320 个历史 token 为止的
> 完整处理链路。B/C/A 三条路线共用这一条链路的同一份代码，区别只在于**权重是否被训练去读这些 token**。

```text
RoboDojo 仿真 ──obs(含 cam_high)──▶ XPolicyLab 客户端 ──ws──▶ π0.5 policy server
                                                                   │
                            ① 采集：每 25 个已执行动作取一帧 ────────┤
                            ② 缓存：per-env deque，保留 20 槽（pad 补齐 + mask）
                                                                   │
                            ③ 编码：SigLIP（So400m/14）→ 16×16 patch
                            ④ 池化：自适应池化 4×4 = 16 token/帧
                            ⑤ 拼装：20 槽 × 16 = 320 token，插在「当前帧图像 token 之后、
                                     prompt token 之前」，pad 槽 attention mask=0
                                                                   │
                                    ⑥ 前向：VLM 前缀 + flow-matching 动作头 → action chunk
                                                                   │
                                    ⑦ 回写：新帧入队、步数 +1；trial reset 时清空 deque
```

这条链路分两段落位：**训练侧**在 `openpi` 数据管线（sidecar → transform → `history_images`），
**推理侧**在 `policy/Pi_05/model.py`（deque）+ `openpi/.../pi0.py`（编码与拼装）。二者必须逐条对齐契约，
否则训练与推理的行为不一致。Phase 2 的产物 `patch_pi05_history.py` 就是这条管线的实现。

1. **相机**：`cam_high`（对应 DM0.5 的 `images_1`）。
2. **槽位**：20 槽，时间顺序 **oldest → newest**，不足补 pad（mask=0）。
3. **采样**：每 25 个**已执行**底层动作采一帧（不是按推理次数）；最新一帧要等下一个 interval 才可见
   （与 `_history_images_for` 的过滤条件一致）。
4. **压缩**：每帧 SigLIP → 16×16 patch → 自适应池化到 4×4=16 token；训练/推理必须同源同参数。
5. **注入位置**：当前帧图像 token 之后、prompt token 之前（训练与推理共用同一约定，写死进配置）。
6. **隔离**：per-env deque，key 为 `(evaluation_id, env_idx)`；`reset` 清空；严禁跨 trial 串味。
7. **训练侧**：历史帧必须来自**同一 episode 且早于当前时刻**；pad 槽在 loss 中不参与（attention mask 屏蔽）。

### 3.3 训练数据构造（实际采用方案）

- 直接下载官方 **prepared LeRobot v3.0**（joint，14 维），**不再跑** `process_data.sh`；
  本地落点为 `robodojo/data/RoboDojo_lerobot_v30_video`。
- 跑一次 `build_history_sidecar.py`：按 episode 读 `cam_high` 视频流，以 1 Hz 抽帧存
  `robodojo/data/history_sidecar/<task>/ep_*.npy`（uint8，N×3×128×128）；
  训练 transform 按 `frame_index` 取严格早于当前时刻的最近 20 帧 + mask。
- 用 `build_cotrain_mix.py` 生成 1200 集训练清单：Memory 6 任务 600 集 +
  其余任务分层抽样 600 集。默认 `HF_LEROBOT_HOME` 指向
  `robodojo/data/hf_lerobot_home`，sidecar 根由 `ROBODOJO_HISTORY_SIDECAR_DIR` 指定。
- **备用**：在 LeRobot 里加一路独立 fps 的视频流（改动大、依赖 LeRobot 版本行为，不首选）。

### 3.4 两段式微调：本项目的改动落在哪一段（重要，别混淆）

官方流程是 `pi05_base ──(RoboDojo sim 数据微调)──▶ 官方 RoboDojo checkpoint`（≈榜单 "Pi0.5"，**无记忆**）。
本项目不重做这一段，而是在其之上做**第二段微调**：

```text
阶段 0（官方已完成；不含任何记忆模块）
  pi05_base ──[数据：sim demos；输入：单帧观测]──▶ 官方 RoboDojo checkpoint

阶段 1（本项目"记忆微调"）
  官方 RoboDojo checkpoint + 本项目的历史通路改动
    ──[数据：同一份 sim demos + history sidecar；输入：当前帧 + 20 槽历史]──▶ 有记忆的 π0.5
```

**改动模块的使用阶段**：

| 改动 | 阶段 0 | 阶段 1（训练） | 评测（推理） |
| --- | --- | --- | --- |
| 模型：`pi0.py` 历史分支 / `Pi0Config` 字段 | ❌ | ✅ | ✅（训练/推理共用同一份模型代码） |
| 数据：`build_history_sidecar.py` + `RoboDojoHistoryTransform` | ❌ | ✅ | ❌（推理用在线队列） |
| 适配器：`Pi_05/model.py` deque + `deploy.yml` history 字段 | ❌ | ❌ | ✅ |

→ 模型侧改动训练与推理共用；数据侧改动只在训练；适配器侧改动只在推理。三者必须按 §3.2 契约对齐，
否则训练学到的输入布局与推理给的不一致，效果归零。

---

## 4. 目录结构规划

```text
<PROJECT_ROOT>/                              # 本文件所在根目录；可用 PI05_ROOT 覆盖
├── Pi0.5-Memory-RoboDojo-实施规划.md       # 本文件
├── DM0.5-RoboDojo-复现规划.md              # 前置文档
├── robodojo/                                # 固定到上游 commit 726e9aa
│   ├── XPolicyLab/                          # 子模块，固定到上游 commit bb9a0b5
│   │   └── policy/Pi_05/
│   │       ├── checkpoints/                 # 权重（不提交）
│   │       ├── openpi/                      # JAX 训练/推理栈（打补丁：history 通路）
│   │       └── deploy.yml                   # 增加 history_* 字段
│   ├── data/RoboDojo_lerobot_v30_video/     # 数据（不提交）
│   ├── data/history_sidecar/                # sidecar：npy 不提交，JSON 元数据提交
│   └── eval_result/RoboDojo/                # 评测结果落点（不提交）
├── scripts/pi05/                            # 本项目脚本（见 §6）
├── logs/pi05/                               # SLURM/验证日志（不提交）
├── models/                                  # 可选 captioner 等模型（不提交）
└── images/                                  # 仿真镜像导出（不提交）
```

**原则**：所有新增脚本放 `scripts/pi05/`，不改动 DM0.5 复现已跑通的 `scripts/phase3_*`、
`run_smoke_stack_bowls.sh` 与 `OpenDM` 适配器；π0.5 侧的源码改动只落在
`XPolicyLab/policy/Pi_05/` 与 `openpi/` 内，并由 `patch_pi05_history.py` 与完整 diff 补丁共同复现（见 §6）。

---

## 5. 分阶段实施计划

### Phase 0：前置就位（预计 1~3 天，下载耗时决定）

**为什么单列**：权重与数据是本项目的两个硬门槛，必须先确认可得性，否则后面全是空转。

| 编号 | 任务 | 命令/落点 | 判据 |
| --- | --- | --- | --- |
| 0A | 下载 π0.5 RoboDojo 权重（来源已核实） | `bash scripts/RoboDojo/download_ckpt.sh <source> pi05`（`<source>`=huggingface/modelscope）→ `XPolicyLab/policy/Pi_05/checkpoints/` | 目录含 `params/` + `assets/`（含 norm stats）；`checkpoint_num=59999` 能对上 |
| 0B | 下载训练数据（来源已核实） | 先 `bash scripts/RoboDojo/download_data.sh <source> demo`（1.5 GB）打通链路，再下 `... lerobot_v3.0`（120 GB，joint） | state/action 维度 = 14；LeRobot key 与 π0.5 repack 期望一致；**不要用 16 维 `lerobot_v3.0_ee`** |
| 0C | openpi 环境安装与冒烟 | `cd policy/Pi_05 && bash install.sh`；`export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3` | 能在 GPU 上 `create_trained_policy` 并出一次 `infer()` |
| 0D | 基线标定 | 单次推理时延 / 显存 / 每个 chunk 实际执行的底层动作数 | 记录下来；**动作数决定 history interval 的对齐**（DM0.5 是 25） |
| 0E | 离线 wiring check | `EVAL_ENV_TYPE=debug bash policy/Pi_05/eval.sh RoboDojo cover_blocks <ckpt> arx_x5 joint 0 0 0 uv <env>` | 产出 `_result.json` |

**风险出口**：0A 或 0B 拿不到 → 只保留 Phase 3（免训练），其余终止，并在文档里明确记录。

#### Phase 0 执行记录（2026-10-03）

- **运行环境**：交互作业 `<JOBID>` @ `gpu01`（2× RTX 4090D，16 CPU，48h）；登录节点无 GPU，
  所有下载/安装/标定都在 `srun --jobid=<JOBID> --overlap` 内执行；日志统一在 `logs/pi05/`。
- **0A 权重**：✅ 完成（20:24）。走 `modelscope` 源（`download_ckpt.sh pi05 modelscope`）；
  3 个 seed 全部校验通过（85/85 文件、69.6/69.6 GB），官方脚本已建立符号链接
  `policy/Pi_05/checkpoints → .cache/robodojo_ckpt_modelscope_repo/ckpt/RoboDojo/Pi_05`。
  seed-0 含 `train_state/`（44.7GB，训练续跑用），seed-1/2 为纯 `params/`（各 12.4GB，推理用）；
  已实测 `assets/arx_x5_sim/norm_stats.json`：state/action 均为 14 维。
  注：seed-2 因主线较慢曾用 `scripts/pi05/ckpt_direct_fetch.sh` 并行直取提前解锁 0D/0E；
  主线完成后，原预取目录被官方脚本归档为 `checkpoints.partial.20261003_202416`。
- **0B 数据**：`demo` ✅ 完成（`data/demo` 符号链接 + `.download_complete`，2.8GB，20 条 episode）。
  `lerobot_v3.0_ee` ⚠️ **勘误**：其 `meta/info.json` 显示 state/action 为 **16 维 EE 空间**
  （`l_x..l_g, r_x..r_g`），与已发布 joint ckpt（14 维 norm stats）不匹配；
  **训练数据应改用 `lerobot_v3.0`（joint-only，state/action=14 维）**。已用 meta 文件离线核验
  `data/RoboDojo_lerobot_v30_video`：state/action=14；相机键 `observation.images.{cam_high,
  cam_left_wrist, cam_right_wrist}` 与 π0.5 repack 完全一致；fps=25 / 3500 episodes / 1.86M frames。
  120GB 本体待 0A 完成后启动（避免抢带宽）。
- **0C 环境**：✅ 完成。`uv 0.12.22` + `CPython 3.11.16`（`~/.local/bin/python3.11`）；
  `uv sync --group lerobot` 解析安装 231 包；冒烟通过：`import openpi` OK、JAX 0.5.3 可见
  `[CudaDevice(id=0), CudaDevice(id=1)]`、`get_config("pi05_base_aloha_full_sim_arx-x5_seed_0")` 加载成功。
  **关键标定预读：`cfg.model.action_horizon=50`**（deploy.py 会把整段 chunk 执行完，即每 chunk 执行
  50 个底层动作；与 DM0.5 的 25 不同，history interval 对齐时以此为准，0D 探针做运行时复核）。
- **0D 基线标定**：✅ 完成（seed-2 路径 `RoboDojo-sim-arx_x5-joint-2/59999`，
  25/25 文件、sha256 与 LFS OID 一致）。实测：加载 40.2s（JAX 首次编译 warmup 18.3s），
  稳态推理 **0.10s/次**（10Hz），动作块 **[50,14]（float64）**，显存峰值 **6.9GB**
  （`XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`）。**每 chunk 实际执行底层动作数 = `action_horizon` = 50**
  （deploy.py 会把整段 chunk 执行完；DM0.5 为 25，Phase 2 历史对齐以 50 为准）。结果：
  `logs/pi05/probe_2026-10-03_20-10-08.json`。
- **0E 离线 wiring check**：✅ 完成（`EVAL_ENV_TYPE=debug`，server GPU0 + client `opendm` 环境）。
  验收证据：server 加载成功、client `[CONNECTED]`、**10/10 episode 各 20 步动作往返无异常、exit=0**，
  日志 `logs/pi05/wiring_debug_seed2_cover_blocks_2026-10-03_20-10-34.log`。
  注意：debug 模式不落 `_result.json`（mock 客户端只打印 episode 过程）。
- **0B（续）**：`lerobot_v3.0`（joint，120GB）已于 20:13 启动下载
  （`logs/pi05/download_data_lerobot_v3.0_modelscope_2026-10-03_20-13-49.log`）。
- **新增脚本**：`scripts/pi05/{download_ckpt.sh, fetch_demos.sh, bootstrap_env.sh,
  probe_pi05_runtime.py, phase0_status.sh, README.md}`。

### Phase 1：Memory 基线复现（预计 1 天）

- 用 `scripts/pi05/phase_mem_pi05.sh --eval-num 6 --seeds 0,1,2` 跑 6 任务 × 3 seeds × 6 集（108 episodes）。
- 产出基线表（逐任务 + 均值 + SR），与 DM0.5 的 47.74 / 44.03 并列。
- **判据**：6/6 任务有结果；记录方差。若基线已 ≥40，项目终止（无需提升）。
- **注意**：这一步同时验证评测链路（server + 容器 client + JAX 推理）在 π0.5 上可用。

#### Phase 1 执行记录（2026-10-04）

- **完成状态**：✅ 18/18 run 全部 rc=0；108/108 episodes（6 任务 × 3 seeds × 6 集）；
  13:03 → 17:07（约 4 小时；单 worker：π0.5 server 走 GPU0、Isaac Sim client 走 GPU1）。
- **命令**：`bash scripts/pi05/phase_mem_pi05.sh --eval-num 6 --seeds 0,1,2`。
  日志根目录 `logs/pi05/phase1_2026-10-04_13-03-01/`（`results.tsv` + 逐 run `.out`）。
- **权重/seed 口径**：固定 `ckpt=RoboDojo-sim-arx_x5-joint-2`（训练 seed-2 权重）；
  `--seed 0/1/2` 为官方"three eval seeds"口径（决定 rollout 随机性；官方 quick-evaluation
  文档：*three eval seeds…Repeat for --seed 1 and --seed 2*）。实测 3 个 eval seed 下布局集合
  相同（layout 0-5）但每集成败模式不同 → rollout 确实在变，方差来自策略轨迹而非布局。
- **结果**（score = 逐集 score 均值 × 100；SR = 成功集占比）：

  | task | s0 | s1 | s2 | 均值 | SR |
  | --- | --- | --- | --- | --- | --- |
  | `cover_blocks` | 22.50 | 22.50 | 22.50 | **22.50** | 16.67% |
  | `match_and_pick_from_conveyor` | 16.67 | 0.00 | 0.00 | **5.56** | 5.56% |
  | `imitate_sorting_sequence` | 0.83 | 0.83 | 1.67 | **1.11** | 0% |
  | `press_by_number` | 0.00 | 0.00 | 0.00 | **0.00** | 0% |
  | `swap_T` | 0.00 | 0.00 | 0.00 | **0.00** | 0% |
  | `swap_blocks` | 0.00 | 0.00 | 0.00 | **0.00** | 0% |
  | **总分** | 6.67 | 3.89 | 4.03 | **4.86** | **3.70%** |

  → 远低于 DM0.5（本集群复现 44.03 / 榜单 47.74），差距 ≈ 39~43 分；
  π0.5 原生"无记忆"确认成立，**判据未触发**（4.86 < 40），项目按计划推进 Phase 2。
- **链路验证**：host π0.5 server + 容器 client + JAX 推理全链路可用，108 集 0 报错；
  结果与 3 路相机视频落 `robodojo/eval_result/RoboDojo/<task>/Pi_05/arx_x5/...`。
- **耗时画像**：平均 ~11 min/run；`imitate_sorting_sequence` 是长杆（episode 上限 1600 步，
  19~45 min/run），其余任务 6~12 min/run。6 集为批内并行（env0~env5），单 run ≈ 启动 4 min + 最长单集。
- **踩坑与修复**：
  1. **server 进程泄漏**：`run_pi05_eval.sh` 原先只 kill `bash robodojo.sh server` 包装进程，
     其下 python 被 reparent 到 PID 1、每 run 泄漏 7.6GB 显存（2~3 个 run 即挤爆 GPU0）。
     已新增 `scripts/pi05/janitor_kill_leaked_servers.sh` 守护（60s 扫一次 PPid=1 的本仓库
     π0.5 server），并修复 `run_pi05_eval.sh` cleanup（按端口精确兜底 `pkill`）。
  2. **results.tsv 缺表头**：编排脚本首跑未写表头，汇总解析会把首行数据当表头；
     已修复脚本（自动写表头），当次 `results.tsv` 已补表头。
- **本阶段新增脚本**：`scripts/pi05/{run_pi05_eval.sh, phase_mem_pi05.sh, phase1_status.sh,
  janitor_kill_leaked_servers.sh}`。

### Phase 2：记忆通路离线活性验证（预计 2~3 天，不训练）

**为什么必须先做**：与 DM0.5 的 Phase 2B-2 同理——若历史没接通，后面所有分数都会被误读。

- 产出 `scripts/pi05/verify_pi05_history_liveness.py` + `.sh`（仿 `scripts/verify_memory_liveness.py`）。
- 同时落第一版补丁 `scripts/pi05/patch_pi05_history.py`（幂等），改动点见附录 B：
  1. `Observation` 增加 `history_images` / `history_mask`；
  2. `embed_prefix` 增加历史分支（SigLIP → 4×4 池化 → 16 token/帧 → 拼在 current image 之后、prompt 之前）；
  3. `Pi0Config` 增加 `history_slots` / `history_tokens_per_slot`（默认 0 = 关闭，向后兼容）；
  4. 权重加载允许新增参数缺失（随机初始化 projector，参考 weight_loaders 的 `missing_regex` 模式）。

测试组（固定 `torch/jax` 随机种子，每组跑 2 次）：

| 组 | 历史内容 | 预期 |
| --- | --- | --- |
| A1 / A2 | 空历史（0/20） | 两次完全一致（差 = 0） |
| B | 20 槽全黑 | 与 A 有显著差异 |
| C | 20 槽全白 | 与 B 有差异（证明读的是内容，不只是"有没有"） |
| D | pad 槽置零 vs 置随机 | 无差异（证明 mask 正确屏蔽了 pad） |

- **判据**：A1=A2；|B−A| > 1e-3；|B−C| 可观测；D 差 < 1e-4；单次前向延迟增幅 < 30%、显存增幅 < 2 GiB。
- **产物**：日志 + 三张对照表 + 补丁文件 + 契约确认（附录 B 更新为"已实现"）。

#### Phase 2 执行记录（2026-10-04）

- **完成状态**：✅ 探针一次通过，全判据达标（含 1 次 OOM 修复）；实际耗时 ~1 小时（远低于 2~3 天预估）。
- **命令**：`srun --jobid=<JOBID> --overlap bash scripts/pi05/verify_pi05_history_liveness.sh`
  （GPU0 单卡；日志 `logs/pi05/verify_history_liveness_2026-10-04_17-33-27.log`）。
- **实测对照表**（固定 `jax.random.key(1234)` + 固定 flow-matching noise，输出逐位可复现）：

  | 组 | 历史内容 | 实测差值 | 判据 | 结论 |
  | --- | --- | --- | --- | --- |
  | A1 / A2 | 空历史（20 槽 pad、mask=0） | 0.000000000 | 差 = 0 | ✅ 完全可复现 |
  | A vs B | 无历史 vs 20 槽全黑 | 0.737505 | > 1e-3 | ✅ 历史通路生效 |
  | B vs C | 全黑 vs 全白 | 0.086627 | 可观测 > 底线 | ✅ 读的是内容而非"有无" |
  | D1 vs D2 | 15 pad（零 vs 随机）+ 5 真槽 | 0.000000000 | < 1e-4 | ✅ mask 正确屏蔽 pad |
  | 延迟 | slots=0 → slots=20 | 85 ms → 87 ms（+1.9%） | 增幅 < 30% | ✅ |
  | 显存 | 两模型同进程 | history 构建峰值 6.28 GiB；同进程 bytes_in_use 12.51 GiB | 增幅 < 2 GiB | ✅（projector 仅 ~8 MB） |

- **结论**：π0.5 记忆通路**已接通**——历史帧经 SigLIP→池化→投影成 16 token/槽 真实进入 VLM 前缀，
  策略输出对历史内容敏感且 pad 槽被正确屏蔽。Phase 3 免训练 A/B 可在此基础上直接开跑。
- **踩坑与修复**：
  1. **缺失参数初始化 OOM**：第一版 `load()` 用 `self.create()` 为新参数做整模随机初始化，
     峰值 9.6 GiB 直接 OOM；改为**就地初始化缺失叶子**（bias→0，其余 `lecun_normal`，复刻
     `nnx.Linear` 默认）后仅新增 ~8 MB。
  2. 探针脚本显式 `XLA_PYTHON_CLIENT_PREALLOCATE=true` + `MEM_FRACTION=.85`，防碎片化 OOM。
- **本阶段修改/新增**：
  - 新增：`scripts/pi05/verify_pi05_history_liveness.{py,sh}`、`scripts/pi05/patch_pi05_history.py`（幂等 8 锚点）；
  - 修改 openpi 源码 4 文件（备份 `*.orig.pi05history`，可用 `--revert` 一键还原）：
    `models/model.py`（`Observation` 字段 / `from_dict` / `preprocess_observation` / `load` 缺失参数就地初始化）、
    `models/pi0.py`（`history_proj` + `embed_prefix` 历史分支）、`models/pi0_config.py`（`history_*` 字段）、
    `policies/policy.py`（`infer` 透传 `history_images/history_mask`）；
  - 附录 B 第 1~4 条 ✅ 已实现；第 5~6 条（推理侧 deque / deploy.yml）留待 Phase 3 评测接入前完成。

### Phase 3：免训练 A/B 筛查（预计 3~5 天，go/no-go 关口）

三个变体，均**不动权重**：

| 变体 | 做法 | 预期 |
| --- | --- | --- |
| B1 拼贴图 | 把最近 4/8/20 帧按网格拼成一张 224×224 图，作为第 4 路图像 key | 最省事，OOD 风险最高 |
| B2 多帧多图 | 取最近 3~4 帧作为独立 image key（避免 token 爆炸） | π0.5 结构原生支持 |
| B3 文字化记忆 | 用 captioner 把历史转成一句"线索描述"注入 prompt | 利用语言通道，最"在分布内"，但依赖 captioner |

- 评测面：`cover_blocks` + `press_by_number` 各 20 trials（seed 0），对比 Phase 1 基线。
- **判据**：
  - 合计提升 **≥15 分** → 进入 Phase 4/5（并把该变体作为训练期的辅助特征候选）；
  - 提升 **5~15 分** → 保留为辅助，主路线仍走训练；
  - 提升 **<5 分** → 免训练路线关闭，直接评估训练预算。

#### Phase 3 执行记录（2026-10-04：公共前置 + B1 + B2）

- **公共前置（附录 B 第 5~6 条）✅**：
  - `policy/Pi_05/model.py`：历史 deque（对齐 DM0.5 口径：每 `history_action_interval` 个动作步抽 1 帧、
    消费时排除"过新"帧、左侧补 None、按 env_idx 隔离、reset 清空）+ B1 拼贴图（2×2 网格、224×224、缺帧补黑）；
  - `deploy.yml`：`history_enabled/history_mode=collage/history_image_key=cam_high/history_slots=20/
    history_fps=1.0/runtime_fps=25.0/history_action_interval=25/history_collage_frames=4/history_collage_size=224`；
  - openpi 补丁新增第 9/9b 条（`aloha_policy.py`）：接受第 4 路图像 `cam_history` → `history_0_rgb`，
    仅在调用方提供时注入（无历史时前缀长度与基线完全一致）；
  - CPU 单测 `scripts/pi05/test_pi05_history_pipeline.py` 6/6 通过（含"无历史=基线不变"回归）。
- **冒烟**（1 集 `cover_blocks`，18:21–18:29）：链路通，server 日志出现
  `[Pi05] history active: env=0 usable=2 collage=4@224`，`_result.json` 正常落盘（510s）。
- **B1 正式评测**（20 trials × 2 任务，seed 0，18:31–19:16；两次 run 各 10 env 均确认 history active）：

  | task | B1（20 trials） | Phase 1 基线（seed0，6 集） | Δ |
  | --- | --- | --- | --- |
  | `cover_blocks` | **9.25**（SR 0%） | 22.50（SR 16.7%） | **−13.25** |
  | `press_by_number` | **0.00**（SR 0%） | 0.00（SR 0%） | 0.00 |
  | 合计均值 | **4.63** | 11.25 | **−6.62** |

- **结论**：B1（拼贴图当第 4 路图像）**未带来提升，且显著拉低 `cover_blocks`**——与"OOD 风险最高"的
  预判一致（前缀 +256 token、位置分布偏移、第 4 路图干扰）。按判据（<5 分）**B1 不作为候选**；
  免训练筛查转入 B2（多帧多图）/ B3（文字化）。
- 口径说明：基线为 6 集、B1 为 20 trials（同 seed），差值含采样方差；但 B1 方向为负且 SR=0，
  判"无提升"可靠。日志：`logs/pi05/phase3_b1_2026-10-04_18-31/results.tsv`。

- **B2 多帧多图（19:26–20:25）**：
  - 实现：`_attach_history` 增加 `multi` 模式（最近 N 帧各作独立 image key `cam_history_0..N-1`
    → 模型键 `history_0_rgb..`，最旧在前）；openpi 补丁新增第 9c/9d 条（放行 `cam_history_*`、
    按索引映射）；CPU 单测扩至 7/7 通过；`deploy.yml: history_mode=multi, history_multi_frames=3`；
  - 冒烟（1 集 `cover_blocks`，19:26–19:35）：链路通，`[Pi05] history active: env=0 usable=2 multi=3`；
  - 正式评测（20 trials × 2 任务，seed 0；两次 run 各 10 env 均确认历史激活）：

  | task | B2（20 trials） | Phase 1 基线（seed0，6 集） | Δ |
  | --- | --- | --- | --- |
  | `cover_blocks` | **22.25**（SR 15%，成功 3/20） | 22.50（SR 16.7%） | −0.25 |
  | `press_by_number` | **0.00**（SR 0%） | 0.00（SR 0%） | 0.00 |
  | 合计均值 | **11.12** | 11.25 | **−0.13** |

  - 结论：B2 与基线**统计上持平**（未提升、也未像 B1 那样明显伤害；B1 合计均值 4.63）。按判据
    <5 分，B2 亦不作候选。累计证据：免训练"图像通道"（B1 拼贴 / B2 多帧）均无法让 π0.5 在
    memory 维度获益——与"记忆必须训进权重"的先验一致；下一步 B3（文字化，走语言通道）或直接
    评估训练预算。日志：`logs/pi05/phase3_b2_2026-10-04_19-41/results.tsv`。

- **B3 文字化（20:25–21:44）**：
  - 实现：`_attach_history` 增加 `text` 模式（取最旧+最新可用历史帧 → Qwen2.5-VL-3B caption
    → prompt 追加 `"... Memory: earlier X; recently Y."`）；`_get_captioner()` 惰性加载（text 模式
    `__init__` 预载）；`_caption_for()` 按 (env, sequence) 缓存、reset 清空、失败不阻断；
    `_generate_caption()` 含截断保护（未句号收尾回切完整句）。CPU 单测 8/8 通过。
  - captioner 实测（独立脚本）：加载 2.4~10.8s、显存 7.0 GiB（峰值 7.6）、0.39~0.84 s/帧；
    抽帧截图人工核对与描述一致（无 BGR/RGB 问题）。
  - 冒烟（1 集 `cover_blocks`，20:44–20:53）：`captioner loaded` + `history active: text_captions=2`
    均出现，JAX+captioner 同卡共存 15.4/24 GB，无失败。
  - 正式评测（20 trials × 2 任务，seed 0；两次 run 各 10 env 均确认历史激活、0 captioner failed、
    0 truncation）：

  | task | B3（20 trials） | Phase 1 基线（seed0，6 集） | Δ |
  | --- | --- | --- | --- |
  | `cover_blocks` | **0.00**（SR 0%，20/20 精确 0 分） | 22.50（SR 16.7%） | −22.50 |
  | `press_by_number` | **0.00**（SR 0%） | 0.00（SR 0%） | 0.00 |
  | 合计均值 | **0.00** | 11.25 | **−11.25** |

  - 归因：录像抽帧对比显示 B3 机器人仍在动作（非冻结），但呈现"到达但抓取失败/推倒物体"模式；
    B1/B2 均 19/20 集有部分分，B3 为 20/20 精确 0 分——文本注入真实伤害策略，与"记忆必须训进
    权重"的先验一致。
  - 结论：按判据 <5 分，**B3 亦未达标 → 免训练路线（B1/B2/B3）整体关闭**，下一步评估训练预算
    （Phase 4 数据管线 + Phase 5C LoRA）。日志：`logs/pi05/phase3_b3_2026-10-04_20-55/results.tsv`。

### Phase 4：历史增强数据管线（预计 3~5 天）

> **Phase 3 → Phase 4/5 交接注意项（2026-10-04 补充，来自 Phase 3 代码复盘）**：
> 1. **评测配置复位**：`deploy.yml` 目前遗留 `history_enabled: true / history_mode: text`。Phase 4 不受
>    影响；但 Phase 5/6 经 XPolicyLab 评测前必须复位并与训练配置严格一致（先 `history_enabled: false`
>    验基线，或切到与训练相同通路），否则会加载 captioner（~7 GiB、0.4~0.8 s/帧）并注入 B3 已证明
>    有害的 text prompt。
> 2. **两条 history 通路必须选型统一**：① `images["cam_history*"]` → `history_<i>_rgb` 裸图像 token
>    （B1/B2 实测走的就是这条，不需要 `history_slots`，效果差）；② 顶层 `history_images (b,n,c,h,w)`
>    → 池化 + Linear → 前缀 token（DM0.5 式，需 train/eval 双开 `history_slots>0`）。B 路线端到端
>    只测过①，**② 仅有 Phase 2 离线单测、从未真实评测**——Phase 4 侧车/transform 与 Phase 5 训练、
>    评测注入模式必须成套选②（本 Phase 4 的 `RoboDojoHistoryTransform` 即按②设计），且现有 XPolicyLab
>    评测侧还没有产出顶层 `history_images` 的模式，需要补。
> 3. **训练权重加载缺参**：`CheckpointWeightLoader._merge_params` 的 `missing_regex=".*lora.*"` 只回填
>    lora 缺参；若 `history_slots>0` 从 RoboDojo ckpt 初始化，需把 regex 扩到 `history_proj`（或自定义
>    loader），并且 LoRA 冻结名单必须把 `history_proj` 列为可训练。
> 4. **缺参静默随机初始化的风险**：Phase 3 给 `BaseModelConfig.load`（serving 路径 `policy_config.py:57`）
>    加了"缺失参数随机初始化"。若 eval 的 `history_slots` 等与训练不一致，会静默得到随机 projector 而
>    不报错。建议 server 启动日志打印 history 配置并断言与训练一致。

1. `scripts/pi05/fetch_demos.sh`：封装官方下载器，拉 prepared `lerobot_v3.0`（joint，120 GB，key 已对齐 π0.5）后按任务过滤；
   memory 6 任务子集与通用数据混采（cotrain），**不需要**下 523 GB hdf5（仅自定义分辨率/字段时用）；

2. 校验已下载的 prepared LeRobot v3.0：`meta/info.json` 的 `fps=25`、`total_episodes=3500`、
   `state/action=14`，相机键为 `observation.images.{cam_high,cam_left_wrist,cam_right_wrist}`。
   **不再运行 `process_data.sh`**；它只在需要自定义分辨率/字段时使用。

3. `scripts/pi05/build_history_sidecar.py`：
   - 输入：LeRobot 数据集 + `cam_high` 视频流；
   - 输出：`data/history_sidecar/<task>/<episode>.npy`（1 Hz 抽帧，uint8，N×3×H×W，H=W=128）；
   - 关键：抽帧时刻必须可被训练 transform 反查到"当前帧之前的最近 20 个 1 Hz 时刻"。

4. openpi 数据管线补丁：新增 `RoboDojoHistoryTransform`（读 sidecar → 组 20 槽 → pad + mask →
   输出 `history_images`/`history_mask`），注册进新 TrainConfig（附录 C）。

5. 单测：`scripts/pi05/check_history_dataset.py`
   - 随机抽 100 个样本：`history_images.shape == (20,3,128,128)`、mask 正确、当前帧不在历史里、
     episode 边界处不跨集；
   - 可视化抽查 3 条 episode 的历史序列（拼成 gif/mp4 人工看一遍）。

- **判据**：100/100 样本校验通过；`--batch-size 4` 的 dataloader 能出 batch；可视化无异常。

- **执行记录（2026-10-04，实际 ~2.5 h，远低于 3~5 天预估）**：
  - 数据侦察：`robodojo/data/RoboDojo_lerobot_v30_video`（v3.0 / 25 fps / 3500 集 / 1,856,102 帧）；
    memory 6 任务 task_index：cover_blocks=3、imitate_sorting_sequence=11、match_and_pick_from_conveyor=16、
    press_by_number=25、swap_T=32、swap_blocks=33（各 100 集，共 600 集 / 328,375 帧；cam_high 254 个 mp4，mem6 涉 51 个）。
  - 边界语义修正：相邻集 `from_ts` 无缝衔接，而 from/to 时长与 `length` 存在 ±1 帧误差 → 抽帧归属改为
    "下一集 from_ts 邻接窗口"，试点缺帧（26 集缺 k=0）清零。
  - sidecar 全量构建（`build_history_sidecar.py --tasks mem6 --workers 8`）：600 集、missing k=0、
    13,407 个采样、631 MB，产出 `robodojo/data/history_sidecar/`（`index.json` / `meta.json` + 6 任务 × 100 `ep_*.npy`；
    128×128 resize_with_pad 黑边、k ↔ local frame=25k）；实测 1.5 分钟。日志
    `logs/pi05/p4_sidecar_full_2026-10-04_22-38-38.log`。
  - 训练侧接入：新增 `openpi/src/openpi/policies/robodojo_history.py`（`RoboDojoHistoryTransform`，口径与
    评测侧 `_history_entries_for` 一致：k_max=frame//25−1、取最近 ≤20 槽、最旧在左、左侧 pad+mask=False；
    无 sidecar → 全 pad，供通用数据 cotrain）；`AlohaInputs` 透传 `episode_index/frame_index/task_index`；
    `config.py` 新增 `history_sidecar_root/history_interval` 字段 + TrainConfig
    `pi05_aloha_sim_history_arx-x5_seed_0`（history_slots=20 / tokens_per_slot=16 / pool_size=4 / batch=128 /
    20k steps，见附录 C）；补齐 assets（`openpi/assets/RoboDojo_assets/arx_x5_sim/norm_stats.json`）。
  - 验收（`check_history_dataset.py`，74.8 s，**PASS**）：A) 随机 100 样本 100/100 通过（shape/dtype/mask 位置/
    当前帧不在历史/不跨集）；B) dataloader batch=4 × 12 batches，8 个 batch 带历史，首个 history 形状
    `(4,20,3,128,128)` float32 ∈[-1,1]；C) 3 条 episode 可视化人工核对通过（1 Hz 步进、动作连续、无跨集）。
    日志 `logs/pi05/p4_check2_2026-10-04_22-45-43.log`，图 `logs/pi05/phase4_vis/check_*.png`。
  - 踩坑修复：jaxtyping 跨字段轴名冲突（history 与 images 同用 `h/w`，128 vs 224 导致类型检查失败）→
    `Observation.history_images` 轴名改为 `*b n c hh ww`（`model.py`）。

### Phase 5：训练（预计 1~2 周）

> **执行记录（2026-10-05，公共前置 + 训练已启动）**：
> - **代码前置（全部默认关闭，不影响其它配置）**：
>   1. `CheckpointWeightLoader` 新增 `missing_regex` 字段（默认 `.*lora.*` 不变）；history 训练配置用
>      `.*lora.*|.*history.*`，解决 train.py 侧 `check_pytree_equality` 因缺 history 键而报错（交接项 3）。
>   2. 关键技巧 1 落地：`Pi0` 的 `history_proj` kernel/bias 零初始化；`BaseModelConfig.load()` 缺参补齐
>      同样改为零填充。step 0 历史通道输出为 0，等价"可学习 scale 初值 0"。
>   3. `Pi0Config` 新增 `freeze_vision_tower`（默认 False）；`get_freeze_filter` 支持并集冻结 `.*img.*`。
>      实测冻结口径：SigLIP 23 个叶子全冻；VLM 9 冻/10 可训（LoRA）；action expert 11 全可训；
>      `history_proj` 可训。
>   4. 训练配置 `pi05_aloha_sim_history_arx-x5_seed_0` 补齐：`paligemma_variant=gemma_2b_lora`、
>      `freeze_filter=Pi0Config(...).get_freeze_filter()`（TrainConfig 默认 Nothing=不冻结，必须显式传）、
>      `ema_decay=None`（LoRA 惯例，省显存）、`missing_regex` 见上。
>   5. 关键技巧 2：新增 `scripts/pi05/build_cotrain_mix.py` → `cotrain_mix_episodes.json`
>      （mem6 600 集 + 其余 29 任务分层抽样 600 集 = 1200 集；帧级 mem6:通用 = 50.9% : 49.1%）。
>      `DataConfig.episodes` + `data_loader` 透传 lerobot `episodes=` 过滤实现。
> - **验收脚本** `scripts/pi05/verify_phase5_setup.py`：A 冻结口径 / B history_proj 零初始化 /
>   C 权重加载结构完整（history 键为 init 占位）/ D cotrain 混采 + history transform 出样本 —— **ALL PASS**。
> - **冒烟与资源标定**：2 卡 bs128、4 卡 bs64(fsdp=2) 均 OOM（单卡激活峰值过高）；
>   最终采用 **4 卡纯数据并行（fsdp=1）+ batch 64**：约 **3.8 s/step**、单卡显存 22.2/24.5 GB、利用率 100%。
> - **正式训练已启动（2026-10-05 10:54）**：exp `RoboDojo-sim-arx_x5-history-lora-seed0`，20k steps
>   （≈1.28M 样本 ≈2 epoch），每 1k 保存 / 2k 保留；日志 `logs/pi05/p5_train_seed0_2026-10-05_10-54-38.log`；
>   ETA ~21 h。训练入口 `scripts/pi05/train_pi05_history.sh`（`--gpus 0,1,2,3 --fsdp-devices 1 --batch-size 64`）。

**5C（默认先跑）冻结主干 + LoRA**

- `scripts/pi05/train_pi05_history.sh`（基于 `train.sh` 改造）：
  - `train_config_name=pi05_aloha_sim_history_arx-x5_seed_0`；
  - `paligemma_variant=gemma_2b_lora`（rank 16 / alpha 16），history projector 全参训练；
  - 其余主干 `freeze_filter` 冻结；
  - 初始化自 Phase 0A 的 RoboDojo 微调 checkpoint（不是 pi05_base）。
  - **加载缺参处理（交接项 3）**：`CheckpointWeightLoader` 的 `missing_regex` 扩到 `history_proj`（或改用
    自定义 loader），否则新历史投影层缺参无法合并；确认 `history_proj` 全参可训、不在 freeze_filter 内。
- **冻结口径（对齐 `pi0_config.get_freeze_filter` 的实际行为，别想当然）**：
  - `gemma_2b_lora` + `gemma_300m`：冻结 VLM (Gemma) 主干，其 LoRA 参数可训；**action expert 仍全参可训**；
  - 若想更保守，用 `gemma_300m_lora`：action expert 主干也冻结，只训其 LoRA；
  - 新增的 history projector 是全参可训；建议在补丁里**额外冻结 SigLIP 视觉塔**，保持当前帧特征不动。
- **关键技巧 1：projector 输出层零初始化**（加可学习 scale，初值 0）。这样 step 0 的行为与基线完全等价，
  训练逐步学会"打开"历史通道，避免随机 token 一开始就污染已收敛的前缀分布。
- **关键技巧 2：数据配比**。只用 memory 6 任务微调会灾难性遗忘（其他维度会掉）→ batch 内
  memory 数据 : 通用 cotrain 数据 ≈ 1:1 混采；至少保证每个 batch 里有非 memory 样本。
- **关键技巧 3（可选）**：训练时对历史帧的时间间隔做 ±20% 抖动，提高推理节拍与训练不一致时的鲁棒性
  （Phase 0D 若发现"每 chunk 实际执行步数"与 25 不一致，此招可兜底）。
- 超参起点：batch 128（显存不足则 64 + 梯度累积）、lr 沿用 openpi 默认、**20k steps 起**
  （继承既有权重，只需学"读历史"），每 5k 存 checkpoint。
- 快速评测：memory 6 任务 × 3 seeds × 6 集。
- **评测前置（交接项 1/4）**：跑快速评测前复位 `deploy.yml`（`history_enabled: false` 或与训练同通路），
  并确认 server 启动日志打印的 history 配置与训练 `Pi0Config` 完全一致。
- **判据**：≥35 且其余维度不掉 >3 分 → 转 native 复测；<25 → 启动 5A。

**5A（兜底）全量微调**

- 触发条件：5C 快速评测 < 25 分（25~35 分先调一轮超参/数据配比，不急着放开主干）；
- 放开主干（`freeze_filter=nnx.Nothing`），60k steps / batch 256 对齐官方 recipe；
- **必须用全量 cotrain 数据**（不能只用 memory 子集），并同样从 RoboDojo 微调 checkpoint 继续训
  （不要退回 pi05_base 重训）；
- 显存不足时降 batch + 梯度累积，FSDP 加到 4 卡；
- 训练成本按 4090D 实测标定（Phase 0D 已有单步时延）。

### Phase 6：Memory 维度正式评测与归档（预计 2~3 天）

> **执行记录（2026-10-06 晚，评测链路就位；训练 19k/20k 收尾中）**：
> - **代码改动（均以 `history_enabled: true` + `history_mode: slots` 显式开启；不开时行为与基线完全一致）**：
>   1. `XPolicyLab/policy/Pi_05/model.py` 新增评测侧 `slots` 历史模式（`_attach_history` 分支）：
>      产出顶层 `history_images (20,3,128,128) uint8` + `history_mask (20,) bool`，与训练侧
>      `RoboDojoHistoryTransform` 同口径（25 步 1 帧、只取严格早于当前帧的样本、最旧在左、pad 槽 mask=False、
>      抽帧时 `resize_with_pad` 黑边缩到 128，与 sidecar 构建一致），由 `Policy.infer` 顶层透传。
>   2. 新增启动一致性断言 `_assert_history_eval_consistency`：slots 模式下逐项校验 deploy 的
>      `history_slots / history_action_interval / history_slot_size` 与 train_config
>      （`model.history_slots` + `RoboDojoHistoryTransform.interval/height`）一致，不一致直接拒绝启动；
>      非 slots 模式（Phase 3 消融）不做断言，互不影响。
>   3. `deploy.yml` 切到 Phase 6 评测配置：`train_config_name=pi05_aloha_sim_history_arx-x5_seed_0`、
>      `history_enabled: true`、`history_mode: slots`、`history_slot_size: 128`、`checkpoint_num: 19999`；
>      文件内注释保留基线回退方式（false + base 配置 + 59999）。
>   4. 新增冒烟 `scripts/pi05/smoke_pi05_eval_slots.py|.sh`：CPU 档验证装配/一致性断言（含负例）/
>      槽位语义/`history_*` 透传；`PI05_SMOKE_REAL=1` 档验证真 checkpoint 端到端活性（有 vs 无历史动作
>      差异、满 20 槽、重复前向一致、单次前向延迟）。
>   5. 新增编排 `scripts/pi05/phase6_kickoff.sh`：等训练最终 checkpoint `19999/params` → GPU 显存回落
>      → 真模型冒烟（门禁，失败不启动评测）→ memory 评测队列（6 任务 × seed0 × 6 集，2 worker：
>      policy GPU 0/2 + env GPU 1/3）；停止：`touch logs/pi05/p6_stop_watch`。
> - **验收（CPU 档已跑，2026-10-06 21:50）**：`logs/pi05/smoke_eval_slots_2026-10-06_21-50-32.log`
>   —— 20 项检查全 PASS（形状/dtype、52 步后 usable=2、pad mask=False、oldest-left、非方形输入 resize 到
>   128、透传一致、负例 history_slots=19 与非 history 训练配置均正确报错）。
> - **真 checkpoint 冒烟 + 快速评测**：kickoff 已武装（日志 `logs/pi05/p6_kickoff_2026-10-06_21-53-13.log`），
>   训练结束、门禁通过后自动执行；快速判据沿用 §5C（≥35 → native 复测；25~35 → 调参重试；<25 → 5A）。
>
> **训练收尾与首轮冒烟（2026-10-06 23:00~23:15）**：
> - 训练于 23:00 保存最终 `19999`（params/assets/train_state 完整，assets 含 `arx_x5_sim/norm_stats.json`）。
>   注意：看护脚本原判定 `last(19999) >= TARGET_STEPS(20000)` 恒假，会误重启 resume（实测 23:06 触发一次），
>   已修：改为 `last + 1 >= TARGET`（`scripts/pi05/watch_pi05_history_train.sh`）；同时 `touch logs/pi05/p5_stop_watch` 停看护。
> - **根因修复（评测侧透传被训练 transform 覆盖）**：history 训练配置自带 `RoboDojoHistoryTransform`，评测时
>   该 transform 因观测无 `episode_index/frame_index` 会把 `history_images` 覆盖成全 pad + mask=False；
>   而 `openpi/policies/policy.py` 的透传补丁原条件 `key not in inputs` 导致 `_attach_history` 的真历史被丢弃
>   （首轮真模型冒烟：有/无历史 Δ 恰为 0.0）。已改为「obs 里存在 history_* 即覆盖」。
> - 修复后真模型冒烟全 PASS（`logs/pi05/p6_real_smoke_fix_2026-10-06_23-13-09.log`）：
>   有 vs 无历史 Δ=1.301e-01、20 槽 vs 2 槽 Δ=5.940e-01、重复前向 Δ=0、满槽 20/20、
>   前向 0.14s（首帧 15s 为 JIT 编译）、显存 6.3GiB；`history_proj` 已训练（kernel absmax≈1.18e-2）。

> **Phase 6 A/B 实测（2026-10-06 23:14 ~ 2026-10-07 20:34）**：
> - 首轮快速评测（10-06 23:14，6 任务 × seed0 × 6 集，history 开）：cover_blocks 8.33 /
>   match_and_pick 0.0 / imitate_sorting 0.83 / 其余 0，均值 ≈1.8；10-07 09:28 的 6 集诊断
>   （history 关，CB+MAP × seed0）出现 MAP 16.67（= 基线 seed0 同值）与 CB 8.33（与开时相同），
>   一度疑「历史在评测域有害」。
> - **加大样本 A/B**（10-07 16:29~20:34，CB+MAP × 3 seeds × 20 集 × {on, off}，2 worker）：
>
>   | 任务 | history ON（s0/s1/s2） | 均值 | history OFF（s0/s1/s2） | 均值 |
>   | --- | --- | --- | --- | --- |
>   | cover_blocks | 13.75 / 26.0 / 26.25 | 22.00 | 18.50 / 22.50 / 21.50 | 20.83 |
>   | match_and_pick | 5.00 / 20.0 / 10.0 | 11.67 | 15.00 / 20.0 / 15.0 | 16.67 |
>   | 合计 | | **16.83** | | **18.75** |
>
>   - 配对差（on − off）：s0 −4.75 / −10.0；s1 +3.5 / 0.0；s2 +4.75 / −5.0（CB / MAP）。
>   - **结论**：总差 −1.9 分被种子波动（单 run 5~26 分）淹没，**历史 on/off 无显著效应**，
>     「6 集有害」证伪；当前 checkpoint 仍在基线量级（严格对齐需补 20 集基线复测），
>     记忆维度未提升 → 触发 §5C 判据（<25）的 5A 全量微调评估。
>   - 基础设施：seed0 首轮 2 个 run 均因 Isaac 冷启动段错误（rc=139，自动重试 2 次后放弃）缺失；
>     19:41 单跑一次补全（偶发故障，非 seed 特定）。A/B 期间 deploy.yml 由 trap 自动切换/还原。
>   - 产物：`logs/pi05/phase6_ab_{on,off}_2026-10-07_16-29-52/`（seed1/2）+
>     `..._2026-10-07_19-41-13/`（seed0 补跑）；编排日志 `logs/pi05/p6_ab_*.log`。

```bash
# 6 任务 × 3 seeds × native 50 trials = 900 episodes
bash scripts/pi05/phase_mem_pi05.sh --eval-num native --seeds 0,1,2 \
  --policy-gpus 0,2 --env-gpus 1,3
```

- 产出：与 DM0.5 榜单（47.74/47.44）、本集群 DM0.5 复现（44.03/43.52）、π0.5 基线三方对照表；
- 备份 `eval_result` 与日志到 `logs/pi05/phase9_eval_v2_<timestamp>/`（仓库内不存在 `runs/pi05_memory/`）；
- 更新本文档：结论 + 环境指纹 + 逐任务表（§8 模板）。

### Phase 7（可选）：五维全量回归

- 目的：确认记忆提升不是以牺牲其他维度换来的；
- 口径：`--eval-num native` + seed 0/1/2，4~8 卡并行；预计数天；
- 判据：其余维度 Δ ≤ 3 分。

---

### Phase 8：记忆关键帧加权（训练侧修复路线，Step 0–4）

> 触发依据：2026-10-09 训练分布上的离线 ON/OFF 对照证明——模型在训练域**完全不使用历史内容**：
> `Δ(zero−on)=+2.6e-7`、`Δ(croll−on)=+2.5e-7`（内容换黑帧/别集帧，loss 不动，≈0），
> 而 `Δ(maskroll−on)=+2.4e-4`（+16%，只动 mask 结构）、`Δ(off−on)=+1.6e-4`（+10%）。
> 即"历史通路"退化成了注意力结构/时钟通道 → 判定为**训练侧问题**（非评测域）。
> 证据文件：`logs/pi05/offline_history_loss_ab_2026-10-09_17-05-15.json`（4 臂 × 400 batch）、
> `logs/pi05/offline_history_loss_ab_2026-10-09_17-57-03.json`（分解臂 × 200 batch）；
> 工具：`scripts/pi05/offline_history_loss_ab.py`（臂：`on/off/shuf/zero/maskroll/croll`）。

目标：对"必须靠回忆才能做对"的关键帧做**采样/损失加权**，把历史内容通道真正打开。

**总判据（唯一 go/no-go 关口）**：记 `Δ = loss_bad_history − loss_on`。内容通道打开时，
用错内容或清零内容应使 loss **变大**，因此 `Δ(croll−on)` 或 `Δ(zero−on)` 应为
**显著正值**（建议 ≥ +1e-4 且 t > +5；当前为 ≈0）。通过后才允许花完整训练+评测预算。

#### Step 0：加权通道打通（2–3 小时，纯代码，不占卡）

- 权重表格式：`keyframe_weights.json` = `{"meta": {...}, "interval": 25, "weights": {"<episode_index>": [w0, w1, ...]}}`，
  1 Hz 分辨率（与历史槽同频），未列出的 episode 默认 `1.0`（通用 cotrain 不受影响）。
- 代码改动（4 处，默认关闭、行为向后兼容）：
  - `openpi/src/openpi/policies/robodojo_history.py`：新增 `keyframe_weights_path` 字段，读表并在
    `__call__` 输出 `loss_weight`（`lru_cache` 加载，与 `_load_index` 同套路）；
  - `openpi/src/openpi/models/model.py`：`Observation` 新增 `loss_weight: *b`（默认 `None`）+ `from_dict` 透传；
  - `openpi/scripts/train.py`：`loss_fn` 改为归一化加权均值 `Σ wᵢℓᵢ / Σ wᵢ`（`loss_weight=None` 时退回原 `jnp.mean`，
    保证不配表时逐位一致）；
  - `openpi/src/openpi/training/config.py`：`LeRobotAlohaDataConfig` 新增 `keyframe_weights_path` 字段并透传给 transform；
    另新增 TrainConfig `pi05_aloha_sim_history_kf_arx-x5_seed_0`（与 history 版同配方 + 权重表，表路径可用
    `ROBODOJO_KEYFRAME_WEIGHTS` 环境变量覆盖，便于 v1→v2 切换）。
- 验收：①不配表时 transform 输出与旧版一致（无 `loss_weight` 键）；②配表时 1 Hz 槽位取到正确权重；
  ③加权均值与手算一致；④5 步训练冒烟无 NaN、loss 尺度与基线相当。

#### Step 1：Tier A 规则表 v1（半天，含抽帧剖面校准）

- 先跑 `build_keyframe_weights.py --profile`：用 sidecar 的 1 Hz 帧算相邻帧视觉变化剖面（按 episode 相对时间分箱、
  跨 600 集聚合），数据化地定位"线索出现/被遮挡/交换"等事件时刻，替代"拍脑袋"划窗。
- 再按 6 个任务语义 + 剖面划关键段（`cover_blocks` 覆盖完成后；`match_and_pick_from_conveyor` 样例物移出视野后；
  `press_by_number` 每次按键后；`swap_T` / `swap_blocks` 交换后；`imitate_sorting_sequence` 示范结束后），
  普通帧 w=1、关键段 w=3~5、封顶 8 → 生成 `keyframe_weights_v1.json`（覆盖全部 600 集）。
- 产出：权重表 v1 + 每任务关键帧占比/权重分布统计 + 剖面报告。
- 验收：关键帧占比落在 20–40%（若某任务找不出关键段，则该任务 w=1 —— 本身是重要发现）。
- 注意：v1 是"能开跑的粗糙表"，不是最终答案；Tier B（Step 3）会替代/修正它。

#### Step 2：短训 + 离线门禁（关键决策点；5–21 小时卡）

- 用 v1 表从 `RoboDojo-sim-arx_x5-joint-2/59999` 起训（复用 `scripts/pi05/train_pi05_history.sh`，换 config/exp 名），
  实测步速 ≈3.8 s/it（4 卡 bs64）→ 20k 步 ≈21 小时；每 2000 步已有 checkpoint。
- 在 4k/6k、10k、20k checkpoint 上各跑一次**便宜门禁**：
  `offline_history_loss_ab.py --arms on,zero,croll --num-batches 100 --batch-size 16`（≈10 分钟/次）。
- 通过标准：`Δ(croll−on)` 或 `Δ(zero−on)` 出现显著正值（越早出现越好；`Δ = arm − on`）。
- 失败分支（20k 仍 ≈0）：①关键帧选错（答案其实在当前帧里）→ 回 Step 3；②结构/优化瓶颈 →
  转 DM0.5 式显式 memory 读写架构（附录 A）。

#### Step 3：Tier B 歧义度打分表 v2（1–2 天，可与 Step 2 并行）

- 定义：某帧的当前观测（图像+state）若能在**其它 episode** 找到视觉近邻、而近邻的**动作 chunk 明显不同**，
  则该帧必须靠历史才能判别 → 关键帧。
- 做法：mem6 的 1 Hz 侧车帧（≈1.3 万张）→ SigLIP 提特征（约几十分钟/单卡）→ 跨 episode 同任务 kNN（k≈8，
  同时校验 state 距离）→ 近邻动作 chunk 分歧度 `sᵢ` → `wᵢ = clip(1+α·sᵢ, 1, 8)`。
- 产出：`keyframe_weights_v2.json` + 诊断报告（`s` 分布直方图、top 帧抽检/可视化）。
- 验收：v2 的 top 20% 高分帧与 v1 规则段重合 >60%；**若 `sᵢ` 几乎全 0 → 说明训练数据本身没有"必须用记忆"的
  监督信号，立即停重训，改数据/课程或架构**（这是本步最重要的诊断价值）。

#### Step 4：最终重训 + 评测（~1 天卡 + 半天评测）

- 用最终表（v2；若与 v1 差异小则用 v1）完整训 20k 步 → 先过离线门禁 → 再跑 Memory 维度评测
  （`phase_mem_pi05.sh`，3 seeds × 20 集，同口径对比基线 22.0 / 11.7）→ 补跑非记忆任务回归。
- 通过标准：Memory 分数显著高于基线、其他任务不退化（目标向 DM0.5 的 47.74 靠拢）。

**资源与顺序**：Step 0+1 ≈ 1 天（人）；Step 2 ≈ 5–21 小时（卡，需 4 卡作业，当前两个 gpu 分区均满，尽早排队）；
Step 3 ≈ 1–2 天（人，可与 Step 2 并行，仅占 1 卡几十分钟）；Step 4 ≈ 1.5 天（卡+评测）。关键路径 ≈ 3 天。

#### Phase 8 执行记录（2026-10-10：Step 0 + Step 1 完成）

- **Step 0（加权通道，4 处既有文件改动，默认关闭）**
  - `openpi/src/openpi/policies/robodojo_history.py`：`RoboDojoHistoryTransform` 新增 `keyframe_weights_path` +
    `_load_keyframe_weights/_keyframe_weight`（1 Hz 槽位 `frame_index // interval`）；配置后额外输出 `loss_weight`。
  - `openpi/src/openpi/models/model.py`：`Observation.loss_weight`（`*b`，默认 `None`）+ `from_dict` 透传。
  - `openpi/scripts/train.py`：`loss_fn` 归一化加权均值 `Σwᵢℓᵢ/Σwᵢ`；`loss_weight=None` 时退回 `jnp.mean`（逐位一致）。
  - `openpi/src/openpi/training/config.py`：`LeRobotAlohaDataConfig.keyframe_weights_path`；新增 TrainConfig
    `pi05_aloha_sim_history_kf_arx-x5_seed_0`（表路径默认 `keyframe_weights_v1.json`，可被 `ROBODOJO_KEYFRAME_WEIGHTS` 覆盖）。
  - 验证（全部通过）：①transform 取值 `0/24→1.0、25→2.0、60→4.0`、越界/未列出→1.0；②不配表时不输出 `loss_weight`（旧行为不变）；
    ③`Observation.from_dict` 透传 `[1., 4.]`、缺省 `None`；④新配置可加载（旧配置字段保持 `None`）；⑤真实训练管线冒烟（单卡 bs16 × 5 步）：
    日志确认 `[0].loss_weight: (16,)@float32`、`Step 0: loss=0.0100`（Phase 5 基线同口径 0.0076，量级一致）、checkpoint 正常保存。
- **Step 1（Tier A 规则表 v1）**
  - 新增 `scripts/pi05/build_keyframe_weights.py`：`--profile`（1 Hz 相邻帧视觉变化剖面，相对时间 10 分箱 × 600 集聚合）
    / `--tier-a`（任务规则 → 权重表）。槽位数学核对：`num_samples == ceil(length/25)`，600/600 集成立。
  - 剖面要点：`imitate_sorting_sequence` 前 40% 平静、0.4 后跃升（示范→执行）；`match_and_pick_from_conveyor` 活动度爬升到结尾
    （峰值中位 0.89）；`cover_blocks` 0.3–0.4 局部峰 + 峰值中位 0.70；`swap_T` 峰值中位 0.22 → 依此划窗（规则见脚本注释）。
  - 产物：`robodojo/data/history_sidecar/keyframe_weights_v1.json`（600 集 / 13,407 槽；关键帧占比 **63.3%**、平均权重 **2.82**）
    + 剖面报告 `logs/pi05/keyframe_profile_v0.json`。
  - 偏差说明：关键帧占比 63.3% 高于初估 20–40% —— v1 采用"事件之后整段"的保守宽窗（首要目标是跑通+验证假设；
    Step 3 的 Tier B 歧义度打分再收窄/校正）。
  - 冒烟 checkpoint：`checkpoints/kf-smoke-20261010/`（9.1 GB，仅验证用）。
- **下一步**：Step 2 短训 + 离线门禁（checkpoint × `--arms on,zero,croll`，出现显著正 Δ 即通过）；
  Step 3 Tier B 打分表（可并行，产出 v2）。当前阻塞：重训需 4 卡作业且分区已满，需尽早排队。

#### Phase 8 执行记录（2026-10-10 下午：Step 2 启动 + Step 3 完成 + 训练表切换）

- **Step 2 训练（后于 2026-10-11 12:49 完成；最终 ckpt=19999）**
  - 15:58 启动 v1 表训练（作业 `<JOBID-TRAIN>`，gpu04，3 卡，`bs48/fsdp=1`，实测 3.7 s/it）；17:05 产出 checkpoint `1000`。
  - 17:08 停旧步骤，改以 **v2 表续训**（`ROBODOJO_KEYFRAME_WEIGHTS=.../keyframe_weights_v2.json` + `--resume`，从 1000 步继续）；
    日志已确认 v2 生效（`keyframe_weights_path='.../keyframe_weights_v2.json'`）。
  - 口径说明：0–1000 步（约 5% 训练量）使用 v1 权重，其后为 v2；两者关键段高度一致（见下），对结论无影响，但归档时须注明。
  - 实际：约 19–21 小时，2026-10-11 12:49 完成 20k 步。
- **Step 2 门禁看护**
  - 新增 `scripts/pi05/p8_gate_watchdog.sbatch`（1 卡）并提交（作业 `<JOBID-GATE>`，gpu01）：每 10 分钟扫描 checkpoint 目录，
    对每个新 checkpoint 自动跑 `offline_history_loss_ab.py --arms on,zero,croll --num-batches 100 --batch-size 16`，
    结果写 `logs/pi05/gate_kf_<step>.json`；17:11 已开始跑 `ckpt=1000`。
- **Step 3 Tier B 完成（数据驱动歧义度打分）**
  - 新增 `scripts/pi05/build_keyframe_weights_tier_b.py`：1 Hz 侧车帧 → ResNet-18(ImageNet) 512-d 编码 →
    同任务跨集 kNN（图像距离 + 0.5×state 距离）→ 近邻前 8 步动作 chunk 的 RMSE（按维度标准化）→ `w=clip(1+3s/s₉₀,1,8)`；
    另新增汇总脚本 `scripts/pi05/summarize_p8_progress.py`（权重表统计 + Tier B 诊断 + 门禁 Δ 序列）。
  - 产物：`robodojo/data/history_sidecar/keyframe_weights_v2.json`（600 集 / 13,407 槽；平均权重 **2.18**、`w>1` 占比 87.1%、
    max 8）+ 诊断 `logs/pi05/keyframe_tierb_diag.json` + 编码缓存 `logs/pi05/keyframe_tierb_emb.npz`。
  - 诊断要点：歧义度中位最高的是 `swap_T`(0.18)、`imitate_sorting_sequence`(0.16)；`match_and_pick_from_conveyor`
    中位仅 0.002 但 p90=0.80 —— 大多数帧无歧义、少数帧强歧义（说明"关键帧"确实稀少且集中）。
    `cover_blocks`/`press_by_number`/`swap_blocks` 介于两者之间。
  - **验收**：v2 top-20% 高分帧与 v1 关键段重合 **89.8%**（验收线 >60% ✓）。
  - **诊断结论**：训练数据中确实存在"同观测、不同动作"的帧（歧义度显著非零）→ 记忆监督信号存在；
    是否足以驱动模型学会用历史，仍由 Step 2 门禁判定（预计 2000/4000 步即可见分晓）。

#### Phase 8 执行记录（2026-10-10 晚间：门禁 ckpt1000/2000 + 6 臂诊断上线）

- **门禁（3 臂 on/zero/croll，100 batch × bs16 = 1600 样本）**
  - ckpt1000（17:51）：`on=0.002459, zero=0.002471, croll=0.002461`；Δ(zero−on)=+1.2e-5（t=+0.77）、
    Δ(croll−on)=+2.2e-6（t=+0.11）→ ≈0，未触发判据。
  - ckpt2000（18:30）：`on=0.002225, zero=0.002226, croll=0.002230`；Δ(zero−on)=+1.3e-6（t=+0.19）、
    Δ(croll−on)=+5.0e-6（t=+0.52）→ ≈0，未触发判据。
  - 判定：2k 步前内容通道仍未打开（符合预期）；**当时文本误写为负门槛**，正确门槛是
    Δ ≥ +1e-4 且 t>+5（见 Phase 9 后的勘误）；继续看 4k/6k（约 22:30 / 次日 01:00）。
    功效分析：若真实效应 Δ=−1e-4，本口径 t≈−10（zero）/−6.9（croll）→ 100 batch 足够，不会漏报。
- **看护修复**：`p8_gate_watchdog.sbatch` XLA 显存比例 0.5→0.9（修复 ckpt1000 首跑 OOM）；连续失败 3 次写 `.fail` 跳过
  （备份 `.bak-20261010_174100`）。
- **门禁升级为 6 臂诊断**：新增 `scripts/pi05/p8_ab6_watchdog.sbatch`（作业 `<JOBID-AB6>`，1 卡，18:33 启动）：
  对所有新 ckpt 跑 `--arms on,off,shuf,zero,maskroll,croll`（约 20 分钟/ckpt，6 臂覆盖原门禁 3 臂），
  输出 `logs/pi05/ab6_kf_<step>.json`，并派生同口径 `logs/pi05/gate_kf_<step>.json`（既有门禁工具链不变）。
  用途：把 Δ(shuf−on) 拆成"mask 结构效应"（maskroll）与"内容效应"（croll），分辨"内容没学会" vs "只学会多一段前缀 binder"。
- **配额/资源事实（重要）**：本账号 QOS `32cpu4gpu64g` 为**全局上限**（跨分区按用户计）：32 CPU / 4 GPU / 64G 内存。
  训练占 3 GPU + 24 CPU + 32G → 全局只剩 1 GPU / 8 CPU / 32G；第 5 张 GPU 作业报 `QOSMaxGRESPerUser`，
  超 32 核作业报 `QOSMaxCpuPerUserLimit`。因此"多申请到的卡"须等 QOS 上限提高（管理员改 QOS）或训练结束（释放 3 卡）后才可用。
- **集群现状（18:30）**：gpu01 的 3 张空闲卡中 2 张于 18:28 被 AlexWen（4h 限额）占用，剩 1 张用于本批次 ab6；其余节点全满。
- **ckpt2000 六臂分解（18:52 完成，100 batch × bs16）**：
  `on=0.0022257, off=0.0021901, shuf=0.0024000, zero=0.0022269, maskroll=0.0024172, croll=0.0022294`；
  Δ(off−on)=−3.6e-5(t=−0.6)、Δ(shuf−on)=+1.7e-4(t=+3.0)、Δ(zero−on)=+1.3e-6(t=+0.2)、
  Δ(maskroll−on)=+1.9e-4(t=+3.1)、Δ(croll−on)=+3.7e-6(t=+0.4)。
  **结论**：模型对 mask 结构错位显著敏感（shuf/maskroll 显著变差），对历史内容完全无感（zero/croll≈0）；
  连"完全无历史"（off）也不比 on 差 → 2k 步时历史输入近似"噪音前缀"，内容通道未打开。
- **"内容无感"机理分析（19:0x）**：代码核对 `robodojo_history.py` —— mask 的 True 个数 = `min(frame_index//25, 20)`，
  即 **mask 本身是严格单调的"剧集时钟"**（已过去多少秒，20 秒后饱和）。据此：
  ① mask 泄露"剧集相位"这个强上下文 → 模型学"读时钟"这条捷径（maskroll 打乱时钟 → loss 显著变差 +1.9e-4, t=+3.1）；
  ② 分箱验证（按 `usable` 分 6 箱）：**每个相位的 Δ(zero−on)、Δ(croll−on) 都 ≈0**（|t|<1），
  maskroll 效应在 usable 少的相位最大（+2.6~2.9e-4）、饱和段（usable=20，各样本 mask 都≈全 True）最小（+4e-5）→ 与"读时钟"完全吻合；
  ③ 任务层：Tier B 歧义度集中于少数帧（大多数帧答案在当前帧）→ 内容对平均 loss 的边际贡献被稀释；
  ④ 目标层：v2 权重把 87% 槽位都设 w>1，加权被摊薄；⑤ 模型层：history 分支是新增模块（2k 步 ≈9.6 万样本梯度）+ LoRA/冻结视觉塔，可塑性有限。
  **结论**：不是"读不懂内容"，而是"没有动力去读"——2k 步内的最省力解是拿 mask 当时间码、把内容当噪音前缀。
- **ckpt 保留策略**：orbax 只保留最新 1 个 checkpoint（ckpt1000 于 18:13 自动删除；数据不丢，`gate_kf_1000.json` 已存档）。
  watchdog 须在 ckpt 存活窗口（约 65 分钟）内完成（6 臂约 20 分钟，安全）。
- **下一步**：盯 ab6/judge 序列到 4k/6k；通过后进 Step 4（先把 `deploy.yml` 的 `train_config_name` 切到
  `pi05_aloha_sim_history_kf_arx-x5_seed_0`，ckpt 用 kf 目录），并等训练收尾后用 4 卡跑 Memory 维度评测。



#### Phase 8 执行记录（2026-10-10 18:5x：断线护航 —— Step 4 自动接力链）

- **背景**：用户可能离线过夜。核实：训练作业 `<JOBID-TRAIN>` 的宿主在 tmux 会话 `nbh`（断 SSH 不受影响），
  ab6 看护 `<JOBID-AB6>` 为 sbatch 作业；两者都会继续跑。唯一缺的是"训练结束后切配置/发评测"的人 → 用 Slurm 依赖链补上。
- **新增 `scripts/pi05/p8_step4_pre.sbatch`（作业 `<JOBID-PRE>`，`--dependency=afterok:<JOBID-TRAIN>`，无 GPU）**：
  训练结束即取消 `p8_ab6`，释放本账号配额（QOS 全局 4 卡上限：训练 3 卡 + ab6 1 卡）。
- **新增 `scripts/pi05/p8_step4_relay.sbatch`（作业 `<JOBID-RELAY>`，`--dependency=afterok:<JOBID-PRE>`，4 卡，`--nodelist=gpu01`）**：
  ① `p8_gate_judge.py --require-pass`：门禁未过 → 写 `logs/pi05/step4_skipped_gate_failed.flag` 并退出（不烧卡）；
  ② 备份并把 `deploy.yml` 切到 `pi05_aloha_sim_history_kf_arx-x5_seed_0`、`checkpoint_num`=kf 目录最大 step；
  ③ `phase_mem_pi05.sh --eval-num 6 --policy-gpus 0,2 --env-gpus 1,3`（2 worker，与基线 22.0/11.7 同口径；EVAL_NUM 可覆盖）。
- **为什么固定 gpu01**：只有 gpu01 的 podman storage 有 `robodojo:isaac6` 镜像（17G）；gpu04 为空（120K）。
- **`p8_gate_judge.py` 增加 `--require-pass`**（无任一判据通过时返回非零，供接力作业判据使用）。
- **⚠️ 变更（19:00，用户要求）**：自动接力链**已取消**（`scancel <JOBID-PRE> <JOBID-RELAY>`）。规则改为：
  **训练跑完即停，不自动触发任何后续评测**；Step 4 评测须用户明确指令后手动启动。
  同时用户要求 **<JOBID-TRAIN> 的 3 卡分配在训练结束后不释放**（保持 allocation 占用，评测/后续任务将来直接
  在其内用 `srun --jobid=<JOBID-TRAIN> --overlap` 运行；allocation 硬上限为作业 2 天墙钟，约 2026-10-12 15:57）。
  gpu04 上评测容器镜像缺失（`/tmp/podman-build-$USER` 为空）；镜像可从
  `images/robodojo-isaac6.tar`（38.35 GB，与 gpu01 上 `localhost/robodojo:isaac6` 38.4GB 对应）载入，
  单次 `podman load` 约 10 分钟。
- **离线期间的风险清单**：①训练若崩溃无人重启（`Requeue=1` + tmux 保护；NFS 04:00 重启是窗口）；②ab6 若挂 →
  门禁 JSON 停止产出 → 接力自动跳过评测（安全失败，不浪费卡）；③其余无人依赖项。


#### Phase 8 执行记录（2026-10-11 12:49：训练完成 + 门禁最终裁决）

- **训练完成**：20k 步跑完（最终 checkpoint step=**19999**，12:49:21 保存完成；保留 5000/10000/15000/19999）。
- **门禁最终裁决：全部 20 个 checkpoint（1000→19999）均未通过**（Δ 始终接近 0；当时文本把门槛误写为负，
  正确方向见 Phase 9 后的勘误；由于 |Δ| 远小于 1e-4，历史结论不变）。
  全程 Δ(zero−on)/Δ(croll−on) 在 ±1e-5 内、|t|≲1（个别 ckpt 最大 |t|≈2.4 且为正值）。
- **最终 19999 六臂（100 batch × bs16）**：`on=0.001579, off=0.001699, shuf=0.001790, zero=0.001582, maskroll=0.001788, croll=0.001582`；
  Δ(off−on)=+1.20e-4(t=+2.21)、Δ(shuf−on)=+2.11e-4(t=+2.29)、Δ(zero−on)=+2.9e-6(t=+0.30)、
  Δ(maskroll−on)=+2.10e-4(t=+2.31)、Δ(croll−on)=+3.1e-6(t=+0.38)。
- **结论（Step 2 门禁失败 → 触发规划失败分支）**：关键帧加权训练 20k 步后，**内容通道仍未打开**
  （zero/croll ≈ 0）；模型确实在用"历史前缀"，但增益完全来自 **mask 结构（剧集时钟）**
  （off 去掉历史变差 +1.2e-4、maskroll 打乱时钟变差 +2.1e-4，二者幅度一致且显著；内容既换掉、清零都不影响 loss）。
  失败分支二选一：① 回 Step 3 复核关键帧选择/歧义度（是否"答案其实在当前帧里"）；② 转 DM0.5 式**显式 memory 读写架构**。
- **资源状态（2026-10-11 13:30）**：`<JOBID-TRAIN>` allocation 按用户要求保持占用（3 卡，硬上限 10-12 15:57）；
  `<JOBID-AB6>`（ab6 看护）已无新 ckpt 可跑，将在 18:33 到达 24h 额度后自动结束。
- **评测暂缓**：按用户 2026-10-10 指示"训练跑完即停，先不要跑后面的内容"，**未启动任何 RoboDojo 仿真评测**；
  Step 4 评测需用户明确指令。

### Phase 9：断时钟捷径 + 去零初始化瓶颈 + 内容对比损失（v2 配方，2026-10-17 起）

> 触发依据（Phase 8 门禁失败结论）：kf 加权训练 20k 步后内容通道仍未打开——
> `Δ(zero−on)=+2.9e-6`、`Δ(croll−on)=+3.1e-6`（内容换掉/清零都不影响 loss），
> 而 `Δ(off−on)=+1.2e-4`、`Δ(maskroll−on)=+2.1e-4`（去掉历史 / 打乱 mask 都显著变差）。
> 即模型学到的"记忆"≈ **mask 结构本身**（20 槽 + 每帧按 `frame_index//25` 递进的可用槽数 =
> 单调剧集时钟），而不是历史内容。本轮把三个最可疑点一次改掉，**只看离线门禁是否翻转**。

三个开关（全部默认关闭、旁路式，不影响既有默认行为）：

1. **断时钟捷径**（数据侧）：旧 `history_mask` 是单调时钟（第 k 帧可用槽 = `min(k//25, 20)`，
   单调递增 → 模型可以直接拿它数时间）。新增 `RoboDojoHistoryTransform(pad_mode=...)`：
   - `zero`（默认，旧行为）：空槽填 0、`mask=False`；
   - `repeat_oldest`（v2）：空槽用**最旧有效帧**回填、`mask` **恒为全 True** → 槽位与时刻解耦。
   评测侧 `policy/Pi_05/model.py:_attach_history` 同口径支持（`history_pad_mode` 配置 + 一致性断言）。
2. **去零初始化瓶颈**（写侧）：`history_proj` 零初始化使 step 0 历史 token 输出恒 0，梯度要先把投影
   "打开"才谈得上用内容。改用**恒等初始化**（`history_inject="proj_identity"`：step 0 输出 = 池化特征，
   等价直注但保留可塑性；`direct` = 完全不建投影）。权重加载 `BaseModelConfig.load()` 在 proj_identity
   时用恒等核补齐缺失的 `history_proj.kernel`（否则会静默退回零初始化瓶颈）。
3. **内容对比损失**（损失侧）：`max(0, margin − (loss_roll − loss_on))`，roll = 同 batch 内 roll=1 的
   错位历史（与离线门禁 `croll` 臂同口径）；`margin = 0.1 × 当批 on-loss 均值`（stop_gradient）、
   β=0.5、只对前 50% 样本算 roll 分支（省算力）。逐样本 hinge + `aux = β·Σhinge/b`：β<1 保证 on 侧
   永远是下降方向（不会为了拉大 gap 反而抬高 on-loss）；`contrast_stop_grad=False` = 对抗式。

v2 配置：`pi05_aloha_sim_history_v2_arx-x5_seed_0`（5C-LoRA 同配方 + 上三项）。
判据（唯一 go/no-go）：`Δ(zero−on)` 或 `Δ(croll−on)` ≥ +1e-4 且 t > +5（`Δ = bad_history − on`）。

#### Phase 9 执行记录（2026-10-17）

- **代码改动（默认关闭、旁路式）**：模型侧 `pi0_config.py` / `pi0.py` / `model.py`（新增 9 条 patch 条目，
  `patch_pi05_history.py` 现共 25 条，`--check` 全绿）。**注意：该脚本只覆盖 5 个 openpi 文件**；
  `training/config.py`、`scripts/train.py`、`training/data_loader.py`、`training/weight_loaders.py`、
  `policy/Pi_05/model.py`、`deploy.yml` 和新增的 `robodojo_history.py` 由完整 diff/overlay 补丁复现，
  不能仅靠 `patch_pi05_history.py` 重建整个工作区。
  数据侧 `robodojo_history.py`（`pad_mode`）、`openpi/src/openpi/training/config.py`
  （`history_pad_mode` 透传 + v2 TrainConfig）、`scripts/train.py`（`loss_fn` 返回 `(loss+aux, info)`，
  `nnx.value_and_grad(has_aux=True)`）；评测侧 `policy/Pi_05/model.py`（`history_pad_mode`）与
  `deploy.yml`（默认 `zero`；切 v2 权重时必须同步改 `repeat_oldest`，一致性断言会强制）。
- **离线验证（CPU 单测 + 单卡分段，不占训练卡）**：
  - `test_pi05_history_v2.py` 7/7 通过：时钟单调性对照（zero 的槽位钟 `[0,0,1,2,8,20]` vs
    repeat_oldest 恒定）、窗口满时逐位一致、无历史全 pad、评测侧同口径、一致性断言、恒等核形状；
  - `verify_pi05_history_v2.py`：proj_zero 历史 token 恒 0 / proj_identity == 池化特征（max_err=0）/
    direct 无 `history_proj` 参数；contrast roll `aux=β·margin/2`、`hinge_frac=1.0`；stop_grad 仍有梯度；
    contrast 关闭时 aux 恒 0；
  - 5 步冒烟（bs12、1 卡）跑通：`Step 0: contrast_aux≈0, hinge_frac=0.667, gap=0.0012 > margin=0.0002,
    loss=loss_main=0.0013, grad_norm=0.04`（三开关在真实训练管线全部生效，无 NaN）。
- **正式训练（已于 2026-10-18 17:03 完成）**：`scripts/pi05/p9_train_v2_launch.sh` → **4 卡（0/1/2/3）**、`fsdp=1`、
  bs48（12/GPU）、20k 步；实际 elapsed **24h50m**；最终 checkpoint `19999`；
  exp = `RoboDojo-sim-arx_x5-history-v2-lora-seed0`（**1200 集**训练清单 = mem6 600 + 通用 600，
  权重起点 `RoboDojo-sim-arx_x5-joint-2/59999`）。
- **启动期踩坑（已修）**：①容器内 `ulimit -u` 软上限仅 256（硬上限 ~3e6）→ 多卡 + DataLoader workers
  触发 `RuntimeError: can't start new thread`；已在 `train_pi05_history.sh` 与两个 launcher 里
  `ulimit -u 8192`。②bs48 @3 卡（16/GPU）叠加对比损失的第二次前向后 OOM（XLA 报告 22 GiB 起、
  单次申请 11.1 GiB）→ **每卡 ≤12 样本**才安全；4 卡 bs48 = 12/GPU 为已验证口径。
- **门禁看护（GPU 3，显存门限保护）**：`scripts/pi05/p9_gate_watchdog.sh`（新 checkpoint 出现即跑
  `on,zero,croll` 三臂 100×16 ≈9 min），输出 `logs/pi05/gate_v2_<step>.json/.npz`；因 4 卡训练占满显存，
  看护加了 `P9_GATE_MIN_FREE_MIB`（默认 9000 MiB）门限 → **训练结束后自动补跑保留的 4 个 ckpt**
  （5000/10000/15000/19999；训练中出现的 1000–4000 会被 keep_period=5000 剪掉）。
  裁判：`python3 scripts/pi05/p8_gate_judge.py --patterns "gate_v2_*.npz"`（新增 `--patterns`，默认值不变）。
- **中途门禁（未取到）**：15:49 启动的**旧看护步骤**（缺 ulimit 修复、无显存门限）对 ckpt 1000–7000 的
  尝试全部失败——`std::system_error: Resource temporarily unavailable`（线程/进程数上限，非门禁结论），
  并在 `logs/pi05/` 留下 `.fail`/`.tries` 标记；2026-10-18 01:2x 已杀掉旧步骤、清除全部标记，
  只保留带显存门限的新看护 → 训练结束（预计 10-18 ≈16:50）后自动补跑保留的 ckpt。
- **对照基线（Phase 8 kf 19999，1600 样本）**：`on=0.001579, zero=0.001582, croll=0.001582`，
  `Δ(zero−on)=+2.9e-6`、`Δ(croll−on)=+3.1e-6` —— v2 要翻转的正是这两个数。
- **状态（归档）**：训练与门禁看护均已在 Slurm 分配 `<JOBID>` 内跑完；训练结束后未自动触发仿真评测。
  当前无训练作业，v2 checkpoint 为 `19999`。

#### Phase 9 门禁结果（2026-10-18 17:46，训练完成后自动补跑）

训练：20k 步全部跑完（elapsed 24h50m，4 卡，最终 ckpt step=19999 于 17:03 保存完成；保留 5000/10000/15000/19999）。
门禁：`p9_gate_watchdog.sh` 在训练结束后（GPU 空闲）自动对 4 个 ckpt 跑 `on/zero/croll`（100×16=1600 样本，≈11 min/个）。

| step | Δ(zero−on) | t | Δ(croll−on) | t | on-loss |
| --- | --- | --- | --- | --- | --- |
| 5000 | **+2.37e-2** | +55.0 | **+1.01e-3** | +7.1 | 0.002228 |
| 10000 | **+1.68e-1** | +58.5 | **+7.07e-3** | +7.1 | 0.002185 |
| 15000 | **+9.28e-2** | +42.2 | **+3.76e-3** | +6.6 | 0.001899 |
| 19999 | **+1.47e-1** | +38.2 | **+6.02e-3** | +6.0 | 0.001724 |
| （kf 基线 19999） | +2.9e-6 | +0.30 | +3.1e-6 | +0.38 | 0.001579 |

**判定（原始事实）**：内容通道**已显著打开**——错位内容 `Δ(croll−on)` 从 +3.1e-6（t≈0.4）升到
**+6.0e-3（t=+6.0）**（≈2000×，|t| 远超 5）；清零内容 `Δ(zero−on)` 从 +2.9e-6 升到 **+1.5e-1（t=+38）**
（zero 臂 loss 0.1492 vs on 0.0017，≈87×）。即"内容换掉/清零 → loss 大幅变差"，与 Phase 8 的
"内容无感"完全相反。训练侧同步可见：`contrast_gap`（同噪声下 loss_roll−loss_on）从 step 0 的 0.0004
增长到 0.03–0.04，hinge 长期有 ~72% 样本激活（`contrast_aux` 因 4 位小数显示为 0.0000）。

> **判据符号已校正（2026-10-20）**：旧文把门槛写成负值，属于符号错误。由于
> `Δ = arm − on`，正确语义是"坏历史 loss 更大"，所以门槛应为 **Δ ≥ +1e-4 且 t > +5**。
> 按修正后的判据，v2 四个保留 checkpoint **全部通过**；Phase 8 的 Δ≈0 仍判为未打开，历史结论不变。

**注意（敏感 ≠ 有效）**：该门禁只证明"模型依赖历史内容"，不证明"回忆能提升任务成功率"。
是否真正涨点需 RoboDojo 仿真评测（须用户明确指令；届时 `deploy.yml` 要同步切
`history_pad_mode: repeat_oldest` + v2 checkpoint）。

---

#### Phase 9 评测结果（2026-10-19，Memory 维度快速口径）

- **部署**：`deploy.yml` 切到 v2（`train_config_name: pi05_aloha_sim_history_v2_arx-x5_seed_0`、
  `history_pad_mode: repeat_oldest`、`checkpoint_num: 19999`；备份 `deploy.yml.phase8kf.bak-20261019_1232`）。
  启动一致性断言通过：`history(slots) 与训练配置一致 … pad_mode=repeat_oldest`，`history active 20/20`。
- **口径**：6 任务 × seeds 0/1/2 × **6 集**（= Phase 1 基线同口径），单 worker（GPU0 policy server / GPU1 Isaac client），
  Slurm 作业 `<JOBID>`；队列日志 `logs/pi05/phase9_eval_v2_20261019_1358/`（13:57:25–17:27:14）。
- **结果**（score；单集满分 1.0，任务分 = 6 集均值 × 100）：

| task | s0 | s1 | s2 | 均值 | 基线 | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| `cover_blocks` | 9.17 | 9.17 | 21.67 | **13.33** | 22.50 | −9.17 |
| `match_and_pick_from_conveyor` | 16.67 | 33.33 | 16.67 | **22.22** | 5.56 | **+16.67** |
| `swap_T` | 0 | 0 | 0 | 0 | 0 | 0 |
| `press_by_number` | 0 | 0 | 0 | 0 | 0 | 0 |
| `imitate_sorting_sequence` | 0.83 | 0.83 | 3.33 | **1.67** | 1.11 | +0.56 |
| `swap_blocks` | rc=3 | rc=3 | rc=3 | n/a | 0.00 | — |
| **均值** | | | | **7.44**（5 任务）/ **6.20**（6 任务，swap_blocks 记 0） | **4.86** | **+1.34** |

> **说明**：SR 5.56%（15 个有效 run；swap_blocks 记 0 则 4.63%）。

- **`swap_blocks` 3/3 失败（仿真侧，根因未闭环）**：容器 client 在 Isaac 启动/场景加载阶段（约 4 min）崩于
  `PhysX Internal CUDA error 700`；policy server 端 **0 次推理**（无 `history active`）→ 模型未参与。
  Phase 6 记忆模型 1/1 + 本轮 3/3（共 4/4）均如此；无记忆基线（2026-10-04）3/3 成功。本轮按 0 分记账，待复跑。
  内核日志对应 `NVRM: Xid 31 … ENGINE GRAPHICS … FAULT_PDE ACCESS_TYPE_VIRT_READ`（仅 env GPU，其他任务 0 次）。
  2026-10-19 的 10-env 对照进一步显示：原场景 3/3 `PHYSX700`；`cover_blocks` PASS；垫子/材质变体仍 `PHYSX700`；
  `skip_reloc` 第 1 次 PASS（482 s）但第 2 次 `PHYSX700`；`no_app_update` 与 `normal_detail` 也 `PHYSX700`。
  → **跳过 articulation relocation 不能稳定通过，触发点会在 reset/场景重建链路中漂移，根因仍未定位**。
  可确认的是崩溃发生在首次动作推理之前，属仿真/场景构建问题，而不是策略权重或历史通路输出错误。
- **结论**：
  1. 离线门禁的"内容通道已打开"**只部分转化为任务收益**：唯一实质提升是 `match_and_pick`
     （5.56 → 22.22，主要来自 seed1 的 2/6 整集成功）。
  2. `cover_blocks` 从 22.50 掉到 13.33：6 集口径下单集翻转即 ±16.67 分（基线 22.5 中 16.67 来自唯一一整集），
     s0/s1 都差那一集；**不是能力崩塌**（v2 的部分分 0.55 高于基线 0.35，且同权重 MAP 解 1/6 与基线同分）。
  3. `press_by_number` 仍为 **0**（DM0.5 = 95.33）——40 分目标的主要杠杆在这里（自运动计数，1 Hz 视觉历史可能不够）。
  4. 总分 **6.20 / 7.44**，较基线 4.86 仅 +1.34，距 40 分判据仍差 ≈34 分；判据未触发，项目继续。
  5. 结论：**"内容敏感"（离线门禁）≠"回忆可提升成功率"**；下一步优先攻 `press_by_number` 与
     `cover_blocks` 整集成功率，而不是继续加码对比损失。

- **运行备注（环境）**：首次启动 18 run 全 rc=3——gpu01 的 `/tmp` podman 存储镜像层被清空
  （仅剩 17M、容器 rootfs 空）；用 `images/robodojo-isaac6.tar`（38.35 GB）`podman load` 恢复
  （≈72 min，存储回到 17G）后重启队列。另：`smoke_pi05_eval_slots.py` 的"真模型 2 槽组装"在
  `repeat_oldest` 下是过期断言（2 个唯一帧会回填 20 槽、mask 全 True），其余核心项通过
  （Δ(on/off)=0.87、满槽、单次前向 0.14 s）。

---

## 6. 脚本清单

### 6.1 新增（全部落在 `scripts/pi05/`）

| 脚本 | 阶段 | 职责 | 关键接口 |
| --- | --- | --- | --- |
| `download_ckpt.sh` | 0A | 拉 π0.5 RoboDojo 权重到 `policy/Pi_05/checkpoints/` | 走 `HF_ENDPOINT=https://hf-mirror.com`；校验 `params/`+`assets/` |
| `fetch_demos.sh` | 0B/4 | 封装官方 `scripts/RoboDojo/download_data.sh`（HF/ModelScope） | 位置参数：`<format> <source>`；训练用 `lerobot_v3.0`。脚本**没有**任务过滤参数，任务子集由 sidecar/训练清单控制 |
| `probe_pi05_runtime.py` | 0C/0D | 标定单次推理时延/显存、每 chunk 实际执行动作数 | 输出 JSON 供后续引用 |
| `patch_pi05_history.py` | 2/4 | 幂等补丁：给 openpi 加 history 通路与 config 字段 | `--check` 只检查；`--revert` 还原 |
| `verify_pi05_history_liveness.py` / `.sh` | 2 | 离线记忆活性验证（A1/A2/B/C/D 五组） | 仿 `scripts/verify_memory_liveness.py` |
| `phase6_ab_history.sh` | 3/6 | 训练后历史开/关 A/B（部署侧切换 `history_enabled`）；B1/B2/B3 仅保留在 `model.py`/`deploy.yml` 中手动切换 | 环境变量 `TASKS/SEEDS/EVAL_NUM/CKPT`；`JOBID` 指向现有分配 |
| `build_history_sidecar.py` | 4 | 从 LeRobot 数据集抽 1 Hz 历史帧存 sidecar | `--repo <path> --tasks mem6 --interval 25 --size 128` |
| `check_history_dataset.py` | 4 | 校验 history 样本对齐/pad/mask/边界 | `--sidecar-root <path> --num 100` + 可视化导出 |
| `build_cotrain_mix.py` | 4/5 | 生成 1200 集 cotrain 清单（mem6 600 + 通用 600） | `--out robodojo/data/history_sidecar/cotrain_mix_episodes.json` |
| `train_pi05_history.sh` | 5/9 | 训练入口（当前实现为 5C-LoRA；没有 `--mode full`） | `--config-name … --exp-name … --gpus 0,1,2,3 --fsdp-devices 1 --batch-size 48 --steps 20000` |
| `phase_mem_pi05.sh` | 1/6 | Memory 维度评测队列（server+容器 client） | 照抄 `scripts/phase3_memory.sh` 的队列/claims 模式，换 π0.5 启动命令与 uv 环境 |
| `offline_history_loss_ab.py` | 6/8 | 训练分布上的离线 ON/OFF 内容对照（6 臂：`on/off/shuf/zero/maskroll/croll`） | `--arms on,zero,croll --num-batches 100 --batch-size 16`；判定 `Δ(croll−on)`/`Δ(zero−on)` 是否显著为正 |
| `run_offline_history_loss_ab.sh` | 8 | 离线 A/B 的 srun 运行器（作业内单卡 bs16） | `JOBID=… NUM_BATCHES=100 ARMS=on,zero,croll bash …` |
| `build_keyframe_weights.py` | 8 | 关键帧权重表：`--profile` 视觉活动剖面（校准用）/ `--tier-a` 生成 v1（Step 3 追加 `--tier-b` 生成 v2） | `--out robodojo/data/history_sidecar/keyframe_weights_v1.json` |
| `build_keyframe_weights_tier_b.py` | 8 | Tier B 歧义度打分（ResNet-18 编码 + 同任务跨集 kNN + 动作 chunk 分歧度）→ v2 权重表 | `--out .../keyframe_weights_v2.json --diag logs/pi05/keyframe_tierb_diag.json` |
| `summarize_p8_progress.py` | 8 | Phase 8 汇总：v1/v2 统计 + Tier B 诊断 + 门禁 Δ 序列 | 无参数；读 `logs/pi05/gate_kf_*.json` |
| `p8_gate_watchdog.sbatch` | 8 | 门禁看护：自动对新 checkpoint 跑离线 on/zero/croll 对照 | `sbatch scripts/pi05/p8_gate_watchdog.sbatch` |
| `p8_gate_judge.py` | 8/9 | 门禁判据自动裁判：读 `gate_*.npz` 输出各 step 的 Δ/t 与通过判定（Δ≥+1e-4 且 t>+5） | `python3 scripts/pi05/p8_gate_judge.py --patterns "gate_v2_*.npz"` |
| `p8_ab6_watchdog.sbatch` | 8 | 6 臂诊断看护（on/off/shuf/zero/maskroll/croll，覆盖门禁 3 臂；自动派生 `gate_kf_<step>.json`） | `sbatch scripts/pi05/p8_ab6_watchdog.sbatch` |
| `test_pi05_history_v2.py` | 9 | CPU 单测（7 项）：两种 pad_mode 的时钟/一致性、无历史 pad、评测侧同口径、恒等核 | `python3 scripts/pi05/test_pi05_history_v2.py`（需限 2 线程） |
| `verify_pi05_history_v2.py` | 9 | GPU 分段验证：`--only pipeline|inject|contrast|contrast_off|contrast_roll|contrast_roll_sg` | 单卡 24GB 需分段跑 |
| `p9_train_v2_launch.sh` | 9 | v2 配方正式训练入口（**4 卡** fsdp=1 bs48 20k，供 `srun` 步骤包装） | `srun --jobid=<alloc> --overlap --ntasks=1 --cpus-per-task=16 bash scripts/pi05/p9_train_v2_launch.sh` |
| `p9_gate_watchdog.sh` | 9 | v2 门禁看护：新 ckpt → `on,zero,croll` 三臂 100×16 → `gate_v2_<step>.json/.npz` | `CUDA_VISIBLE_DEVICES=3 bash scripts/pi05/p9_gate_watchdog.sh` |
| `p9_gate_launch.sh` | 9 | 门禁看护的 srun 步骤包装（默认 GPU 3，`P9_GATE_GPUS` 可换） | `setsid nohup srun --jobid=<alloc> --overlap --cpus-per-task=8 bash scripts/pi05/p9_gate_launch.sh &` |

> Phase 9 起 `p8_gate_judge.py` 支持 `--patterns`（默认 `gate_kf_*.npz,ab6_kf_*.npz` 不变）；
> v2 门禁判据：`python3 scripts/pi05/p8_gate_judge.py --patterns "gate_v2_*.npz"`。

### 6.2 复用（不修改）

| 脚本 | 用途 |
| --- | --- |
| `scripts/phase3_memory.sh` 的队列/claims/看护模式 | `phase_mem_pi05.sh` 的骨架来源 |
| `scripts/phase3q_progress.py` / `phase3q_requeue.py` / `phase3q_status.sh` | 进度与看护（如需 π0.5 版则复制改路径） |
| `scripts/phase4_report.py` | 结果聚合（新增 π0.5 分支即可） |
| `scripts/run_smoke_stack_bowls.sh` | 冒烟参考实现（π0.5 版需另写 `run_smoke_pi05.sh`，因为它是 conda/opendm 专用） |
| 容器镜像 `robodojo:isaac6` | 仿真侧环境（Phase 0C 沿用 DM0.5 的镜像与 `podman save/load` 流程） |

### 6.3 改动的既有文件（两类复现方式）

> **边界必须说清楚**：`patch_pi05_history.py` 只管理 5 个 openpi 文件
> （`models/model.py`、`models/pi0.py`、`models/pi0_config.py`、`policies/policy.py`、`policies/aloha_policy.py`）。
> 表中其余文件和新增 `robodojo_history.py` **不在该脚本覆盖范围内**，必须用
> `patches/xpolicylab.patch` + `overlay/`（发布仓库）复现；RoboDojo 侧改动另见 `patches/robodojo.patch`。

| 文件 | 改动 | 是否影响 DM0.5 链路 |
| --- | --- | --- |
| `openpi/src/openpi/models/pi0.py` | `embed_prefix` 增加 history 分支 | 否（默认 `history_slots=0` 时行为不变） |
| `openpi/src/openpi/models/pi0_config.py` | 增加 `history_*` 字段（默认关闭） | 否 |
| `openpi/src/openpi/models/model.py` | `Observation` 增加 `history_images`/`history_mask`（可选字段） | 否 |
| `openpi/src/openpi/training/config.py` | 新增 history TrainConfig（附录 C） | 否（新增条目） |
| `openpi/src/openpi/training/weight_loaders.py` | 允许 history projector 参数缺失时随机初始化 | 否（新增分支） |
| `openpi/src/openpi/policies/robodojo_history.py`（新增） | `RoboDojoHistoryTransform`：sidecar → 20 槽 `history_images/history_mask` | 否（仅被 history TrainConfig 挂载） |
| `openpi/src/openpi/policies/aloha_policy.py` | `AlohaInputs` 透传 `episode_index/frame_index/task_index` | 否（键缺省时行为不变） |
| `openpi/src/openpi/training/config.py`（history 片段） | `history_sidecar_root/history_interval` 字段 + `pi05_aloha_sim_history_arx-x5_seed_0` | 否（新增条目/默认关闭） |
| `openpi/src/openpi/models/model.py`（history 片段） | history 轴名 `*b n c hh ww`（避免与 images 的 h/w 冲突） | 否 |
| `policy/Pi_05/model.py` | per-env 历史 deque（抄 OpenDM 的 `_history_by_env` 逻辑）+ reset 清理 | 否（仅 Pi_05） |
| `policy/Pi_05/deploy.yml` | 增加 `history_enabled/history_slots/history_fps/...` | 否（新增字段） |
| `openpi/src/openpi/models/model.py`（Phase 8 片段） | `Observation` 增加 `loss_weight`（可选字段）+ `from_dict` 透传 | 否（默认 None，等价旧行为） |
| `openpi/scripts/train.py`（Phase 8 片段） | `loss_fn` 支持归一化加权均值（无权重时退回 `jnp.mean`） | 否（`loss_weight=None` 时逐位一致） |
| `openpi/src/openpi/policies/robodojo_history.py`（Phase 8 片段） | 读取关键帧权重表并输出 `loss_weight`（默认路径 None = 关闭） | 否（仅 history TrainConfig 挂载） |
| `openpi/src/openpi/training/config.py`（Phase 8 片段） | `keyframe_weights_path` 字段 + `pi05_aloha_sim_history_kf_arx-x5_seed_0` | 否（新增条目/默认关闭） |
| `openpi/src/openpi/models/pi0_config.py`（Phase 9 片段） | `history_inject`（proj_zero/proj_identity/direct）+ `contrast_*` 字段与校验 | 否（默认 `proj_zero`/`none` = 旧行为） |
| `openpi/src/openpi/models/pi0.py`（Phase 9 片段） | `_identity_kernel_init`、`history_inject` 分支、`compute_loss` 拆 `_prepare_loss_inputs`+`_action_loss`、`compute_contrast_aux` | 否（默认关闭时逐位一致） |
| `openpi/src/openpi/models/model.py`（Phase 9 片段） | `load()` 在 proj_identity 时用恒等核补齐缺失的 `history_proj.kernel` | 否（默认 proj_zero 走原零初始化分支） |
| `openpi/src/openpi/policies/robodojo_history.py`（Phase 9 片段） | `pad_mode`（`zero` 默认 / `repeat_oldest`）+ 校验 | 否（默认 `zero` = 旧 mask 语义） |
| `openpi/src/openpi/training/config.py`（Phase 9 片段） | `history_pad_mode` 透传 + `pi05_aloha_sim_history_v2_arx-x5_seed_0` | 否（新增条目/默认 zero） |
| `openpi/scripts/train.py`（Phase 9 片段） | `loss_fn` 返回 `(loss+aux, info)`；`value_and_grad(has_aux=True)`；info 含 `loss_main/contrast_*` | 否（contrast 关闭时 aux=0） |
| `policy/Pi_05/model.py`（Phase 9 片段） | `history_pad_mode` 配置 + `_attach_history` 同口径回填（一致性断言） | 否（默认 `zero`） |
| `policy/Pi_05/deploy.yml`（Phase 9 片段） | `history_pad_mode: zero`（须与训练配置一致） | 否（新增字段） |

> `patch_pi05_history.py` 的 25 条规则必须保持幂等；打补丁后先跑
> `python3 scripts/pi05/patch_pi05_history.py --check`（应显示 25/25 已应用），
> 再跑 `python -c "import openpi"` + 一次 debug wiring check。

---

## 7. 命令速查

```bash
ROOT="${PI05_ROOT:-$(pwd)}"          # 项目根；公开复现时不要写死集群绝对路径
cd "$ROOT"

# ---------- Phase 0：权重 / 环境 / 冒烟 ----------
bash scripts/pi05/download_ckpt.sh                       # 0A 权重（ModelScope/HF）
bash scripts/pi05/fetch_demos.sh lerobot_v3.0            # 0B prepared joint 数据（约 120 GB）
srun --jobid=<JOBID> --overlap bash scripts/pi05/bootstrap_env.sh   # 0C uv 环境
srun --jobid=<JOBID> --overlap "$ROOT/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python" \
  "$ROOT/scripts/pi05/probe_pi05_runtime.py"             # 0D 标定
EVAL_ENV_TYPE=debug bash robodojo/XPolicyLab/policy/Pi_05/eval.sh \
  RoboDojo cover_blocks <ckpt> arx_x5 joint 0 0 0 uv <eval_env_conda_env>      # 0E

# ---------- Phase 1/6：Memory 维度评测（复用 DM0.5 的 server+容器架构） ----------
srun --jobid=<JOBID> --overlap bash scripts/pi05/phase_mem_pi05.sh \
  --eval-num 6 --seeds 0,1,2 --policy-gpus 0 --env-gpus 1        # 快速（108 episodes）
srun --jobid=<JOBID> --overlap bash scripts/pi05/phase_mem_pi05.sh \
  --eval-num native --seeds 0,1,2 --policy-gpus 0,2 --env-gpus 1,3   # 正式（900 episodes）

# ---------- Phase 2：离线记忆活性 ----------
srun --jobid=<JOBID> --overlap bash scripts/pi05/verify_pi05_history_liveness.sh

# ---------- Phase 3：免训练 A/B（手动切 deploy.yml；没有一键 ablation 脚本） ----------
# history_mode: collage | multi | text，配合 phase_mem_pi05.sh 做两任务小样本对比。
# 仓库内没有 ablation_pi05_memory.sh，请勿按旧版文档调用。

# ---------- Phase 4：数据管线 ----------
PY="$ROOT/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
"$PY" scripts/pi05/build_history_sidecar.py --repo "$ROOT/robodojo/data/RoboDojo_lerobot_v30_video" \
  --tasks mem6 --interval 25 --size 128 --workers 8
"$PY" scripts/pi05/build_cotrain_mix.py
"$PY" scripts/pi05/check_history_dataset.py \
  --sidecar-root "$ROOT/robodojo/data/history_sidecar" --num 100

# ---------- Phase 5：训练 ----------
bash scripts/pi05/train_pi05_history.sh \
  --config-name pi05_aloha_sim_history_v2_arx-x5_seed_0 \
  --exp-name RoboDojo-sim-arx_x5-history-v2-lora-seed0 \
  --gpus 0,1,2,3 --fsdp-devices 1 --batch-size 48 --steps 20000
# 当前入口只实现 5C-LoRA，没有 --mode full；全量微调需另行构造 TrainConfig。

# ---------- Phase 6：归档 ----------
# phase_mem_pi05.sh 会在 --log-dir 下生成 results.tsv；没有 report_pi05_memory.py。
```

**集群硬约束提醒**（与 DM0.5 文档 §8 一致）：策略 server 与仿真 client **必须分卡**；
每个 worker 峰值 ≈23 GiB；同一 podman storage root 上不要并发 `podman run`/`migrate`；
作业 48 h 墙钟上限；NFS 每天 04:00 重启（避开 03:55–04:10 写 `/share`）。

---

## 8. 结果记录模板

### 8.1 环境指纹（每次评测都要记）

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-10-03 至 2026-10-19（Phase 0 → Phase 9 归档） |
| 节点 / GPU | gpu01；基线 2× RTX 4090D，Phase 9 训练 4×，快速评测 3×（交互作业 `<JOBID>`） |
| NVIDIA 驱动 | 595.80（全集群一致） |
| RoboDojo commit | `726e9aaabfaa642203722eb126f5eaf0f37f3e1ad`（2026-09-19） |
| XPolicyLab commit | `bb9a0b5f5136a74503b679af830bfd0a3a837d5c`（2026-09-17；π0.5 补丁前基线） |
| openpi 版本/commit | 目录内嵌于 XPolicyLab，同一 commit `bb9a0b5f`（2026-09-17）；jaxlib 0.5.3 / Python 3.11.16 |
| 仿真镜像 | `robodojo:isaac6`（Isaac Sim 6.0.1 + Isaac Lab 2.3.2） |
| π0.5 权重 | 基线 `RoboDojo-sim-arx_x5-joint-{0,1,2}/59999`；v2 记忆权重 `RoboDojo-sim-arx_x5-history-v2-lora-seed0/19999`（来源 ModelScope `RoboDojo-Benchmark/RoboDojo`，2026-10-03 下载并校验） |
| 训练配置 | 基线 `pi05_base_aloha_full_sim_arx-x5_seed_0`；v2 `pi05_aloha_sim_history_v2_arx-x5_seed_0`（LoRA r16/α16，bs48，20k steps，seed0） |
| 评测口径 | 基线/快速对照 `--eval-num 6`，seeds 0/1/2；native 50 trials **尚未跑** |

> **2026-10-19（Phase 9 v2）评测指纹**：gpu01 / Slurm `<JOBID>`（3× RTX 4090D）；权重
> `RoboDojo-sim-arx_x5-history-v2-lora-seed0/19999`；训练配置 `pi05_aloha_sim_history_v2_arx-x5_seed_0`
> （LoRA r16/α16、20k 步、bs48/4 卡）；部署 `history_pad_mode: repeat_oldest`；
> 口径 6 任务 × 3 seeds × 6 集；日志 `logs/pi05/phase9_eval_v2_20261019_1358/`。

### 8.2 总分对照

| 口径 | π0.5 基线 | π0.5 免训练最佳 | π0.5 训练后 | DM0.5 本集群复现 | DM0.5 榜单 |
| --- | --- | --- | --- | --- | --- |
| Memory score | **4.86** | 未计（仅离线筛查） | **6.20**（5 任务 7.44） | 44.03 | 47.74 |
| Memory SR | **3.70%** | 未计 | **5.56%**（15 有效 run） | 43.52 | 47.44 |
| 其他四维（Δ vs 基线） | — | — | 尚未回归 | — | — |

> π0.5 训练后 = Phase 9 v2（6 任务 × 3 seeds × 6 集）。`swap_blocks` 3/3 为仿真侧 `PhysX CUDA error 700`
> （server 0 次推理），按 0 分记账；只算 5 个有效任务为 **7.44** / SR 5.56%。

### 8.3 逐任务对照

| task | π0.5 基线 | π0.5 训练后 | DM0.5 复现 | DM0.5 榜单 | Δ |
| --- | --- | --- | --- | --- | --- |
| `cover_blocks` | 22.50 | 13.33 | 100.00 | 100.00 | −9.17 |
| `match_and_pick_from_conveyor` | 5.56 | 22.22 | 77.78 | 70.67 | +16.67 |
| `press_by_number` | 0.00 | 0.00 | 83.33 | 95.33 | 0.00 |
| `swap_T` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `imitate_sorting_sequence` | 1.11 | 1.67 | 3.06 | 1.80 | +0.56 |
| `swap_blocks` | 0.00 | n/a（rc=3×3） | 0.00 | 18.67 | — |
| **均值** | **4.86** | **6.20** | **44.03** | **47.74** | **+1.34** |

> 6 任务口径（`swap_blocks` 记 0）= 6.20；5 个有效任务口径 = 7.44。

---

## 9. 风险与已知坑

| # | 风险 | 影响 | 应对 |
| --- | --- | --- | --- |
| 1 | ~~权重不可得~~ **已核实可下**（HF `ckpt/RoboDojo/Pi_05`，2026-10-01）；残余风险仅是**下载网络** | 拿不到则全项目无法启动 | 用 `modelscope` 源；HF 源需覆盖 `HF_REPO_URL`（`download_ckpt.sh` 走 git clone，不认 `HF_ENDPOINT`） |
| 2 | ~~数据不可得~~ **已核实可下**（HF `data/RoboDojo_lerobot_v30_video` 等 7 种导出，2026-10-01）；残余风险是**格式选错** | 训练路线（A/C）全部失效或维度错 | **首选 `lerobot_v3.0`（joint-only，state/action=14 维）**，与已发布 `RoboDojo-sim-arx_x5-joint-*` ckpt 及 `RoboDojo_sim_arx-x5_v30` repack 一致；~~`lerobot_v3.0_ee`~~ 为 16 维 EE 空间，仅在改训 EE 模型时才用。2026-10-03 已用 `meta/info.json` 离线核验两版维度 |
| 3 | **前缀分布偏移**：π0.5 动作头是 flow matching + adaRMS，对前缀敏感 | memory 涨、其他维度崩 | 每次评测都带非 memory 维度门禁（L5）；先在 LoRA 模式验证 |
| 4 | **新参数初始化**：pi05_base 权重没有 history projector，加载时 missing keys | 加载失败或静默错位 | 在 weight loader 里显式允许缺失并随机初始化；加载后核对参数树 |
| 5 | **JAX/FSDP 新形状**：history 输入改变 JIT 形状与显存 | 编译失败或 OOM | 先把 `history_slots` 做成静态形状（20 恒定）；显存不足降 batch + 梯度累积 |
| 6 | **历史对齐错误**（跨 episode、用了未来帧、pad 未屏蔽） | 分数虚高，上线即失效 | `check_history_dataset.py` 100 样本断言 + 3 条 episode 可视化人工复核 |
| 7 | **多环境/reset 串味** | trial 之间互相泄漏记忆，分数不可信 | 照抄 OpenDM 的 `(evaluation_id, env_idx)` key + `reset` 清空（`policy/OpenDM/model.py:509`） |
| 8 | **`press_by_number` 是自运动记忆**（要数自己按了几次） | 纯 1 Hz 视觉历史可能不够，该项上不去 | 备选：提高该任务的采样节拍、加 1 维动作/状态历史；先按 DM0.5 同款视觉历史执行 |
| 9 | **评测方差**：`swap_blocks` 已有 0 vs 18.67 的环境漂移先例 | 误判提升/退化 | 至少 3 seeds × 6 集出结论；正式分用 native 50 trials |
| 10 | **集群约束**：48 h 墙钟、NFS 04:00 重启、策略/仿真必须分卡 | 长作业被杀/结果丢失 | 按 seed/任务分片；关键产物先写节点本地盘再 rsync；避开 03:55–04:10 |
| 11 | **训练算力**：4090D 24 GB，官方 recipe 是 batch 256 / FSDP 2 | 训不动或周期拉长 | 用 LoRA 起步；全量时降 batch + 累积 + 提高到 4 卡 FSDP；Phase 0D 先标定单步时延 |
| 12 | **免训练路线可能全面掉点**：拼贴/多图是 OOD 输入 | 基线被污染 | A/B 只在两个任务上小样本试，不动主线；发现掉点立刻停 |
| 13 | **补丁污染主干**：改 openpi 影响未来升级/其他策略 | 维护困难 | 所有改动走幂等补丁脚本 + 默认关闭的开关；`--revert` 可还原 |

---

## 10. 已决策 / 已核实问题

1. **权重来源**：HF/ModelScope `RoboDojo-Benchmark/RoboDojo` 的 `ckpt/RoboDojo/Pi_05`；已下载并校验
   3 个 seed（85/85 文件，`RoboDojo-sim-arx_x5-joint-{0,1,2}/59999`）。发布仓库只记录来源，不分发权重。
2. **数据来源**：官方 prepared `data/RoboDojo_lerobot_v30_video`（LeRobot v3.0，3500 集、1,856,102 帧、
   25 FPS、14 维 joint），已下载到 `robodojo/data/RoboDojo_lerobot_v30_video`。上游仓库标注
   Apache-2.0，但再分发数据前仍应以其最新许可与榜单规则为准；本项目只提交元数据和小型清单。
3. **动作块节拍**：0D 实测 `action_horizon=50`，deploy 会执行完整 chunk；历史采样采用每 25 个底层动作 1 帧
   （1 Hz@25 FPS），与训练 sidecar 和评测 deque 一致。
4. **免训练 A/B**：B1/B2/B3 已做过筛查，只作为机理验证，不作为最终交付；没有一键 ablation 脚本。
5. **训练预算**：Phase 5/8/9 已消耗多轮 20k-step 训练；v2 在 4×4090D 上耗时 24h50m。后续 native 900-episode
   评测与全量回归仍需单独申请预算。
6. **发布边界**：代码与文档可发布；数据、权重、checkpoint、日志、镜像不进入 Git。
7. **仍未闭合的问题**：`swap_blocks` 的 PhysX CUDA 700 根因未定位；`press_by_number` 为 0 的能力缺口未解；
   正式 native 50-trial 分数与五维回归尚未跑。

---

## 附录 A：DM0.5 记忆实现代码地图（移植参考）

| 功能 | 文件:行 |
| --- | --- |
| 契约参数（slots/fps/interval/tokens） | `robodojo/XPolicyLab/policy/OpenDM/deploy.yml` |
| 历史采集与可见性过滤 | `robodojo/XPolicyLab/policy/OpenDM/model.py:379`、`:403` |
| 每 env 隔离与 reset | `robodojo/XPolicyLab/policy/OpenDM/model.py:220`、`:509` |
| 16 token/帧 常量 | `robodojo/XPolicyLab/policy/OpenDM/opendm/opendm/constants/robot.py:5` |
| SigLIP 历史特征投影 | `robodojo/XPolicyLab/policy/OpenDM/opendm/opendm/model/dm05/dm05_arch.py:61` |
| 前缀/KV 注入 | `robodojo/XPolicyLab/policy/OpenDM/opendm/opendm/model/dm05/dm05_arch.py:996` |
| 离线活性验证脚本 | `scripts/verify_memory_liveness.py` / `.sh`（改造对象） |

## 附录 B：π0.5 侧改造点（Phase 2 落地清单 ｜ 状态：6 条全部 ✅ 已实现；完整总账见附录 E）

1. ✅ **`Observation` 扩展**（`openpi/src/openpi/models/model.py`）：增加可选 `history_images`（B,N,3,H,W）
   与 `history_mask`（B,N），默认 None；`from_dict` 读取时 uint8→[-1,1]；`preprocess_observation` 透传（必要时 resize）。
2. ✅ **`embed_prefix` 历史分支**（`openpi/src/openpi/models/pi0.py`）：
   - 位置：当前相机图像 token 之后、tokenized prompt 之前；
   - 每帧：SigLIP（复用 `PaliGemma.img`）→ 16×16 patch → 均值池化 4×4=16 token → `history_proj` 线性投影；
   - mask：pad 槽 input_mask=0，ar_mask 与图像 token 同规则（互相全注意）；
   - 形状：`history_slots` 固定为 20（静态形状，利于 JIT）。
   - 实测：`verify_pi05_history_liveness.py` 2026-10-04 全判据通过（效应 0.7375 / 内容 0.0866 / pad 0.0 / 延迟 +1.9%）。
3. ✅ **`Pi0Config` 字段**（`pi0_config.py`）：`history_slots=0`（默认关闭，保持向后兼容）、
   `history_tokens_per_slot=16`、`history_pool_size=4`。
4. ✅ **权重加载**（改在 `models/model.py::BaseModelConfig.load()`，比 `weight_loaders.py` 更通用）：
   缺失叶子就地初始化（bias→0、其余 `lecun_normal`），官方 checkpoint 缺 history projector 时自动随机
   初始化且不 OOM；`--check/--revert` 幂等可还原。
5. **推理侧 deque**（`policy/Pi_05/model.py`）：新增 `_history_by_env` / `_sequence_by_env`，
   在 `update_obs` 里 append，在 `get_action_batch` 里取 20 槽；`reset` 清空；key 用 `(evaluation_id, env_idx)`。
6. **deploy.yml**：新增 `history_enabled`、`history_image_key: cam_high`、`history_slots: 20`、
   `history_fps: 1.0`、`history_action_interval: 25`、`history_tokens_per_slot: 16`。

## 附录 C：训练配置草案（openpi TrainConfig）

```python
TrainConfig(
    name="pi05_aloha_sim_history_arx-x5_seed_0",
    model=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",   # C 路线；A 路线改回 "gemma_2b"
        history_slots=20,
        history_tokens_per_slot=16,
        history_pool_size=4,
    ),
    data=LeRobotAlohaDataConfig(
        repo_id="RoboDojo_sim_arx-x5_v30_history",   # Phase 4 生成
        assets=AssetsConfig(assets_dir=str(_ROBODOJO_ASSETS_DIR), asset_id="arx_x5_sim"),
        repack_transforms=_transforms.Group(inputs=[
            _transforms.RepackTransform({
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            }),
        ]),
        extra_transforms=_transforms.Group(inputs=[
            RoboDojoHistoryTransform(slots=20, fps=1.0, interval=25),  # 新增
        ]),
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "<RoboDojo π0.5 微调权重路径>/params"   # Phase 0A 落位
    ),
    seed=0,
    batch_size=128,          # 4090D 24GB 起点；不足改 64 + 梯度累积
    fsdp_devices=2,
    num_train_steps=20000,   # C 路线；A 路线 60000
)
```

> 注：`extra_transforms` 字段名以 openpi 实际 API 为准（落地时对照 `LeRobotAlohaDataConfig` 的
> transforms 组装顺序调整）；本项目约定**历史帧必须在当前帧之前、且同一 episode 内**。

---

## 附：时间线汇总（乐观估计，含等待下载）

| 阶段 | 内容 | 工作日 |
| --- | --- | --- |
| Phase 0 | 权重+数据就位、环境、基线标定 | 1~3 |
| Phase 1 | Memory 基线复现（108 eps） | 1 |
| Phase 2 | 记忆通路离线活性 + 补丁 ✅ 2026-10-04 | 2~3（实际 ~1 小时） |
| Phase 3 | 免训练 A/B（go/no-go）：公共前置 ✅、B1/B2/B3 全部完成且未达标 → 免训练关闭（2026-10-04） | 3~5 |
| Phase 4 | 历史增强数据管线 ✅ 2026-10-04（600 集 sidecar 0 缺失 + 训练 transform 验收 PASS） | 3~5（实际 ~2.5 小时） |
| Phase 5 | 训练（LoRA 先行） | 5~10 |
| Phase 6 | 正式评测（900 eps）+ 归档 | 2~3 |
| Phase 7（可选） | 五维全量回归 | 3~5 |
| **合计** | | **约 4~6 周**（受数据/权重可得性影响最大） |

---

## 附录 D：与「通用记忆升级路线」的差异对照（若项目升级，先读这里）

> 背景：有人会问"能不能抛开 RoboDojo，单独把 π0.5 改造成通用记忆模型，再喂给各种 benchmark 评测"。
> 可以，但那是一条独立的大路线。两者的关键差异如下。

| 维度 | 当前方案（RoboDojo 专用） | 通用记忆升级路线 |
| --- | --- | --- |
| **目标判据** | Memory 维度 ≥40（对齐 DM0.5 47.74），单一数字可对照 | 跨 N 个 benchmark 的记忆增益 + zero-shot 迁移；无单一对照数字，需设计统计口径 |
| **数据** | 一份 RoboDojo sim demos（120 GB）+ memory 子集/通用混采 | N 个 benchmark 数据（RoboDojo / RMBench / RoboTwin / 真机）+ 配比设计 + 各 benchmark sidecar |
| **记忆契约** | 固定 20 槽 @1 Hz、`cam_high`、16 token/帧（抄 DM0.5，已验证） | 参数化：槽数/频率/相机/时间尺度随 benchmark 变化，可能需学习型压缩（参考 RISE/SimpleMemVLA/Mem-0） |
| **训练形态** | 单段第二微调（LoRA 先行），1~2 GPU 天/轮 | 多 benchmark 混训 + 每 benchmark 一次适配微调；数周 GPU；存在负迁移/灾难性遗忘风险 |
| **工程面** | 1 个适配器（Pi_05）+ 1 套 sidecar 管线 | 每个 benchmark 的输入契约与评测接入 + 多 checkpoint 回归矩阵 + 自动化 |
| **最大不确定性** | 以工程确定性为主（数据/权重/训练是否收敛） | **记忆能力能否跨域迁移是开放研究问题**；且没有现成对照物（可对照 Mem-0 / RISE / SimpleMemVLA） |
| **与当前方案的关系** | — | 当前方案是其**前置里程碑**：机制、数据管线、活性验证方法、评测协议都在当前方案里产出 |

**执行建议**：先按本文档在单 benchmark 上打通（拿到机制 + 方法论 + 可复用的代码），再泛化到
多 benchmark；**不要一上来就做多 benchmark 混训**——否则失败时无法区分是机制问题、数据配比问题还是评测问题。

---

## 附录 E：π0.5 记忆改动总账（截至 2026-10-15）

> 范围：为给 π0.5 增加"历史/记忆"能力而做的**全部代码、数据、训练与评测侧改动**。
> 改动文件清单可用 `git -C robodojo/XPolicyLab status --porcelain` 核对（11 个既有文件 M + 1 个新文件 ??
> = `policies/robodojo_history.py`）；openpi 侧既有文件的改动由 `scripts/pi05/patch_pi05_history.py`
> 幂等管理（`--check` 检查 / `--revert` 还原，备份为 `*.orig.pi05history`）。
> 所有改动均为**旁路式、默认关闭**（`history_slots=0` 或 `history_enabled: false` 时与官方 π0.5 行为一致）。

### E.1 模型本体（openpi/src/openpi）

| 文件 | 改动内容 | 状态 / 验证 |
| --- | --- | --- |
| `models/pi0_config.py` | `Pi0Config` 新增 `history_slots`（默认 0=关闭）、`history_tokens_per_slot=16`、`history_pool_size=4` | ✅ 默认关闭时行为不变 |
| `models/pi0.py` | ① 新增 `history_proj` 线性投影（**零初始化**：step 0 历史 token 输出 0，训练中逐步"打开"）；② `embed_prefix` 历史分支：SigLIP → 16×16 patch → 4×4 均值池化 → 每帧 16 token，插在**当前图像 token 之后、prompt 之前**；pad 槽 `input_mask=0`，ar_mask 与图像 token 同规则 | ✅ 活性验证 PASS（2026-10-04：注意力效应 0.7375 / 内容效应 0.0866 / pad 0.0 / 推理延迟 +1.9%） |
| `models/model.py` | ① `Observation` 新增 `history_images`(b,n,c,128,128 ∈[−1,1])、`history_mask`(b,n)、`loss_weight`(b,)；② `from_dict` uint8→[−1,1]；③ `preprocess_observation` 透传（必要时 resize）；④ `BaseModelConfig.load()` 对缺失的 history 参数**零初始化补齐**（白名单仅含 "history" 键；其余缺失仍报错） | ✅ |
| `policies/policy.py` | `Policy.infer` 把 `history_images/history_mask` 从 obs 透传给模型输入 | ✅ |
| `policies/aloha_policy.py` | `AlohaInputs` 支持 `cam_history` / `history_0_rgb` / `cam_history_*`（Phase 3 免训练 A/B 的 B1 拼贴 / B2 多图方案） | ✅ |
| `policies/robodojo_history.py`（**新文件**） | `RoboDojoHistoryTransform`：20 槽 × 128×128、1 Hz（每 25 帧 1 帧）、只取严格早于当前帧、最旧在左、空槽 mask=False；Phase 8 扩展：可选 `keyframe_weights_path` → 额外输出 `loss_weight` | ✅ 数据校验 100 样本断言通过 |

### E.2 训练侧（关键帧加权）

| 文件 | 改动内容 |
| --- | --- |
| `training/config.py` | `LeRobotAlohaDataConfig` 新增 `keyframe_weights_path`；新增两个 TrainConfig：`pi05_aloha_sim_history_arx-x5_seed_0`（纯历史通路）与 `pi05_aloha_sim_history_kf_arx-x5_seed_0`（+关键帧加权；表路径默认 v1，可用 `ROBODOJO_KEYFRAME_WEIGHTS` 环境变量覆盖为 v2） |
| `scripts/train.py` | loss 改为**归一化加权均值 Σwᵢℓᵢ/Σwᵢ**；`loss_weight=None` 时退回 `jnp.mean`（与旧实现逐位一致） |
| `training/data_loader.py`、`training/weight_loaders.py` | 配套改动（数据管线透传 `loss_weight`、历史参数加载） |

### E.3 评测 / 部署侧（RoboDojo XPolicyLab 的 π0.5 服务）

| 文件 | 改动内容 |
| --- | --- |
| `policy/Pi_05/model.py` | 推理侧历史缓冲：`_configure_history`（读 deploy.yml 的 history_*，并与训练 config **逐项断言一致**，不一致直接报错）；`_history_by_env` deque（key=`(evaluation_id, env_idx)`，每 `history_action_interval=25` 步记 1 帧、取最近 20 槽 + mask）；`reset` 清空；保留 collage/multi/text 三种消融模式 |
| `policy/Pi_05/deploy.yml` | 新增 `history_enabled: true`、`history_mode: slots`、`history_image_key: cam_high`、`history_slots: 20`、`history_slot_size: 128`、`history_fps: 1.0`、`history_action_interval: 25`；消融用 `history_collage_*` / `history_multi_*` / `history_captioner_*` |

### E.4 工具链（`scripts/pi05/`）

| 类别 | 脚本 |
| --- | --- |
| 数据管线 | `build_history_sidecar.py`（抽 1 Hz 历史帧存 sidecar）、`build_cotrain_mix.py`（1200 集混料清单）、`check_history_dataset.py`（对齐/pad/mask 校验）、`test_pi05_history_pipeline.py` |
| 关键帧权重表 | `build_keyframe_weights.py`（v1：视觉活动剖面 + 任务规则）、`build_keyframe_weights_tier_b.py`（v2：同观测不同动作的歧义度打分 + 跨集 kNN） |
| 训练 | `train_pi05_history.sh`（LoRA/全量入口）、`watch_pi05_history_train.sh`、`verify_phase5_setup.py`、`patch_pi05_history.py`（改动管理器） |
| 离线诊断 | `verify_pi05_history_liveness.py/.sh`（活性五组判据）、`offline_history_loss_ab.py`（6 臂 ON/OFF 分解）、`run_offline_history_loss_ab.sh`、`p8_gate_judge.py`（门禁判据裁判）、`summarize_p8_progress.py`（汇总）、`p8_gate_watchdog.sbatch`、`p8_ab6_watchdog.sbatch`（自动看护） |
| 消融 / 评测 | `phase6_ab_history.sh`（训练后历史开/关 A/B）、`phase_mem_pi05.sh`（6 任务 × 3 seeds 评测队列）、`run_pi05_eval.sh`（server + 容器 client 执行器） |

### E.5 数据与实验产物

- **侧车数据**：`robodojo/data/history_sidecar/`——mem6 共 600 集、1 Hz、128×128，632 MB，13,407 槽；
  + `cotrain_mix_episodes.json`（混料清单）。
- **关键帧权重表**：`keyframe_weights_v1.json`（w>1 占 63.3%、均值 2.82、max 4）→
  `keyframe_weights_v2.json`（w>1 占 87.1%、均值 2.18、max 8）。
- **训练产物**（每个 ckpt ≈9.1 GB）：
  - `RoboDojo-sim-arx_x5-history-lora-seed0`（Phase 5：4 卡 bs64、20k 步；保留 2000→19999 每 2000 步）；
  - `RoboDojo-sim-arx_x5-history-kf-lora-seed0`（Phase 8：3 卡 bs48、20k 步，0–1000 步用 v1 表、其后用 v2 表；保留 5000/10000/15000/19999）。
  - `RoboDojo-sim-arx_x5-history-v2-lora-seed0`（Phase 9：4 卡 bs48、20k 步；保留 5000/10000/15000/19999）。

### E.6 效果时间线（结论）

1. **2026-10-04 离线活性**：历史通路接通（注意力效应 0.74 / 内容 0.087 / pad 0 / 延迟 +1.9%）；
2. **Phase 5 训练 + 评测**：记忆分无提升；离线 ON≈OFF；
3. **2026-10-09 诊断**：判定"内容通道未打开"（训练侧问题）；
4. **Phase 8 关键帧加权 20k**：**门禁 20/20 全未通过**——`Δ(zero−on)`/`Δ(croll−on)` ≈ 0（内容无感），
   而 `Δ(off−on)`≈+1.2e-4、`Δ(maskroll−on)`≈+2.1e-4（只使用了"历史前缀 + mask 时钟结构"这一捷径）。
   → "关键帧加权"路线被证伪；后续可选：①改 mask 设计断掉时钟捷径；②上 DM0.5 式显式 memory 读写架构。
5. **2026-10-19 Phase 9 v2 评测**（6 任务 × 3 seeds × 6 集）：**6.20**（`swap_blocks` 记 0）/ **7.44**（5 任务），
  基线 4.86；唯一实质提升是 `match_and_pick`（5.56→22.22），`cover_blocks` 22.5→13.3，`press_by_number` 仍 0。
  内容通道已打开（离线门禁）但任务级收益有限；最终榜单口径（native 50 trials）仍未跑。

### E.7 已知问题 / 注意事项

- `patch_pi05_history.py --check` 当前为 **25/25 已应用**（2026-10-19 复核），覆盖
  `models/model.py`、`models/pi0.py`、`models/pi0_config.py`、`policies/policy.py`、`policies/aloha_policy.py`。
  **它不是全量改动管理器**：训练配置、`scripts/train.py`、data loader、weight loader、评测适配器和
  `robodojo_history.py` 不在其中，发布时必须使用 `patches/xpolicylab.patch` 与 `overlay/`。
- `deploy.yml` 当前指向 Phase 9 v2：`pi05_aloha_sim_history_v2_arx-x5_seed_0`、
  `checkpoint_num: 19999`、`history_pad_mode: repeat_oldest`。回退基线时需同步关闭 history 并切回
  `pi05_base_aloha_full_sim_arx-x5_seed_0/59999`。
- `swap_blocks` 的 PhysX CUDA 700 尚未根因闭环：跳过 articulation relocation 仅一次通过、第二次仍崩，
  不能把该变量当作已验证修复；后续实验需同时记录场景重建阶段、GPU/驱动状态和可复现率。
- 评测容器镜像只在 gpu01 的 podman storage 有（38.4 GB）；gpu04 为空，需要时可用
  `images/robodojo-isaac6.tar`（38.35 GB）`podman load`（≈10 分钟）。
