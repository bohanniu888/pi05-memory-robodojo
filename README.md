# π0.5 × RoboDojo Memory 维度实验

面向 RoboDojo 基准的 π0.5 记忆条件实验。本仓库包含复现实验所需的项目文档、脚本、补丁、配置和精简元数据，
不包含 RoboDojo/XPolicyLab 源码树、模型权重、数据集、checkpoint、日志或容器镜像。

## 状态

Phase 9 v2 LoRA 训练已完成（20k steps，最终 checkpoint `19999`）。当前没有训练作业在运行。

快速评测（6 任务 × 3 seeds × 6 集，并非官方 50-trial 协议）：

| 指标 | π0.5 基线 | π0.5 v2 记忆 | DM0.5 复现 | DM0.5 榜单 |
| --- | ---: | ---: | ---: | ---: |
| Memory 得分（score） | **4.86** | **6.20**（5 个有效任务：7.44） | 44.03 | 47.74 |
| Memory 成功率（SR） | **3.70%** | **5.56%** | 43.52 | 47.44 |

目标是 ≥40，因此项目尚未完成。任务层面唯一实质提升是 `match_and_pick_from_conveyor`（5.56 → 22.22）。
`press_by_number` 仍为 0，`cover_blocks` 从 22.50 降至 13.33，`swap_blocks` 在三个 seed 的仿真中
均因 `PhysX CUDA error 700` 失败。官方 native 50-trial 分数和五维回归尚未运行。

## 改动内容

- 在当前相机 token 与语言提示之间注入 20 个槽位、1 Hz 的 `cam_high` 历史通路。
- 训练侧使用 sidecar 变换；推理侧使用按环境隔离、在 reset 时清空的双端队列（deque）。
- 从官方 RoboDojo π0.5 joint checkpoint 出发进行 LoRA 微调。
- v2 改动：用 `repeat_oldest` 填充消除单调递增的 mask 时钟，历史投影采用恒等初始化，并加入内容对比损失。
- 1,200 集 cotrain 混料：Memory 6 个任务 × 100 集 + 600 集分层通用数据。

完整实验记录见 [docs/IMPLEMENTATION_PLAN.zh-CN.md](docs/IMPLEMENTATION_PLAN.zh-CN.md)，证据摘要见
[docs/RESULTS.md](docs/RESULTS.md)，可执行步骤见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。
文档审计和已修正的过时表述见 [docs/AUDIT_2026-10-20.md](docs/AUDIT_2026-10-20.md)。

## 仓库结构

```text
docs/                 实施规划与结果
scripts/pi05/         数据、训练、评测与诊断
scripts/setup/        上游检出与补丁应用
patches/              两个上游仓库的已跟踪文件 diff
overlay/              无法用“修改已跟踪文件”的 diff 表示的新文件
configs/              v2 记忆策略的 deploy.yml 快照
sidecar_metadata/     index/meta/cotrain/keyframe 元数据（不含 .npy 负载）
third_party/          上游许可证文本
```

仅含源码的检出刻意保持较小（约 1.3 MB，不含 `.git`）。`git status` 和
`bash scripts/setup/check_release.sh` 都属于发布门禁。

## 上游基线

| 仓库 | 提交 | 许可证 |
| --- | --- | --- |
| [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) | `726e9aabfaa642203722eb126f5eaf0f37f3e1ad` | MIT |
| [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | `bb9a0b5f5136a74503b679af830bfd0a3a837d5c` | Apache-2.0 |

`XPolicyLab` 是 RoboDojo 的子模块，但记忆相关改动是在其工作树中完成的。检出上述确切提交后，需要同时应用
RoboDojo 和 XPolicyLab 补丁。

## 快速开始

```bash
git clone <THIS_REPOSITORY> pi05-memory-robodojo
cd pi05-memory-robodojo

bash scripts/setup/bootstrap_upstream.sh
bash scripts/setup/apply_overlay.sh

# 数据和权重需单独下载；参见 docs/REPRODUCTION.md。
bash scripts/pi05/download_ckpt.sh
bash scripts/pi05/fetch_demos.sh lerobot_v3.0
```

项目脚本会根据自身位置解析根目录。仅当上游检出位于本仓库之外时，才需要设置 `PI05_ROOT`。GPU 任务必须在
原集群已分配的 Slurm 作业内运行，或在具备兼容 Isaac Sim/podman 环境的等价机器上运行。

## 数据规模

| 数据集 | 集数 | 帧数 |
| --- | ---: | ---: |
| 官方预处理 LeRobot v3.0 | 3,500 | 1,856,102 |
| v2 训练混料 | 1,200 | 644,703 |
| Memory 6 任务子集 | 600 | 328,375 |
| 通用 cotrain 子集 | 600 | 316,328 |

sidecar 包含 13,407 个 1 Hz 历史采样（约 632 MB 的 `.npy`）。本仓库只收录其小型 JSON 元数据。

## 已知问题

- `swap_blocks` 根因尚未定位。十环境对照在原始场景中产生 `PhysX CUDA error 700`；`skip_reloc` 一次通过、
  一次失败。不能把 articulation relocation 视为已验证的修复。
- v2 门禁最初使用了错误的符号。按 `Δ = bad_history − on` 定义，内容通路有效时应满足
  `Δ ≥ +1e-4` 且 `t > +5`；v2 checkpoint 满足这一修正后的判据。
- 6 集口径的任务分数方差很大。`cover_blocks` 只要单次成功翻转就会变化 ±16.67。
- 本仓库不宣称任何 native 50-trial 结果。

## 许可与署名

项目专属文档和脚本采用 Apache-2.0 许可。RoboDojo 仍为 MIT，XPolicyLab 仍为 Apache-2.0；被打补丁或覆盖的文件
保留其上游许可证。另见 [NOTICE.md](NOTICE.md) 和 `third_party/`。数据和权重不随仓库分发；请从 RoboDojo
官方仓库获取，并遵守其当前许可证和基准规则。

本仓库是研究代码，不是可用于生产的机器人控制器。
