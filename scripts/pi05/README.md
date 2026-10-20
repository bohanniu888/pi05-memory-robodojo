# scripts/pi05 — π0.5 Memory 项目运行手册

本目录是 π0.5 × RoboDojo 记忆改造的工具链。源码基线固定在：

- RoboDojo `726e9aabfaa642203722eb126f5eaf0f37f3e1ad`
- XPolicyLab（RoboDojo 子模块）`bb9a0b5f5136a74503b679af830bfd0a3a837d5c`

所有命令默认在项目根执行。脚本支持 `PI05_ROOT` 覆盖根目录；未设置时，
Bash 脚本按自身位置解析，Python 脚本按 `Path(__file__).resolve().parents[2]` 解析。
公开复现时不要把集群绝对路径写回脚本。

## 目录约定

```text
<PROJECT_ROOT>/
├── robodojo/                         # 上游仓库 + XPolicyLab 子模块
├── robodojo/data/RoboDojo_lerobot_v30_video/
├── robodojo/data/history_sidecar/    # npy + 小型 JSON
├── scripts/pi05/
└── logs/pi05/                        # 不入 Git
```

数据与权重不随仓库分发。权重来源：HF/ModelScope `RoboDojo-Benchmark/RoboDojo`
的 `ckpt/RoboDojo/Pi_05`；训练数据使用 prepared `data/RoboDojo_lerobot_v30_video`
（joint-only，14 维），不要下载 16 维 `lerobot_v3.0_ee`。

## 环境

```bash
ROOT="${PI05_ROOT:-$PWD}"
cd "$ROOT"

# 安装 openpi/uv 环境；GPU 侧在 Slurm 分配内执行
srun --jobid=<JOBID> --overlap bash scripts/pi05/bootstrap_env.sh
```

登录节点无 GPU。训练、评测和 GPU 探针统一通过 `srun --jobid=<JOBID> --overlap` 进入分配。

## Phase 0：权重、数据与探针

```bash
bash scripts/pi05/download_ckpt.sh                 # 默认 ModelScope；落点 Pi_05/checkpoints
bash scripts/pi05/fetch_demos.sh lerobot_v3.0      # joint，约 120 GB
bash scripts/pi05/fetch_demos.sh demo               # 可选：小样本链路验证
srun --jobid=<JOBID> --overlap \
  "$ROOT/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python" \
  "$ROOT/scripts/pi05/probe_pi05_runtime.py"
```

`fetch_demos.sh` 的接口是位置参数 `<format> <source>`，没有任务过滤参数。
任务子集由 sidecar 的 `--tasks mem6` 和训练清单控制。

## Phase 2/4：历史通路与 sidecar

```bash
python3 scripts/pi05/patch_pi05_history.py --check

PY="$ROOT/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
"$PY" scripts/pi05/build_history_sidecar.py \
  --repo "$ROOT/robodojo/data/RoboDojo_lerobot_v30_video" \
  --tasks mem6 --interval 25 --size 128 --workers 8
"$PY" scripts/pi05/build_cotrain_mix.py
"$PY" scripts/pi05/check_history_dataset.py \
  --sidecar-root "$ROOT/robodojo/data/history_sidecar" --num 100
```

`patch_pi05_history.py` 只管理 5 个 openpi 文件，当前检查结果应为 **25/25 已应用**。
它不管理 `training/config.py`、`scripts/train.py`、data/weight loader、评测适配器和
`robodojo_history.py`；完整复现请用发布仓库的 `patches/` 与 `overlay/`。

## Phase 5/9：训练

当前入口只实现 5C-LoRA，没有 `--mode` 参数。正式 v2 训练：

```bash
srun --jobid=<JOBID> --overlap --ntasks=1 --cpus-per-task=16 \
  bash scripts/pi05/p9_train_v2_launch.sh
```

等价的显式命令：

```bash
bash scripts/pi05/train_pi05_history.sh \
  --config-name pi05_aloha_sim_history_v2_arx-x5_seed_0 \
  --exp-name RoboDojo-sim-arx_x5-history-v2-lora-seed0 \
  --gpus 0,1,2,3 --fsdp-devices 1 \
  --batch-size 48 --steps 20000 --save-interval 1000 --overwrite
```

关键环境变量：

- `HF_LEROBOT_HOME`：默认 `robodojo/data/hf_lerobot_home`
- `ROBODOJO_HISTORY_SIDECAR_DIR`：默认 `robodojo/data/history_sidecar`
- `OPENPI_LOCAL_CACHE_ROOT`：默认 `/tmp/openpi-cache-$(hostname)`

## Phase 6：Memory 评测

```bash
# 快速口径：6 任务 × 3 seeds × 6 集 = 108 episodes
srun --jobid=<JOBID> --overlap bash scripts/pi05/phase_mem_pi05.sh \
  --eval-num 6 --seeds 0,1,2 --policy-gpus 0 --env-gpus 1

# 正式口径：6 任务 × 3 seeds × 50 trials = 900 episodes（尚未归档结果）
srun --jobid=<JOBID> --overlap bash scripts/pi05/phase_mem_pi05.sh \
  --eval-num native --seeds 0,1,2 --policy-gpus 0,2 --env-gpus 1,3
```

评测目录会生成 `results.tsv`、每 run 的 `_result.json` 和 server/client 日志。
策略 server 与 Isaac 仿真 client 必须分卡；同一 podman storage root 不要并发执行
`podman run`/`migrate`。

## 诊断与门禁

| 脚本 | 用途 |
| --- | --- |
| `verify_pi05_history_liveness.py/.sh` | 历史注入活性与内容敏感性 |
| `offline_history_loss_ab.py` / `run_offline_history_loss_ab.sh` | ON/OFF/zero/croll 等 6 臂离线对照 |
| `p8_gate_judge.py` | 读取 `gate_*.npz`，判定 Δ/t 门槛 |
| `p9_gate_watchdog.sh` / `p9_gate_launch.sh` | 新 checkpoint 的 v2 门禁看护 |
| `test_pi05_history_v2.py` | zero/repeat_oldest、投影初始化与一致性 CPU 单测 |
| `verify_pi05_history_v2.py` | proj_zero/proj_identity/direct 与对比损失分段验证 |
| `diag_swap_blocks_*.sh` | swap_blocks PhysX 700 的对照实验（根因未闭环） |

CPU 单测使用 openpi 的虚拟环境，不是系统 `python3`：

```bash
PY="$ROOT/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" scripts/pi05/test_pi05_history_v2.py
```

## 当前结论（2026-10-20）

- v2 训练完成，最终 checkpoint `19999`；当前没有训练作业。
- 快速评测：**6.20**（6 任务，swap_blocks 记 0）/ **7.44**（5 任务），基线 4.86。
- 唯一实质提升：`match_and_pick_from_conveyor` 5.56 → 22.22。
- `press_by_number` 仍为 0；`cover_blocks` 22.50 → 13.33（6 集口径翻转噪声大）。
- `swap_blocks` 3/3 在仿真侧 `PhysX CUDA error 700`，根因未定位。
- native 50-trial 正式分数与五维回归尚未跑。

## 发布卫生

不得提交：`robodojo/data/`、`history_sidecar/*.npy`、checkpoints、`models/`、
`images/`、`logs/`、`cache*/`、`.venv/`、podman storage、备份文件和 `__pycache__/`。
发布前运行：

```bash
git status --short
git grep -nE '/share/|/(home|Users)/|hf_[A-Za-z0-9]{10,}|wandb|WANDB|password|token'
```
