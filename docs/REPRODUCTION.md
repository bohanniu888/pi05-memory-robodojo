# 复现指南

本指南针对源码级复现。复现仓库只提供源码补丁、脚本和元数据；以下外部产物不随仓库分发，必须另外取得或生成：

| 产物 | 获取方式 | 是否自动下载 |
| --- | --- | --- |
| 上游源码 | `bootstrap_upstream.sh` 克隆 RoboDojo 和 XPolicyLab | 是 |
| 官方基础 π0.5 checkpoint | `download_ckpt.sh` | 是 |
| 官方 LeRobot v3.0 数据 | `fetch_demos.sh lerobot_v3.0` | 是 |
| 历史 sidecar | 第 6 步从数据生成 | 否 |
| v2 LoRA checkpoint | 第 7 步训练，或从外部位置提供 | 否 |
| 仿真镜像与 Assets | 第 8 步单独准备 | 否 |

## 1. 前置条件

- 安装 Git、**Git LFS**、Bash、Python 3.11 和 `uv` 的 Linux 环境。
- 用于评测的 NVIDIA GPU 环境，并具备相同或兼容的 Isaac Sim/Isaac Lab 软件栈。归档环境使用 NVIDIA 驱动
  `595.80`；更换驱动后需重新验证仿真器兼容性。
- `nvidia-smi`；评测还需要 podman。只有原集群编排才需要 Slurm。
- 预处理后的 LeRobot 数据集约需 120 GB；另需为 sidecar、基础/v2 checkpoint 和约 17 GB 的仿真镜像预留空间。

Git LFS 是下载数据和基础权重的硬依赖。两个下载脚本都会直接调用 `git lfs`，执行前先检查并初始化：

```bash
git lfs version
git lfs install
```

如果命令不存在，Ubuntu 可安装 `git-lfs`，conda 环境可用
`conda install -c conda-forge git-lfs`。缺少 Git LFS 时，脚本会报 `git-lfs not found`，或只留下 LFS 指针文件。

设置根目录，并让所有命令相对于该目录执行：

```bash
export PI05_ROOT="$(pwd)"
```

## 2. 检出确切的上游基线

```bash
bash scripts/setup/bootstrap_upstream.sh
```

该脚本将 RoboDojo 克隆到 `robodojo/`，检出
`726e9aabfaa642203722eb126f5eaf0f37f3e1ad`，初始化 XPolicyLab 子模块，并将其检出到
`bb9a0b5f5136a74503b679af830bfd0a3a837d5c`。

如需完整的仿真器检出，请手动初始化其余子模块：

```bash
git -C robodojo submodule update --init third_party/IsaacLab third_party/curobo
```

## 3. 应用项目改动

```bash
bash scripts/setup/apply_overlay.sh
```

该脚本应用 `patches/robodojo.patch`、`patches/xpolicylab.patch`，并复制新文件覆盖层。对于已应用的补丁，
脚本可重复执行。验证五个 openpi 补丁目标：

```bash
python3 scripts/pi05/patch_pi05_history.py --check
```

预期输出以 25 行标记为 `已应用` 的记录结尾，且没有 `锚点未命中`。

## 4. 安装策略环境

```bash
srun --jobid=<JOBID> --overlap bash scripts/pi05/bootstrap_env.sh
```

在非 Slurm 机器上，确认其 CUDA 假设后，直接运行内部的 `policy/Pi_05/install.sh`。

## 5. 下载基础权重和数据

### 5.1 官方基础 π0.5 权重

```bash
bash scripts/pi05/download_ckpt.sh
bash scripts/pi05/download_ckpt.sh pi05 huggingface  # 可选：改走 HF 镜像
```

该脚本下载的是 RoboDojo 官方**基础** π0.5 checkpoint（如
`RoboDojo-sim-arx_x5-joint-{0,1,2}/59999`），不会下载本项目的 v2 LoRA checkpoint。默认使用 ModelScope，
目标位置为 `robodojo/XPolicyLab/policy/Pi_05/checkpoints/`，该路径通常是指向
`robodojo/.cache/robodojo_ckpt_modelscope_repo/` 的符号链接。

安装 Git LFS 后可用以下命令确认基础权重：

```bash
ls -d robodojo/XPolicyLab/policy/Pi_05/checkpoints/RoboDojo-sim-arx_x5-joint-*/59999
```

### 5.2 官方训练数据

```bash
bash scripts/pi05/fetch_demos.sh lerobot_v3.0
```

训练数据是预处理后的 joint-only LeRobot v3.0 导出（14 维 state/action）。不要用 `lerobot_v3.0_ee`
替代已发布的 joint checkpoint。训练前验证 `meta/info.json`：

```text
total_episodes = 3500
total_frames   = 1856102
fps            = 25
state/action   = 14
```

默认位置：

- 基础权重：`robodojo/XPolicyLab/policy/Pi_05/checkpoints/`
- 数据：`robodojo/data/RoboDojo_lerobot_v30_video/`
- sidecar：`robodojo/data/history_sidecar/`

`download_ckpt.sh` 和 `fetch_demos.sh` 都不会下载历史 sidecar；sidecar 必须在第 6 步从 LeRobot 数据生成。

## 6. 构建历史 sidecar 与 cotrain 清单

```bash
PY="$PI05_ROOT/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"

"$PY" scripts/pi05/build_history_sidecar.py \
  --repo "$PI05_ROOT/robodojo/data/RoboDojo_lerobot_v30_video" \
  --tasks mem6 --interval 25 --size 128 --workers 8

"$PY" scripts/pi05/build_cotrain_mix.py
"$PY" scripts/pi05/check_history_dataset.py \
  --sidecar-root "$PI05_ROOT/robodojo/data/history_sidecar" --num 100
```

预期：600 集、13,407 个历史采样、没有缺失采样，且 `cotrain_mix_episodes.json` 中有 1,200 集。

## 7. 训练 v2 LoRA 策略

```bash
srun --jobid=<JOBID> --overlap --ntasks=1 --cpus-per-task=16 \
  bash scripts/pi05/p9_train_v2_launch.sh
```

归档配置为 4×RTX 4090D、FSDP=1、batch 48（每卡 12）、20,000 steps、seed 0。已记录的运行耗时
24 小时 50 分钟。checkpoint 目录由 `--exp-name` 选择；如果最终 step 不同，请调整
`configs/deploy.history.yml` 中的 `checkpoint_num`。

归档结果对应的 v2 checkpoint 必须位于：

```text
robodojo/XPolicyLab/policy/Pi_05/checkpoints/
  RoboDojo-sim-arx_x5-history-v2-lora-seed0/
    19999/
      assets/
      params/
```

其中 `19999/params` 和 `19999/assets` 是评测加载所需的最低内容；`train_state/` 只在续训时需要。
`download_ckpt.sh` 不提供这个目录，复现仓库也不分发 v2 权重。若没有保留的训练产物，必须重新运行本节的
20k 训练；若从外部下载 v2 checkpoint，请保留上述完整目录层级，不要把 `params/` 直接放进 `checkpoints/`。
`configs/deploy.history.yml` 默认 `checkpoint_num: 19999`，如使用其他保留 step 需同步修改。

## 8. 准备仿真镜像与 Assets

评测脚本默认使用 `IMAGE=robodojo:isaac6`，并且固定使用独立的 podman storage：

```text
/tmp/podman-build-$USER/{root,run}
```

因此镜像必须存在于这个 storage 中，而不是只存在于 root 用户的默认 podman storage。归档镜像基于
Isaac Sim 6.0.1 + Isaac Lab 2.3.2，大小约 17 GB；镜像和 NVIDIA Isaac Sim 基础镜像都不随本仓库分发。

先检查镜像是否已存在：

```bash
export PODMAN_ROOT="/tmp/podman-build-$USER/root"
export PODMAN_RUNROOT="/tmp/podman-build-$USER/run"

podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" \
  --storage-driver overlay image exists robodojo:isaac6
```

如果不存在，可在应用覆盖层后、于仓库根目录构建。构建前必须完成第 2 节的
`third_party/IsaacLab` 和 `third_party/curobo` 子模块初始化，否则镜像内的仿真器依赖不完整：

```bash
mkdir -p "$PODMAN_ROOT" "$PODMAN_RUNROOT"
podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" \
  --storage-driver overlay build \
  --format docker --network host \
  --build-arg ISAACSIM_IMAGE=nvcr.m.daocloud.io/nvidia/isaac-sim:6.0.1 \
  -f robodojo/docker/Dockerfile.isaac6 \
  -t robodojo:isaac6 robodojo
```

如果已有预构建镜像归档，也可以载入同一 storage：

```bash
podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" \
  --storage-driver overlay load -i /path/to/robodojo-isaac6.tar
```

此外，评测容器会挂载 `robodojo/Assets/`。该目录需按 RoboDojo 官方安装说明准备，不能用镜像构建替代；
缺少 Assets 时仿真会在加载场景前失败。

## 9. 评测

先切换到与 v2 checkpoint 对应的部署配置，并确认 `checkpoint_num`、`train_config_name` 和
`history_pad_mode` 与 checkpoint 一致：

```bash
cp configs/deploy.history.yml robodojo/XPolicyLab/policy/Pi_05/deploy.yml
```

```bash
# 与归档结果兼容的快速协议；必须显式指定 v2 checkpoint。
srun --jobid=<JOBID> --overlap bash scripts/pi05/phase_mem_pi05.sh \
  --ckpt RoboDojo-sim-arx_x5-history-v2-lora-seed0 \
  --eval-num 6 --seeds 0,1,2 --policy-gpus 0 --env-gpus 1

# 官方规模协议，尚未归档结果。
srun --jobid=<JOBID> --overlap bash scripts/pi05/phase_mem_pi05.sh \
  --ckpt RoboDojo-sim-arx_x5-history-v2-lora-seed0 \
  --eval-num native --seeds 0,1,2 --policy-gpus 0,2 --env-gpus 1,3
```

`phase_mem_pi05.sh` 的 `--ckpt` 默认值是基础权重 `RoboDojo-sim-arx_x5-joint-2`；漏传该参数会评测基线而不是
v2。server 会断言历史模式、槽位数、pad 模式和训练配置一致，不一致时直接拒绝启动。基线运行时应禁用历史，
并使用官方基础配置/checkpoint。

## 10. 发布检查

```bash
bash scripts/setup/check_release.sh
```

该检查会拒绝大型产物、上游检出、集群绝对路径和常见密钥模式。
