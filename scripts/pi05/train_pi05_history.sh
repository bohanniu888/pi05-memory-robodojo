#!/usr/bin/env bash
# Phase 5C 训练入口：π0.5 + 历史增强（冻结主干 + LoRA，history projector 全参可训）。
#
# 用法（在 srun 作业内执行，见文末示例）：
#   bash scripts/pi05/train_pi05_history.sh --gpus 0,1                      # 正式 20k steps / bs128
#   bash scripts/pi05/train_pi05_history.sh --gpus 0 --batch-size 16 --steps 10 --overwrite   # 冒烟
#   bash scripts/pi05/train_pi05_history.sh --gpus 0,1 --resume             # 断点续训
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POLICY_DIR="${ROOT}/robodojo/XPolicyLab/policy/Pi_05"
OPENPI_DIR="${POLICY_DIR}/openpi"
PY="${OPENPI_DIR}/.venv/bin/python"

config_name="pi05_aloha_sim_history_arx-x5_seed_0"
exp_name="RoboDojo-sim-arx_x5-history-lora-seed0"
gpus="0,1"
batch_size=128
steps=""
fsdp_devices=""
overwrite=0
resume=0
use_wandb=0
save_interval=""
keep_period=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-name) config_name="$2"; shift 2 ;;
    --exp-name) exp_name="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --fsdp-devices) fsdp_devices="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --resume) resume=1; shift ;;
    --wandb) use_wandb=1; shift ;;
    --save-interval) save_interval="$2"; shift 2 ;;
    --keep-period) keep_period="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

gpu_count=$(awk -F',' '{print NF}' <<<"${gpus}")
if [[ -z "${fsdp_devices}" ]]; then
  fsdp_devices=$(( gpu_count < 2 ? 1 : 2 ))
fi
if (( batch_size % gpu_count != 0 )); then
  echo "[train] batch_size(${batch_size}) 必须能被 GPU 数(${gpu_count}) 整除" >&2
  exit 2
fi
if (( overwrite == 1 && resume == 1 )); then
  echo "[train] --overwrite 与 --resume 不可同时使用" >&2
  exit 2
fi

ckpt_dir="${POLICY_DIR}/checkpoints/${exp_name}"
mkdir -p "${ckpt_dir}"

# ---- 环境 ----
export CUDA_VISIBLE_DEVICES="${gpus}"
export HF_LEROBOT_HOME="${ROOT}/robodojo/data/hf_lerobot_home"
export ROBODOJO_HISTORY_SIDECAR_DIR="${ROOT}/robodojo/data/history_sidecar"
# 共享存储只读数据 + 本地缓存（避免 NFS 锁竞争）。
LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/openpi-cache-$(hostname)}"
mkdir -p "${LOCAL_CACHE_ROOT}/hf/datasets" "${LOCAL_CACHE_ROOT}/jax"
export HF_DATASETS_CACHE="${LOCAL_CACHE_ROOT}/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="${LOCAL_CACHE_ROOT}/jax"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
# [p9] 强制无缓冲 stdout：Step 日志（pbar.write）实时落盘，后台 nohup / 被 kill 时不丢。
export PYTHONUNBUFFERED=1
# [p9] 容器内 RLIMIT_NPROC 软上限可能很低（gpu01 实测 256，硬上限 ~3e6）：多卡训练
# + DataLoader workers 会触发 "RuntimeError: can't start new thread"。硬上限允许时抬高软上限。
ulimit -u 8192 2>/dev/null || true

if (( use_wandb == 1 )); then
  wandb_arg="--wandb-enabled"
else
  wandb_arg="--no-wandb-enabled"
fi

echo "[train] config=${config_name}"
echo "[train] exp=${exp_name} ckpt=${ckpt_dir}"
echo "[train] gpus=${gpus} fsdp=${fsdp_devices} batch=${batch_size} steps=${steps:-<config default>}"
echo "[train] host=$(hostname) sidecar=${ROBODOJO_HISTORY_SIDECAR_DIR}"
echo "[train] log=$(date +%Y-%m-%d_%H-%M-%S)"

cd "${OPENPI_DIR}"
"${PY}" scripts/train.py "${config_name}" \
  --exp-name="${exp_name}" \
  --fsdp-devices="${fsdp_devices}" \
  --checkpoint-dir-override="${ckpt_dir}" \
  --batch-size="${batch_size}" \
  ${steps:+--num-train-steps="${steps}"} \
  --seed=0 \
  "${wandb_arg}" \
  ${save_interval:+--save-interval="${save_interval}"} \
  ${keep_period:+--keep-period="${keep_period}"} \
  $( (( overwrite == 1 )) && echo "--overwrite" ) \
  $( (( resume == 1 )) && echo "--resume" )

# 示例（在 gpu01 的 srun 作业内后台跑正式训练）：
#   setsid nohup bash scripts/pi05/train_pi05_history.sh --gpus 0,1 --overwrite \
#     > logs/pi05/p5_train_$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &
