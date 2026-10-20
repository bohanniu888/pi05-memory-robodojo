#!/usr/bin/env bash
# Phase 9：v2 配方（pad_mode=repeat_oldest + history_inject=proj_identity + contrast roll）
# 的 5C-LoRA 20k 正式训练入口。供 `srun ... bash scripts/pi05/p9_train_v2_launch.sh` 调用，
# 避免多行 payload 被压平；日志由调用方重定向。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# 登录节点 RLIMIT_NPROC 限制下也需要这些；容器内同样安全。
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
# [p9] 同上：抬高进程/线程上限（硬上限允许时生效）。
ulimit -u 8192 2>/dev/null || true

exec bash scripts/pi05/train_pi05_history.sh \
  --config-name pi05_aloha_sim_history_v2_arx-x5_seed_0 \
  --exp-name RoboDojo-sim-arx_x5-history-v2-lora-seed0 \
  --gpus 0,1,2,3 \
  --fsdp-devices 1 \
  --batch-size 48 \
  --steps 20000 \
  --save-interval 1000 \
  --overwrite
