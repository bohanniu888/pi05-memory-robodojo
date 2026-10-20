#!/usr/bin/env bash
# Phase 9 门禁看护的 srun 步骤包装（在 <JOBID> 分配内与训练并行，默认用 GPU 3）。
# 用法：setsid nohup srun --jobid=<JOBID> --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 \
#         bash scripts/pi05/p9_gate_launch.sh > logs/pi05/p9_gate_v2_<ts>.log 2>&1 &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
# [p9] 抬高进程/线程上限（硬上限允许时生效）。
ulimit -u 8192 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="${P9_GATE_GPUS:-3}"

exec bash scripts/pi05/p9_gate_watchdog.sh
