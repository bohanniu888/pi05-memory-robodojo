#!/usr/bin/env bash
# Phase 2：在 policy GPU 上跑 π0.5 历史通路活性探针。
#   srun --jobid=<jid> --overlap bash scripts/pi05/verify_pi05_history_liveness.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
GPU="${POLICY_GPU:-0}"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG="$ROOT/logs/pi05/verify_history_liveness_${RUN_ID}.log"

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=.85
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONWARNINGS=ignore::UserWarning

echo "[H.sh] host=$(hostname) start=$(date -Is) gpu=$GPU"
echo "[H.sh] log=$LOG"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

"$ROOT/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python" -u \
    "$ROOT/scripts/pi05/verify_pi05_history_liveness.py" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[H.sh] exit=$rc  log=$LOG"
exit "$rc"
