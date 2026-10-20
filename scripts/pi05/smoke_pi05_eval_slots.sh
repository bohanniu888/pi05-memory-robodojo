#!/usr/bin/env bash
# Phase 6 前置：π0.5+memory 评测链路冒烟（slots 历史模式）。
#   CPU 组装语义（随时可跑）:   srun --jobid=<jid> --overlap bash scripts/pi05/smoke_pi05_eval_slots.sh
#   真模型活性（需空 GPU）:     PI05_SMOKE_REAL=1 POLICY_GPU=0 srun --jobid=<jid> --overlap bash scripts/pi05/smoke_pi05_eval_slots.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PY="$ROOT/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG="$ROOT/logs/pi05/smoke_eval_slots_${RUN_ID}.log"
EXTRA_ARGS=()

if [[ "${PI05_SMOKE_REAL:-0}" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-.85}"
    EXTRA_ARGS+=(--real-policy)
else
    export CUDA_VISIBLE_DEVICES=""
    export JAX_PLATFORMS=cpu
fi
export PYTHONWARNINGS=ignore::UserWarning

echo "[smoke.sh] host=$(hostname) start=$(date -Is) real=${PI05_SMOKE_REAL:-0} gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "[smoke.sh] log=$LOG"
"$PY" -u "$ROOT/scripts/pi05/smoke_pi05_eval_slots.py" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[smoke.sh] exit=$rc log=$LOG"
exit "$rc"
