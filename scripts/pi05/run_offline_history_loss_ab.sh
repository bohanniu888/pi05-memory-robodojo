#!/usr/bin/env bash
# 离线历史 ON/OFF 对照运行器（在 GPU 作业内跑；单卡 bs16，显存约 10~12GB）。
#
# 用法：
#   JOBID=76429 NUM_BATCHES=400 setsid nohup bash scripts/pi05/run_offline_history_loss_ab.sh \
#     > logs/pi05/offline_ab_$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &
# 环境变量：JOBID（默认 76429）、GPU（默认 0）、NUM_BATCHES（默认 200）、BATCH_SIZE（默认 16）、NUM_WORKERS（默认 8）
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
JOBID="${JOBID:-76429}"
GPU="${GPU:-0}"
NUM_BATCHES="${NUM_BATCHES:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ARMS="${ARMS:-on,off,shuf,zero}"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_JSON="${ROOT}/logs/pi05/offline_history_loss_ab_${TS}.json"

echo "[runner] job=${JOBID} gpu=${GPU} batches=${NUM_BATCHES} bs=${BATCH_SIZE} arms=${ARMS} out=${OUT_JSON}"

srun --jobid="${JOBID}" --overlap bash -c "
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${GPU}
export HF_LEROBOT_HOME=${ROOT}/robodojo/data/hf_lerobot_home
export ROBODOJO_HISTORY_SIDECAR_DIR=${ROOT}/robodojo/data/history_sidecar
LOCAL=/tmp/openpi-cache-\$(hostname); mkdir -p \"\$LOCAL/hf/datasets\" \"\$LOCAL/jax\"
export HF_DATASETS_CACHE=\"\$LOCAL/hf/datasets\"
export JAX_COMPILATION_CACHE_DIR=\"\$LOCAL/jax\"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
PY=${ROOT}/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
cd ${ROOT}/robodojo/XPolicyLab/policy/Pi_05/openpi
\"\$PY\" ${ROOT}/scripts/pi05/offline_history_loss_ab.py \\
    --num-batches ${NUM_BATCHES} --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS} \\
    --arms ${ARMS} \\
    --out-json ${OUT_JSON}
"
