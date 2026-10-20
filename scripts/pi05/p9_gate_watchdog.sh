#!/usr/bin/env bash
# Phase 9 门禁看护（v2 配方）：对 v2 训练 checkpoint 自动跑离线 ON/ZERO/CROLL 内容对照。
# 判据：Δ(zero−on) 或 Δ(croll−on) ≤ -1e-4 且 t < -5 → 内容通道打开（门禁翻转）。
#
# 运行方式（在 <JOBID> 分配内、与训练并行的独立 srun 步骤，占用 GPU 3）：
#   CUDA_VISIBLE_DEVICES=3 bash scripts/pi05/p9_gate_watchdog.sh
# 由 p9_gate_launch.sh 包装调用。输出：logs/pi05/gate_v2_<step>.json/.npz。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CKPT_ROOT="${ROOT}/robodojo/XPolicyLab/policy/Pi_05/checkpoints/RoboDojo-sim-arx_x5-history-v2-lora-seed0"
CONFIG_NAME="pi05_aloha_sim_history_v2_arx-x5_seed_0"
PY="${ROOT}/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
OUT_DIR="${ROOT}/logs/pi05"
SLEEP_SEC="${P9_GATE_SLEEP_SEC:-300}"
# 目标 GPU 需有这么多空闲显存才开跑（MiB）；训练占满卡时自动等待（训练结束即补跑）。
MIN_FREE_MIB="${P9_GATE_MIN_FREE_MIB:-9000}"
mkdir -p "${OUT_DIR}"

_free_mib() {
  local dev="${1:-0}" v=""
  v=$(nvidia-smi -i "${dev}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  if [[ -z "${v}" ]]; then
    v=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  fi
  echo "${v:-0}"
}

export HF_LEROBOT_HOME="${ROOT}/robodojo/data/hf_lerobot_home"
export ROBODOJO_HISTORY_SIDECAR_DIR="${ROOT}/robodojo/data/history_sidecar"
LOCAL=/tmp/openpi-cache-gate-$(hostname); mkdir -p "$LOCAL/hf/datasets" "$LOCAL/jax"
export HF_DATASETS_CACHE="$LOCAL/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="$LOCAL/jax"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

cd "${ROOT}/robodojo/XPolicyLab/policy/Pi_05/openpi" || exit 1
echo "[p9-gate] start host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} min_free=${MIN_FREE_MIB}MiB ckpt_root=${CKPT_ROOT} $(date -Is)"
while true; do
  free_now=$(_free_mib "${CUDA_VISIBLE_DEVICES:-0}")
  if (( free_now < MIN_FREE_MIB )); then
    sleep "${SLEEP_SEC}"
    continue
  fi
  for ck in $(ls "${CKPT_ROOT}" 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
    ckdir="${CKPT_ROOT}/${ck}"
    out="${OUT_DIR}/gate_v2_${ck}.json"
    fail="${OUT_DIR}/gate_v2_${ck}.fail"
    [[ -f "${out}" ]] && continue
    [[ -f "${fail}" ]] && continue
    # 完整性：orbax 完成写入才有这两个标记（避免读到半成品）。
    [[ -f "${ckdir}/_CHECKPOINT_METADATA" && -f "${ckdir}/params/_METADATA" ]] || continue
    tries_file="${OUT_DIR}/.gate_v2_${ck}.tries"
    tries=$(cat "${tries_file}" 2>/dev/null || echo 0)
    if (( tries >= 3 )); then
      echo "[p9-gate] ckpt=${ck} failed ${tries} times, giving up" | tee "${fail}"
      continue
    fi
    echo "[p9-gate] run ckpt=${ck} (try $((tries+1))/3) $(date -Is)"
    if "${PY}" "${ROOT}/scripts/pi05/offline_history_loss_ab.py" \
        --config-name "${CONFIG_NAME}" \
        --ckpt-dir "${ckdir}" \
        --arms on,zero,croll --num-batches 100 --batch-size 16 --num-workers 4 \
        --out-json "${out}" >"${OUT_DIR}/gate_v2_${ck}.runlog" 2>&1; then
      echo "[p9-gate] ckpt=${ck} OK $(date -Is)"
    else
      echo $((tries+1)) > "${tries_file}"
      echo "[p9-gate] ckpt=${ck} FAIL (try $((tries+1))/3) $(date -Is)"
      tail -5 "${OUT_DIR}/gate_v2_${ck}.runlog"
    fi
  done
  sleep "${SLEEP_SEC}"
done
