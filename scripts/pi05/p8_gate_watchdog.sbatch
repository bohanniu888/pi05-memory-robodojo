#!/usr/bin/env bash
# Phase 8 Step 2 门禁看护：对 kf 训练最新 checkpoint 自动跑离线 ON/OFF 内容对照（on/zero/croll）。
# 判定：Δ(croll−on) / Δ(zero−on) 显著为负 → 加权生效（内容通道打开）。
# 变更记录（2026-10-10 17:4x）：XLA 显存比例 0.5→0.9（修复 ckpt1000 OOM）；连续失败 3 次写 .fail 标记并跳过。
#SBATCH --job-name=p8_gate
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pi05/p8_gate_slurm_%j.log

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CKPT_ROOT="${ROOT}/robodojo/XPolicyLab/policy/Pi_05/checkpoints/RoboDojo-sim-arx_x5-history-kf-lora-seed0"
PY="${ROOT}/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
OUT_DIR="${ROOT}/logs/pi05"
mkdir -p "${OUT_DIR}"

export HF_LEROBOT_HOME="${ROOT}/robodojo/data/hf_lerobot_home"
export ROBODOJO_HISTORY_SIDECAR_DIR="${ROOT}/robodojo/data/history_sidecar"
LOCAL=/tmp/openpi-cache-gate-$(hostname); mkdir -p "$LOCAL/hf/datasets" "$LOCAL/jax"
export HF_DATASETS_CACHE="$LOCAL/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="$LOCAL/jax"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd "${ROOT}/robodojo/XPolicyLab/policy/Pi_05/openpi" || exit 1
echo "[gate] host=$(hostname) ckpt_root=${CKPT_ROOT} $(date -Is)"
while true; do
  for ck in $(ls "${CKPT_ROOT}" 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
    out="${OUT_DIR}/gate_kf_${ck}.json"
    fail="${OUT_DIR}/gate_kf_${ck}.fail"
    [[ -f "${out}" ]] && continue
    [[ -f "${fail}" ]] && continue
    if [[ ! -d "${CKPT_ROOT}/${ck}/params" ]]; then continue; fi
    tries_file="${OUT_DIR}/.gate_kf_${ck}.tries"
    tries=$(cat "${tries_file}" 2>/dev/null || echo 0)
    if (( tries >= 3 )); then
      echo "[gate] ckpt=${ck} failed ${tries} times, giving up" | tee "${fail}"
      continue
    fi
    echo "[gate] run ckpt=${ck} (try $((tries+1))/3) $(date -Is)"
    if "${PY}" "${ROOT}/scripts/pi05/offline_history_loss_ab.py" \
        --ckpt-dir "${CKPT_ROOT}/${ck}" \
        --arms on,zero,croll --num-batches 100 --batch-size 16 --num-workers 4 \
        --out-json "${out}" >"${OUT_DIR}/gate_kf_${ck}.runlog" 2>&1; then
      echo "[gate] ckpt=${ck} OK $(date -Is)"
    else
      echo $((tries+1)) > "${tries_file}"
      echo "[gate] ckpt=${ck} FAIL (try $((tries+1))/3) $(date -Is)"
      tail -5 "${OUT_DIR}/gate_kf_${ck}.runlog"
    fi
  done
  sleep 600
done
