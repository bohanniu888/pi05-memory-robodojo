#!/usr/bin/env bash
# Phase 8 Step 3 诊断看护：对 kf 训练各 checkpoint 跑 6 臂离线分解
#   arms = on,off,shuf,zero,maskroll,croll
# 目的：区分"内容没学会" vs "mask 结构主导"；对 ckpt1000(v1权重表) 与 ckpt2000+(v2) 做横向对比。
# 输出：logs/pi05/ab6_kf_<step>.json / .runlog；已存在则跳过。
# 变更记录（2026-10-10 18:3x）：新建（三卡批次）。
#SBATCH --job-name=p8_ab6
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pi05/p8_ab6_slurm_%j.log

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CKPT_ROOT="${ROOT}/robodojo/XPolicyLab/policy/Pi_05/checkpoints/RoboDojo-sim-arx_x5-history-kf-lora-seed0"
PY="${ROOT}/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
OUT_DIR="${ROOT}/logs/pi05"
mkdir -p "${OUT_DIR}"

export HF_LEROBOT_HOME="${ROOT}/robodojo/data/hf_lerobot_home"
export ROBODOJO_HISTORY_SIDECAR_DIR="${ROOT}/robodojo/data/history_sidecar"
LOCAL=/tmp/openpi-cache-ab6-$(hostname); mkdir -p "$LOCAL/hf/datasets" "$LOCAL/jax"
export HF_DATASETS_CACHE="$LOCAL/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="$LOCAL/jax"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd "${ROOT}/robodojo/XPolicyLab/policy/Pi_05/openpi" || exit 1
echo "[ab6] host=$(hostname) ckpt_root=${CKPT_ROOT} $(date -Is)"
while true; do
  for ck in $(ls "${CKPT_ROOT}" 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
    out="${OUT_DIR}/ab6_kf_${ck}.json"
    fail="${OUT_DIR}/ab6_kf_${ck}.fail"
    [[ -f "${out}" ]] && continue
    [[ -f "${fail}" ]] && continue
    if [[ ! -d "${CKPT_ROOT}/${ck}/params" ]]; then continue; fi
    tries_file="${OUT_DIR}/.ab6_kf_${ck}.tries"
    tries=$(cat "${tries_file}" 2>/dev/null || echo 0)
    if (( tries >= 3 )); then
      echo "[ab6] ckpt=${ck} failed ${tries} times, giving up" | tee "${fail}"
      continue
    fi
    echo "[ab6] run ckpt=${ck} (try $((tries+1))/3) $(date -Is)"
    if "${PY}" "${ROOT}/scripts/pi05/offline_history_loss_ab.py" \
        --ckpt-dir "${CKPT_ROOT}/${ck}" \
        --arms on,off,shuf,zero,maskroll,croll --num-batches 100 --batch-size 16 --num-workers 4 \
        --out-json "${out}" >"${OUT_DIR}/ab6_kf_${ck}.runlog" 2>&1; then
      # 由 6 臂结果派生门禁口径 JSON（仅 on/zero/croll + 对应 Δ），保证既有 gate 工具链不变
      python3 - "$out" "${OUT_DIR}/gate_kf_${ck}.json" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
def keep(k):
    if k == "n": return True
    if k in ("on", "zero", "croll"): return True
    if k.startswith("\u0394(") and ("zero" in k or "croll" in k) and "off" not in k and "shuf" not in k: return True
    return False
g = {k: v for k, v in d.items() if not isinstance(v, dict)}
g["derived_from"] = "ab6 6-arm"
for sec in ("overall", "usable_ge_min", "usable_lt_min"):
    if sec in d: g[sec] = {k: v for k, v in d[sec].items() if keep(k)}
json.dump(g, open(dst, "w"), ensure_ascii=False, indent=1)
PYEOF
      echo "[ab6] ckpt=${ck} OK $(date -Is)"
    else
      echo $((tries+1)) > "${tries_file}"
      echo "[ab6] ckpt=${ck} FAIL (try $((tries+1))/3) $(date -Is)"
      tail -5 "${OUT_DIR}/ab6_kf_${ck}.runlog"
    fi
  done
  sleep 300
done
