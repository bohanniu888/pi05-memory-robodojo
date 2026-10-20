#!/usr/bin/env bash
# Phase 8 Step 4 接力·第二棒：门禁通过 → 切 deploy.yml 到 kf 配置 → 4 卡（2 worker）跑 Memory 维度评测。
# 提交：sbatch --dependency=afterok:<pre_jobid> scripts/pi05/p8_step4_relay.sbatch
# 可覆盖：P8_STEP4_EVAL_NUM（默认 6，与基线 22.0/11.7 同口径）、P8_STEP4_LOG_DIR、P8_STEP4_CKPT、P8_STEP4_CONFIG
#SBATCH --job-name=p8_step4
#SBATCH --partition=gpu2node
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pi05/p8_step4_slurm_%j.log

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT" || exit 1
export PATH="/share/software/anaconda3/2024.10/bin:$HOME/.local/bin:$PATH"

KF_CFG="${P8_STEP4_CONFIG:-pi05_aloha_sim_history_kf_arx-x5_seed_0}"
CKPT_NAME="${P8_STEP4_CKPT:-RoboDojo-sim-arx_x5-history-kf-lora-seed0}"
CKPT_ROOT="$ROOT/robodojo/XPolicyLab/policy/Pi_05/checkpoints/$CKPT_NAME"
DEPLOY="$ROOT/robodojo/XPolicyLab/policy/Pi_05/deploy.yml"
EVAL_NUM="${P8_STEP4_EVAL_NUM:-6}"
RUN_ROOT="${P8_STEP4_LOG_DIR:-$ROOT/logs/pi05/phase8_final_$(date +%Y-%m-%d_%H-%M-%S)}"

echo "[step4] host=$(hostname) start=$(date -Is) eval_num=$EVAL_NUM ckpt=$CKPT_NAME"
nvidia-smi -L || true

# ① 门禁裁判（未通过 → 不烧卡）
if ! python3 scripts/pi05/p8_gate_judge.py --require-pass; then
  echo "[step4] 门禁未通过 → 跳过评测 $(date -Is)"
  date -Is > logs/pi05/step4_skipped_gate_failed.flag
  exit 0
fi

# ② 选最终 ckpt
MAXSTEP=$(ls "$CKPT_ROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
if [[ -z "${MAXSTEP:-}" ]]; then
  echo "[step4] 找不到 ckpt 目录 $CKPT_ROOT → 退出"
  exit 0
fi
echo "[step4] ckpt=$CKPT_NAME step=$MAXSTEP"

# ③ 切 deploy.yml（备份一次，幂等）
if ! grep -q "^train_config_name: ${KF_CFG}$" "$DEPLOY"; then
  cp "$DEPLOY" "$DEPLOY.bak-step4-$(date +%Y%m%d_%H%M%S)"
fi
sed -i "s/^train_config_name:.*/train_config_name: ${KF_CFG}/" "$DEPLOY"
sed -i "s/^checkpoint_num:.*/checkpoint_num: ${MAXSTEP}/" "$DEPLOY"
grep -E "^(train_config_name|checkpoint_num):" "$DEPLOY"

# ④ 评测（4 卡 = 2 worker，共享队列）
mkdir -p "$RUN_ROOT"
date -Is > logs/pi05/step4_launched.flag
bash scripts/pi05/phase_mem_pi05.sh \
  --eval-num "$EVAL_NUM" \
  --ckpt "$CKPT_NAME" \
  --policy-gpus 0,2 --env-gpus 1,3 \
  --log-dir "$RUN_ROOT"
rc=$?
echo "[step4] eval rc=$rc $(date -Is)"
exit "$rc"
