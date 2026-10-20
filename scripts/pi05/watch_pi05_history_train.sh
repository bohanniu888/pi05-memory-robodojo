#!/usr/bin/env bash
# Phase 5 训练看护：训练进程退出且未达目标 step 时自动 --resume 续训（最多 MAX_RETRY 次）。
# 用法：setsid nohup bash scripts/pi05/watch_pi05_history_train.sh > logs/pi05/p5_watch_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# 停止看护：touch logs/pi05/p5_stop_watch
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT}"

TARGET_STEPS="${TARGET_STEPS:-20000}"
MAX_RETRY="${MAX_RETRY:-8}"
JOBID="${JOBID:-75430}"
EXP="${EXP:-RoboDojo-sim-arx_x5-history-lora-seed0}"
GPUS="${GPUS:-0,1,2,3}"
CKPT_DIR="robodojo/XPolicyLab/policy/Pi_05/checkpoints/${EXP}"
PATTERN="scripts/train.py pi05_aloha_sim_history_arx-x5_seed_0"

latest_step() {
  ls -1 "${CKPT_DIR}" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1
}

for attempt in $(seq 1 "${MAX_RETRY}"); do
  while pgrep -f "${PATTERN}" >/dev/null; do sleep 60; done
  last=$(latest_step)
  # 训练 loop 的最后一步是 num_train_steps-1（save 条件 step == num_train_steps-1），
  # 因此完成判定用 last+1 >= TARGET，避免把 19999/20000 误判为"未完成"而无限 resume。
  if [[ -n "${last}" && "$(( last + 1 ))" -ge "${TARGET_STEPS}" ]]; then
    echo "[watch] $(date '+%F %T') reached step ${last} >= ${TARGET_STEPS}, done."
    exit 0
  fi
  if [[ -f "${ROOT}/logs/pi05/p5_stop_watch" ]]; then
    echo "[watch] $(date '+%F %T') stop flag found, exiting."
    exit 0
  fi
  echo "[watch] $(date '+%F %T') attempt ${attempt}: not running (last ckpt=${last:-none}), resuming..."
  srun --jobid="${JOBID}" --overlap bash scripts/pi05/train_pi05_history.sh \
    --gpus "${GPUS}" --fsdp-devices 1 --batch-size 64 --steps "${TARGET_STEPS}" \
    --save-interval 1000 --keep-period 2000 --exp-name "${EXP}" --resume
  echo "[watch] $(date '+%F %T') training exited (rc=$?), cooling down 60s"
  sleep 60
done

echo "[watch] $(date '+%F %T') max retries reached; manual check needed"
exit 1
