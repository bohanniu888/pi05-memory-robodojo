#!/usr/bin/env bash
# Phase 6 诊断：训练权重 + 历史置空（history_enabled=false → 训练 transform 输出全 pad + mask=0，
# 等价训练时的"无记忆样本"口径），用于区分「策略本身退化」vs「历史通路在评测域有害」。
# 跑完自动恢复 deploy.yml（trap EXIT）。用法：
#   setsid nohup bash scripts/pi05/phase6_diag_nohist.sh > logs/pi05/p6_diag_nohist_$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT" || exit 1
JOBID="${JOBID:-75430}"
TASKS="${TASKS:-cover_blocks,match_and_pick_from_conveyor}"
RUN_ROOT="${RUN_ROOT:-$ROOT/logs/pi05/phase6_diag_nohist_$(date +%Y-%m-%d_%H-%M-%S)}"
DEPLOY="$ROOT/robodojo/XPolicyLab/policy/Pi_05/deploy.yml"
BACKUP="$DEPLOY.phase6diag.bak"

cp "$DEPLOY" "$BACKUP"
restore() {
    cp "$BACKUP" "$DEPLOY"
    echo "[diag] $(date -Is) deploy.yml 已恢复为 history_enabled=true（备份 $BACKUP）"
}
trap restore EXIT

sed -i 's/^history_enabled: true$/history_enabled: false/' "$DEPLOY"
grep -n "^history_enabled" "$DEPLOY"
echo "[diag] $(date -Is) run_root=$RUN_ROOT tasks=$TASKS ckpt=history-lora-seed0@19999（历史置空）"

srun --jobid="$JOBID" --overlap bash scripts/pi05/phase_mem_pi05.sh \
    --eval-num 6 --seeds 0 \
    --ckpt RoboDojo-sim-arx_x5-history-lora-seed0 \
    --tasks "$TASKS" \
    --policy-gpus 0,2 --env-gpus 1,3 \
    --log-dir "$RUN_ROOT"
rc=$?
echo "[diag] queue rc=$rc"
cat "$RUN_ROOT/results.tsv" 2>/dev/null
