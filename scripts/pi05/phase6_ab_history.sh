#!/usr/bin/env bash
# Phase 6 A/B：训练权重（history-lora-19999）× {历史开, 历史关} × 2 任务 × 3 seeds × 20 集。
# 目的：加大样本，判断"历史通路在评测域有害"是真实效应还是 6 集噪声。
# 用法：setsid nohup bash scripts/pi05/phase6_ab_history.sh > logs/pi05/p6_ab_$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &
# 停止：pkill -f phase6_ab_history.sh（deploy.yml 由 trap 自动恢复）
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT" || exit 1
JOBID="${JOBID:?请设置 JOBID=<现有分配>}"
CKPT="${CKPT:-RoboDojo-sim-arx_x5-history-lora-seed0}"
TASKS="${TASKS:-cover_blocks,match_and_pick_from_conveyor}"
SEEDS="${SEEDS:-0,1,2}"
EVAL_NUM="${EVAL_NUM:-20}"
DEPLOY="$ROOT/robodojo/XPolicyLab/policy/Pi_05/deploy.yml"
BACKUP="$DEPLOY.phase6ab.bak"
TS="$(date +%Y-%m-%d_%H-%M-%S)"

cp "$DEPLOY" "$BACKUP"
restore() {
    cp "$BACKUP" "$DEPLOY"
    echo "[ab] $(date -Is) deploy.yml 已恢复为 history_enabled=true（备份 $BACKUP）"
}
trap restore EXIT

run_condition() {
    local cond="$1" enabled="$2" root="$3" rc
    sed -i "s/^history_enabled: .*/history_enabled: ${enabled}/" "$DEPLOY"
    echo "[ab] $(date -Is) === 条件 ${cond}（history_enabled=${enabled}）root=$root ==="
    echo "[ab] tasks=$TASKS seeds=$SEEDS eval_num=$EVAL_NUM ckpt=$CKPT"
    srun --jobid="$JOBID" --overlap bash scripts/pi05/phase_mem_pi05.sh \
        --eval-num "$EVAL_NUM" --seeds "$SEEDS" --tasks "$TASKS" \
        --ckpt "$CKPT" --policy-gpus 0,2 --env-gpus 1,3 \
        --log-dir "$root"
    rc=$?
    echo "[ab] $(date -Is) === 条件 ${cond} 结束 rc=$rc ==="
    echo "[ab] --- ${cond} results.tsv ---"
    cat "$root/results.tsv" 2>/dev/null
}

run_condition "hist_on" true "$ROOT/logs/pi05/phase6_ab_on_$TS"
run_condition "hist_off" false "$ROOT/logs/pi05/phase6_ab_off_$TS"

echo "[ab] $(date -Is) 全部结束。对比："
echo "  on : $ROOT/logs/pi05/phase6_ab_on_$TS/results.tsv"
echo "  off: $ROOT/logs/pi05/phase6_ab_off_$TS/results.tsv"
