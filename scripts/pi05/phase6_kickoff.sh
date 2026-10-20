#!/usr/bin/env bash
# Phase 6 编排：等训练收尾 → 真模型 slots 冒烟 → memory 评测队列（6 任务 × 1 seed × 6 集，2 worker）。
#
# 用法（login 节点，ssh 断开也不受影响）：
#   setsid nohup bash scripts/pi05/phase6_kickoff.sh > logs/pi05/p6_kickoff_$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &
# 取消：touch logs/pi05/p6_stop_watch（每个阶段开始前检查，最多等到当前阶段结束）
#
# 阶段门（任一失败即停止，不启动评测）：
#   [1] 等 checkpoints/<ckpt>/19999/params 出现（训练最终步保存；训练日志 40min 无更新视为停滞）
#   [2] GPU 显存回落到 <2GiB/卡
#   [3] PI05_SMOKE_REAL=1 真模型冒烟（组装 + 端到端活性）通过
#   [4] phase_mem_pi05.sh 评测队列（CKPT=history 训练 checkpoint；policy/env 分卡）
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT" || exit 1

JOBID="${JOBID:-75430}"
CKPT_NAME="${CKPT_NAME:-RoboDojo-sim-arx_x5-history-lora-seed0}"
FINAL_STEP="${FINAL_STEP:-19999}"
WATCH_LOG="${WATCH_LOG:-$ROOT/logs/pi05/p5_watch_20261006_111952.log}"
STOP_FILE="$ROOT/logs/pi05/p6_stop_watch"
CKPT_DIR="$ROOT/robodojo/XPolicyLab/policy/Pi_05/checkpoints/$CKPT_NAME"
EVAL_NUM="${EVAL_NUM:-6}"
SEEDS="${SEEDS:-0}"
POLL_SECS="${POLL_SECS:-60}"
STALE_SECS="${STALE_SECS:-2400}"
WAIT_TIMEOUT_SECS="${WAIT_TIMEOUT_SECS:-28800}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ROOT="${RUN_ROOT:-$ROOT/logs/pi05/phase6_mem_$TIMESTAMP}"

log() { echo "[p6] $(date -Is) $*"; }
check_stop() { [[ -f "$STOP_FILE" ]]; }

log "start host=$(hostname) job=$JOBID ckpt=$CKPT_NAME@$FINAL_STEP run_root=$RUN_ROOT"
mkdir -p "$RUN_ROOT"

if [[ -f "$STOP_FILE" ]]; then
    log "检测到 stop 文件 $STOP_FILE，直接退出"
    exit 0
fi

# ---------------------------------------------------------------- [1] 等训练收尾 ---
log "阶段1/4：等待 $CKPT_DIR/$FINAL_STEP 落盘（训练最终步；orbax 的 params 是目录，用 _CHECKPOINT_METADATA+params 判定）"
waited=0
while [[ ! -f "$CKPT_DIR/$FINAL_STEP/_CHECKPOINT_METADATA" || ! -d "$CKPT_DIR/$FINAL_STEP/params" ]]; do
    if check_stop; then log "stop 文件出现，退出等待"; exit 0; fi
    if (( waited >= WAIT_TIMEOUT_SECS )); then
        log "✗ 等待超时 ${WAIT_TIMEOUT_SECS}s，未出现 $FINAL_STEP/params；请人工检查训练"; exit 2
    fi
    if [[ -f "$WATCH_LOG" ]]; then
        stale=$(( $(date +%s) - $(stat -c %Y "$WATCH_LOG") ))
        if (( stale > STALE_SECS )); then
            log "✗ 训练日志已 ${stale}s 无更新（>${STALE_SECS}s），按停滞处理；退出"; exit 3
        fi
    fi
    if (( waited % 600 == 0 )); then
        log "  等待中 ${waited}s；最新进度：$(grep -o 'Progress on: [0-9.]*kit/[0-9.]*kit' "$WATCH_LOG" 2>/dev/null | tail -1)"
    fi
    sleep "$POLL_SECS"
    waited=$(( waited + POLL_SECS ))
done
if [[ -f "$CKPT_DIR/$FINAL_STEP/assets/arx_x5_sim/norm_stats.json" || -f "$CKPT_DIR/$FINAL_STEP/assets/arx_x5_sim/norm_stats.npz" ]]; then
    log "✓ 最终 checkpoint 已就绪：$CKPT_DIR/$FINAL_STEP（含 assets/arx_x5_sim）"
else
    log "✓ 最终 checkpoint 已就绪：$CKPT_DIR/$FINAL_STEP（⚠ 未发现 assets/arx_x5_sim，继续由冒烟验证）"
fi

# ------------------------------------------------------- [2] 等 GPU 显存回落 ---
log "阶段2/4：等待 GPU 显存回落（<2GiB/卡）"
drain_waited=0
while :; do
    used="$(srun --jobid="$JOBID" --overlap --quiet nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd, -)"
    max_used="$(echo "$used" | tr ',' '\n' | sort -n | tail -1)"
    if [[ -n "${max_used:-}" ]] && (( max_used < 2048 )); then
        log "✓ GPU 已回落：used=$used MiB"
        break
    fi
    if (( drain_waited >= 900 )); then
        log "⚠ GPU 显存 ${used:-?} MiB 仍高（等待>900s），继续尝试冒烟"
        break
    fi
    sleep 30
    drain_waited=$(( drain_waited + 30 ))
done

if check_stop; then log "stop 文件出现，退出（不启动冒烟/评测）"; exit 0; fi

# ------------------------------------------------------------- [3] 真模型冒烟 ---
SMOKE_LOG="$ROOT/logs/pi05/p6_real_smoke_${TIMESTAMP}.log"
log "阶段3/4：真模型 slots 冒烟 → $SMOKE_LOG"
PI05_SMOKE_REAL=1 POLICY_GPU=0 srun --jobid="$JOBID" --overlap bash scripts/pi05/smoke_pi05_eval_slots.sh >"$SMOKE_LOG" 2>&1
smoke_rc=$?
if (( smoke_rc != 0 )); then
    log "✗ 真模型冒烟失败 rc=$smoke_rc，停止（不启动评测）；日志：$SMOKE_LOG"
    tail -n 25 "$SMOKE_LOG"
    exit 4
fi
log "✓ 真模型冒烟通过：$(grep -E 'PASS|✓' "$SMOKE_LOG" | tail -3 | tr '\n' ' ')"
log "  活性/延迟摘要：$(grep -E '动作差异|延迟|bytes_in_use' "$SMOKE_LOG" | tail -4 | tr '\n' ' ')"

if check_stop; then log "stop 文件出现，退出（不启动评测）"; exit 0; fi

# ------------------------------------------------------------- [4] 评测队列 ---
EVAL_LOG="$RUN_ROOT/phase_mem_queue.log"
log "阶段4/4：Phase 6 memory 评测队列（eval_num=$EVAL_NUM seeds=$SEEDS ckpt=$CKPT_NAME）→ $EVAL_LOG"
srun --jobid="$JOBID" --overlap bash scripts/pi05/phase_mem_pi05.sh \
    --eval-num "$EVAL_NUM" \
    --seeds "$SEEDS" \
    --ckpt "$CKPT_NAME" \
    --policy-gpus 0,2 \
    --env-gpus 1,3 \
    --log-dir "$RUN_ROOT" >"$EVAL_LOG" 2>&1
eval_rc=$?
log "评测队列结束 rc=$eval_rc"
echo
log "================= Phase 6 memory 结果 ================="
cat "$RUN_ROOT/results.tsv" 2>/dev/null || true
echo
grep -E "score 均值|Memory" "$EVAL_LOG" | tail -8 || true
log "全部结束。结果目录：$RUN_ROOT"
