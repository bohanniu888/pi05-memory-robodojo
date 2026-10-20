#!/usr/bin/env bash
# Phase 0E：离线 wiring check（debug mock 客户端，不需要 Isaac 容器）。
#
# 用法（在 GPU 作业内）：
#   srun --jobid=<jid> --overlap bash scripts/pi05/phase0_wiring_check.sh [seed] [task]
# 参数：
#   seed  0|1|2（对应 ckpt RoboDojo-sim-arx_x5-joint-<seed>），默认 2
#   task  默认 cover_blocks
#
# 判据：server 成功加载策略；client [CONNECTED]；10 个 episode × 20 步动作往返无异常。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SEED="${1:-2}"
TASK="${2:-cover_blocks}"
LOG_DIR="$ROOT/logs/pi05"
mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG="$LOG_DIR/wiring_debug_seed${SEED}_${TASK}_${RUN_ID}.log"

export EVAL_ENV_TYPE=debug
# 策略 server 用 GPU0，debug 客户端不需要 GPU（给 1 以保持与仿真一致的分卡约定）
POLICY_GPU=0
CLIENT_GPU=1
EVAL_ENV_CONDA="${PI05_CLIENT_CONDA_ENV:-opendm}"

echo "[0E] host=$(hostname) seed=$SEED task=$TASK env=$EVAL_ENV_CONDA log=$LOG"
cd "$ROOT/robodojo/XPolicyLab" || exit 1

bash policy/Pi_05/eval.sh \
    RoboDojo "$TASK" "RoboDojo-sim-arx_x5-joint-${SEED}" \
    arx_x5 joint 0 "$POLICY_GPU" "$CLIENT_GPU" uv "$EVAL_ENV_CONDA" \
    2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[0E] exit=$rc"
exit "$rc"
