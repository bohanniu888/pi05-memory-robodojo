#!/usr/bin/env bash
# Phase 0A：下载 π0.5 的 RoboDojo 训练权重（封装官方 robodojo/scripts/RoboDojo/download_ckpt.sh）
#
# 用法：
#   bash scripts/pi05/download_ckpt.sh                    # 默认 pi05 + modelscope
#   bash scripts/pi05/download_ckpt.sh pi05 huggingface   # 走 hf-mirror（git 通道）
#
# 环境变量：
#   PI05_CKPT_SOURCE  覆盖默认 source（modelscope|huggingface）
#   HF_REPO_URL       自定义 HF 仓库地址（默认替换为 hf-mirror）
#
# 落点：robodojo/XPolicyLab/policy/Pi_05/checkpoints/（指向 .cache 的符号链接）
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REPO="$ROOT/robodojo"
POLICY="${1:-pi05}"
SOURCE="${2:-${PI05_CKPT_SOURCE:-modelscope}}"
LOG_DIR="$ROOT/logs/pi05"
mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG="$LOG_DIR/download_ckpt_${POLICY}_${SOURCE}_${RUN_ID}.log"

case "$SOURCE" in
  modelscope) ;;
  huggingface)
    export HF_REPO_URL="${HF_REPO_URL:-https://hf-mirror.com/datasets/RoboDojo-Benchmark/RoboDojo}"
    ;;
  *) echo "[0A] source 必须是 huggingface 或 modelscope，收到: $SOURCE" >&2; exit 2 ;;
esac

echo "[0A] host=$(hostname) policy=$POLICY source=$SOURCE"
echo "[0A] log=$LOG"
cd "$REPO" || exit 1
bash scripts/RoboDojo/download_ckpt.sh "$SOURCE" "$POLICY" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[0A] exit=$rc"
if [[ $rc -eq 0 ]]; then
  ls -l "XPolicyLab/policy/Pi_05/checkpoints/" 2>/dev/null | head -5
fi
exit "$rc"
