#!/usr/bin/env bash
# Phase 0B/4：下载 RoboDojo 数据（封装官方 robodojo/scripts/RoboDojo/download_data.sh）
#
# 用法：
#   bash scripts/pi05/fetch_demos.sh                     # demo（约 1.5GB，默认，先打通链路）
#   bash scripts/pi05/fetch_demos.sh lerobot_v3.0        # 120GB，joint-only 14 维，π0.5 训练首选
#   bash scripts/pi05/fetch_demos.sh lerobot_v3.0 huggingface
#
# 注意：`lerobot_v3.0_ee` 是 16 维 EE 空间，与当前 14 维 joint checkpoint 不匹配，只用于 EE 改训。
#
# 环境变量：
#   PI05_DATA_SOURCE  覆盖默认 source（modelscope|huggingface）
#   HF_REPO_URL       自定义 HF 仓库地址（默认替换为 hf-mirror）
#
# 落点：robodojo/data/<DATA_DIR_NAME>/（指向 .cache 的符号链接）
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REPO="$ROOT/robodojo"
FORMAT="${1:-demo}"
SOURCE="${2:-${PI05_DATA_SOURCE:-modelscope}}"
LOG_DIR="$ROOT/logs/pi05"
mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG="$LOG_DIR/download_data_${FORMAT}_${SOURCE}_${RUN_ID}.log"

case "$SOURCE" in
  modelscope) ;;
  huggingface)
    export HF_REPO_URL="${HF_REPO_URL:-https://hf-mirror.com/datasets/RoboDojo-Benchmark/RoboDojo}"
    ;;
  *) echo "[0B] source 必须是 huggingface 或 modelscope，收到: $SOURCE" >&2; exit 2 ;;
esac

echo "[0B] host=$(hostname) format=$FORMAT source=$SOURCE"
echo "[0B] log=$LOG"
cd "$REPO" || exit 1
bash scripts/RoboDojo/download_data.sh "$SOURCE" "$FORMAT" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[0B] exit=$rc"
exit "$rc"
