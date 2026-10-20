#!/usr/bin/env bash
# Phase 0C：安装 uv + openpi(π0.5) 运行环境（在 GPU 节点作业内运行）
#
# 用法：
#   srun --jobid=<jid> --overlap bash scripts/pi05/bootstrap_env.sh
#
# 说明：install.sh 会执行 `uv sync --group lerobot`（openpi/.venv），再安装 XPolicyLab 本地包。
# 落点：robodojo/XPolicyLab/policy/Pi_05/openpi/.venv
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POLICY_DIR="$ROOT/robodojo/XPolicyLab/policy/Pi_05"
LOG_DIR="$ROOT/logs/pi05"
mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG="$LOG_DIR/bootstrap_env_${RUN_ID}.log"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[0C] uv 未找到，尝试安装到 ~/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "[0C] uv 安装失败" >&2; exit 1; }
echo "[0C] uv: $(uv --version)"
echo "[0C] log=$LOG"

cd "$POLICY_DIR" || exit 1
bash install.sh 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[0C] exit=$rc  log=$LOG"
exit "$rc"
