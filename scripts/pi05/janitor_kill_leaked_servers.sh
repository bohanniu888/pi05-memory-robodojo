#!/usr/bin/env bash
# 守护：清理泄漏的 π0.5 policy server。
# 背景：run_pi05_eval.sh 只 kill 了 `bash robodojo.sh server` 包装进程，其下层的
# python（openpi server）会被 reparent 到 PID 1 继续占显存与端口（实测 run1 泄漏 7.6GB）。
# 本守护每 60s 扫一次：命令行匹配本仓库 Pi_05 openpi、且 PPid=1（孤儿）的进程，TERM→KILL。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PATTERN="$ROOT/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
LOG="$ROOT/logs/pi05/janitor.log"
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) janitor 启动" >> "$LOG"
while true; do
    for pid in $(pgrep -u "$USER" -f "$PATTERN" 2>/dev/null); do
        ppid=$(awk '/^PPid:/{print $2}' /proc/$pid/status 2>/dev/null)
        [[ -z "$ppid" ]] && continue
        if [[ "$ppid" == "1" ]]; then
            echo "$(date -Is) 清理泄漏 server pid=$pid" >> "$LOG"
            kill "$pid" 2>/dev/null
            sleep 5
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    sleep 60
done
