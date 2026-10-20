#!/usr/bin/env bash
# Phase 0 状态检查：一条命令看清 0A/0B/0C 三道门槛。
#
# 用法：
#   bash scripts/pi05/phase0_status.sh            # 快照
#   bash scripts/pi05/phase0_status.sh --watch 120  # 每 120s 刷新
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REPO="$ROOT/robodojo"
POLICY="$REPO/XPolicyLab/policy/Pi_05"
CKPT_DIR="$POLICY/checkpoints"
DEMO_DIR="$REPO/data/demo"

snapshot() {
  echo "===== Phase 0 状态 ($(date '+%F %T')) ====="
  local ckpt_mb demo_mb
  ckpt_mb=$(du -sm "$REPO/.cache/robodojo_ckpt_modelscope_repo" 2>/dev/null | cut -f1)
  demo_mb=$(du -sm "$REPO/.cache/robodojo_data_modelscope_demo_repo" 2>/dev/null | cut -f1)

  echo "[0A] ckpt cache: ${ckpt_mb:-0} MB (目标 ~69000 MB)"
  if compgen -G "$CKPT_DIR/RoboDojo-sim-arx_x5-joint-*/59999/params" >/dev/null; then
    echo "[0A] ✅ checkpoints 落位: $(ls -d "$CKPT_DIR"/RoboDojo-sim-arx_x5-joint-*/59999 2>/dev/null | tr '\n' ' ')"
  else
    echo "[0A] ⏳ 未落位（$CKPT_DIR）"
  fi

  if [[ -e "$DEMO_DIR/.download_complete" ]]; then
    echo "[0B] ✅ demo 完成: $(du -shL "$DEMO_DIR" 2>/dev/null | cut -f1) ($DEMO_DIR)"
  else
    echo "[0B] ⏳ demo cache: ${demo_mb:-0} MB (目标 ~5900 MB)"
  fi

  if [[ -x "$POLICY/openpi/.venv/bin/python" ]]; then
    local ver
    ver=$("$POLICY/openpi/.venv/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null)
    if "$POLICY/openpi/.venv/bin/python" -c 'import openpi' >/dev/null 2>&1; then
      echo "[0C] ✅ openpi venv 可导入: $POLICY/openpi/.venv (python ${ver:-?})"
    else
      echo "[0C] ⏳ openpi venv 存在但尚不可导入 (python ${ver:-?})，uv sync 可能仍在进行"
    fi
  else
    echo "[0C] ⏳ openpi venv 未就绪"
  fi

  if compgen -G "$ROOT/logs/pi05/probe_*.json" >/dev/null; then
    echo "[0D] ✅ 探针结果: $(ls -t "$ROOT"/logs/pi05/probe_*.json | head -1)"
  else
    echo "[0D] ⏳ 无探针结果"
  fi

  if [[ -n "$(find "$ROOT/logs/pi05" -name 'wiring_debug_*.log' -print -quit 2>/dev/null)" ]]; then
    local wlog
    wlog=$(ls -t "$ROOT"/logs/pi05/wiring_debug_*.log | head -1)
    if grep -q "MAIN\] eval finished" "$wlog" 2>/dev/null; then
      echo "[0E] ✅ debug wiring 通过: $wlog"
    else
      echo "[0E] ⏳ wiring 运行中: $wlog"
    fi
  elif [[ -n "$(find "$POLICY/eval_result" -name '*_result.json' -print -quit 2>/dev/null)" ]]; then
    echo "[0E] ✅ 有 _result.json（sim 评测）"
  else
    echo "[0E] ⏳ 无调试 wiring 结果"
  fi
}

if [[ "${1:-}" == "--watch" ]]; then
  interval="${2:-120}"
  while true; do
    clear
    snapshot
    sleep "$interval"
  done
else
  snapshot
fi
