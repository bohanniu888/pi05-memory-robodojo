#!/usr/bin/env bash
# 干净 GPU 上的因果配对：只改 RLIMIT_NPROC，并在崩溃后冷却，避免 Xid 残留污染。
set -uo pipefail
export BASH_ENV=/dev/null
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT" || exit 1
ulimit -u 8192 2>/dev/null || true

MEM_CKPT=${MEM_CKPT:-RoboDojo-sim-arx_x5-history-v2-lora-seed0}
OUT="$ROOT/logs/pi05/diag_swap_blocks_clean_gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "out=$OUT" | tee "$OUT/summary.txt"
echo "driver_ulimit_soft=$(ulimit -Su) driver_ulimit_hard=$(ulimit -Hu)" | tee -a "$OUT/summary.txt"

run_case() {
    local name="$1" nproc="$2" env_gpu="$3" policy_gpu="$4" port="$5"
    local out="$OUT/${name}.out"
    local t0; t0=$(date +%s)
    echo "=== [$name] env_gpu=$env_gpu policy_gpu=$policy_gpu RLIMIT_NPROC=$nproc 开始 $(date +%H:%M:%S)" | tee -a "$OUT/summary.txt"
    PODMAN_NPROC="$nproc" ROBODOJO_MAX_BASH_RETRIES=1 ROBODOJO_FATAL_RESTART_COUNT=4 \
    TASK=swap_blocks SEED=0 EVAL_NUM=1 PORT="$port" POLICY_GPU="$policy_gpu" ENV_GPU="$env_gpu" CKPT="$MEM_CKPT" \
    CLIENT_TIMEOUT=600 SERVER_TIMEOUT=300 \
    bash scripts/pi05/run_pi05_eval.sh > "$out" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))

    local cl=""
    cl=$(sed -n 's/^\[Phase1\] client_log=//p' "$out" | head -1)
    local nenv="?"; [[ -n "$cl" && -f "$cl" ]] && nenv=$(grep -a "Number of environments" "$cl" | tail -1 | tr -dc '0-9')
    local resets=0; [[ -n "$cl" && -f "$cl" ]] && resets=$(grep -ac "Resetting all environments" "$cl")
    local physx="no"; { grep -aq "Error code 700" "$out" || { [[ -n "$cl" && -f "$cl" ]] && grep -aq "Error code 700" "$cl"; }; } && physx="yes"
    local render="no"; [[ -n "$cl" && -f "$cl" ]] && grep -aq "\[render_sync\]" "$cl" && render="yes"
    local thread="no"; { grep -aq "can't start new thread" "$out" || { [[ -n "$cl" && -f "$cl" ]] && grep -aq "can't start new thread" "$cl"; }; } && thread="yes"
    local first_step="no"; [[ -n "$cl" && -f "$cl" ]] && grep -aq "env0 step:" "$cl" && first_step="yes"
    local res=""; res=$(find robodojo/eval_result/RoboDojo/swap_blocks -name _result.json -newermt "@$t0" 2>/dev/null | sort | tail -1)
    local score=""; if [[ -n "$res" ]]; then score=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('score'))" "$res" 2>/dev/null); fi

    local verdict="NO_RESET_RC=${rc}"
    if [[ "$physx" == "yes" ]]; then verdict="PHYSX700"
    elif [[ "$thread" == "yes" ]]; then verdict="THREAD_LIMIT"
    elif [[ "$render" == "yes" && "$first_step" == "yes" ]]; then verdict="STEPPING_RC=${rc}"
    elif [[ "$render" == "yes" ]]; then verdict="RENDER_OK_RC=${rc}"
    elif [[ "$resets" -gt 0 ]]; then verdict="RESET_OK_NO_RENDER_RC=${rc}"
    fi
    if [[ "$rc" == "0" && -n "$res" ]]; then verdict="PASS"; fi

    printf '[%s] env_gpu=%s nproc=%s 用时=%ss verdict=%s rc=%s envs=%s resets=%s physx700=%s render=%s step=%s thread_err=%s score=%s\n' \
        "$name" "$env_gpu" "$nproc" "$dt" "$verdict" "$rc" "$nenv" "$resets" "$physx" "$render" "$first_step" "$thread" "${score:-}" | tee -a "$OUT/summary.txt"
}

cooldown() {
    local sec="$1"
    echo "=== 冷却 ${sec}s，避免 Xid 后 GPU 上下文污染 $(date +%H:%M:%S) ===" | tee -a "$OUT/summary.txt"
    sleep "$sec"
}

# GPU0 在本轮从未跑过仿真，是干净卡。先证明高限可完整跑完。
run_case gpu0_high_before 8192:8192 0 2 26450
# 同一张 GPU0、同任务/缓存/seed，只把容器线程上限降到 1024。
run_case gpu0_low_1024 1024:1024 0 2 26451
# 崩溃后冷却，再回到高限。
cooldown 240
run_case gpu0_high_after 8192:8192 0 2 26452

echo "=== 全部结束 $(date +%H:%M:%S) ===" | tee -a "$OUT/summary.txt"
