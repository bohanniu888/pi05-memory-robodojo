#!/usr/bin/env bash
# swap_blocks 根因配对：只改 RLIMIT_NPROC，其他条件固定（10 envs, GPU1, 原任务, 现用缓存）。
set -uo pipefail
export BASH_ENV=/dev/null
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT" || exit 1
ulimit -u 8192 2>/dev/null || true

MEM_CKPT=${MEM_CKPT:-RoboDojo-sim-arx_x5-history-v2-lora-seed0}
OUT="$ROOT/logs/pi05/diag_swap_blocks_nproc_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "out=$OUT" | tee "$OUT/summary.txt"
echo "driver_ulimit_soft=$(ulimit -Su) driver_ulimit_hard=$(ulimit -Hu)" | tee -a "$OUT/summary.txt"

run_case() {
    local name="$1" host_low="$2" nproc="$3" port="$4"
    local out="$OUT/${name}.out"
    local t0; t0=$(date +%s)
    echo "=== [$name] host_low=$host_low RLIMIT_NPROC=$nproc 开始 $(date +%H:%M:%S)" | tee -a "$OUT/summary.txt"
    if [[ "$host_low" == "1" ]]; then
        (
            ulimit -S -u 256
            ulimit -H -u 512
            PODMAN_NPROC="$nproc" ROBODOJO_MAX_BASH_RETRIES=1 ROBODOJO_FATAL_RESTART_COUNT=4 \
            TASK=swap_blocks SEED=0 EVAL_NUM=1 PORT="$port" POLICY_GPU=0 ENV_GPU=1 CKPT="$MEM_CKPT" \
            CLIENT_TIMEOUT=600 SERVER_TIMEOUT=300 \
            bash scripts/pi05/run_pi05_eval.sh
        ) > "$out" 2>&1
    else
        PODMAN_NPROC="$nproc" ROBODOJO_MAX_BASH_RETRIES=1 ROBODOJO_FATAL_RESTART_COUNT=4 \
        TASK=swap_blocks SEED=0 EVAL_NUM=1 PORT="$port" POLICY_GPU=0 ENV_GPU=1 CKPT="$MEM_CKPT" \
        CLIENT_TIMEOUT=600 SERVER_TIMEOUT=300 \
        bash scripts/pi05/run_pi05_eval.sh > "$out" 2>&1
    fi
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
    local host_line=""; host_line=$(grep -a 'host ulimit -u' "$out" | head -1)

    local res=""; res=$(find robodojo/eval_result/RoboDojo/swap_blocks -name _result.json -newermt "@$t0" 2>/dev/null | sort | tail -1)
    local score=""; if [[ -n "$res" ]]; then score=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('score'))" "$res" 2>/dev/null); fi

    local verdict="NO_RESET_RC=${rc}"
    if [[ "$physx" == "yes" ]]; then
        verdict="PHYSX700"
    elif [[ "$thread" == "yes" ]]; then
        verdict="THREAD_LIMIT"
    elif [[ "$render" == "yes" && "$first_step" == "yes" ]]; then
        verdict="STEPPING_RC=${rc}"
    elif [[ "$render" == "yes" ]]; then
        verdict="RENDER_OK_RC=${rc}"
    elif [[ "$resets" -gt 0 ]]; then
        verdict="RESET_OK_NO_RENDER_RC=${rc}"
    fi
    if [[ "$rc" == "0" && -n "$res" ]]; then verdict="PASS"; fi

    printf '[%s] host_low=%s nproc=%s 用时=%ss verdict=%s rc=%s envs=%s resets=%s physx700=%s render=%s step=%s thread_err=%s score=%s %s\n' \
        "$name" "$host_low" "$nproc" "$dt" "$verdict" "$rc" "$nenv" "$resets" "$physx" "$render" "$first_step" "$thread" "${score:-}" "$host_line" | tee -a "$OUT/summary.txt"
}

# 完全复刻此前崩溃环境：host 与 container 都是 soft=256/hard=512。
run_case h256_c256_nenv10_gpu1 1 256:512 26430
# 只压 container，host 保持 8192：判断故障点在宿主还是容器。
run_case h8192_c256_nenv10_gpu1 0 256:512 26431
# 剂量响应。
run_case h8192_c512_nenv10_gpu1 0 512:512 26432
run_case h8192_c1024_nenv10_gpu1 0 1024:1024 26433
run_case h8192_c8192_nenv10_gpu1 0 8192:8192 26434

echo "=== 全部结束 $(date +%H:%M:%S) ===" | tee -a "$OUT/summary.txt"
