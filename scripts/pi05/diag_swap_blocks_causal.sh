#!/usr/bin/env bash
# swap_blocks 根因实验：单变量、一次失败不回退、显式打印容器线程上限。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT" || exit 1
ulimit -u 8192 2>/dev/null || true

MEM_CKPT=${MEM_CKPT:-RoboDojo-sim-arx_x5-history-v2-lora-seed0}
BASE_CKPT=${BASE_CKPT:-RoboDojo-sim-arx_x5-joint-2}
CFG="$ROOT/logs/pi05/diag_cfg"
OUT="$ROOT/logs/pi05/diag_swap_blocks_causal_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "out=$OUT" | tee "$OUT/summary.txt"
echo "host_ulimit_soft=$(ulimit -Su) host_ulimit_hard=$(ulimit -Hu)" | tee -a "$OUT/summary.txt"

run_case() {
    local name="$1" mounts="$2" env_gpu="$3" policy_gpu="$4" ckpt="$5" port="$6"
    local out="$OUT/${name}.out"
    local t0; t0=$(date +%s)
    echo "=== [$name] env_gpu=$env_gpu policy_gpu=$policy_gpu ckpt=$ckpt mounts=${mounts:-none} port=$port 开始 $(date +%H:%M:%S)" | tee -a "$OUT/summary.txt"
    ROBODOJO_MAX_BASH_RETRIES=1 ROBODOJO_FATAL_RESTART_COUNT=4 \
    EXTRA_MOUNTS="$mounts" TASK=swap_blocks SEED=0 EVAL_NUM=1 PORT="$port" \
        POLICY_GPU="$policy_gpu" ENV_GPU="$env_gpu" CKPT="$ckpt" \
        CLIENT_TIMEOUT=900 SERVER_TIMEOUT=300 \
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

    printf '[%s] 用时=%ss verdict=%s rc=%s envs=%s resets=%s physx700=%s render=%s step=%s thread_err=%s score=%s result=%s\n' \
        "$name" "$dt" "$verdict" "$rc" "$nenv" "$resets" "$physx" "$render" "$first_step" "$thread" "${score:-}" "${res:-none}" | tee -a "$OUT/summary.txt"
}

# 1) 复现：10 envs，原任务，记忆 ckpt，GPU1。
run_case c02_nenv10_gpu1 "" 1 0 "$MEM_CKPT" 26411
# 2) 正对照：1 env，GPU1。若只有在少 env 时才不崩，根因指向并行环境缩放。
run_case c01_nenv1_gpu1 "-v $CFG/sim_config_nenv1.yml:/workspace/RoboDojo/env_cfg/sim/sim_config.yml:ro" 1 0 "$MEM_CKPT" 26410
# 3) 阈值：2 env。
run_case c03_nenv2_gpu1 "-v $CFG/sim_config_nenv2.yml:/workspace/RoboDojo/env_cfg/sim/sim_config.yml:ro" 1 0 "$MEM_CKPT" 26412
# 4) 阈值：4 env。
run_case c04_nenv4_gpu1 "-v $CFG/sim_config_nenv4.yml:/workspace/RoboDojo/env_cfg/sim/sim_config.yml:ro" 1 0 "$MEM_CKPT" 26413
# 5) 阈值：8 env。
run_case c05_nenv8_gpu1 "-v $CFG/sim_config_nenv8.yml:/workspace/RoboDojo/env_cfg/sim/sim_config.yml:ro" 1 0 "$MEM_CKPT" 26414
# 6) 资产对照：10 envs，cushion 换成 cube。
run_case c06_nenv10_cube_gpu1 "-v $CFG/swap_blocks_cube_mats.yml:/workspace/RoboDojo/task/RoboDojo/config/swap_blocks.yml:ro" 1 0 "$MEM_CKPT" 26415
# 7) 硬件对照：10 envs 放到 GPU0（policy server 去 GPU2）。
run_case c07_nenv10_gpu0 "" 0 2 "$MEM_CKPT" 26416
# 8) GPU0 的 1-env 正对照。
run_case c08_nenv1_gpu0 "-v $CFG/sim_config_nenv1.yml:/workspace/RoboDojo/env_cfg/sim/sim_config.yml:ro" 0 2 "$MEM_CKPT" 26417

echo "=== 全部结束 $(date +%H:%M:%S) ===" | tee -a "$OUT/summary.txt"
