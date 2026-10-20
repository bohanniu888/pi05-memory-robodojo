#!/usr/bin/env bash
# swap_blocks reset 崩溃的根因矩阵。
#
# 设计原则：
#   * 每个 case 都是全新的 policy server + Podman 容器；
#   * 关闭自动重启，避免第二次启动掩盖第一次失败；
#   * 从客户端日志读取实际 num_envs，挂载失败判为 INVALID，不作为因果证据；
#   * 记录环境 GPU 的峰值显存/利用率，帮助排除或确认 OOM；
#   * 仍在原 Slurm allocation 内运行，不额外申请资源。
#
# 用法（在 gpu01 的 allocation 内）：
#   CASE_SET=threshold bash scripts/pi05/diag_swap_blocks_rootcause.sh
#   CASE_SET=controls  bash scripts/pi05/diag_swap_blocks_rootcause.sh
set -uo pipefail
export BASH_ENV=/dev/null

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT" || exit 1
ulimit -u 8192 2>/dev/null || true

MEM_CKPT=${MEM_CKPT:-RoboDojo-sim-arx_x5-history-v2-lora-seed0}
BASE_CKPT=${BASE_CKPT:-RoboDojo-sim-arx_x5-joint-2}
ENV_GPU=${ENV_GPU:-2}
POLICY_GPU=${POLICY_GPU:-0}
SEED=${SEED:-0}
CFG="$ROOT/logs/pi05/diag_cfg"
CASE_SET=${CASE_SET:-threshold}
DIAG_STAGES=${DIAG_STAGES:-1}
OUT="$ROOT/logs/pi05/diag_swap_blocks_rootcause_${CASE_SET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

STAGE_MOUNTS=""
if [[ "$DIAG_STAGES" == "1" ]]; then
    STAGE_MOUNTS="-v $ROOT/robodojo/env/diagnostics.py:/workspace/RoboDojo/env/diagnostics.py:ro"
    STAGE_MOUNTS+=" -v $ROOT/robodojo/env/environment/task_env.py:/workspace/RoboDojo/env/environment/task_env.py:ro"
    STAGE_MOUNTS+=" -v $ROOT/robodojo/env/scene_manager/scene_manager.py:/workspace/RoboDojo/env/scene_manager/scene_manager.py:ro"
    STAGE_MOUNTS+=" -v $ROOT/robodojo/env/scene_manager/objects/articulation.py:/workspace/RoboDojo/env/scene_manager/objects/articulation.py:ro"
    STAGE_MOUNTS+=" -v $ROOT/robodojo/src/eval_client/eval_env.py:/workspace/RoboDojo/src/eval_client/eval_env.py:ro"
fi

echo "out=$OUT" | tee "$OUT/summary.txt"
echo "case_set=$CASE_SET env_gpu=$ENV_GPU policy_gpu=$POLICY_GPU seed=$SEED" | tee -a "$OUT/summary.txt"
echo "host_ulimit_soft=$(ulimit -Su) host_ulimit_hard=$(ulimit -Hu)" | tee -a "$OUT/summary.txt"

run_case() {
    local name="$1" expected_envs="$2" task="$3" mounts="$4" ckpt="$5" nproc="$6" port="$7" seed="$8" cooldown_after="$9" art_mode="${10:-}"
    local out="$OUT/${name}.out"
    local mem="$OUT/${name}.gpu.csv"
    local thr="$OUT/${name}.threads.csv"
    local t0; t0=$(date +%s)

    echo "=== [$name] task=$task expected_envs=$expected_envs env_gpu=$ENV_GPU policy_gpu=$POLICY_GPU nproc=${nproc:-<image-default>} art_reloc=${art_mode:-normal} ckpt=$ckpt seed=$seed 开始 $(date +%H:%M:%S)" | tee -a "$OUT/summary.txt"

    (while true; do
        nvidia-smi --id="$ENV_GPU" --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true
        sleep 1
    done) > "$mem" 2>/dev/null &
    local mem_pid=$!
    (while true; do
        printf '%s %s\n' "$(date +%s)" "$(ps -u "$USER" -L --no-headers 2>/dev/null | wc -l)"
        sleep 1
    done) > "$thr" 2>/dev/null &
    local thr_pid=$!

    local all_mounts="$mounts"
    [[ "$DIAG_STAGES" == "1" ]] && all_mounts="$STAGE_MOUNTS $mounts"

    PODMAN_NPROC="$nproc" DIAG_STAGES="$DIAG_STAGES" ART_RELOC_MODE="$art_mode" \
    ROBODOJO_MAX_BASH_RETRIES=0 ROBODOJO_FATAL_RESTART_COUNT=4 \
    EXTRA_MOUNTS="$all_mounts" TASK="$task" SEED="$seed" EVAL_NUM=1 PORT="$port" \
    POLICY_GPU="$POLICY_GPU" ENV_GPU="$ENV_GPU" CKPT="$ckpt" \
    CLIENT_TIMEOUT=900 SERVER_TIMEOUT=300 \
    bash scripts/pi05/run_pi05_eval.sh > "$out" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    kill "$mem_pid" "$thr_pid" 2>/dev/null || true
    wait "$mem_pid" "$thr_pid" 2>/dev/null || true

    local cl=""
    cl=$(sed -n 's/^\[Phase1\] client_log=//p' "$out" | head -1)
    local nenv=""
    [[ -n "$cl" && -f "$cl" ]] && nenv=$(grep -a "Number of environments" "$cl" | tail -1 | tr -dc '0-9')
    local resets=0; [[ -n "$cl" && -f "$cl" ]] && resets=$(grep -ac "Resetting all environments" "$cl")
    local physx="no"; { grep -aq "Error code 700" "$out" || { [[ -n "$cl" && -f "$cl" ]] && grep -aq "Error code 700" "$cl"; }; } && physx="yes"
    local render="no"; [[ -n "$cl" && -f "$cl" ]] && grep -aq "\[render_sync\]" "$cl" && render="yes"
    local thread="no"; { grep -aq "can't start new thread" "$out" || { [[ -n "$cl" && -f "$cl" ]] && grep -aq "can't start new thread" "$cl"; }; } && thread="yes"
    local first_step="no"; [[ -n "$cl" && -f "$cl" ]] && grep -aq "env0 step:" "$cl" && first_step="yes"
    local last_stage="none"
    if [[ -n "$cl" && -f "$cl" ]]; then
        last_stage=$(grep -aE "Scene creation|Starting the simulation|Completed setting up|Resetting all environments|\[render_sync\]|env0 step:" "$cl" | tail -1 | sed 's/[[:space:]]\+/ /g' | cut -c1-100)
        [[ -z "$last_stage" ]] && last_stage="startup"
    fi

    local peak_mem=""; peak_mem=$(awk -F', ' 'NF>=2 {gsub(/^ +| +$/,"",$2); if (($2+0)>m) m=$2+0} END {print m+0}' "$mem" 2>/dev/null)
    local peak_util=""; peak_util=$(awk -F', ' 'NF>=3 {gsub(/^ +| +$/,"",$3); if (($3+0)>m) m=$3+0} END {print m+0}' "$mem" 2>/dev/null)
    local peak_threads=""; peak_threads=$(awk 'NF>=2 {if (($2+0)>m) m=$2+0} END {print m+0}' "$thr" 2>/dev/null)

    local res=""; res=$(find "robodojo/eval_result/RoboDojo/$task" -name _result.json -newermt "@$t0" 2>/dev/null | sort | tail -1)
    local score=""; if [[ -n "$res" ]]; then score=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('score'))" "$res" 2>/dev/null); fi

    local verdict="NO_RESET_RC=${rc}"
    if [[ -z "$nenv" ]]; then
        verdict="STARTUP_OR_INVALID_RC=${rc}"
    elif [[ "$nenv" != "$expected_envs" ]]; then
        verdict="INVALID_ENVS_${nenv}_EXPECTED_${expected_envs}"
    elif [[ "$physx" == "yes" ]]; then
        verdict="PHYSX700"
    elif [[ "$thread" == "yes" ]]; then
        verdict="THREAD_LIMIT"
    elif [[ "$render" == "yes" && "$first_step" == "yes" ]]; then
        verdict="STEPPING_RC=${rc}"
    elif [[ "$render" == "yes" ]]; then
        verdict="RENDER_OK_RC=${rc}"
    elif [[ "$resets" -gt 0 ]]; then
        verdict="RESET_NO_RENDER_RC=${rc}"
    fi
    if [[ "$rc" == "0" && -n "$res" ]]; then verdict="PASS"; fi

    printf '[%s] task=%s 用时=%ss verdict=%s rc=%s envs=%s resets=%s physx700=%s render=%s step=%s thread_err=%s peak_gpu_mem=%sMiB peak_util=%s%% peak_threads=%s score=%s stage=%s\n' \
        "$name" "$task" "$dt" "$verdict" "$rc" "${nenv:-?}" "$resets" "$physx" "$render" "$first_step" "$thread" "${peak_mem:-?}" "${peak_util:-?}" "${peak_threads:-?}" "${score:-}" "$last_stage" \
        | tee -a "$OUT/summary.txt"

    if [[ "$verdict" == "PHYSX700" || "$verdict" == THREAD_LIMIT* || "$verdict" == STEPPING_RC=* ]]; then
        echo "=== 冷却 ${cooldown_after}s，避免 Xid 影响下一个 case $(date +%H:%M:%S) ===" | tee -a "$OUT/summary.txt"
        sleep "$cooldown_after"
    fi
}

port=26500

if [[ "$CASE_SET" == "threshold" ]]; then
    # 只改 num_envs；全部使用高 nproc、同一张 env GPU、同一 seed/ckpt。
    run_case n01_expected1 1 swap_blocks "-v $CFG/sim_config_nenv1.yml:/workspace/RoboDojo/env_cfg/sim/sim_config.yml:ro" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 0; port=$((port+1))
    run_case n02_expected2 2 swap_blocks "-v $CFG/sim_config_nenv2.yml:/workspace/RoboDojo/env_cfg/sim/sim_config.yml:ro" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 0; port=$((port+1))
    run_case n04_expected4 4 swap_blocks "-v $CFG/sim_config_nenv4.yml:/workspace/RoboDojo/env_cfg/sim/sim_config.yml:ro" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 0; port=$((port+1))
    run_case n08_expected8 8 swap_blocks "-v $CFG/sim_config_nenv8.yml:/workspace/RoboDojo/env_cfg/sim/sim_config.yml:ro" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 60; port=$((port+1))
    run_case n09_expected9 9 swap_blocks "-v $CFG/sim_config_nenv9.yml:/workspace/RoboDojo/env_cfg/sim/sim_config.yml:ro" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 60; port=$((port+1))
    run_case n10_control_rep1 10 swap_blocks "" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 60; port=$((port+1))
    run_case n10_control_rep2 10 swap_blocks "" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 60; port=$((port+1))
elif [[ "$CASE_SET" == "controls" ]]; then
    # 先跑无 4K 材质的资产对照，降低 Xid 对后续 case 的污染。
    # cube_cushion/00001 是 12.6MB USDZ（含 4K JPG），00000 只有 5KB。
    # 两者都保持 Geometry 类型与同一布局，只改资产索引。
    run_case n10_cushion_small_rep1 10 swap_blocks "-v $CFG/swap_blocks_cushion_small.yml:/workspace/RoboDojo/task/RoboDojo/config/swap_blocks.yml:ro" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 60; port=$((port+1))
    run_case n10_cover_blocks 10 cover_blocks "" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 60; port=$((port+1))
    run_case n10_cube_mats 10 swap_blocks "-v $CFG/swap_blocks_cube_mats.yml:/workspace/RoboDojo/task/RoboDojo/config/swap_blocks.yml:ro" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 60; port=$((port+1))
    run_case n10_original_rep1 10 swap_blocks "" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 120; port=$((port+1))
    run_case n10_original_rep2 10 swap_blocks "" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 120; port=$((port+1))
    run_case n10_cushion_small_rep2 10 swap_blocks "-v $CFG/swap_blocks_cushion_small.yml:/workspace/RoboDojo/task/RoboDojo/config/swap_blocks.yml:ro" "$MEM_CKPT" 8192:8192 "$port" "$SEED" 60; port=$((port+1))
elif [[ "$CASE_SET" == "model" ]]; then
    # 排除模型/policy server 参与：崩溃若与记忆 ckpt 无关，这三项应与原配置同分布。
    run_case n10_base_ckpt 10 swap_blocks "" "$BASE_CKPT" 8192:8192 "$port" "$SEED" 120; port=$((port+1))
    run_case n10_image_default_nproc 10 swap_blocks "" "$MEM_CKPT" "" "$port" "$SEED" 120; port=$((port+1))
    run_case n10_seed1 10 swap_blocks "" "$MEM_CKPT" 8192:8192 "$port" 1 120; port=$((port+1))
elif [[ "$CASE_SET" == "focused" ]]; then
    # 已定位到 reload_env_scene(env0) -> relocate_stale_objects(articulation)
    # -> SpringButton_2_7.relocate_offscreen()。这里做最后的组内拆解。
    run_case n10_skip_reloc_rep1 10 swap_blocks "" "$MEM_CKPT" 8192:8192 "$port" 0 60 skip; port=$((port+1))
    run_case n10_no_app_update 10 swap_blocks "" "$MEM_CKPT" 8192:8192 "$port" 0 60 no_app_update; port=$((port+1))
    run_case n10_normal_detail 10 swap_blocks "" "$MEM_CKPT" 8192:8192 "$port" 0 120 normal; port=$((port+1))
    run_case n10_skip_reloc_rep2 10 swap_blocks "" "$MEM_CKPT" 8192:8192 "$port" 0 60 skip; port=$((port+1))
else
    echo "unknown CASE_SET=$CASE_SET" >&2
    exit 2
fi

echo "=== 全部结束 $(date +%H:%M:%S) ===" | tee -a "$OUT/summary.txt"
