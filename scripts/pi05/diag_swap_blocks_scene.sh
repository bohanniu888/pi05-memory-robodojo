#!/usr/bin/env bash
# swap_blocks 第二轮：场景内容 / 并行 env 数二分（EVAL_NUM=1，GPU1，现用缓存）。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT" || exit 1
ulimit -u 8192 2>/dev/null || true
CKPT=${CKPT:-RoboDojo-sim-arx_x5-history-v2-lora-seed0}
CFG="$ROOT/logs/pi05/diag_cfg"
OUT="$ROOT/logs/pi05/diag_swap_blocks_scene_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "out=$OUT" | tee "$OUT/summary.txt"

run_case() {
    local name="$1" mounts="$2" port="$3"
    local t0; t0=$(date +%s)
    echo "=== [$name] mounts=${mounts:-none} port=$port 开始 $(date +%H:%M:%S)" | tee -a "$OUT/summary.txt"
    EXTRA_MOUNTS="$mounts" TASK=swap_blocks SEED=0 EVAL_NUM=1 PORT="$port" \
        POLICY_GPU=0 ENV_GPU=1 CKPT="$CKPT" CLIENT_TIMEOUT=2400 \
        bash scripts/pi05/run_pi05_eval.sh > "$OUT/${name}.out" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    local cl; cl=$(ls -t logs/pi05/p1_client_swap_blocks_s0_*.log 2>/dev/null | head -1)
    local physx="no"; grep -aq "Error code 700" "$OUT/${name}.out" && physx="yes"
    if [[ -n "$cl" ]] && grep -aq "Error code 700" "$cl"; then physx="yes"; fi
    local nenv="?"; [[ -n "$cl" ]] && nenv=$(grep -a "Number of environments" "$cl" | tail -1 | tr -dc '0-9')
    local res; res=$(find robodojo/eval_result/RoboDojo/swap_blocks -name _result.json -newermt "@$t0" 2>/dev/null | sort | tail -1)
    local score=""; if [[ -n "$res" ]]; then score=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('score'))" "$res" 2>/dev/null); fi
    printf '[%s] rc=%s 用时=%ss envs=%s physx700=%s score=%s result=%s\n' \
        "$name" "$rc" "$dt" "$nenv" "$physx" "${score:-}" "${res:-none}" | tee -a "$OUT/summary.txt"
}

# D：num_envs=1（缩放/并行相关？）
run_case t5_nenv1 "-v $CFG/sim_config_nenv1.yml:/workspace/RoboDojo/env_cfg/sim/sim_config.yml:ro" 26310
# E：10 envs，但三个 cube_cushion 垫子换成刚体 cube（垫子资产相关？）
run_case t6_cube_mats "-v $CFG/swap_blocks_cube_mats.yml:/workspace/RoboDojo/task/RoboDojo/config/swap_blocks.yml:ro" 26311
# F：num_envs=4（阈值/资源）
run_case t7_nenv4 "-v $CFG/sim_config_nenv4.yml:/workspace/RoboDojo/env_cfg/sim/sim_config.yml:ro" 26312

echo "=== 全部结束 $(date +%H:%M:%S) ===" | tee -a "$OUT/summary.txt"
