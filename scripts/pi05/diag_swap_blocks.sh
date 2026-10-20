#!/usr/bin/env bash
# swap_blocks 崩溃根因判定：同一权重/同一任务，只改 env GPU 与 OV 缓存来源。
# 每组 EVAL_NUM=1（崩溃发生在首次 reset，与集数无关），记录 rc / PhysX 700 / 是否产出 _result.json。
#
# 用法（在已分配 GPU 的作业内）：
#   srun --jobid=<JOBID> --overlap --nodes=1 --ntasks=1 bash scripts/pi05/diag_swap_blocks.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT" || exit 1
ulimit -u 8192 2>/dev/null || true
CKPT=${CKPT:-RoboDojo-sim-arx_x5-history-v2-lora-seed0}
OUT="$ROOT/logs/pi05/diag_swap_blocks_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "out=$OUT" | tee "$OUT/summary.txt"

run_case() {
    local name="$1" env_gpu="$2" cache="$3" port="$4"
    mkdir -p "$cache/ov" "$cache/warp" "$cache/nvidia" "$cache/pip"
    local t0; t0=$(date +%s)
    echo "=== [$name] env_gpu=$env_gpu cache=$cache port=$port 开始 $(date +%H:%M:%S)" | tee -a "$OUT/summary.txt"

    CACHE_ROOT="$cache" TASK=swap_blocks SEED=0 EVAL_NUM=1 PORT="$port" \
        POLICY_GPU=0 ENV_GPU="$env_gpu" CKPT="$CKPT" CLIENT_TIMEOUT=2400 \
        bash scripts/pi05/run_pi05_eval.sh > "$OUT/${name}.out" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))

    local cl; cl=$(ls -t logs/pi05/p1_client_swap_blocks_s0_*.log 2>/dev/null | head -1)
    local physx="no"; grep -aq "Error code 700" "$OUT/${name}.out" && physx="yes"
    if [[ -n "$cl" ]] && grep -aq "Error code 700" "$cl"; then physx="yes"; fi
    local res; res=$(find robodojo/eval_result/RoboDojo/swap_blocks -name _result.json -newermt "@$t0" 2>/dev/null | sort | tail -1)

    printf '[%s] rc=%s 用时=%ss physx700=%s result=%s\n' \
        "$name" "$rc" "$dt" "$physx" "${res:-none}" | tee -a "$OUT/summary.txt"
    if [[ "$physx" == "yes" && -n "$cl" ]]; then
        { grep -a "NVRM\|Xid" "$cl" | head -2; } >> "$OUT/summary.txt" 2>/dev/null || true
    fi
}

# 控制组：GPU1 + 现用缓存（预期复现崩溃）
run_case t1_gpu1_cache_old 1 "$ROOT/cache" 26300
# A：换 env GPU 2（卡相关？）
run_case t2_gpu2_cache_old 2 "$ROOT/cache" 26301
# B：GPU1 + 全新 OV/Warp 缓存（缓存损坏？）
mkdir -p "$ROOT/cache_diag_fresh"
run_case t3_gpu1_cache_fresh 1 "$ROOT/cache_diag_fresh" 26302
# C：GPU2 + 全新缓存（组合确认）
run_case t4_gpu2_cache_fresh 2 "$ROOT/cache_diag_fresh" 26303

echo "=== 全部结束 $(date +%H:%M:%S) ===" | tee -a "$OUT/summary.txt"
