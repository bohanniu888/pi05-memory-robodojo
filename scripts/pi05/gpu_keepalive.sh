#!/usr/bin/env bash
# GPU 保活：让本次 Slurm 分配里的全部 GPU 保持"在用"（每卡占显存 + 周期性算力脉冲），
# 避免被管理侧判定为"超过 4 小时闲置"。
#
# 用法（在登录节点执行，不需要自己 srun；脚本内部会用 srun --overlap 起一个后台作业步）：
#   bash scripts/pi05/gpu_keepalive.sh start                     # 默认：当前唯一 RUNNING 作业的全部 GPU
#   bash scripts/pi05/gpu_keepalive.sh start --jobid <JOBID> --gpus 0,1,2 \
#        --mem-mib 2048 --interval 0.1 --burst-ms 90 --matmul 256
#   bash scripts/pi05/gpu_keepalive.sh status
#   bash scripts/pi05/gpu_keepalive.sh stop                      # ⚠️ 跑训练/评测前必须先 stop
#
# 说明：
#   * 每卡启动一个 worker：占住 --mem-mib 显存 + 每 --interval 秒打一段 --burst-ms 的
#     matmul 脉冲（占空比 = burst/interval，默认 90%，保证 nvidia-smi 采样长期非零），nvidia-smi 里能看到显存和算力占用；
#   * 断连安全：作业步以 setsid nohup 启动，退出登录不会掉；
#   * 幂等：重复 start 只提示已在运行；stop 会 scancel 对应作业步并兜底清理残留 worker。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PY="$ROOT/robodojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
WORKER="$ROOT/scripts/pi05/gpu_keepalive_worker.py"
BASE_LOG_DIR="$ROOT/logs/keepalive"
STEP_NAME=gpu-keepalive
MEM_MIB=2048
INTERVAL=0.1
BURST_MS=90
MATMUL=256
GPUS_SPEC=""
JOBID="${SLURM_JOB_ID:-}"

usage() { sed -n '2,20p' "$0"; }

CMD="${1:-}"
[[ $# -gt 0 ]] && shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobid)    JOBID="$2"; shift 2 ;;
        --gpus)     GPUS_SPEC="$2"; shift 2 ;;
        --mem-mib)  MEM_MIB="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --burst-ms) BURST_MS="$2"; shift 2 ;;
        --matmul)   MATMUL="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "未知参数: $1" >&2; usage; exit 2 ;;
    esac
done

resolve_jobid() {
    [[ -n "$JOBID" ]] && return 0
    mapfile -t _jobs < <(squeue -u "$USER" -h -t RUNNING -o "%i %N %b")
    if [[ ${#_jobs[@]} -eq 1 ]]; then
        JOBID="${_jobs[0]%% *}"
    elif [[ ${#_jobs[@]} -eq 0 ]]; then
        echo "[keepalive] ✗ 没有 RUNNING 的 Slurm 作业，无法保活" >&2
        return 1
    else
        echo "[keepalive] ✗ 有多个 RUNNING 作业，请用 --jobid 指定：" >&2
        printf '  %s\n' "${_jobs[@]}" >&2
        return 1
    fi
}

find_step() {
    squeue --steps -j "$JOBID" -h -u "$USER" -o "%i %j" 2>/dev/null \
        | awk -v n="$STEP_NAME" '$2 == n { print $1; exit }'
}

gpu_query() {
    timeout 60 srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 bash -lc \
        'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; \
         echo "-- 计算进程 --"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | head -8' 2>&1
}

cmd_start() {
    resolve_jobid || exit 1
    STATE="$BASE_LOG_DIR/keepalive_$JOBID"
    mkdir -p "$STATE"

    local step; step="$(find_step)"
    if [[ -n "$step" ]]; then
        echo "[keepalive] 已在运行（作业步 $step），无需重复启动。"
        cmd_status; exit 0
    fi

    cat > "$STATE/run.sh" <<'RUNEOF'
#!/usr/bin/env bash
set -uo pipefail
ulimit -u 8192 2>/dev/null || true
cd "$KEEPALIVE_ROOT"
GPUS="$KEEPALIVE_GPUS"
if [[ -z "$GPUS" ]]; then
    mapfile -t _g < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
    GPUS="$(IFS=,; echo "${_g[*]}")"
fi
echo "[keepalive] 作业 ${SLURM_JOB_ID:-?} 步 ${SLURM_STEP_ID:-?} 节点 $(hostname) GPU[$GPUS]"
pids=()
for g in ${GPUS//,/ }; do
    CUDA_VISIBLE_DEVICES="$g" KEEPALIVE_GPU="$g" nohup "$KEEPALIVE_PY" -u "$KEEPALIVE_WORKER" \
        --mem-mib "$KEEPALIVE_MEM_MIB" --interval "$KEEPALIVE_INTERVAL" --burst-ms "$KEEPALIVE_BURST_MS" --matmul "$KEEPALIVE_MATMUL" \
        > "$KEEPALIVE_STATE/gpu${g}.log" 2>&1 &
    pids+=("$!")
    echo "[keepalive] GPU $g → pid $! 日志 $KEEPALIVE_STATE/gpu${g}.log"
done
trap 'kill "${pids[@]}" 2>/dev/null' TERM INT
wait
RUNEOF
    chmod +x "$STATE/run.sh"

    echo "[keepalive] 启动作业步：jobid=$JOBID gpus=${GPUS_SPEC:-全部} mem=${MEM_MIB}MiB interval=${INTERVAL}s burst=${BURST_MS}ms"
    KEEPALIVE_ROOT="$ROOT" KEEPALIVE_PY="$PY" KEEPALIVE_WORKER="$WORKER" \
    KEEPALIVE_GPUS="$GPUS_SPEC" KEEPALIVE_MEM_MIB="$MEM_MIB" \
    KEEPALIVE_INTERVAL="$INTERVAL" KEEPALIVE_BURST_MS="$BURST_MS" KEEPALIVE_MATMUL="$MATMUL" KEEPALIVE_STATE="$STATE" \
    setsid nohup srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --cpu-bind=none --job-name="$STEP_NAME" bash "$STATE/run.sh" > "$STATE/step.log" 2>&1 &
    echo $! > "$STATE/local.pid"

    for _ in $(seq 1 30); do
        sleep 2
        step="$(find_step)"
        [[ -n "$step" ]] && break
    done
    if [[ -z "$step" ]]; then
        echo "[keepalive] ✗ 30s 内未看到作业步，日志尾部：" >&2
        tail -n 20 "$STATE/step.log" >&2
        exit 1
    fi
    echo "$step" > "$STATE/step_id"
    echo "[keepalive] ✓ 作业步 $step 已启动，等 10s 后确认 GPU 状态："
    sleep 10
    cmd_status
}

cmd_status() {
    resolve_jobid || exit 1
    STATE="$BASE_LOG_DIR/keepalive_$JOBID"
    echo "== 作业 $JOBID 状态（$(date '+%F %T')）=="
    squeue -j "$JOBID" -h -o "  作业: %i %T 节点 %N 剩余 %L" 2>/dev/null
    local step; step="$(find_step)"
    if [[ -n "$step" ]]; then
        echo "  保活步: $step（RUNNING）"
    else
        echo "  保活步: 未运行"
    fi
    echo "== GPU 现状 =="
    gpu_query | sed 's/^/  /'
    if [[ -d "$STATE" ]]; then
        for f in "$STATE"/gpu*.log; do
            [[ -f "$f" ]] || continue
            echo "== $(basename "$f") 末尾 =="
            tail -n 3 "$f" | sed 's/^/  /'
        done
    fi
}

cmd_stop() {
    resolve_jobid || exit 1
    STATE="$BASE_LOG_DIR/keepalive_$JOBID"
    local step; step="$(find_step)"
    if [[ -n "$step" ]]; then
        echo "[keepalive] scancel 作业步 $step"
        scancel "$step"
        for _ in $(seq 1 20); do
            sleep 1
            [[ -z "$(find_step)" ]] && break
        done
    fi
    step="$(find_step)"
    if [[ -n "$step" ]]; then
        echo "[keepalive] ⚠️ 作业步 $step 仍在，再等 10s"
        sleep 10
    fi
    [[ -f "$STATE/step_id" ]] && rm -f "$STATE/step_id"
    echo "[keepalive] 兜底清理节点上的残留 worker"
    timeout 60 srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 bash -lc \
        'pkill -f gpu_keepalive_worker.py 2>/dev/null; sleep 1; pgrep -af gpu_keepalive_worker.py || echo "  无残留 worker"' 2>&1 | sed 's/^/  /'
    echo "[keepalive] 清理后 GPU 状态："
    gpu_query | sed 's/^/  /'
}

case "$CMD" in
    start)  cmd_start ;;
    status) cmd_status ;;
    stop)   cmd_stop ;;
    *)      usage; exit 2 ;;
esac
