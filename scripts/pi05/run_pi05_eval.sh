#!/usr/bin/env bash
# Phase 1 执行器：π0.5 policy server（host，uv/openpi）+ RoboDojo 仿真 client（podman 容器）。
#
# 架构对齐 DM0.5 的 scripts/run_smoke_stack_bowls.sh：
#   * server 在 host 上跑（XPolicyLab/policy/Pi_05，uv 环境），绑定 0.0.0.0
#   * client 在容器内跑（Isaac Sim 6），--network host 直连 127.0.0.1:PORT
#   * server / client 必须分卡（同卡必 OOM，DM0.5 实测）
#
# 用法（在 GPU 作业内）：
#   TASK=cover_blocks SEED=0 EVAL_NUM=6 srun --jobid=<jid> --overlap bash scripts/pi05/run_pi05_eval.sh
# 可覆盖的环境变量：TASK/SEED/PORT/POLICY_GPU/ENV_GPU/EVAL_NUM/IMAGE/CKPT/CLIENT_TIMEOUT/TIMEOUT
set -uo pipefail
ulimit -u 8192 2>/dev/null || true
echo "[Phase1] host ulimit -u soft=$(ulimit -Su) hard=$(ulimit -Hu)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REPO="$ROOT/robodojo"
LOG_DIR="$ROOT/logs/pi05"
mkdir -p "$LOG_DIR"

TASK="${TASK:-cover_blocks}"
SEED="${SEED:-0}"
PORT="${PORT:-26200}"
POLICY_GPU="${POLICY_GPU:-0}"
ENV_GPU="${ENV_GPU:-1}"
EVAL_NUM="${EVAL_NUM:-6}"
IMAGE="${IMAGE:-robodojo:isaac6}"
CKPT="${CKPT:-RoboDojo-sim-arx_x5-joint-2}"
# Isaac Sim 6 在 app.close() 可能挂住；_result.json 在关闭前已写好，超时兜底。
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-7200}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-600}"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
SERVER_LOG="$LOG_DIR/p1_server_${TASK}_s${SEED}_${RUN_ID}.log"
CLIENT_LOG="$LOG_DIR/p1_client_${TASK}_s${SEED}_${RUN_ID}.log"
START_TS="$(date +%s)"

echo "==================================================================="
echo "[Phase1] host=$(hostname) start=$(date -Is)"
echo "[Phase1] task=$TASK seed=$SEED eval_num=$EVAL_NUM port=$PORT ckpt=$CKPT"
echo "[Phase1] policy_gpu=$POLICY_GPU env_gpu=$ENV_GPU image=$IMAGE"
echo "[Phase1] server_log=$SERVER_LOG"
echo "[Phase1] client_log=$CLIENT_LOG"
nvidia-smi --query-gpu=index,name,memory.used --format=csv

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
ENV_GPU_UUID="${GPU_UUIDS[$ENV_GPU]}"
if [[ -z "${ENV_GPU_UUID:-}" ]]; then
    echo "[Phase1] ✗ 取不到 GPU $ENV_GPU 的 UUID（可见卡数 ${#GPU_UUIDS[@]}）"
    exit 1
fi
echo "[Phase1] 仿真卡: 索引 $ENV_GPU → $ENV_GPU_UUID（容器内为 cuda:0）"

# ---------------------------------------------------------------- podman ---
export XDG_RUNTIME_DIR="/tmp/xdg-$USER"
mkdir -p "$XDG_RUNTIME_DIR"
STORAGE="/tmp/podman-build-$USER"
PODMAN=(podman --root "$STORAGE/root" --runroot "$STORAGE/run" --storage-driver overlay)

CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"
mkdir -p "$CACHE_ROOT/ov" "$CACHE_ROOT/warp" "$CACHE_ROOT/nvidia" "$CACHE_ROOT/pip"
PODMAN_NPROC="${PODMAN_NPROC-8192:8192}"
PODMAN_ULIMIT_ARGS=()
if [[ -n "$PODMAN_NPROC" ]]; then
    PODMAN_ULIMIT_ARGS+=(--ulimit nproc="$PODMAN_NPROC")
fi
echo "[Phase1] container RLIMIT_NPROC=${PODMAN_NPROC:-<image-default>}"

# ------------------------------------------------- 1) policy server (host) ---
echo
echo "=== [1/3] 启动 π0.5 policy server（host，GPU $POLICY_GPU，uv 环境）==="

source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$REPO" || exit 1
bash scripts/robodojo.sh server \
    --policy-dir XPolicyLab/policy/Pi_05 \
    --task "$TASK" \
    --ckpt "$CKPT" \
    --policy-env uv \
    --action-type joint \
    --seed "$SEED" \
    --policy-gpu "$POLICY_GPU" \
    --policy-port "$PORT" \
    --bind-host 0.0.0.0 \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[Phase1] 关闭 policy server (pid=$SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    # 包装进程被 kill 后，下层 python 会被 reparent 到 PID 1 并继续占显存
    # （2026-10-04 Phase 1 实测：每次泄漏 7.6GB）。这里按端口精确兜底收掉本 run 的 server。
    pkill -u "$USER" -f "policy/Pi_05/openpi/.venv/bin/python .*port=$PORT" 2>/dev/null || true
    sleep 1
    pkill -9 -u "$USER" -f "policy/Pi_05/openpi/.venv/bin/python .*port=$PORT" 2>/dev/null || true
}
trap cleanup EXIT

server_up="false"
for i in $(seq 1 "$SERVER_TIMEOUT"); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[Phase1] ✗ policy server 进程已退出，日志尾部："
        tail -n 30 "$SERVER_LOG"
        exit 1
    fi
    if timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$PORT" 2>/dev/null; then
        echo "[Phase1] ✓ policy server 已监听 0.0.0.0:$PORT（等待 ${i}s）"
        server_up="true"
        break
    fi
    if [[ $((i % 30)) -eq 0 ]]; then
        echo "[Phase1]   等待中 ${i}s，日志尾部：$(tail -n 1 "$SERVER_LOG" 2>/dev/null)"
    fi
    sleep 1
done
if [[ "$server_up" != "true" ]]; then
    echo "[Phase1] ✗ ${SERVER_TIMEOUT}s 内未监听端口，日志尾部："
    tail -n 30 "$SERVER_LOG"
    exit 1
fi
grep -E "Model loaded|policy server ready|listening|Using normalization|running on" "$SERVER_LOG" | tail -3

# ----------------------------------------------- 2) simulator client (容器) ---
echo
echo "=== [2/3] 启动仿真客户端（容器内，GPU $ENV_GPU）==="

MOUNTS=()
# 关键：把宿主仓库的 XPolicyLab 挂进容器，确保 client 用的是最新 π0.5 适配器
# （容器镜像里烘焙的是制作镜像时的旧快照，不一定含 Pi_05）。
MOUNTS+=(-v "$REPO/XPolicyLab:/workspace/RoboDojo/XPolicyLab:ro")
# 诊断用：EXTRA_MOUNTS 允许覆盖镜像内配置（如 env_cfg/sim/sim_config.yml、task/.../config/swap_blocks.yml）
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
    read -r -a _extra_mounts <<< "$EXTRA_MOUNTS"
    MOUNTS+=("${_extra_mounts[@]}")
fi
echo "[Phase1] extra_mounts=${EXTRA_MOUNTS:-none}"

DIAG_ENV_ARGS=()
if [[ "${DIAG_STAGES:-0}" == "1" ]]; then
    DIAG_ENV_ARGS+=(-e ROBODOJO_DIAG_STAGES=1)
    echo "[Phase1] diag_stages=1"
fi
if [[ -n "${ART_RELOC_MODE:-}" ]]; then
    DIAG_ENV_ARGS+=(-e ROBODOJO_DIAG_ART_RELOC_MODE="$ART_RELOC_MODE")
    echo "[Phase1] art_reloc_mode=$ART_RELOC_MODE"
fi
if [[ "${SHIM:-1}" == "1" ]]; then
    MOUNTS+=(-v "$REPO/docker/patch/sitecustomize.py:/opt/robodojo-patch/sitecustomize.py:ro")
fi

timeout "$CLIENT_TIMEOUT" "${PODMAN[@]}" run --rm \
    --device "nvidia.com/gpu=$ENV_GPU_UUID" \
    --network host \
    --ipc host \
    "${PODMAN_ULIMIT_ARGS[@]}" \
    -e ROBODOJO_MAX_BASH_RETRIES="${ROBODOJO_MAX_BASH_RETRIES:-2}" \
    -e ROBODOJO_FATAL_RESTART_COUNT="${ROBODOJO_FATAL_RESTART_COUNT:-0}" \
    -e ROBODOJO_RUN_ID="${RUN_ID}_${TASK}_s${SEED}" \
    "${MOUNTS[@]}" \
    "${DIAG_ENV_ARGS[@]}" \
    -v "$CACHE_ROOT/ov:/root/.cache/ov" \
    -v "$CACHE_ROOT/warp:/root/.cache/warp" \
    -v "$CACHE_ROOT/nvidia:/root/.cache/nvidia" \
    -v "$CACHE_ROOT/pip:/root/.cache/pip" \
    -v "$REPO/Assets:/workspace/RoboDojo/Assets:ro" \
    -v "$REPO/Assets:$REPO/Assets:ro" \
    -v "$REPO/eval_result:/workspace/RoboDojo/eval_result" \
    "$IMAGE" \
    bash scripts/robodojo.sh client \
        --task "$TASK" \
        --policy-name Pi_05 \
        --policy-host 127.0.0.1 \
        --policy-port "$PORT" \
        --ckpt "$CKPT" \
        --action-type joint \
        --seed "$SEED" \
        --eval-num "$EVAL_NUM" \
        --env-gpu 0 \
    >"$CLIENT_LOG" 2>&1
client_rc=$?
echo "[Phase1] 客户端退出码 rc=$client_rc（124 = 命中 ${CLIENT_TIMEOUT}s 超时）"
echo "[Phase1] 客户端日志尾部："
tail -n 20 "$CLIENT_LOG"

# ------------------------------------------------------------ 3) verify ---
echo
echo "=== [3/3] 验收：结果文件 ==="
result_json="$(find "$REPO/eval_result" -name "_result.json" -newermt "@$START_TS" 2>/dev/null | sort | tail -1)"
if [[ -z "$result_json" ]]; then
    echo "[Phase1] ✗ 未找到 _result.json"
    exit 3
fi
echo "[Phase1] ✓ _result.json: $result_json"
python3 - "$result_json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print("[Phase1] score=%s success_rate=%s eval_time=%s" % (
    data.get("score"), data.get("success_rate"), data.get("eval_time")))
PY
echo "[Phase1] 完成 end=$(date -Is) 用时 $(( $(date +%s) - START_TS ))s"
