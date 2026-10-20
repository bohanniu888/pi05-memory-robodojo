#!/usr/bin/env bash
# Phase 1：π0.5 RoboDojo memory 基线评测（6 任务 × 3 seeds × EVAL_NUM 集）。
#
# 编排模型对齐 DM0.5 的 scripts/phase3_memory.sh：
#   * 队列 = 6 个 memory 任务 × 3 个 seed，每行 "<task> <seed> <port>"
#   * claims/<行号> 原子领取（mkdir 互斥），支持多 worker / 多进程共享队列
#   * 每个 run 调用 scripts/pi05/run_pi05_eval.sh（host 起 π0.5 server + 容器起 client）
#   * 汇总 results.tsv（task/seed/rc/eval_time/score/SR/秒/result_json）+ 逐任务均值
#
# 为什么自己排队列：官方 `robodojo.sh benchmark --dimension memory` 假定 client
# 跑在本机原生环境；本集群必须把 client 放进容器，沿用"host server + 容器 client"模型。
#
# 关键约束（DM0.5 实测，别改）：
#   * policy server 与仿真 client 必须分卡（同卡必 OOM）。
#   * `podman system migrate` 全进程只在开头做一次，并发做会杀掉运行中的容器。
#
# 用法（在已分配 GPU 的作业内）：
#   srun --jobid=<jid> --overlap bash scripts/pi05/phase_mem_pi05.sh --eval-num 6 --seeds 0,1,2
#   多 worker（需 4 卡）：--policy-gpus 0,2 --env-gpus 1,3
#   共享队列（多进程续跑）：--log-dir logs/pi05/phase1_<ts>
#     —— 队列已存在则复用，results.tsv 追加；已完成的行用 claims 跳过。
#   DRY_RUN=1 只排期。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT" || exit 1

TASKS=(cover_blocks match_and_pick_from_conveyor swap_T press_by_number imitate_sorting_sequence swap_blocks)
SEEDS_SPEC="0,1,2"
EVAL_NUM_SPEC="6"
RUN_ROOT_OVERRIDE=""
POLICY_GPUS_SPEC="0"
ENV_GPUS_SPEC="1"
CKPT_SPEC="RoboDojo-sim-arx_x5-joint-2"
PORT_BASE=26200

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-gpus) POLICY_GPUS_SPEC="$2"; shift 2 ;;
        --env-gpus) ENV_GPUS_SPEC="$2"; shift 2 ;;
        --seeds) SEEDS_SPEC="$2"; shift 2 ;;
        --tasks) IFS=',' read -r -a TASKS <<< "$2"; shift 2 ;;
        --eval-num) EVAL_NUM_SPEC="$2"; shift 2 ;;
        --ckpt) CKPT_SPEC="$2"; shift 2 ;;
        --log-dir) RUN_ROOT_OVERRIDE="$2"; shift 2 ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

IFS=',' read -r -a SEEDS <<< "$SEEDS_SPEC"
IFS=',' read -r -a POLICY_GPUS <<< "$POLICY_GPUS_SPEC"
IFS=',' read -r -a ENV_GPUS <<< "$ENV_GPUS_SPEC"

if [[ "${#POLICY_GPUS[@]}" -ne "${#ENV_GPUS[@]}" ]]; then
    echo "[phase1] --policy-gpus 与 --env-gpus 数量必须一致" >&2
    exit 2
fi
WORKERS="${#POLICY_GPUS[@]}"
echo "[phase1] workers=$WORKERS  policy_gpus=${POLICY_GPUS_SPEC}  env_gpus=${ENV_GPUS_SPEC}"
echo "[phase1] tasks=${TASKS[*]}"
echo "[phase1] seeds=${SEEDS_SPEC}  eval_num=$EVAL_NUM_SPEC  ckpt=$CKPT_SPEC"

if [[ -n "$RUN_ROOT_OVERRIDE" ]]; then
    RUN_ROOT="$RUN_ROOT_OVERRIDE"
    SHARED=1
else
    RUN_ROOT="$ROOT/logs/pi05/phase1_$(date +%Y-%m-%d_%H-%M-%S)"
    SHARED=0
fi
mkdir -p "$RUN_ROOT"
QUEUE="$RUN_ROOT/queue.txt"
CLAIM_DIR="$RUN_ROOT/claims"
mkdir -p "$CLAIM_DIR"

if [[ "$SHARED" == "1" && -s "$QUEUE" ]]; then
    echo "[phase1] 共享模式：复用已有队列（$(wc -l < "$QUEUE") 行）"
else
    : > "$QUEUE"
    angle=0
    for seed in "${SEEDS[@]}"; do
        for task in "${TASKS[@]}"; do
            port=$((PORT_BASE + angle))
            echo "$task $seed $port" >> "$QUEUE"
            angle=$((angle + 1))
        done
    done
fi

echo "[phase1] 队列 $(wc -l < "$QUEUE") 个 run；日志根目录: $RUN_ROOT"
cat "$QUEUE" | nl | sed 's/^/    /'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[phase1] DRY_RUN=1，仅排期不执行"
    exit 0
fi

export XDG_RUNTIME_DIR="/tmp/xdg-$USER"
mkdir -p "$XDG_RUNTIME_DIR"
STORAGE="/tmp/podman-build-$USER"
podman --root "$STORAGE/root" --runroot "$STORAGE/run" --storage-driver overlay \
    system migrate >/dev/null 2>&1 || true
echo "[phase1] podman storage 已迁移（只此一次）"

RESULTS="$RUN_ROOT/results.tsv"
RESULTS_HEADER=$'task\tseed\trc\teval_time\tscore\tSR\t秒\tresult'
if [[ "$SHARED" == "1" ]]; then
    [[ -s "$RESULTS" ]] || printf '%s\n' "$RESULTS_HEADER" > "$RESULTS"
else
    printf '%s\n' "$RESULTS_HEADER" > "$RESULTS"
fi

claim_next() {
    local line_no=0 entry
    while IFS= read -r entry; do
        line_no=$((line_no + 1))
        if mkdir "$CLAIM_DIR/$line_no" 2>/dev/null; then
            echo "$entry"
            return 0
        fi
    done < "$QUEUE"
    return 1
}

worker_loop() {
    local worker="$1" entry task seed port
    local start elapsed result_json score sr eval_time
    while entry="$(claim_next)"; do
        read -r task seed port <<< "$entry"
        start=$(date +%s)
        echo "[phase1][w$worker] 开始 $task seed=$seed port=$port $(date +%H:%M:%S)"

        TASK="$task" SEED="$seed" EVAL_NUM="$EVAL_NUM_SPEC" PORT="$port" \
            IMAGE=robodojo:isaac6 CKPT="$CKPT_SPEC" \
            POLICY_GPU="${POLICY_GPUS[$worker]}" ENV_GPU="${ENV_GPUS[$worker]}" \
            CLIENT_TIMEOUT=21600 \
            bash scripts/pi05/run_pi05_eval.sh \
            > "$RUN_ROOT/${task}_seed${seed}_w${worker}.out" 2>&1
        rc=$?

        elapsed=$(( $(date +%s) - start ))
        # 只在本任务、本 ckpt 的目录里找，避免 2 worker 并发时抓到其它任务的最新 json（2026-10-06 实测错配）。
        result_json=$(find "robodojo/eval_result/RoboDojo/$task" -path "*ckpt_name=$CKPT_SPEC*" -name "_result.json" -newermt "@$start" 2>/dev/null | sort | tail -1)
        if [[ -n "$result_json" ]]; then
            read -r score sr eval_time <<< "$(python3 - "$result_json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(data.get("score", ""), data.get("success_rate", ""), data.get("eval_time", ""))
PY
)"
        else
            score=""; sr=""; eval_time=""
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$task" "$seed" "$rc" "$eval_time" "$score" "$sr" "$elapsed" "$result_json" >> "$RESULTS"
        echo "[phase1][w$worker] 完成 $task seed=$seed rc=$rc score=$score SR=$sr 用时 $(( elapsed / 60 ))min"
    done
}

for ((w = 0; w < WORKERS; w++)); do
    worker_loop "$w" &
done
wait

echo
echo "================= Phase 1 π0.5 Memory 基线汇总 ================="
printf 'task\tseed\trc\teval_time\tscore\tSR\t秒\tresult\n'
sort "$RESULTS"
echo
python3 - "$RESULTS" <<'PY'
import csv, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
by_task = defaultdict(list)
by_seed = defaultdict(list)
for row in rows:
    if row["score"]:
        by_task[row["task"]].append(float(row["score"]))
        by_seed[row["seed"]].append(float(row["score"]))

if not by_task:
    print("（没有成功产出分数的 run）")
    raise SystemExit(0)

print("任务\truns\tscore 均值")
means = []
for task, scores in sorted(by_task.items()):
    mean = sum(scores) / len(scores)
    means.append(mean)
    print(f"{task}\t{len(scores)}\t{mean:.2f}")

print()
print("seed\t runs\tscore 均值")
for seed, scores in sorted(by_seed.items()):
    print(f"{seed}\t{len(scores)}\t{sum(scores) / len(scores):.2f}")

print(f"\nπ0.5 Memory 基线 score 均值（{len(means)} 个任务）：{sum(means) / len(means):.2f}")
print("参考：DM0.5 Memory = 47.74；判据：若 π0.5 基线 ≥40 则项目终止")
PY
echo "[phase1] 全部结束 end=$(date -Is)"
