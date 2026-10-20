#!/usr/bin/env bash
# Phase 1 巡检：队列进度 / 当前 run / 结果表 / 汇总（随时安全，只读）。
# 用法：bash scripts/pi05/phase1_status.sh [日志根目录]
#       默认取 logs/pi05/phase1_* 中最新的一个。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_ROOT="${1:-$(ls -dt "$ROOT"/logs/pi05/phase1_* 2>/dev/null | head -1)}"
if [[ -z "${RUN_ROOT:-}" || ! -d "$RUN_ROOT" ]]; then
    echo "找不到 phase1 日志目录"; exit 1
fi
echo "== Phase1 日志根目录: $RUN_ROOT =="
total=$(wc -l < "$RUN_ROOT/queue.txt" 2>/dev/null || echo 0)
claimed=$(find "$RUN_ROOT/claims" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
done_n=$(($(tail -n +2 "$RUN_ROOT/results.tsv" 2>/dev/null | wc -l)))
echo "队列: $total 个 run；已领取: $claimed；已产出结果行: $done_n"
echo
if [[ -s "$RUN_ROOT/results.tsv" ]]; then
    echo "--- results.tsv（最新在前） ---"
    (head -1 "$RUN_ROOT/results.tsv"; tail -n +2 "$RUN_ROOT/results.tsv" | sort -k7 -n -r) | head -25
    echo
    echo "--- 逐任务 score 均值 & 总均值（截至当前） ---"
    python3 - "$RUN_ROOT/results.tsv" <<'PY'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
by_task = defaultdict(list)
for r in rows:
    if r["score"]:
        by_task[r["task"]].append(float(r["score"]))
if not by_task:
    print("（暂无有效分数）"); raise SystemExit
means = []
for t, s in sorted(by_task.items()):
    m = sum(s)/len(s); means.append(m)
    print(f"  {t}\truns={len(s)}\t均值 {m:.2f}")
print(f"  合计: {sum(means)/len(means):.2f}（{len(means)}/6 任务）")
PY
else
    echo "（results.tsv 尚无结果行；进行中）"
fi
echo
echo "--- 当前活跃 run ---"
ps -eo pid,etime,args | grep -E "run_pi05_eval|phase_mem_pi05" | grep -v grep | head -3 | cut -c1-160
srun --jobid="${JOBID:?请设置 JOBID=<现有分配>}" --overlap --quiet bash -c 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader' 2>/dev/null
