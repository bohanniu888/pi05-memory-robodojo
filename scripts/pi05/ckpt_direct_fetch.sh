#!/usr/bin/env bash
# 0A 加速器：绕过 git-lfs，直接从 hf-mirror 并行预取某个 seed 的 59999 checkpoint。
# 用途：主线 git-lfs 下载较慢时，先拿到一个可用于 0D/0E 的 seed。
#
# 用法（在 GPU 作业内）：
#   bash scripts/pi05/ckpt_direct_fetch.sh [seed] [dest_root]
# 默认 seed=2，落点 policy/Pi_05/checkpoints/RoboDojo-sim-arx_x5-joint-<seed>/59999/
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SEED="${1:-2}"
DEST="${2:-$ROOT/robodojo/XPolicyLab/policy/Pi_05/checkpoints/RoboDojo-sim-arx_x5-joint-${SEED}/59999}"
BASE="https://hf-mirror.com/datasets/RoboDojo-Benchmark/RoboDojo/resolve/main"
LIST=/tmp/seed${SEED}_files.txt

mkdir -p "$DEST"
curl -s "https://hf-mirror.com/api/datasets/RoboDojo-Benchmark/RoboDojo/tree/main/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-${SEED}/59999?recursive=true" \
  | python3 -c "
import json,sys
for x in json.load(sys.stdin):
    if x['type']=='file':
        print(x['size'], x['path'].split('59999/')[-1])
" > "$LIST"
echo "[0A-direct] seed=$SEED files=$(wc -l < "$LIST") dest=$DEST"
export DEST BASE SEED
xargs -a "$LIST" -P6 -n2 bash -c '
  size="$1"; rel="$2"; out="$DEST/$rel"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out" && "$(stat -c%s "$out")" == "$size" ]]; then echo "SKIP $rel"; exit 0; fi
  curl -sL --retry 3 --retry-delay 2 -o "$out.part" \
    "$BASE/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-${SEED}/59999/$rel" \
    && mv "$out.part" "$out" && echo "OK $rel $(stat -c%s "$out")/$size" || echo "FAIL $rel"
' _
python3 - "$DEST" "$LIST" <<'PY'
import os, sys
dest, lst = sys.argv[1], sys.argv[2]
ok = bad = 0
for line in open(lst):
    size, rel = line.strip().split(' ', 1)
    p = os.path.join(dest, rel)
    if os.path.exists(p) and os.path.getsize(p) == int(size):
        ok += 1
    else:
        bad += 1
        print("BAD", rel)
print(f"[0A-direct] complete={ok} bad={bad}")
sys.exit(1 if bad else 0)
PY
