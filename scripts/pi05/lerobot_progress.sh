#!/usr/bin/env bash
# 查看 lerobot_v3.0（joint，120GB）后台下载进度。
# 用法：bash scripts/pi05/lerobot_progress.sh [format]
#   format 默认 lerobot_v3.0，可选 lerobot_v3.0_ee / demo / hdf5
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
FORMAT="${1:-lerobot_v3.0}"
case "$FORMAT" in
  lerobot_v3.0)    REMOTE_DIR="data/RoboDojo_lerobot_v30_video";    CACHE="$ROOT/robodojo/.cache/robodojo_data_modelscope_lerobot_v3.0_repo" ;;
  lerobot_v3.0_ee) REMOTE_DIR="data/RoboDojo_ee_lerobot_v30_video"; CACHE="$ROOT/robodojo/.cache/robodojo_data_modelscope_lerobot_v3.0_ee_repo" ;;
  demo)            REMOTE_DIR="data/demo";                          CACHE="$ROOT/robodojo/.cache/robodojo_data_modelscope_demo_repo" ;;
  *) echo "未知 format: $FORMAT"; exit 2 ;;
esac

python3 - "$REMOTE_DIR" "$CACHE" <<'PY'
import json, os, subprocess, sys, time
remote_dir, cache = sys.argv[1], sys.argv[2]
url = f"https://hf-mirror.com/api/datasets/RoboDojo-Benchmark/RoboDojo/tree/main/{remote_dir}?recursive=true"
files = [x for x in json.loads(subprocess.run(["curl", "-s", url], capture_output=True, text=True).stdout)
         if x["type"] == "file"]
total = sum(x.get("size", 0) for x in files)
done = sum(x.get("size", 0) for x in files
           if os.path.exists(cache + "/" + x["path"]) and os.path.getsize(cache + "/" + x["path"]) == x.get("size", 0))
print(f"{remote_dir}: {done/1e9:.1f}/{total/1e9:.1f} GB ({100*done/total:.1f}%)  文件 {sum(1 for x in files if os.path.exists(cache+'/'+x['path']) and os.path.getsize(cache+'/'+x['path'])==x.get('size',0))}/{len(files)}")
print("cache 目录占用:", subprocess.run(["du","-sm",cache],capture_output=True,text=True).stdout.split()[0], "MB")
PY
