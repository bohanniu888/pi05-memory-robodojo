#!/usr/bin/env bash
# Phase 8 Step 4 接力·第一棒：训练(<JOBID-TRAIN>)结束后取消 ab6 看护，释放本账号 GPU/CPU/内存配额。
# 提交：sbatch --dependency=afterok:<JOBID-TRAIN> scripts/pi05/p8_step4_pre.sbatch
#SBATCH --job-name=p8_step4_pre
#SBATCH --partition=gpu2node
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/pi05/p8_step4_pre_slurm_%j.log

set -uo pipefail
echo "[pre] host=$(hostname) start=$(date -Is)"
for j in $(squeue -u "$USER" -h -o "%i|%j" | awk -F'|' '$2=="p8_ab6"{print $1}'); do
  echo "[pre] scancel $j (p8_ab6)"
  scancel "$j"
done
sleep 15
echo "[pre] 当前作业："
squeue -u "$USER" -o "%.8i %.10T %.16j %b %C" || true
echo "[pre] done $(date -Is)"
