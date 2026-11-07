#!/usr/bin/env bash
# PACER training probe: same command as the crashed stage-1 run, but with
# per-iteration stats enabled so the log shows how far it got before dying.
#
# Does NOT modify any source file. The only writable artefact is a temporary
# copy of the train config (deleted at exit) and a separate probe output dir.
#
# usage: pacer_train_probe.sh <tag> <num_envs> <max_iterations> <gpu> <timeout_s>
set -uo pipefail

TAG="${1:-probe}"
NUM_ENVS="${2:-1024}"
MAX_IT="${3:-500}"
GPU="${4:-3}"
TMO="${5:-3600}"

source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"

LOG="/data/ljx/tmp/pacer_${TAG}.log"
OUT_DIR="output/exp/pacer_${TAG}"
CFG_SRC="pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage1.yaml"
CFG_TMP="pacer/data/cfg/train/rlg/_probe_${TAG}.yaml"

# mirror all output into $LOG so the local driver can poll and fetch it
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

export CUDA_VISIBLE_DEVICES="$GPU"
# 关键：SIGSEGV 时打印 Python 调用栈（不改任何源码）
export PYTHONFAULTHANDLER=1

mkdir -p "$OUT_DIR"
if [ ! -f "$OUT_DIR/Humanoid.pth" ]; then
  cp output/release/pacer/Humanoid.pth "$OUT_DIR/Humanoid.pth"
fi

# temp config: turn print_stats on, nothing else
python - "$CFG_SRC" "$CFG_TMP" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
out = []
n = 0
for ln in open(src, encoding="utf-8").read().splitlines():
    if ln.strip().startswith("print_stats:"):
        ind = ln[: len(ln) - len(ln.lstrip())]
        ln = ind + "print_stats: True"
        n += 1
    out.append(ln)
open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("temp config written: %s (print_stats patched: %d)" % (dst, n))
PY

echo "================ PROBE START $(date -Is) ================"
echo "tag=$TAG envs=$NUM_ENVS max_iter=$MAX_IT gpu=$GPU timeout=$TMO"
echo "out_dir=$OUT_DIR"
echo "--- GPU before ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "$GPU" 2>/dev/null || echo "nvidia-smi failed"
echo "--- checkpoint being resumed ---"
python - "$OUT_DIR/Humanoid.pth" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu")
print("epoch=%s frame=%s" % (ck.get("epoch"), ck.get("frame")))
PY
echo "========================================================="

stdbuf -oL -eL timeout "$TMO" python pacer/run.py \
  --task HumanoidPedestrianTerrain \
  --cfg_env pacer/data/cfg/pacer_stage1_stairs.yaml \
  --cfg_train "$CFG_TMP" \
  --motion_file data/amass/pkls/amass_isaac_train_stairs_flat.pkl \
  --network_path "$OUT_DIR" \
  --epoch -1 \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_IT" \
  --headless --no_log
RC=$?

echo ""
echo "================ PROBE END $(date -Is) ================"
echo "exit_code=$RC"
echo "--- GPU after ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "$GPU" 2>/dev/null || true
echo "--- kernel segfault lines (if readable) ---"
dmesg 2>/dev/null | grep -i "segfault\|trap" | tail -n 5 || echo "dmesg not readable"
echo "--- saved checkpoints in $OUT_DIR ---"
ls -l --time-style=full-iso "$OUT_DIR" 2>/dev/null | tail -n 8
rm -f "$CFG_TMP"
echo "======================================================"
