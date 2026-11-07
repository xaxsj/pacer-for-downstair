#!/usr/bin/env bash
# Convert the KIT stair subset into a PACER motionlib pkl.
# Uses the existing PACER processing scripts; no PACER core files are modified.
set -euo pipefail

source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
export PYTHONPATH="$PACER_ROOT/script/shims:${PYTHONPATH:-}"

RAW=/data/ljx/datasets/amass_kit_raw
DB=/data/ljx/datasets/amass_kit_db
OUT=data/amass/pkls/amass_isaac_kit_stairs.pkl

rm -rf "$RAW" "$DB"
mkdir -p "$RAW" "$DB" data/amass/pkls

# process_amass_raw.py iterates over these sequence names; only KIT is populated.
SEQS=(
  ACCAD BMLmovi BioMotionLab_NTroje CMU DFaust_67 EKUT Eyes_Japan_Dataset
  HumanEva KIT MPI_HDM05 MPI_Limits MPI_mosh SFU SSM_synced TCD_handMocap
  TotalCapture Transitions_mocap BMLhandball DanceDB
)
for seq in "${SEQS[@]}"; do
  if [ "$seq" = "KIT" ]; then
    ln -s /data/ljx/datasets/KIT "$RAW/KIT"
  else
    mkdir -p "$RAW/$seq"
  fi
done

echo "== 1.3a raw AMASS db =="
python uhc/data_process/process_amass_raw.py --dir "$RAW" --out_dir "$DB"

if [ ! -f sample_data/amass_copycat_occlusion_v3.pkl ]; then
  # The v2 occlusion file marks KIT stair clips as irrecoverable; use an empty
  # occlusion map so the stair subset is not filtered out.
  python - <<'PY'
import joblib
joblib.dump({}, "sample_data/amass_copycat_occlusion_v3.pkl")
PY
fi

echo "== 1.3b processed AMASS db =="
python script/run_process_amass_db.py --path "$DB/amass_db_smplh.pt"

echo "== 1.3c PACER motionlib =="
python script/run_convert_amass_isaac.py \
  --in_file sample_data/amass_copycat_take5_train.pkl \
  --out_file "$OUT"

echo "== done: $OUT =="
ls -lh "$OUT"
