#!/usr/bin/env bash
# Convert the 195 KIT stairs + 120 KIT flat locomotion clips into a balanced
# PACER motionlib, then merge with the base motion file.
set -euo pipefail

source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"

RAW=/data/ljx/datasets/amass_kit_raw
DB=/data/ljx/datasets/amass_kit_db
KIT_ALL=data/amass/pkls/amass_isaac_kit_all.pkl
MERGED=data/amass/pkls/amass_isaac_train_stairs_flat.pkl

echo "== raw AMASS db =="
python uhc/data_process/process_amass_raw.py --dir "$RAW" --out_dir "$DB"

echo "== processed AMASS db =="
python script/run_process_amass_db.py --path "$DB/amass_db_smplh.pt"

echo "== all KIT clips -> PACER motionlib =="
export PYTHONPATH="$PACER_ROOT/script/shims:$PACER_ROOT:${PYTHONPATH:-}"
python script/convert_all_kit_stairs.py \
  --in_file sample_data/amass_copycat_take5_train.pkl \
  --out_file "$KIT_ALL"

echo "== merge with base =="
python script/merge_kit_motionlib.py \
  --base sample_data/amass_isaac_standing_upright_slim.pkl \
  --stairs "$KIT_ALL" \
  --out "$MERGED" \
  --report data/amass/pkls/kit_all_report.json

echo "== verify =="
python script/verify_kit_motionlib.py --motion "$MERGED"
