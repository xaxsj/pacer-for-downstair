#!/usr/bin/env bash
set -u
source /data/ljx/scripts/pacer_env.sh
cd "$PACER_ROOT"
for tag in release stage1 diverse; do
  for mesh in wide_stairs narrow_stairs; do
    for seed in 0 1; do
      st="experiment_result/validation/${tag}/${mesh}/seed_${seed}/states.pkl"
      [ -f "$st" ] || continue
      python script/diagnose_hook.py \
        --states "$st" --mesh "data/mesh/${mesh}.pkl" \
        --out "experiment_result/validation/${tag}_${mesh}_seed_${seed}_hook.csv" >/dev/null 2>&1 || true
    done
  done
done
python - <<'PY'
import csv
from pathlib import Path
rows=[]
for p in sorted(Path("experiment_result/validation").glob("*_hook.csv")):
    with open(p) as fh:
        for r in csv.DictReader(fh):
            r["file"]=p.name
            rows.append(r)
out=Path("experiment_result/validation/hook_summary.csv")
with open(out,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()) if rows else ["file"])
    w.writeheader(); w.writerows(rows)
print("hook rows",len(rows),"->",out)
# first hook/stall per file
first={}
for r in rows:
    k=r["file"]
    if k not in first: first[k]={"file":k,"first_hook":None,"first_stall":None,"min_gap":999}
    if r["hook"]=="1" and first[k]["first_hook"] is None: first[k]["first_hook"]=int(r["frame"])
    if r["stall"]=="1" and first[k]["first_stall"] is None: first[k]["first_stall"]=int(r["frame"])
    first[k]["min_gap"]=min(first[k]["min_gap"],float(r["gap"]))
for k in sorted(first): print(first[k])
PY
