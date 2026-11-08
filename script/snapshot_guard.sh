#!/usr/bin/env bash
# Snapshot guard: the trainer overwrites <stage>/Humanoid.pth on every save, so
# intermediate training records would otherwise be lost. Poll the file and copy
# each new version into <stage>/snapshots/, tagged with the current epoch.
set -u
cd /data/ljx/code/pacer-main
ROOT=${ROOT:-output/exp/pacer_curriculum}
SLEEP=${SLEEP:-30}
while true; do
  for d in "$ROOT"/stage*; do
    [ -d "$d" ] || continue
    f="$d/Humanoid.pth"
    [ -f "$f" ] || continue
    mkdir -p "$d/snapshots"
    stamp=$(stat -c %Y "$f")
    last="$d/snapshots/.last"
    if [ ! -f "$last" ] || [ "$(cat "$last")" != "$stamp" ]; then
      size_before=$(stat -c %s "$f")
      ep=$(grep -a "Ep:" "/data/ljx/tmp/curriculum_$(basename "$d").log" 2>/dev/null \
           | tail -1 | sed 's/.*Ep: \([0-9]*\).*/\1/')
      tmp="$d/snapshots/.tmp_${ep:-unknown}.pth"
      cp "$f" "$tmp" 2>/dev/null || continue
      size_after=$(stat -c %s "$tmp")
      if [ "$size_before" = "$size_after" ]; then
        mv "$tmp" "$d/snapshots/Humanoid_${ep:-unknown}.pth"
        echo "$stamp" > "$last"
        echo "[snapshot] $d/snapshots/Humanoid_${ep:-unknown}.pth ($size_after bytes)"
      else
        rm -f "$tmp"
      fi
    fi
  done
  sleep "$SLEEP"
done
