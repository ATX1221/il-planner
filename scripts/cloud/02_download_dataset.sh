#!/usr/bin/env bash
# Run ON the instance, after 01_setup_instance.sh.
#
# The nuPlan bucket is PUBLICLY readable - no login, no signed URLs. Verified 2026-08-29:
# every Content-Length below matches the figure on the nuscenes.org download page, and the
# maps archive is byte-identical to the copy already on the laptop (971,557,640).
#
# Peak-disk strategy: download ONE archive, extract it, delete the zip, then move on.
# Peak = (everything extracted so far) + (the one zip in flight), so the archives are
# ordered LARGEST-ZIP-FIRST and maps last, which puts the biggest zip alongside the
# smallest amount of already-extracted data. Projected peak ~493 GB against 559 GB free.
set -euo pipefail

DATA_ROOT=${NUPLAN_DATA_ROOT:-/data/nuplan/dataset}
BASE="https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public"
EXPAND=1.7          # extracted bytes per zipped byte; measured 1.68 on the mini split

# name|zipped GiB  (order matters - see header)
ARCHIVES=(
    "nuplan-v1.1/nuplan-v1.1_val.zip|90.3"
    "nuplan-v1.1/nuplan-v1.1_test.zip|89.3"
    "nuplan-v1.1/nuplan-v1.1_train_boston.zip|35.5"
    "nuplan-v1.1/nuplan-v1.1_train_singapore.zip|32.6"
    "nuplan-v1.1/nuplan-v1.1_train_pittsburgh.zip|28.5"
    "nuplan-v1.1/nuplan-maps-v1.0.zip|0.9"
)
# DELIBERATELY EXCLUDED: nuplan-v1.1_train_vegas_*.zip (850 GB, 90% of the train split and
# the lowest turn-density per GB of any city - wide multi-lane stroads).

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

for entry in "${ARCHIVES[@]}"; do
    path=${entry%%|*}
    zipgb=${entry##*|}
    fname=$(basename "$path")

    echo
    echo "=============== $fname (${zipgb} GB zipped) ==============="
    if [ -f ".done_${fname}" ]; then
        echo "already extracted, skipping"
        continue
    fi

    # Guard: refuse to start if this archive cannot fit alongside its own expansion.
    freegb=$(df -BG --output=avail "$DATA_ROOT" | tail -1 | tr -dc '0-9')
    needgb=$(awk -v z="$zipgb" -v e="$EXPAND" 'BEGIN{printf "%.0f", z + z*e}')
    echo "free ${freegb} GB / need ~${needgb} GB for this archive"
    if [ "$freegb" -lt "$needgb" ]; then
        echo "NOT ENOUGH DISK. Stopping before $fname rather than filling the volume." >&2
        echo "Drop the test split (89 GB) from ARCHIVES, or grow the EBS volume." >&2
        exit 1
    fi

    # aria2c with 16 parallel connections, NOT wget. The bucket is in ap-northeast-1
    # (Tokyo) and this instance is in us-east-1, so a single TCP stream tops out around
    # 20 MB/s on the fat long-distance pipe. Measured 2026-08-29 on the 927 MB maps file:
    #   wget single stream  20.8 MB/s
    #   aria2c -x16        184.0 MB/s      <- 9x, turns 3.8 hours into ~25 minutes
    # --continue resumes a partial instead of restarting; falloc is instant on ext4.
    aria2c -x16 -s16 -k10M --continue=true --file-allocation=falloc \
           --console-log-level=warn --summary-interval=30 \
           -d "$DATA_ROOT" -o "$fname" "$BASE/$path"
    echo "--- extracting ---"
    unzip -q -o "$fname" -d "$DATA_ROOT"
    rm -f "$fname"
    touch ".done_${fname}"
    df -h "$DATA_ROOT" | tail -1
done

echo
echo "=============== consolidating log DBs ==============="
# get_db_filenames_from_load_path uses Path(load_path).iterdir() - FLAT, not recursive - so
# a data_root pointing at splits/trainval would silently miss splits/test. Hardlink every
# .db into one directory: same filesystem, so this costs no additional disk.
ALL="$DATA_ROOT/nuplan-v1.1/splits/all"
mkdir -p "$ALL"
count=0
while IFS= read -r db; do
    case "$db" in *"/splits/all/"*) continue ;; esac
    ln -f "$db" "$ALL/$(basename "$db")" 2>/dev/null || cp -n "$db" "$ALL/"
    count=$((count+1))
done < <(find "$DATA_ROOT" -name '*.db' -not -path '*/maps/*')

echo "hardlinked $count log DBs into $ALL"
echo
echo "=============== layout ==============="
find "$DATA_ROOT" -maxdepth 3 -type d | head -20
echo
echo "unique log DBs : $(find "$ALL" -name '*.db' | wc -l)"
echo "maps (.gpkg)   : $(find "$DATA_ROOT" -name '*.gpkg' | wc -l)"
du -sh "$DATA_ROOT"
df -h "$DATA_ROOT" | tail -1
echo
echo "Next:  DB_DIR=$ALL bash /tmp/03_build_cache.sh"
