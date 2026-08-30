#!/usr/bin/env bash
# Run ON the instance, after 02_download_dataset.sh.
#
# Builds the feature cache with YOUR SimpleFeatureBuilder. This is CPU-only - no GPU and
# no G-instance quota required - so it can run while the quota request is still pending.
set -euo pipefail

DEVKIT=$HOME/nuplan-devkit
CACHE_DIR=${CACHE_DIR:-/data/nuplan/exp/cache}
PER_TYPE=${PER_TYPE:-2000}          # num_scenarios_per_type - the balanced filter
THREADS=${THREADS:-4}               # c6i.2xlarge has 8 vCPU but only 16 GB RAM; each ray
                                    # worker loads the map gpkg, so 8 workers OOMs. 4 is safe.
TRAINING=${TRAINING:-tf_multi_noego_balanced635k}
S3_BUCKET=${S3_BUCKET:-}            # e.g. s3://my-nuplan-bucket ; empty = skip upload

cd "$DEVKIT"

# The archives do NOT extract to the layout the docs show, and each one lands in its OWN
# directory: val.zip -> data/cache/val/, test.zip -> data/cache/test/, and each city
# archive its own. Verified on this instance 2026-08-29: 737 val logs under
# /data/nuplan/dataset/data/cache/val. Taking dirname of the first .db found would pick
# ONE split and silently ignore every other one - a quiet 5x data loss.
# So consolidate every .db into a single directory as SYMLINKS (free - no copying of
# 300 GB) and point data_root at that. get_db_filenames_from_load_path() just iterdir()s
# and filters on suffix == '.db', so symlinks are resolved normally.
DB_DIR=${DB_DIR:-$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/trainval}
mkdir -p "$DB_DIR"
linked=0
while IFS= read -r db; do
    ln -sf "$db" "$DB_DIR/$(basename "$db")" && linked=$((linked+1))
done < <(find "$NUPLAN_DATA_ROOT" -name '*.db' -not -path '*/maps/*' -not -path "$DB_DIR/*")
if [ "$linked" -eq 0 ]; then
    echo "No .db files found under $NUPLAN_DATA_ROOT - did 02_download_dataset.sh finish?" >&2
    exit 1
fi
echo "==> log DBs   : $DB_DIR  ($linked logs consolidated as symlinks)"
echo "==> cache to  : $CACHE_DIR"
echo "==> per type  : $PER_TYPE"
echo

python nuplan/planning/script/run_training.py \
    py_func=cache \
    +training="$TRAINING" \
    experiment_name=cache_build \
    group=/data/nuplan/exp \
    hydra.searchpath="[pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments, file://$DEVKIT/IL_Planner/config]" \
    scenario_builder=nuplan \
    scenario_builder.data_root="$DB_DIR" \
    scenario_filter=training_scenarios \
    scenario_filter.num_scenarios_per_type="$PER_TYPE" \
    cache.cache_path="$CACHE_DIR" \
    cache.force_feature_computation=false \
    worker=ray_distributed \
    worker.threads_per_node="$THREADS"

echo
echo "===== cache built ====="
echo "scenarios: $(find "$CACHE_DIR" -mindepth 3 -maxdepth 3 -type d | wc -l)"
du -sh "$CACHE_DIR"

if [ -n "$S3_BUCKET" ]; then
    echo
    echo "==> uploading cache to $S3_BUCKET/cache"
    aws s3 sync "$CACHE_DIR" "$S3_BUCKET/cache" --only-show-errors
    echo "done. The GPU box can now train straight from $S3_BUCKET/cache"
fi

cat <<MSG

Next:
  - Terminate this CPU instance once the cache is in S3. You do NOT need the 312 GB of
    .db files again; training reads only the cache.
  - On the GPU box (once the quota clears):

      aws s3 sync $S3_BUCKET/cache /data/nuplan/exp/cache
      python nuplan/planning/script/run_training.py \\
          py_func=train +training=$TRAINING \\
          experiment_name=tf_multi_noego_balanced635k \\
          group=/data/nuplan/exp \\
          hydra.searchpath="[pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments, file://\$HOME/nuplan-devkit/IL_Planner/config]" \\
          scenario_builder=nuplan scenario_builder.data_root=$DB_DIR \\
          scenario_filter=training_scenarios scenario_filter.num_scenarios_per_type=$PER_TYPE \\
          cache.cache_path=/data/nuplan/exp/cache \\
          cache.force_feature_computation=false \\
          lightning.trainer.params.max_epochs=30

    NOTE cache.force_feature_computation MUST stay false. It is the first term of an 'or'
    in compute_or_load_feature, so true recomputes every feature every epoch and turns a
    short run into an all-nighter.
MSG
