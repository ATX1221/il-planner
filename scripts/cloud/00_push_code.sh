#!/usr/bin/env bash
# RUN THIS LOCALLY, not on the instance.
#
# IL_Planner/ is untracked in git, and the cache is built with YOUR SimpleFeatureBuilder -
# cache_data() constructs the model to ask it for its feature builders - so the instance
# needs this code before it can cache anything.
set -euo pipefail

: "${HOST:?set HOST=ubuntu@<ec2-public-dns>}"
: "${KEY:?set KEY=/path/to/your-key.pem}"

LOCAL_DEVKIT="${LOCAL_DEVKIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/..}"

echo "==> packing IL_Planner + devkit source (excluding exp/, dataset/, caches)"
tar czf /tmp/ilplanner.tgz \
    -C "$LOCAL_DEVKIT" \
    --exclude='IL_Planner/exp' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    IL_Planner

echo "==> copying to $HOST"
scp -i "$KEY" -o StrictHostKeyChecking=accept-new /tmp/ilplanner.tgz "$HOST:/tmp/"
scp -i "$KEY" -o StrictHostKeyChecking=accept-new "$LOCAL_DEVKIT/IL_Planner/cloud/"*.sh "$HOST:/tmp/"

echo
echo "Done. Now ssh in and run the setup script:"
echo "    ssh -i $KEY $HOST"
echo "    bash /tmp/01_setup_instance.sh"
