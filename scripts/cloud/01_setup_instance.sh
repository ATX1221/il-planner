#!/usr/bin/env bash
# Run ON the EC2 instance. Prepares disk, python env, devkit, and your IL_Planner code.
#
# IMPORTANT: on a default Ubuntu AMI the ROOT volume is only ~8 GB with ~3.7 GB free, and
# the nuplan conda env (PyTorch + CUDA libs) is 7-10 GB. So conda, pip caches, the devkit
# and the dataset ALL live on the big volume; $HOME only gets a symlink.
set -euo pipefail

DATA_ROOT=/data

echo "############ 1. disk ############"
lsblk
echo
# A disk is free only if NEITHER the disk NOR ANY of its partitions is mounted. Checking
# only the disk's own row is wrong: the root disk (nvme0n1) has an empty MOUNTPOINT on its
# own line because it is the PARTITIONS (nvme0n1p1 -> /) that carry the mounts. That bug
# made this script try to mkfs the root volume; mkfs refused, but do not rely on that.
CAND=""
for _d in $(lsblk -dno NAME,TYPE | awk '$2=="disk"{print $1}'); do
    if [ -z "$(lsblk -rno MOUNTPOINT "/dev/$_d" | tr -d '[:space:]')" ]; then
        CAND=$_d
        break
    fi
done
if [ -n "${CAND:-}" ]; then
    echo "==> found unmounted disk /dev/$CAND - formatting and mounting at $DATA_ROOT"
    sudo file -s "/dev/$CAND" | grep -q ext4 || sudo mkfs -t ext4 "/dev/$CAND"
    sudo mkdir -p "$DATA_ROOT"
    mountpoint -q "$DATA_ROOT" || sudo mount "/dev/$CAND" "$DATA_ROOT"
    grep -q "$DATA_ROOT" /etc/fstab || \
        echo "/dev/$CAND $DATA_ROOT ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
else
    echo "==> no separate volume found; using $DATA_ROOT on the root disk"
    sudo mkdir -p "$DATA_ROOT"
fi
sudo chown -R "$USER:$USER" "$DATA_ROOT"
df -h / "$DATA_ROOT"

echo "############ 2. packages ############"
sudo apt-get update -qq
sudo apt-get install -y -qq unzip wget aria2 git build-essential awscli

echo "############ 3. miniforge (on $DATA_ROOT, NOT root) ############"
export CONDA_PKGS_DIRS=$DATA_ROOT/conda_pkgs
export PIP_CACHE_DIR=$DATA_ROOT/pip_cache
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
if [ ! -d "$DATA_ROOT/miniforge3" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
    bash /tmp/mf.sh -b -p "$DATA_ROOT/miniforge3"
    rm -f /tmp/mf.sh
fi
# shellcheck disable=SC1091
source "$DATA_ROOT/miniforge3/etc/profile.d/conda.sh"

echo "############ 4. devkit (on $DATA_ROOT, symlinked into \$HOME) ############"
if [ ! -d "$DATA_ROOT/nuplan-devkit" ]; then
    git clone -q https://github.com/motional/nuplan-devkit.git "$DATA_ROOT/nuplan-devkit"
fi
[ -e "$HOME/nuplan-devkit" ] || ln -s "$DATA_ROOT/nuplan-devkit" "$HOME/nuplan-devkit"
cd "$DATA_ROOT/nuplan-devkit"
git checkout -q e924167 2>/dev/null || echo "(staying on default branch)"

echo "############ 5. your IL_Planner code ############"
tar xzf /tmp/ilplanner.tgz -C "$DATA_ROOT/nuplan-devkit"
ls "$DATA_ROOT/nuplan-devkit/IL_Planner" | head -5

echo "############ 6. conda env ############"
if ! conda env list | grep -q '^nuplan '; then
    conda env create -f environment.yml
fi
conda activate nuplan
pip install -q -e .

echo "############ 7. env vars ############"
grep -q 'NUPLAN_DATA_ROOT' "$HOME/.bashrc" || cat >> "$HOME/.bashrc" <<ENVEOF

# --- nuplan ---
export CONDA_PKGS_DIRS=$DATA_ROOT/conda_pkgs
export PIP_CACHE_DIR=$DATA_ROOT/pip_cache
source $DATA_ROOT/miniforge3/etc/profile.d/conda.sh
conda activate nuplan
export NUPLAN_DATA_ROOT=$DATA_ROOT/nuplan/dataset
export NUPLAN_MAPS_ROOT=$DATA_ROOT/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=$DATA_ROOT/nuplan/exp
export PYTHONPATH=$DATA_ROOT/nuplan-devkit/IL_Planner:$DATA_ROOT/nuplan-devkit:\$PYTHONPATH
ENVEOF
mkdir -p "$DATA_ROOT/nuplan/dataset" "$DATA_ROOT/nuplan/exp"

echo
echo "==================================================================="
df -h / "$DATA_ROOT"
echo "Setup done. Start a fresh shell so the env vars load:"
echo "    exec bash -l"
echo "Then put your signed download URLs in $DATA_ROOT/urls.txt and run:"
echo "    bash /tmp/02_download_dataset.sh"
echo "==================================================================="
