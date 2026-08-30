"""Train a model from a config in config/training/.

    python scripts/train.py tf_multi_noego_balanced635k --cache /path/to/cache
    python scripts/train.py tf_single_ego_balanced635k --epochs 20

Reads only from the feature cache (cache.use_cache_without_dataset=true), so the trainval
.db files are not needed on this machine.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Optional core exclusion, e.g. IL_PLANNER_EXCLUDE_CPUS=10,11 to work around cores that
# miscompute. Unset on most machines, in which case this is a no-op.
_faulty = {int(c) for c in os.environ.get("IL_PLANNER_EXCLUDE_CPUS", "").split(",")
           if c.strip()} & set(os.sched_getaffinity(0))
if _faulty:
    os.sched_setaffinity(0, set(os.sched_getaffinity(0)) - _faulty)

REPO = Path(__file__).resolve().parent.parent
DEVKIT = REPO.parent

p = argparse.ArgumentParser()
p.add_argument("config", help="name in config/training/, without .yaml")
p.add_argument("--cache", default=os.environ.get("IL_CACHE", str(Path.home() / "nuplan/exp/cache")))
p.add_argument("--exp", default=str(REPO / "exp"))
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--workers", type=int, default=10)
p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
a = p.parse_args()

os.environ.setdefault("NUPLAN_DATA_ROOT", str(DEVKIT / "nuplan/dataset"))
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(DEVKIT / "nuplan/dataset/maps"))
os.environ.setdefault("NUPLAN_EXP_ROOT", str(Path.home() / "nuplan/exp"))
os.environ["PYTHONPATH"] = f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

cmd = [
    sys.executable, str(DEVKIT / "nuplan/planning/script/run_training.py"),
    "py_func=train", f"+training={a.config}",
    f"experiment_name={a.config}", f"group={a.exp}",
    f"hydra.searchpath=[pkg://nuplan.planning.script.config.common,"
    f" pkg://nuplan.planning.script.experiments, file://{REPO}/config]",
    f"cache.cache_path={a.cache}",
    "cache.use_cache_without_dataset=true",
    # Must stay false: it is the first term of an `or` in compute_or_load_feature, so true
    # recomputes every feature every epoch.
    "cache.force_feature_computation=false",
    # val_loss = reg + cls, and cls legitimately rises as modes specialise, so it selects
    # far too early. Displacement error is the honest criterion.
    "lightning.trainer.checkpoint.monitor=metrics/val_avg_displacement_error",
    "lightning.trainer.checkpoint.mode=min",
    f"lightning.trainer.params.max_epochs={a.epochs}",
    "lightning.trainer.params.gpus=1",
    "lightning.trainer.params.accelerator=null",
    f"data_loader.params.batch_size={a.batch_size}",
    f"data_loader.params.num_workers={a.workers}",
    # Calls the model at each epoch end and pushes outputs through to_device(); our models
    # return raw Tensors, so it raises and kills the run before a checkpoint is written.
    "~callbacks.visualization_callback",
    *a.override,
]
raise SystemExit(subprocess.run(cmd, cwd=DEVKIT).returncode)
