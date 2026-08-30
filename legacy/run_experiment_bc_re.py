"""
Runs the full nuPlan workflow for BC_model_v0Re.BasicVectorMapMLP as one script instead of
three separate terminal commands: train -> simulate -> print the nuBoard command to view
results.

This does exactly what nuplan_framework.ipynb's notebook cells do - it builds a config
with hydra.compose(...) and calls each script's main() function directly, instead of
going through the command line.
"""
import os
from pathlib import Path

# This machine's CPU core 10 is faulty and silently computes wrong results: pinned tests
# gave 4 failures / 15 runs on cpu 10 vs 0 failures / 60 runs on cpus 0, 4, 20 and 26,
# and both hard crashes seen earlier reported "on cpu 10". The symptoms (SystemError:
# unknown opcode, ints where strings belong, random SIGSEGV) are corrupted computation,
# not a bug in this code. Core 11 is excluded too - it is core 10's hyperthread sibling,
# so both share the same faulty physical core. Child processes inherit this affinity.
_FAULTY_CPUS = {10, 11}
os.sched_setaffinity(0, set(os.sched_getaffinity(0)) - _FAULTY_CPUS)

os.environ.setdefault("NUPLAN_DATA_ROOT", "/home/ubuntu/nuplan-devkit/nuplan/dataset")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/home/ubuntu/nuplan-devkit/nuplan/dataset/maps")

import subprocess
import sys

# Make BC_model_v0.py (sitting in this folder) importable, so hydra can resolve
# "_target_: BC_model_v0Re.BasicVectorMapMLP". sys.path covers the in-process training step;
# PYTHONPATH is inherited by the simulation subprocess, which resolves the same _target_ in
# a fresh interpreter. The folder name containing a space is fine - only the module name
# itself has to be a valid Python identifier.
_THIS_DIR = str(Path(__file__).parent)
sys.path.insert(0, _THIS_DIR)
os.environ["PYTHONPATH"] = _THIS_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

import hydra
import torch

from nuplan.planning.script.run_training import main as main_train

# This environment has torch 2.8 (which defaults torch.load to weights_only=True for
# security) paired with a much older pytorch_lightning (1.3.8) whose checkpoint loading
# code predates that change and doesn't pass weights_only=False itself. That combination
# fails to load ANY checkpoint here, including ones we just trained ourselves in this
# same environment - which are fully trusted, so it's safe to restore the old default.
_original_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_trusted

def compose_with_retry(config_path, config_name, overrides, attempts=8):
    """
    hydra.compose() on this machine intermittently (~6% of process starts) fails with
    memory-corruption symptoms while parsing config yaml - SystemError: unknown opcode,
    TypeError on PyYAML's Reader.pointer, or a bogus yaml ParserError on a file that is
    provably well-formed (verified: git-clean, parses 15/15 standalone, compiles cleanly).
    It is not deterministic and not caused by the configs, so just retry it.
    """
    for attempt in range(1, attempts + 1):
        try:
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            hydra.initialize(config_path=config_path)
            return hydra.compose(config_name=config_name, overrides=overrides)
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"  [compose attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}] retrying...")


EXPERIMENT_NAME = "BC_model_v0Re_experiment"

# All training/simulation output (checkpoints, logs, metrics) is written under this
# folder instead of the default ~/nuplan/exp. Wrapped in quotes in the overrides below
# because the path contains a space ("IL planner personal").
EXP_DIR = "/home/ubuntu/nuplan-devkit/IL planner personal/exp"

# ---------------------------------------------------------------------------
# Step 1: Train
# ---------------------------------------------------------------------------
train_cfg = compose_with_retry(
    config_path="../nuplan/planning/script/config/training",
    config_name="default_training",
    overrides=[
        f"experiment_name={EXPERIMENT_NAME}",
        f"group='{EXP_DIR}'",
        "+training=training_simple_vector_model",
        # Swap in our own model. simple_vector_model.yaml is left untouched: it still
        # supplies the constructor arguments (hidden_size, num_output_features, the
        # trajectory samplings), and only the class they are passed to is redirected.
        # This works because BasicVectorMapMLP takes the same arguments as
        # VectorMapSimpleMLP - once that stops being true, write a real model yaml.
        "model._target_=BC_model_v0Re.BasicVectorMapMLP",
        # 20 scenarios (~16 training samples) was a smoke test - far too little to learn
        # anything, which is why the first run scored 0 on every metric.
        "scenario_filter.limit_total_scenarios=2000",
        # Default batch_size is 2, which is very small for 2000 samples.
        "data_loader.params.batch_size=32",
        # Feature caching runs through the same ray worker that OOM'd during simulation
        # (one process per CPU core = 32 here, each loading its own copy of torch).
        "worker.threads_per_node=8",
        # Disable kinematic_agent_augmentation (enabled by training_simple_vector_model).
        # It perturbs the ego pose then runs an ipopt/casadi solver to re-smooth the
        # trajectory; on degenerate scenarios (e.g. near-zero velocity) that solver hits a
        # divide-by-zero and raises SIGFPE, which is a hardware signal - so the try/except
        # RuntimeError in kinematic_agent_augmentation.py:78 cannot catch it and the whole
        # process dies. 20 scenarios never sampled a bad one; 2000 did.
        "data_augmentation=[]",
    ],
)

print("\n===== STEP 1: TRAINING =====\n")
main_train(train_cfg)

# Find the checkpoint that training just produced. One file per epoch is saved here,
# named "epoch=<N>.ckpt" (see ModelCheckpointAtEpochEnd.on_epoch_end in
# nuplan/planning/training/callbacks/checkpoint_callback.py). Sort by the epoch number
# itself, not alphabetically - "epoch=10.ckpt" would otherwise sort before "epoch=2.ckpt".
training_output_dir = Path(train_cfg.output_dir)
checkpoint_dir = training_output_dir / "checkpoints"
checkpoint_path = max(checkpoint_dir.iterdir(), key=lambda p: int(p.stem.split("=")[1]))
print(f"\nFound checkpoint (last epoch): {checkpoint_path}\n")

# ---------------------------------------------------------------------------
# Step 2: Simulate, using the checkpoint from step 1.
#
# Run as a separate OS process (not an in-process hydra.compose + main_simulation call
# like step 1) because training leaves an initialized CUDA context in this process, and
# simulation's default worker (ray_distributed) forks child processes on Linux. Forking a
# process with an active CUDA context is a well-known cause of native segfaults - it isn't
# something GlobalHydra.clear() can fix, since that only resets Hydra's config state, not
# the CUDA driver state. subprocess.run() starts a fresh process image instead of forking
# this one, so simulation never inherits training's CUDA context at all.
# ---------------------------------------------------------------------------
print("\n===== STEP 2: SIMULATION =====\n")
sim_cmd = [
    sys.executable,
    str(Path(__file__).parent / "_run_simulation_patched.py"),
    f"experiment_name={EXPERIMENT_NAME}",
    f"group='{EXP_DIR}'",
    "+simulation=open_loop_boxes",
    "planner=ml_planner",
    "model=simple_vector_model",
    # Same redirect as in training - the checkpoint holds BasicVectorMapMLP weights, so
    # the simulation side has to build the same class to load them into.
    "model._target_=BC_model_v0Re.BasicVectorMapMLP",
    "planner.ml_planner.model_config=${model}",
    f"planner.ml_planner.checkpoint_path='{checkpoint_path}'",
    "scenario_filter.num_scenarios_per_type=3",
    # Default (null) spawns one worker process per CPU core (32 here), each loading
    # its own full copy of torch + the model - that exhausted this machine's 30GB of
    # RAM well before the actual (small) simulation workload needed it.
    "worker.threads_per_node=4",
]
# Retried for the same reason as compose_with_retry - a crash here can also be the
# intermittent corruption (which can arrive as a SIGSEGV, killing the whole process,
# so it has to be retried at the process level rather than caught in Python).
for attempt in range(1, 6):
    result = subprocess.run(sim_cmd, cwd="/home/ubuntu/nuplan-devkit")
    if result.returncode == 0:
        break
    print(f"  [simulation attempt {attempt}/5 failed with exit {result.returncode}] retrying...")
else:
    raise RuntimeError("Simulation failed 5 times in a row - not the intermittent fault.")

# job_name for the open_loop_boxes bundle is "open_loop_boxes" (see
# nuplan/planning/script/experiments/simulation/open_loop_boxes.yaml), so the run's output
# folder is the newest timestamped subfolder under this path.
simulation_experiment_dir = Path(EXP_DIR) / EXPERIMENT_NAME / "open_loop_boxes"
simulation_output_dir = max(simulation_experiment_dir.iterdir(), key=lambda p: p.stat().st_mtime)
print(f"\nSimulation output written to: {simulation_output_dir}\n")

# ---------------------------------------------------------------------------
# Step 3: nuBoard - printed, not launched here, since it starts a long-running
# web server (blocking) meant to be opened interactively when you're ready.
# ---------------------------------------------------------------------------
print("===== STEP 3: To view results, run =====\n")
print(
    # PYTHONPATH is required, not optional. SimulationLog pickles the whole planner -
    # MLPlanner -> our model -> SimpleFeatureBuilder - so opening a scenario unpickles
    # it and needs this folder importable. nuBoard is a separate process and inherits
    # nothing from here, so without this the metric tabs load fine and only the
    # visualization dies, with bokeh hiding the real ModuleNotFoundError behind a
    # "coroutine was never awaited" warning.
    f"PYTHONPATH={_THIS_DIR} \\\n"
    "python nuplan/planning/script/run_nuboard.py \\\n"
    f'    simulation_path="[{simulation_output_dir}]"\n'
)
