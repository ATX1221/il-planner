"""Evaluate an existing checkpoint - no training.

The three challenges differ in who controls ego and how other agents behave:
  open_loop_boxes                ego replays the log; predictions are compared, never
                                 executed, so error cannot compound.
  closed_loop_nonreactive_agents ego tracks its own trajectory; other agents replay the
                                 log and ignore it. Error compounds.
  closed_loop_reactive_agents    same ego, but other agents run IDM and react.
"""
import argparse
import os
from pathlib import Path

# Optional core exclusion, e.g. IL_PLANNER_EXCLUDE_CPUS=10,11 to work around cores that
# miscompute. Child processes inherit this, and _run_simulation_patched.py re-applies it
# because ray resets worker affinity.
_FAULTY_CPUS = {int(c) for c in os.environ.get("IL_PLANNER_EXCLUDE_CPUS", "").split(",") if c.strip()}
if _FAULTY_CPUS:
    os.sched_setaffinity(0, set(os.sched_getaffinity(0)) - _FAULTY_CPUS)

REPO_DIR = Path(__file__).resolve().parent.parent
# Defaults assume this repo sits beside a nuplan-devkit checkout; override with env vars.
DEVKIT_DIR = os.environ.get("NUPLAN_DEVKIT_ROOT", str(REPO_DIR.parent / "nuplan-devkit"))
os.environ.setdefault("NUPLAN_DATA_ROOT", f"{DEVKIT_DIR}/nuplan/dataset")
os.environ.setdefault("NUPLAN_MAPS_ROOT", f"{DEVKIT_DIR}/nuplan/dataset/maps")

import subprocess
import sys

# Makes "_target_: il_planner.models.transformer_planner_single.TransformerPlanner" and the
# resolvable in the simulation subprocess, which is a fresh interpreter.
_THIS_DIR = str(Path(__file__).parent)
sys.path.insert(0, _THIS_DIR)
os.environ["PYTHONPATH"] = _THIS_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

EXP_DIR = os.environ.get("NUPLAN_EXP_ROOT", str(REPO_DIR / "exp"))

# The challenge name is BOTH the +simulation= bundle and the job_name that bundle sets,
# which is the folder the results land in (see the yamls in
# nuplan/planning/script/experiments/simulation/). Keeping them one string is what lets
# the output directory be found without re-parsing the config.
CHALLENGES = (
    "open_loop_boxes",
    "closed_loop_nonreactive_agents",
    "closed_loop_reactive_agents",
)


def find_checkpoint(experiment_name: str, job_name: str) -> Path:
    """
    Newest best_model/*.ckpt under exp/<experiment_name>/<job_name>/<timestamp>/.

    best_model/ holds the epoch that minimised val_loss, which is NOT the last epoch:
    at 20k scenarios tf_single's best was epoch 11 of 30 and tf_multi's was epoch 2 of 30, both
    overfitting well before the run ended. Falls back to the highest-numbered file in
    checkpoints/ if the ModelCheckpoint callback never wrote a best_model/.
    """
    job_dir = Path(EXP_DIR) / experiment_name / job_name
    if not job_dir.is_dir():
        raise SystemExit(
            f"No training runs found at {job_dir}\n"
            f"Train the model first with run_experiment_{job_name[-3:]}.py, or pass a "
            f"checkpoint path as an argument."
        )

    runs = sorted((d for d in job_dir.iterdir() if d.is_dir()), key=lambda p: p.name)
    for run in reversed(runs):                     # newest run first
        best = list((run / "best_model").glob("*.ckpt"))
        if best:
            return max(best, key=lambda p: p.stat().st_mtime)
        checkpoints = run / "checkpoints"
        if checkpoints.is_dir() and any(checkpoints.iterdir()):
            # Sort by epoch NUMBER - "epoch=10" sorts before "epoch=2" alphabetically.
            return max(checkpoints.iterdir(), key=lambda p: int(p.stem.split("=")[1]))

    raise SystemExit(f"Found {len(runs)} run(s) under {job_dir} but none contain a checkpoint.")


def simulate(experiment_name: str, job_name: str, model_config: str) -> None:
    """Parse the command line, then evaluate the checkpoint on each requested challenge."""
    parser = argparse.ArgumentParser(description=f"Simulate {model_config} without retraining.")
    parser.add_argument(
        "-c", "--challenge", default="open_loop_boxes", choices=(*CHALLENGES, "all"),
        help="which simulation to run (default: open_loop_boxes)",
    )
    parser.add_argument("--ckpt", default=None, help="checkpoint (default: newest best_model)")
    parser.add_argument(
        "--experiment", default=None,
        help="experiment folder to read the checkpoint from and write results into "
             "(default: this script's own). Use it to score an ablation run.",
    )
    parser.add_argument(
        "-o", "--model-override", action="append", default=[], metavar="KEY=VALUE",
        help="extra hydra override for the MODEL, repeatable. Simulation must build the "
             "same architecture the checkpoint was trained with, so an ablation trained "
             "with model.use_ego_history=false must be simulated with it too - otherwise "
             "load_state_dict fails on the missing _ego_state_encoder weights.",
    )
    parser.add_argument(
        "--scenarios", type=int, default=50,
        help="scenarios per type (default: 50 -> 631 total). Use a small number for a "
             "quick smoke test; closed loop is much slower than open loop.",
    )
    args = parser.parse_args()

    experiment_name = args.experiment or experiment_name
    checkpoint_path = (
        Path(args.ckpt).resolve() if args.ckpt else find_checkpoint(experiment_name, job_name)
    )
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint_path}")

    challenges = CHALLENGES if args.challenge == "all" else (args.challenge,)
    for challenge in challenges:
        _run_one(experiment_name, model_config, checkpoint_path, challenge, args.scenarios,
                 args.model_override)


def _run_one(
    experiment_name: str, model_config: str, checkpoint_path: Path, challenge: str, scenarios: int,
    model_overrides: list = (),
) -> None:
    """Run one challenge on one checkpoint and report the score."""
    print(f"\n===== {model_config}: {challenge} =====\n\ncheckpoint: {checkpoint_path}\n")

    sim_cmd = [
        sys.executable,
        str(Path(__file__).parent / "_run_simulation_patched.py"),
        f"experiment_name={experiment_name}",
        f"group='{EXP_DIR}'",
        f"+simulation={challenge}",
        # Overrides the placeholder planner each challenge bundle names (simple_planner
        # for the closed-loop ones, log_future_planner for open loop).
        "planner=ml_planner",
        # Simulation must build the SAME class the checkpoint was trained with, otherwise
        # load_state_dict fails on missing/mismatched keys. The model yamls live in
        # IL_Planner/config, so the searchpath below is what makes them discoverable.
        "hydra.searchpath=[pkg://nuplan.planning.script.config.common,"
        " pkg://nuplan.planning.script.experiments,"
        f" file://{REPO_DIR}/config]",
        f"model={model_config}",
        *model_overrides,
        "planner.ml_planner.model_config=${model}",
        f"planner.ml_planner.checkpoint_path='{checkpoint_path}'",
        # Replace the WHOLE filter, don't just set .scenario_types on it. The simulation
        # default is one_continuous_log, which pins log_names to a single 15-minute drive:
        # the 2026-08-25 tf_single run scored 241 scenarios that all came from that one log and
        # not one contained a turn above 10 degrees. This filter is the official benchmark
        # set and draws from all 64 logs - 631 scenarios, 155 of them real turns.
        "scenario_filter=nuplan_challenge_scenarios",
        f"scenario_filter.num_scenarios_per_type={scenarios}",
        # Default (null) spawns one worker per CPU core (32 here), each loading its own
        # full copy of torch + the model - that exhausted this machine's 30GB of RAM.
        "worker.threads_per_node=4",
    ]

    # Retried at the PROCESS level: the intermittent corruption on this machine can arrive
    # as a SIGSEGV, which kills the interpreter outright and cannot be caught in Python.
    for attempt in range(1, 6):
        if subprocess.run(sim_cmd, cwd=DEVKIT_DIR).returncode == 0:
            break
        print(f"  [simulation attempt {attempt}/5 failed] retrying...")
    else:
        raise SystemExit("Simulation failed 5 times in a row - not the intermittent fault.")

    # Each bundle sets job_name to its own name, which is the folder results land in.
    output_dir = max(
        (Path(EXP_DIR) / experiment_name / challenge).iterdir(),
        key=lambda p: p.stat().st_mtime,
    )
    print(f"\nSimulation output written to: {output_dir}\n")
    _print_score(output_dir)

    print("===== To view results, run =====\n")
    print(
        # PYTHONPATH is required, not optional. SimulationLog pickles the whole planner -
        # MLPlanner -> our model -> SimpleFeatureBuilder - so opening a scenario unpickles
        # it and needs this folder importable. nuBoard is a separate process and inherits
        # nothing from here; without it the metric tabs load and only the visualization
        # dies, with bokeh hiding the real ModuleNotFoundError behind a "coroutine was
        # never awaited" warning.
        f"PYTHONPATH={_THIS_DIR} \\\n"
        "python nuplan/planning/script/run_nuboard.py \\\n"
        f'    simulation_path="[{output_dir}]"\n'
    )


def _print_score(output_dir: Path) -> None:
    """Final weighted score plus a per-type breakdown, so turns can be read separately."""
    try:
        import pandas as pd

        aggregated = list((output_dir / "aggregator_metric").glob("*.parquet"))
        if aggregated:
            df = pd.read_parquet(aggregated[0])
            final = df[df["scenario"] == "final_score"] if "scenario" in df else df
            print(f"FINAL SCORE: {float(final['score'].iloc[0]):.4f}\n")

        # Per-type means. The headline score is a type-WEIGHTED aggregate, so these do not
        # average to it - they are here to show which types the model is losing on.
        frames = [
            pd.read_parquet(f)[["scenario_type", "scenario_name", "metric_score"]]
            for f in (output_dir / "metrics").glob("*.parquet")
            if {"scenario_type", "metric_score"} <= set(pd.read_parquet(f).columns)
        ]
        if frames:
            per_type = (
                pd.concat(frames)
                .groupby(["scenario_type", "scenario_name"])["metric_score"].mean()
                .groupby("scenario_type").agg(["mean", "count"])
                .sort_values("mean")
            )
            print("per-type mean metric score (worst first):")
            for scenario_type, row in per_type.iterrows():
                turn = "  <- TURNS" if any(
                    k in scenario_type for k in ("turn", "lateral", "changing_lane")
                ) else ""
                print(f"  {row['mean']:.3f}  n={int(row['count']):>3}  {scenario_type}{turn}")
            print()
    except Exception as error:                     # reporting must never fail the run
        print(f"(could not summarise metrics: {type(error).__name__}: {error})")
