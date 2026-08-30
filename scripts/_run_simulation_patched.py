"""
Thin wrapper around nuplan's run_simulation.py that applies the torch.load weights_only
patch (see run_experiment_simple_vector.py for why it's needed) before running simulation.

This has to be a separate file, not just code inside run_experiment_simple_vector.py,
because the patch only affects the Python process it runs in - and simulation is launched
as its own fresh subprocess (also to avoid inheriting training's CUDA context, see
run_experiment_simple_vector.py). A fresh subprocess imports torch unpatched unless the
patch is applied at the very start of that subprocess's own script, which is what this
file does before handing off to the real run_simulation entrypoint.
"""
import os

# Re-applied here as well as in simulate.py: ray resets worker CPU affinity, so inheriting
# it from the parent process is not sufficient. See IL_PLANNER_EXCLUDE_CPUS there.
_excluded = {int(c) for c in os.environ.get("IL_PLANNER_EXCLUDE_CPUS", "").split(",") if c.strip()}
if _excluded:
    os.sched_setaffinity(0, set(os.sched_getaffinity(0)) - _excluded)

import torch

_original_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    # Simulation runs on CPU by default anyway (number_of_gpus_allocated_per_simulation: 0
    # in default_simulation.yaml). Forcing map_location here avoids torch trying to restore
    # the checkpoint's tensors onto GPU, which was segfaulting - deserializing CUDA tensor
    # storage while implicitly triggering CUDA init mid-load is a known native crash risk.
    kwargs.setdefault("map_location", "cpu")
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_trusted

from nuplan.planning.script.run_simulation import main

if __name__ == "__main__":
    main()
