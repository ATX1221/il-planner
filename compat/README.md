# Backward-compatibility shims

`SimulationLog` pickles the planner, which stores module paths as they were at run time.
Runs produced before the package restructure reference the flat module names
(`simple_feature`, `BC_model_v2`, ...). To replay those in nuBoard, put this directory on
the path:

    PYTHONPATH=<repo>:<repo>/compat python nuplan/planning/script/run_nuboard.py ...

Not needed for new runs.
