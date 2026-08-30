"""
Smoke test: real scenario -> feature -> collate -> model.forward -> Trajectory.

    cd /home/ubuntu/nuplan-devkit/IL_Planner && python test_model_v1.py

Catches shape/dtype bugs in seconds instead of minutes into a hydra training run.
"""
import os

os.sched_setaffinity(0, set(os.sched_getaffinity(0)) - {10, 11})
os.environ.setdefault("NUPLAN_DATA_ROOT", "/home/ubuntu/nuplan-devkit/nuplan/dataset")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/home/ubuntu/nuplan-devkit/nuplan/dataset/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import (
    get_test_nuplan_scenario,
)

from simple_feature import SimpleFeature, SimpleFeatureBuilder
from BC_model_v1 import BasicVectorMapMLP

FUTURE = TrajectorySampling(num_poses=16, time_horizon=8.0)
HIDDEN = 128
NUM_OUT = FUTURE.num_poses * 3          # x, y, heading per pose

print("=" * 70)
print("1. build model")
print("=" * 70)
builder = SimpleFeatureBuilder()
model = BasicVectorMapMLP(
    num_output_features=NUM_OUT,
    hidden_size=HIDDEN,
    future_trajectory_sampling=FUTURE,
    feature_builder=builder,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"  T                = {model.T}")
print(f"  agent_mlp_dim    = {model.agent_mlp_dim}  (= T * {SimpleFeature.agent_feature_dim()})")
print(f"  num_output_feats = {NUM_OUT}")
print(f"  parameters       = {n_params:,}")
print("  PASS")

print()
print("=" * 70)
print("2. build a batch of 2 with different agent counts")
print("=" * 70)
scenario = get_test_nuplan_scenario()
f1 = builder.get_features_from_scenario(scenario).to_feature_tensor()
f2 = builder.get_features_from_scenario(scenario).to_feature_tensor()
n_keep = max(1, f2.data["agent"]["position"].shape[0] - 5)
f2.data["agent"] = {k: v[:n_keep] for k, v in f2.data["agent"].items()}

batch = SimpleFeature.collate([f1, f2])
for k, v in batch.data["agent"].items():
    print(f"    agent/{k:<11} {str(tuple(v.shape)):<20} {v.dtype}")
assert batch.data["agent"]["position"].dtype == torch.float32, \
    f"expected float32, got {batch.data['agent']['position'].dtype} -> add .astype(np.float32) in normalize"
print("  PASS")

print()
print("=" * 70)
print("3. forward pass")
print("=" * 70)
model.eval()
with torch.no_grad():
    out = model({"simple_feature": batch})

traj = out["trajectory"]
print(f"  returned keys : {list(out.keys())}")
print(f"  trajectory    : {tuple(traj.data.shape)}   (expect (2, {FUTURE.num_poses}, 3))")
assert traj.data.shape == (2, FUTURE.num_poses, 3), traj.data.shape
assert torch.isfinite(traj.data).all(), "non-finite output - masked_fill value leaked through the max"
print(f"  finite        : yes")
print(f"  sample 0 first 3 poses:\n{traj.data[0, :3]}")
print("  PASS")

print()
print("=" * 70)
print("4. backward pass (gradients reach every parameter)")
print("=" * 70)
model.train()
out = model({"simple_feature": batch})
loss = out["trajectory"].data.square().mean()
loss.backward()
missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
print(f"  loss = {loss.item():.4f}")
if missing:
    print("  FAIL - no/NaN gradient for:")
    for n in missing:
        print(f"    {n}")
    raise SystemExit(1)
print(f"  all {sum(1 for _ in model.parameters())} parameter tensors got finite gradients")
print("  PASS")

print()
print("=" * 70)
print("ALL PASSED - model consumes SimpleFeature end to end.")
print("Next: wire it into a hydra training run.")
print("=" * 70)
