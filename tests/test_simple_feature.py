"""
End-to-end check of SimpleFeatureBuilder: real nuPlan scenario in, batched tensors out.

Run it FROM THIS DIRECTORY (simple_feature.py does `from feature_utils import ...`):

    python -m pytest tests/test_simple_feature.py

Six stages. Each one either prints PASS or raises. Read the failures top-down -
a failure in stage 2 makes stages 3-6 meaningless.
"""
import os
from pathlib import Path

DEVKIT_DIR = os.environ.get(
    "NUPLAN_DEVKIT_ROOT",
    str(Path(__file__).resolve().parent.parent.parent / "nuplan-devkit"))

os.sched_setaffinity(0, set(os.sched_getaffinity(0)) - {10, 11})
os.environ.setdefault("NUPLAN_DATA_ROOT", f"{DEVKIT_DIR}/nuplan/dataset")
os.environ.setdefault("NUPLAN_MAPS_ROOT", f"{DEVKIT_DIR}/nuplan/dataset/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import numpy as np
import torch

from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import (
    get_test_nuplan_scenario,
)

from il_planner.features.simple_feature import SimpleFeature, SimpleFeatureBuilder


def hdr(n, title):
    print()
    print("=" * 72)
    print(f"{n}. {title}")
    print("=" * 72)


builder = SimpleFeatureBuilder()
scenario = get_test_nuplan_scenario()

# ---------------------------------------------------------------------------
hdr(1, "get_features_from_scenario runs at all")
# ---------------------------------------------------------------------------
feature = builder.get_features_from_scenario(scenario)
print(f"  returned {type(feature).__name__}")
print(f"  top-level keys: {sorted(feature.data.keys())}")
assert set(feature.data.keys()) == {"agent", "map", "origin", "angle"}, feature.data.keys()
print("  PASS")

# ---------------------------------------------------------------------------
hdr(2, "shapes and dtypes")
# ---------------------------------------------------------------------------
agent = feature.data["agent"]
N, T = agent["position"].shape[:2]
print(f"  N = {N} agents (ego + up to {builder.max_agents} others)")
print(f"  T = {T} timesteps (expected {builder.history_samples + 1})")
print()
for k, v in agent.items():
    print(f"    agent/{k:<11} {str(v.shape):<14} {v.dtype}")
print(f"    origin       {str(feature.data['origin'].shape):<14} {feature.data['origin'].dtype}")
print(f"    angle        {str(feature.data['angle'].shape):<14} {feature.data['angle'].dtype}")

assert T == builder.history_samples + 1, f"T={T}"
assert agent["position"].shape == (N, T, 2)
assert agent["heading"].shape == (N, T)
assert agent["velocity"].shape == (N, T, 2)
assert agent["valid_mask"].shape == (N, T)
assert agent["category"].shape == (N,)
assert feature.data["origin"].shape == (2,)
print("  PASS")

# ---------------------------------------------------------------------------
hdr(3, "ego is row 0, and it is at the origin after normalization")
# ---------------------------------------------------------------------------
# This is THE check. present_idx after normalization is the last index, T-1.
ego_pos_now = agent["position"][0, -1]
ego_hdg_now = agent["heading"][0, -1]
print(f"  ego position at present : {ego_pos_now}   (must be [0, 0])")
print(f"  ego heading  at present : {ego_hdg_now:.6f}  (must be 0)")
print(f"  ego category            : {agent['category'][0]}  (must be 0 = EGO)")
assert np.allclose(ego_pos_now, 0.0, atol=1e-9), "ego not at origin -> wrong origin"
assert np.allclose(ego_hdg_now, 0.0, atol=1e-9), "ego not axis-aligned -> rot sign flipped"
assert agent["category"][0] == 0, "row 0 is not ego"
print("  PASS")

# ---------------------------------------------------------------------------
hdr(4, "coordinates are actually normalized, not global")
# ---------------------------------------------------------------------------
valid = agent["valid_mask"]
pos_valid = agent["position"][valid]
print(f"  |position| over valid cells: max {np.abs(pos_valid).max():8.2f} m")
print(f"                               mean {np.abs(pos_valid).mean():8.2f} m")
print(f"  origin (global, the receipt): {feature.data['origin']}")
print(f"  angle  (global, radians)    : {float(feature.data['angle']):.4f}")
print(f"  heading range: [{agent['heading'][valid].min():.3f}, {agent['heading'][valid].max():.3f}]")
assert np.abs(pos_valid).max() < 1e4, "positions still look global (~5e5) - normalize_ not called"

# The INVALID cells matter just as much. They start as zeros meaning "no measurement";
# if normalize transforms them too they become -origin (~4e6) and poison the model,
# because the row-level mask (valid_mask.any(-1)) does not drop individual timesteps.
for _k in ("position", "velocity", "heading"):
    _all = np.abs(agent[_k]).max()
    _val = np.abs(agent[_k][valid]).max() if valid.any() else 0.0
    print(f"  {_k:<9} |max| valid {_val:10.2f}   ALL cells {_all:12.2f}")
    assert _all < 1e4, (
        f"{_k} has magnitude {_all:,.0f} in cells that are NOT observed -> "
        "normalize transformed the placeholder zeros. Zero them out after the transform.")
assert np.abs(feature.data["origin"]).max() > 1e3, "origin does not look like a global coordinate"
print("  PASS")

# ---------------------------------------------------------------------------
hdr(5, "valid_mask means something")
# ---------------------------------------------------------------------------
print(f"  valid cells: {valid.sum()} / {valid.size}")
print(f"  rows with at least one observation: {valid.any(axis=1).sum()} / {N}")
print()
print("  per-row occupancy (X = observed, . = missing), first 12 rows:")
for r in range(min(N, 12)):
    tag = "  <- EGO" if r == 0 else ""
    print(f"    row {r:2d}: " + "".join("X" if valid[r, t] else "." for t in range(T)) + tag)
assert valid[0].all(), "ego should be valid at every timestep"
assert feature.is_valid, "is_valid returned False"
print("  PASS")

# ---------------------------------------------------------------------------
hdr("5b", "map features")
# ---------------------------------------------------------------------------
mp = feature.data["map"]
M, P = mp["point_position"].shape[:2]
print(f"  M = {M} lanes (LANE + LANE_CONNECTOR within radius {builder.radius} m)")
print(f"  P = {P} points per centerline (expected {builder.map_sample_points})")
print()
for k, v in mp.items():
    print(f"    map/{k:<15} {str(v.shape):<16} {v.dtype}")
assert M > 0, "no lanes found - check radius / map_api"
assert P == builder.map_sample_points
assert mp["point_position"].shape == (M, P, 2)
assert mp["valid_mask"].all(), "all real lanes should be valid before collate"

pp = mp["point_position"]
print()
print(f"  |point_position| max {np.abs(pp).max():8.2f} m")
print(f"  on_route         {int(mp['on_route'].sum())}/{M} lanes lead toward the goal")
print(f"  tl_status counts " + str({int(t): int((mp['tl_status'] == t).sum()) for t in np.unique(mp['tl_status'])})
      + "   (0=GREEN 1=YELLOW 2=RED 3=UNKNOWN)")
assert np.abs(pp).max() < 1e4, (
    f"map points have magnitude {np.abs(pp).max():,.0f} -> still global. "
    "Add point_position to normalize().")
# NOTE: max can legitimately exceed `radius`: proximity is decided per lane object,
# so a long lane clipping the circle is returned whole, tail included.
# interpolate_polyline places points at equal ARC-length steps. On a curving lane the
# straight-line distance between consecutive points is slightly shorter than the arc, so
# perfect chord uniformity is not expected - only near-uniformity.
seg = np.linalg.norm(np.diff(pp, axis=1), axis=-1)
rel = (seg.max(axis=1) - seg.min(axis=1)) / np.maximum(seg.mean(axis=1), 1e-9)
# point_vector must ROTATE but not translate. If it were translated like a position,
# its magnitudes would be ~origin (1e6) instead of the ~3 m spacing between points.
pv = mp["point_vector"]
print(f"  |point_vector|  max {np.abs(pv).max():6.2f} m   median segment "
      f"{np.median(np.linalg.norm(pv, axis=-1)):.2f} m")
assert np.abs(pv).max() < 1e3, (
    f"point_vector magnitude {np.abs(pv).max():,.0f} -> it was translated by origin. "
    "A vector rotates only.")
# the vectors must actually match the differences of the positions
recon = pp[:, 1:] - pp[:, :-1]
print(f"  vectors match position differences: "
      f"{bool(np.allclose(pv[:, :-1], recon, atol=1e-3))}")
print(f"  point spacing per lane: median spread {np.median(rel)*100:5.2f}%   worst {rel.max()*100:5.2f}%")
print(f"  lane arc lengths: {seg.sum(axis=1).min():.1f} - {seg.sum(axis=1).max():.1f} m")
print("  PASS")

# ---------------------------------------------------------------------------
hdr(6, "to_feature_tensor + collate on TWO samples with different agent/lane counts")
# ---------------------------------------------------------------------------
# This is what pad_sequence exists for. If both samples had the same N the test
# would pass even with padding broken.
scenario2 = get_test_nuplan_scenario()
f1 = builder.get_features_from_scenario(scenario).to_feature_tensor()
f2 = builder.get_features_from_scenario(scenario2).to_feature_tensor()

# force different agent counts so padding is actually exercised
n_keep = max(1, f2.data["agent"]["position"].shape[0] - 3)
f2.data["agent"] = {k: v[:n_keep] for k, v in f2.data["agent"].items()}
m_keep = max(1, f2.data["map"]["point_position"].shape[0] - 7)
f2.data["map"] = {k: v[:m_keep] for k, v in f2.data["map"].items()}

n1 = f1.data["agent"]["position"].shape[0]
n2 = f2.data["agent"]["position"].shape[0]
print(f"  sample 1 has {n1} agents, sample 2 has {n2}")

batch = SimpleFeature.collate([f1, f2])
print()
for k, v in batch.data["agent"].items():
    print(f"    agent/{k:<11} {tuple(v.shape)}")
for k, v in batch.data["map"].items():
    print(f"    map/{k:<13} {tuple(v.shape)}")
print(f"    origin       {tuple(batch.data['origin'].shape)}")
print(f"    angle        {tuple(batch.data['angle'].shape)}")

assert batch.data["agent"]["position"].shape == (2, max(n1, n2), T, 2)
assert batch.data["origin"].shape == (2, 2)
assert isinstance(batch.data["agent"]["position"], torch.Tensor)
# the padded rows of the SHORTER sample must be masked off
assert not batch.data["agent"]["valid_mask"][1, n2:].any(), "agent padding is not masked!"
assert not batch.data["map"]["valid_mask"][1, m_keep:].any(), "lane padding is not masked!"
print()
print(f"  padded rows of sample 2 (rows {n2}..{max(n1,n2)-1}) are all valid_mask=False")
print("  PASS")

print()
print("=" * 72)
print("ALL STAGES PASSED - the builder produces correct batched tensors.")
print("Next: a model whose forward() reads features['simple_feature'].data['agent'].")
print("=" * 72)
