"""
End-to-end smoke test:
1. Load a real scenario from the local nuPlan mini dataset (real map + real logged data).
2. Run VectorMapFeatureBuilder and AgentsFeatureBuilder against it (tests the feature-building
   side of the "entire nuplan stack" - map API, scenario DB, etc.).
3. Feed the resulting features into BasicVectorMapMLP.forward() (tests BC_model_v0 runs end-to-end).

This does not train anything - it just checks nothing crashes and shapes make sense.
"""
import os

# Point the devkit at the local copy of the nuPlan mini dataset + maps.
os.environ.setdefault("NUPLAN_DATA_ROOT", "/home/ubuntu/nuplan-devkit/nuplan/dataset")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/home/ubuntu/nuplan-devkit/nuplan/dataset/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import torch

from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder

from BC_model_v0 import BasicVectorMapMLP

past_trajectory_sampling = TrajectorySampling(num_poses=4, time_horizon=1.5)
future_trajectory_sampling = TrajectorySampling(num_poses=12, time_horizon=6)

print("Loading a real scenario from the local mini dataset...")
scenario = get_test_nuplan_scenario()
print(f"Loaded scenario: {scenario.scenario_name} (log: {scenario.log_name})")

print("\nBuilding VectorMap features from the real map...")
vector_map_builder = VectorMapFeatureBuilder(radius=20)
vector_map_feature = vector_map_builder.get_features_from_scenario(scenario)
print(f"  vector_map.coords[0].shape = {vector_map_feature.coords[0].shape}  (num_lane_segments, 2, 2)")

print("\nBuilding Agents features from the real ego + tracked objects history...")
agents_builder = AgentsFeatureBuilder(past_trajectory_sampling)
agents_feature = agents_builder.get_features_from_scenario(scenario)
print(f"  agents.ego[0].shape    = {agents_feature.ego[0].shape}     (num_frames, ego_state_dim)")
print(f"  agents.agents[0].shape = {agents_feature.agents[0].shape}  (num_frames, num_agents, agent_state_dim)")

# Feature builders return numpy arrays; the model needs torch tensors.
features = {
    "vector_map": vector_map_feature.to_feature_tensor(),
    "agents": agents_feature.to_feature_tensor(),
}

print("\nBuilding BasicVectorMapMLP and running forward()...")
model = BasicVectorMapMLP(
    num_output_features=future_trajectory_sampling.num_poses * 3,  # (x, y, heading) per future pose
    hidden_size=128,
    vector_map_feature_radius=20,
    past_trajectory_sampling=past_trajectory_sampling,
    future_trajectory_sampling=future_trajectory_sampling,
)

with torch.no_grad():
    predictions = model.forward(features)

trajectory = predictions["trajectory"]
print(f"  predicted trajectory.data.shape = {trajectory.data.shape}  (batch, num_future_poses, state_dim)")
print("\nSuccess: real nuPlan features flowed through BC_model_v0 without errors.")
