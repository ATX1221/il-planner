"""
Behaviour-cloning planner over SimpleFeature. Three PointNet branches -> concat -> decode.

    ego    (B, T*7)      -> _ego_mlp                      -> (B, H)
    agents (B, N-1, T*7) -> _agent_mlp -> masked max-pool -> (B, H)
    lanes  (B, M, P*2+5) -> _map_mlp   -> masked max-pool -> (B, H)  -> (B, 3H) -> trajectory

Ego is kept out of the agent pool: a channel-wise max over 33 rows discards ego's own
heading, and val_avg_heading_error stalled at 0.59 rad because of it (0.26 with a
dedicated ego encoder).
"""
from typing import cast

import torch
from torch import nn

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from simple_feature import SimpleFeature, SimpleFeatureBuilder


def mlp(input_size, output_size, hidden_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


def convert_predictions_to_trajectory(predictions, trajectory_state_size):
    """(B, num_output_features) -> (B, num_poses, 3)."""
    num_batches = predictions.shape[0]
    return predictions.reshape(num_batches, -1, trajectory_state_size)


class BasicVectorMapMLP(TorchModuleWrapper):
    def __init__(
        self,
        num_output_features: int,
        hidden_size: int,
        future_trajectory_sampling: TrajectorySampling,
        feature_builder: SimpleFeatureBuilder = SimpleFeatureBuilder(),
    ):
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling)],
            future_trajectory_sampling=future_trajectory_sampling,
        )

        self.hidden_size = hidden_size
        self.T = feature_builder.history_samples + 1   # from the builder, so they can't disagree

        # Same input width, separate weights: each branch specialises.
        self.agent_mlp_dim = self.T * SimpleFeature.agent_feature_dim()
        self._agent_mlp = mlp(self.agent_mlp_dim, hidden_size, hidden_size)
        self._ego_mlp = mlp(self.agent_mlp_dim, hidden_size, hidden_size)

        P = feature_builder.map_sample_points
        self.map_mlp_dim = P * 2 * 2 + 1 + 4           # centerline + vectors + on_route + tl one-hot
        self._map_mlp = mlp(self.map_mlp_dim, hidden_size, hidden_size)

        self._mlp = mlp(3 * hidden_size, num_output_features, hidden_size)

        self.trajectory_state_size = Trajectory.state_size()

    def forward(self, features: FeaturesType) -> TargetsType:
        feature = cast(SimpleFeature, features["simple_feature"])
        agent = feature.data["agent"]
        map_ = feature.data["map"]

        position = agent["position"]        # (B, N, T, 2)
        heading = agent["heading"]          # (B, N, T)
        velocity = agent["velocity"]        # (B, N, T, 2)
        valid_mask = agent["valid_mask"]    # (B, N, T) bool

        # cos/sin, not radians: -pi and +pi are one direction but opposite numbers.
        heading_enc = torch.stack([heading.cos(), heading.sin()], dim=-1)   # (B, N, T, 2)
        # mask as a feature too, so a real (0,0) differs from an unobserved slot.
        valid = valid_mask.unsqueeze(-1).to(position.dtype)                 # (B, N, T, 1)

        x = torch.cat([position, heading_enc, velocity, valid], dim=-1)     # (B, N, T, 7)
        x = x.reshape(x.shape[0], x.shape[1], -1)                           # (B, N, T*7)

        # Linear maps over the last axis, so no per-sample loop anywhere below.
        ego_vec = self._ego_mlp(x[:, 0])                                    # (B, H)  row 0 is ego

        agents = self._agent_mlp(x[:, 1:])                                  # (B, N-1, H)
        valid_agents = valid_mask[:, 1:].any(dim=-1).unsqueeze(-1)          # (B, N-1, 1)
        # finfo of the POOLED tensor: under AMP it is fp16, where float32's min is -inf.
        agents = agents.masked_fill(~valid_agents, torch.finfo(agents.dtype).min)
        agents_vec = torch.max(agents, dim=1).values                        # (B, H)
        # Zero agents -> the max is finfo.min -> NaN downstream. Ego no longer guards this.
        agents_vec = torch.where(
            valid_agents.any(dim=1), agents_vec, torch.zeros_like(agents_vec)
        )

        point_position = map_["point_position"]                             # (B, M, P, 2)
        B, M = point_position.shape[0], point_position.shape[1]
        # tl_status is a label, not a quantity: as an int, RED would be twice YELLOW and
        # GREEN (=0) would look like an absent lane.
        tl = torch.nn.functional.one_hot(map_["tl_status"].long(), 4).to(point_position.dtype)
        on_route = map_["on_route"].unsqueeze(-1).to(point_position.dtype)  # (B, M, 1)

        # point_vector states lane direction explicitly; without it the net has to recover
        # it by differencing adjacent points, a ~3 m signal inside ~50 m values.
        point_vector = map_["point_vector"]                                 # (B, M, P, 2)
        lanes = torch.cat(
            [point_position.reshape(B, M, -1), point_vector.reshape(B, M, -1), tl, on_route],
            dim=-1,
        )
        lanes = self._map_mlp(lanes)                                        # (B, M, H)
        # Already (B, M) - lanes have no time axis - and every sample has >=1 lane.
        lanes = lanes.masked_fill(
            ~map_["valid_mask"].unsqueeze(-1), torch.finfo(lanes.dtype).min
        )
        map_vec = torch.max(lanes, dim=1).values                            # (B, H)

        pooled = torch.cat([ego_vec, agents_vec, map_vec], dim=-1)          # (B, 3H)
        predictions = self._mlp(pooled)
        trajectory_data = convert_predictions_to_trajectory(
            predictions, self.trajectory_state_size
        )
        return {"trajectory": Trajectory(data=trajectory_data)}
