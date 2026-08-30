from typing import cast

import torch
from torch import nn

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)


def create_mlp(input_size: int, output_size: int, hidden_size: int = 128) -> torch.nn.Module:
    """
    Create MLP
    :param input_size: input feature size
    :param output_size: output feature size
    :param hidden_size: hidden layer
    :return: sequential network
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


def convert_predictions_to_trajectory(predictions: torch.Tensor, trajectory_state_size: int) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :param trajectory_state_size: trajectory state size
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.reshape(num_batches, -1, trajectory_state_size)

        
class BasicVectorMapMLP(TorchModuleWrapper):
    """
    Starting-point behavior-cloning model, adapted from nuplan's VectorMapSimpleMLP.
    Same modeling logic (map + ego + agents each through a small MLP, max-pooled, concatenated,
    then a final MLP predicts the trajectory) but with the TorchScript export machinery
    (ScriptableTorchModuleWrapper / scriptable_forward) removed in favor of one plain forward().
    """

    def __init__(
        self,
        num_output_features: int,
        hidden_size: int,
        vector_map_feature_radius: int,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
    ):
        """
        Initialize the model.
        :param num_output_features: number of target features
        :param hidden_size: size of hidden layers of MLP
        :param vector_map_feature_radius: The query radius scope relative to the current ego-pose.
        :param past_trajectory_sampling: Sampling parameters for past trajectory
        :param future_trajectory_sampling: Sampling parameters for future trajectory
        """
        super().__init__(
            feature_builders=[
                VectorMapFeatureBuilder(radius=vector_map_feature_radius),
                AgentsFeatureBuilder(past_trajectory_sampling),
            ],
            target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling)],
            future_trajectory_sampling=future_trajectory_sampling,
        )

        self._hidden_size = hidden_size

        # Vectormap feature input size is 2D start lane coord + 2D end lane coord
        self.vectormap_mlp = create_mlp(
            input_size=2 * VectorMap.lane_coord_dim(), output_size=self._hidden_size, hidden_size=self._hidden_size
        )

        # Ego trajectory feature
        self.ego_mlp = create_mlp(
            input_size=(past_trajectory_sampling.num_poses + 1) * Agents.ego_state_dim(),
            output_size=self._hidden_size,
            hidden_size=self._hidden_size,
        )

        # Agent trajectory feature
        self._agent_mlp_dim = (past_trajectory_sampling.num_poses + 1) * Agents.agents_states_dim()
        self.agent_mlp = create_mlp(
            input_size=self._agent_mlp_dim,
            output_size=self._hidden_size,
            hidden_size=self._hidden_size,
        )

        # Final mlp
        self._mlp = create_mlp(
            input_size=3 * self._hidden_size, output_size=num_output_features, hidden_size=self._hidden_size
        )

        self._vector_map_flatten_lane_coord_dim = VectorMap.flatten_lane_coord_dim()
        self._trajectory_state_size = Trajectory.state_size()

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict a trajectory from vector-map and agent-history features.
        :param features: dict containing {"vector_map": VectorMap, "agents": Agents}
        :return: dict containing {"trajectory": Trajectory}
        """
        vector_map_data = cast(VectorMap, features["vector_map"])
        ego_agents_feature = cast(Agents, features["agents"])

        # Each of these is a list with one tensor per sample in the batch, because the
        # number of lanes/agents differs sample-to-sample and can't be stacked into one tensor.
        ego_past_trajectory = ego_agents_feature.ego
        agents_past_trajectory = ego_agents_feature.agents
        vector_map_coords = vector_map_data.coords

        batch_size = len(vector_map_coords)

        vector_map_feature = []
        agents_feature = []
        ego_feature = []

        # Map and agent features have different sizes across the batch, so process one sample at a time.
        for sample_idx in range(batch_size):
            sample_ego_feature = self.ego_mlp(ego_past_trajectory[sample_idx].view(1, -1))
            ego_feature.append(torch.max(sample_ego_feature, dim=0).values)

            vectormap_coords = vector_map_coords[sample_idx].reshape(-1, self._vector_map_flatten_lane_coord_dim)
            if vectormap_coords.numel() == 0:
                vectormap_coords = torch.zeros(
                    (1, self._vector_map_flatten_lane_coord_dim),
                    dtype=vectormap_coords.dtype,
                    device=vectormap_coords.device,
                )
            sample_vectormap_feature = self.vectormap_mlp(vectormap_coords)
            vector_map_feature.append(torch.max(sample_vectormap_feature, dim=0).values)

            this_agents_feature = agents_past_trajectory[sample_idx]
            agents_multiplier = float(min(this_agents_feature.shape[1], 1))

            if this_agents_feature.shape[1] > 0:  # at least one valid agent in this sample
                # <num_frames, num_agents, feature_dim> -> <num_agents, num_frames * feature_dim>
                orig_shape = this_agents_feature.shape
                flattened_agents = this_agents_feature.transpose(1, 0).reshape(orig_shape[1], -1)
            else:
                flattened_agents = torch.zeros(
                    (this_agents_feature.shape[0], self._agent_mlp_dim),
                    device=sample_vectormap_feature.device,
                    dtype=sample_vectormap_feature.dtype,
                )

            sample_agent_feature = self.agent_mlp(flattened_agents)
            sample_agent_feature *= agents_multiplier  # zero out if there were no agents at all
            agents_feature.append(torch.max(sample_agent_feature, dim=0).values)

        vector_map_feature = torch.cat(vector_map_feature).reshape(batch_size, -1)
        ego_feature = torch.cat(ego_feature).reshape(batch_size, -1)
        agents_feature = torch.cat(agents_feature).reshape(batch_size, -1)

        input_features = torch.cat([vector_map_feature, ego_feature, agents_feature], dim=1)

        predictions = self._mlp(input_features)
        trajectory_data = convert_predictions_to_trajectory(predictions, self._trajectory_state_size)

        return {"trajectory": Trajectory(data=trajectory_data)}
