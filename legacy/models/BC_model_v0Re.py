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



def mlp(input_size, output_size, hidden_size):
    return nn.Sequential(
        nn.Linear(input_size,hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size,output_size)
    )

def convert_predictions_to_trajectory(predictions,trajectory_state_size):
    num_batches = predictions.shape[0]
    return predictions.reshape(num_batches, -1, trajectory_state_size)

class BasicVectorMapMLP(TorchModuleWrapper):

    def __init__(
        self,
        num_output_features: int,
        hidden_size: int,
        vector_map_feature_radius: int,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,      
    ):

        super().__init__(
        feature_builders = [
            AgentsFeatureBuilder(past_trajectory_sampling),
            VectorMapFeatureBuilder(radius = vector_map_feature_radius)
        ],
        target_builders = [EgoTrajectoryTargetBuilder(future_trajectory_sampling)],
        future_trajectory_sampling=future_trajectory_sampling
        )

        self.hidden_size = hidden_size

        
        #vector map encoding shape: Nx4 
        self._vector_map_flatten_lane_coord_dim = VectorMap.flatten_lane_coord_dim()
        self.vectormap_mlp = mlp(
            input_size=self._vector_map_flatten_lane_coord_dim, 
            output_size=hidden_size, 
            hidden_size=hidden_size
            )

        #Ego trajectory encoding, original shape: frames x ego_state_dim 
        self.ego_mlp = mlp(input_size = (past_trajectory_sampling.num_poses + 1) * Agents.ego_state_dim(),output_size=hidden_size, hidden_size=hidden_size)

        #Agents
        self.agent_mlp_dim = (past_trajectory_sampling.num_poses+1)*Agents.agents_states_dim()
        self.agent_mlp = mlp(input_size=self.agent_mlp_dim,output_size=hidden_size,hidden_size=hidden_size)


        #final mlp 
        self._mlp = mlp(
            input_size = 3 * self.hidden_size, output_size=num_output_features, hidden_size=hidden_size
        )

        self.trajectory_state_size = Trajectory.state_size()


    def forward(self,features):

        vector_map_data = cast(VectorMap,features["vector_map"])
        ego_agents_features = cast(Agents, features["agents"])

        vector_map_coords = vector_map_data.coords # sample x num_lane_segments x 2 x 2 ---> N x 4 
        past_ego_trajectory = ego_agents_features.ego # sample x num_frames x states
        past_agent_trajectory = ego_agents_features.agents # sample x num_agents x numframes x states

        batch_size = len(past_ego_trajectory)


        vector_map_features = []

        ego_features = []
        agents_features = []

        for sample in range(batch_size):
            #ego 
            sample_ego_feature = self.ego_mlp(past_ego_trajectory[sample].reshape(1,-1))
            ego_features.append(torch.max(sample_ego_feature, dim=0).values)

            #vector map
            vectormap_coords = vector_map_coords[sample].reshape(-1,self._vector_map_flatten_lane_coord_dim)
            if vectormap_coords.numel() == 0:
                vectormap_coords = torch.zeros(
                    (1,self._vector_map_flatten_lane_coord_dim),
                    dtype = vectormap_coords.dtype,
                    device= vectormap_coords.device,
                    )
            sample_map_feature = self.vectormap_mlp(vectormap_coords)
            vector_map_features.append(torch.max(sample_map_feature, dim = 0).values)

            #agents
            sample_agents_feature = past_agent_trajectory[sample]

            if sample_agents_feature.shape[1] > 0:
                orig_shape = sample_agents_feature.shape
                flatten_agents = sample_agents_feature.transpose(0,1).reshape(orig_shape[1],-1)
            else:
                flatten_agents = torch.zeros(
                    (sample_agents_feature.shape[0],self.agent_mlp_dim), 
                    device=sample_map_feature.device,
                    dtype=sample_map_feature.dtype,
                    )
                
            if sample_agents_feature.shape[1] > 0:
                agents_multiplier = 1
            else:
                agents_multiplier = 0

            sample_agents_feature = self.agent_mlp(flatten_agents)
            sample_agents_feature *= agents_multiplier
            agents_features.append(torch.max(sample_agents_feature, dim=0).values)

        vector_map_features = torch.cat(vector_map_features).reshape(batch_size,-1)
        ego_features = torch.cat(ego_features).reshape(batch_size,-1)
        agents_features = torch.cat(agents_features).reshape(batch_size,-1)

        input_feature = torch.cat([vector_map_features,ego_features,agents_features],dim = 1)
        predictions = self._mlp(input_feature)
        trajectory_data = convert_predictions_to_trajectory(predictions, self.trajectory_state_size)
        
        return {"trajectory": Trajectory(data=trajectory_data)}




