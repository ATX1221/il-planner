"""Single-mode transformer planner.

Same encoder as transformer_planner but emits one trajectory, trained with plain L2.
Kept separate rather than num_modes=1 so the two can be compared without shared-code risk.
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

from il_planner.features.simple_feature import SimpleFeature, SimpleFeatureBuilder
from il_planner.models.ego_state_encoder import StateAttentionEncoder


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


class TransformerPlanner(TorchModuleWrapper):
    def __init__(
        self,
        num_output_features: int,
        hidden_size: int,
        future_trajectory_sampling: TrajectorySampling,
        feature_builder: SimpleFeatureBuilder = SimpleFeatureBuilder(),
        num_heads: int = 8,
        encoder_depth: int = 2,
        dropout: float = 0.1,
        num_modes: int = 6,
        use_ego_history: bool = True,
        state_channel: int = 6,
        state_dropout: float = 0.75,
    ):
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling)],
            future_trajectory_sampling=future_trajectory_sampling,
        )
        
        self.hidden_size = hidden_size
        self.T = feature_builder.history_samples + 1   # from the builder, so they can't disagree

        # Tokenizers: one entity -> one H-dim token. Ego shares the agent tokenizer and is
        # distinguished by being token 0, not by having its own weights.
        self.agent_mlp_dim = self.T * SimpleFeature.agent_feature_dim()
        self._agent_mlp = mlp(self.agent_mlp_dim, hidden_size, hidden_size)

        # use_ego_history=False replaces token 0 - which the agent tokenizer built from
        # ego's whole 21-step past - with an encoding of the PRESENT ego state alone.
        # The other agents keep their history; only ego loses it. See ego_state_encoder.py
        # for why: ego's own past is the one input that becomes self-referential once the
        # model is steering, and extrapolating it is the shortcut that wins open loop and
        # loses closed loop.
        self.use_ego_history = use_ego_history
        self._ego_state_encoder = (
            None if use_ego_history
            else StateAttentionEncoder(state_channel, hidden_size, state_dropout)
        )

        P = feature_builder.map_sample_points
        self.map_mlp_dim = P * 2 * 2 + 1 + 4           # centerline + vectors + on_route + tl one-hot
        self._map_mlp = mlp(self.map_mlp_dim, hidden_size, hidden_size)

        # norm_first=True (pre-norm) trains stably without an LR warmup, which this config
        # does not have. batch_first=True because our tensors are (B, seq, dim).
        self._blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=4 * hidden_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ) for _ in range(encoder_depth)
        )

        self._norm = nn.LayerNorm(hidden_size)

        self.trajectory_state_size = Trajectory.state_size()

        self.num_modes = num_modes
        self.future_steps = future_trajectory_sampling.num_poses

        self._mlp = mlp(hidden_size, num_output_features, hidden_size)

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

        agents = torch.cat([position, heading_enc, velocity, valid], dim=-1)     # (B, N, T, 7)
        agents = agents.reshape(agents.shape[0], agents.shape[1], -1)            # (B, N, T*7)

        # One token per agent. Linear maps over the last axis, so no loop anywhere here.
        agents = self._agent_mlp(agents)                                         # (B, N, H)

        if self._ego_state_encoder is not None:
            # Overwrite ego's token AFTER the tokenizer has run: the tensor keeps its
            # shape, key_padding_mask keeps its meaning, and ego stays token 0. Ego is
            # always valid, so no mask update is needed.
            agents = torch.cat(
                [self._ego_state_encoder(feature.data["current_state"]).unsqueeze(1),
                 agents[:, 1:]],
                dim=1,
            )                                                                    # (B, N, H)

        # --- lanes ---
        point_position = map_["point_position"]                             # (B, M, P, 2)
        # P is unpacked, not left to reshape's -1, because -1 is AMBIGUOUS when the
        # tensor is empty: closed-loop simulation can drive ego somewhere with no lane
        # inside the builder's 50 m radius, giving M == 0 and zero elements to divide up.
        # That crashed 3 of 28 scenarios on the first closed-loop run (2026-08-26).
        B, M, P = point_position.shape[0], point_position.shape[1], point_position.shape[2]
        # tl_status is a label, not a quantity: as an int, RED would be twice YELLOW and
        # GREEN (=0) would look like an absent lane.
        tl = torch.nn.functional.one_hot(map_["tl_status"].long(), 4).to(point_position.dtype)
        on_route = map_["on_route"].unsqueeze(-1).to(point_position.dtype)  # (B, M, 1)

        # point_vector states lane direction explicitly; without it the net has to recover
        # it by differencing adjacent points, a ~3 m signal inside ~50 m values.
        point_vector = map_["point_vector"]                                 # (B, M, P, 2)
        lanes = torch.cat(
            [point_position.reshape(B, M, P * 2), point_vector.reshape(B, M, P * 2), tl, on_route],
            dim=-1,
        )
        lanes = self._map_mlp(lanes)                                        # (B, M, H)

        # --- one sequence: joint attention over ego, agents and map ---
        x = torch.cat([agents,lanes], dim = 1)                              # (B, N+M, H)

        # True = ignore. Same ~valid convention as v1's masked_fill, but no trailing axis:
        # this masks TOKENS, not channels, and is applied to the attention WEIGHTS rather
        # than the feature values - zeroing a feature would not exclude it from a weighted
        # sum the way -inf excludes it from a max.
        # any(-1) on agents collapses T; lane valid_mask is already (B, M).
        # Ego is token 0 and always valid, so no sample can have every key masked, which
        # would make softmax over all -inf produce NaN.
        key_padding_mask = torch.cat(
            [~valid_mask.any(dim=-1), ~map_["valid_mask"]], dim = 1
            )                                                               # (B, N+M)

        for layer in self._blocks:
            x = layer(x, src_key_padding_mask = key_padding_mask)
        x = self._norm(x)    # (B, N+M, H)

        # Ego's token, having absorbed whatever it attended to: a learned,
        # relevance-weighted summary rather than a channel-wise max.
        ego_cmp = x[:,0]    # (B,H)

        predictions = self._mlp(ego_cmp)                                    # (B, num_output_features)
        trajectory_data = convert_predictions_to_trajectory(
            predictions, self.trajectory_state_size
        )
        return {"trajectory": Trajectory(data=trajectory_data)}
