"""Vectorised scene features for the IL planner.

Builds ego history, up to `max_agents` neighbour tracks, and sampled map polylines in an
ego-centric frame. Also emits per-agent future targets, used only when the model sets
`predict_agents`.

The cache key is the builder's CONFIG, never its source, so changing this file does not
invalidate an existing cache - clear it manually after edits.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Type

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    to_tensor,
)
from il_planner.features.feature_utils import rotate_round_z_axis, normalize_angle, interpolate_polyline
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap, PolygonMapObject
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,

)
logger = logging.getLogger(__name__)


def _walk(data, fn):
    """Apply fn to every leaf of a nested dict. Saves repeating this in 4 methods below."""
    if isinstance(data, dict):
        return {k: _walk(v, fn) for k, v in data.items()}
    return fn(data)


class SimpleFeature(AbstractModelFeature):
    """One sample's features (or a batch's, after collate)."""

    # The exact keys every sample must carry, spelled out instead of being read off
    # whichever sample happens to land first in a batch. Two reasons:
    #   - a sample that is missing one is named in the error instead of surfacing as a
    #     bare `KeyError: 'point_vector'` from inside collate's dict comprehension,
    #   - is_valid can check it, which is what stops a feature cached by an OLDER
    #     version of this file from being loaded as if it were current. The cache key
    #     is (feature name, builder config) and ignores this source file entirely, so
    #     nothing else catches that.
    AGENT_KEYS: Tuple[str, ...] = ("position", "heading", "velocity", "valid_mask", "category")
    MAP_KEYS: Tuple[str, ...] = ("point_position", "point_vector", "on_route", "tl_status", "valid_mask")
    GROUP_KEYS: Dict[str, Tuple[str, ...]] = {"agent": AGENT_KEYS, "map": MAP_KEYS}
    # current_state batches like origin/angle (one vector per sample, torch.stack'd).
    # Only meaningful when the model runs with use_ego_history=False - see
    # ego_state_encoder.py - but it is always built, so the schema always requires it.
    SCALAR_KEYS: Tuple[str, ...] = ("origin", "angle", "current_state")
    # Supervision for the auxiliary agent-prediction head. OPTIONAL because it is the one
    # thing the simulation path cannot produce - at inference there is no future to read.
    # schema_errors() therefore must not demand it, and collate() only batches it when
    # every sample in the batch has it (i.e. during training).
    OPTIONAL_AGENT_KEYS: Tuple[str, ...] = ("target", "target_valid_mask")

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data

    def schema_errors(self) -> List[str]:
        """Human-readable description of every way this sample departs from the schema above."""
        errors = []
        for group, keys in self.GROUP_KEYS.items():
            values = self.data.get(group)
            if not isinstance(values, dict):
                errors.append(f"data[{group!r}] is {type(values).__name__}, expected dict")
                continue
            missing = [key for key in keys if key not in values]
            if missing:
                errors.append(f"data[{group!r}] missing {missing}, has {sorted(values)}")
        missing_scalars = [key for key in self.SCALAR_KEYS if key not in self.data]
        if missing_scalars:
            errors.append(f"data missing {missing_scalars}, has {sorted(self.data)}")
        return errors

    # ------------------------------------------------------------------
    # THE method that matters. Everything else in this class is boilerplate.
    # ------------------------------------------------------------------
    @classmethod
    def collate(cls, batch: List["SimpleFeature"]) -> "SimpleFeature":
        """
        Turn a list of per-sample features into ONE batched feature.

        This is what makes a transformer possible: pad_sequence pads every sample
        up to the largest agent/lane count in the batch, giving (B, N_max, ...)
        instead of the ragged python lists nuplan's Agents/VectorMap produce.
        That is exactly why your current forward() has a per-sample loop and
        PlanTF's does not.

        ANSWER KEY: nuplan_feature.py:21  (this is a direct simplification of it)
        """
        # Screen the batch against the declared schema first. A sample that fails is
        # dropped rather than killing the run: one bad sample in 32 costs nothing, and
        # a 30-epoch run dying at epoch 0 with a bare KeyError costs the whole run.
        # The drop is logged with what was wrong, so it can never pass unnoticed.
        usable: List["SimpleFeature"] = []
        rejected: List[str] = []
        for index, sample in enumerate(batch):
            errors = sample.schema_errors()
            if errors:
                rejected.append(f"sample {index}: {'; '.join(errors)}")
            else:
                usable.append(sample)

        if rejected:
            detail = " | ".join(rejected)
            if not usable:
                raise ValueError(f"Every sample in this batch failed the SimpleFeature schema - {detail}")
            logger.error(
                f"Dropping {len(rejected)} of {len(batch)} samples that failed the SimpleFeature "
                f"schema - {detail}"
            )

        batched: Dict[str, Any] = {}
        for group, keys in cls.GROUP_KEYS.items():
            if group == "agent":
                keys = keys + tuple(
                    k for k in cls.OPTIONAL_AGENT_KEYS
                    if all(k in b.data["agent"] for b in usable)
                )
            batched[group] = {
                key: pad_sequence([b.data[group][key] for b in usable], batch_first=True)
                for key in keys
            }

        for group in cls.SCALAR_KEYS:
            batched[group] = torch.stack([b.data[group] for b in usable]) # why dim = 0

        return cls(data=batched)

    # ------------------------------------------------------------------
    # Boilerplate required by AbstractModelFeature - already done for you.
    # ------------------------------------------------------------------
    def to_feature_tensor(self) -> "SimpleFeature":
        return SimpleFeature(data=_walk(self.data, to_tensor))

    def to_device(self, device: torch.device) -> "SimpleFeature":
        return SimpleFeature(data=_walk(self.data, lambda x: x.to(device)))

    def serialize(self) -> Dict[str, Any]:
        return self.data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "SimpleFeature":
        return cls(data=data)

    def unpack(self) -> List["SimpleFeature"]:
        raise NotImplementedError("Only needed if you split a batch back into samples.")

    @property
    def is_valid(self) -> bool:
        """Lets the caching pipeline throw away broken samples instead of crashing.

        Checked in two places by nuplan's compute_or_load_feature: a freshly built
        feature is only written to the cache when this is True, and a feature read back
        from the cache is asserted on. Including the schema check here is what makes a
        cache entry written by an older version of this file fail at load, naming the
        cache, instead of travelling on to collate as a missing key.
        """
        if self.schema_errors():
            return False
        return bool(self.data["agent"]["valid_mask"].any())

    @staticmethod
    def agent_feature_dim() -> int:
        """Per-agent, per-timestep width: position(2) + velocity(2) + heading as cos/sin(2) + valid(1). Return 7"""
        return 7



class SimpleFeatureBuilder(AbstractFeatureBuilder):
    """Builds SimpleFeature from a scenario (training) or from live simulation input."""

    def __init__(self, radius: float = 50.0, history_horizon: float = 2.0,
                 sample_interval: float = 0.1, max_agents: int = 32,
                 map_sample_points: int = 20, future_horizon: float = 8.0) -> None:
        self.radius = radius
        self.history_horizon = history_horizon
        self.sample_interval = sample_interval
        self.history_samples = int(history_horizon / sample_interval)
        # Only used for the auxiliary agent-prediction target. Matches the ego horizon so
        # the two supervisions describe the same slice of the future.
        self.future_horizon = future_horizon
        self.future_samples = int(future_horizon / sample_interval)
        self.max_agents = max_agents
        self.map_sample_points = map_sample_points

        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]

    # ------------------------------------------------------------------
    # Identity - these two wire you into the framework.
    # ------------------------------------------------------------------
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """The dict key your model's forward() will look up: features["simple_feature"]."""
        return "simple_feature"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        return SimpleFeature

    # ------------------------------------------------------------------
    # The two entry points. Note they differ ONLY in where the raw data comes
    # from - both then call the same _build_feature. Keeping that single shared
    # path is what stops training and simulation from silently diverging.
    # ANSWER KEY: nuplan_feature_builder.py:76 and :128
    # ------------------------------------------------------------------
    def get_features_from_scenario(self, scenario: AbstractScenario) -> SimpleFeature:
        present_ego_state = [scenario.initial_ego_state]
        past_ego_state = [
            ego_state for ego_state in scenario.get_ego_past_trajectory(
            iteration=0, time_horizon=self.history_horizon, num_samples=self.history_samples)
            ]
        ego_state_list = past_ego_state + present_ego_state


        present_tracked_object = [scenario.initial_tracked_objects.tracked_objects]
        past_tracked_objects = [
            tracked_object.tracked_objects for tracked_object in scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.history_horizon, num_samples=self.history_samples)
        ]

        # Past + present + FUTURE in one timeline, so _build_feature's existing
        # (N, T) machinery produces history and supervision in a single pass and they
        # cannot disagree about agent identity, ordering or frame. The split happens at
        # present_idx afterwards. ANSWER KEY: nuplan_feature_builder.py:104
        future_ego_states = list(scenario.get_ego_future_trajectory(
            iteration=0, time_horizon=self.future_horizon, num_samples=self.future_samples))
        ego_state_list = ego_state_list + future_ego_states

        future_tracked_objects = [
            tracked_object.tracked_objects for tracked_object in scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_horizon, num_samples=self.future_samples)
        ]

        tracked_objects_list = (
            past_tracked_objects + present_tracked_object + future_tracked_objects
        )

        return self._build_feature(
            # NOT -1 any more: the present is in the middle of the timeline, not at its end.
            present_idx=self.history_samples,
            ego_states=ego_state_list,
            tracked_objects_list= tracked_objects_list,
            map_api=scenario.map_api,
            route_roadblock_ids=scenario.get_route_roadblock_ids(),
            traffic_light_status=scenario.get_traffic_light_status_at_iteration(0),
        )


    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> SimpleFeature:

        history = current_input.history
        ego_states = history.ego_states
        tracked_objects_list = [observation.tracked_objects for observation in history.observations]

        horizon = self.history_samples + 1

        return self._build_feature(
            present_idx=-1,
            ego_states=ego_states[-horizon:],
            tracked_objects_list= tracked_objects_list[-horizon:],
            map_api=initialization.map_api,
            route_roadblock_ids=initialization.route_roadblock_ids,
            traffic_light_status=current_input.traffic_light_data,
        )

    # ------------------------------------------------------------------
    # Shared construction.
    # ANSWER KEY: nuplan_feature_builder.py:147-190
    # ------------------------------------------------------------------
    def _build_feature(self, present_idx, ego_states: List[EgoState], tracked_objects_list,
                       map_api, route_roadblock_ids, traffic_light_status) -> SimpleFeature:
        
        if present_idx < 0:
            present_idx += len(tracked_objects_list)

        present_ego_state = ego_states[present_idx]
        query_xy = present_ego_state.rear_axle
        origin = query_xy.array
        angle = present_ego_state.rear_axle.heading

        data = {}

        ego_features = self._get_ego_features(ego_states)
        agent_features = self._get_agent_features(query_xy,present_idx,tracked_objects_list)

        data["agent"] = {}
        for k in agent_features.keys():
            data["agent"][k] = np.concatenate(
                [ego_features[k][None,...], agent_features[k]], axis = 0,
                )

        data["map"] = self._get_map_features(
                map_api=map_api, 
                query_xy=query_xy, 
                route_roadblock_ids=route_roadblock_ids, 
                traffic_light_status=traffic_light_status,
                radius=self.radius,
                sample_points=self.map_sample_points,
                )
        
        # index 0 has no predecessor to difference against for yaw rate; the caller
        # always passes present_idx = T-1, so this only guards the degenerate T == 1 case.
        previous_ego_state = ego_states[present_idx - 1] if present_idx > 0 else present_ego_state
        data["current_state"] = self._get_ego_current_state(present_ego_state, previous_ego_state)

        data["origin"] = origin
        data["angle"] = np.array(angle, dtype=np.float64)
        SimpleFeatureBuilder.normalize(data,origin,angle)

        # Split the timeline: everything up to and including present_idx is INPUT, the
        # rest is supervision for the agent-prediction head. Simulation never appends a
        # future, so present_idx is the last index there and this is a no-op.
        if present_idx < data["agent"]["position"].shape[1] - 1:
            SimpleFeatureBuilder.split_future(data, present_idx)

        return SimpleFeature(data=data)

    @staticmethod
    def split_future(data, present_idx: int) -> None:
        """
        Move timesteps after present_idx out of the history arrays and into
        agent["target"], expressed RELATIVE TO EACH AGENT'S OWN PRESENT POSE.

        Relative, not ego-frame, because the head predicts how each agent MOVES: an agent
        40 m away that carries straight on should look identical to one 5 m away doing the
        same thing. Absolute ego-frame targets would make the head relearn that for every
        position on the map. ANSWER KEY: nuplan_feature.py normalize(), target_position.

        Ego occupies row 0 here too. Its row is redundant - nuPlan's EgoTrajectoryTargetBuilder
        already supplies the ego target the main loss uses - but keeping it costs one row
        and keeps every array's agent axis aligned.
        """
        agent = data["agent"]
        position, heading, valid = agent["position"], agent["heading"], agent["valid_mask"]

        present_position = position[:, present_idx][:, None, :]          # (N, 1, 2)
        present_heading = heading[:, present_idx][:, None]               # (N, 1)

        target_position = position[:, present_idx + 1:] - present_position
        target_heading = normalize_angle(heading[:, present_idx + 1:] - present_heading)
        target = np.concatenate([target_position, target_heading[..., None]], axis=-1)

        # An agent must be visible NOW and at the future step for that step to supervise
        # anything: a target relative to a pose we never observed is meaningless.
        target_valid = valid[:, present_idx + 1:] & valid[:, present_idx][:, None]
        target[~target_valid] = 0.0

        agent["target"] = target.astype(np.float32)
        agent["target_valid_mask"] = target_valid

        # Truncate the inputs back to history only.
        for key in ("position", "heading", "velocity", "valid_mask"):
            agent[key] = agent[key][:, : present_idx + 1]

    def _get_ego_current_state(self, ego_state: EgoState, previous_ego_state: EgoState):
        """
        Ego at the PRESENT step only - the whole input when use_ego_history=False.

        Channels 0:3 are zeroed by normalize() (ego is the origin of its own frame), so
        they are constants that carry no information. The model relies on that: its
        StateAttentionEncoder never drops the first three and randomly drops the rest,
        which is only a meaningful regulariser because the rest are the informative ones.

        ANSWER KEY: nuplan_feature_builder.py:192 (_get_ego_current_state)
        """
        steering_angle, yaw_rate = self._additional_ego_states(ego_state, previous_ego_state)

        state = np.zeros(7, dtype=np.float64)
        state[0:2] = ego_state.rear_axle.array
        state[2] = ego_state.rear_axle.heading
        state[3] = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        state[4] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        state[5] = steering_angle
        state[6] = yaw_rate
        return state

    def _additional_ego_states(self, ego_state: EgoState, previous_ego_state: EgoState):
        """Steering angle and yaw rate, neither of which EgoState exposes directly."""
        velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        yaw_rate = normalize_angle(
            ego_state.rear_axle.heading - previous_ego_state.rear_axle.heading
        ) / self.sample_interval

        # Below ~0.2 m/s the heading is numerically meaningless, so the difference above is
        # noise divided by dt - a large fake yaw rate. PlanTF zeroes both for this reason.
        if abs(velocity) < 0.2:
            return 0.0, 0.0

        # abs(): a reversing car must not flip the sign of the inferred steering angle.
        wheel_base = ego_state.car_footprint.vehicle_parameters.wheel_base
        steering_angle = np.arctan(yaw_rate * wheel_base / abs(velocity))
        return float(steering_angle), float(yaw_rate)

    def _get_ego_features(self, ego_states: List[EgoState]): 
        """Ego over T history steps: position, heading, velocity, valid_mask(all True).
        ANSWER KEY: nuplan_feature_builder.py:206-244"""

        T = len(ego_states)

        position = np.zeros((T,2),dtype = np.float64)
        heading = np.zeros((T,),dtype=np.float64)
        velocity = np.zeros((T,2),dtype=np.float64)
        valid_mask = np.ones((T,),dtype = bool)
        category = np.array(self.interested_objects_types.index(TrackedObjectType.EGO),dtype=np.int8)

        for t, state in enumerate(ego_states):
            position[t] = state.rear_axle.array
            heading[t] = state.rear_axle.heading
            velocity[t] = rotate_round_z_axis(state.dynamic_car_state.rear_axle_velocity_2d.array, 
                                           -state.rear_axle.heading)

        return {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "valid_mask": valid_mask,
            "category": category,
        }

    def _get_agent_features(self, query_xy, present_idx: int, tracked_objects_list: List[TrackedObjects]):

        #pre-allocate zeros by shape
        present_tracked_objects = tracked_objects_list[present_idx]
        present_agents = present_tracked_objects.get_tracked_objects_of_types(
            self.interested_objects_types
        )

        N, T = min(len(present_agents), self.max_agents), len(tracked_objects_list)
    
        position = np.zeros((N,T,2), dtype = np.float64)
        heading = np.zeros((N,T), dtype=np.float64)
        velocity = np.zeros((N,T,2),dtype=np.float64)
        valid_mask = np.zeros((N,T), dtype=bool)
        category = np.zeros((N,), dtype=np.int8)

        if N == 0:
            return {
                "position": position,
                "heading": heading,
                "velocity": velocity,
                "valid_mask": valid_mask,
                "category": category,
            }
        #sort by distance
        agent_pos = np.array([agent.center.array for agent in present_agents])
        agent_ids = np.array([agent.track_token for agent in present_agents])
        distance = np.linalg.norm(agent_pos-query_xy.array, axis=1)
        sorted_agent_id = agent_ids[np.argsort(distance)[:self.max_agents]]
        sorted_agent_id = {agent_id: idx for idx, agent_id in enumerate(sorted_agent_id)}
        
        #use sorted token ids to fill in the preallocated spots
        for t, tracked_objects in enumerate(tracked_objects_list):
            if len(tracked_objects) == 0:
                return {
                        "position": position,
                        "heading": heading,
                        "velocity": velocity,
                        "valid_mask": valid_mask,
                        "category": category,
                        }
                
            for agent in tracked_objects:
                if agent.track_token not in sorted_agent_id:
                    continue

                idx = sorted_agent_id[agent.track_token]
                position[idx][t] = agent.center.array
                heading[idx][t] = agent.center.heading
                velocity[idx][t] = agent.velocity.array
                valid_mask[idx][t] = True

                if t == present_idx:
                    category[idx] = self.interested_objects_types.index(agent.tracked_object_type)
            
        return {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "valid_mask": valid_mask,
            "category": category,
        }

    def _get_map_features(
        self,
        map_api: AbstractMap,
        query_xy: Point2D,
        route_roadblock_ids: List[str],
        traffic_light_status: List[TrafficLightStatusData],
        radius: float,
        sample_points: int = 20,
        ):
        """
        Centerline only - skip left/right boundaries and crosswalks for now.
        Use map_api.get_proximal_map_objects(...) then sample each lane's
        baseline_path into self.map_sample_points points.

        ANSWER KEY: nuplan_feature_builder.py:309-437, but you only need the parts
        touching `centerline`, `polygon_on_route` and `polygon_tl_status`.
        Roughly two thirds of that function is boundaries/crosswalks you cut.
        """
        map_objects = map_api.get_proximal_map_objects(
            point=query_xy,
            radius=radius,
            layers=[SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        )
        lanes = map_objects[SemanticMapLayer.LANE] + map_objects[SemanticMapLayer.LANE_CONNECTOR]
        M = len(lanes) # (M,) variable

        point_position = np.zeros((M,sample_points,2), dtype=np.float64) # (M,P,2)
        point_vector = np.zeros((M,sample_points,2), dtype=np.float64)   # (M,P,2) segment directions
        on_route = np.zeros(M, dtype=bool)
        tl_status = np.full(M, TrafficLightStatusType.UNKNOWN, dtype = np.int8)
        valid_mask = np.ones(M, dtype=bool)

        route_ids = {int(r) for r in route_roadblock_ids}
        tls = {tl.lane_connector_id: tl.status for tl in traffic_light_status}

        for i,lane in enumerate(lanes):
            # P+1 samples -> P positions and P difference vectors, like PlanTF.
            path = np.stack([p.array for p in lane.baseline_path.discrete_path], axis = 0)
            pts = interpolate_polyline(path, sample_points + 1)
            point_position[i] = pts[:-1]
            point_vector[i] = pts[1:] - pts[:-1]

            on_route[i] = int(lane.get_roadblock_id()) in route_ids
            tl_status[i] = tls.get(int(lane.id), TrafficLightStatusType.UNKNOWN)

        return {
            "point_position": point_position,
            "point_vector": point_vector,
            "on_route": on_route,
            "tl_status": tl_status,
            "valid_mask": valid_mask,
        }
    @staticmethod
    def normalize(data, origin, angle):
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)],
             [np.sin(angle), np.cos(angle)]]
        )
      
        data["agent"]["position"] = ((data["agent"]["position"] - origin) @ rot_mat).astype(np.float32)
        data["agent"]["velocity"] = (data["agent"]["velocity"] @ rot_mat).astype(np.float32)
        data["agent"]["heading"] = (normalize_angle(data["agent"]["heading"] - angle)).astype(np.float32)
        data["map"]["point_position"] = ((data["map"]["point_position"] - origin) @ rot_mat).astype(np.float32)
        # A vector is a difference of two points, so the origin cancels: rotate only.
        data["map"]["point_vector"] = (data["map"]["point_vector"] @ rot_mat).astype(np.float32)

        # Ego is the origin of the frame we just rotated into, so its own position and
        # heading are identically zero. Store them that way rather than as float noise.
        data["current_state"][:3] = 0.0
        data["current_state"] = data["current_state"].astype(np.float32)

        #set states of unobserved agents to zero 
        vm = data["agent"]["valid_mask"]
        data["agent"]["position"][~vm] = 0.0
        data["agent"]["velocity"][~vm] = 0.0
        data["agent"]["heading"][~vm] = 0.0
    
    