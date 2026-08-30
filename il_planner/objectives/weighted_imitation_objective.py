"""
ImitationObjective with a separate weight on the heading term.

nuPlan's version is mean(MSE(xy)) + mean(L1(heading)). Two problems with that here:
  - MSE's gradient shrinks with the error, L1's is constant +/-1, so once position error
    drops the heading term is ~8% of the update.
  - The open_loop_boxes score weights all four of its metrics equally, and two of them
    are heading. Heading is 50% of the score for 8% of the training signal.

heading_weight scales only the heading term, leaving xy alone.
ANSWER KEY: nuplan/planning/training/modeling/objectives/imitation_objective.py
"""
from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import (
    extract_scenario_type_weight,
)
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class WeightedImitationObjective(AbstractObjective):
    def __init__(
        self,
        scenario_type_loss_weighting: Dict[str, float],
        weight: float = 1.0,
        heading_weight: float = 10.0,
    ):
        self._name = "weighted_imitation_objective"
        self._weight = weight
        self._heading_weight = heading_weight
        self._fn_xy = torch.nn.modules.loss.MSELoss(reduction="none")
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction="none")
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        return ["trajectory"]

    def compute(
        self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType
    ) -> torch.Tensor:
        predicted_trajectory = cast(Trajectory, predictions["trajectory"])
        targets_trajectory = cast(Trajectory, targets["trajectory"])

        loss_weights = extract_scenario_type_weight(
            scenarios, self._scenario_type_loss_weighting, device=predicted_trajectory.xy.device
        )
        # Reshape the per-scenario weight to broadcast over the trailing axes.
        shape_xy = tuple([-1] + [1] * (predicted_trajectory.xy.dim() - 1))
        shape_heading = tuple([-1] + [1] * (predicted_trajectory.heading.dim() - 1))

        xy_loss = self._fn_xy(predicted_trajectory.xy, targets_trajectory.xy) * loss_weights.view(shape_xy)
        heading_loss = (
            self._fn_heading(predicted_trajectory.heading, targets_trajectory.heading)
            * loss_weights.view(shape_heading)
        )

        # The ONLY change from nuPlan's version is _heading_weight on the second term.
        return self._weight * (
            torch.mean(xy_loss) + self._heading_weight * torch.mean(heading_loss)
        )
