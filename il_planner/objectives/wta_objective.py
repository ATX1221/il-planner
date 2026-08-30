from typing import Dict, List, cast

import torch
import torch.nn.functional as F

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import (
    extract_scenario_type_weight,
)
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

class WinnerTakesAllObjective(AbstractObjective):
    """
    Multi-modal imitation loss.

    Regression gradient goes ONLY to the mode that already landed closest; the other K-1
    are left alone. That is what stops the modes collapsing onto the mean of the possible
    futures - which is what a single-mode MSE model is forced to predict when the scene is
    ambiguous, and which is not a maneuver any car performs.

    A separate classification head learns WHICH mode won, so inference can pick one.
    """
    def __init__(
      self,
      scenario_type_loss_weighting: Dict[str,float],
      weight: float = 1.0,
      cls_weight: float = 1.0,
    ):
        self._name = "wta_objective"
        self._weight = weight
        self._cls_weight = cls_weight
        self.scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        return ["trajectory"]

    def compute(
            self, predictions:FeaturesType, targets: TargetsType, scenario: ScenarioListType
    ) -> torch.Tensor:

        modes = predictions["modes"]                            # (B, K, T, 3)
        pi = predictions["probability"]                                  # (B,)
        target = cast(Trajectory, targets["trajectory"]).data 
        B = modes.shape[0]

        ade = torch.norm(modes[..., :2] - target[:, None, :, :2], dim = -1)   # (B,K,T)
        best = ade.sum(dim = -1).argmin(dim = -1)  # (B,)
        best_traj = modes[torch.arange(B), best] # (B, T, 3)

        w = extract_scenario_type_weight(
            scenario,self.scenario_type_loss_weighting, device=modes.device
        )

        reg = F.smooth_l1_loss(best_traj, target, reduction="none").mean(dim=(1,2)) #(B,)
        reg = (reg * w).mean()

        cls = F.cross_entropy(pi, best.detach(), reduction= "none")
        cls = (cls * w).mean()

        return self._weight * reg + self._cls_weight * cls

    

class AgentPredictionObjective(AbstractObjective):
    """
    Auxiliary supervision: how does every OTHER agent move over the next horizon?

    Listed as its own objective rather than folded into the WTA loss so nuPlan logs it
    separately under objectives/{train,val}_agent_prediction - the reg/cls split we
    deliberately skipped is worth having here, because this term's whole job is to keep
    falling and dilute the rising cls term.

    NOTE the training yaml must use objective_aggregate_mode: sum. With 'mean' the loss
    becomes the AVERAGE of the objectives, which silently rescales every existing term.
    """

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0):
        self._name = "agent_prediction"
        self._weight = weight
        self.scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        # The supervision rides in the FEATURE, not the target dict: it comes from the
        # feature builder's own future timeline, not from nuPlan's target builder.
        return ["trajectory"]

    def compute(
        self, predictions: FeaturesType, targets: TargetsType, scenario: ScenarioListType
    ) -> torch.Tensor:
        prediction = predictions["agent_prediction"]                    # (B, N-1, T, 2)
        target = predictions["agent_target"][:, 1:, :, :2]              # drop ego row
        mask = predictions["agent_target_valid_mask"][:, 1:]            # (B, N-1, T)

        if not mask.any():
            return prediction.sum() * 0.0

        # Masked mean: padded agents and unobserved future steps must not dilute the loss.
        return self._weight * F.smooth_l1_loss(prediction[mask], target[mask])
