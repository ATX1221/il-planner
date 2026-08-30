from typing import List, cast
import torch
from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class MinADE(AbstractTrainingMetric):
    """
    Best-of-K displacement error: grades the MODES alone, with the classifier removed.

    Read it against avg_displacement_error, which grades loc[argmax(pi)] - the mode
    actually driven. The difference between the two is what mode selection costs, in
    metres. On the 2026-08-25 tf_multi run that was 0.77 m of 1.71 m total at epoch 22.
    """

    def __init__(self, name: str = "min_ade_6") -> None:
        self._name = name

    def name(self) -> str:
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        modes = predictions["modes"]                               # (B, K, T, 3)
        target = cast(Trajectory, targets["trajectory"]).data      # (B, T, 3)
        ade = torch.norm(modes[..., :2] - target[:, None, :, :2], dim = -1)  # (B,K,T)
        ade = ade.mean(-1) # (B, K)
        ade = ade.min(-1).values #(B,)
        return ade.mean(-1)