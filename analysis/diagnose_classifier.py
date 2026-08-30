"""Why is the mode classifier bad? Measure, don't guess."""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DEVKIT_DIR = os.environ.get("NUPLAN_DEVKIT_ROOT", str(REPO_DIR.parent / "nuplan-devkit")), sys
os.sched_setaffinity(0, set(os.sched_getaffinity(0)) - {10, 11})
for k, v in [("NUPLAN_DATA_ROOT", f"{DEVKIT_DIR}/nuplan/dataset"),
             ("NUPLAN_MAPS_ROOT", f"{DEVKIT_DIR}/nuplan/dataset/maps"),
             ("NUPLAN_EXP_ROOT", str(REPO_DIR / "exp"))]:
    os.environ.setdefault(k, v)
sys.path.insert(0, str(REPO_DIR))
import logging; logging.disable(logging.INFO)
import torch, torch.nn.functional as F, hydra

_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False, "map_location": "cpu"})

from nuplan.planning.script.builders.training_builder import build_lightning_datamodule
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.worker_pool_builder import build_worker

EXP = str(REPO_DIR / "exp")
# exp/ directories keep their original run names; see the mapping in docs/results.md.
# Override with CKPT=<path> in the environment.
CKPT = os.environ.get(
    "CKPT",
    f"{EXP}/BC_model_v2b_20k_experiment/bc_model_v2b/2026.08.25.23.38.20/checkpoints/epoch=22.ckpt")

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=f"{DEVKIT_DIR}/nuplan/planning/script/config/training")
cfg = hydra.compose(config_name="default_training", overrides=[
    "experiment_name=probe", f"group={EXP}",
    "hydra.searchpath=[pkg://nuplan.planning.script.config.common,"
    " pkg://nuplan.planning.script.experiments,"
    f" file://{REPO_DIR}/config]",
    "+training=tf_multi_ego_mini13k", "~callbacks.visualization_callback",
    "scenario_filter.limit_total_scenarios=20000",
    "data_loader.params.batch_size=32", "worker.threads_per_node=8",
    "cache.force_feature_computation=true", "lightning.trainer.params.max_epochs=1",
])
worker = build_worker(cfg)
model = build_torch_module_wrapper(cfg.model)
dm = build_lightning_datamodule(cfg, worker, model); dm.setup("fit")
val = dm.val_dataloader()

sd = torch.load(CKPT)["state_dict"]
model.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")})
model.eval()
K = model.num_modes

acc = n = 0
ade_oracle = ade_chosen = ade_random = ade_worst = 0.0
ent = 0.0
conf_when_right = conf_when_wrong = nr = nw = 0.0
spread = 0.0
with torch.no_grad():
    for i, (feats, tgts, _) in enumerate(val):
        if i >= 12:
            break
        out = model(feats)
        modes, pi = out["modes"], out["probability"]
        tgt = tgts["trajectory"].data
        B = modes.shape[0]
        ade = torch.norm(modes[..., :2] - tgt[:, None, :, :2], dim=-1).mean(-1)   # (B,K)
        best = ade.argmin(-1)
        chosen = pi.argmax(-1)

        acc += (chosen == best).sum().item()
        ade_oracle += ade.min(-1).values.sum().item()
        ade_chosen += ade[torch.arange(B), chosen].sum().item()
        ade_random += ade.mean(-1).sum().item()          # expectation of a random pick
        ade_worst += ade.max(-1).values.sum().item()
        # how much better is the best mode than the average one? if modes are near-identical
        # the classifier CANNOT matter, however good it is.
        spread += (ade.max(-1).values - ade.min(-1).values).sum().item()

        p = pi.softmax(-1)
        ent += (-(p * p.clamp_min(1e-9).log()).sum(-1)).sum().item()
        right = chosen == best
        conf_when_right += p.max(-1).values[right].sum().item(); nr += right.sum().item()
        conf_when_wrong += p.max(-1).values[~right].sum().item(); nw += (~right).sum().item()
        n += B

import math
print(f"\n=== classifier diagnosis: K={K}, {n} val samples, checkpoint epoch 22\n")
print(f"top-1 accuracy at picking the best mode : {acc/n*100:5.1f}%   (chance = {100/K:.1f}%)")
print(f"softmax entropy                         : {ent/n:5.3f}   (uniform = {math.log(K):.3f}, certain = 0)")
print(f"mean confidence when RIGHT              : {conf_when_right/max(nr,1):5.3f}")
print(f"mean confidence when WRONG              : {conf_when_wrong/max(nw,1):5.3f}")
print(f"\nADE by selection strategy (metres):")
print(f"  oracle  (always the best mode)        : {ade_oracle/n:6.3f}")
print(f"  learned (argmax pi)  <- what you drive: {ade_chosen/n:6.3f}")
print(f"  random  (expectation of a coin flip)  : {ade_random/n:6.3f}")
print(f"  worst   (always the worst mode)       : {ade_worst/n:6.3f}")
gain = (ade_random - ade_chosen) / max(ade_random - ade_oracle, 1e-9)
print(f"\n  classifier captures {gain*100:5.1f}% of the available oracle-vs-random gain")
print(f"  mode spread (worst - best ADE)        : {spread/n:6.3f} m")

# does gradient actually reach the pi head?
model.train()
feats, tgts, scen = next(iter(val))
out = model(feats)
modes, pi = out["modes"], out["probability"]
tgt = tgts["trajectory"].data
best = torch.norm(modes[..., :2] - tgt[:, None, :, :2], dim=-1).mean(-1).argmin(-1)
model.zero_grad()
F.cross_entropy(pi, best.detach()).backward()
gpi = sum(p.grad.abs().sum().item() for p in model.pi.parameters() if p.grad is not None)
gproj = sum(p.grad.abs().sum().item() for p in model.multimodal_proj.parameters() if p.grad is not None)
gloc = sum(p.grad.abs().sum().item() for p in model.loc.parameters() if p.grad is not None)
print(f"\ngradient from cls reaching  self.pi              : {gpi:.4f}")
print(f"gradient from cls reaching  self.multimodal_proj : {gproj:.4f}")
print(f"gradient from cls reaching  self.loc             : {gloc:.4f}  (should be 0)")
