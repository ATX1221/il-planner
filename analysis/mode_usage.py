import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DEVKIT_DIR = os.environ.get("NUPLAN_DEVKIT_ROOT", str(REPO_DIR.parent / "nuplan-devkit"))
os.sched_setaffinity(0, set(os.sched_getaffinity(0)) - {10, 11})
for k, v in {"NUPLAN_DATA_ROOT": f"{DEVKIT_DIR}/nuplan/dataset",
             "NUPLAN_MAPS_ROOT": f"{DEVKIT_DIR}/nuplan/dataset/maps",
             "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
             "NUPLAN_EXP_ROOT": str(REPO_DIR / "exp")}.items():
    os.environ.setdefault(k, v)
import torch
from hydra import initialize, compose
from nuplan.planning.script.builders.training_builder import build_lightning_datamodule, build_lightning_module
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.worker_pool_builder import build_worker

# exp/ directories keep their original run names; see the mapping in docs/results.md.
# Override with CKPT=<path> in the environment.
CKPT = os.environ.get(
    "CKPT",
    "exp/BC_model_v2b_experiment/bc_model_v2b/2026.08.25.08.33.42/best_model/epoch=36-step=1479.ckpt")
with initialize(config_path="../nuplan/planning/script/config/training"):
    cfg = compose("default_training", overrides=["experiment_name=m","group=/tmp/m",
      "hydra.searchpath=[pkg://nuplan.planning.script.config.common,"
      " pkg://nuplan.planning.script.experiments,"
      f" file://{REPO_DIR}/config]",
      "+training=tf_multi_ego_mini13k","scenario_filter.limit_total_scenarios=300",
      "data_loader.params.batch_size=32","worker.threads_per_node=4"])
worker = build_worker(cfg)
tm = build_torch_module_wrapper(cfg.model)
lm = build_lightning_module(cfg, tm)
dm = build_lightning_datamodule(cfg, worker, tm); dm.setup("fit")
lm.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"])
model = lm.model.eval()

K = tm.num_modes
win = torch.zeros(K); pick = torch.zeros(K); agree = n = 0
spread = []
with torch.no_grad():
    for i, (f, t, s) in enumerate(dm.val_dataloader()):
        out = model(f)
        modes, pi = out["modes"], out["probability"]
        tgt = t["trajectory"].data
        ade = torch.norm(modes[..., :2] - tgt[:, None, :, :2], dim=-1).sum(-1)   # (B,K)
        best = ade.argmin(-1); chosen = pi.argmax(-1)
        win += torch.bincount(best, minlength=K).float()
        pick += torch.bincount(chosen, minlength=K).float()
        agree += (best == chosen).sum().item(); n += len(best)
        # how different are the modes from each other?
        spread.append((modes[:, :, -1, :2].std(dim=1)).mean().item())
        if i >= 9: break

print(f"samples: {n}")
print(f"\nwinner  (argmin over geometry) : {win.int().tolist()}")
print(f"chosen  (argmax over pi)       : {pick.int().tolist()}")
print(f"\nclassifier agrees with the true winner: {100*agree/n:.1f}%   (random would be {100/K:.1f}%)")
print(f"mean std across modes of the FINAL waypoint: {sum(spread)/len(spread):.3f} m")
print("   (near 0 = all six modes predict the same trajectory = collapsed)")
