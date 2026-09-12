# IL-Planner

A vectorised transformer planner for [nuPlan](https://www.nuscenes.org/nuplan), built by
starting from the devkit's vector-map baseline and progressively rebuilding it, using
[PlanTF](https://github.com/jchengai/planTF) as a reference for the encoder design and the
ego-history ablation.

Evaluated on Test14-random and Test14-hard over the 1,349 held-out nuPlan test logs.

**`tf_multi` — 755K parameters, 634,630 training scenarios, 62.15 NR-CLS on Test14-random.**

https://github.com/user-attachments/assets/f4290524-6a28-4597-856f-87eeb19d55c6

## What this project set out to test

Whether a small planner's closed-loop score is limited by **how much** data it sees or by
**which** scenarios it sees — and whether open-loop displacement error is a usable proxy for
closed-loop driving.

Both questions were answered by controlled ablations where a single variable changes and
model, features, validation set and benchmark are held fixed.

## Results

<p>Baselines are PlanTF's published numbers.</p>

<table>
  <thead>
    <tr>
      <th rowspan="2">Type</th>
      <th rowspan="2">Planner</th>
      <th colspan="3">Test14-Random</th>
      <th colspan="3">Test14-Hard</th>
    </tr>
    <tr>
      <th>OLS↑</th>
      <th>NR-CLS↑</th>
      <th>R-CLS↑</th>
      <th>OLS↑</th>
      <th>NR-CLS↑</th>
      <th>R-CLS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Expert</td>
      <td>LogReplay</td>
      <td>100.0</td>
      <td>94.03</td>
      <td>75.86</td>
      <td>100.0</td>
      <td>85.96</td>
      <td>68.80</td>
    </tr>
    <tr>
      <td>Rule</td>
      <td>PDM-Closed</td>
      <td>46.32</td>
      <td>90.05</td>
      <td>91.64</td>
      <td>26.43</td>
      <td>65.07</td>
      <td>75.18</td>
    </tr>
    <tr>
      <td>Rule</td>
      <td>IDM</td>
      <td>34.15</td>
      <td>70.39</td>
      <td>72.42</td>
      <td>20.07</td>
      <td>56.16</td>
      <td>62.26</td>
    </tr>
    <tr>
      <td>Learning</td>
      <td>PlanTF</td>
      <td>87.07</td>
      <td>86.48</td>
      <td>80.59</td>
      <td>83.32</td>
      <td>72.68</td>
      <td>61.70</td>
    </tr>
    <tr>
      <td>Learning</td>
      <td>PlanCNN</td>
      <td>62.93</td>
      <td>69.66</td>
      <td>67.54</td>
      <td>52.40</td>
      <td>49.47</td>
      <td>52.16</td>
    </tr>
    <tr>
      <td>Learning</td>
      <td>UrbanDriver</td>
      <td>82.44</td>
      <td>63.27</td>
      <td>61.02</td>
      <td>76.90</td>
      <td>51.54</td>
      <td>49.07</td>
    </tr>
    <tr>
      <td><b>Learning</b></td>
      <td><b><code>tf_multi/noego/balanced635k</code></b></td>
      <td><b>66.19</b></td>
      <td><b>62.15</b></td>
      <td><b>60.12</b></td>
      <td><b>64.22</b></td>
      <td><b>44.16</b></td>
      <td><b>43.63</b></td>
    </tr>
    <tr>
      <td>Learning</td>
      <td>GC-PGP</td>
      <td>77.33</td>
      <td>55.99</td>
      <td>51.39</td>
      <td>73.78</td>
      <td>43.22</td>
      <td>39.63</td>
    </tr>
    <tr>
      <td>Learning</td>
      <td>PDM-Open</td>
      <td>84.14</td>
      <td>52.80</td>
      <td>57.23</td>
      <td>79.06</td>
      <td>33.51</td>
      <td>35.83</td>
    </tr>
  </tbody>
</table>


### Ablations — Test14-random, NR-CLS, 261 scenarios

| run | modes | ego hist | training set | val ADE | NR-CLS |
|---|---|---|---|---|---|
| `tf_multi/ego/balanced635k` | 6 | on | 634,630 balanced | 1.972 | 51.84 |
| `tf_multi/noego/balanced635k` | 6 | off | 634,630 balanced, 6.29% turns | 2.497 | **62.15** |
| `tf_multi/noego/uniform635k` | 6 | off | 634,630 proportional, 0.809% turns | 2.465 | 56.44 |
| `tf_single/ego/mini13k` | 1 | on | 13,400 | — | 23.21 |

"Balanced" and "proportional" describe how the training set was drawn from nuPlan's raw
scenario pool; see Finding 3.

## Findings

### 1. Data volume dominates, but it does not lift every failure mode equally

Same architecture, same benchmark, only the training-set size changes:

| | scenarios | NR-CLS |
|---|---|---|
| `tf_single/ego/mini13k` | 13,400 | 23.21 |
| `tf_single/ego/balanced635k` | 634,630 | **51.84** |

47x the data more than doubles the score, but not evenly across failure modes:

```
gate failure (%)              13.4k    635k
drivable_area_compliance      46.7  →  10.7     4.4x
driving_direction_compliance  13.0  →   2.3     5.7x
no_ego_at_fault_collisions    47.9  →  24.5     2.0x
ego_is_making_progress        18.4  →  10.0     1.8x
```

Every gate improves, but not at the same rate. Road-geometry competence — staying inside the
drivable area, facing the right way — improves fastest and is close to solved. Collisions
improve half as much and are left as the binding constraint at 24.5%, against PDM-Closed's
3.1%. Scaling the dataset further would keep helping, but this split suggests it would run
into interaction with other agents long before it runs out of road-geometry errors to fix.

### 2. Open-loop error does not predict closed-loop driving

PlanTF identified this for ego history, attributing it to a copycat shortcut. The same
divergence shows up here in two further places with unrelated causes:

| # | change | open loop | closed loop |
|---|---|---|---|
| 1 | ego history on | better — 1.972 vs 2.497 ADE | worse — 51.84 vs 62.15 |
| 2 | proportional sampling | better — 2.465 vs 2.497 ADE | worse — 56.44 vs 62.15 |
| 3 | Test14-hard vs Test14-random | −3.0% | **−28.9%** |

Each time, the model with the lower displacement error drives worse. Only case 1 is
architectural, so the divergence looks like a property of the metrics themselves: ADE
averages smoothly over every timestep, while the closed-loop score multiplies four binary
gates and is decided by rare failures. Validation ADE is therefore not usable for
checkpoint selection.

The ego-history failure has a clear signature: it fails `ego_is_making_progress` 3x more
often than the no-history model while scoring *better* on drivable area. It does not steer
badly — it stalls, reading its own deceleration as reason to decelerate further.

### 3. At equal dataset size, composition still matters

nuPlan tags every scenario with a single type (`stationary`, `following_lane_with_lead`,
`starting_right_turn`, and ~70 others). The pool drawn on here is 7,615,338 scenarios from
the 5,603 Boston, Pittsburgh and Singapore training logs — the Las Vegas split, roughly 90%
of nuPlan's training data by volume, was not used. That pool is heavily skewed toward a few
common types. Two ways to draw a fixed-size training set from it:

- **Proportional** — keep the natural frequencies. The devkit default.
- **Type-balanced** — cap every type at the same ceiling, 25,000 here. Rare types keep all
  their scenarios; common types are truncated.

Both caches hold exactly 634,630 scenarios, so only the composition differs:

| | balanced | proportional |
|---|---|---|
| turn scenarios | 39,924 (6.29%) | 5,137 (0.809%) |
| `stationary` | 25,000 | 131,603 |
| NR-CLS | **62.15** | 56.44 |

Balancing wins more often than it loses. Of the 261 test scenarios, 188 scored differently
under the two models, and balanced was ahead on 120 of them — a margin unlikely to come from
chance (p = 1.5e-4).

The size of the win needs one caveat. The headline 56.44 → 62.15 is the benchmark's own
score, which weights all 14 scenario types equally, and balancing helps most on the types
that are rare in the raw data. Averaged flat across all 261 scenarios instead, the gap is
+1.3 points rather than +5.7, and that flat average is noisy enough to include zero. Both
numbers are real; the benchmark's weighting is what separates them.

Balancing did **not** help turns specifically (+0.0146 mean delta on turn types against
+0.0123 elsewhere). The likelier mechanism is redundancy removal: cutting `stationary` from
131,603 to 25,000 frees ~106,000 slots for varied data of every kind.

## Layout

```
il_planner/                  the installable package
  features/
    simple_feature.py        SimpleFeatureBuilder - ego history, up to 32 neighbour agent
                             tracks, and sampled map polylines in an ego-centric frame.
                             Also writes per-agent future targets, used only when a model
                             sets predict_agents. Cache keys derive from this builder's
                             CONFIG, never its source.
    feature_utils.py         geometry helpers (frame rotation, polyline sampling)
  models/
    transformer_planner.py       tf_multi - K-mode head, trained winner-take-all
    transformer_planner_single.py tf_single - one trajectory, plain L2
    ego_state_encoder.py     StateAttentionEncoder, used when use_ego_history=False.
                             Embeds each ego scalar as its own token and pools with a
                             learned query, so a dropped channel masks a key rather than
                             feeding a zero the model could read as a measurement.
  objectives/
    wta_objective.py         WinnerTakesAllObjective + AgentPredictionObjective
    weighted_imitation_objective.py  scenario-type-weighted L2
  metrics/
    min_ade_metric.py        minADE over the mode dimension

config/                      hydra configs, mirroring the devkit layout so they resolve
  model/                     tf_multi.yaml, tf_single.yaml - architecture + feature builder
  training/                  one file per run, named <arch>_<egohist>_<dataset>
  objective/  training_metric/

scripts/
  train.py                   single runner; takes a config/training name
  simulate.py                evaluates a checkpoint on a challenge + scenario filter
  cloud/                     helpers for bulk feature-cache construction

analysis/                    diagnostics behind the findings above
  diagnose_classifier.py     mode-selection accuracy, calibration, oracle gap
  mode_usage.py              how often each of the K modes wins

legacy/                      earlier MLP models, their configs and best checkpoints
compat/                      import shims so simulation logs recorded before the package
                             restructure still unpickle in nuBoard
tests/                       feature-builder and model shape tests
docs/results.md              full record, including exp/ directory name mapping
```

## Setup

```bash
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit && pip install -e .
git clone <this repo> && cd il-planner
pip install -r requirements.txt && pip install -e .
```

`requirements.txt` departs from the devkit's pins in two places: torch 2.8 rather than 1.9
(the models use `nn.TransformerEncoderLayer(norm_first=True)`, added in 1.10), and a
matching torchvision. `torch-scatter` is omitted — nothing here imports it.

## Training

```bash
python scripts/train.py tf_multi_noego_balanced635k --cache ~/nuplan/exp/cache
```

Reads only from the feature cache, so the trainval `.db` files are not needed once it is
built. Checkpoints are selected on `metrics/val_avg_displacement_error`, not `val_loss` —
`val_loss` includes the mode-classification term, which legitimately rises as modes
specialise and selects far too early.

## Evaluation

```bash
python scripts/simulate.py --ckpt <path> -c closed_loop_nonreactive_agents --filter test14-random
```

`data_root` must point at the test logs only. `test14-random` selects by scenario type with
no log filter, so aimed at a full dataset it will silently draw scenarios from the training
split.

## Acknowledgements

Built on [nuplan-devkit](https://github.com/motional/nuplan-devkit); the initial model was
adapted from its vector-map baseline. Encoder design, the ego-history ablation and the
Test14 benchmark definitions follow [planTF](https://github.com/jchengai/planTF). PDM
baselines from [tuplan_garage](https://github.com/autonomousvision/tuplan_garage).
