# Results

All scores on the 1,349 held-out nuPlan test logs. `exp/` is gitignored; this file is the
canonical record.

## Run name mapping

`exp/` directories keep their original names — `.nuboard` files embed absolute paths, so
renaming them breaks replay.

| exp/ directory | run name | notes |
|---|---|---|
| `BC_model_v2c_noagent_635k` | `tf_multi/noego/balanced635k` | main model |
| `BC_model_v2c_uniform_635k` | `tf_multi/noego/uniform635k` | PlanTF-style sampling control |
| `BC_model_v2a_635k_experiment` | `tf_single/ego/balanced635k` | single-mode, ego history |
| `BC_model_v2a_20k_experiment` | `tf_single/ego/mini13k` | 13.4k data-scaling reference |
| `BC_model_v2b_20k_experiment` | `tf_multi/ego/mini13k` | |
| `BC_model_v2b_20k_noegohist_experiment` | `tf_multi/noego/mini13k` | |
| `BC_model_v2c_20k_experiment` | `tf_multi/noego/agents/mini13k` | only run with agent prediction |
| `BC_model_v0*`, `BC_model_v1*`, `simple_vector_pipeline_*` | legacy MLPs | see `legacy/` |

## Main table — Test14-random / Test14-hard

Baselines are PlanTF's published numbers. IDM and PDM-Closed were also re-run locally as a
setup check and matched the published values, so the table is treated as comparable.

| Method | rand OLS | rand NR-CLS | rand R-CLS | hard OLS | hard NR-CLS | hard R-CLS |
|---|---|---|---|---|---|---|
| LogReplay (expert) | 100.0 | 94.03 | 75.86 | 100.0 | 85.96 | 68.80 |
| PDM-Closed | 46.32 | 90.05 | 91.64 | 26.43 | 65.07 | 75.18 |
| IDM | 34.15 | 70.39 | 72.42 | 20.07 | 56.16 | 62.26 |
| PlanTF | 87.07 | 86.48 | 80.59 | 83.32 | 72.68 | 61.70 |
| PlanCNN | 62.93 | 69.66 | 67.54 | 52.40 | 49.47 | 52.16 |
| UrbanDriver | 82.44 | 63.27 | 61.02 | 76.90 | 51.54 | 49.07 |
| **tf_multi/noego/balanced635k** | **66.19** | **62.15** | **60.12** | **64.22** | **44.16** | **43.63** |
| GC-PGP | 77.33 | 55.99 | 51.39 | 73.78 | 43.22 | 39.63 |
| PDM-Open | 84.14 | 52.80 | 57.23 | 79.06 | 33.51 | 35.83 |

## Ablations (Test14-random, NR-CLS, 261 scenarios)

| run | modes | ego hist | agent pred | training set | val ADE | NR-CLS |
|---|---|---|---|---|---|---|
| `tf_multi/noego/balanced635k` | 6 | off | off | 634,630 balanced, 6.29% turns | 2.4973 | **62.15** |
| `tf_multi/noego/uniform635k` | 6 | off | off | 634,630 proportional, 0.809% turns | 2.4652 | 56.44 |
| `tf_single/ego/balanced635k` | 1 | on | — | 634,630 balanced | 1.9718 | 51.84 |
| `tf_single/ego/mini13k` | 1 | on | — | 13,400 | — | 23.21 |

## Open loop does not predict closed loop

| # | change | open loop | closed loop |
|---|---|---|---|
| 1 | ego history on | better (1.972 vs 2.497 ADE) | worse (51.84 vs 62.15) |
| 2 | proportional sampling | better (2.465 vs 2.497 ADE) | worse (56.44 vs 62.15) |
| 3 | Test14-hard vs random | −3.0% | −28.9% |

## Balanced vs proportional — significance

The benchmark score is type-weighted; the raw per-scenario difference is smaller.

| test | result |
|---|---|
| weighted benchmark score | +5.71 points (+10.1%) |
| sign test, 188 non-tied scenarios | 120 vs 68, z=3.79, **p = 1.5e-4** |
| paired t-test on magnitudes | +0.0128, 95% CI [−0.0038, +0.0294], **not significant** |

Report the direction, not a large effect size.

## Data scaling (controlled: same architecture, same benchmark)

| | scenarios | NR-CLS |
|---|---|---|
| `tf_single/ego/mini13k` | 13,400 | 23.21 |
| `tf_single/ego/balanced635k` | 634,630 | 51.84 |

+123%. Gate failures:

| gate | 13.4k | 635k |
|---|---|---|
| drivable_area_compliance | 46.7 | 10.7 |
| driving_direction_compliance | 13.0 | 2.3 |
| no_ego_at_fault_collisions | 47.9 | 24.5 |
| ego_is_making_progress | 18.4 | 10.0 |

## Gate failures (% of scenarios scoring 0)

| gate | balanced | uniform | ego-history | hard/nonreact |
|---|---|---|---|---|
| no_ego_at_fault_collisions | 20.7 | 24.5 | 24.5 | 30.5 |
| drivable_area_compliance | 13.4 | 12.6 | 10.7 | 19.1 |
| ego_is_making_progress | 3.4 | 6.9 | 10.0 | 11.0 |
| driving_direction_compliance | 4.2 | 4.6 | 2.3 | 7.0 |

Collisions are the binding constraint (20.7% vs PDM's 3.1%). The ego-history model fails
`ego_is_making_progress` 3x more often — it stalls itself rather than steering badly.

## Training data

Raw train pool over 5,603 Boston/Pittsburgh/Singapore logs: 7,615,338 scenarios,
34.3% `unknown`, 39,924 turn scenarios (0.524%).

| | balanced | proportional |
|---|---|---|
| train scenarios | 634,630 | 634,630 |
| turn scenarios | 39,924 (6.29%) | 5,137 (0.809%) |
| share of available turns | 100% | 12.7% |
| `unknown` | 25,000 | 0 |
| `stationary` | 25,000 | 131,603 |
| distinct types | 56 | 53 |

`limit_total_scenarios` drops all `unknown` first, then samples the remainder
proportionally — so PlanTF's filter is already mildly rebalanced, and at the tail it deletes
rare types outright (56 types vs 53).

## Known limitations

- `starting_left_turn` has zero training examples at any dataset size: the scenario builder
  assigns one label per scenario, so left turns are absorbed into other tags. It is still one
  of the 14 evaluated types, and it is the worst (0.647, largest gap to PDM at +0.300).
- n=20 per scenario type. One scenario flipping moves a type mean by 0.05, so per-type
  numbers are illustrative, not evidence.
- `tf_single/ego/mini13k` uses 16 output poses vs 80 for the 635k runs. Measured
  interpolation error is ~0.024 m against ~2 m model ADE, but it is a second variable.
