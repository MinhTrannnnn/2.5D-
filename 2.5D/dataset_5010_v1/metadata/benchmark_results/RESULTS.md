# Paper-ready 2.5D dataset results

## Scope

- 5,010 maps in 1,670 matched Easy/Medium/Hard triplets
- 10 published Start-Goal tasks per triplet; the canonical long task is used here
- 6 planners and 5 trials for each stochastic planner
- 90,180 raw trial rows, aggregated to 30,060 map-algorithm experimental units
- Mean and sample standard deviation are computed across maps after within-map trial aggregation

## Integrity audit

- Audit valid: **True**
- Successful paths failing validation: 0
- Missing path images: 0
- Relief ordering violations: 0
- Navigability ordering violations: 0
- A*/Dijkstra optimal-cost mismatches: 0

## Overall planner results

The success statistic is the mean per-map trial success rate. Cost is normalized
by the A* optimum on the same map and Start-Goal task. Path metrics are computed
over successful solutions only.

| Algorithm | Success (%) | Cost ratio to A* | Runtime (s) |
|---|---:|---:|---:|
| BFS | 100.00 ± 0.00 | 1.034 ± 0.031 | 0.075 ± 0.022 |
| Dijkstra | 100.00 ± 0.00 | 1.000 ± 0.000 | 0.077 ± 0.021 |
| A* | 100.00 ± 0.00 | 1.000 ± 0.000 | 0.035 ± 0.017 |
| PRM | 88.86 ± 26.67 | 1.103 ± 0.132 | 0.106 ± 0.009 |
| RRT-Connect | 93.89 ± 19.92 | 1.392 ± 0.248 | 0.045 ± 0.100 |
| RRT* | 92.57 ± 22.39 | 1.025 ± 0.081 | 3.789 ± 0.995 |

## Interpretation notes

- A* and Dijkstra provide a cross-check of the common weighted objective.
- BFS is a minimum-hop baseline and is not expected to minimize weighted 2.5D cost.
- PRM, RRT-Connect, and RRT* are fixed-budget stochastic baselines without hidden retries.
- Easy, Medium, and Hard are paired terrain variants, not independent samples.
- The canonical benchmark evaluates the longest published task per map; the other
  nine tasks remain part of the released dataset but are not included in this table.

## Runtime caveat

- runtime was measured during a multi-worker throughput run; report the execution protocol and avoid presenting it as isolated single-core latency

## Generated files

- `tables/planner_map_level.csv`: one row per map and algorithm
- `tables/planner_summary_family_difficulty.csv`: paper-level family/difficulty summary
- `tables/planner_summary_overall.csv`: overall and by-difficulty summary
- `tables/dataset_difficulty_summary.csv`: terrain calibration summary
- `tables/paired_dataset_difficulty_effects.csv`: paired E/M/H terrain changes
- `tables/paired_planner_difficulty_effects.csv`: paired planner changes
- `figures/dataset_difficulty_overview.png`
- `figures/planner_benchmark_overview.png`
- `figures/planner_success_heatmap.png`
- `figures/planner_cost_ratio_heatmap.png`
