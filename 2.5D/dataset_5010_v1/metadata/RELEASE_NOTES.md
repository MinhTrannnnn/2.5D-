# Release notes — dataset_5010_v1

## Included

- 5,010 paired 2.5D terrain maps
- 50,100 published map-task records
- 10,020 terrain and analysis visualizations
- 30,060 canonical planner route visualizations
- 90,180 canonical-task benchmark trials
- Map-level benchmark tables, paired-difficulty summaries, and four figures

## Verified

- Structural validator: 5,010 maps and 50,100 map-task records valid
- Planner audit: zero invalid successful paths and zero missing route images
- Difficulty audit: zero relief or navigability ordering violations
- Optimality cross-check: exact A*/Dijkstra weighted-cost agreement on all maps

## Statistical unit

The five stochastic trials are nested within each map-algorithm unit. Benchmark
mean and sample standard deviation are computed across maps after within-map
aggregation. Easy, Medium, and Hard members are paired terrain variants.

## Runtime

Planner runtime was recorded during a three-worker throughput run. Report this
protocol explicitly; the supplied timing is not isolated single-core latency.

## Packaging

This directory is release-ready but intentionally not duplicated into a local
29 GB archive. Create the final archive only after choosing the repository or
upload service and its preferred archive/chunking format.
