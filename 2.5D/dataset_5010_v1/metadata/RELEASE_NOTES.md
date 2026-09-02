# Release notes — dataset_5010_v1

## Included

- 5,010 paired 2.5D terrain maps
- 5,010 pickle-free NPZ raster companions
- 50,100 published map-task records
- 10,020 terrain and analysis visualizations
- 30,060 canonical planner route visualizations
- 90,180 canonical-task benchmark trials
- Map-level benchmark tables, paired-difficulty summaries, and four figures
- Fixed train/validation/test indexes assigned at matched-triplet level and
  stratified exactly by terrain family

## Verified

- Structural validator: 5,010 maps and 50,100 map-task records valid
- Planner audit: zero invalid successful paths and zero missing route images
- Difficulty audit: zero relief or navigability ordering violations
- Optimality cross-check: exact A*/Dijkstra weighted-cost agreement on all maps
- Split audit: all 1,670 matched groups assigned once, with zero group leakage
- NPZ audit: all 5,010 companions pass CRC, field-set, NPY-header, shape, and
  payload-length checks

## Statistical unit

The five stochastic trials are nested within each map-algorithm unit. Benchmark
mean and sample standard deviation are computed across maps after within-map
aggregation. Easy, Medium, and Hard members are paired terrain variants.

## Fixed splits

The published split unit is `matched_group_id`, so all Easy, Medium, and Hard
members and their ten shared tasks remain together. Within each of the five
terrain families, 234 groups are assigned to train, 50 to validation, and 50 to
test. The deterministic SHA-256 assignment protocol and seed are recorded in
`metadata/splits/split_protocol.json`.

## Runtime

Planner runtime was recorded during a three-worker throughput run. Report this
protocol explicitly; the supplied timing is not isolated single-core latency.

## NPZ companion

JSON remains the canonical self-describing map record. The same-stem NPZ files
provide the 13 raster layers as `float32` or `uint8` arrays for faster numerical
loading with `numpy.load(..., allow_pickle=False)`. Per-file paths and hashes
are published in `metadata/npz_manifest.csv` and `metadata/SHA256SUMS_NPZ.txt`.

## Packaging

The Zenodo deposit is partitioned into one compact core ZIP, one NPZ companion
ZIP, and five terrain-family ZIPs. Split indexes, documentation, summary tables,
reports, and checksums are in the core ZIP; the unchanged JSON map and image
payload remains in the family ZIPs.
