# Paired 2.5D Terrain Dataset and Planning Benchmark

> **Permanent archive.** The complete version 1.0.0 dataset is archived on
> Zenodo at <https://doi.org/10.5281/zenodo.22074838>. The GitHub repository
> versions the schema, reproducibility material, summary tables, checksums,
> paper figures, and contact sheets; the large map, full-resolution image, and
> raw-trial payload is intentionally excluded from ordinary Git history.

This release contains deterministic 2.5D height-field terrains for ground-robot
path-planning research. Each `(x, y)` cell has one elevation `z`; the robot
footprint, local grade, step height, roughness, traversability cost, and binary
collision grid are explicitly represented.

## Release contents

- 5,010 map JSON files
- 1,670 matched Easy/Medium/Hard terrain triplets
- 5 terrain families, with 334 triplets per family
- 10 shared Start-Goal tasks per triplet: 3 short, 4 medium, and 3 long
- 50,100 map-task records
- 5,010 terrain PNGs and 5,010 traversability-analysis PNGs
- 30,060 canonical route PNGs: one per map and planner
- 90,180 raw canonical-task benchmark trials

Terrain families:

- `smooth_obstacles`
- `rolling`
- `mountain`
- `rugged`
- `plateau`

## Matched difficulty design

Easy, Medium, and Hard members share the same base terrain realization and the
same ten Start-Goal coordinates. They are paired samples, not statistically
independent maps. Relief severity strictly increases and navigable fraction
strictly decreases within every one of the 1,670 triplets.

General families target 72-94%, 54-76%, and 40-65% navigable area for Easy,
Medium, and Hard. Plateau uses family-calibrated bands of 80-90%, 73-82%, and
69-76% because traversable mesa tops remain valid robot poses.

## Planner benchmark

Six fixed-protocol planners are included:

- BFS
- Dijkstra
- A*
- PRM
- RRT-Connect
- RRT*

The benchmark uses the canonical long-distance task on every map. BFS,
Dijkstra, and A* run once. PRM, RRT-Connect, and RRT* run five deterministic
seeded trials without hidden retries. All planners and the final validator share
the same 8-connected movement, no-corner-cutting, footprint collision, edge
slope, and weighted 2.5D cost semantics.

For paper statistics, stochastic trials are first aggregated within each map.
Mean and sample standard deviation are then computed across maps, so repeated
trials are not incorrectly treated as independent terrain samples.

Runtime was measured during a three-worker throughput run. It should be reported
with that protocol and not described as isolated single-core latency.

## Structure

```text
dataset_5010_v1/
|-- LICENSE
|-- maps/<family>/
|-- images/
|   |-- terrain/<family>/
|   |-- analysis/<family>/
|   |-- paths/<family>/
|   `-- contact_sheets/
|-- metadata/
|   |-- dataset_summary.csv
|   |-- task_summary.csv
|   |-- manifest.json
|   |-- validation_report.json
|   |-- pathfinding_benchmark_v1_trials.csv
|   |-- pathfinding_benchmark_v1_summary.csv
|   |-- pathfinding_benchmark_v1_protocol.json
|   |-- RELEASE_NOTES.md
|   |-- release_manifest.json
|   |-- release_filelist.csv
|   |-- SHA256SUMS_CORE.txt
|   |-- SHA256SUMS_MAPS.txt
|   `-- benchmark_results/
|       |-- RESULTS.md
|       |-- benchmark_audit.json
|       |-- tables/
|       `-- figures/
+-- README.md
```

Map and image files are partitioned by family. Each family contains 1,002 map
JSON files and the corresponding images. `metadata/dataset_summary.csv` is the
canonical cross-family map index; tasks are embedded in each map JSON and
indexed compactly by `metadata/task_summary.csv`.

## Reading a map

See [`SCHEMA.md`](SCHEMA.md) for the complete field dictionary, array indexing,
coordinate conversion, collision encoding, terrain-analysis layers, task
semantics, and weighted path objective. A dependency-free reader is provided at
`examples/load_map.py` and can be run from the dataset directory:

```bash
python examples/load_map.py
python examples/load_map.py maps/plateau/terrain_plateau_010_hard.json
```

Important conventions: raster layers are indexed `[y][x]`, task cells are
stored `[x, y]`, `grid == 0` is traversable, and planners use
`terrain_analysis.support_elevation` rather than raw surface elevation for edge
grade and weighted cost.

## Validation status

The final audit reports:

- 5,010/5,010 maps structurally valid
- 50,100/50,100 map-task records valid
- zero invalid successful planner paths
- zero missing terrain, analysis, or route images
- zero Easy/Medium/Hard ordering violations
- exact A*/Dijkstra weighted-cost agreement on all 5,010 canonical tasks

See `metadata/validation_report.json` and
`metadata/benchmark_results/benchmark_audit.json` for machine-readable reports.
The release inventory and integrity lists are in `metadata/release_manifest.json`,
`metadata/release_filelist.csv`, and `metadata/SHA256SUMS_*.txt`.

## Licence

The dataset is released under the Creative Commons Attribution 4.0
International (CC BY 4.0) License. See `LICENSE` for the licence notice and
canonical terms.

## Representation limit

This is a 2.5D height-field dataset rather than a voxel or mesh-world dataset.
It cannot represent caves, bridges, overhangs, or multiple surfaces at the same
`(x, y)` coordinate.
