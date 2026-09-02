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
- 5,010 pickle-free NPZ raster companions for faster array loading
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
|-- npz/<family>/
|-- images/
|   |-- terrain/<family>/
|   |-- analysis/<family>/
|   |-- paths/<family>/
|   `-- contact_sheets/
|-- metadata/
|   |-- dataset_summary.csv
|   |-- task_summary.csv
|   |-- splits/
|   |   |-- train.csv
|   |   |-- validation.csv
|   |   |-- test.csv
|   |   `-- split_protocol.json
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
|   |-- SHA256SUMS_NPZ.txt
|   |-- npz_conversion.json
|   |-- npz_manifest.csv
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

## Fast NPZ raster access

Every JSON map has a same-stem companion under `npz/<family>/`. JSON remains
the canonical, self-describing record for identifiers, tasks, robot settings,
generation parameters, and metrics. NPZ is a derived array-only view intended
for faster repeated loading in numerical and learning pipelines.

Each NPZ contains 13 named `128 x 128` arrays. Continuous layers are stored as
`float32`; `grid` and the five binary masks are `uint8`. No object arrays or
pickled values are used:

```python
import numpy as np

with np.load(
    "npz/mountain/terrain_mountain_001_hard.npz",
    allow_pickle=False,
) as raster:
    elevation = raster["elevation"]
    collision = raster["grid"]
    support_z = raster["support_elevation"]
```

The JSON and NPZ files are linked by their identical stem. Field names, dtypes,
conversion validation, per-file paths, sizes, and hashes are recorded in
`SCHEMA.md`, `metadata/npz_conversion.json`, `metadata/npz_manifest.csv`, and
`metadata/SHA256SUMS_NPZ.txt`.

## Fixed splits

The optional learning split is published as three group-level indexes under
`metadata/splits/`; it does not duplicate or move map files. The assignment
unit is `matched_group_id`: all Easy, Medium, and Hard members of a base terrain,
together with their ten shared tasks, always remain in the same split. This
prevents paired variants of one terrain from leaking across training and
evaluation sets.

Each terrain family contributes 234 groups to train, 50 to validation, and 50
to test. The totals are therefore 1,170/250/250 groups, corresponding to
3,510/750/750 maps. Assignment is deterministic: within each family, groups are
ranked by SHA-256 using seed `20260830`, then allocated in those fixed counts.
The exact method, counts, validation flags, and file hashes are recorded in
`metadata/splits/split_protocol.json`. These splits are intended for learned
methods; the supplied six-planner reference benchmark was run over all maps and
was not used to train a model.

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

## Usage notes

- Task classes are defined on the Hard member and copied to its paired Easy and
  Medium members. `long` denotes a pair selected from the 0.75--0.98 band of
  ranked weighted distances; the longest selected task is canonical.
- The released collision and traversability layers use a circular 0.45-m
  reference footprint. To use another robot, recompute the footprint-aware
  layers from raw `elevation` with that robot's geometry and limits. Do not
  dilate the released `grid` again unless an additional safety margin is
  intentionally required.
- Treat a matched triplet, not its three maps, as the independent unit in
  paired statistical analyses. Likewise, repeated stochastic-planner seeds are
  nested trials, not independent terrain samples.
- The representation is a single-valued 2.5D height field. It does not encode
  overhangs, caves, bridges, dynamics, deformable ground, or sensor noise, and
  the reference robot thresholds are not universal vehicle limits.

## Validation status

The final audit reports:

- 5,010/5,010 maps structurally valid
- 50,100/50,100 map-task records valid
- zero invalid successful planner paths
- zero missing terrain, analysis, or route images
- zero Easy/Medium/Hard ordering violations
- exact A*/Dijkstra weighted-cost agreement on all 5,010 canonical tasks
- 5,010/5,010 NPZ companions pass CRC, field-set, shape, dtype-header, and
  payload-length checks

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
