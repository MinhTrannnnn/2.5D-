# Matched 2.5D Terrain Dataset and Planning Benchmark

This project generates deterministic 2.5D height-field terrains for
ground-robot path-planning experiments. Every `(x, y)` cell has one elevation
`z`; non-traversable regions remain visible terrain rather than holes.

## Dataset design

The generator provides five terrain families:

- `smooth_obstacles`
- `rolling`
- `mountain`
- `rugged`
- `plateau`

Each independently seeded base terrain produces a matched Easy, Medium, and
Hard triplet. Feature position, orientation, and scale change with the seed.
Members of a triplet share their large-scale terrain realization and ten
Start-Goal tasks, while relief and local detail change with difficulty.
The ten tasks are stratified into three short, four medium, and three long
routes. One task is marked as the canonical visualization task.

Easy, Medium, and Hard maps in one triplet are deliberately paired samples,
not three statistically independent terrain realizations.

## Terrain and robot model

For each elevation field, the pipeline derives:

- point slope and least-squares support-plane slope over the footprint;
- footprint-scale plane-fit roughness;
- local step height;
- weighted traversability cost;
- one footprint-conditioned binary collision grid.

Default maps are 128 by 128 cells at 0.25 m/cell. The robot defaults are a
0.45 m footprint radius, 28 degree maximum support-surface grade, 0.65 m
maximum local step, and 0.10 m maximum roughness. Surface detail is localized
and calibrated per family; smooth and plateau families do not reuse the
high-frequency detail model of rugged terrain.

Start-Goal connectivity, all six planners, and final path validation share the
same movement protocol:

- eight-connected cell-centre movement;
- no diagonal corner cutting;
- the same footprint support elevation for edge grade and path cost;
- the same footprint-conditioned collision grid;
- symmetric Bresenham discretization for PRM/RRT local connections.

The common reported path objective is

```text
sum(sqrt(planar_distance^2 + dz^2)
    * (1 + slope_weight * (abs(dz) / planar_distance)^2))
```

BFS is retained as a minimum-hop baseline. Dijkstra and A* optimize the common
weighted objective. PRM and RRT-Connect are fixed-budget stochastic baselines.
RRT* continues sampling and rewiring after its first solution and returns the
lowest-cost goal connection found after the complete fixed budget.

## Reproducible environment

The exact validated software environment is pinned in `environment.yml` and
`requirements-lock.txt`. Create it with Conda using

```bash
conda env create -f environment.yml
conda activate paired-25d
python -m unittest discover -s tests -v
```

Alternatively, install `requirements-lock.txt` into a Python 3.9.6 virtual
environment. See `REPRODUCIBILITY.md` for the full setup, validation, analysis,
and release-check commands. Use the shorter `requirements.txt` only for an
unpinned development installation.

## Preview workflow

Run these commands from this source directory. The scripts resolve their
default dataset path relative to the project, not the shell working directory.

```bash
python generate_25d_dataset.py --workers 6 --render-all-images --clean
python validate_dataset.py
python render_25d_path_previews.py
python benchmark_pathplanning.py --stochastic-trials 5 --workers 6
```

The default generator creates 30 preview maps: two matched triplets per family
times three difficulties. Analysis images include elevation, aggregate
traversability, and a categorical slope/roughness/step diagnostic. Images are
rendered only for the first triplet unless `--render-all-images` is supplied.

The benchmark has no hidden retries. It writes raw trial records, aggregated
mean/standard-deviation tables, and the exact protocol to `metadata/`.

Both generation and benchmarking checkpoint completed work. If a run is
interrupted, repeat the identical command with `--resume`. A resumed command
may use a different worker count, but every scientific parameter, seed, map
selection, planner budget, and trial count must remain unchanged.

## Dataset structure

```text
dataset_preview/
|-- maps/
|   |-- mountain/                Easy/Medium/Hard JSON triplets
|   |-- plateau/
|   |-- rolling/
|   |-- rugged/
|   `-- smooth_obstacles/
|-- images/
|   |-- terrain/<family>/         optional 3D terrain views
|   |-- analysis/<family>/        optional traversability analysis
|   |-- paths/<family>/           optional representative planner routes
|   `-- contact_sheets/           compact human-review sheets
|-- metadata/
|   |-- dataset_summary.csv
|   |-- task_summary.csv
|   |-- manifest.json
|   `-- validation_report.json
`-- README.md
```

Tasks are embedded in each map JSON. `metadata/task_summary.csv` is a compact
index of unique matched tasks rather than a second copy of their definitions.
Partitioning by the five terrain families keeps matched difficulty triplets
together and avoids a single directory containing all 5,000-10,000 map files.
For 5,010 maps, each family directory contains 1,002 JSON files.

## Scaling rule

With all five families and three difficulties, the number of map files is

```text
15 * instances_per_family
```

For example:

- `--instances-per-family 334` creates 5,010 maps (1,670 matched triplets);
- `--instances-per-family 666` creates 9,990 maps (3,330 matched triplets).

Create the 5,010-map final dataset with six worker processes:

```bash
python generate_25d_dataset.py \
  --output /absolute/path/to/dataset_5010_v1 \
  --instances-per-family 334 \
  --dataset-mode final \
  --workers 6
```

Resume the same run after an interruption:

```bash
python generate_25d_dataset.py \
  --output /absolute/path/to/dataset_5010_v1 \
  --instances-per-family 334 \
  --dataset-mode final \
  --workers 6 \
  --resume
```

Generation checkpoints one complete Easy/Medium/Hard matched group at a time.
Each record includes SHA-256 hashes for its three JSON maps. Missing or changed
files cause only that group to be regenerated. Final CSVs and the manifest are
rebuilt deterministically from checkpoint records.

The benchmark checkpoints one map after all requested task, algorithm, and
trial combinations for that map finish. For a dataset-paper baseline analogous
to the 2D benchmark, evaluate the single canonical long-route task on every map
and render its six planner paths:

```bash
python benchmark_pathplanning.py \
  --dataset /absolute/path/to/dataset_5010_v1 \
  --stochastic-trials 5 \
  --task-scope canonical \
  --render-all-canonical-paths \
  --workers 6 \
  --output-prefix pathfinding_benchmark_v1

python benchmark_pathplanning.py \
  --dataset /absolute/path/to/dataset_5010_v1 \
  --stochastic-trials 5 \
  --task-scope canonical \
  --render-all-canonical-paths \
  --workers 6 \
  --output-prefix pathfinding_benchmark_v1 \
  --resume
```

This produces 5,010 paired benchmark tasks, 90,180 trial rows, and up to 30,060
path PNGs. Images are written as each map completes and are safe to resume.
Omit `--task-scope canonical` only for the much larger all-ten-task experiment,
which produces 901,800 trial rows. Rendering every task would create 300,600
path images and is deliberately unsupported by the standard layout.

Generation renders only the first matched group of each family by default. To
materialize both terrain and analysis PNGs for every already-generated map,
without regenerating any JSON, run:

```bash
python render_25d_dataset_images.py \
  --dataset /absolute/path/to/dataset_5010_v1 \
  --workers 6 \
  --output-prefix map_images_full_v1

python render_25d_dataset_images.py \
  --dataset /absolute/path/to/dataset_5010_v1 \
  --workers 6 \
  --output-prefix map_images_full_v1 \
  --resume
```

The renderer skips durable existing PNGs, checkpoints completed maps, and
updates `dataset_summary.csv` and `manifest.json` after all 10,020 PNGs exist.
The map JSON, elevation, traversability, and ten Start-Goal tasks remain the
canonical data regardless of whether the optional PNGs have been rendered.

After the benchmark and image renderer are complete, audit the release and
build map-level benchmark tables and figures with:

```bash
python analyze_25d_benchmark_results.py \
  --dataset /absolute/path/to/dataset_5010_v1 \
  --benchmark-prefix pathfinding_benchmark_v1
```

The analysis verifies pairing, expected trial counts and seeds, path validity,
image completeness, and A*/Dijkstra cost agreement. It first aggregates the
five stochastic trials within each map, then reports mean and sample standard
deviation across maps. Outputs are written to
`metadata/benchmark_results/{tables,figures}` with a machine-readable audit and a
concise results report.

After validation and analysis both pass, prepare the release inventory and
checksums without creating a duplicate archive:

```bash
python prepare_25d_release.py \
  --dataset /absolute/path/to/dataset_5010_v1 \
  --benchmark-prefix pathfinding_benchmark_v1
```

This writes release notes, a complete file inventory, map/core SHA-256 lists,
and the machine-readable `metadata/release_manifest.json`. Create an archive
only after selecting the target repository and its preferred archive or chunk
format.

Parallel execution preserves paths, seeds, costs, and success results, but
simultaneous CPU load affects measured runtime. Use `--workers 1` for a paper
table intended to report isolated single-process planner latency; use multiple
workers when throughput and completion time are the priority.

## Representation limit

This is 2.5D rather than a voxel representation. It cannot represent overhangs,
caves, bridges, or multiple surfaces at one `(x, y)` coordinate.
