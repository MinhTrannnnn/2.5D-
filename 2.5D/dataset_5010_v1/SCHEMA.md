# 2.5D map schema

This document describes schema `3.1-preview`, used by every JSON file below
`maps/<family>/`. Despite the historical schema label, `dataset_5010_v1` is the
validated final dataset. Each file is one deterministic height-field map and
contains all geometric, traversability, robot, and task data needed by a
planner.

## Indexing and coordinate convention

Arrays are `size x size` nested lists indexed as `[row][column]`, or `[y][x]`.
Grid cells and task endpoints are stored as `[x, y]` pairs. Convert a cell to
metric map coordinates with

```text
X = x * navigation.cell_size
Y = y * navigation.cell_size
Z = elevation[y][x]
```

The origin is `[0, 0, 0]`, units are metres, and the default map is `128 x 128`
cells at `0.25 m/cell`. Do not index an array as `array[x][y]`.

## Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Map schema identifier. |
| `instance_id` | string | Unique map identifier, including family, triplet index, and difficulty. |
| `matched_group_id` | string | Identifier shared by the Easy/Medium/Hard variants. |
| `instance_index` | integer | One-based triplet index within a terrain family. |
| `terrain_family` | string | One of `smooth_obstacles`, `rolling`, `mountain`, `rugged`, or `plateau`. |
| `difficulty` | string | `easy`, `medium`, or `hard`. |
| `base_seed` | integer | Seed shared by the matched triplet. |
| `realized_seed` | integer | Deterministic seed for this realized terrain variant. |
| `task_seed` | integer | Seed used to select the ten matched tasks. |
| `size` | integer | Width and height of every raster layer. |
| `representation` | string | Declares a single-valued 2.5D height field. |
| `grid` | integer matrix | Footprint-conditioned collision grid: `0` traversable, `1` blocked. |
| `elevation` | float matrix | Raw terrain-surface elevation in metres. |
| `terrain_analysis` | object | Derived geometry and traversability layers described below. |
| `tasks` | array | Ten matched Start-Goal tasks. |
| `canonical_task_id` | string | Long-distance task used by the released planner benchmark. |
| `start`, `goal` | `[x,y]` | Convenience copies of the canonical task endpoints. |
| `start_z`, `goal_z` | float | Raw surface elevation at the canonical endpoints. |
| `coordinate_frame` | object | Machine-readable axes, origin, and units. |
| `navigation` | object | Robot constraints and planner movement semantics. |
| `generation` | object | Generator configuration and realized terrain parameters. |
| `difficulty_definition` | object | Acceptance bounds used for this family/difficulty. |
| `metrics` | object | Map-level summary statistics. |

`seed` and `terrain_profile` are compatibility aliases for `base_seed` and
`terrain_family` respectively.

## Raster layers

Every layer below has the same shape as `elevation`.

| Field | Units/encoding | Meaning |
|---|---|---|
| `grid` | `0` free, `1` blocked | Final robot-centre collision grid, including the footprint-safe map border. |
| `elevation` | m | Raw surface height `z(y,x)`. |
| `point_slope_degrees` | degrees | Slope from the gradient of a lightly smoothed elevation field. |
| `support_elevation` | m | Intercept of the least-squares support plane fitted over the circular robot footprint; planners use this layer for edge grade and cost. |
| `footprint_slope_degrees` | degrees | Grade of the fitted support plane. |
| `roughness` | m RMS | RMS elevation residual about the fitted support plane. |
| `step_height` | m | Maximum local adjacent height discontinuity observed over the footprint. |
| `traversability_cost` | dimensionless | Weighted sum of normalized footprint slope, roughness, and step height. |
| `blocked_by_slope` | `0/1` | Footprint slope exceeds `navigation.max_slope_degrees`. |
| `blocked_by_roughness` | `0/1` | Roughness exceeds `navigation.max_roughness`. |
| `blocked_by_step` | `0/1` | Step height exceeds `navigation.max_step_height`. |
| `blocked_by_combined_cost` | `0/1` | Aggregate traversability cost is at least one. |
| `centre_blocked` | `0/1` | Union of the four blocking causes before the footprint-safe border is forced blocked. |

The collision layers already describe valid robot-centre poses for the stated
footprint. Applying another obstacle dilation would double-count the footprint
and is incorrect.

## Navigation object

The released protocol uses eight-connected cell-centre motion without diagonal
corner cutting. `footprint_radius`, `endpoint_margin`, `max_step_height`, and
`max_roughness` are in metres. `max_slope_degrees` is the readable grade limit;
`max_slope` is its rise/run form. `path_slope_weight` is the penalty coefficient
in the common path objective. `traversability_weights` contains the slope,
roughness, and step weights used to construct `traversability_cost`; they sum to
one.

For an edge with planar length `d`, support-elevation change `dz`, and
`w = navigation.path_slope_weight`, the benchmark cost is

```text
sqrt(d^2 + dz^2) * (1 + w * (abs(dz) / d)^2)
```

## Task object

Each map contains ten task objects. Easy, Medium, and Hard members of a matched
group use identical `[x,y]` endpoints.

| Field | Meaning |
|---|---|
| `task_index` | One-based index within the map. |
| `task_id` | Difficulty-specific unique identifier. |
| `matched_task_id` | Identifier shared across the matched difficulty triplet. |
| `start`, `goal` | Endpoint cells in `[x,y]` order. |
| `distance_class` | `short`, `medium`, or `long`; the distribution is 3/4/3. |
| `reference_weighted_distance` | Weighted Dijkstra distance used to stratify the matched task. |
| `reference_distance_difficulty` | Difficulty variant on which the shared task distance was measured. |
| `canonical_visualization` | `true` for the long task used by the released benchmark. |

## Minimal reader

```python
import json
from pathlib import Path

map_path = Path("maps/mountain/terrain_mountain_001_easy.json")
data = json.loads(map_path.read_text(encoding="utf-8"))

cell_size = float(data["navigation"]["cell_size"])
grid = data["grid"]
surface_z = data["elevation"]
support_z = data["terrain_analysis"]["support_elevation"]
canonical = next(t for t in data["tasks"] if t["canonical_visualization"])

x, y = canonical["start"]
assert grid[y][x] == 0
xyz = (x * cell_size, y * cell_size, surface_z[y][x])
planner_support_height = support_z[y][x]
print(data["instance_id"], canonical["task_id"], xyz, planner_support_height)
```

A runnable version with structural checks is provided in `examples/load_map.py`.
Dataset-level indexes are in `metadata/dataset_summary.csv` and
`metadata/task_summary.csv`; the map JSON remains the canonical source.
