#!/usr/bin/env python3
"""Audit a completed 2.5D benchmark and build paper-ready tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ("BFS", "Dijkstra", "A_Star", "PRM", "RRT_Connect", "RRT_Star")
STOCHASTIC = frozenset(("PRM", "RRT_Connect", "RRT_Star"))
FAMILIES = ("smooth_obstacles", "rolling", "mountain", "rugged", "plateau")
DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_LABELS = ("Easy", "Medium", "Hard")
ALGORITHM_LABELS = {
    "BFS": "BFS",
    "Dijkstra": "Dijkstra",
    "A_Star": "A*",
    "PRM": "PRM",
    "RRT_Connect": "RRT-Connect",
    "RRT_Star": "RRT*",
}
FAMILY_LABELS = {
    "smooth_obstacles": "Smooth obstacles",
    "rolling": "Rolling",
    "mountain": "Mountain",
    "rugged": "Rugged",
    "plateau": "Plateau",
}
COLORS = {
    "BFS": "#0072B2",
    "Dijkstra": "#009E73",
    "A_Star": "#E69F00",
    "PRM": "#CC79A7",
    "RRT_Connect": "#56B4E9",
    "RRT_Star": "#D55E00",
}
FAMILY_COLORS = {
    "smooth_obstacles": "#0072B2",
    "rolling": "#009E73",
    "mountain": "#D55E00",
    "rugged": "#CC79A7",
    "plateau": "#E69F00",
}
DATASET_METRICS = (
    "relief_severity",
    "max_elevation",
    "mean_point_slope_degrees",
    "mean_footprint_slope_degrees",
    "mean_roughness",
    "blocked_by_slope_fraction",
    "blocked_by_roughness_fraction",
    "blocked_by_step_fraction",
    "navigable_fraction",
    "collision_boundary_density",
    "connected_components",
    "largest_component_fraction",
    "largest_component_share",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _optional_float(value: str) -> Optional[float]:
    return None if value == "" else float(value)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    items = [float(value) for value in values if value is not None]
    if not items:
        return math.nan, math.nan
    return (
        statistics.fmean(items),
        statistics.stdev(items) if len(items) > 1 else 0.0,
    )


def _round(value: float, digits: int = 6) -> float | str:
    return "" if math.isnan(value) else round(value, digits)


def _mean_std_text(mean: float, std: float, digits: int = 3) -> str:
    if math.isnan(mean):
        return "N/A"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _family_key(value: str) -> int:
    return FAMILIES.index(value)


def _difficulty_key(value: str) -> int:
    return DIFFICULTIES.index(value)


def _algorithm_key(value: str) -> int:
    return ALGORITHMS.index(value)


def _typed_trial(row: dict[str, str]) -> dict:
    typed = dict(row)
    typed["trial"] = int(row["trial"])
    typed["success"] = _bool(row["success"])
    typed["path_valid"] = _bool(row["path_valid"]) if row["path_valid"] else None
    typed["canonical_task"] = _bool(row["canonical_task"])
    for key in (
        "runtime_sec",
        "path_length_2d_m",
        "path_length_3d_m",
        "weighted_path_cost",
        "elevation_gain_m",
    ):
        typed[key] = _optional_float(row[key])
    typed["path_cells"] = int(row["path_cells"])
    typed["explored_edges"] = int(row["explored_edges"])
    return typed


def _typed_map(row: dict[str, str]) -> dict:
    typed = dict(row)
    typed["instance_index"] = int(row["instance_index"])
    typed["task_count"] = int(row["task_count"])
    for key in DATASET_METRICS:
        typed[key] = float(row[key])
    return typed


def _audit(
    dataset_root: Path,
    maps: Sequence[dict],
    trials: Sequence[dict],
    protocol: dict,
) -> tuple[dict, dict[str, float]]:
    errors: list[str] = []
    warnings: list[str] = []
    map_by_id = {row["instance_id"]: row for row in maps}
    if len(map_by_id) != len(maps):
        errors.append("dataset_summary contains duplicate instance_id values")
    if len(maps) != 5010:
        errors.append(f"expected 5010 maps, found {len(maps)}")
    if len(trials) != 90180:
        errors.append(f"expected 90180 benchmark rows, found {len(trials)}")
    if protocol.get("task_scope") != "canonical":
        errors.append("benchmark protocol is not canonical-task scope")
    if protocol.get("stochastic_trials") != 5:
        errors.append("benchmark protocol does not contain five stochastic trials")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in maps:
        groups[row["matched_group_id"]].append(row)
        if row["task_count"] != 10:
            errors.append(f"{row['instance_id']} does not contain ten tasks")
        for key in ("map_file", "terrain_image_file", "traversability_image_file"):
            if not row.get(key) or not (dataset_root / row[key]).is_file():
                errors.append(f"{row['instance_id']}: missing {key}")

    relief_violations = 0
    navigability_violations = 0
    triplet_errors = 0
    for members in groups.values():
        if {row["difficulty"] for row in members} != set(DIFFICULTIES):
            triplet_errors += 1
            continue
        ordered = sorted(members, key=lambda row: _difficulty_key(row["difficulty"]))
        relief = [row["relief_severity"] for row in ordered]
        navigability = [row["navigable_fraction"] for row in ordered]
        relief_violations += int(not (relief[0] < relief[1] < relief[2]))
        navigability_violations += int(
            not (navigability[0] > navigability[1] > navigability[2])
        )
    if triplet_errors:
        errors.append(f"{triplet_errors} matched groups do not contain E/M/H")
    if relief_violations:
        errors.append(f"{relief_violations} triplets violate relief ordering")
    if navigability_violations:
        errors.append(
            f"{navigability_violations} triplets violate navigability ordering"
        )

    grouped_trials: dict[tuple[str, str], list[dict]] = defaultdict(list)
    invalid_success_paths = 0
    noncanonical_rows = 0
    nonlong_rows = 0
    for row in trials:
        if row["terrain_id"] not in map_by_id:
            errors.append(f"unknown terrain_id in trials: {row['terrain_id']}")
            continue
        grouped_trials[(row["terrain_id"], row["algorithm"])].append(row)
        invalid_success_paths += int(row["success"] and row["path_valid"] is not True)
        noncanonical_rows += int(not row["canonical_task"])
        nonlong_rows += int(row["distance_class"] != "long")
    if invalid_success_paths:
        errors.append(f"{invalid_success_paths} successful paths failed validation")
    if noncanonical_rows:
        errors.append(f"{noncanonical_rows} trial rows are not canonical")
    if nonlong_rows:
        errors.append(f"{nonlong_rows} trial rows are not long-distance tasks")

    count_errors = 0
    seed_errors = 0
    missing_path_images = 0
    for terrain_id, map_row in map_by_id.items():
        for algorithm in ALGORITHMS:
            rows = grouped_trials.get((terrain_id, algorithm), [])
            expected = 5 if algorithm in STOCHASTIC else 1
            count_errors += int(len(rows) != expected)
            if algorithm in STOCHASTIC and rows:
                seeds = [row["seed"] for row in rows]
                seed_errors += int(len(set(seeds)) != expected or "" in seeds)
            image_path = (
                dataset_root
                / "images"
                / "paths"
                / map_row["family"]
                / f"{terrain_id}_{algorithm}.png"
            )
            missing_path_images += int(
                not image_path.is_file() or image_path.stat().st_size == 0
            )
    if count_errors:
        errors.append(f"{count_errors} map-algorithm groups have wrong trial counts")
    if seed_errors:
        errors.append(f"{seed_errors} stochastic groups have invalid seeds")
    if missing_path_images:
        errors.append(f"{missing_path_images} planner path images are missing")

    optimal_by_terrain: dict[str, dict[str, float]] = defaultdict(dict)
    for row in trials:
        if (
            row["algorithm"] in {"A_Star", "Dijkstra"}
            and row["success"]
            and row["weighted_path_cost"] is not None
        ):
            optimal_by_terrain[row["terrain_id"]][row["algorithm"]] = row[
                "weighted_path_cost"
            ]
    optimal_mismatches = 0
    maximum_optimal_cost_delta = 0.0
    for terrain_id in map_by_id:
        values = optimal_by_terrain.get(terrain_id, {})
        if set(values) != {"A_Star", "Dijkstra"}:
            optimal_mismatches += 1
            continue
        delta = abs(values["A_Star"] - values["Dijkstra"])
        maximum_optimal_cost_delta = max(maximum_optimal_cost_delta, delta)
        optimal_mismatches += int(delta > 1e-6)
    if optimal_mismatches:
        errors.append(
            f"{optimal_mismatches} maps have inconsistent A*/Dijkstra optimal cost"
        )

    failures = Counter()
    for row in trials:
        if not row["success"]:
            failures[row["algorithm"]] += 1

    if protocol.get("parallel_execution", {}).get("workers_used") != [1]:
        warnings.append(
            "runtime was measured during a multi-worker throughput run; report the "
            "execution protocol and avoid presenting it as isolated single-core latency"
        )

    report = {
        "schema_version": "1.0",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "maps": len(maps),
            "matched_triplets": len(groups),
            "trial_rows": len(trials),
            "map_algorithm_groups": len(grouped_trials),
            "terrain_images": sum(
                1 for _ in dataset_root.joinpath("images", "terrain").rglob("*.png")
            ),
            "analysis_images": sum(
                1 for _ in dataset_root.joinpath("images", "analysis").rglob("*.png")
            ),
            "path_images": sum(
                1 for _ in dataset_root.joinpath("images", "paths").rglob("*.png")
            ),
        },
        "checks": {
            "invalid_success_paths": invalid_success_paths,
            "noncanonical_trial_rows": noncanonical_rows,
            "nonlong_trial_rows": nonlong_rows,
            "wrong_trial_count_groups": count_errors,
            "invalid_stochastic_seed_groups": seed_errors,
            "missing_path_images": missing_path_images,
            "relief_ordering_violations": relief_violations,
            "navigability_ordering_violations": navigability_violations,
            "astar_dijkstra_cost_mismatches": optimal_mismatches,
            "max_astar_dijkstra_cost_delta": maximum_optimal_cost_delta,
        },
        "failed_trials_by_algorithm": dict(sorted(failures.items())),
    }
    astar_cost = {
        terrain_id: values["A_Star"]
        for terrain_id, values in optimal_by_terrain.items()
        if "A_Star" in values
    }
    return report, astar_cost


def _build_map_level(
    maps: Sequence[dict],
    trials: Sequence[dict],
    astar_cost: dict[str, float],
) -> list[dict]:
    map_by_id = {row["instance_id"]: row for row in maps}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in trials:
        grouped[(row["terrain_id"], row["algorithm"])].append(row)
    output = []
    for (terrain_id, algorithm), rows in grouped.items():
        map_row = map_by_id[terrain_id]
        rows.sort(key=lambda row: row["trial"])
        successful = [row for row in rows if row["success"]]
        runtime_mean, runtime_std = _mean_std(row["runtime_sec"] for row in rows)
        path_mean, path_std = _mean_std(
            row["path_length_3d_m"] for row in successful
        )
        cost_mean, cost_std = _mean_std(
            row["weighted_path_cost"] for row in successful
        )
        gain_mean, gain_std = _mean_std(row["elevation_gain_m"] for row in successful)
        ratios = [
            row["weighted_path_cost"] / astar_cost[terrain_id]
            for row in successful
            if terrain_id in astar_cost and astar_cost[terrain_id] > 0
        ]
        ratio_mean, ratio_std = _mean_std(ratios)
        output.append(
            {
                "terrain_id": terrain_id,
                "matched_group_id": map_row["matched_group_id"],
                "instance_index": map_row["instance_index"],
                "family": map_row["family"],
                "difficulty": map_row["difficulty"],
                "algorithm": algorithm,
                "trial_count": len(rows),
                "successes": len(successful),
                "trial_success_rate": round(len(successful) / len(rows), 6),
                "any_trial_success": bool(successful),
                "all_trials_success": len(successful) == len(rows),
                "runtime_mean_sec": _round(runtime_mean),
                "runtime_within_map_std_sec": _round(runtime_std),
                "path_length_3d_mean_m": _round(path_mean),
                "path_length_3d_within_map_std_m": _round(path_std),
                "weighted_cost_mean": _round(cost_mean),
                "weighted_cost_within_map_std": _round(cost_std),
                "cost_ratio_to_astar_mean": _round(ratio_mean),
                "cost_ratio_to_astar_within_map_std": _round(ratio_std),
                "elevation_gain_mean_m": _round(gain_mean),
                "elevation_gain_within_map_std_m": _round(gain_std),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            _family_key(row["family"]),
            row["instance_index"],
            _difficulty_key(row["difficulty"]),
            _algorithm_key(row["algorithm"]),
        ),
    )


def _numeric_values(rows: Sequence[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key, "") != ""]


def _planner_summary(map_level: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in map_level:
        grouped[(row["algorithm"], row["family"], row["difficulty"])].append(row)
    output = []
    for algorithm in ALGORITHMS:
        for family in FAMILIES:
            for difficulty in DIFFICULTIES:
                rows = grouped[(algorithm, family, difficulty)]
                success_mean, success_std = _mean_std(
                    float(row["trial_success_rate"]) for row in rows
                )
                runtime_mean, runtime_std = _mean_std(
                    _numeric_values(rows, "runtime_mean_sec")
                )
                path_mean, path_std = _mean_std(
                    _numeric_values(rows, "path_length_3d_mean_m")
                )
                cost_mean, cost_std = _mean_std(
                    _numeric_values(rows, "weighted_cost_mean")
                )
                ratio_mean, ratio_std = _mean_std(
                    _numeric_values(rows, "cost_ratio_to_astar_mean")
                )
                gain_mean, gain_std = _mean_std(
                    _numeric_values(rows, "elevation_gain_mean_m")
                )
                output.append(
                    {
                        "algorithm": algorithm,
                        "family": family,
                        "difficulty": difficulty,
                        "map_tasks": len(rows),
                        "trials": sum(int(row["trial_count"]) for row in rows),
                        "successful_trials": sum(int(row["successes"]) for row in rows),
                        "trial_success_rate_percent": _round(success_mean * 100),
                        "trial_success_rate_map_std_percent": _round(success_std * 100),
                        "maps_with_any_success_percent": _round(
                            100
                            * statistics.fmean(
                                float(row["any_trial_success"]) for row in rows
                            )
                        ),
                        "maps_with_all_trials_success_percent": _round(
                            100
                            * statistics.fmean(
                                float(row["all_trials_success"]) for row in rows
                            )
                        ),
                        "runtime_map_mean_sec": _round(runtime_mean),
                        "runtime_between_map_std_sec": _round(runtime_std),
                        "path_length_3d_map_mean_m": _round(path_mean),
                        "path_length_3d_between_map_std_m": _round(path_std),
                        "weighted_cost_map_mean": _round(cost_mean),
                        "weighted_cost_between_map_std": _round(cost_std),
                        "cost_ratio_to_astar_map_mean": _round(ratio_mean),
                        "cost_ratio_to_astar_between_map_std": _round(ratio_std),
                        "elevation_gain_map_mean_m": _round(gain_mean),
                        "elevation_gain_between_map_std_m": _round(gain_std),
                    }
                )
    return output


def _overall_summary(map_level: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in map_level:
        grouped[(row["algorithm"], row["difficulty"])].append(row)
        grouped[(row["algorithm"], "all")].append(row)
    output = []
    for algorithm in ALGORITHMS:
        for difficulty in (*DIFFICULTIES, "all"):
            rows = grouped[(algorithm, difficulty)]
            success_mean, success_std = _mean_std(
                float(row["trial_success_rate"]) for row in rows
            )
            runtime_mean, runtime_std = _mean_std(
                _numeric_values(rows, "runtime_mean_sec")
            )
            cost_mean, cost_std = _mean_std(
                _numeric_values(rows, "weighted_cost_mean")
            )
            ratio_mean, ratio_std = _mean_std(
                _numeric_values(rows, "cost_ratio_to_astar_mean")
            )
            path_mean, path_std = _mean_std(
                _numeric_values(rows, "path_length_3d_mean_m")
            )
            output.append(
                {
                    "algorithm": algorithm,
                    "difficulty": difficulty,
                    "map_tasks": len(rows),
                    "trials": sum(int(row["trial_count"]) for row in rows),
                    "trial_success_rate_percent": _round(success_mean * 100),
                    "trial_success_rate_map_std_percent": _round(success_std * 100),
                    "runtime_map_mean_sec": _round(runtime_mean),
                    "runtime_between_map_std_sec": _round(runtime_std),
                    "path_length_3d_map_mean_m": _round(path_mean),
                    "path_length_3d_between_map_std_m": _round(path_std),
                    "weighted_cost_map_mean": _round(cost_mean),
                    "weighted_cost_between_map_std": _round(cost_std),
                    "cost_ratio_to_astar_map_mean": _round(ratio_mean),
                    "cost_ratio_to_astar_between_map_std": _round(ratio_std),
                }
            )
    return output


def _dataset_summary(maps: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in maps:
        grouped[(row["family"], row["difficulty"])].append(row)
    output = []
    for family in FAMILIES:
        for difficulty in DIFFICULTIES:
            rows = grouped[(family, difficulty)]
            result = {
                "family": family,
                "difficulty": difficulty,
                "maps": len(rows),
            }
            for metric in DATASET_METRICS:
                mean_value, std_value = _mean_std(row[metric] for row in rows)
                result[f"{metric}_mean"] = _round(mean_value)
                result[f"{metric}_std"] = _round(std_value)
            output.append(result)
    return output


def _paired_effect_rows(
    rows: Sequence[dict],
    *,
    metrics: Sequence[str],
    identity_key: str,
    family_key: str = "family",
    difficulty_key: str = "difficulty",
    extra: Optional[dict[str, str]] = None,
) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[(row[family_key], row[identity_key])][row[difficulty_key]] = row
    output = []
    comparisons = (("medium", "easy"), ("hard", "medium"), ("hard", "easy"))
    for family in FAMILIES:
        family_groups = [
            members
            for (row_family, _), members in grouped.items()
            if row_family == family and set(members) == set(DIFFICULTIES)
        ]
        for metric in metrics:
            for high, low in comparisons:
                deltas = []
                for members in family_groups:
                    high_value = members[high].get(metric, "")
                    low_value = members[low].get(metric, "")
                    if high_value == "" or low_value == "":
                        continue
                    deltas.append(float(high_value) - float(low_value))
                mean_value, std_value = _mean_std(deltas)
                result = dict(extra or {})
                result.update(
                    {
                        "family": family,
                        "metric": metric,
                        "comparison": f"{high}_minus_{low}",
                        "paired_groups": len(deltas),
                        "mean_delta": _round(mean_value),
                        "std_delta": _round(std_value),
                        "median_delta": _round(
                            float(np.median(deltas)) if deltas else math.nan
                        ),
                        "positive_delta_fraction": _round(
                            (
                                sum(delta > 0 for delta in deltas) / len(deltas)
                                if deltas
                                else math.nan
                            )
                        ),
                    }
                )
                output.append(result)
    return output


def _planner_paired_effects(map_level: Sequence[dict]) -> list[dict]:
    output = []
    for algorithm in ALGORITHMS:
        rows = [row for row in map_level if row["algorithm"] == algorithm]
        output.extend(
            _paired_effect_rows(
                rows,
                metrics=(
                    "trial_success_rate",
                    "runtime_mean_sec",
                    "weighted_cost_mean",
                    "cost_ratio_to_astar_mean",
                ),
                identity_key="matched_group_id",
                extra={"algorithm": algorithm},
            )
        )
    return output


def _pivot_table(
    summary: Sequence[dict],
    key: str,
    *,
    digits: int,
    include_std_key: Optional[str] = None,
) -> list[dict]:
    lookup = {
        (row["family"], row["difficulty"], row["algorithm"]): row
        for row in summary
    }
    output = []
    for family in FAMILIES:
        for difficulty in DIFFICULTIES:
            result = {"family": family, "difficulty": difficulty}
            for algorithm in ALGORITHMS:
                row = lookup[(family, difficulty, algorithm)]
                value = float(row[key]) if row[key] != "" else math.nan
                if include_std_key:
                    std = (
                        float(row[include_std_key])
                        if row[include_std_key] != ""
                        else math.nan
                    )
                    result[algorithm] = _mean_std_text(value, std, digits)
                else:
                    result[algorithm] = (
                        "N/A" if math.isnan(value) else f"{value:.{digits}f}"
                    )
            output.append(result)
    return output


def _figure_dataset_difficulty(summary: Sequence[dict], output_path: Path) -> None:
    lookup = {(row["family"], row["difficulty"]): row for row in summary}
    panels = (
        ("navigable_fraction", "Navigable area (%)", 100.0),
        ("relief_severity", "Relief severity", 1.0),
        ("mean_footprint_slope_degrees", "Mean footprint slope (deg)", 1.0),
        ("connected_components", "Connected components", 1.0),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150, squeeze=False)
    x = np.arange(len(DIFFICULTIES))
    for axis, (metric, label, scale) in zip(axes.ravel(), panels):
        for family in FAMILIES:
            means = [
                float(lookup[(family, difficulty)][f"{metric}_mean"]) * scale
                for difficulty in DIFFICULTIES
            ]
            stds = [
                float(lookup[(family, difficulty)][f"{metric}_std"]) * scale
                for difficulty in DIFFICULTIES
            ]
            axis.errorbar(
                x,
                means,
                yerr=stds,
                marker="o",
                linewidth=2.0,
                capsize=3,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
        axis.set_xticks(x, DIFFICULTY_LABELS)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        frameon=False,
    )
    figure.suptitle(
        "Matched 2.5D terrain difficulty calibration", fontsize=17, y=0.995
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _figure_planner_overview(overall: Sequence[dict], output_path: Path) -> None:
    lookup = {
        (row["algorithm"], row["difficulty"]): row
        for row in overall
        if row["difficulty"] in DIFFICULTIES
    }
    panels = (
        (
            "trial_success_rate_percent",
            "trial_success_rate_map_std_percent",
            "Success rate (%)",
            False,
        ),
        (
            "cost_ratio_to_astar_map_mean",
            "cost_ratio_to_astar_between_map_std",
            "Weighted-cost ratio to A*",
            False,
        ),
        (
            "runtime_map_mean_sec",
            "runtime_between_map_std_sec",
            "Runtime (s, log scale)",
            True,
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=150)
    x = np.arange(len(DIFFICULTIES))
    for axis, (metric, std_metric, label, log_scale) in zip(axes, panels):
        for algorithm in ALGORITHMS:
            means = [
                float(lookup[(algorithm, difficulty)][metric])
                for difficulty in DIFFICULTIES
            ]
            confidence_intervals = [
                1.96
                * float(lookup[(algorithm, difficulty)][std_metric])
                / math.sqrt(float(lookup[(algorithm, difficulty)]["map_tasks"]))
                for difficulty in DIFFICULTIES
            ]
            lower = [
                min(interval, mean * 0.95) if log_scale else interval
                for mean, interval in zip(means, confidence_intervals)
            ]
            axis.errorbar(
                x,
                means,
                yerr=(
                    np.asarray((lower, confidence_intervals))
                    if log_scale
                    else confidence_intervals
                ),
                marker="o",
                linewidth=1.8,
                capsize=2.5,
                color=COLORS[algorithm],
                label=ALGORITHM_LABELS[algorithm],
            )
        axis.set_xticks(x, DIFFICULTY_LABELS)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        if log_scale:
            axis.set_yscale("log")
        if metric == "trial_success_rate_percent":
            axis.set_ylim(0.0, 102.5)
        if metric == "cost_ratio_to_astar_map_mean":
            axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=6,
        frameon=False,
    )
    figure.suptitle(
        "Canonical long-route planner benchmark (map-level mean, 95% CI)",
        fontsize=17,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.855))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _heatmap(
    summary: Sequence[dict],
    *,
    metric: str,
    title: str,
    colorbar_label: str,
    output_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    lookup = {
        (row["family"], row["difficulty"], row["algorithm"]): row
        for row in summary
    }
    row_keys = [(family, difficulty) for family in FAMILIES for difficulty in DIFFICULTIES]
    values = np.asarray(
        [
            [float(lookup[(family, difficulty, algorithm)][metric]) for algorithm in ALGORITHMS]
            for family, difficulty in row_keys
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(12, 9), dpi=150)
    image = axis.imshow(values, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(ALGORITHMS)), [ALGORITHM_LABELS[a] for a in ALGORITHMS])
    axis.set_yticks(
        np.arange(len(row_keys)),
        [f"{FAMILY_LABELS[f]} — {d.title()}" for f, d in row_keys],
    )
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            normalized = (value - np.nanmin(values)) / max(
                np.nanmax(values) - np.nanmin(values), 1e-12
            )
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if normalized > 0.55 else "white",
            )
    axis.set_title(title, fontsize=16, pad=14)
    figure.colorbar(image, ax=axis, shrink=0.85, label=colorbar_label)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _results_markdown(
    audit: dict,
    overall: Sequence[dict],
    protocol: dict,
) -> str:
    overall_lookup = {
        row["algorithm"]: row for row in overall if row["difficulty"] == "all"
    }
    table_lines = [
        "| Algorithm | Success (%) | Cost ratio to A* | Runtime (s) |",
        "|---|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        row = overall_lookup[algorithm]
        table_lines.append(
            "| {algorithm} | {success} | {ratio} | {runtime} |".format(
                algorithm=ALGORITHM_LABELS[algorithm],
                success=_mean_std_text(
                    float(row["trial_success_rate_percent"]),
                    float(row["trial_success_rate_map_std_percent"]),
                    2,
                ),
                ratio=_mean_std_text(
                    float(row["cost_ratio_to_astar_map_mean"]),
                    float(row["cost_ratio_to_astar_between_map_std"]),
                    3,
                ),
                runtime=_mean_std_text(
                    float(row["runtime_map_mean_sec"]),
                    float(row["runtime_between_map_std_sec"]),
                    3,
                ),
            )
        )
    warning_lines = "\n".join(f"- {warning}" for warning in audit["warnings"])
    return f"""# Paper-ready 2.5D dataset results

## Scope

- 5,010 maps in 1,670 matched Easy/Medium/Hard triplets
- 10 published Start-Goal tasks per triplet; the canonical long task is used here
- 6 planners and {protocol['stochastic_trials']} trials for each stochastic planner
- 90,180 raw trial rows, aggregated to 30,060 map-algorithm experimental units
- Mean and sample standard deviation are computed across maps after within-map trial aggregation

## Integrity audit

- Audit valid: **{audit['valid']}**
- Successful paths failing validation: {audit['checks']['invalid_success_paths']}
- Missing path images: {audit['checks']['missing_path_images']}
- Relief ordering violations: {audit['checks']['relief_ordering_violations']}
- Navigability ordering violations: {audit['checks']['navigability_ordering_violations']}
- A*/Dijkstra optimal-cost mismatches: {audit['checks']['astar_dijkstra_cost_mismatches']}

## Overall planner results

The success statistic is the mean per-map trial success rate. Cost is normalized
by the A* optimum on the same map and Start-Goal task. Path metrics are computed
over successful solutions only.

{chr(10).join(table_lines)}

## Interpretation notes

- A* and Dijkstra provide a cross-check of the common weighted objective.
- BFS is a minimum-hop baseline and is not expected to minimize weighted 2.5D cost.
- PRM, RRT-Connect, and RRT* are fixed-budget stochastic baselines without hidden retries.
- Easy, Medium, and Hard are paired terrain variants, not independent samples.
- The canonical benchmark evaluates the longest published task per map; the other
  nine tasks remain part of the released dataset but are not included in this table.

## Runtime caveat

{warning_lines or '- None'}

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
"""


def analyze(
    dataset_root: Path,
    benchmark_prefix: str,
    output_root: Path,
) -> dict:
    metadata_root = dataset_root / "metadata"
    trial_path = metadata_root / f"{benchmark_prefix}_trials.csv"
    protocol_path = metadata_root / f"{benchmark_prefix}_protocol.json"
    map_summary_path = metadata_root / "dataset_summary.csv"
    for path in (trial_path, protocol_path, map_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Loading dataset and benchmark tables...", flush=True)
    maps = [_typed_map(row) for row in _read_csv(map_summary_path)]
    trials = [_typed_trial(row) for row in _read_csv(trial_path)]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    print("Auditing pairing, trials, costs, paths, and images...", flush=True)
    audit, astar_cost = _audit(dataset_root, maps, trials, protocol)
    if not audit["valid"]:
        _write_json(output_root / "benchmark_audit.json", audit)
        raise RuntimeError(f"benchmark audit failed: {audit['errors'][:3]}")

    print("Aggregating stochastic trials at map level...", flush=True)
    map_level = _build_map_level(maps, trials, astar_cost)
    planner_summary = _planner_summary(map_level)
    overall = _overall_summary(map_level)
    terrain_summary = _dataset_summary(maps)
    paired_terrain = _paired_effect_rows(
        maps,
        metrics=(
            "relief_severity",
            "navigable_fraction",
            "mean_footprint_slope_degrees",
            "collision_boundary_density",
            "connected_components",
            "largest_component_share",
        ),
        identity_key="matched_group_id",
    )
    paired_planner = _planner_paired_effects(map_level)

    tables_root = output_root / "tables"
    figures_root = output_root / "figures"
    _write_csv(tables_root / "planner_map_level.csv", map_level)
    _write_csv(
        tables_root / "planner_summary_family_difficulty.csv", planner_summary
    )
    _write_csv(tables_root / "planner_summary_overall.csv", overall)
    _write_csv(tables_root / "dataset_difficulty_summary.csv", terrain_summary)
    _write_csv(tables_root / "paired_dataset_difficulty_effects.csv", paired_terrain)
    _write_csv(tables_root / "paired_planner_difficulty_effects.csv", paired_planner)
    _write_csv(
        tables_root / "table_success_rate_percent.csv",
        _pivot_table(
            planner_summary,
            "trial_success_rate_percent",
            digits=2,
            include_std_key="trial_success_rate_map_std_percent",
        ),
    )
    _write_csv(
        tables_root / "table_weighted_cost_ratio.csv",
        _pivot_table(
            planner_summary,
            "cost_ratio_to_astar_map_mean",
            digits=3,
            include_std_key="cost_ratio_to_astar_between_map_std",
        ),
    )
    _write_csv(
        tables_root / "table_runtime_sec.csv",
        _pivot_table(
            planner_summary,
            "runtime_map_mean_sec",
            digits=3,
            include_std_key="runtime_between_map_std_sec",
        ),
    )
    _write_csv(
        tables_root / "table_weighted_cost.csv",
        _pivot_table(
            planner_summary,
            "weighted_cost_map_mean",
            digits=3,
            include_std_key="weighted_cost_between_map_std",
        ),
    )

    print("Rendering paper figures...", flush=True)
    _figure_dataset_difficulty(
        terrain_summary, figures_root / "dataset_difficulty_overview.png"
    )
    _figure_planner_overview(
        overall, figures_root / "planner_benchmark_overview.png"
    )
    _heatmap(
        planner_summary,
        metric="trial_success_rate_percent",
        title="Planner success rate by terrain family and difficulty",
        colorbar_label="Success rate (%)",
        output_path=figures_root / "planner_success_heatmap.png",
        vmin=0.0,
        vmax=100.0,
    )
    _heatmap(
        planner_summary,
        metric="cost_ratio_to_astar_map_mean",
        title="Successful-path weighted cost relative to A*",
        colorbar_label="Cost ratio",
        output_path=figures_root / "planner_cost_ratio_heatmap.png",
        vmin=1.0,
    )

    _write_json(output_root / "benchmark_audit.json", audit)
    _write_text(output_root / "RESULTS.md", _results_markdown(audit, overall, protocol))
    result = {
        "valid": True,
        "output_root": str(output_root),
        "map_level_rows": len(map_level),
        "family_difficulty_summary_rows": len(planner_summary),
        "overall_summary_rows": len(overall),
        "figure_count": 4,
    }
    _write_json(output_root / "analysis_state.json", result)
    print(
        f"Benchmark results complete: {len(map_level)} map-algorithm rows, "
        f"{len(planner_summary)} grouped rows, 4 figures.",
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "dataset_5010_v1",
    )
    parser.add_argument("--benchmark-prefix", default="pathfinding_benchmark_v1")
    parser.add_argument(
        "--output",
        type=Path,
        help="default: <dataset>/metadata/benchmark_results",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset.resolve()
    output_root = (
        args.output.resolve()
        if args.output
        else dataset_root / "metadata" / "benchmark_results"
    )
    analyze(dataset_root, args.benchmark_prefix, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
