#!/usr/bin/env python3
"""Generate a small, paired family-by-difficulty 2.5D review dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from generate_and_benchmark import (
    save_showcase_visualization,
    save_traversability_visualization,
)
from terrain_generation import (
    Cell,
    TerrainConfig,
    analyze_terrain,
    connected_components,
    generate_elevation,
    sample_start_goal_pairs,
)


FAMILIES = (
    "smooth_obstacles",
    "rolling",
    "mountain",
    "rugged",
    "plateau",
)


@dataclass(frozen=True)
class DifficultySpec:
    min_navigable: float
    max_navigable: float
    target_navigable: float
    min_largest_component: float
    min_severity: float
    max_severity: float
    max_components: int
    min_largest_component_share: float
    max_boundary_density: float


DIFFICULTIES = {
    "easy": DifficultySpec(0.72, 0.94, 0.84, 0.65, 0.18, 1.05, 40, 0.90, 0.080),
    "medium": DifficultySpec(0.54, 0.76, 0.65, 0.45, 0.48, 2.40, 60, 0.84, 0.085),
    "hard": DifficultySpec(0.40, 0.65, 0.52, 0.34, 0.90, 4.80, 75, 0.80, 0.090),
}

FAMILY_DIFFICULTIES = {
    # Flat mesa tops remain valid robot poses even when their enclosing cliffs
    # are impassable. Plateau difficulty therefore uses a family-calibrated
    # occupancy band instead of injecting texture to mimic other families.
    "plateau": {
        "easy": DifficultySpec(0.80, 0.90, 0.84, 0.70, 0.18, 1.25, 25, 0.90, 0.060),
        "medium": DifficultySpec(0.73, 0.82, 0.78, 0.62, 1.20, 3.20, 30, 0.90, 0.060),
        "hard": DifficultySpec(0.69, 0.76, 0.73, 0.55, 2.80, 6.00, 35, 0.88, 0.060),
    }
}


def _difficulty_spec(family: str, difficulty: str) -> DifficultySpec:
    return FAMILY_DIFFICULTIES.get(family, {}).get(
        difficulty,
        DIFFICULTIES[difficulty],
    )


@dataclass(frozen=True)
class DetailSpec:
    correlation_sigma_cells: float
    strength_scale: float
    maximum_strength_m: float
    activation_quantile: float


DETAIL_SPECS = {
    "smooth_obstacles": DetailSpec(2.8, 0.24, 0.045, 0.58),
    "rolling": DetailSpec(2.3, 0.38, 0.070, 0.52),
    "mountain": DetailSpec(1.7, 0.58, 0.105, 0.44),
    "rugged": DetailSpec(1.2, 0.78, 0.145, 0.34),
    "plateau": DetailSpec(2.8, 0.28, 0.055, 0.58),
}


@dataclass
class TerrainCandidate:
    elevation: np.ndarray
    layers: dict[str, np.ndarray]
    grid: np.ndarray
    components: list[set[Cell]]
    navigable_fraction: float
    largest_component_fraction: float
    severity: float
    detail_strength: float
    boundary_density: float
    largest_component_share: float
    score: float


def _rounded(values: np.ndarray, decimals: int = 4) -> list[list[float]]:
    return np.round(values, decimals).tolist()


def _detail_field(seed: int, size: int, family: str) -> np.ndarray:
    """Create family-aware, spatially localized surface detail."""

    spec = DETAIL_SPECS[family]
    rng = np.random.default_rng(seed ^ 0x5F3759DF)
    micro = ndimage.gaussian_filter(
        rng.standard_normal((size, size)),
        sigma=spec.correlation_sigma_cells,
        mode="reflect",
    )
    micro /= max(float(micro.std()), 1e-12)
    envelope = ndimage.gaussian_filter(
        rng.random((size, size)), sigma=size / 12.0, mode="reflect"
    )
    cutoff = float(np.quantile(envelope, spec.activation_quantile))
    envelope = np.clip(envelope - cutoff, 0.0, None)
    envelope /= max(float(envelope.max()), 1e-12)
    envelope = ndimage.gaussian_filter(envelope, sigma=1.2, mode="reflect")
    return micro * envelope


def _apply_severity(
    base_elevation: np.ndarray,
    detail: np.ndarray,
    severity: float,
    family: str,
) -> tuple[np.ndarray, float]:
    """Change difficulty while retaining the same underlying terrain."""

    detail_spec = DETAIL_SPECS[family]
    raw_detail_strength = (
        0.010 * severity
        + 0.060 * max(0.0, severity - 0.75) ** 1.24
    )
    detail_strength = min(
        raw_detail_strength * detail_spec.strength_scale,
        detail_spec.maximum_strength_m,
    )
    elevation = base_elevation * severity + detail * detail_strength
    if severity < 0.62:
        elevation = ndimage.gaussian_filter(
            elevation,
            sigma=(0.62 - severity) * 1.15,
            mode="reflect",
        )
    elevation -= elevation.min()
    return elevation, detail_strength


def _approximate_largest_component(grid: np.ndarray) -> float:
    free = grid == 0
    labels, count = ndimage.label(free, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return float(sizes.max() / grid.size)


def _boundary_density(grid: np.ndarray) -> float:
    """Fraction of cardinal cell boundaries that change collision state."""

    horizontal = np.count_nonzero(grid[:, 1:] != grid[:, :-1])
    vertical = np.count_nonzero(grid[1:, :] != grid[:-1, :])
    comparisons = grid.shape[0] * (grid.shape[1] - 1) + (
        grid.shape[0] - 1
    ) * grid.shape[1]
    return float((horizontal + vertical) / comparisons)


def _rank_candidates(
    *,
    base_elevation: np.ndarray,
    detail: np.ndarray,
    family: str,
    difficulty: str,
    config: TerrainConfig,
    severity_samples: int,
) -> tuple[list[TerrainCandidate], str]:
    """Return quality-controlled candidates nearest a difficulty target."""

    spec = _difficulty_spec(family, difficulty)
    lightweight = []
    severities = np.geomspace(
        spec.min_severity,
        spec.max_severity,
        severity_samples,
    )
    diagnostic = "no candidate evaluated"
    for severity_value in severities:
        severity = float(severity_value)
        elevation, detail_strength = _apply_severity(
            base_elevation,
            detail,
            severity,
            family,
        )
        # The collision grid is derived from exactly the precision published
        # in JSON, so a reader can reproduce every blocked cell bit-for-bit.
        elevation = np.round(elevation, 6)
        layers = analyze_terrain(elevation, config)
        grid = layers["collision_grid"].astype(np.uint8)
        navigable = float(np.mean(grid == 0))
        largest_approx = _approximate_largest_component(grid)
        boundary_density = _boundary_density(grid)
        band_error = max(0.0, spec.min_navigable - navigable) + max(
            0.0, navigable - spec.max_navigable
        )
        component_error = max(
            0.0,
            spec.min_largest_component - largest_approx,
        )
        score = (
            abs(navigable - spec.target_navigable)
            + 4.0 * band_error
            + 3.0 * component_error
        )
        lightweight.append(
            (
                score,
                elevation,
                layers,
                grid,
                navigable,
                largest_approx,
                severity,
                detail_strength,
                boundary_density,
            )
        )

    feasible: list[TerrainCandidate] = []
    for (
        score,
        elevation,
        layers,
        grid,
        navigable,
        largest_approx,
        severity,
        detail_strength,
        boundary_density,
    ) in sorted(lightweight, key=lambda item: item[0]):
        diagnostic = (
            f"difficulty={difficulty}, navigable={navigable:.3f}, "
            f"largest~={largest_approx:.3f}, boundary={boundary_density:.3f}, "
            f"severity={severity:.3f}"
        )
        if not spec.min_navigable <= navigable <= spec.max_navigable:
            continue
        if largest_approx < spec.min_largest_component * 0.95:
            continue
        if boundary_density > spec.max_boundary_density:
            continue
        components = connected_components(
            grid,
            layers["support_elevation"],
            config,
        )
        if not components:
            continue
        largest_fraction = len(components[0]) / grid.size
        largest_component_share = largest_fraction / max(navigable, 1e-12)
        if largest_fraction < spec.min_largest_component:
            continue
        if len(components) > spec.max_components:
            continue
        if largest_component_share < spec.min_largest_component_share:
            continue
        feasible.append(
            TerrainCandidate(
                elevation=elevation,
                layers=layers,
                grid=grid,
                components=components,
                navigable_fraction=navigable,
                largest_component_fraction=largest_fraction,
                severity=severity,
                detail_strength=detail_strength,
                boundary_density=boundary_density,
                largest_component_share=largest_component_share,
                score=score,
            )
        )
        if len(feasible) == 6:
            break
    return feasible, diagnostic


def _rank_matched_combinations(
    candidate_lists: dict[str, list[TerrainCandidate]],
    difficulties: Sequence[str],
) -> list[dict[str, TerrainCandidate]]:
    """Rank triplets with increasing severity and decreasing free space."""

    combinations = []
    for values in itertools.product(*(candidate_lists[item] for item in difficulties)):
        severities = [candidate.severity for candidate in values]
        navigable = [candidate.navigable_fraction for candidate in values]
        if any(
            severities[index] >= severities[index + 1]
            for index in range(len(values) - 1)
        ):
            continue
        if any(
            navigable[index] <= navigable[index + 1] + 0.01
            for index in range(len(values) - 1)
        ):
            continue
        combinations.append(
            (
                sum(candidate.score for candidate in values),
                dict(zip(difficulties, values)),
            )
        )
    combinations.sort(key=lambda item: item[0])
    return [combination for _, combination in combinations]


def _tasks_are_connected(
    tasks: Sequence[dict],
    candidates: dict[str, TerrainCandidate],
) -> bool:
    for candidate in candidates.values():
        for task in tasks:
            start = tuple(task["start"])
            goal = tuple(task["goal"])
            if not any(
                start in component and goal in component
                for component in candidate.components
            ):
                return False
    return True


def _serialize_payload(
    *,
    instance_id: str,
    matched_group_id: str,
    instance_index: int,
    family: str,
    difficulty: str,
    base_seed: int,
    realized_seed: int,
    generation_attempt: int,
    task_seed: int,
    candidate: TerrainCandidate,
    common_tasks: Sequence[dict],
    reference_difficulty: str,
    config: TerrainConfig,
) -> dict:
    spec = _difficulty_spec(family, difficulty)
    elevation = candidate.elevation
    layers = candidate.layers
    grid = candidate.grid
    tasks = []
    for task in common_tasks:
        task_index = int(task["task_index"])
        tasks.append(
            {
                **task,
                "task_id": f"{instance_id}_sg{task_index:02d}",
                "matched_task_id": f"{matched_group_id}_sg{task_index:02d}",
                "reference_distance_difficulty": reference_difficulty,
            }
        )
    canonical = next(task for task in tasks if task["canonical_visualization"])
    start = canonical["start"]
    goal = canonical["goal"]
    start_x, start_y = start
    goal_x, goal_y = goal
    point_slope = layers["slope_degrees"]
    support_elevation = layers["support_elevation"]
    footprint_slope = layers["footprint_slope_degrees"]
    roughness = layers["roughness"]
    step_height = layers["step_height"]
    centre_blocked = layers["centre_blocked"]
    traversability_cost = layers["traversability_cost"]
    generation = config.as_serializable_dict()
    generation.update(
        {
            "terrain_family": family,
            "difficulty": difficulty,
            "difficulty_target_navigable_fraction": spec.target_navigable,
            "relief_severity": round(candidate.severity, 6),
            "detail_strength_m": round(candidate.detail_strength, 6),
            "detail_correlation_sigma_cells": DETAIL_SPECS[
                family
            ].correlation_sigma_cells,
            "detail_localized": True,
            "slope_estimator": "least-squares plane over circular footprint",
            "navigation_elevation": "footprint support-plane intercept",
            "matched_generation": True,
        }
    )
    return {
        "schema_version": "3.1-preview",
        "instance_id": instance_id,
        "matched_group_id": matched_group_id,
        "instance_index": instance_index,
        "representation": "continuous 2.5D elevation terrain",
        "terrain_family": family,
        "terrain_profile": family,
        "difficulty": difficulty,
        "seed": base_seed,
        "base_seed": base_seed,
        "realized_seed": realized_seed,
        "task_seed": task_seed,
        "size": config.size,
        "grid": grid.tolist(),
        "elevation": _rounded(elevation, 6),
        "terrain_analysis": {
            "point_slope_degrees": _rounded(point_slope, 3),
            "support_elevation": _rounded(support_elevation, 6),
            "footprint_slope_degrees": _rounded(footprint_slope, 3),
            "roughness": _rounded(roughness),
            "step_height": _rounded(step_height),
            "traversability_cost": _rounded(traversability_cost, 3),
            "blocked_by_slope": layers["blocked_by_slope"]
            .astype(np.uint8)
            .tolist(),
            "blocked_by_roughness": layers["blocked_by_roughness"]
            .astype(np.uint8)
            .tolist(),
            "blocked_by_step": layers["blocked_by_step"].astype(np.uint8).tolist(),
            "blocked_by_combined_cost": layers["blocked_by_combined_cost"]
            .astype(np.uint8)
            .tolist(),
            "centre_blocked": centre_blocked.astype(np.uint8).tolist(),
        },
        "tasks": tasks,
        "canonical_task_id": canonical["task_id"],
        "start": start,
        "goal": goal,
        "start_z": round(float(elevation[start_y, start_x]), 4),
        "goal_z": round(float(elevation[goal_y, goal_x]), 4),
        "coordinate_frame": {
            "name": "map",
            "units": "m",
            "origin_xyz": [0.0, 0.0, 0.0],
            "x_axis": "column * cell_size",
            "y_axis": "row * cell_size",
            "z_axis": "elevation[row][column]",
        },
        "navigation": {
            "neighbors": 8,
            "diagonal_corner_cutting": False,
            "cell_size": config.cell_size,
            "footprint_radius": config.footprint_radius,
            "endpoint_margin": config.endpoint_margin,
            "max_slope": config.max_slope,
            "max_slope_degrees": config.max_slope_degrees,
            "max_step_height": config.max_step_height,
            "max_roughness": config.max_roughness,
            "slope_weight": config.slope_weight,
            "path_slope_weight": config.slope_weight,
            "traversability_weights": {
                "slope": config.traversability_slope_weight,
                "roughness": config.traversability_roughness_weight,
                "step": config.traversability_step_weight,
            },
        },
        "generation": generation,
        "difficulty_definition": {
            "min_navigable_fraction": spec.min_navigable,
            "max_navigable_fraction": spec.max_navigable,
            "target_navigable_fraction": spec.target_navigable,
            "min_largest_component_fraction": spec.min_largest_component,
            "max_connected_components": spec.max_components,
            "min_largest_component_share": spec.min_largest_component_share,
            "max_collision_boundary_density": spec.max_boundary_density,
        },
        "metrics": {
            "min_elevation": round(float(elevation.min()), 4),
            "max_elevation": round(float(elevation.max()), 4),
            "mean_elevation": round(float(elevation.mean()), 4),
            "mean_point_slope_degrees": round(float(point_slope.mean()), 4),
            "max_point_slope_degrees": round(float(point_slope.max()), 4),
            "mean_footprint_slope_degrees": round(
                float(footprint_slope.mean()), 4
            ),
            "max_footprint_slope_degrees": round(float(footprint_slope.max()), 4),
            "mean_roughness": round(float(roughness.mean()), 4),
            "max_roughness": round(float(roughness.max()), 4),
            "max_step_height": round(float(step_height.max()), 4),
            "centre_blocked_fraction": round(float(centre_blocked.mean()), 6),
            "blocked_by_slope_fraction": round(
                float(layers["blocked_by_slope"].mean()), 6
            ),
            "blocked_by_roughness_fraction": round(
                float(layers["blocked_by_roughness"].mean()), 6
            ),
            "blocked_by_step_fraction": round(
                float(layers["blocked_by_step"].mean()), 6
            ),
            "collision_boundary_density": round(candidate.boundary_density, 6),
            "navigable_fraction": round(candidate.navigable_fraction, 6),
            "connected_components": len(candidate.components),
            "largest_component_fraction": round(
                candidate.largest_component_fraction, 6
            ),
            "largest_component_share": round(
                candidate.largest_component_share, 6
            ),
            "task_count": len(tasks),
            "canonical_reference_weighted_distance": canonical[
                "reference_weighted_distance"
            ],
            "generation_attempt": generation_attempt,
        },
    }


def _build_matched_group(
    *,
    matched_group_id: str,
    instance_index: int,
    family: str,
    base_seed: int,
    difficulties: Sequence[str],
    config: TerrainConfig,
    max_attempts: int,
    severity_samples: int,
    task_count: int,
) -> tuple[dict[str, dict], dict]:
    last_diagnostic = "no attempt completed"
    for attempt in range(max_attempts):
        realized_seed = base_seed + attempt * 104729
        base_elevation = generate_elevation(config, realized_seed, family)
        detail = _detail_field(realized_seed, config.size, family)
        candidate_lists: dict[str, list[TerrainCandidate]] = {}
        diagnostics = []
        for difficulty in difficulties:
            candidates, diagnostic = _rank_candidates(
                base_elevation=base_elevation,
                detail=detail,
                family=family,
                difficulty=difficulty,
                config=config,
                severity_samples=severity_samples,
            )
            candidate_lists[difficulty] = candidates
            diagnostics.append(diagnostic)
        if any(not candidates for candidates in candidate_lists.values()):
            last_diagnostic = "; ".join(diagnostics)
            continue

        combinations = _rank_matched_combinations(candidate_lists, difficulties)
        if not combinations:
            last_diagnostic = "difficulty candidates exist but are not monotonic"
            continue
        reference_difficulty = difficulties[-1]
        task_seed = realized_seed ^ 0x6A09E667
        for candidates in combinations[:12]:
            reference_candidate = candidates[reference_difficulty]
            try:
                common_tasks, _ = sample_start_goal_pairs(
                    reference_candidate.grid,
                    reference_candidate.layers["support_elevation"],
                    config,
                    count=task_count,
                    seed=task_seed,
                )
            except ValueError as error:
                last_diagnostic = str(error)
                continue
            if not _tasks_are_connected(common_tasks, candidates):
                last_diagnostic = "sampled tasks are not connected in every difficulty"
                continue

            payloads = {}
            for difficulty, candidate in candidates.items():
                instance_id = f"{matched_group_id}_{difficulty}"
                payloads[difficulty] = _serialize_payload(
                    instance_id=instance_id,
                    matched_group_id=matched_group_id,
                    instance_index=instance_index,
                    family=family,
                    difficulty=difficulty,
                    base_seed=base_seed,
                    realized_seed=realized_seed,
                    generation_attempt=attempt + 1,
                    task_seed=task_seed,
                    candidate=candidate,
                    common_tasks=common_tasks,
                    reference_difficulty=reference_difficulty,
                    config=config,
                )
            shared_tasks = []
            for task in common_tasks:
                task_index = int(task["task_index"])
                shared_tasks.append(
                    {
                        **task,
                        "matched_task_id": (
                            f"{matched_group_id}_sg{task_index:02d}"
                        ),
                        "reference_distance_difficulty": reference_difficulty,
                    }
                )
            task_payload = {
                "schema_version": "1.0-preview",
                "matched_group_id": matched_group_id,
                "family": family,
                "instance_index": instance_index,
                "base_seed": base_seed,
                "realized_seed": realized_seed,
                "task_seed": task_seed,
                "reference_difficulty": reference_difficulty,
                "tasks": shared_tasks,
            }
            return payloads, task_payload
    raise RuntimeError(
        f"could not create matched group {matched_group_id} after "
        f"{max_attempts} attempts: {last_diagnostic}"
    )


def _atomic_write_json(payload: dict, path: Path, *, compact: bool = False) -> None:
    """Durably replace one JSON file without exposing a partial write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            separators=(",", ":") if compact else None,
            indent=None if compact else 2,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _map_row(
    payload: dict,
    output_root: Path,
    *,
    render_images: bool,
) -> dict:
    instance_id = payload["instance_id"]
    family = payload["terrain_family"]
    map_path = output_root / "maps" / family / f"{instance_id}.json"
    terrain_path = (
        output_root / "images" / "terrain" / family / f"{instance_id}.png"
    )
    analysis_path = (
        output_root
        / "images"
        / "analysis"
        / family
        / f"{instance_id}_analysis.png"
    )
    metrics = payload["metrics"]
    return {
        "instance_id": instance_id,
        "matched_group_id": payload["matched_group_id"],
        "instance_index": payload["instance_index"],
        "family": payload["terrain_family"],
        "difficulty": payload["difficulty"],
        "base_seed": payload["base_seed"],
        "realized_seed": payload["realized_seed"],
        "relief_severity": payload["generation"]["relief_severity"],
        "detail_strength_m": payload["generation"]["detail_strength_m"],
        "size": payload["size"],
        "cell_size": payload["navigation"]["cell_size"],
        "task_count": len(payload["tasks"]),
        "canonical_task_id": payload["canonical_task_id"],
        "max_elevation": metrics["max_elevation"],
        "mean_point_slope_degrees": metrics["mean_point_slope_degrees"],
        "mean_footprint_slope_degrees": metrics[
            "mean_footprint_slope_degrees"
        ],
        "mean_roughness": metrics["mean_roughness"],
        "blocked_by_slope_fraction": metrics["blocked_by_slope_fraction"],
        "blocked_by_roughness_fraction": metrics[
            "blocked_by_roughness_fraction"
        ],
        "blocked_by_step_fraction": metrics["blocked_by_step_fraction"],
        "navigable_fraction": metrics["navigable_fraction"],
        "collision_boundary_density": metrics["collision_boundary_density"],
        "connected_components": metrics["connected_components"],
        "largest_component_fraction": metrics["largest_component_fraction"],
        "largest_component_share": metrics["largest_component_share"],
        "map_file": map_path.relative_to(output_root).as_posix(),
        "terrain_image_file": (
            terrain_path.relative_to(output_root).as_posix()
            if render_images
            else ""
        ),
        "traversability_image_file": (
            analysis_path.relative_to(output_root).as_posix()
            if render_images
            else ""
        ),
    }


def _task_rows(task_payload: dict) -> list[dict]:
    rows = []
    for task in task_payload["tasks"]:
        rows.append(
            {
                "matched_group_id": task_payload["matched_group_id"],
                "family": task_payload["family"],
                "instance_index": task_payload["instance_index"],
                "matched_task_id": task["matched_task_id"],
                "task_index": task["task_index"],
                "start_x": task["start"][0],
                "start_y": task["start"][1],
                "goal_x": task["goal"][0],
                "goal_y": task["goal"][1],
                "distance_class": task["distance_class"],
                "reference_weighted_distance": task[
                    "reference_weighted_distance"
                ],
                "reference_distance_difficulty": task[
                    "reference_distance_difficulty"
                ],
                "canonical_visualization": task["canonical_visualization"],
            }
        )
    return rows


def _generate_group_worker(job: dict) -> dict:
    """Generate and atomically publish one matched difficulty group."""

    output_root = Path(job["output_root"])
    config = TerrainConfig(**job["config"])
    difficulties = tuple(job["difficulties"])
    payloads, task_payload = _build_matched_group(
        matched_group_id=job["matched_group_id"],
        instance_index=job["instance_index"],
        family=job["family"],
        base_seed=job["base_seed"],
        difficulties=difficulties,
        config=config,
        max_attempts=job["max_attempts"],
        severity_samples=job["severity_samples"],
        task_count=job["task_count"],
    )
    map_rows = []
    map_hashes = {}
    navigable = {}
    render_images = bool(job["render_images"])
    for difficulty in difficulties:
        payload = payloads[difficulty]
        map_path = (
            output_root
            / "maps"
            / payload["terrain_family"]
            / f"{payload['instance_id']}.json"
        )
        _atomic_write_json(payload, map_path, compact=True)
        relative_path = map_path.relative_to(output_root).as_posix()
        map_hashes[relative_path] = _sha256_file(map_path)
        map_rows.append(
            _map_row(payload, output_root, render_images=render_images)
        )
        navigable[difficulty] = payload["metrics"]["navigable_fraction"]
    return {
        "record_type": "completed_group",
        "matched_group_id": job["matched_group_id"],
        "family": job["family"],
        "instance_index": job["instance_index"],
        "map_rows": map_rows,
        "task_rows": _task_rows(task_payload),
        "map_hashes": map_hashes,
        "navigable_fraction": navigable,
    }


def _append_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_csv_atomic(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_checkpoints(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A crash can leave only the final append incomplete.
                if any(item.strip() for item in handle):
                    raise ValueError(
                        f"invalid checkpoint record at line {line_number}: {path}"
                    )
                break
            records[record["matched_group_id"]] = record
    return records


def _checkpoint_is_valid(output_root: Path, record: dict) -> bool:
    hashes = record.get("map_hashes", {})
    if not hashes:
        return False
    return all(
        (output_root / relative_path).is_file()
        and _sha256_file(output_root / relative_path) == expected_hash
        for relative_path, expected_hash in hashes.items()
    )


def _generation_signature(args: argparse.Namespace, config: TerrainConfig) -> dict:
    return {
        "schema_version": "1.0",
        "generator_revision": "family-detail-support-plane-v3",
        "families": list(args.families),
        "difficulties": list(args.difficulties),
        "instances_per_family": args.instances_per_family,
        "task_count": args.task_count,
        "seed": args.seed,
        "max_attempts": args.max_attempts,
        "severity_samples": args.severity_samples,
        "render_all_images": bool(args.render_all_images),
        "dataset_mode": args.dataset_mode,
        "terrain_configuration": asdict(config),
    }


def _signature_hash(signature: dict) -> str:
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _render_map_images_worker(job: dict) -> dict:
    """Render both review images for one map, skipping durable outputs."""

    map_path = Path(job["map_path"])
    terrain_path = Path(job["terrain_path"])
    analysis_path = Path(job["analysis_path"])
    with map_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    force = bool(job.get("force", False))
    rendered = 0
    if force or not _nonempty_file(terrain_path):
        temporary_path = terrain_path.with_name(
            f".{terrain_path.stem}.{os.getpid()}.tmp.png"
        )
        try:
            save_showcase_visualization(payload, temporary_path)
            os.replace(temporary_path, terrain_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        rendered += 1
    if force or not _nonempty_file(analysis_path):
        temporary_path = analysis_path.with_name(
            f".{analysis_path.stem}.{os.getpid()}.tmp.png"
        )
        try:
            save_traversability_visualization(payload, temporary_path)
            os.replace(temporary_path, analysis_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        rendered += 1
    return {"instance_id": payload["instance_id"], "rendered": rendered}


def _render_dataset_images(
    output_root: Path,
    map_rows: Sequence[dict],
    workers: int,
    *,
    force: bool = False,
) -> int:
    jobs = []
    for row in map_rows:
        if not row["terrain_image_file"]:
            continue
        terrain_path = output_root / row["terrain_image_file"]
        analysis_path = output_root / row["traversability_image_file"]
        if (
            not force
            and _nonempty_file(terrain_path)
            and _nonempty_file(analysis_path)
        ):
            continue
        jobs.append(
            {
                "map_path": str(output_root / row["map_file"]),
                "terrain_path": str(terrain_path),
                "analysis_path": str(analysis_path),
                "force": force,
            }
        )
    if not jobs:
        return 0

    completed = 0

    def report(result: dict) -> None:
        nonlocal completed
        completed += 1
        if completed == len(jobs) or completed % 25 == 0:
            print(
                f"Rendered map images {completed}/{len(jobs)}: "
                f"{result['instance_id']}"
            )

    if workers == 1:
        for job in jobs:
            report(_render_map_images_worker(job))
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        futures = {
            executor.submit(_render_map_images_worker, job): job for job in jobs
        }
        try:
            for future in as_completed(futures):
                report(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    return len(jobs)


def _safe_clean(output_root: Path) -> None:
    """Remove only a directory previously produced by this preview script."""

    output_root = output_root.resolve()
    project_root = Path(__file__).resolve().parents[1]
    if output_root == project_root or project_root not in output_root.parents:
        raise ValueError(f"refusing to clean unsafe output path: {output_root}")
    if not output_root.exists():
        return
    manifest_path = output_root / "metadata" / "manifest.json"
    sentinel_path = output_root / ".paired_preview_output"
    if not manifest_path.is_file() and not sentinel_path.is_file():
        raise ValueError(
            f"refusing to clean unrecognized directory: {output_root}"
        )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("generated_by") != Path(__file__).name:
            raise ValueError(f"refusing to clean non-preview dataset: {output_root}")
    shutil.rmtree(output_root)


def _save_review_contact_sheet(
    terrain_root: Path,
    output_path: Path,
    families: Sequence[str],
    difficulties: Sequence[str],
    instance_index: int,
) -> None:
    """Create one compact matched family-by-difficulty review sheet."""

    figure, axes = plt.subplots(
        len(families),
        len(difficulties),
        figsize=(18, 6.0 * len(families)),
        dpi=100,
        squeeze=False,
        facecolor="#eef2f2",
    )
    for row, family in enumerate(families):
        for column, difficulty in enumerate(difficulties):
            instance_id = (
                f"terrain_{family}_{instance_index:03d}_{difficulty}"
            )
            image_path = terrain_root / family / f"{instance_id}.png"
            axes[row, column].imshow(plt.imread(image_path))
            axes[row, column].set_axis_off()
            axes[row, column].set_title(
                f"{family.replace('_', ' ').title()} — {difficulty.title()}",
                fontsize=14,
                fontweight="semibold",
                pad=8,
            )
    figure.suptitle(
        f"Matched 2.5D terrain family × difficulty — triplet {instance_index:03d}",
        fontsize=22,
        fontweight="semibold",
        y=0.998,
    )
    figure.subplots_adjust(
        left=0.008,
        right=0.992,
        bottom=0.004,
        top=0.988,
        wspace=0.012,
        hspace=0.045,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor=figure.get_facecolor(), pad_inches=0.02)
    plt.close(figure)


def _write_dataset_readme(
    output_root: Path,
    total_maps: int,
    total_tasks: int,
    families: Sequence[str],
    instances_per_family: int,
    dataset_mode: str,
) -> None:
    family_lines = "\n".join(f"- {family}" for family in families)
    title = "Dataset" if dataset_mode == "final" else "Preview"
    readme = f"""# Paired 2.5D Terrain {title}

This set contains {total_maps} maps and {total_tasks} map-task records.
It has {instances_per_family} matched terrain group per family. Easy, Medium,
and Hard members of a group share the same base terrain seed and the same
Start-Goal coordinates.

Families

{family_lines}

Protocol

- General families: Easy 72-94, Medium 54-76, Hard 40-65 percent navigable
- Plateau: Easy 80-90, Medium 73-82, Hard 69-76 percent navigable
- Difficulty also constrains connected-component count, largest-component share,
  and collision-boundary density
- Ten shared tasks per group: 3 short, 4 medium, 3 long
- One longest task is marked canonical for preview rendering

Structure

    {output_root.name}/
    |-- maps/<family>/
    |-- images/terrain/<family>/
    |-- images/analysis/<family>/
    |-- images/paths/<family>/
    |-- images/contact_sheets/
    |-- metadata/dataset_summary.csv
    |-- metadata/task_summary.csv
    |-- metadata/manifest.json
    |-- metadata/generation_state.json
    |-- metadata/generation_progress.jsonl
    |-- metadata/validation_report.json
    +-- README.md

Tasks are embedded in each map JSON. The task_summary.csv file is a compact
cross-map index rather than a second copy of task definitions.

Map and optional image files are partitioned by terrain family. This keeps the
three matched difficulty members adjacent while avoiding one directory with
thousands of files. At 5,010 maps, each family directory contains 1,002 JSON
files; dataset_summary.csv remains the canonical cross-family index.

The standardized planner writes raw trials, family-by-difficulty mean and
standard deviation summaries, and its fixed-budget protocol into metadata/.
Generation can be resumed using the identical command plus --resume.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def generate_preview(args: argparse.Namespace) -> None:
    config = TerrainConfig(
        size=args.size,
        cell_size=args.cell_size,
        footprint_radius=args.footprint_radius,
        endpoint_margin=args.endpoint_margin,
        max_slope_degrees=args.max_slope_degrees,
        max_step_height=args.max_step_height,
        max_roughness=args.max_roughness,
        slope_weight=args.slope_weight,
        traversability_slope_weight=args.traversability_slope_weight,
        traversability_roughness_weight=args.traversability_roughness_weight,
        traversability_step_weight=args.traversability_step_weight,
        min_navigable_fraction=0.20,
        max_navigable_fraction=0.95,
        min_largest_component_fraction=0.20,
    )
    config.validate()
    difficulties = tuple(
        item for item in DIFFICULTIES if item in set(args.difficulties)
    )
    if len(difficulties) != len(args.difficulties):
        raise ValueError("difficulty list contains duplicates")
    output_root = args.output.resolve()
    if args.clean:
        _safe_clean(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"output directory is not empty: {output_root}; use --resume to "
            "continue a compatible run or --clean to rebuild it"
        )

    images_root = output_root / "images"
    terrain_root = images_root / "terrain"
    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    sentinel_path = output_root / ".paired_preview_output"
    state_path = metadata_root / "generation_state.json"
    checkpoint_path = metadata_root / "generation_progress.jsonl"
    signature = _generation_signature(args, config)
    signature_hash = _signature_hash(signature)
    if args.resume:
        if not sentinel_path.is_file() or not state_path.is_file():
            raise ValueError(
                f"cannot resume unrecognized or incomplete output: {output_root}"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature_hash") != signature_hash:
            raise ValueError(
                "resume configuration differs from the original generation run"
            )
    else:
        sentinel_path.write_text(
            "generated by generate_25d_dataset.py\n",
            encoding="utf-8",
        )
        state = {
            "schema_version": "1.0",
            "generated_by": Path(__file__).name,
            "status": "in_progress",
            "signature_hash": signature_hash,
            "signature": signature,
            "workers_used": [],
        }
        _atomic_write_json(state, state_path)

    group_count = len(args.families) * args.instances_per_family
    jobs = []
    for family in args.families:
        family_seed_offset = FAMILIES.index(family) * 1_000_003
        for instance_index in range(1, args.instances_per_family + 1):
            base_seed = (
                args.seed
                + family_seed_offset
                + (instance_index - 1) * 10_007
            )
            matched_group_id = f"terrain_{family}_{instance_index:03d}"
            jobs.append(
                {
                    "output_root": str(output_root),
                    "matched_group_id": matched_group_id,
                    "instance_index": instance_index,
                    "family": family,
                    "base_seed": base_seed,
                    "difficulties": difficulties,
                    "config": asdict(config),
                    "max_attempts": args.max_attempts,
                    "severity_samples": args.severity_samples,
                    "task_count": args.task_count,
                    "render_images": (
                        instance_index == 1 or args.render_all_images
                    ),
                }
            )

    loaded_records = _load_checkpoints(checkpoint_path)
    completed_records = {
        group_id: record
        for group_id, record in loaded_records.items()
        if _checkpoint_is_valid(output_root, record)
    }
    pending_jobs = [
        job
        for job in jobs
        if job["matched_group_id"] not in completed_records
    ]
    if completed_records:
        print(
            f"Resume verified {len(completed_records)}/{group_count} completed groups; "
            f"{len(pending_jobs)} remain"
        )
    if pending_jobs and args.workers not in state.setdefault("workers_used", []):
        state["workers_used"].append(args.workers)
        state["status"] = "in_progress"
        _atomic_write_json(state, state_path)

    completed_count = len(completed_records)

    def commit_result(result: dict) -> None:
        nonlocal completed_count
        _append_checkpoint(checkpoint_path, result)
        completed_records[result["matched_group_id"]] = result
        completed_count += 1
        navigation_values = ", ".join(
            f"{difficulty}={result['navigable_fraction'][difficulty]:.1%}"
            for difficulty in difficulties
        )
        print(
            f"[{completed_count:04d}/{group_count:04d}] "
            f"{result['matched_group_id']}: {navigation_values}; "
            f"{args.task_count} shared S-G tasks"
        )

    try:
        if args.workers == 1:
            for job in pending_jobs:
                commit_result(_generate_group_worker(job))
        elif pending_jobs:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            futures = {
                executor.submit(_generate_group_worker, job): job
                for job in pending_jobs
            }
            try:
                for future in as_completed(futures):
                    commit_result(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
    except BaseException:
        state["status"] = "interrupted"
        state["completed_groups"] = len(completed_records)
        _atomic_write_json(state, state_path)
        raise

    if len(completed_records) != group_count:
        raise RuntimeError(
            f"generation stopped with {len(completed_records)}/{group_count} groups"
        )

    family_order = {family: index for index, family in enumerate(args.families)}
    difficulty_order = {
        difficulty: index for index, difficulty in enumerate(difficulties)
    }
    ordered_records = sorted(
        completed_records.values(),
        key=lambda record: (
            family_order[record["family"]],
            int(record["instance_index"]),
        ),
    )
    map_rows = [row for record in ordered_records for row in record["map_rows"]]
    map_rows.sort(
        key=lambda row: (
            family_order[row["family"]],
            int(row["instance_index"]),
            difficulty_order[row["difficulty"]],
        )
    )
    task_rows = [row for record in ordered_records for row in record["task_rows"]]
    task_rows.sort(
        key=lambda row: (
            family_order[row["family"]],
            int(row["instance_index"]),
            int(row["task_index"]),
        )
    )
    _write_csv_atomic(metadata_root / "dataset_summary.csv", map_rows)
    _write_csv_atomic(metadata_root / "task_summary.csv", task_rows)
    try:
        rendered_map_count = _render_dataset_images(
            output_root,
            map_rows,
            args.workers,
            force=args.force_render_images,
        )
    except BaseException:
        state["status"] = "interrupted"
        state["completed_groups"] = group_count
        state["phase"] = "rendering_map_images"
        _atomic_write_json(state, state_path)
        raise

    manifest = {
        "schema_version": "2.1",
        "generated_by": Path(__file__).name,
        "purpose": (
            "small matched family x difficulty review set"
            if args.dataset_mode == "preview"
            else "matched 2.5D terrain dataset"
        ),
        "preview_only": args.dataset_mode == "preview",
        "status": "complete",
        "families": list(args.families),
        "difficulties": list(difficulties),
        "instances_per_family": args.instances_per_family,
        "maps_per_matched_group": len(difficulties),
        "tasks_per_matched_group": args.task_count,
        "matched_difficulty_design": True,
        "shared_start_goal_across_difficulties": True,
        "task_storage": "embedded in each map JSON and indexed by metadata/task_summary.csv",
        "total_matched_groups": group_count,
        "total_maps": len(map_rows),
        "total_unique_tasks": len(task_rows),
        "total_map_task_records": len(map_rows) * args.task_count,
        "render_style": "terrain_natural_accessible_v1",
        "visual_palette": {
            "showcase": "natural_accessible",
            "elevation_analysis": "cividis",
            "traversability_cost": "viridis_r",
            "categorical_causes": "okabe_ito_derived",
        },
        "storage_layout": "family_partitioned_v1",
        "image_layout": {
            "terrain": "images/terrain/<family>",
            "analysis": "images/analysis/<family>",
            "paths": "images/paths/<family>",
            "contact_sheets": "images/contact_sheets",
            "terrain_contact_sheet": (
                "images/contact_sheets/terrain_triplet_001.png"
            ),
            "path_contact_sheet": (
                "images/contact_sheets/path_hard.png"
            ),
        },
        "representative_images_only": not args.render_all_images,
        "representative_instance_index": 1,
        "path_planning_outputs_included": False,
        "generation_execution": {
            "workers_used": state.get("workers_used") or [args.workers],
            "parallel_image_rendering": True,
            "maps_with_requested_images": sum(
                bool(row["terrain_image_file"]) for row in map_rows
            ),
            "maps_rendered_this_invocation": rendered_map_count,
            "resume_supported": True,
            "checkpoint_unit": "matched terrain group",
            "checkpoint_file": "metadata/generation_progress.jsonl",
            "signature_hash": signature_hash,
        },
        "configuration": config.as_serializable_dict(),
    }
    _atomic_write_json(manifest, metadata_root / "manifest.json")
    # Only instance 001 has review images in the scalable/default mode.
    # Contact sheets for every instance are valid only when every image was
    # explicitly requested with --render-all-images.
    contact_sheet_indices = (
        range(1, args.instances_per_family + 1)
        if args.render_all_images
        else (1,)
    )
    for instance_index in contact_sheet_indices:
        contact_sheet_name = (
            "terrain_triplet_001.png"
            if instance_index == 1
            else f"terrain_triplet_{instance_index:03d}.png"
        )
        _save_review_contact_sheet(
            terrain_root,
            images_root / "contact_sheets" / contact_sheet_name,
            args.families,
            difficulties,
            instance_index,
        )
    _write_dataset_readme(
        output_root,
        total_maps=len(map_rows),
        total_tasks=len(map_rows) * args.task_count,
        families=args.families,
        instances_per_family=args.instances_per_family,
        dataset_mode=args.dataset_mode,
    )
    state["status"] = "complete"
    state["phase"] = "complete"
    state["completed_groups"] = group_count
    state["total_maps"] = len(map_rows)
    _atomic_write_json(state, state_path)
    print(f"Completed {len(map_rows)} maps in {output_root}")


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "dataset_preview",
    )
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument(
        "--difficulties",
        nargs="+",
        choices=tuple(DIFFICULTIES),
        default=tuple(DIFFICULTIES),
    )
    parser.add_argument("--instances-per-family", type=int, default=2)
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--footprint-radius", type=float, default=0.45)
    parser.add_argument("--endpoint-margin", type=float, default=1.50)
    parser.add_argument("--max-slope-degrees", type=float, default=28.0)
    parser.add_argument("--max-step-height", type=float, default=0.65)
    parser.add_argument("--max-roughness", type=float, default=0.10)
    parser.add_argument("--slope-weight", type=float, default=3.0)
    parser.add_argument("--traversability-slope-weight", type=float, default=0.45)
    parser.add_argument(
        "--traversability-roughness-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument("--traversability-step-weight", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--severity-samples", type=int, default=34)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, os.cpu_count() or 1)),
        help="parallel matched-group workers (default: up to 6)",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("preview", "final"),
        default="preview",
    )
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--clean", action="store_true")
    lifecycle.add_argument("--resume", action="store_true")
    parser.add_argument("--render-all-images", action="store_true")
    parser.add_argument(
        "--force-render-images",
        action="store_true",
        help="overwrite requested terrain/analysis PNGs without changing maps",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.instances_per_family < 1:
        raise ValueError("instances-per-family must be at least one")
    if args.task_count < 1:
        raise ValueError("task-count must be at least one")
    if args.workers < 1:
        raise ValueError("workers must be at least one")
    generate_preview(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
