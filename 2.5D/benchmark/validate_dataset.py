#!/usr/bin/env python3
"""Validate paired 2.5D preview data and reproduce its traversability grids."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Sequence

import numpy as np

from terrain_generation import TerrainConfig, analyze_terrain, connected_components


DIFFICULTY_ORDER = ("easy", "medium", "hard")


class Validation:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.maps_checked = 0
        self.tasks_checked = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _config_from_manifest(manifest: dict) -> TerrainConfig:
    names = {field.name for field in fields(TerrainConfig)}
    raw = manifest["configuration"]
    return TerrainConfig(**{key: raw[key] for key in names if key in raw})


def _task_signature(task: dict) -> tuple:
    return (
        int(task["task_index"]),
        tuple(task["start"]),
        tuple(task["goal"]),
        task["distance_class"],
        float(task["reference_weighted_distance"]),
        bool(task["canonical_visualization"]),
    )


def _validate_map(
    path: Path,
    payload: dict,
    config: TerrainConfig,
    validation: Validation,
) -> None:
    label = payload.get("instance_id", path.name)
    size = int(payload.get("size", -1))
    grid = np.asarray(payload.get("grid"), dtype=np.uint8)
    elevation = np.asarray(payload.get("elevation"), dtype=float)
    validation.require(
        grid.shape == (size, size),
        f"{label}: grid shape {grid.shape} does not match size {size}",
    )
    validation.require(
        elevation.shape == (size, size),
        f"{label}: elevation shape {elevation.shape} does not match size {size}",
    )
    if grid.shape != elevation.shape or grid.ndim != 2:
        return
    validation.require(
        bool(np.all(np.isfinite(elevation))),
        f"{label}: elevation contains non-finite values",
    )
    validation.require(
        bool(np.all((grid == 0) | (grid == 1))),
        f"{label}: grid is not binary",
    )

    recomputed = analyze_terrain(elevation, config)
    support_elevation = recomputed["support_elevation"]
    reproduced_grid = recomputed["collision_grid"].astype(np.uint8)
    mismatch_count = int(np.count_nonzero(grid != reproduced_grid))
    validation.require(
        mismatch_count == 0,
        f"{label}: {mismatch_count} collision cells cannot be reproduced",
    )

    navigable = float(np.mean(grid == 0))
    reported_navigable = float(payload["metrics"]["navigable_fraction"])
    validation.require(
        abs(navigable - reported_navigable) <= 1e-6,
        f"{label}: reported navigable fraction is inconsistent",
    )
    definition = payload["difficulty_definition"]
    validation.require(
        float(definition["min_navigable_fraction"])
        <= navigable
        <= float(definition["max_navigable_fraction"]),
        f"{label}: navigable fraction is outside its difficulty band",
    )

    reported_support = payload.get("terrain_analysis", {}).get(
        "support_elevation"
    )
    if reported_support is not None:
        support_mismatch = float(
            np.max(
                np.abs(
                    np.asarray(reported_support, dtype=float)
                    - support_elevation
                )
            )
        )
        validation.require(
            support_mismatch <= 1.1e-6,
            f"{label}: support elevation cannot be reproduced",
        )

    components = connected_components(grid, support_elevation, config)
    largest_fraction = len(components[0]) / grid.size if components else 0.0
    largest_share = largest_fraction / max(navigable, 1e-12)
    validation.require(
        abs(
            largest_fraction
            - float(payload["metrics"]["largest_component_fraction"])
        )
        <= 1e-6,
        f"{label}: reported largest-component fraction is inconsistent",
    )
    validation.require(
        largest_fraction
        >= float(definition["min_largest_component_fraction"]),
        f"{label}: largest component is below the required minimum",
    )
    validation.require(
        len(components) <= int(definition.get("max_connected_components", 10**9)),
        f"{label}: too many traversable components",
    )
    validation.require(
        largest_share
        >= float(definition.get("min_largest_component_share", 0.0)),
        f"{label}: largest component contains too little free space",
    )
    boundary_changes = np.count_nonzero(grid[:, 1:] != grid[:, :-1]) + np.count_nonzero(
        grid[1:, :] != grid[:-1, :]
    )
    boundary_comparisons = grid.shape[0] * (grid.shape[1] - 1) + (
        grid.shape[0] - 1
    ) * grid.shape[1]
    boundary_density = float(boundary_changes / boundary_comparisons)
    validation.require(
        boundary_density
        <= float(definition.get("max_collision_boundary_density", 1.0))
        + 1e-12,
        f"{label}: collision mask is too fragmented",
    )

    tasks = payload.get("tasks", [])
    validation.require(
        len(tasks) == int(payload["metrics"]["task_count"]),
        f"{label}: task count is inconsistent",
    )
    validation.require(
        sum(bool(task.get("canonical_visualization")) for task in tasks) == 1,
        f"{label}: exactly one task must be canonical",
    )
    class_counts = {
        item: sum(task.get("distance_class") == item for task in tasks)
        for item in ("short", "medium", "long")
    }
    if len(tasks) == 10:
        validation.require(
            class_counts == {"short": 3, "medium": 4, "long": 3},
            f"{label}: expected task split 3 short, 4 medium, 3 long",
        )
    for task in tasks:
        task_label = task.get("task_id", f"{label}/unknown")
        start = tuple(int(value) for value in task["start"])
        goal = tuple(int(value) for value in task["goal"])
        for point_name, point in (("start", start), ("goal", goal)):
            x, y = point
            validation.require(
                0 <= x < size and 0 <= y < size,
                f"{task_label}: {point_name} lies outside the map",
            )
            if 0 <= x < size and 0 <= y < size:
                validation.require(
                    grid[y, x] == 0,
                    f"{task_label}: {point_name} lies in a blocked cell",
                )
        validation.require(
            any(start in component and goal in component for component in components),
            f"{task_label}: Start and Goal are not connected",
        )
        validation.tasks_checked += 1

    canonical = [
        task for task in tasks if bool(task.get("canonical_visualization"))
    ]
    if len(canonical) == 1:
        validation.require(
            payload.get("start") == canonical[0]["start"]
            and payload.get("goal") == canonical[0]["goal"],
            f"{label}: backward-compatible Start/Goal do not match canonical task",
        )
        validation.require(
            payload.get("canonical_task_id") == canonical[0].get("task_id"),
            f"{label}: canonical_task_id is inconsistent",
        )
    validation.maps_checked += 1


def validate_dataset(dataset_root: Path, write_report: bool = True) -> Validation:
    validation = Validation()
    manifest_path = dataset_root / "metadata" / "manifest.json"
    validation.require(manifest_path.is_file(), "metadata/manifest.json is missing")
    if not manifest_path.is_file():
        return validation
    manifest = _load_json(manifest_path)
    validation.require(
        manifest.get("generated_by") == "generate_25d_dataset.py",
        "dataset was not produced by the paired preview generator",
    )
    config = _config_from_manifest(manifest)
    try:
        config.validate()
    except ValueError as error:
        validation.errors.append(f"invalid manifest configuration: {error}")
        return validation

    map_paths = sorted(dataset_root.joinpath("maps").rglob("terrain_*.json"))
    validation.require(bool(map_paths), "no terrain JSON files were found")
    # Validate one full payload at a time.  Retaining every decoded elevation,
    # analysis field, and grid requires tens of gigabytes for a 5k-map
    # dataset, even though the cross-difficulty checks need only a few scalar
    # values and compact task signatures.
    groups: dict[str, list[dict]] = {}
    expected_map_task_records = 0
    for index, path in enumerate(map_paths, start=1):
        payload = _load_json(path)
        _validate_map(path, payload, config, validation)
        expected_map_task_records += len(payload.get("tasks", []))
        groups.setdefault(payload["matched_group_id"], []).append(
            {
                "difficulty": payload["difficulty"],
                "base_seed": payload["base_seed"],
                "realized_seed": payload["realized_seed"],
                "task_signatures": tuple(
                    _task_signature(task) for task in payload["tasks"]
                ),
                "relief_severity": float(
                    payload["generation"]["relief_severity"]
                ),
                "navigable_fraction": float(
                    payload["metrics"]["navigable_fraction"]
                ),
            }
        )
        if index == len(map_paths) or index % 250 == 0:
            print(f"Validated maps {index}/{len(map_paths)}", flush=True)

    for group_id, members in sorted(groups.items()):
        members.sort(key=lambda item: DIFFICULTY_ORDER.index(item["difficulty"]))
        difficulties = [item["difficulty"] for item in members]
        expected = list(manifest["difficulties"])
        validation.require(
            difficulties == expected,
            f"{group_id}: difficulty members are {difficulties}, expected {expected}",
        )
        validation.require(
            len({item["base_seed"] for item in members}) == 1,
            f"{group_id}: members do not share a base seed",
        )
        validation.require(
            len({item["realized_seed"] for item in members}) == 1,
            f"{group_id}: members do not share a realized seed",
        )
        task_signatures = [item["task_signatures"] for item in members]
        validation.require(
            all(signature == task_signatures[0] for signature in task_signatures[1:]),
            f"{group_id}: task coordinates/classes differ across difficulties",
        )
        severities = [item["relief_severity"] for item in members]
        navigable = [item["navigable_fraction"] for item in members]
        validation.require(
            all(
                severities[index] < severities[index + 1]
                for index in range(len(severities) - 1)
            ),
            f"{group_id}: relief severity is not strictly increasing",
        )
        validation.require(
            all(
                navigable[index] > navigable[index + 1]
                for index in range(len(navigable) - 1)
            ),
            f"{group_id}: navigable fraction is not strictly decreasing",
        )
    validation.require(
        len(map_paths) == int(manifest["total_maps"]),
        "manifest total_maps does not match files on disk",
    )
    validation.require(
        len(groups) == int(manifest["total_matched_groups"]),
        "manifest total_matched_groups does not match files on disk",
    )
    validation.require(
        expected_map_task_records == int(manifest["total_map_task_records"]),
        "manifest total_map_task_records is inconsistent",
    )

    if write_report:
        report = {
            "valid": not validation.errors,
            "maps_checked": validation.maps_checked,
            "map_task_records_checked": validation.tasks_checked,
            "matched_groups_checked": len(groups),
            "error_count": len(validation.errors),
            "warning_count": len(validation.warnings),
            "errors": validation.errors,
            "warnings": validation.warnings,
        }
        report_path = dataset_root / "metadata" / "validation_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return validation


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "dataset_preview",
    )
    parser.add_argument("--no-report", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation = validate_dataset(
        args.dataset.resolve(),
        write_report=not args.no_report,
    )
    if validation.errors:
        print(
            f"FAILED: {len(validation.errors)} errors after checking "
            f"{validation.maps_checked} maps"
        )
        for error in validation.errors:
            print(f"- {error}")
        return 1
    print(
        f"VALID: {validation.maps_checked} maps and "
        f"{validation.tasks_checked} map-task records checked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
