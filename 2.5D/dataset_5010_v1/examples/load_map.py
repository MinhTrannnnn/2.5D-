#!/usr/bin/env python3
"""Load one released 2.5D map and print its canonical planning task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_map(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    size = int(payload["size"])
    layer_names = (
        "elevation",
        "grid",
    )
    for name in layer_names:
        layer = payload[name]
        if len(layer) != size or any(len(row) != size for row in layer):
            raise ValueError(f"{name} is not a {size} x {size} layer")

    analysis = payload["terrain_analysis"]
    for name in (
        "support_elevation",
        "footprint_slope_degrees",
        "roughness",
        "step_height",
        "traversability_cost",
    ):
        layer = analysis[name]
        if len(layer) != size or any(len(row) != size for row in layer):
            raise ValueError(f"terrain_analysis.{name} has an invalid shape")
    return payload


def main() -> int:
    dataset_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "map",
        nargs="?",
        type=Path,
        default=(
            dataset_root
            / "maps"
            / "mountain"
            / "terrain_mountain_001_easy.json"
        ),
    )
    args = parser.parse_args()
    payload = load_map(args.map.resolve())

    canonical = next(
        task for task in payload["tasks"] if task["canonical_visualization"]
    )
    cell_size = float(payload["navigation"]["cell_size"])
    coordinates = {}
    for label in ("start", "goal"):
        x, y = canonical[label]
        if payload["grid"][y][x] != 0:
            raise ValueError(f"canonical {label} is blocked")
        coordinates[label] = (
            x * cell_size,
            y * cell_size,
            payload["elevation"][y][x],
        )

    print(f"map: {payload['instance_id']}")
    print(f"family/difficulty: {payload['terrain_family']}/{payload['difficulty']}")
    print(f"canonical task: {canonical['task_id']}")
    print(f"start xyz (m): {coordinates['start']}")
    print(f"goal xyz (m): {coordinates['goal']}")
    print(f"collision grid: {payload['size']} x {payload['size']} (0 free, 1 blocked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
