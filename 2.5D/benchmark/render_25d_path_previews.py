"""Render one A* route draped over each selected 2.5D terrain family."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from benchmark_pathplanning import (
    path_length,
    plot_algorithm_result_25d,
    run_astar,
    weighted_path_cost,
)


def _save_contact_sheet(
    dataset_root: Path,
    rows: list[dict],
    difficulty: str,
) -> Path:
    output_path = (
        dataset_root / "images" / "contact_sheets" / f"path_{difficulty}.png"
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 12),
        dpi=100,
        squeeze=False,
        facecolor="#eef2f2",
    )
    flat_axes = axes.ravel()
    for axis, row in zip(flat_axes, rows):
        axis.imshow(plt.imread(dataset_root / row["image_file"]))
        axis.set_axis_off()
        axis.set_title(
            row["family"].replace("_", " ").title(),
            fontsize=15,
            fontweight="semibold",
            pad=8,
        )
    for axis in flat_axes[len(rows):]:
        axis.set_axis_off()
    figure.suptitle(
        f"A* routes draped over 2.5D terrain - {difficulty.title()}",
        fontsize=21,
        fontweight="semibold",
        y=0.99,
    )
    figure.subplots_adjust(
        left=0.006,
        right=0.994,
        bottom=0.008,
        top=0.965,
        wspace=0.012,
        hspace=0.05,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor=figure.get_facecolor(), pad_inches=0.02)
    plt.close(figure)
    return output_path


def render_previews(dataset_root: Path, difficulty: str) -> None:
    map_paths = sorted(dataset_root.joinpath("maps").rglob("terrain_*.json"))
    selected = []
    for map_path in map_paths:
        with map_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            payload.get("difficulty") == difficulty
            and int(payload.get("instance_index", 1)) == 1
        ):
            selected.append((map_path, payload))

    if not selected:
        raise RuntimeError(
            f"no {difficulty!r} maps found below {dataset_root / 'maps'}"
        )

    rows = []
    for map_path, payload in selected:
        navigation = payload["navigation"]
        navigation_elevation = payload.get("terrain_analysis", {}).get(
            "support_elevation",
            payload["elevation"],
        )
        path, _, elapsed = run_astar(
            payload["grid"],
            payload["start"],
            payload["goal"],
            navigation_elevation,
            max_slope=navigation["max_slope"],
            cell_size=navigation["cell_size"],
            slope_weight=navigation.get(
                "path_slope_weight",
                navigation.get("slope_weight", 3.0),
            ),
        )
        length = path_length(path, navigation_elevation, navigation["cell_size"])
        common_cost = weighted_path_cost(
            path,
            navigation_elevation,
            navigation["cell_size"],
            navigation.get(
                "path_slope_weight",
                navigation.get("slope_weight", 3.0),
            ),
        )
        family = payload.get(
            "terrain_family", payload.get("terrain_profile", "other")
        )
        output_path = (
            dataset_root
            / "images"
            / "paths"
            / family
            / f"{payload['instance_id']}_A_Star.png"
        )
        plot_algorithm_result_25d(
            payload,
            path,
            f"A* path ({length:.2f} m)",
            output_path,
        )
        rows.append(
            {
                "terrain_id": payload["instance_id"],
                "family": family,
                "difficulty": payload["difficulty"],
                "success": bool(path),
                "path_length_25d": round(length, 4),
                "weighted_path_cost": round(common_cost, 4),
                "time_sec": round(elapsed, 6),
                "image_file": output_path.relative_to(dataset_root).as_posix(),
                "map_file": map_path.relative_to(dataset_root).as_posix(),
            }
        )
        print(
            payload["instance_id"],
            "success" if path else "failed",
            f"length={length:.2f} m",
        )

    summary_path = dataset_root / "metadata" / "path_preview_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    contact_sheet_path = _save_contact_sheet(dataset_root, rows, difficulty)

    manifest_path = dataset_root / "metadata" / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["path_planning_outputs_included"] = True
        manifest["path_preview"] = {
            "algorithm": "A_Star",
            "difficulty": difficulty,
            "count": len(rows),
            "directory": "images/paths/<family>",
            "contact_sheet": contact_sheet_path.relative_to(dataset_root).as_posix(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "dataset_preview",
    )
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium", "hard"),
        default="medium",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_previews(args.dataset.resolve(), args.difficulty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
