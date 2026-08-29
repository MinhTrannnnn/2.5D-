#!/usr/bin/env python3
"""Render side-by-side showcase palette trials without changing the default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from benchmark_pathplanning import path_length, run_astar
from generate_and_benchmark import save_showcase_visualization


PALETTE_LABELS = {
    "natural_current": "A — Current natural (baseline)",
    "natural_accessible": "B — Natural accessible",
    "scientific_colorblind": "C — Scientific colorblind",
    "neutral_contrast": "D — Neutral high-contrast",
}


def render_palette_trials(dataset_root: Path, terrain_id: str) -> Path:
    matches = list(
        dataset_root.joinpath("maps").rglob(f"{terrain_id}.json")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {terrain_id}.json below "
            f"{dataset_root / 'maps'}, found {len(matches)}"
        )
    with matches[0].open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    navigation = payload["navigation"]
    navigation_elevation = payload.get("terrain_analysis", {}).get(
        "support_elevation", payload["elevation"]
    )
    path, _, _ = run_astar(
        payload["grid"],
        payload["start"],
        payload["goal"],
        navigation_elevation,
        max_slope=navigation["max_slope"],
        cell_size=navigation["cell_size"],
        slope_weight=navigation.get(
            "path_slope_weight", navigation.get("slope_weight", 3.0)
        ),
    )
    if not path:
        raise RuntimeError(f"A* could not solve canonical task for {terrain_id}")
    length = path_length(path, navigation_elevation, navigation["cell_size"])

    output_root = dataset_root / "images" / "color_trials"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for palette_name, label in PALETTE_LABELS.items():
        output_path = output_root / f"{terrain_id}_{palette_name}.png"
        save_showcase_visualization(
            payload,
            output_path,
            path=path,
            route_label=f"A* path ({length:.2f} m)",
            palette_name=palette_name,
        )
        outputs.append((palette_name, label, output_path))
        print(f"rendered {label}: {output_path}")

    sheet_path = output_root / f"{terrain_id}_palette_comparison.png"
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(20, 14),
        dpi=100,
        squeeze=False,
        facecolor="#f3f4f2",
    )
    for axis, (_, label, image_path) in zip(axes.ravel(), outputs):
        axis.imshow(plt.imread(image_path))
        axis.set_axis_off()
        axis.set_title(label, fontsize=17, fontweight="semibold", pad=8)
    figure.suptitle(
        "2.5D dataset figure palette trials — identical map and A* path",
        fontsize=22,
        fontweight="semibold",
        y=0.988,
    )
    figure.subplots_adjust(
        left=0.006,
        right=0.994,
        bottom=0.006,
        top=0.920,
        wspace=0.012,
        hspace=0.065,
    )
    figure.savefig(sheet_path, facecolor=figure.get_facecolor(), pad_inches=0.02)
    plt.close(figure)
    return sheet_path


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "dataset_preview",
    )
    parser.add_argument(
        "--terrain-id",
        default="terrain_mountain_001_hard",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sheet_path = render_palette_trials(
        args.dataset.resolve(), args.terrain_id
    )
    print(f"comparison sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
