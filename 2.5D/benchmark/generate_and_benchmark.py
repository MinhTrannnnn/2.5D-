#!/usr/bin/env python3
"""Visualization helpers and compatibility entry point for paired 2.5D data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
    BoundaryNorm,
    LightSource,
    LinearSegmentedColormap,
    ListedColormap,
    to_rgb,
    to_rgba,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d


SHOWCASE_PALETTES = {
    "natural_current": {
        "terrain": (
            "#315f57",
            "#5f8064",
            "#91a36f",
            "#c7b982",
            "#ad835e",
            "#765744",
            "#ddd7c8",
        ),
        "blocked_tint": "#b3331f",
        "blocked_alpha": 0.48,
        "blocked_cell": "#b94b38",
        "blocked_edge": "#7f1d1d",
        "free_cell": "#dcebd7",
        "path": "#8b1cf6",
        "start": "#079455",
        "goal": "#dc2626",
        "wire": "#203b42",
        "background": "#eef2f2",
    },
    "natural_accessible": {
        "terrain": (
            "#365f5b",
            "#68847a",
            "#9aa68a",
            "#c9bc91",
            "#a68168",
            "#725d55",
            "#dedbd2",
        ),
        "blocked_tint": "#3f3f46",
        "blocked_alpha": 0.66,
        "blocked_cell": "#3f3f46",
        "blocked_edge": "#18181b",
        "free_cell": "#e8efe9",
        "path": "#00a6d6",
        "start": "#0072b2",
        "goal": "#e69f00",
        "wire": "#263f47",
        "background": "#f2f4f3",
    },
    "scientific_colorblind": {
        "terrain": "cividis",
        "blocked_tint": "#d55e00",
        "blocked_alpha": 0.70,
        "blocked_cell": "#d55e00",
        "blocked_edge": "#7c2d12",
        "free_cell": "#f2f2ed",
        "path": "#cc79a7",
        "start": "#0072b2",
        "goal": "#e69f00",
        "wire": "#263746",
        "background": "#f5f6f3",
    },
    "neutral_contrast": {
        "terrain": (
            "#f2f1ed",
            "#d9d8d2",
            "#bebdb7",
            "#9f9e99",
            "#7d7c79",
            "#5f6061",
        ),
        "blocked_tint": "#b2182b",
        "blocked_alpha": 0.76,
        "blocked_cell": "#b2182b",
        "blocked_edge": "#67000d",
        "free_cell": "#f7f7f5",
        "path": "#2166ac",
        "start": "#009e73",
        "goal": "#e69f00",
        "wire": "#30343b",
        "background": "#f7f7f5",
    },
}


def save_showcase_visualization(
    payload: dict,
    output_path: Path,
    path: Optional[Sequence[Sequence[int]]] = None,
    route_label: Optional[str] = None,
    palette_name: str = "natural_accessible",
) -> None:
    """Render a publication-style 2.5D terrain, optionally with a route."""

    if palette_name not in SHOWCASE_PALETTES:
        choices = ", ".join(sorted(SHOWCASE_PALETTES))
        raise ValueError(f"unknown showcase palette {palette_name!r}; use {choices}")
    palette = SHOWCASE_PALETTES[palette_name]

    elevation = np.asarray(payload["elevation"], dtype=float)
    grid = np.asarray(payload["grid"], dtype=np.uint8)
    cell_size = float(payload["navigation"]["cell_size"])
    size = elevation.shape[0]
    coordinates = np.arange(size, dtype=float) * cell_size
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    extent = float(coordinates[-1])

    terrain_spec = palette["terrain"]
    terrain_cmap = (
        plt.get_cmap(terrain_spec)
        if isinstance(terrain_spec, str)
        else LinearSegmentedColormap.from_list(
            f"terrain_{palette_name}", terrain_spec
        )
    )
    light = LightSource(azdeg=320, altdeg=48)
    facecolors = light.shade(
        elevation,
        cmap=terrain_cmap,
        vert_exag=1.25,
        blend_mode="soft",
    )
    # The collision grid is the actual planner constraint. Keep the terrain
    # geometry visible, but tint forbidden cells so a paper reader can see why
    # a route avoids them without consulting a separate occupancy image.
    blocked_mask = grid != 0
    blocked_tint = np.asarray(to_rgb(palette["blocked_tint"]), dtype=float)
    blocked_alpha = float(palette["blocked_alpha"])
    facecolors[blocked_mask, :3] = (
        (1.0 - blocked_alpha) * facecolors[blocked_mask, :3]
        + blocked_alpha * blocked_tint
    )
    facecolors[blocked_mask, 3] = 1.0

    background = palette["background"]
    figure = plt.figure(figsize=(13, 9), dpi=120, facecolor=background)
    # Use nearly the full canvas for the terrain.  The previous 90%-wide axis
    # left a large decorative margin that made the actual map unnecessarily
    # small in exported figures.
    axis = figure.add_axes((0.010, 0.080, 0.975, 0.825), projection="3d")
    axis.set_facecolor(background)
    axis.plot_surface(
        x_grid,
        y_grid,
        elevation,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    # Show the planning discretization without drawing all 16,384 cell edges.
    # Every wire interval represents eight cells (2 m at the default scale).
    axis.plot_wireframe(
        x_grid,
        y_grid,
        elevation + 0.006,
        rstride=8,
        cstride=8,
        color=palette["wire"],
        linewidth=0.55,
        alpha=0.24,
    )

    path_array = None
    if path:
        path_array = np.asarray(path, dtype=int)
        path_x = path_array[:, 0]
        path_y = path_array[:, 1]
        path_lift = max(0.055, float(np.ptp(elevation)) * 0.010)
        path_z = elevation[path_y, path_x] + path_lift
        metric_x = path_x.astype(float) * cell_size
        metric_y = path_y.astype(float) * cell_size
        # A white halo keeps the route legible over both dark and light terrain.
        axis.plot(
            metric_x,
            metric_y,
            path_z,
            color="#ffffff",
            linewidth=7.0,
            alpha=0.96,
            solid_capstyle="round",
            zorder=70,
        )
        axis.plot(
            metric_x,
            metric_y,
            path_z + 0.002,
            color=palette["path"],
            linewidth=4.0,
            alpha=1.0,
            solid_capstyle="round",
            zorder=71,
        )

    marker_lift = max(0.65, float(np.ptp(elevation)) * 0.075)
    projected_markers = []
    for point, color, label in (
        (payload["start"], palette["start"], "S"),
        (payload["goal"], palette["goal"], "G"),
    ):
        x, y = point
        surface_z = float(elevation[y, x])
        marker_z = surface_z + marker_lift
        axis.plot(
            [x * cell_size, x * cell_size],
            [y * cell_size, y * cell_size],
            [surface_z + 0.03, marker_z],
            color="#f8fafc",
            linewidth=4.0,
            zorder=40,
        )
        axis.plot(
            [x * cell_size, x * cell_size],
            [y * cell_size, y * cell_size],
            [surface_z + 0.03, marker_z],
            color="#1f2937",
            linewidth=1.5,
            zorder=41,
        )
        projected_markers.append(
            (x * cell_size, y * cell_size, marker_z, color, label)
        )

    horizontal_margin = max(0.6, extent * 0.025)
    z_min = float(elevation.min())
    z_max = float(elevation.max())
    z_margin = max(0.14, float(np.ptp(elevation)) * 0.035)
    axis.set_xlim(-horizontal_margin, extent + horizontal_margin)
    axis.set_ylim(-horizontal_margin, extent + horizontal_margin)
    axis.set_zlim(z_min - z_margin, z_max + marker_lift + 0.25)
    axis.set_box_aspect((1.0, 1.0, 0.36), zoom=1.17)
    axis.view_init(elev=38, azim=-132)
    axis.set_proj_type("ortho")
    ticks = np.arange(0.0, extent + 0.1, 5.0)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.zaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.set_xlabel("X (m)", labelpad=12, fontsize=11, fontweight="semibold")
    axis.set_ylabel("Y (m)", labelpad=12, fontsize=11, fontweight="semibold")
    axis.set_zlabel(
        "Z elevation (m)", labelpad=10, fontsize=11, fontweight="semibold"
    )
    axis.tick_params(axis="both", which="major", labelsize=9, pad=2)
    axis.grid(True)
    for dimension in (axis.xaxis, axis.yaxis, axis.zaxis):
        dimension.pane.fill = False
        dimension.pane.set_edgecolor((0.31, 0.39, 0.41, 0.52))
        dimension._axinfo["grid"].update(
            {"color": (0.31, 0.39, 0.41, 0.22), "linewidth": 0.7}
        )

    figure.text(
        0.5,
        0.965,
        "2.5D NAVIGATION TERRAIN",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#3e6268",
    )
    title_family = payload.get("terrain_family", payload["terrain_profile"])
    title_text = title_family.replace("_", " ").title()
    if payload.get("difficulty"):
        title_text += f" — {payload['difficulty'].title()}"
    figure.text(
        0.5,
        0.928,
        title_text,
        ha="center",
        va="center",
        fontsize=21,
        fontweight="semibold",
        color="#172b32",
    )
    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=palette["start"],
            markeredgecolor="white", markeredgewidth=1.5, markersize=11,
            label="Start",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=palette["goal"],
            markeredgecolor="white", markeredgewidth=1.5, markersize=11,
            label="Goal",
        ),
        Line2D(
            [0], [0], color=palette["wire"], linewidth=1.2, alpha=0.55,
            label="8-cell major grid",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none",
            markerfacecolor=palette["blocked_cell"],
            markeredgecolor=palette["blocked_edge"], markersize=9,
            label="Non-traversable",
        ),
    ]
    if path:
        legend_handles.append(
            Line2D(
                [0], [0], color=palette["path"], linewidth=4.0,
                label=route_label or "Planned path",
            )
        )
    # Anchor the legend to the canvas rather than the projected 3D axis.  This
    # keeps it in the true upper-left corner and leaves a stable visual gap
    # between the key and the terrain for every camera angle/profile.
    figure.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.014, 0.974),
        borderaxespad=0.0,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cad5d7",
        framealpha=0.94,
        fontsize=10,
    )

    if path_array is not None:
        # The 3D surface is made of faces between elevation samples, whereas
        # collision checking is defined at discrete robot-centre cells.  A
        # route close to a tinted face can consequently look as though it
        # crosses an obstacle in perspective.  This inset renders the exact
        # grid used by the planner, without interpolation or projection.
        # Keep this diagnostic in the true upper-right margin, symmetric with
        # the figure-level legend on the left and clear of the terrain body.
        inset = figure.add_axes((0.835, 0.738, 0.150, 0.220), zorder=200)
        occupancy_cmap = ListedColormap(
            (palette["free_cell"], palette["blocked_cell"])
        )
        inset.imshow(
            blocked_mask.astype(np.uint8),
            origin="lower",
            interpolation="nearest",
            cmap=occupancy_cmap,
            vmin=0,
            vmax=1,
            extent=(-0.5, size - 0.5, -0.5, size - 0.5),
        )
        path_x = path_array[:, 0]
        path_y = path_array[:, 1]
        inset.plot(
            path_x,
            path_y,
            color="white",
            linewidth=3.8,
            solid_capstyle="round",
            zorder=3,
        )
        inset.plot(
            path_x,
            path_y,
            color=palette["path"],
            linewidth=2.1,
            solid_capstyle="round",
            zorder=4,
        )
        start_x, start_y = payload["start"]
        goal_x, goal_y = payload["goal"]
        inset.scatter(
            [start_x, goal_x],
            [start_y, goal_y],
            s=32,
            c=[palette["start"], palette["goal"]],
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        inset.set_xlim(-0.5, size - 0.5)
        inset.set_ylim(-0.5, size - 0.5)
        inset.set_aspect("equal")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title(
            "Exact planner cell grid",
            fontsize=9,
            fontweight="semibold",
            color="#172b32",
            pad=4,
        )
        blocked_path_cells = int(
            np.count_nonzero(blocked_mask[path_y, path_x])
        )
        check_color = "#067647" if blocked_path_cells == 0 else "#b42318"
        inset.text(
            0.5,
            0.025,
            f"Path on blocked cells: {blocked_path_cells}",
            transform=inset.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=check_color,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": check_color,
                "linewidth": 0.9,
                "alpha": 0.94,
            },
            zorder=6,
        )
        for spine in inset.spines.values():
            spine.set_color("#365159")
            spine.set_linewidth(0.9)

    figure.canvas.draw()
    for marker_x, marker_y, marker_z, color, label in projected_markers:
        projected_x, projected_y, _ = proj3d.proj_transform(
            marker_x, marker_y, marker_z, axis.get_proj()
        )
        display_point = axis.transData.transform((projected_x, projected_y))
        figure_x, figure_y = figure.transFigure.inverted().transform(display_point)
        figure.text(
            figure_x,
            figure_y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
            zorder=1000,
            bbox={
                "boxstyle": "circle,pad=0.38",
                "facecolor": color,
                "edgecolor": "white",
                "linewidth": 2.2,
            },
        )

    blocked_fraction = 1.0 - float(np.mean(grid == 0))
    figure.text(
        0.5,
        0.025,
        (
            f"{size} x {size} cells   |   {cell_size:.2f} m/cell   |   "
            f"Elevation {z_min:.2f}-{z_max:.2f} m   |   "
            f"Non-traversable {blocked_fraction:.1%}   |   8-connected grid"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#365159",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor=figure.get_facecolor(), pad_inches=0.02)
    plt.close(figure)


def save_traversability_visualization(payload: dict, output_path: Path) -> None:
    elevation = np.asarray(payload["elevation"], dtype=float)
    analysis = payload["terrain_analysis"]
    cost = np.asarray(analysis["traversability_cost"], dtype=float)
    grid = np.asarray(payload["grid"], dtype=np.uint8)
    cell_size = float(payload["navigation"]["cell_size"])
    extent_m = (elevation.shape[0] - 1) * cell_size
    extent = (0, extent_m, extent_m, 0)

    figure, axes = plt.subplots(1, 3, figsize=(20, 6.2), dpi=120)
    elevation_image = axes[0].imshow(
        elevation,
        cmap="cividis",
        origin="upper",
        extent=extent,
    )
    axes[0].contour(
        np.linspace(0, extent_m, elevation.shape[1]),
        np.linspace(0, extent_m, elevation.shape[0]),
        elevation,
        levels=14,
        colors="black",
        linewidths=0.35,
        alpha=0.38,
    )
    axes[0].set_title("Elevation and contours")
    figure.colorbar(elevation_image, ax=axes[0], shrink=0.82, label="elevation (m)")

    cost_image = axes[1].imshow(
        np.clip(cost, 0.0, 1.5),
        cmap="viridis_r",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        extent=extent,
    )
    blocked_overlay = np.zeros((*grid.shape, 4), dtype=float)
    blocked_overlay[grid != 0] = to_rgba("#3f3f46", alpha=0.62)
    axes[1].imshow(blocked_overlay, origin="upper", extent=extent)
    axes[1].set_title("Traversability cost (overlay = non-traversable)")
    figure.colorbar(cost_image, ax=axes[1], shrink=0.82, label="terrain cost")

    blocked_by_slope = np.asarray(analysis["blocked_by_slope"], dtype=bool)
    blocked_by_roughness = np.asarray(
        analysis["blocked_by_roughness"], dtype=bool
    )
    blocked_by_step = np.asarray(analysis["blocked_by_step"], dtype=bool)
    cause_count = (
        blocked_by_slope.astype(np.uint8)
        + blocked_by_roughness.astype(np.uint8)
        + blocked_by_step.astype(np.uint8)
    )
    causes = np.zeros(grid.shape, dtype=np.uint8)
    causes[blocked_by_slope] = 1
    causes[blocked_by_roughness] = 2
    causes[blocked_by_step] = 3
    causes[cause_count > 1] = 4
    footprint_cells = int(payload["generation"].get("footprint_cells", 1))
    border = np.zeros(grid.shape, dtype=bool)
    border[:footprint_cells, :] = True
    border[-footprint_cells:, :] = True
    border[:, :footprint_cells] = True
    border[:, -footprint_cells:] = True
    causes[border] = 5
    cause_colors = (
        "#e8efe9",
        "#e69f00",
        "#cc79a7",
        "#0072b2",
        "#3f3f46",
        "#8f8f8f",
    )
    cause_labels = (
        "Traversable",
        "Support slope",
        "Roughness",
        "Step height",
        "Multiple constraints",
        "Footprint boundary",
    )
    axes[2].imshow(
        causes,
        cmap=ListedColormap(cause_colors),
        norm=BoundaryNorm(np.arange(-0.5, 6.5, 1.0), 6),
        origin="upper",
        extent=extent,
        interpolation="nearest",
    )
    axes[2].set_title("Why each robot-centre pose is blocked")
    axes[2].legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=label)
            for color, label in zip(cause_colors, cause_labels)
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.94,
    )

    for axis in axes:
        for point, color, marker, label in (
            (payload["start"], "#0072b2", "o", "Start"),
            (payload["goal"], "#e69f00", "*", "Goal"),
        ):
            x, y = point
            axis.scatter(
                x * cell_size,
                y * cell_size,
                c=color,
                marker=marker,
                s=75 if marker == "o" else 120,
                edgecolors="white",
                linewidths=1.0,
                label=label,
            )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal")
    axes[1].legend(loc="lower right", framealpha=0.9)
    figure.suptitle(
        f"{payload['instance_id']} - terrain analysis",
        fontsize=15,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    from generate_25d_dataset import build_parser as build_paired_parser

    return build_paired_parser()


def main(argv: Optional[Sequence[str]] = None) -> int:
    from generate_25d_dataset import main as generate_paired_dataset

    return generate_paired_dataset(argv)


if __name__ == "__main__":
    raise SystemExit(main())
