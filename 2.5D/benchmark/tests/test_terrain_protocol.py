"""Regression tests for the paired 2.5D terrain protocol."""

from __future__ import annotations

import math
import unittest

import numpy as np

from terrain_generation import (
    TerrainConfig,
    analyze_terrain,
    generate_elevation,
    sample_start_goal_pairs,
)


class TerrainProtocolTests(unittest.TestCase):
    def test_release_terrain_families_are_deterministic(self) -> None:
        config = TerrainConfig(
            size=48,
            endpoint_margin=0.5,
            min_navigable_fraction=0.10,
            max_navigable_fraction=0.99,
            min_largest_component_fraction=0.10,
        )
        for family in (
            "smooth_obstacles",
            "rolling",
            "mountain",
            "rugged",
            "plateau",
        ):
            with self.subTest(family=family):
                first = generate_elevation(config, seed=12345, profile=family)
                second = generate_elevation(config, seed=12345, profile=family)
                self.assertEqual(first.shape, (config.size, config.size))
                self.assertTrue(np.all(np.isfinite(first)))
                self.assertGreaterEqual(float(first.min()), 0.0)
                self.assertTrue(np.array_equal(first, second))

    def test_unknown_terrain_family_is_rejected(self) -> None:
        config = TerrainConfig(
            size=48,
            endpoint_margin=0.5,
            min_navigable_fraction=0.10,
            max_navigable_fraction=0.99,
            min_largest_component_fraction=0.10,
        )
        with self.assertRaisesRegex(ValueError, "unknown terrain profile"):
            generate_elevation(config, seed=12345, profile="unsupported")

    def test_traversability_weights_must_sum_to_one(self) -> None:
        config = TerrainConfig(
            traversability_slope_weight=0.5,
            traversability_roughness_weight=0.5,
            traversability_step_weight=0.5,
        )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            config.validate()

    def test_flat_plane_has_zero_geometric_penalties(self) -> None:
        config = TerrainConfig(
            size=48,
            endpoint_margin=0.5,
            min_navigable_fraction=0.10,
            max_navigable_fraction=0.99,
            min_largest_component_fraction=0.10,
        )
        elevation = np.zeros((config.size, config.size), dtype=float)
        layers = analyze_terrain(elevation, config)
        for key in (
            "slope_degrees",
            "footprint_slope_degrees",
            "roughness",
            "step_height",
            "traversability_cost",
        ):
            self.assertTrue(np.allclose(layers[key], 0.0), key)

    def test_collision_grid_has_no_second_footprint_dilation(self) -> None:
        config = TerrainConfig(
            size=48,
            endpoint_margin=0.5,
            min_navigable_fraction=0.10,
            max_navigable_fraction=0.99,
            min_largest_component_fraction=0.10,
        )
        elevation = np.zeros((config.size, config.size), dtype=float)
        elevation[24, 24] = 1.0
        layers = analyze_terrain(elevation, config)
        border = config.footprint_cells
        interior_collision = layers["collision_grid"][
            border:-border,
            border:-border,
        ]
        interior_centres = layers["centre_blocked"][
            border:-border,
            border:-border,
        ]
        self.assertTrue(np.array_equal(interior_collision, interior_centres))

    def test_support_plane_recovers_grade_without_creating_roughness(self) -> None:
        config = TerrainConfig(
            size=48,
            endpoint_margin=0.5,
            min_navigable_fraction=0.10,
            max_navigable_fraction=0.99,
            min_largest_component_fraction=0.10,
        )
        target_degrees = 12.0
        x = np.arange(config.size, dtype=float) * config.cell_size
        elevation = np.repeat(
            (x * math.tan(math.radians(target_degrees)))[None, :],
            config.size,
            axis=0,
        )
        layers = analyze_terrain(elevation, config)
        interior = slice(4, -4)
        support_slope = layers["footprint_slope_degrees"][interior, interior]
        roughness = layers["roughness"][interior, interior]
        self.assertTrue(np.allclose(support_slope, target_degrees, atol=1e-6))
        self.assertTrue(np.allclose(roughness, 0.0, atol=1e-7))

    def test_task_sampling_is_deterministic_and_stratified(self) -> None:
        config = TerrainConfig(
            size=40,
            endpoint_margin=0.5,
            min_navigable_fraction=0.10,
            max_navigable_fraction=0.99,
            min_largest_component_fraction=0.10,
        )
        elevation = np.zeros((config.size, config.size), dtype=float)
        grid = analyze_terrain(elevation, config)["collision_grid"].astype(np.uint8)
        first, _ = sample_start_goal_pairs(
            grid,
            elevation,
            config,
            count=10,
            seed=42,
        )
        second, _ = sample_start_goal_pairs(
            grid,
            elevation,
            config,
            count=10,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            [task["distance_class"] for task in first],
            ["short"] * 3 + ["medium"] * 4 + ["long"] * 3,
        )
        self.assertEqual(
            sum(task["canonical_visualization"] for task in first),
            1,
        )
        pairs = {
            (tuple(task["start"]), tuple(task["goal"]))
            for task in first
        }
        self.assertEqual(len(pairs), 10)


if __name__ == "__main__":
    unittest.main()
