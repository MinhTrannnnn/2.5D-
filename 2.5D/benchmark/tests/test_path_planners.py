"""Regression tests for standardized 2.5D path planning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_pathplanning import (
    _line_cells,
    is_collision_free,
    is_path_valid,
    run_astar,
    run_benchmark,
    run_bfs,
    run_dijkstra,
    run_prm,
    run_rrt_star,
    weighted_path_cost,
)


class PathPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = [[0] * 24 for _ in range(24)]
        self.elevation = [[0.0] * 24 for _ in range(24)]

    def test_bfs_uses_the_common_eight_connected_action_space(self) -> None:
        path, _, _ = run_bfs(
            self.grid,
            (1, 1),
            (6, 6),
            self.elevation,
            max_slope=1.0,
            cell_size=1.0,
        )
        self.assertEqual(len(path), 6)
        self.assertTrue(
            is_path_valid(
                path,
                self.grid,
                self.elevation,
                (1, 1),
                (6, 6),
                1.0,
                1.0,
            )
        )

    def test_diagonal_rasterization_has_no_phantom_cardinal_cell(self) -> None:
        cases = (
            ((1, 2), (2, 1)),
            ((2, 1), (1, 2)),
            ((2, 2), (3, 3)),
            ((3, 3), (2, 2)),
        )
        for start, goal in cases:
            with self.subTest(start=start, goal=goal):
                self.assertEqual(_line_cells(start, goal), [start, goal])

    def test_line_rasterization_is_symmetric_and_eight_connected(self) -> None:
        cases = (
            ((1, 1), (18, 7)),
            ((2, 20), (19, 3)),
            ((12, 2), (12, 21)),
            ((20, 19), (3, 5)),
        )
        for start, goal in cases:
            with self.subTest(start=start, goal=goal):
                forward = _line_cells(start, goal)
                backward = _line_cells(goal, start)
                self.assertEqual(forward, list(reversed(backward)))
                self.assertTrue(
                    all(
                        max(abs(b[0] - a[0]), abs(b[1] - a[1])) == 1
                        for a, b in zip(forward, forward[1:])
                    )
                )

    def test_grid_search_and_validator_share_diagonal_slope_semantics(self) -> None:
        grid = [[0] * 4 for _ in range(4)]
        elevation = [[0.0] * 4 for _ in range(4)]
        start = (1, 2)
        goal = (2, 1)
        # This cardinal side cell is deliberately steep.  It is required to be
        # collision-free for the no-corner-cutting rule, but it is not part of
        # the direct diagonal cell-centre transition.
        elevation[2][2] = 10.0
        self.assertTrue(
            is_collision_free(
                grid,
                elevation,
                start,
                goal,
                max_slope=0.5,
                cell_size=1.0,
            )
        )
        for planner in (run_bfs, run_dijkstra, run_astar):
            with self.subTest(planner=planner.__name__):
                path, _, _ = planner(
                    grid,
                    start,
                    goal,
                    elevation,
                    max_slope=0.5,
                    cell_size=1.0,
                )
                self.assertEqual(path, [start, goal])
                self.assertTrue(
                    is_path_valid(
                        path,
                        grid,
                        elevation,
                        start,
                        goal,
                        0.5,
                        1.0,
                    )
                )
        grid[2][2] = 1
        self.assertFalse(
            is_collision_free(
                grid,
                elevation,
                start,
                goal,
                max_slope=0.5,
                cell_size=1.0,
            )
        )

    def test_single_cell_path_still_requires_a_free_endpoint(self) -> None:
        self.grid[4][4] = 1
        self.assertFalse(
            is_path_valid(
                [(4, 4)],
                self.grid,
                self.elevation,
                (4, 4),
                (4, 4),
                1.0,
                1.0,
            )
        )

    def test_astar_and_dijkstra_agree_on_common_weighted_cost(self) -> None:
        self.elevation[8][8] = 1.5
        arguments = (
            self.grid,
            (2, 2),
            (18, 18),
            self.elevation,
        )
        dijkstra_path, _, _ = run_dijkstra(
            *arguments,
            max_slope=10.0,
            cell_size=0.25,
            slope_weight=4.0,
        )
        astar_path, _, _ = run_astar(
            *arguments,
            max_slope=10.0,
            cell_size=0.25,
            slope_weight=4.0,
        )
        self.assertAlmostEqual(
            weighted_path_cost(
                dijkstra_path,
                self.elevation,
                0.25,
                4.0,
            ),
            weighted_path_cost(
                astar_path,
                self.elevation,
                0.25,
                4.0,
            ),
            places=9,
        )

    def test_prm_uses_slope_weight_in_roadmap_edge_costs(self) -> None:
        with patch(
            "benchmark_pathplanning.weighted_path_cost",
            wraps=weighted_path_cost,
        ) as evaluator:
            run_prm(
                self.grid,
                (1, 1),
                (22, 22),
                self.elevation,
                max_slope=1.0,
                cell_size=0.25,
                slope_weight=7.0,
                num_samples=80,
                k_neighbors=10,
                seed=4,
            )
        self.assertTrue(
            any(call.args[3] == 7.0 for call in evaluator.call_args_list)
        )

    def test_rrt_star_uses_cost_and_returns_a_valid_fixed_budget_path(self) -> None:
        with patch(
            "benchmark_pathplanning.weighted_path_cost",
            wraps=weighted_path_cost,
        ) as evaluator:
            path, _, _ = run_rrt_star(
                self.grid,
                (1, 1),
                (22, 22),
                self.elevation,
                max_slope=1.0,
                cell_size=0.25,
                slope_weight=9.0,
                max_iter=600,
                seed=7,
            )
        self.assertTrue(path)
        self.assertTrue(
            is_path_valid(
                path,
                self.grid,
                self.elevation,
                (1, 1),
                (22, 22),
                1.0,
                0.25,
            )
        )
        self.assertTrue(
            any(call.args[3] == 9.0 for call in evaluator.call_args_list)
        )

    def test_benchmark_consumes_every_task_without_retries(self) -> None:
        tasks = [
            {
                "task_index": 1,
                "task_id": "terrain_test_001_easy_sg01",
                "matched_task_id": "terrain_test_001_sg01",
                "start": [1, 1],
                "goal": [8, 8],
                "distance_class": "short",
                "canonical_visualization": False,
            },
            {
                "task_index": 2,
                "task_id": "terrain_test_001_easy_sg02",
                "matched_task_id": "terrain_test_001_sg02",
                "start": [1, 1],
                "goal": [22, 22],
                "distance_class": "long",
                "canonical_visualization": True,
            },
        ]
        payload = {
            "instance_id": "terrain_test_001_easy",
            "matched_group_id": "terrain_test_001",
            "instance_index": 1,
            "terrain_family": "test",
            "terrain_profile": "test",
            "difficulty": "easy",
            "seed": 10,
            "task_seed": 20,
            "grid": self.grid,
            # Raw terrain alternates between extreme heights. The benchmark
            # must use the published footprint support surface for navigation.
            "elevation": [
                [10.0 * ((x + y) % 2) for x in range(24)]
                for y in range(24)
            ],
            "terrain_analysis": {"support_elevation": self.elevation},
            "tasks": tasks,
            "start": tasks[1]["start"],
            "goal": tasks[1]["goal"],
            "navigation": {
                "max_slope": 1.0,
                "cell_size": 0.25,
                "path_slope_weight": 3.0,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            maps = root / "maps"
            maps.mkdir()
            (root / "metadata").mkdir()
            (maps / "terrain_test_001_easy.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            trials, summaries = run_benchmark(
                root,
                algorithm_names=("BFS",),
                output_prefix="test",
            )
            self.assertEqual(len(trials), 2)
            self.assertEqual(len(summaries), 1)
            self.assertTrue(all(row["success"] for row in trials))
            protocol = json.loads(
                (root / "metadata" / "test_protocol.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(protocol["hidden_retries"])
            resumed_trials, resumed_summaries = run_benchmark(
                root,
                algorithm_names=("BFS",),
                output_prefix="test",
                resume=True,
            )
            self.assertEqual(resumed_trials, trials)
            self.assertEqual(resumed_summaries, summaries)
            self.assertEqual(
                len(
                    (root / "metadata" / "test_progress.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                1,
            )
            state = json.loads(
                (root / "metadata" / "test_state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(state["status"], "complete")
            canonical_trials, _ = run_benchmark(
                root,
                algorithm_names=("BFS",),
                task_scope="canonical",
                output_prefix="canonical",
            )
            self.assertEqual(len(canonical_trials), 1)
            self.assertEqual(
                canonical_trials[0]["task_id"],
                "terrain_test_001_easy_sg02",
            )
            canonical_protocol = json.loads(
                (root / "metadata" / "canonical_protocol.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(canonical_protocol["task_scope"], "canonical")
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                run_benchmark(
                    root,
                    algorithm_names=("BFS",),
                    max_tasks_per_map=1,
                    output_prefix="test",
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
