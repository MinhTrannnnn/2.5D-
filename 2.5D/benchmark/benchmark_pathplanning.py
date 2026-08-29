"""Path-planning algorithms for full-frame 2.5D elevation terrains."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Sequence

from generate_and_benchmark import save_showcase_visualization
from terrain_generation import (
    movement_cost as _movement_cost,
    transition_cost as _transition_cost,
)


Cell = tuple[int, int]
Grid = list[list[int]]
HeightField = list[list[Optional[float]]]


def heuristic(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def path_length(
    path: Sequence[Cell],
    elevation: Optional[HeightField] = None,
    cell_size: float = 1.0,
) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for first, second in zip(path, path[1:]):
        planar = heuristic(first, second) * cell_size
        if elevation is None:
            total += planar
        else:
            z0 = elevation[first[1]][first[0]]
            z1 = elevation[second[1]][second[0]]
            assert z0 is not None and z1 is not None
            total += math.hypot(planar, z1 - z0)
    return total


def planar_path_length(
    path: Sequence[Cell],
    cell_size: float = 1.0,
) -> float:
    """Return horizontal path length in metres."""

    return sum(
        heuristic(first, second) * cell_size
        for first, second in zip(path, path[1:])
    )


def weighted_path_cost(
    path: Sequence[Cell],
    elevation: HeightField,
    cell_size: float = 1.0,
    slope_weight: float = 3.0,
) -> float:
    """Evaluate a path with the common 2.5D objective used by all planners."""

    total = 0.0
    for first, second in zip(path, path[1:]):
        edge_cost, _ = _movement_cost(
            first,
            second,
            elevation,
            cell_size,
            slope_weight,
        )
        total += edge_cost
    return total


def path_elevation_gain(
    path: Sequence[Cell],
    elevation: HeightField,
) -> float:
    """Return cumulative positive elevation change in metres."""

    total = 0.0
    for first, second in zip(path, path[1:]):
        z0 = elevation[first[1]][first[0]]
        z1 = elevation[second[1]][second[0]]
        assert z0 is not None and z1 is not None
        total += max(0.0, z1 - z0)
    return total


def _neighbors(
    grid: Grid,
    elevation: HeightField,
    current: Cell,
    max_slope: float,
    cell_size: float,
    slope_weight: float,
) -> Iterable[tuple[Cell, float]]:
    directions = (
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )
    x, y = current
    for dx, dy in directions:
        neighbor = (x + dx, y + dy)
        cost = _transition_cost(
            grid,
            elevation,
            current,
            neighbor,
            max_slope,
            cell_size,
            slope_weight,
        )
        if cost is not None:
            yield neighbor, cost


def _reconstruct(came_from: dict[Cell, Cell], start: Cell, goal: Cell) -> list[Cell]:
    if goal != start and goal not in came_from:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def run_bfs(
    grid: Grid,
    start: Sequence[int],
    goal: Sequence[int],
    elevation: HeightField,
    max_slope: float = 1.25,
    cell_size: float = 1.0,
    slope_weight: float = 3.0,
) -> tuple[list[Cell], list[tuple[Cell, Cell]], float]:
    started = time.perf_counter()
    start_cell = tuple(start)
    goal_cell = tuple(goal)
    queue = deque([start_cell])
    came_from: dict[Cell, Optional[Cell]] = {start_cell: None}
    explored_edges = []
    while queue:
        current = queue.popleft()
        if current == goal_cell:
            break
        for neighbor, _ in _neighbors(
            grid,
            elevation,
            current,
            max_slope,
            cell_size,
            slope_weight,
        ):
            if neighbor not in came_from:
                came_from[neighbor] = current
                explored_edges.append((current, neighbor))
                queue.append(neighbor)
    parents = {cell: parent for cell, parent in came_from.items() if parent is not None}
    path = _reconstruct(parents, start_cell, goal_cell)
    return path, explored_edges, time.perf_counter() - started


def _run_weighted_grid_search(
    grid: Grid,
    start: Sequence[int],
    goal: Sequence[int],
    elevation: HeightField,
    max_slope: float,
    cell_size: float,
    slope_weight: float,
    use_heuristic: bool,
) -> tuple[list[Cell], list[tuple[Cell, Cell]], float]:
    started = time.perf_counter()
    start_cell = tuple(start)
    goal_cell = tuple(goal)
    queue: list[tuple[float, float, Cell]] = [(0.0, 0.0, start_cell)]
    scores = {start_cell: 0.0}
    came_from: dict[Cell, Cell] = {}
    explored_edges = []
    while queue:
        _, queued_score, current = heapq.heappop(queue)
        if queued_score != scores.get(current):
            continue
        if current == goal_cell:
            break
        current_score = scores[current]
        for neighbor, edge_cost in _neighbors(
            grid,
            elevation,
            current,
            max_slope,
            cell_size,
            slope_weight,
        ):
            candidate = current_score + edge_cost
            if candidate < scores.get(neighbor, math.inf):
                scores[neighbor] = candidate
                came_from[neighbor] = current
                priority = candidate
                if use_heuristic:
                    priority += heuristic(neighbor, goal_cell) * cell_size
                heapq.heappush(queue, (priority, candidate, neighbor))
                explored_edges.append((current, neighbor))
    path = _reconstruct(came_from, start_cell, goal_cell)
    return path, explored_edges, time.perf_counter() - started


def run_astar(
    grid: Grid,
    start: Sequence[int],
    goal: Sequence[int],
    elevation: HeightField,
    max_slope: float = 1.25,
    cell_size: float = 1.0,
    slope_weight: float = 3.0,
) -> tuple[list[Cell], list[tuple[Cell, Cell]], float]:
    return _run_weighted_grid_search(
        grid, start, goal, elevation, max_slope, cell_size, slope_weight, True
    )


def run_dijkstra(
    grid: Grid,
    start: Sequence[int],
    goal: Sequence[int],
    elevation: HeightField,
    max_slope: float = 1.25,
    cell_size: float = 1.0,
    slope_weight: float = 3.0,
) -> tuple[list[Cell], list[tuple[Cell, Cell]], float]:
    return _run_weighted_grid_search(
        grid, start, goal, elevation, max_slope, cell_size, slope_weight, False
    )


def _line_cells(first: Cell, second: Cell) -> list[Cell]:
    """Return a symmetric, 8-connected Bresenham rasterization of an edge.

    Canonicalizing the endpoint order makes an undirected edge traverse the
    same cells in either direction.  In particular, one diagonal grid move
    remains one diagonal move; it is never changed into two cardinal moves by
    floating-point rounding.
    """

    reverse = tuple(first) > tuple(second)
    start, goal = (second, first) if reverse else (first, second)
    x, y = start
    goal_x, goal_y = goal
    delta_x = abs(goal_x - x)
    step_x = 1 if x < goal_x else -1
    delta_y = -abs(goal_y - y)
    step_y = 1 if y < goal_y else -1
    error = delta_x + delta_y
    cells: list[Cell] = []
    while True:
        cells.append((x, y))
        if x == goal_x and y == goal_y:
            break
        doubled_error = 2 * error
        if doubled_error >= delta_y:
            error += delta_y
            x += step_x
        if doubled_error <= delta_x:
            error += delta_x
            y += step_y
    return list(reversed(cells)) if reverse else cells


def _expand_waypoints(waypoints: Sequence[Cell]) -> list[Cell]:
    """Rasterize sparse sampling-planner waypoints into adjacent grid cells."""

    if not waypoints:
        return []
    expanded = [waypoints[0]]
    for first, second in zip(waypoints, waypoints[1:]):
        expanded.extend(_line_cells(first, second)[1:])
    return expanded


def is_collision_free(
    grid: Grid,
    elevation: HeightField,
    first: Cell,
    second: Cell,
    max_slope: float = 1.25,
    cell_size: float = 1.0,
) -> bool:
    cells = _line_cells(first, second)
    width = len(grid[0])
    height = len(grid)
    for cell in cells:
        x, y = cell
        if not (0 <= x < width and 0 <= y < height) or grid[y][x] != 0:
            return False
    return all(
        _transition_cost(
            grid,
            elevation,
            current,
            neighbor,
            max_slope,
            cell_size,
            0.0,
        )
        is not None
        for current, neighbor in zip(cells, cells[1:])
    )


def _roadmap_shortest_path(
    nodes: Sequence[Cell],
    adjacency: Sequence[list[tuple[int, float]]],
) -> list[Cell]:
    goal_index = 1
    distances = {0: 0.0}
    previous: dict[int, int] = {}
    queue = [(0.0, 0)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances.get(current):
            continue
        if current == goal_index:
            break
        for neighbor, edge_cost in adjacency[current]:
            candidate = distance + edge_cost
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                previous[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    if goal_index not in distances:
        return []
    indices = [goal_index]
    while indices[-1] != 0:
        indices.append(previous[indices[-1]])
    indices.reverse()
    expanded: list[Cell] = []
    for first_index, second_index in zip(indices, indices[1:]):
        segment = _line_cells(nodes[first_index], nodes[second_index])
        expanded.extend(segment if not expanded else segment[1:])
    return expanded or [nodes[0]]


def run_prm(
    grid: Grid,
    start: Sequence[int],
    goal: Sequence[int],
    elevation: HeightField,
    max_slope: float = 1.25,
    cell_size: float = 1.0,
    slope_weight: float = 3.0,
    num_samples: int = 500,
    k_neighbors: int = 12,
    seed: Optional[int] = None,
) -> tuple[list[Cell], list[tuple[Cell, Cell]], float]:
    started = time.perf_counter()
    rng = random.Random(seed)
    free = [
        (x, y)
        for y, row in enumerate(grid)
        for x, value in enumerate(row)
        if value == 0
    ]
    start_cell = tuple(start)
    goal_cell = tuple(goal)
    candidates = [cell for cell in free if cell not in (start_cell, goal_cell)]
    samples = rng.sample(candidates, min(num_samples, len(candidates)))
    nodes = [start_cell, goal_cell, *samples]
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    explored_edges = []
    candidate_edges: set[tuple[int, int]] = set()
    for index, node in enumerate(nodes):
        nearest = sorted(
            (
                (heuristic(node, other), other_index)
                for other_index, other in enumerate(nodes)
                if other_index != index
            ),
            key=lambda item: item[0],
        )[:k_neighbors]
        for _, other_index in nearest:
            candidate_edges.add(tuple(sorted((index, other_index))))
    for index, other_index in sorted(candidate_edges):
        node = nodes[index]
        other = nodes[other_index]
        if not is_collision_free(
            grid, elevation, node, other, max_slope, cell_size
        ):
            continue
        segment = _line_cells(node, other)
        edge_cost = weighted_path_cost(
            segment,
            elevation,
            cell_size,
            slope_weight,
        )
        adjacency[index].append((other_index, edge_cost))
        adjacency[other_index].append((index, edge_cost))
        explored_edges.append((node, other))
    path = _roadmap_shortest_path(nodes, adjacency)
    return path, explored_edges, time.perf_counter() - started


def _nearest(nodes: Sequence[Cell], target: Cell) -> int:
    return min(range(len(nodes)), key=lambda index: heuristic(nodes[index], target))


def _steer(source: Cell, target: Cell, step_size: float) -> Cell:
    distance = heuristic(source, target)
    if distance <= step_size:
        return target
    ratio = step_size / distance
    return (
        round(source[0] + (target[0] - source[0]) * ratio),
        round(source[1] + (target[1] - source[1]) * ratio),
    )


def _tree_path(nodes: Sequence[Cell], parents: Sequence[Optional[int]], index: int) -> list[Cell]:
    path = [nodes[index]]
    while parents[index] is not None:
        index = parents[index]  # type: ignore[assignment]
        path.append(nodes[index])
    path.reverse()
    return path


def run_rrt_connect(
    grid: Grid,
    start: Sequence[int],
    goal: Sequence[int],
    elevation: HeightField,
    max_slope: float = 1.25,
    cell_size: float = 1.0,
    slope_weight: float = 3.0,
    step_size: float = 4.0,
    max_iter: int = 5000,
    seed: Optional[int] = None,
) -> tuple[list[Cell], list[tuple[Cell, Cell]], float]:
    started = time.perf_counter()
    rng = random.Random(seed)
    free = [
        (x, y)
        for y, row in enumerate(grid)
        for x, value in enumerate(row)
        if value == 0
    ]
    trees = [
        ([tuple(start)], [None]),
        ([tuple(goal)], [None]),
    ]
    explored_edges = []
    for iteration in range(max_iter):
        active = iteration % 2
        other = 1 - active
        nodes, parents = trees[active]
        other_nodes, other_parents = trees[other]
        sample = rng.choice(free)
        near_index = _nearest(nodes, sample)
        new_node = _steer(nodes[near_index], sample, step_size)
        if new_node == nodes[near_index] or not is_collision_free(
            grid, elevation, nodes[near_index], new_node, max_slope, cell_size
        ):
            continue
        nodes.append(new_node)
        parents.append(near_index)
        new_index = len(nodes) - 1
        explored_edges.append((nodes[near_index], new_node))

        other_index = _nearest(other_nodes, new_node)
        while True:
            extension = _steer(other_nodes[other_index], new_node, step_size)
            if extension == other_nodes[other_index] or not is_collision_free(
                grid,
                elevation,
                other_nodes[other_index],
                extension,
                max_slope,
                cell_size,
            ):
                break
            other_nodes.append(extension)
            other_parents.append(other_index)
            next_index = len(other_nodes) - 1
            explored_edges.append((other_nodes[other_index], extension))
            other_index = next_index
            if extension == new_node:
                first = _tree_path(nodes, parents, new_index)
                second = _tree_path(other_nodes, other_parents, other_index)
                if active == 0:
                    waypoints = first + list(reversed(second))[1:]
                else:
                    waypoints = second + list(reversed(first))[1:]
                return (
                    _expand_waypoints(waypoints),
                    explored_edges,
                    time.perf_counter() - started,
                )
    return [], explored_edges, time.perf_counter() - started


def run_rrt_star(
    grid: Grid,
    start: Sequence[int],
    goal: Sequence[int],
    elevation: HeightField,
    max_slope: float = 1.25,
    cell_size: float = 1.0,
    slope_weight: float = 3.0,
    step_size: float = 3.0,
    search_radius: float = 7.0,
    max_iter: int = 6000,
    seed: Optional[int] = None,
) -> tuple[list[Cell], list[tuple[Cell, Cell]], float]:
    started = time.perf_counter()
    rng = random.Random(seed)
    free = [
        (x, y)
        for y, row in enumerate(grid)
        for x, value in enumerate(row)
        if value == 0
    ]
    start_cell = tuple(start)
    goal_cell = tuple(goal)
    nodes = [start_cell]
    parents: list[Optional[int]] = [None]
    children: list[set[int]] = [set()]
    costs = [0.0]
    explored_edges = []
    for _ in range(max_iter):
        sample = goal_cell if rng.random() < 0.08 else rng.choice(free)
        nearest_index = _nearest(nodes, sample)
        new_node = _steer(nodes[nearest_index], sample, step_size)
        if new_node == nodes[nearest_index] or new_node in nodes:
            continue
        if not is_collision_free(
            grid, elevation, nodes[nearest_index], new_node, max_slope, cell_size
        ):
            continue
        near_indices = [
            index
            for index, node in enumerate(nodes)
            if heuristic(node, new_node) <= search_radius
            and is_collision_free(
                grid, elevation, node, new_node, max_slope, cell_size
            )
        ]
        parent_index = nearest_index
        parent_cost = costs[nearest_index] + weighted_path_cost(
            _line_cells(nodes[nearest_index], new_node),
            elevation,
            cell_size,
            slope_weight,
        )
        for index in near_indices:
            candidate = costs[index] + weighted_path_cost(
                _line_cells(nodes[index], new_node),
                elevation,
                cell_size,
                slope_weight,
            )
            if candidate < parent_cost:
                parent_index = index
                parent_cost = candidate
        nodes.append(new_node)
        parents.append(parent_index)
        children.append(set())
        costs.append(parent_cost)
        new_index = len(nodes) - 1
        children[parent_index].add(new_index)
        explored_edges.append((nodes[parent_index], new_node))

        for index in near_indices:
            if index == parent_index:
                continue
            rewired_cost = parent_cost + weighted_path_cost(
                _line_cells(new_node, nodes[index]),
                elevation,
                cell_size,
                slope_weight,
            )
            if rewired_cost < costs[index]:
                old_parent = parents[index]
                assert old_parent is not None
                children[old_parent].discard(index)
                parents[index] = new_index
                children[new_index].add(index)
                cost_delta = rewired_cost - costs[index]
                stack = [index]
                while stack:
                    descendant = stack.pop()
                    costs[descendant] += cost_delta
                    stack.extend(children[descendant])

    goal_candidates = []
    for index, node in enumerate(nodes):
        if heuristic(node, goal_cell) > step_size:
            continue
        if not is_collision_free(
            grid,
            elevation,
            node,
            goal_cell,
            max_slope,
            cell_size,
        ):
            continue
        connection_cost = weighted_path_cost(
            _line_cells(node, goal_cell),
            elevation,
            cell_size,
            slope_weight,
        )
        goal_candidates.append((costs[index] + connection_cost, index))
    if not goal_candidates:
        return [], explored_edges, time.perf_counter() - started
    _, goal_parent_index = min(goal_candidates)
    waypoints = _tree_path(nodes, parents, goal_parent_index)
    if waypoints[-1] != goal_cell:
        waypoints.append(goal_cell)
    return (
        _expand_waypoints(waypoints),
        explored_edges,
        time.perf_counter() - started,
    )


def plot_algorithm_result_25d(
    payload: dict,
    path: Sequence[Cell],
    algorithm_name: str,
    save_path: Path,
) -> None:
    """Drape a planned route over the same 2.5D terrain used by the planner."""

    route_label = (
        algorithm_name
        if "path" in algorithm_name.lower()
        else f"{algorithm_name} path"
    )
    save_showcase_visualization(
        payload,
        save_path,
        path=path,
        route_label=route_label,
    )


PLANNERS = {
    "BFS": run_bfs,
    "Dijkstra": run_dijkstra,
    "A_Star": run_astar,
    "PRM": run_prm,
    "RRT_Connect": run_rrt_connect,
    "RRT_Star": run_rrt_star,
}
STOCHASTIC_PLANNERS = frozenset(("PRM", "RRT_Connect", "RRT_Star"))
PLANNER_SEED_OFFSETS = {
    "PRM": 11,
    "RRT_Connect": 23,
    "RRT_Star": 37,
}


def is_path_valid(
    path: Sequence[Cell],
    grid: Grid,
    elevation: HeightField,
    start: Sequence[int],
    goal: Sequence[int],
    max_slope: float,
    cell_size: float,
) -> bool:
    """Verify endpoints, collision clearance, corner rules, and edge slope."""

    if not path or tuple(path[0]) != tuple(start) or tuple(path[-1]) != tuple(goal):
        return False
    if not is_collision_free(
        grid,
        elevation,
        tuple(path[0]),
        tuple(path[0]),
        max_slope,
        cell_size,
    ):
        return False
    return all(
        is_collision_free(
            grid,
            elevation,
            first,
            second,
            max_slope,
            cell_size,
        )
        for first, second in zip(path, path[1:])
    )


def _tasks_for_payload(payload: dict) -> list[dict]:
    tasks = payload.get("tasks")
    if tasks:
        return tasks
    return [
        {
            "task_index": 1,
            "task_id": f"{payload['instance_id']}_sg01",
            "matched_task_id": f"{payload['instance_id']}_sg01",
            "start": payload["start"],
            "goal": payload["goal"],
            "distance_class": "unspecified",
            "canonical_visualization": True,
        }
    ]


def _select_tasks(payload: dict, task_scope: str) -> list[dict]:
    """Select the published task set or its single canonical member."""

    tasks = _tasks_for_payload(payload)
    if task_scope == "all":
        return tasks
    if task_scope != "canonical":
        raise ValueError(f"unknown task scope: {task_scope}")
    canonical = [
        task for task in tasks if bool(task.get("canonical_visualization", False))
    ]
    if len(canonical) != 1:
        raise ValueError(
            f"{payload['instance_id']} must contain exactly one canonical task; "
            f"found {len(canonical)}"
        )
    return canonical


def _trial_seed(
    payload: dict,
    task: dict,
    algorithm_name: str,
    trial: int,
) -> int:
    base_seed = int(payload.get("task_seed", payload.get("seed", 0)))
    task_index = int(task.get("task_index", 1))
    return (
        base_seed
        + task_index * 1009
        + PLANNER_SEED_OFFSETS[algorithm_name] * 1_000_003
        + trial
    )


def _mean_and_std(values: Sequence[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    mean_value = statistics.fmean(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return round(mean_value, 6), round(std_value, 6)


def _write_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty benchmark table: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_benchmark_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_benchmark_checkpoints(path: Path) -> dict[str, dict]:
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
                if any(item.strip() for item in handle):
                    raise ValueError(
                        f"invalid benchmark checkpoint line {line_number}"
                    )
                break
            records[record["map_file"]] = record
    return records


def _benchmark_map_worker(job: dict) -> dict:
    """Run every requested algorithm/trial for one map checkpoint unit."""

    map_path = Path(job["map_path"])
    with map_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    grid = payload["grid"]
    terrain_elevation = payload["elevation"]
    elevation = payload.get("terrain_analysis", {}).get(
        "support_elevation",
        terrain_elevation,
    )
    navigation = payload["navigation"]
    max_slope = float(navigation["max_slope"])
    cell_size = float(navigation["cell_size"])
    slope_weight = float(
        navigation.get("path_slope_weight", navigation.get("slope_weight", 3.0))
    )
    tasks = _select_tasks(payload, job["task_scope"])
    if job["max_tasks_per_map"] is not None:
        tasks = tasks[: job["max_tasks_per_map"]]

    trial_rows = []
    representatives = {}
    for task in tasks:
        for algorithm_name in job["algorithm_names"]:
            algorithm = PLANNERS[algorithm_name]
            trial_count = (
                job["stochastic_trials"]
                if algorithm_name in STOCHASTIC_PLANNERS
                else 1
            )
            for trial in range(trial_count):
                seed: Optional[int] = None
                extra_arguments = {}
                budget_type = "deterministic graph search"
                budget_value: int | str = "complete"
                if algorithm_name in STOCHASTIC_PLANNERS:
                    seed = _trial_seed(payload, task, algorithm_name, trial)
                    extra_arguments["seed"] = seed
                if algorithm_name == "PRM":
                    extra_arguments.update(
                        num_samples=job["prm_samples"],
                        k_neighbors=job["prm_neighbors"],
                    )
                    budget_type = "samples"
                    budget_value = job["prm_samples"]
                elif algorithm_name == "RRT_Connect":
                    extra_arguments.update(
                        max_iter=job["rrt_connect_iterations"],
                        step_size=job["rrt_connect_step_size"],
                    )
                    budget_type = "iterations"
                    budget_value = job["rrt_connect_iterations"]
                elif algorithm_name == "RRT_Star":
                    extra_arguments.update(
                        max_iter=job["rrt_star_iterations"],
                        step_size=job["rrt_star_step_size"],
                        search_radius=job["rrt_star_search_radius"],
                    )
                    budget_type = "iterations"
                    budget_value = job["rrt_star_iterations"]

                path, explored_edges, elapsed = algorithm(
                    grid,
                    task["start"],
                    task["goal"],
                    elevation,
                    max_slope=max_slope,
                    cell_size=cell_size,
                    slope_weight=slope_weight,
                    **extra_arguments,
                )
                valid = is_path_valid(
                    path,
                    grid,
                    elevation,
                    task["start"],
                    task["goal"],
                    max_slope,
                    cell_size,
                )
                success = bool(path) and valid
                length_2d = planar_path_length(path, cell_size) if success else None
                length_3d = (
                    path_length(path, elevation, cell_size) if success else None
                )
                common_cost = (
                    weighted_path_cost(path, elevation, cell_size, slope_weight)
                    if success
                    else None
                )
                elevation_gain = (
                    path_elevation_gain(path, elevation) if success else None
                )
                trial_rows.append(
                    {
                        "terrain_id": payload["instance_id"],
                        "matched_group_id": payload.get(
                            "matched_group_id", payload["instance_id"]
                        ),
                        "family": payload.get(
                            "terrain_family",
                            payload.get("terrain_profile", "other"),
                        ),
                        "difficulty": payload.get("difficulty", "unspecified"),
                        "task_id": task.get(
                            "task_id", task.get("matched_task_id", "unknown")
                        ),
                        "matched_task_id": task.get(
                            "matched_task_id", task.get("task_id", "unknown")
                        ),
                        "distance_class": task.get(
                            "distance_class", "unspecified"
                        ),
                        "canonical_task": bool(
                            task.get("canonical_visualization", False)
                        ),
                        "algorithm": algorithm_name,
                        "trial": trial + 1,
                        "seed": "" if seed is None else seed,
                        "budget_type": budget_type,
                        "budget_value": budget_value,
                        "success": success,
                        "path_valid": valid if path else "",
                        "runtime_sec": round(elapsed, 6),
                        "path_length_2d_m": (
                            "" if length_2d is None else round(length_2d, 6)
                        ),
                        "path_length_3d_m": (
                            "" if length_3d is None else round(length_3d, 6)
                        ),
                        "weighted_path_cost": (
                            "" if common_cost is None else round(common_cost, 6)
                        ),
                        "elevation_gain_m": (
                            "" if elevation_gain is None else round(elevation_gain, 6)
                        ),
                        "path_cells": len(path) if success else 0,
                        "explored_edges": len(explored_edges),
                    }
                )
                should_capture = (
                    job["render_scope"] != "none"
                    and bool(task.get("canonical_visualization", False))
                    and (
                        job["render_scope"] == "canonical_all"
                        or int(payload.get("instance_index", 1)) == 1
                    )
                )
                existing_representative = representatives.get(algorithm_name)
                if should_capture and (
                    existing_representative is None
                    or (
                        not existing_representative["success"]
                        and success
                    )
                ):
                    representatives[algorithm_name] = {
                        "start": task["start"],
                        "goal": task["goal"],
                        "path": path,
                        "success": success,
                        "trial": trial + 1,
                    }

    record = {
        "record_type": "completed_map_benchmark",
        "map_file": job["map_file"],
        "map_sha256": _sha256_file(map_path),
        "terrain_id": payload["instance_id"],
        "family": payload.get(
            "terrain_family", payload.get("terrain_profile", "other")
        ),
        "tasks": len(tasks),
        "trial_rows": trial_rows,
        "representatives": representatives,
    }
    if representatives:
        render_result = _render_benchmark_map_worker(
            {
                "map_path": str(map_path),
                "output_root": job["output_root"],
                "representatives": representatives,
            }
        )
        record["rendered_images"] = render_result["rendered"]
    return record


def _render_benchmark_map_worker(job: dict) -> dict:
    """Render every retained planner route for one map."""

    map_path = Path(job["map_path"])
    output_root = Path(job["output_root"])
    with map_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    family = payload.get(
        "terrain_family", payload.get("terrain_profile", "other")
    )
    rendered = 0
    for algorithm_name, representative in job["representatives"].items():
        image_path = (
            output_root
            / "images"
            / "paths"
            / family
            / f"{payload['instance_id']}_{algorithm_name}.png"
        )
        if image_path.is_file() and image_path.stat().st_size > 0:
            continue
        render_payload = dict(payload)
        render_payload["start"] = representative["start"]
        render_payload["goal"] = representative["goal"]
        temporary_path = image_path.with_name(
            f".{image_path.stem}.{os.getpid()}.tmp.png"
        )
        try:
            display_name = (
                algorithm_name
                if representative.get("success", bool(representative["path"]))
                else f"{algorithm_name} — no valid path"
            )
            plot_algorithm_result_25d(
                render_payload,
                representative["path"],
                display_name,
                temporary_path,
            )
            os.replace(temporary_path, image_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        rendered += 1
    return {"terrain_id": payload["instance_id"], "rendered": rendered}


def _render_benchmark_images(
    dataset_root: Path,
    ordered_records: Sequence[dict],
    workers: int,
) -> int:
    jobs = []
    for record in ordered_records:
        if not record["representatives"]:
            continue
        missing = False
        family = record.get("family")
        if not family:
            with (dataset_root / record["map_file"]).open(
                "r", encoding="utf-8"
            ) as handle:
                payload = json.load(handle)
            family = payload.get(
                "terrain_family", payload.get("terrain_profile", "other")
            )
        for algorithm_name in record["representatives"]:
            image_path = (
                dataset_root
                / "images"
                / "paths"
                / family
                / f"{record['terrain_id']}_{algorithm_name}.png"
            )
            if not image_path.is_file() or image_path.stat().st_size == 0:
                missing = True
                break
        if missing:
            jobs.append(
                {
                    "map_path": str(dataset_root / record["map_file"]),
                    "output_root": str(dataset_root),
                    "representatives": record["representatives"],
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
                f"Rendered planner paths {completed}/{len(jobs)}: "
                f"{result['terrain_id']}"
            )

    if workers == 1:
        for job in jobs:
            report(_render_benchmark_map_worker(job))
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        futures = {
            executor.submit(_render_benchmark_map_worker, job): job for job in jobs
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


def _benchmark_signature(
    map_files: Sequence[str],
    *,
    algorithm_names: Sequence[str],
    stochastic_trials: int,
    prm_samples: int,
    prm_neighbors: int,
    rrt_connect_iterations: int,
    rrt_star_iterations: int,
    rrt_connect_step_size: float,
    rrt_star_step_size: float,
    rrt_star_search_radius: float,
    render_scope: str,
    task_scope: str,
    max_tasks_per_map: Optional[int],
) -> dict:
    signature = {
        "schema_version": "1.0",
        "map_files": list(map_files),
        "algorithms": list(algorithm_names),
        "stochastic_trials": stochastic_trials,
        "prm_samples": prm_samples,
        "prm_neighbors": prm_neighbors,
        "rrt_connect_iterations": rrt_connect_iterations,
        "rrt_star_iterations": rrt_star_iterations,
        "rrt_connect_step_size": rrt_connect_step_size,
        "rrt_star_step_size": rrt_star_step_size,
        "rrt_star_search_radius": rrt_star_search_radius,
        "render_scope": render_scope,
        "max_tasks_per_map": max_tasks_per_map,
    }
    # Keep the legacy all-task signature byte-for-byte compatible so an
    # existing checkpoint can still be resumed after this option was added.
    if task_scope != "all":
        signature["task_scope"] = task_scope
    return signature


def _signature_hash(signature: dict) -> str:
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_benchmark(
    dataset_root: Path,
    *,
    algorithm_names: Sequence[str] = tuple(PLANNERS),
    stochastic_trials: int = 5,
    prm_samples: int = 500,
    prm_neighbors: int = 12,
    rrt_connect_iterations: int = 5000,
    rrt_star_iterations: int = 6000,
    rrt_connect_step_size: float = 4.0,
    rrt_star_step_size: float = 3.0,
    rrt_star_search_radius: float = 7.0,
    render_representatives: bool = False,
    render_all_canonical_paths: bool = False,
    task_scope: str = "all",
    max_maps: Optional[int] = None,
    max_tasks_per_map: Optional[int] = None,
    output_prefix: str = "pathfinding",
    workers: int = 1,
    resume: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Run a fixed-budget, task-level benchmark with no hidden retries."""

    if stochastic_trials < 1:
        raise ValueError("stochastic_trials must be at least one")
    if workers < 1:
        raise ValueError("workers must be at least one")
    if task_scope not in {"all", "canonical"}:
        raise ValueError("task_scope must be 'all' or 'canonical'")
    if task_scope == "canonical" and max_tasks_per_map is not None:
        raise ValueError(
            "--max-tasks-per-map cannot be combined with --task-scope canonical"
        )
    if render_representatives and render_all_canonical_paths:
        raise ValueError(
            "choose either representative or all-canonical path rendering"
        )
    render_scope = (
        "canonical_all"
        if render_all_canonical_paths
        else "representative"
        if render_representatives
        else "none"
    )
    unknown = set(algorithm_names) - set(PLANNERS)
    if unknown:
        raise ValueError(f"unknown planners: {sorted(unknown)}")
    map_paths = sorted(dataset_root.joinpath("maps").rglob("terrain_*.json"))
    if max_maps is not None:
        map_paths = map_paths[:max_maps]
    if not map_paths:
        raise FileNotFoundError(f"no terrain maps found below {dataset_root / 'maps'}")

    metadata_root = dataset_root / "metadata"
    state_path = metadata_root / f"{output_prefix}_state.json"
    checkpoint_path = metadata_root / f"{output_prefix}_progress.jsonl"
    trials_path = metadata_root / f"{output_prefix}_trials.csv"
    summary_path = metadata_root / f"{output_prefix}_summary.csv"
    protocol_path = metadata_root / f"{output_prefix}_protocol.json"
    map_files = [path.relative_to(dataset_root).as_posix() for path in map_paths]
    signature = _benchmark_signature(
        map_files,
        algorithm_names=algorithm_names,
        stochastic_trials=stochastic_trials,
        prm_samples=prm_samples,
        prm_neighbors=prm_neighbors,
        rrt_connect_iterations=rrt_connect_iterations,
        rrt_star_iterations=rrt_star_iterations,
        rrt_connect_step_size=rrt_connect_step_size,
        rrt_star_step_size=rrt_star_step_size,
        rrt_star_search_radius=rrt_star_search_radius,
        render_scope=render_scope,
        task_scope=task_scope,
        max_tasks_per_map=max_tasks_per_map,
    )
    signature_hash = _signature_hash(signature)
    output_files = (
        state_path,
        checkpoint_path,
        trials_path,
        summary_path,
        protocol_path,
    )
    if resume:
        if not state_path.is_file():
            raise ValueError(f"no benchmark state to resume: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature_hash") != signature_hash:
            raise ValueError(
                "resume benchmark configuration differs from the original run"
            )
    else:
        existing = [path for path in output_files if path.exists()]
        if existing:
            raise FileExistsError(
                f"benchmark output already exists: {existing[0]}; use --resume "
                "or choose another --output-prefix"
            )
        state = {
            "schema_version": "1.0",
            "status": "in_progress",
            "signature_hash": signature_hash,
            "signature": signature,
            "workers_used": [],
        }
        _atomic_write_json(state, state_path)

    loaded_records = _load_benchmark_checkpoints(checkpoint_path)
    completed_records = {}
    selected_map_files = set(map_files)
    for map_file, record in loaded_records.items():
        map_path = dataset_root / map_file
        if (
            map_file in selected_map_files
            and map_path.is_file()
            and _sha256_file(map_path) == record.get("map_sha256")
        ):
            completed_records[map_file] = record

    common_job = {
        "algorithm_names": tuple(algorithm_names),
        "stochastic_trials": stochastic_trials,
        "prm_samples": prm_samples,
        "prm_neighbors": prm_neighbors,
        "rrt_connect_iterations": rrt_connect_iterations,
        "rrt_star_iterations": rrt_star_iterations,
        "rrt_connect_step_size": rrt_connect_step_size,
        "rrt_star_step_size": rrt_star_step_size,
        "rrt_star_search_radius": rrt_star_search_radius,
        "render_scope": render_scope,
        "task_scope": task_scope,
        "max_tasks_per_map": max_tasks_per_map,
        "output_root": str(dataset_root),
    }
    jobs = [
        {
            **common_job,
            "map_file": map_file,
            "map_path": str(dataset_root / map_file),
        }
        for map_file in map_files
        if map_file not in completed_records
    ]
    if completed_records:
        print(
            f"Resume verified {len(completed_records)}/{len(map_files)} maps; "
            f"{len(jobs)} remain"
        )
    if jobs and workers not in state.setdefault("workers_used", []):
        state["workers_used"].append(workers)
        state["status"] = "in_progress"
        _atomic_write_json(state, state_path)
    completed_count = len(completed_records)

    def commit_result(record: dict) -> None:
        nonlocal completed_count
        _append_benchmark_checkpoint(checkpoint_path, record)
        completed_records[record["map_file"]] = record
        completed_count += 1
        print(
            f"[{completed_count:05d}/{len(map_files):05d}] "
            f"{record['terrain_id']}: {record['tasks']} tasks"
        )

    try:
        if workers == 1:
            for job in jobs:
                commit_result(_benchmark_map_worker(job))
        elif jobs:
            executor = ProcessPoolExecutor(max_workers=workers)
            futures = {
                executor.submit(_benchmark_map_worker, job): job for job in jobs
            }
            try:
                for future in as_completed(futures):
                    commit_result(future.result())
            except BaseException as error:
                if isinstance(error, KeyboardInterrupt):
                    print(
                        "\nStop requested; waiting for active planner workers...",
                        flush=True,
                    )
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
    except BaseException:
        state["status"] = "interrupted"
        state["completed_maps"] = len(completed_records)
        _atomic_write_json(state, state_path)
        raise

    if len(completed_records) != len(map_files):
        raise RuntimeError(
            f"benchmark stopped with {len(completed_records)}/{len(map_files)} maps"
        )
    ordered_records = [completed_records[map_file] for map_file in map_files]
    trial_rows = [
        row for record in ordered_records for row in record["trial_rows"]
    ]

    worker_rendered_map_count = sum(
        int(record.get("rendered_images", 0) > 0)
        for record in completed_records.values()
    )
    final_pass_rendered_map_count = 0
    if render_scope != "none":
        try:
            final_pass_rendered_map_count = _render_benchmark_images(
                dataset_root, ordered_records, workers
            )
        except BaseException:
            state["status"] = "interrupted"
            state["completed_maps"] = len(map_files)
            state["phase"] = "rendering_path_images"
            _atomic_write_json(state, state_path)
            raise

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in trial_rows:
        grouped[(row["algorithm"], row["family"], row["difficulty"])].append(row)
    summary_rows = []
    for (algorithm_name, family, difficulty), rows in sorted(grouped.items()):
        successful = [row for row in rows if row["success"]]
        runtime_mean, runtime_std = _mean_and_std(
            [float(row["runtime_sec"]) for row in rows]
        )
        length_mean, length_std = _mean_and_std(
            [float(row["path_length_3d_m"]) for row in successful]
        )
        cost_mean, cost_std = _mean_and_std(
            [float(row["weighted_path_cost"]) for row in successful]
        )
        gain_mean, gain_std = _mean_and_std(
            [float(row["elevation_gain_m"]) for row in successful]
        )
        summary_rows.append(
            {
                "algorithm": algorithm_name,
                "family": family,
                "difficulty": difficulty,
                "tasks": len({row["task_id"] for row in rows}),
                "trials": len(rows),
                "successes": len(successful),
                "success_rate": round(len(successful) / len(rows), 6),
                "runtime_mean_sec": runtime_mean,
                "runtime_std_sec": runtime_std,
                "path_length_3d_mean_m": length_mean,
                "path_length_3d_std_m": length_std,
                "weighted_cost_mean": cost_mean,
                "weighted_cost_std": cost_std,
                "elevation_gain_mean_m": gain_mean,
                "elevation_gain_std_m": gain_std,
            }
        )

    _write_rows(trials_path, trial_rows)
    _write_rows(summary_path, summary_rows)
    protocol = {
        "schema_version": "1.1",
        "algorithms": list(algorithm_names),
        "stochastic_algorithms": sorted(STOCHASTIC_PLANNERS & set(algorithm_names)),
        "stochastic_trials": stochastic_trials,
        "movement": {
            "neighbors": 8,
            "diagonal_corner_cutting": False,
            "sampling_edge_discretization": "symmetric Bresenham 8-connected path",
            "collision_grid_is_footprint_conditioned": True,
            "edge_slope_limit_enforced": True,
            "elevation_for_navigation": (
                "footprint support-plane elevation when present; raw terrain fallback"
            ),
        },
        "common_objective": (
            "sum(sqrt(planar_distance^2 + dz^2) * "
            "(1 + slope_weight * (abs(dz) / planar_distance)^2))"
        ),
        "fixed_budgets": {
            "PRM": {
                "samples": prm_samples,
                "neighbors": prm_neighbors,
            },
            "RRT_Connect": {
                "iterations": rrt_connect_iterations,
                "step_size_cells": rrt_connect_step_size,
            },
            "RRT_Star": {
                "iterations": rrt_star_iterations,
                "step_size_cells": rrt_star_step_size,
                "search_radius_cells": rrt_star_search_radius,
                "returns": "lowest-cost goal connection after full budget",
            },
        },
        "hidden_retries": False,
        "task_scope": task_scope,
        "tasks_per_map_limit": max_tasks_per_map,
        "maps_limit": max_maps,
        "path_render_scope": render_scope,
        "parallel_execution": {
            "workers_used": state.get("workers_used") or [workers],
            "parallel_image_rendering": render_scope != "none",
            "maps_with_retained_paths": sum(
                bool(record["representatives"]) for record in ordered_records
            ),
            "maps_rendered_by_benchmark_workers": worker_rendered_map_count,
            "maps_repaired_in_final_render_pass": final_pass_rendered_map_count,
            "checkpoint_unit": "one map with all requested tasks and trials",
            "resume_supported": True,
            "checkpoint_file": checkpoint_path.relative_to(dataset_root).as_posix(),
            "signature_hash": signature_hash,
        },
    }
    _atomic_write_json(protocol, protocol_path)
    manifest_path = dataset_root / "metadata" / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        benchmarks = manifest.setdefault("planner_benchmarks", {})
        benchmarks[output_prefix] = {
            "task_scope": task_scope,
            "algorithms": list(algorithm_names),
            "stochastic_trials": stochastic_trials,
            "path_render_scope": render_scope,
            "map_count": len(map_files),
            "trial_row_count": len(trial_rows),
            "path_image_count_expected": (
                len(map_files) * len(algorithm_names)
                if render_scope == "canonical_all"
                else None
            ),
            "trials_file": trials_path.relative_to(dataset_root).as_posix(),
            "summary_file": summary_path.relative_to(dataset_root).as_posix(),
            "protocol_file": protocol_path.relative_to(dataset_root).as_posix(),
        }
        _atomic_write_json(manifest, manifest_path)
    state["status"] = "complete"
    state["phase"] = "complete"
    state["completed_maps"] = len(map_files)
    state["total_trial_rows"] = len(trial_rows)
    _atomic_write_json(state, state_path)
    return trial_rows, summary_rows


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run a fixed-budget benchmark of six 2.5D path planners."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "dataset_preview",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(PLANNERS),
        default=tuple(PLANNERS),
    )
    parser.add_argument("--stochastic-trials", type=int, default=5)
    parser.add_argument("--prm-samples", type=int, default=500)
    parser.add_argument("--prm-neighbors", type=int, default=12)
    parser.add_argument("--rrt-connect-iterations", type=int, default=5000)
    parser.add_argument("--rrt-star-iterations", type=int, default=6000)
    parser.add_argument("--rrt-connect-step-size", type=float, default=4.0)
    parser.add_argument("--rrt-star-step-size", type=float, default=3.0)
    parser.add_argument("--rrt-star-search-radius", type=float, default=7.0)
    rendering = parser.add_mutually_exclusive_group()
    rendering.add_argument("--render-representatives", action="store_true")
    rendering.add_argument(
        "--render-all-canonical-paths",
        action="store_true",
        help="render one canonical task for every map and requested algorithm",
    )
    parser.add_argument("--max-maps", type=int)
    parser.add_argument("--max-tasks-per-map", type=int)
    parser.add_argument(
        "--task-scope",
        choices=("all", "canonical"),
        default="all",
        help=(
            "benchmark all published tasks or the one canonical long-route "
            "task per map"
        ),
    )
    parser.add_argument("--output-prefix", default="pathfinding")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, os.cpu_count() or 1)),
        help="parallel per-map workers (default: up to 6)",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_benchmark(
            args.dataset.resolve(),
            algorithm_names=args.algorithms,
            stochastic_trials=args.stochastic_trials,
            prm_samples=args.prm_samples,
            prm_neighbors=args.prm_neighbors,
            rrt_connect_iterations=args.rrt_connect_iterations,
            rrt_star_iterations=args.rrt_star_iterations,
            rrt_connect_step_size=args.rrt_connect_step_size,
            rrt_star_step_size=args.rrt_star_step_size,
            rrt_star_search_radius=args.rrt_star_search_radius,
            render_representatives=args.render_representatives,
            render_all_canonical_paths=args.render_all_canonical_paths,
            task_scope=args.task_scope,
            max_maps=args.max_maps,
            max_tasks_per_map=args.max_tasks_per_map,
            output_prefix=args.output_prefix,
            workers=args.workers,
            resume=args.resume,
        )
    except KeyboardInterrupt:
        print(
            "\nStopped safely after completed maps were checkpointed. "
            "Run the identical command with --resume to continue.",
            flush=True,
        )
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
