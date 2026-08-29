"""Procedural 2.5D terrain generation and traversability analysis.

Every (x, y) location owns exactly one elevation value.  Obstacles are not
holes in the height field: they are terrain regions rejected by the robot's
slope, step-height, roughness, or footprint constraints.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
from scipy import ndimage


Cell = tuple[int, int]


@dataclass(frozen=True)
class TerrainConfig:
    size: int = 128
    cell_size: float = 0.25
    footprint_radius: float = 0.45
    endpoint_margin: float = 1.50
    max_slope_degrees: float = 28.0
    max_step_height: float = 0.65
    max_roughness: float = 0.10
    slope_weight: float = 3.0
    traversability_slope_weight: float = 0.45
    traversability_roughness_weight: float = 0.25
    traversability_step_weight: float = 0.30
    min_navigable_fraction: float = 0.28
    max_navigable_fraction: float = 0.88
    min_largest_component_fraction: float = 0.24

    @property
    def max_slope(self) -> float:
        """Maximum rise/run used by the path planners."""

        return math.tan(math.radians(self.max_slope_degrees))

    @property
    def footprint_cells(self) -> int:
        return max(1, math.ceil(self.footprint_radius / self.cell_size))

    @property
    def analysis_window(self) -> int:
        return self.footprint_cells * 2 + 1

    @property
    def endpoint_margin_cells(self) -> int:
        return max(self.footprint_cells + 1, math.ceil(self.endpoint_margin / self.cell_size))

    def validate(self) -> None:
        if self.size < 32:
            raise ValueError("size must be at least 32")
        if self.cell_size <= 0 or self.footprint_radius <= 0 or self.endpoint_margin <= 0:
            raise ValueError("cell_size, footprint_radius, and endpoint_margin must be positive")
        if not 1.0 <= self.max_slope_degrees < 89.0:
            raise ValueError("max_slope_degrees must be between 1 and 89")
        if self.max_step_height <= 0 or self.max_roughness <= 0:
            raise ValueError("step and roughness thresholds must be positive")
        if self.slope_weight < 0:
            raise ValueError("slope_weight must be non-negative")
        terrain_weights = (
            self.traversability_slope_weight,
            self.traversability_roughness_weight,
            self.traversability_step_weight,
        )
        if any(weight < 0 for weight in terrain_weights):
            raise ValueError("traversability weights must be non-negative")
        if not math.isclose(sum(terrain_weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("traversability weights must sum to one")
        if not 0 < self.min_navigable_fraction < self.max_navigable_fraction < 1:
            raise ValueError("invalid navigable-fraction limits")
        if not 0 < self.min_largest_component_fraction < 1:
            raise ValueError("min_largest_component_fraction must be between zero and one")

    def as_serializable_dict(self) -> dict:
        result = asdict(self)
        result["max_slope"] = self.max_slope
        result["footprint_cells"] = self.footprint_cells
        result["analysis_window"] = self.analysis_window
        result["endpoint_margin_cells"] = self.endpoint_margin_cells
        return result


def _normalized_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    sigma: float,
) -> np.ndarray:
    values = ndimage.gaussian_filter(
        rng.standard_normal(shape), sigma=sigma, mode="reflect"
    )
    values -= values.mean()
    standard_deviation = values.std()
    if standard_deviation > 1e-12:
        values /= standard_deviation
    return values


def _elliptic_feature(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    angle: float,
) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    dx = x_grid - center_x
    dy = y_grid - center_y
    along = cosine * dx + sine * dy
    across = -sine * dx + cosine * dy
    return np.exp(-0.5 * ((along / radius_x) ** 2 + (across / radius_y) ** 2))


def _profile_parameters(profile: str) -> dict[str, float | int]:
    profiles: dict[str, dict[str, float | int]] = {
        "rolling": {
            "base_relief": 0.24,
            "hills": 9,
            "hill_amplitude": 2.2,
            "ridges": 1,
            "ridge_amplitude": 2.4,
            "mesas": 1,
            "rough_patches": 2,
            "rough_amplitude": 0.10,
        },
        "mountain": {
            "base_relief": 0.38,
            "hills": 12,
            "hill_amplitude": 3.8,
            "ridges": 3,
            "ridge_amplitude": 4.2,
            "mesas": 1,
            "rough_patches": 2,
            "rough_amplitude": 0.14,
        },
        "rugged": {
            "base_relief": 0.48,
            "hills": 13,
            "hill_amplitude": 2.8,
            "ridges": 2,
            "ridge_amplitude": 3.2,
            "mesas": 2,
            "rough_patches": 6,
            "rough_amplitude": 0.26,
        },
    }
    if profile not in profiles:
        raise ValueError(f"unknown terrain profile: {profile}")
    return profiles[profile]


def _generate_special_profile(
    profile: str,
    rng: np.random.Generator,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    size: int,
) -> np.ndarray:
    """Generate the smooth-obstacle and plateau release families."""

    base = 0.055 * _normalized_noise(rng, (size, size), size / 12.0)
    if profile == "smooth_obstacles":
        # A seeded compound mountain plus isolated smooth domes. The layout
        # changes between instances while retaining the family morphology.
        elevation = base.copy()
        map_radius = float(np.max(np.abs(x_grid)))
        cluster_x = rng.uniform(-0.22, 0.22) * map_radius
        cluster_y = rng.uniform(-0.08, 0.28) * map_radius
        cluster_angle = rng.uniform(0.0, 2.0 * math.pi)
        features = []
        for offset in (-3.8, 0.0, 3.8):
            features.append(
                (
                    cluster_x + math.cos(cluster_angle) * offset + rng.uniform(-1.0, 1.0),
                    cluster_y + math.sin(cluster_angle) * offset + rng.uniform(-1.0, 1.0),
                    rng.uniform(3.0, 4.7),
                    rng.uniform(3.0, 4.8),
                    rng.uniform(2.6, 3.9),
                )
            )
        for _ in range(2):
            features.append(
                (
                    rng.uniform(-0.62, 0.62) * map_radius,
                    rng.uniform(-0.68, -0.36) * map_radius,
                    rng.uniform(1.2, 1.8),
                    rng.uniform(1.2, 1.8),
                    rng.uniform(1.2, 1.8),
                )
            )
        for center_x, center_y, radius_x, radius_y, amplitude in features:
            elevation += amplitude * _elliptic_feature(
                x_grid,
                y_grid,
                center_x,
                center_y,
                radius_x,
                radius_y,
                rng.uniform(-0.16, 0.16),
            )
        return elevation

    if profile == "plateau":
        elevation = base + 0.08 * _normalized_noise(
            rng, (size, size), size / 16.0
        )
        map_radius = float(np.max(np.abs(x_grid)))
        plateau_count = int(rng.integers(3, 5))
        phase = rng.uniform(0.0, 2.0 * math.pi)
        features = []
        for index in range(plateau_count):
            position_angle = (
                phase
                + 2.0 * math.pi * index / plateau_count
                + rng.uniform(-0.32, 0.32)
            )
            radial_position = rng.uniform(0.28, 0.58) * map_radius
            features.append(
                (
                    math.cos(position_angle) * radial_position,
                    math.sin(position_angle) * radial_position,
                    rng.uniform(2.3, 4.8),
                    rng.uniform(2.0, 4.3),
                    rng.uniform(0.0, math.pi),
                    rng.uniform(2.2, 3.8),
                    rng.uniform(0.075, 0.13),
                )
            )
        for center_x, center_y, radius_x, radius_y, angle, amplitude, softness in features:
            cosine = math.cos(angle)
            sine = math.sin(angle)
            dx = x_grid - center_x
            dy = y_grid - center_y
            along = (cosine * dx + sine * dy) / radius_x
            across = (-sine * dx + cosine * dy) / radius_y
            radius = np.sqrt(along * along + across * across)
            elevation += amplitude / (
                1.0 + np.exp(np.clip((radius - 1.0) / softness, -40, 40))
            )
        return elevation
    raise ValueError(f"unknown special terrain profile: {profile}")


def generate_elevation(
    config: TerrainConfig,
    seed: int,
    profile: str,
) -> np.ndarray:
    """Generate a continuous full-frame height field in meters."""

    config.validate()
    rng = np.random.default_rng(seed)
    size = config.size
    extent = (size - 1) * config.cell_size
    coordinates = np.linspace(-extent / 2.0, extent / 2.0, size)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)

    if profile in {
        "smooth_obstacles",
        "plateau",
    }:
        elevation = _generate_special_profile(
            profile, rng, x_grid, y_grid, size
        )
        elevation = ndimage.gaussian_filter(elevation, sigma=0.55, mode="reflect")
        elevation -= elevation.min()
        return elevation

    parameters = _profile_parameters(profile)

    elevation = (
        _normalized_noise(rng, (size, size), size / 7.0)
        * float(parameters["base_relief"])
        + _normalized_noise(rng, (size, size), size / 18.0)
        * float(parameters["base_relief"])
        * 0.42
    )

    margin = extent * 0.36
    for _ in range(int(parameters["hills"])):
        center_x = rng.uniform(-margin, margin)
        center_y = rng.uniform(-margin, margin)
        radius_x = rng.uniform(1.3, 4.8)
        radius_y = rng.uniform(1.2, 4.4)
        amplitude = rng.uniform(-0.55, 1.0) * float(parameters["hill_amplitude"])
        elevation += amplitude * _elliptic_feature(
            x_grid,
            y_grid,
            center_x,
            center_y,
            radius_x,
            radius_y,
            rng.uniform(0.0, math.pi),
        )

    for _ in range(int(parameters["ridges"])):
        feature = _elliptic_feature(
            x_grid,
            y_grid,
            rng.uniform(-margin, margin),
            rng.uniform(-margin, margin),
            rng.uniform(4.0, 9.0),
            rng.uniform(0.55, 1.35),
            rng.uniform(0.0, math.pi),
        )
        elevation += rng.uniform(0.65, 1.0) * float(parameters["ridge_amplitude"]) * feature

    for _ in range(int(parameters["mesas"])):
        center_x = rng.uniform(-margin, margin)
        center_y = rng.uniform(-margin, margin)
        radius_x = rng.uniform(1.4, 3.5)
        radius_y = rng.uniform(1.3, 3.1)
        angle = rng.uniform(0.0, math.pi)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        dx = x_grid - center_x
        dy = y_grid - center_y
        along = (cosine * dx + sine * dy) / radius_x
        across = (-sine * dx + cosine * dy) / radius_y
        radius = np.sqrt(along * along + across * across)
        edge_softness = rng.uniform(0.07, 0.16)
        plateau = 1.0 / (1.0 + np.exp(np.clip((radius - 1.0) / edge_softness, -40, 40)))
        elevation += rng.uniform(1.3, 3.2) * plateau

    detail_noise = _normalized_noise(rng, (size, size), 0.8)
    for _ in range(int(parameters["rough_patches"])):
        patch = _elliptic_feature(
            x_grid,
            y_grid,
            rng.uniform(-margin, margin),
            rng.uniform(-margin, margin),
            rng.uniform(1.5, 4.2),
            rng.uniform(1.4, 3.8),
            rng.uniform(0.0, math.pi),
        )
        elevation += float(parameters["rough_amplitude"]) * patch * detail_noise

    # A slight global tilt avoids unnaturally level map boundaries.
    elevation += rng.uniform(-0.018, 0.018) * x_grid
    elevation += rng.uniform(-0.018, 0.018) * y_grid
    elevation = ndimage.gaussian_filter(elevation, sigma=0.55, mode="reflect")
    elevation -= elevation.min()
    return elevation


def _disk(radius: int) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1)
    x_grid, y_grid = np.meshgrid(offsets, offsets)
    return x_grid * x_grid + y_grid * y_grid <= radius * radius


def _plane_fit_geometry(
    elevation: np.ndarray,
    footprint: np.ndarray,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return support height, support-plane slope, and RMS roughness.

    Grade is estimated from one least-squares plane over the physical robot
    footprint. Local deviations from that plane remain in the roughness layer,
    so a single sub-footprint bump does not masquerade as whole-body grade.
    Every returned value is already a robot-centre property and must not be
    dilated by the footprint again.
    """

    radius = footprint.shape[0] // 2
    offsets = np.arange(-radius, radius + 1, dtype=float) * cell_size
    x_kernel, y_kernel = np.meshgrid(offsets, offsets)
    mask = footprint.astype(float)
    x_kernel *= mask
    y_kernel *= mask
    samples = float(mask.sum())
    normalized_mask = mask / samples
    mean_height = ndimage.convolve(elevation, normalized_mask, mode="reflect")
    mean_square = ndimage.convolve(
        elevation * elevation, normalized_mask, mode="reflect"
    )
    sum_x_squared = float(np.sum(x_kernel * x_kernel))
    sum_y_squared = float(np.sum(y_kernel * y_kernel))
    slope_x = ndimage.convolve(elevation, x_kernel, mode="reflect") / sum_x_squared
    slope_y = ndimage.convolve(elevation, y_kernel, mode="reflect") / sum_y_squared
    coordinate_variance_x = sum_x_squared / samples
    coordinate_variance_y = sum_y_squared / samples
    residual_variance = (
        mean_square
        - mean_height * mean_height
        - slope_x * slope_x * coordinate_variance_x
        - slope_y * slope_y * coordinate_variance_y
    )
    support_slope_degrees = np.degrees(
        np.arctan(np.hypot(slope_x, slope_y))
    )
    roughness = np.sqrt(np.maximum(residual_variance, 0.0))
    return mean_height, support_slope_degrees, roughness


def _local_step_height(elevation: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    adjacent_step = np.zeros_like(elevation)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = ndimage.shift(
                elevation,
                shift=(dy, dx),
                order=0,
                mode="nearest",
                prefilter=False,
            )
            adjacent_step = np.maximum(adjacent_step, np.abs(elevation - shifted))
    return ndimage.maximum_filter(adjacent_step, footprint=footprint, mode="nearest")


def analyze_terrain(elevation: np.ndarray, config: TerrainConfig) -> dict[str, np.ndarray]:
    """Compute pointwise and footprint-conditioned terrain layers.

    Every value used for collision classification represents a candidate
    robot-centre pose.  This avoids the former double inflation in which
    roughness and step height were computed over the footprint and then the
    resulting blocked centres were dilated by the footprint a second time.
    """

    smoothed = ndimage.gaussian_filter(elevation, sigma=0.7, mode="reflect")
    gradient_y, gradient_x = np.gradient(smoothed, config.cell_size, config.cell_size)
    slope_ratio = np.hypot(gradient_x, gradient_y)
    point_slope_degrees = np.degrees(np.arctan(slope_ratio))
    footprint = _disk(config.footprint_cells)
    (
        support_elevation,
        footprint_slope_degrees,
        roughness,
    ) = _plane_fit_geometry(
        elevation,
        footprint,
        config.cell_size,
    )
    step_height = _local_step_height(elevation, footprint)

    normalized_slope = footprint_slope_degrees / config.max_slope_degrees
    normalized_roughness = roughness / config.max_roughness
    normalized_step = step_height / config.max_step_height
    traversability_cost = (
        config.traversability_slope_weight * normalized_slope
        + config.traversability_roughness_weight * normalized_roughness
        + config.traversability_step_weight * normalized_step
    )
    blocked_by_slope = footprint_slope_degrees > config.max_slope_degrees
    blocked_by_roughness = roughness > config.max_roughness
    blocked_by_step = step_height > config.max_step_height
    blocked_by_combined_cost = traversability_cost >= 1.0
    centre_blocked = (
        blocked_by_slope
        | blocked_by_roughness
        | blocked_by_step
        | blocked_by_combined_cost
    )
    collision_grid = centre_blocked.copy()
    border = config.footprint_cells
    collision_grid[:border, :] = True
    collision_grid[-border:, :] = True
    collision_grid[:, :border] = True
    collision_grid[:, -border:] = True
    return {
        "slope_degrees": point_slope_degrees,
        "support_elevation": support_elevation,
        "footprint_slope_degrees": footprint_slope_degrees,
        "roughness": roughness,
        "step_height": step_height,
        "traversability_cost": traversability_cost,
        "blocked_by_slope": blocked_by_slope,
        "blocked_by_roughness": blocked_by_roughness,
        "blocked_by_step": blocked_by_step,
        "blocked_by_combined_cost": blocked_by_combined_cost,
        "raw_blocked": centre_blocked,
        "centre_blocked": centre_blocked,
        "collision_grid": collision_grid,
    }


def movement_cost(
    first: Cell,
    second: Cell,
    elevation: Sequence[Sequence[Optional[float]]],
    cell_size: float,
    slope_weight: float,
) -> tuple[float, float]:
    """Return weighted 2.5D cost and rise/run for one cell transition."""

    planar_cells = math.hypot(first[0] - second[0], first[1] - second[1])
    planar_distance = planar_cells * cell_size
    z0 = elevation[first[1]][first[0]]
    z1 = elevation[second[1]][second[0]]
    assert z0 is not None and z1 is not None
    dz = abs(float(z1) - float(z0))
    slope = dz / planar_distance
    spatial_distance = math.hypot(planar_distance, dz)
    return spatial_distance * (1.0 + slope_weight * slope * slope), slope


def transition_cost(
    grid: Sequence[Sequence[int]],
    elevation: Sequence[Sequence[Optional[float]]],
    first: Cell,
    second: Cell,
    max_slope: float,
    cell_size: float,
    slope_weight: float,
) -> Optional[float]:
    """Return cost for one valid 8-connected, no-corner-cutting move."""

    height = len(grid)
    width = len(grid[0]) if height else 0
    x, y = first
    nx, ny = second
    if not (
        0 <= x < width
        and 0 <= y < height
        and 0 <= nx < width
        and 0 <= ny < height
    ):
        return None
    dx = nx - x
    dy = ny - y
    if (dx == 0 and dy == 0) or max(abs(dx), abs(dy)) != 1:
        return None
    if grid[y][x] != 0 or grid[ny][nx] != 0:
        return None
    if dx and dy and (grid[y][nx] != 0 or grid[ny][x] != 0):
        return None
    cost, slope = movement_cost(
        first,
        second,
        elevation,
        cell_size,
        slope_weight,
    )
    if slope > max_slope + 1e-12:
        return None
    return cost


def _candidate_neighbors(x: int, y: int) -> Iterable[tuple[int, int]]:
    for dx, dy in (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ):
        yield x + dx, y + dy


def iter_valid_neighbors(
    cell: Cell,
    grid: np.ndarray,
    elevation: np.ndarray,
    config: TerrainConfig,
) -> Iterator[tuple[Cell, float]]:
    x, y = cell
    for nx, ny in _candidate_neighbors(x, y):
        neighbor = (nx, ny)
        cost = transition_cost(
            grid,
            elevation,
            cell,
            neighbor,
            config.max_slope,
            config.cell_size,
            config.slope_weight,
        )
        if cost is not None:
            yield neighbor, cost


def connected_components(
    grid: np.ndarray,
    elevation: np.ndarray,
    config: TerrainConfig,
) -> list[set[Cell]]:
    remaining = {
        (int(x), int(y))
        for y, x in np.argwhere(grid == 0)
    }
    components: list[set[Cell]] = []
    while remaining:
        seed = next(iter(remaining))
        remaining.remove(seed)
        component = {seed}
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbor, _ in iter_valid_neighbors(current, grid, elevation, config):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    components.sort(key=len, reverse=True)
    return components


def _dijkstra(
    start: Cell,
    allowed: set[Cell],
    grid: np.ndarray,
    elevation: np.ndarray,
    config: TerrainConfig,
) -> dict[Cell, float]:
    distances = {start: 0.0}
    queue: list[tuple[float, Cell]] = [(0.0, start)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances.get(current):
            continue
        for neighbor, edge_cost in iter_valid_neighbors(
            current, grid, elevation, config
        ):
            if neighbor not in allowed:
                continue
            candidate = distance + edge_cost
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def sample_start_goal_pairs(
    grid: np.ndarray,
    elevation: np.ndarray,
    config: TerrainConfig,
    *,
    count: int = 10,
    seed: int = 0,
) -> tuple[list[dict], list[set[Cell]]]:
    """Select deterministic short-, medium-, and long-range navigation tasks.

    Candidate distances are measured with the same weighted graph used for
    connectivity.  For the standard ten-task protocol, three pairs are drawn
    from the short band, four from the medium band, and three from the long
    band.  Other task counts are distributed proportionally across the bands.
    """

    if count < 1:
        raise ValueError("task count must be at least one")
    components = connected_components(grid, elevation, config)
    if not components or len(components[0]) < 2:
        raise ValueError("terrain has no usable traversable component")
    largest = components[0]
    margin = config.endpoint_margin_cells
    height, width = grid.shape
    endpoint_candidates = sorted(
        (
            cell
            for cell in largest
            if margin <= cell[0] < width - margin
            and margin <= cell[1] < height - margin
        ),
        key=lambda cell: (cell[1], cell[0]),
    )
    if len(endpoint_candidates) < 2:
        endpoint_candidates = sorted(largest, key=lambda cell: (cell[1], cell[0]))
    if len(endpoint_candidates) < 2:
        raise ValueError("terrain has fewer than two endpoint candidates")

    rng = np.random.default_rng(seed)
    origin_count = min(max(16, count * 3), len(endpoint_candidates))
    if origin_count == len(endpoint_candidates):
        origins = endpoint_candidates
    else:
        origin_indices = sorted(
            int(index)
            for index in rng.choice(
                len(endpoint_candidates),
                size=origin_count,
                replace=False,
            )
        )
        origins = [endpoint_candidates[index] for index in origin_indices]

    candidate_pairs: dict[tuple[Cell, Cell], float] = {}
    for origin in origins:
        distances = _dijkstra(origin, largest, grid, elevation, config)
        for target in endpoint_candidates:
            if target == origin or target not in distances:
                continue
            first, second = sorted((origin, target))
            pair_key = (first, second)
            candidate_pairs[pair_key] = distances[target]
    ranked = sorted(
        (
            (distance, pair[0], pair[1])
            for pair, distance in candidate_pairs.items()
        ),
        key=lambda item: (item[0], item[1][1], item[1][0], item[2][1], item[2][0]),
    )
    if len(ranked) < count:
        raise ValueError(f"only {len(ranked)} distinct endpoint pairs are available")

    if count == 10:
        band_counts = {"short": 3, "medium": 4, "long": 3}
    else:
        short_count = max(1, round(count * 0.3))
        long_count = max(1, round(count * 0.3)) if count > 1 else 0
        if short_count + long_count > count:
            long_count = max(0, count - short_count)
        band_counts = {
            "short": short_count,
            "medium": count - short_count - long_count,
            "long": long_count,
        }
    quantile_ranges = {
        "short": (0.20, 0.42),
        "medium": (0.45, 0.72),
        "long": (0.75, 0.98),
    }
    selected: list[tuple[str, float, Cell, Cell]] = []
    used_pairs: set[tuple[Cell, Cell]] = set()
    for distance_class in ("short", "medium", "long"):
        band_count = band_counts[distance_class]
        if band_count == 0:
            continue
        lower, upper = quantile_ranges[distance_class]
        lower_index = min(len(ranked) - 1, int(lower * (len(ranked) - 1)))
        upper_index = min(len(ranked), max(lower_index + 1, int(upper * len(ranked))))
        band = ranked[lower_index:upper_index]
        if len(band) < band_count:
            band = ranked
        positions = np.linspace(0, len(band) - 1, band_count + 2)[1:-1]
        for position in positions:
            start_index = int(round(float(position)))
            for offset in range(len(band)):
                distance, first, second = band[(start_index + offset) % len(band)]
                ordered_first, ordered_second = sorted((first, second))
                pair_key = (ordered_first, ordered_second)
                if pair_key in used_pairs:
                    continue
                used_pairs.add(pair_key)
                selected.append((distance_class, distance, first, second))
                break

    if len(selected) != count:
        for distance, first, second in reversed(ranked):
            ordered_first, ordered_second = sorted((first, second))
            pair_key = (ordered_first, ordered_second)
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)
            selected.append(("long", distance, first, second))
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(f"could select only {len(selected)} of {count} endpoint pairs")

    tasks = []
    canonical_index = max(range(len(selected)), key=lambda index: selected[index][1])
    for index, (distance_class, distance, start, goal) in enumerate(selected, start=1):
        tasks.append(
            {
                "task_index": index,
                "start": [int(start[0]), int(start[1])],
                "goal": [int(goal[0]), int(goal[1])],
                "distance_class": distance_class,
                "reference_weighted_distance": round(float(distance), 4),
                "canonical_visualization": index - 1 == canonical_index,
            }
        )
    return tasks, components
