#!/usr/bin/env python3
"""Run the standardized multi-trial benchmark for stochastic 2.5D planners."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from benchmark_pathplanning import run_benchmark


STOCHASTIC_ALGORITHMS = ("PRM", "RRT_Connect", "RRT_Star")


def run_extended_stochastic_benchmark(
    dataset_root: Path,
    trials: int = 5,
    *,
    prm_samples: int = 500,
    prm_neighbors: int = 12,
    rrt_connect_iterations: int = 5000,
    rrt_star_iterations: int = 6000,
    workers: int = 1,
    resume: bool = False,
) -> None:
    """Compatibility entry point backed by the common benchmark protocol."""

    run_benchmark(
        dataset_root,
        algorithm_names=STOCHASTIC_ALGORITHMS,
        stochastic_trials=trials,
        prm_samples=prm_samples,
        prm_neighbors=prm_neighbors,
        rrt_connect_iterations=rrt_connect_iterations,
        rrt_star_iterations=rrt_star_iterations,
        output_prefix="stochastic_pathfinding",
        workers=workers,
        resume=resume,
    )


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "dataset_preview",
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--prm-samples", type=int, default=500)
    parser.add_argument("--prm-neighbors", type=int, default=12)
    parser.add_argument("--rrt-connect-iterations", type=int, default=5000)
    parser.add_argument("--rrt-star-iterations", type=int, default=6000)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, os.cpu_count() or 1)),
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.trials < 1:
        raise ValueError("trials must be at least one")
    run_extended_stochastic_benchmark(
        args.dataset.resolve(),
        args.trials,
        prm_samples=args.prm_samples,
        prm_neighbors=args.prm_neighbors,
        rrt_connect_iterations=args.rrt_connect_iterations,
        rrt_star_iterations=args.rrt_star_iterations,
        workers=args.workers,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
