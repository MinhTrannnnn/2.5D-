#!/usr/bin/env python3
"""Render terrain and traversability PNGs for every existing 2.5D map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

from generate_25d_dataset import _render_map_images_worker


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_progress_map_files(path: Path) -> set[str]:
    map_files: set[str] = set()
    if not path.is_file():
        return map_files
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if any(item.strip() for item in handle):
                    raise ValueError(
                        f"invalid image checkpoint line {line_number}: {path}"
                    )
                break
            map_files.add(record["map_file"])
    return map_files


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _output_paths(dataset_root: Path, map_path: Path) -> tuple[Path, Path]:
    family = map_path.parent.name
    terrain_path = dataset_root / "images" / "terrain" / family / f"{map_path.stem}.png"
    analysis_path = (
        dataset_root
        / "images"
        / "analysis"
        / family
        / f"{map_path.stem}_analysis.png"
    )
    return terrain_path, analysis_path


def _signature(map_files: Sequence[str]) -> dict:
    return {
        "schema_version": "1.0",
        "renderer_revision": "terrain-analysis-natural-accessible-v1",
        "map_files": list(map_files),
        "outputs": ("terrain", "analysis"),
    }


def _signature_hash(signature: dict) -> str:
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _update_dataset_indexes(dataset_root: Path, total_maps: int) -> None:
    summary_path = dataset_root / "metadata" / "dataset_summary.csv"
    if summary_path.is_file():
        with summary_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames
        if fieldnames and rows:
            for row in rows:
                family = row["family"]
                terrain_id = row["instance_id"]
                row["terrain_image_file"] = (
                    f"images/terrain/{family}/{terrain_id}.png"
                )
                row["traversability_image_file"] = (
                    f"images/analysis/{family}/{terrain_id}_analysis.png"
                )
            temporary = summary_path.with_name(
                f".{summary_path.name}.{os.getpid()}.tmp"
            )
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, summary_path)

    manifest_path = dataset_root / "metadata" / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["representative_images_only"] = False
        manifest["map_image_scope"] = "all_maps"
        manifest["full_map_images"] = {
            "terrain_count": total_maps,
            "analysis_count": total_maps,
            "renderer": "render_25d_dataset_images.py",
            "resume_supported": True,
        }
        _atomic_write_json(manifest, manifest_path)


def render_dataset_images(
    dataset_root: Path,
    *,
    workers: int,
    output_prefix: str,
    max_maps: Optional[int] = None,
    resume: bool = False,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be at least one")
    map_paths = sorted(dataset_root.joinpath("maps").rglob("terrain_*.json"))
    if max_maps is not None:
        map_paths = map_paths[:max_maps]
    if not map_paths:
        raise FileNotFoundError(f"no terrain maps found below {dataset_root / 'maps'}")

    metadata_root = dataset_root / "metadata"
    state_path = metadata_root / f"{output_prefix}_state.json"
    progress_path = metadata_root / f"{output_prefix}_progress.jsonl"
    map_files = [path.relative_to(dataset_root).as_posix() for path in map_paths]
    signature = _signature(map_files)
    signature_hash = _signature_hash(signature)

    if resume:
        if not state_path.is_file():
            raise ValueError(f"no image-render state to resume: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature_hash") != signature_hash:
            raise ValueError(
                "resume image-render configuration differs from the original run"
            )
    else:
        existing = [path for path in (state_path, progress_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"image-render output already exists: {existing[0]}; use "
                "--resume or choose another --output-prefix"
            )
        state = {
            "schema_version": "1.0",
            "status": "in_progress",
            "signature_hash": signature_hash,
            "signature": signature,
            "workers_used": [],
        }
        _atomic_write_json(state, state_path)

    logged_map_files = _load_progress_map_files(progress_path)
    complete_before = 0
    jobs = []
    for map_path, map_file in zip(map_paths, map_files):
        terrain_path, analysis_path = _output_paths(dataset_root, map_path)
        if _nonempty(terrain_path) and _nonempty(analysis_path):
            complete_before += 1
            if map_file not in logged_map_files:
                _append_progress(
                    progress_path,
                    {
                        "record_type": "completed_map_images",
                        "map_file": map_file,
                        "terrain_image_file": terrain_path.relative_to(
                            dataset_root
                        ).as_posix(),
                        "analysis_image_file": analysis_path.relative_to(
                            dataset_root
                        ).as_posix(),
                        "reconciled_existing_output": True,
                    },
                )
                logged_map_files.add(map_file)
            continue
        jobs.append(
            {
                "map_file": map_file,
                "map_path": str(map_path),
                "terrain_path": str(terrain_path),
                "analysis_path": str(analysis_path),
                "force": False,
            }
        )

    if workers not in state.setdefault("workers_used", []):
        state["workers_used"].append(workers)
    state["status"] = "in_progress"
    state["completed_maps"] = complete_before
    _atomic_write_json(state, state_path)

    completed = complete_before
    rendered_images = 0

    def commit(job: dict, result: dict) -> None:
        nonlocal completed, rendered_images
        terrain_path = Path(job["terrain_path"])
        analysis_path = Path(job["analysis_path"])
        if not (_nonempty(terrain_path) and _nonempty(analysis_path)):
            raise RuntimeError(f"image rendering incomplete for {job['map_file']}")
        if job["map_file"] not in logged_map_files:
            _append_progress(
                progress_path,
                {
                    "record_type": "completed_map_images",
                    "map_file": job["map_file"],
                    "terrain_image_file": terrain_path.relative_to(
                        dataset_root
                    ).as_posix(),
                    "analysis_image_file": analysis_path.relative_to(
                        dataset_root
                    ).as_posix(),
                },
            )
            logged_map_files.add(job["map_file"])
        completed += 1
        rendered_images += int(result["rendered"])
        if completed == len(map_paths) or completed % 25 == 0:
            print(
                f"Rendered map images {completed}/{len(map_paths)}: "
                f"{result['instance_id']}",
                flush=True,
            )

    try:
        if workers == 1:
            for job in jobs:
                commit(job, _render_map_images_worker(job))
        elif jobs:
            executor = ProcessPoolExecutor(max_workers=workers)
            futures = {
                executor.submit(_render_map_images_worker, job): job for job in jobs
            }
            try:
                for future in as_completed(futures):
                    commit(futures[future], future.result())
            except BaseException as error:
                if isinstance(error, KeyboardInterrupt):
                    print(
                        "\nStop requested; waiting for active image workers...",
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
        state["completed_maps"] = completed
        state["rendered_images_this_invocation"] = rendered_images
        _atomic_write_json(state, state_path)
        raise

    state["status"] = "complete"
    state["completed_maps"] = len(map_paths)
    state["rendered_images_this_invocation"] = rendered_images
    _atomic_write_json(state, state_path)
    if max_maps is None:
        _update_dataset_indexes(dataset_root, len(map_paths))
    print(
        f"Completed terrain and analysis images for {len(map_paths)} maps.",
        flush=True,
    )
    return state


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "dataset_preview",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, os.cpu_count() or 1)),
    )
    parser.add_argument("--output-prefix", default="map_images_full_v1")
    parser.add_argument("--max-maps", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        render_dataset_images(
            args.dataset.resolve(),
            workers=args.workers,
            output_prefix=args.output_prefix,
            max_maps=args.max_maps,
            resume=args.resume,
        )
    except KeyboardInterrupt:
        print(
            "Stopped safely. Run the identical command with --resume to continue.",
            flush=True,
        )
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
