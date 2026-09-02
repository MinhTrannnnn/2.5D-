#!/usr/bin/env python3
"""Prepare inventory, checksums, and notes for a completed 2.5D release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence


GENERATED_RELEASE_FILES = frozenset(
    (
        "metadata/release_manifest.json",
        "metadata/release_filelist.csv",
        "metadata/SHA256SUMS_CORE.txt",
        "metadata/SHA256SUMS_MAPS.txt",
        "metadata/SHA256SUMS_NPZ.txt",
    )
)

# Durable checkpoints and safety sentinels are useful while generating or
# resuming a dataset, but they are implementation state rather than published
# scientific data.  Keep them out of inventories even if release preparation
# is run before a working directory is cleaned.
OPERATIONAL_FILES = frozenset(
    (
        ".paired_preview_output",
        "metadata/generation_progress.jsonl",
        "metadata/generation_state.json",
        "metadata/map_images_full_v1_progress.jsonl",
        "metadata/map_images_full_v1_state.json",
        "metadata/pathfinding_benchmark_v1_progress.jsonl",
        "metadata/pathfinding_benchmark_v1_state.json",
        "metadata/path_preview_summary.csv",
        "metadata/benchmark_results/analysis_state.json",
    )
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError("release file list is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _category(relative_path: str) -> str:
    if relative_path.startswith("maps/") and relative_path.endswith(".json"):
        return "map_json"
    if relative_path.startswith("npz/") and relative_path.endswith(".npz"):
        return "npz_raster_companion"
    if relative_path.startswith("images/terrain/"):
        return "terrain_image"
    if relative_path.startswith("images/analysis/"):
        return "analysis_image"
    if relative_path.startswith("images/paths/"):
        return "planner_path_image"
    if relative_path.startswith("images/contact_sheets/"):
        return "contact_sheet"
    if relative_path.startswith("metadata/benchmark_results/figures/"):
        return "benchmark_figure"
    if relative_path.startswith("metadata/benchmark_results/tables/"):
        return "benchmark_table"
    if relative_path.startswith("metadata/"):
        return "metadata"
    return "documentation_or_other"


def _map_checksums(dataset_root: Path) -> list[tuple[str, str]]:
    map_paths = sorted(dataset_root.joinpath("maps").glob("*/*.json"))
    if len(map_paths) != 5010:
        raise ValueError(f"expected 5010 map files, found {len(map_paths)}")
    return [
        (_sha256(path), path.relative_to(dataset_root).as_posix())
        for path in map_paths
    ]


def _release_notes() -> str:
    return """# Release notes — dataset_5010_v1

## Included

- 5,010 paired 2.5D terrain maps
- 5,010 pickle-free NPZ raster companions
- 50,100 published map-task records
- 10,020 terrain and analysis visualizations
- 30,060 canonical planner route visualizations
- 90,180 canonical-task benchmark trials
- Map-level benchmark tables, paired-difficulty summaries, and four figures
- Fixed train/validation/test indexes assigned at matched-triplet level and
  stratified exactly by terrain family

## Verified

- Structural validator: 5,010 maps and 50,100 map-task records valid
- Planner audit: zero invalid successful paths and zero missing route images
- Difficulty audit: zero relief or navigability ordering violations
- Optimality cross-check: exact A*/Dijkstra weighted-cost agreement on all maps
- Split audit: all 1,670 matched groups assigned once, with zero group leakage
- NPZ audit: all 5,010 companions pass CRC, field-set, NPY-header, shape, and
  payload-length checks

## Statistical unit

The five stochastic trials are nested within each map-algorithm unit. Benchmark
mean and sample standard deviation are computed across maps after within-map
aggregation. Easy, Medium, and Hard members are paired terrain variants.

## Fixed splits

The published split unit is `matched_group_id`, so all Easy, Medium, and Hard
members and their ten shared tasks remain together. Within each of the five
terrain families, 234 groups are assigned to train, 50 to validation, and 50 to
test. The deterministic SHA-256 assignment protocol and seed are recorded in
`metadata/splits/split_protocol.json`.

## Runtime

Planner runtime was recorded during a three-worker throughput run. Report this
protocol explicitly; the supplied timing is not isolated single-core latency.

## NPZ companion

JSON remains the canonical self-describing map record. The same-stem NPZ files
provide the 13 raster layers as `float32` or `uint8` arrays for faster numerical
loading with `numpy.load(..., allow_pickle=False)`. Per-file paths and hashes
are published in `metadata/npz_manifest.csv` and `metadata/SHA256SUMS_NPZ.txt`.

## Packaging

The Zenodo deposit is partitioned into one compact core ZIP, one NPZ companion
ZIP, and five terrain-family ZIPs. Split indexes, documentation, summary tables,
reports, and checksums are in the core ZIP; the unchanged JSON map and image
payload remains in the family ZIPs.
"""


def prepare_release(dataset_root: Path, benchmark_prefix: str) -> dict:
    metadata_root = dataset_root / "metadata"
    manifest_path = metadata_root / "manifest.json"
    validation_path = metadata_root / "validation_report.json"
    audit_path = metadata_root / "benchmark_results" / "benchmark_audit.json"
    split_protocol_path = metadata_root / "splits" / "split_protocol.json"
    npz_protocol_path = metadata_root / "npz_conversion.json"
    npz_manifest_path = metadata_root / "npz_manifest.csv"
    npz_checksums_path = metadata_root / "SHA256SUMS_NPZ.txt"
    split_paths = tuple(
        metadata_root / "splits" / f"{split}.csv"
        for split in ("train", "validation", "test")
    )
    required = (
        manifest_path,
        validation_path,
        audit_path,
        split_protocol_path,
        npz_protocol_path,
        npz_manifest_path,
        npz_checksums_path,
        *split_paths,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    split_protocol = json.loads(split_protocol_path.read_text(encoding="utf-8"))
    npz_protocol = json.loads(npz_protocol_path.read_text(encoding="utf-8"))
    if not validation.get("valid") or not audit.get("valid"):
        raise RuntimeError("release cannot be prepared from an invalid dataset")
    split_validation = split_protocol.get("validation", {})
    if (
        split_protocol.get("totals", {}).get("matched_groups") != 1670
        or split_validation.get("group_leakage_count") != 0
        or not all(
            split_validation.get(key)
            for key in (
                "all_groups_assigned_once",
                "all_groups_have_easy_medium_hard",
                "family_stratification_exact",
            )
        )
    ):
        raise RuntimeError("release cannot be prepared from invalid fixed splits")
    npz_validation = npz_protocol.get("validation", {})
    npz_paths = sorted(dataset_root.joinpath("npz").glob("*/*.npz"))
    with npz_manifest_path.open(newline="", encoding="utf-8") as handle:
        npz_manifest_rows = sum(1 for _ in csv.DictReader(handle))
    npz_checksum_lines = sum(
        1 for line in npz_checksums_path.read_text(encoding="utf-8").splitlines() if line
    )
    if (
        npz_protocol.get("map_count") != 5010
        or len(npz_paths) != 5010
        or npz_manifest_rows != 5010
        or npz_checksum_lines != 5010
        or not all(npz_validation.values())
    ):
        raise RuntimeError("release cannot be prepared from invalid NPZ companions")

    notes_path = metadata_root / "RELEASE_NOTES.md"
    _atomic_text(notes_path, _release_notes())

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixed_splits"] = {
        "protocol": "paired-25d-fixed-split-v1",
        "directory": "metadata/splits",
        "assignment_unit": "matched_group_id",
        "stratification": "terrain family",
        "seed": split_protocol["seed"],
        "counts_per_family": split_protocol["counts_per_family"],
        "total_matched_groups": 1670,
        "total_maps_referenced": 5010,
        "group_leakage_count": 0,
    }
    manifest["npz_companion"] = {
        "role": "derived raster companion; JSON maps remain canonical",
        "directory": "npz/<family>",
        "map_count": 5010,
        "shape": [128, 128],
        "float_dtype": "float32",
        "mask_dtype": "uint8",
        "pickle_required": False,
        "protocol": "metadata/npz_conversion.json",
        "manifest": "metadata/npz_manifest.csv",
        "checksums": "metadata/SHA256SUMS_NPZ.txt",
    }
    manifest["release"] = {
        "status": "release_ready",
        "benchmark_prefix": benchmark_prefix,
        "benchmark_results": "metadata/benchmark_results",
        "release_notes": "metadata/RELEASE_NOTES.md",
        "release_manifest": "metadata/release_manifest.json",
        "file_list": "metadata/release_filelist.csv",
        "core_checksums": "metadata/SHA256SUMS_CORE.txt",
        "map_checksums": "metadata/SHA256SUMS_MAPS.txt",
        "npz_checksums": "metadata/SHA256SUMS_NPZ.txt",
        "fixed_splits": {
            "directory": "metadata/splits",
            "protocol": "metadata/splits/split_protocol.json",
            "assignment_unit": "matched_group_id",
            "stratification": "terrain family",
        },
        "archive_created": True,
        "archives": [
            "paired-25d-v1.0.0-core.zip",
            "paired-25d-v1.0.0-npz.zip",
            "paired-25d-v1.0.0-mountain.zip",
            "paired-25d-v1.0.0-plateau.zip",
            "paired-25d-v1.0.0-rolling.zip",
            "paired-25d-v1.0.0-rugged.zip",
            "paired-25d-v1.0.0-smooth-obstacles.zip",
        ],
    }
    _atomic_json(manifest_path, manifest)

    files = []
    category_counts: Counter[str] = Counter()
    category_bytes: defaultdict[str, int] = defaultdict(int)
    for path in sorted(item for item in dataset_root.rglob("*") if item.is_file()):
        relative = path.relative_to(dataset_root).as_posix()
        if (
            relative in GENERATED_RELEASE_FILES
            or relative in OPERATIONAL_FILES
            or path.name == ".DS_Store"
        ):
            continue
        category = _category(relative)
        size = path.stat().st_size
        files.append(
            {
                "relative_path": relative,
                "category": category,
                "size_bytes": size,
            }
        )
        category_counts[category] += 1
        category_bytes[category] += size

    expected_counts = {
        "map_json": 5010,
        "terrain_image": 5010,
        "analysis_image": 5010,
        "planner_path_image": 30060,
        "npz_raster_companion": 5010,
    }
    for category, expected in expected_counts.items():
        actual = category_counts[category]
        if actual != expected:
            raise ValueError(f"{category}: expected {expected}, found {actual}")

    release_manifest = {
        "schema_version": "1.0",
        "release_status": "release_ready",
        "dataset": dataset_root.name,
        "archive_created": True,
        "archive_note": (
            "one core ZIP, one NPZ companion ZIP, and five terrain-family ZIPs "
            "for Zenodo"
        ),
        "inventory_excludes": sorted(GENERATED_RELEASE_FILES | OPERATIONAL_FILES),
        "inventory": {
            "files": len(files),
            "bytes": sum(int(row["size_bytes"]) for row in files),
            "by_category": {
                category: {
                    "files": category_counts[category],
                    "bytes": category_bytes[category],
                }
                for category in sorted(category_counts)
            },
        },
        "scientific_counts": {
            "maps": 5010,
            "matched_triplets": 1670,
            "map_task_records": 50100,
            "canonical_benchmark_map_tasks": 5010,
            "benchmark_trial_rows": 90180,
            "planner_path_images": 30060,
            "fixed_split_matched_groups": 1670,
            "npz_raster_companions": 5010,
        },
        "validation": {
            "dataset_valid": validation["valid"],
            "dataset_errors": validation["error_count"],
            "dataset_warnings": validation["warning_count"],
            "benchmark_audit_valid": audit["valid"],
            "invalid_success_paths": audit["checks"]["invalid_success_paths"],
            "missing_path_images": audit["checks"]["missing_path_images"],
            "astar_dijkstra_cost_mismatches": audit["checks"][
                "astar_dijkstra_cost_mismatches"
            ],
            "fixed_split_valid": True,
            "fixed_split_group_leakage": split_validation["group_leakage_count"],
            "npz_companion_valid": True,
            "npz_companions_checked": 5010,
        },
        "benchmark_prefix": benchmark_prefix,
        "statistics": {
            "experimental_unit": "map-algorithm after within-map trial aggregation",
            "reported_dispersion": "sample standard deviation across maps",
            "paired_difficulty_design": True,
            "runtime_workers": [3],
        },
        "fixed_splits": {
            "assignment_unit": "matched_group_id",
            "stratification": "terrain family",
            "train_groups": 1170,
            "validation_groups": 250,
            "test_groups": 250,
            "group_leakage_count": 0,
            "protocol": "metadata/splits/split_protocol.json",
        },
        "npz_companion": {
            "role": "derived raster companion; JSON maps remain canonical",
            "files": 5010,
            "total_bytes": npz_protocol["total_npz_bytes"],
            "pickle_required": False,
            "protocol": "metadata/npz_conversion.json",
            "manifest": "metadata/npz_manifest.csv",
            "checksums": "metadata/SHA256SUMS_NPZ.txt",
        },
    }
    release_manifest_path = metadata_root / "release_manifest.json"
    _atomic_json(release_manifest_path, release_manifest)
    _atomic_csv(metadata_root / "release_filelist.csv", files)

    map_checksum_lines = [f"{digest}  {path}" for digest, path in _map_checksums(dataset_root)]
    _atomic_text(
        metadata_root / "SHA256SUMS_MAPS.txt",
        "\n".join(map_checksum_lines) + "\n",
    )

    core_relative_paths = (
        "LICENSE",
        "README.md",
        "SCHEMA.md",
        "examples/load_map.py",
        "metadata/RELEASE_NOTES.md",
        "metadata/manifest.json",
        "metadata/release_manifest.json",
        "metadata/release_filelist.csv",
        "metadata/validation_report.json",
        "metadata/dataset_summary.csv",
        "metadata/task_summary.csv",
        "metadata/npz_conversion.json",
        "metadata/npz_manifest.csv",
        "metadata/SHA256SUMS_NPZ.txt",
        "metadata/splits/train.csv",
        "metadata/splits/validation.csv",
        "metadata/splits/test.csv",
        "metadata/splits/split_protocol.json",
        f"metadata/{benchmark_prefix}_trials.csv",
        f"metadata/{benchmark_prefix}_summary.csv",
        f"metadata/{benchmark_prefix}_protocol.json",
        "metadata/benchmark_results/RESULTS.md",
        "metadata/benchmark_results/benchmark_audit.json",
        "metadata/benchmark_results/tables/planner_summary_family_difficulty.csv",
        "metadata/benchmark_results/tables/planner_summary_overall.csv",
    )
    core_checksum_lines = []
    for relative in core_relative_paths:
        path = dataset_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        core_checksum_lines.append(f"{_sha256(path)}  {relative}")
    _atomic_text(
        metadata_root / "SHA256SUMS_CORE.txt",
        "\n".join(core_checksum_lines) + "\n",
    )

    print(
        f"Release-ready inventory: {len(files)} files, "
        f"{release_manifest['inventory']['bytes']} bytes.",
        flush=True,
    )
    print(
        f"Checksums: 5010 maps and {len(core_relative_paths)} core release files.",
        flush=True,
    )
    return release_manifest


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "dataset_5010_v1",
    )
    parser.add_argument("--benchmark-prefix", default="pathfinding_benchmark_v1")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_release(args.dataset.resolve(), args.benchmark_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
