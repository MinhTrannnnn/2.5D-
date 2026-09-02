#!/usr/bin/env python3
"""Build the Zenodo core/NPZ archives and refresh the archive checksum list."""

from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path
from typing import Optional, Sequence


EXCLUDED_NAMES = {
    ".DS_Store",
    "analysis_state.json",
    "generation_progress.jsonl",
    "generation_state.json",
    "map_images_full_v1_progress.jsonl",
    "map_images_full_v1_state.json",
    "path_preview_summary.csv",
    "pathfinding_benchmark_v1_progress.jsonl",
    "pathfinding_benchmark_v1_state.json",
}
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _core_files(dataset_root: Path) -> list[Path]:
    paths = [
        dataset_root / "LICENSE",
        dataset_root / "README.md",
        dataset_root / "SCHEMA.md",
    ]
    paths.extend((dataset_root / "examples").glob("*"))
    paths.extend((dataset_root / "images" / "contact_sheets").glob("*"))
    paths.extend((dataset_root / "metadata").glob("*"))
    paths.extend((dataset_root / "metadata" / "splits").glob("*"))
    paths.extend((dataset_root / "metadata" / "benchmark_results").rglob("*"))
    selected = sorted(
        {
            path
            for path in paths
            if path.is_file() and path.name not in EXCLUDED_NAMES
        },
        key=lambda path: path.relative_to(dataset_root.parent).as_posix(),
    )
    if not selected:
        raise ValueError("core archive file list is empty")
    return selected


def _stored_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def package(
    dataset_root: Path, archive_root: Path, version: str
) -> tuple[Path, Path, Path]:
    archive_root.mkdir(parents=True, exist_ok=True)
    core_archive = archive_root / f"paired-25d-{version}-core.zip"
    temporary_archive = core_archive.with_name(f".{core_archive.name}.{os.getpid()}.tmp")
    temporary_archive.unlink(missing_ok=True)
    with zipfile.ZipFile(
        temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in _core_files(dataset_root):
            archive.write(path, path.relative_to(dataset_root.parent).as_posix())
    os.replace(temporary_archive, core_archive)

    npz_paths = sorted((dataset_root / "npz").glob("*/*.npz"))
    if len(npz_paths) != 5010:
        raise ValueError(f"expected 5010 NPZ companions, found {len(npz_paths)}")
    npz_metadata = [
        dataset_root / "metadata" / "npz_conversion.json",
        dataset_root / "metadata" / "npz_manifest.csv",
        dataset_root / "metadata" / "SHA256SUMS_NPZ.txt",
    ]
    for path in npz_metadata:
        if not path.is_file():
            raise FileNotFoundError(path)
    npz_archive = archive_root / f"paired-25d-{version}-npz.zip"
    npz_temporary = npz_archive.with_name(f".{npz_archive.name}.{os.getpid()}.tmp")
    npz_temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(npz_temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in [*npz_metadata, *npz_paths]:
            relative = path.relative_to(dataset_root.parent).as_posix()
            archive.writestr(
                _stored_zip_info(relative),
                path.read_bytes(),
                compress_type=zipfile.ZIP_STORED,
            )
    os.replace(npz_temporary, npz_archive)

    expected_archives = [
        archive_root / f"paired-25d-{version}-{suffix}.zip"
        for suffix in (
            "core",
            "npz",
            "mountain",
            "plateau",
            "rolling",
            "rugged",
            "smooth-obstacles",
        )
    ]
    for path in expected_archives:
        if not path.is_file():
            raise FileNotFoundError(path)

    checksum_path = archive_root / "SHA256SUMS.txt"
    checksum_temporary = checksum_path.with_name(
        f".{checksum_path.name}.{os.getpid()}.tmp"
    )
    checksum_temporary.write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in expected_archives)
        + "\n",
        encoding="utf-8",
    )
    os.replace(checksum_temporary, checksum_path)
    return core_archive, npz_archive, checksum_path


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=project_root / "dataset_5010_v1"
    )
    parser.add_argument(
        "--archives", type=Path, default=project_root / "release_archives"
    )
    parser.add_argument("--version", default="v1.0.0")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    core_archive, npz_archive, checksum_path = package(
        args.dataset.resolve(), args.archives.resolve(), args.version
    )
    print(core_archive)
    print(npz_archive)
    print(checksum_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
