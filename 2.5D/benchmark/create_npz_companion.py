#!/usr/bin/env python3
"""Create deterministic, pickle-free NPZ raster companions for every JSON map."""

from __future__ import annotations

import argparse
import array
import ast
import csv
import hashlib
import itertools
import json
import os
import struct
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Sequence


FLOAT_FIELDS = (
    "elevation",
    "point_slope_degrees",
    "support_elevation",
    "footprint_slope_degrees",
    "roughness",
    "step_height",
    "traversability_cost",
)
UINT8_FIELDS = (
    "grid",
    "blocked_by_slope",
    "blocked_by_roughness",
    "blocked_by_step",
    "blocked_by_combined_cost",
    "centre_blocked",
)
ALL_FIELDS = FLOAT_FIELDS + UINT8_FIELDS
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npy_payload(values: list, dtype: str, size: int) -> bytes:
    if len(values) != size or any(len(row) != size for row in values):
        raise ValueError(f"expected a {size}x{size} raster")
    flattened = itertools.chain.from_iterable(values)
    if dtype == "uint8":
        body = bytes(flattened)
        descriptor = "|u1"
    elif dtype == "float32":
        encoded = array.array("f", flattened)
        if sys.byteorder != "little":
            encoded.byteswap()
        body = encoded.tobytes()
        descriptor = "<f4"
    else:
        raise ValueError(dtype)

    header = str(
        {
            "descr": descriptor,
            "fortran_order": False,
            "shape": (size, size),
        }
    )
    padding = (16 - ((10 + len(header) + 1) % 16)) % 16
    header = (header + " " * padding + "\n").encode("latin-1")
    if len(header) > 65535:
        raise ValueError("NumPy v1 header is unexpectedly large")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + body


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _validate_npy(payload: bytes, dtype: str, size: int) -> None:
    if payload[:8] != b"\x93NUMPY\x01\x00":
        raise ValueError("invalid NumPy v1 magic/version")
    header_length = struct.unpack("<H", payload[8:10])[0]
    header = ast.literal_eval(payload[10 : 10 + header_length].decode("latin-1"))
    expected_descriptor = "|u1" if dtype == "uint8" else "<f4"
    if header != {
        "descr": expected_descriptor,
        "fortran_order": False,
        "shape": (size, size),
    }:
        raise ValueError(f"unexpected NPY header: {header}")
    item_size = 1 if dtype == "uint8" else 4
    if len(payload) - 10 - header_length != size * size * item_size:
        raise ValueError("NPY payload length does not match its declared shape")


def _validate_npz(path: Path, size: int) -> None:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"corrupt NPZ member in {path}")
        expected = {f"{field}.npy" for field in ALL_FIELDS}
        if set(archive.namelist()) != expected:
            raise ValueError(f"unexpected NPZ fields in {path}")
        for field in ALL_FIELDS:
            dtype = "float32" if field in FLOAT_FIELDS else "uint8"
            _validate_npy(archive.read(f"{field}.npy"), dtype, size)


def _convert_one(arguments: tuple[str, str]) -> dict[str, object]:
    source_value, output_value = arguments
    source = Path(source_value)
    output = Path(output_value)
    source_bytes = source.read_bytes()
    record = json.loads(source_bytes)
    size = int(record["size"])
    if size != 128:
        raise ValueError(f"{source}: expected size 128, found {size}")

    analysis = record["terrain_analysis"]
    values = {"elevation": record["elevation"], "grid": record["grid"]}
    for field in ALL_FIELDS:
        if field not in values:
            values[field] = analysis[field]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for field in ALL_FIELDS:
            dtype = "float32" if field in FLOAT_FIELDS else "uint8"
            archive.writestr(
                _zip_info(f"{field}.npy"),
                _npy_payload(values[field], dtype, size),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
    os.replace(temporary, output)
    _validate_npz(output, size)
    return {
        "instance_id": record["instance_id"],
        "family": record["terrain_family"],
        "difficulty": record["difficulty"],
        "json_file": source.as_posix(),
        "npz_file": output.as_posix(),
        "json_size_bytes": len(source_bytes),
        "npz_size_bytes": output.stat().st_size,
        "json_sha256": _sha256_bytes(source_bytes),
        "npz_sha256": _sha256(output),
    }


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
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


def create_npz_companion(dataset_root: Path, workers: int) -> dict[str, object]:
    map_paths = sorted((dataset_root / "maps").glob("*/*.json"))
    if len(map_paths) != 5010:
        raise ValueError(f"expected 5010 JSON maps, found {len(map_paths)}")
    output_root = dataset_root / "npz"
    arguments = [
        (
            path.as_posix(),
            (output_root / path.parent.name / f"{path.stem}.npz").as_posix(),
        )
        for path in map_paths
    ]
    rows: list[dict[str, object]] = []
    if workers == 1:
        converted = map(_convert_one, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        converted = executor.map(_convert_one, arguments)
    try:
        for index, row in enumerate(converted, start=1):
            row["json_file"] = Path(str(row["json_file"])).relative_to(
                dataset_root
            ).as_posix()
            row["npz_file"] = Path(str(row["npz_file"])).relative_to(
                dataset_root
            ).as_posix()
            rows.append(row)
            if index % 250 == 0 or index == len(arguments):
                print(f"NPZ companions: {index}/{len(arguments)}", flush=True)
    finally:
        if workers != 1:
            executor.shutdown()

    expected_outputs = {Path(output) for _, output in arguments}
    actual_outputs = set(output_root.glob("*/*.npz"))
    if actual_outputs != expected_outputs:
        raise ValueError("NPZ output inventory does not match the JSON map inventory")

    metadata_root = dataset_root / "metadata"
    manifest_path = metadata_root / "npz_manifest.csv"
    _atomic_csv(manifest_path, rows)
    checksum_path = metadata_root / "SHA256SUMS_NPZ.txt"
    _atomic_text(
        checksum_path,
        "\n".join(f"{row['npz_sha256']}  {row['npz_file']}" for row in rows)
        + "\n",
    )
    total_json_bytes = sum(int(row["json_size_bytes"]) for row in rows)
    total_npz_bytes = sum(int(row["npz_size_bytes"]) for row in rows)
    protocol: dict[str, object] = {
        "schema_version": "1.0",
        "format": "NumPy NPZ (ZIP_DEFLATED collection of NPY v1 arrays)",
        "role": "derived raster companion; JSON maps remain canonical",
        "generator": "create_npz_companion.py",
        "map_count": len(rows),
        "shape": [128, 128],
        "float32_fields": list(FLOAT_FIELDS),
        "uint8_fields": list(UINT8_FIELDS),
        "pickle_required": False,
        "recommended_reader": "numpy.load(path, allow_pickle=False)",
        "total_json_bytes": total_json_bytes,
        "total_npz_bytes": total_npz_bytes,
        "npz_to_json_size_ratio": total_npz_bytes / total_json_bytes,
        "manifest": "metadata/npz_manifest.csv",
        "checksums": "metadata/SHA256SUMS_NPZ.txt",
        "validation": {
            "all_json_maps_converted": True,
            "all_npz_archives_pass_crc": True,
            "all_member_sets_match_schema": True,
            "all_npy_headers_and_payload_lengths_valid": True,
        },
    }
    protocol_path = metadata_root / "npz_conversion.json"
    _atomic_text(protocol_path, json.dumps(protocol, indent=2) + "\n")
    return protocol


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=project_root / "dataset_5010_v1"
    )
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    protocol = create_npz_companion(args.dataset.resolve(), args.workers)
    print(json.dumps(protocol, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
