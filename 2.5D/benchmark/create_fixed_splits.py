#!/usr/bin/env python3
"""Create deterministic group-level train/validation/test splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence


SPLIT_COUNTS_PER_FAMILY = {"train": 234, "validation": 50, "test": 50}
DIFFICULTIES = ("easy", "medium", "hard")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _assignment_rank(seed: int, matched_group_id: str) -> str:
    value = f"paired-25d-fixed-split-v1:{seed}:{matched_group_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_splits(dataset_root: Path, seed: int) -> dict[str, object]:
    summary_path = dataset_root / "metadata" / "dataset_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    groups: dict[str, dict[str, object]] = {}
    family_order: list[str] = []
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group_id = row["matched_group_id"]
            family = row["family"]
            difficulty = row["difficulty"]
            if family not in family_order:
                family_order.append(family)
            group = groups.setdefault(
                group_id,
                {
                    "matched_group_id": group_id,
                    "family": family,
                    "instance_index": int(row["instance_index"]),
                    "members": {},
                },
            )
            if group["family"] != family:
                raise ValueError(f"family mismatch within {group_id}")
            members = group["members"]
            assert isinstance(members, dict)
            if difficulty in members:
                raise ValueError(f"duplicate {difficulty} member in {group_id}")
            members[difficulty] = {
                "instance_id": row["instance_id"],
                "map_file": row["map_file"],
            }

    if len(groups) != 1670:
        raise ValueError(f"expected 1670 matched groups, found {len(groups)}")
    for group_id, group in groups.items():
        members = group["members"]
        assert isinstance(members, dict)
        if set(members) != set(DIFFICULTIES):
            raise ValueError(f"{group_id} does not contain exactly Easy/Medium/Hard")

    by_family: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for group in groups.values():
        by_family[str(group["family"])].append(group)
    if set(by_family) != set(family_order):
        raise ValueError("family indexing is inconsistent")

    split_rows: dict[str, list[dict[str, object]]] = {
        split: [] for split in SPLIT_COUNTS_PER_FAMILY
    }
    assignment: dict[str, str] = {}
    for family in family_order:
        family_groups = sorted(
            by_family[family],
            key=lambda group: (
                _assignment_rank(seed, str(group["matched_group_id"])),
                str(group["matched_group_id"]),
            ),
        )
        expected = sum(SPLIT_COUNTS_PER_FAMILY.values())
        if len(family_groups) != expected:
            raise ValueError(
                f"{family}: expected {expected} groups, found {len(family_groups)}"
            )
        offset = 0
        for split, count in SPLIT_COUNTS_PER_FAMILY.items():
            for group in family_groups[offset : offset + count]:
                group_id = str(group["matched_group_id"])
                if group_id in assignment:
                    raise ValueError(f"group leakage detected for {group_id}")
                assignment[group_id] = split
                members = group["members"]
                assert isinstance(members, dict)
                split_rows[split].append(
                    {
                        "split": split,
                        "matched_group_id": group_id,
                        "family": family,
                        "instance_index": group["instance_index"],
                        "easy_instance_id": members["easy"]["instance_id"],
                        "medium_instance_id": members["medium"]["instance_id"],
                        "hard_instance_id": members["hard"]["instance_id"],
                        "easy_map_file": members["easy"]["map_file"],
                        "medium_map_file": members["medium"]["map_file"],
                        "hard_map_file": members["hard"]["map_file"],
                    }
                )
            offset += count

    if set(assignment) != set(groups):
        raise ValueError("split assignment is incomplete")

    split_root = dataset_root / "metadata" / "splits"
    for split, rows in split_rows.items():
        rows.sort(key=lambda row: (family_order.index(str(row["family"])), int(row["instance_index"])))
        _atomic_csv(split_root / f"{split}.csv", rows)

    family_counts: dict[str, dict[str, int]] = {}
    for split, rows in split_rows.items():
        counts = Counter(str(row["family"]) for row in rows)
        family_counts[split] = {family: counts[family] for family in family_order}

    split_files = {
        f"{split}.csv": {
            "matched_groups": len(rows),
            "maps_referenced": 3 * len(rows),
            "unique_paired_tasks": 10 * len(rows),
            "sha256": _sha256(split_root / f"{split}.csv"),
        }
        for split, rows in split_rows.items()
    }
    assignment_digest = hashlib.sha256(
        "\n".join(f"{group_id},{assignment[group_id]}" for group_id in sorted(assignment)).encode(
            "utf-8"
        )
    ).hexdigest()
    protocol: dict[str, object] = {
        "schema_version": "1.0",
        "protocol": "paired-25d-fixed-split-v1",
        "source": "metadata/dataset_summary.csv",
        "source_sha256": _sha256(summary_path),
        "assignment_unit": "matched_group_id",
        "stratification": "terrain family",
        "assignment_method": (
            "within each family, sort groups by SHA-256 of "
            "'paired-25d-fixed-split-v1:<seed>:<matched_group_id>', then take "
            "234 train, 50 validation, and 50 test groups"
        ),
        "seed": seed,
        "counts_per_family": SPLIT_COUNTS_PER_FAMILY,
        "family_counts": family_counts,
        "files": split_files,
        "totals": {
            "matched_groups": len(assignment),
            "maps_referenced": 3 * len(assignment),
            "unique_paired_tasks": 10 * len(assignment),
        },
        "validation": {
            "all_groups_assigned_once": True,
            "group_leakage_count": 0,
            "all_groups_have_easy_medium_hard": True,
            "family_stratification_exact": True,
        },
        "assignment_sha256": assignment_digest,
    }
    _atomic_json(split_root / "split_protocol.json", protocol)
    return protocol


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=project_root / "dataset_5010_v1"
    )
    parser.add_argument("--seed", type=int, default=20260830)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = create_splits(args.dataset.resolve(), args.seed)
    print(json.dumps(protocol["totals"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
