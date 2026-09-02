#!/usr/bin/env python3
"""Build self-contained Overleaf ZIP packages from validated project figures."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PAPER_ROOT.parent / "dataset_5010_v1"
FIGURE_ROOT = PAPER_ROOT / "figures"
DIST_ROOT = PAPER_ROOT / "dist"

PIPELINE_FIGURES = (
    "terrain_triplet_001.png",
    "dataset_difficulty_overview.png",
    "a_star_mountain_hard.png",
    "bfs_mountain_hard.png",
    "dijkstra_mountain_hard.png",
    "prm_mountain_hard.png",
    "rrt_connect_mountain_hard.png",
    "rrt_star_mountain_hard.png",
    "analysis_smooth_obstacles_hard.png",
    "analysis_rugged_hard.png",
    "analysis_plateau_hard.png",
)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_zip(
    source_tex: Path,
    source_figures: Path,
    output_zip: Path,
    readme: str,
    supplemental_files: tuple[Path, ...],
    temporary_prefix: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix=temporary_prefix) as temporary:
        root = Path(temporary)
        shutil.copy2(source_tex, root / "main.tex")
        target_figures = root / "figures"
        target_figures.mkdir()
        for filename in PIPELINE_FIGURES:
            source_figure = source_figures / filename
            if not source_figure.is_file():
                raise FileNotFoundError(source_figure)
            shutil.copy2(source_figure, target_figures / filename)
        for supplemental_file in supplemental_files:
            if not supplemental_file.is_file():
                raise FileNotFoundError(supplemental_file)
            shutil.copy2(supplemental_file, root / supplemental_file.name)
        (root / "README.txt").write_text(readme, encoding="utf-8")
        manifest_lines = []
        for item in sorted(path for path in root.rglob("*") if path.is_file()):
            manifest_lines.append(
                f"{_sha256(item)}  {item.relative_to(root).as_posix()}"
            )
        (root / "SHA256SUMS.txt").write_text(
            "\n".join(manifest_lines) + "\n", encoding="utf-8"
        )

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        temporary_zip = output_zip.with_suffix(".tmp.zip")
        temporary_zip.unlink(missing_ok=True)
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(path for path in root.rglob("*") if path.is_file()):
                archive.write(item, item.relative_to(root).as_posix())
        temporary_zip.replace(output_zip)


def _copy_release_figures(target: Path, terrain_source: Path) -> None:
    route_source = (
        DATASET_ROOT
        / "images"
        / "paths"
        / "mountain"
        / "terrain_mountain_001_hard_A_Star.png"
    )
    route_sources = {
        "a_star_mountain_hard.png": route_source,
        "bfs_mountain_hard.png": DATASET_ROOT
        / "images"
        / "paths"
        / "mountain"
        / "terrain_mountain_001_hard_BFS.png",
        "dijkstra_mountain_hard.png": DATASET_ROOT
        / "images"
        / "paths"
        / "mountain"
        / "terrain_mountain_001_hard_Dijkstra.png",
        "prm_mountain_hard.png": DATASET_ROOT
        / "images"
        / "paths"
        / "mountain"
        / "terrain_mountain_001_hard_PRM.png",
        "rrt_connect_mountain_hard.png": DATASET_ROOT
        / "images"
        / "paths"
        / "mountain"
        / "terrain_mountain_001_hard_RRT_Connect.png",
        "rrt_star_mountain_hard.png": DATASET_ROOT
        / "images"
        / "paths"
        / "mountain"
        / "terrain_mountain_001_hard_RRT_Star.png",
    }
    analysis_sources = {
        "analysis_smooth_obstacles_hard.png": DATASET_ROOT
        / "images"
        / "analysis"
        / "smooth_obstacles"
        / "terrain_smooth_obstacles_001_hard_analysis.png",
        "analysis_rugged_hard.png": DATASET_ROOT
        / "images"
        / "analysis"
        / "rugged"
        / "terrain_rugged_001_hard_analysis.png",
        "analysis_plateau_hard.png": DATASET_ROOT
        / "images"
        / "analysis"
        / "plateau"
        / "terrain_plateau_001_hard_analysis.png",
    }
    difficulty_source = (
        DATASET_ROOT
        / "metadata"
        / "benchmark_results"
        / "figures"
        / "dataset_difficulty_overview.png"
    )
    if not terrain_source.is_file():
        raise FileNotFoundError(terrain_source)
    for source in (*route_sources.values(), *analysis_sources.values()):
        if not source.is_file():
            raise FileNotFoundError(source)
    if not difficulty_source.is_file():
        raise FileNotFoundError(difficulty_source)

    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(terrain_source, target / "terrain_triplet_001.png")
    shutil.copy2(difficulty_source, target / "dataset_difficulty_overview.png")
    for filename, source in route_sources.items():
        shutil.copy2(source, target / filename)
    for filename, source in analysis_sources.items():
        shutil.copy2(source, target / filename)


def _build_language(language: str) -> None:
    # Figure 1 is a compact layout composed only from the 15 pipeline-rendered
    # terrain images for map index 001. The layout removes repeated margins,
    # titles and legends; it does not alter terrain values, blocked regions, or
    # Start--Goal locations. Rebuild both language variants with
    # make_figure1_options.sh.
    if language == "vi":
        terrain_source = (
            PAPER_ROOT / "figure1_options" / "option_d_clean_15_panels.png"
        )
        source_tex = PAPER_ROOT / "main_vi.tex"
        source_figures = FIGURE_ROOT / "vi"
        output_zip = DIST_ROOT / "Scientific_Data_Vietnamese_Overleaf.zip"
        supplements = (
            PAPER_ROOT / "SUBMISSION_CHECKLIST_VI.md",
            PAPER_ROOT / "FIGURE_PROVENANCE_VI.md",
        )
        readme = (
            "Upload this ZIP as a new Overleaf project. main.tex is the main document.\n"
            "All referenced figures are included in figures/.\n"
            "This Vietnamese review copy compiles with Overleaf's default pdfLaTeX engine.\n"
        )
        temporary_prefix = "scientific-data-vi-"
    elif language == "en":
        terrain_source = (
            PAPER_ROOT / "figure1_options" / "option_d_clean_15_panels_en.png"
        )
        source_tex = PAPER_ROOT / "main_en.tex"
        source_figures = FIGURE_ROOT / "en"
        output_zip = DIST_ROOT / "Scientific_Data_English_Overleaf.zip"
        supplements = (
            PAPER_ROOT / "SUBMISSION_CHECKLIST_EN.md",
            PAPER_ROOT / "FIGURE_PROVENANCE_EN.md",
        )
        readme = (
            "Upload this ZIP as a new Overleaf project. main.tex is the main document.\n"
            "All referenced figures are included in figures/.\n"
            "This English submission draft compiles with Overleaf's default pdfLaTeX engine.\n"
        )
        temporary_prefix = "scientific-data-en-"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_tex.is_file():
        raise FileNotFoundError(source_tex)
    _copy_release_figures(source_figures, terrain_source)
    _write_zip(
        source_tex,
        source_figures,
        output_zip,
        readme,
        supplements,
        temporary_prefix,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build self-contained English or Vietnamese Overleaf packages."
    )
    parser.add_argument(
        "--language",
        choices=("vi", "en", "all"),
        default="vi",
        help="package to build (default: vi)",
    )
    arguments = parser.parse_args()
    languages = ("vi", "en") if arguments.language == "all" else (arguments.language,)
    for language in languages:
        _build_language(language)
    print(f"Wrote {', '.join(languages)} package(s) to {DIST_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
