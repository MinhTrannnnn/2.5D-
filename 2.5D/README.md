# Paired 2.5D terrain dataset

This directory contains the reusable benchmark implementation and the validated
dataset release. Generated paper sources are intentionally kept separate so a
new manuscript can be developed without mixing publication files with the
scientific data pipeline.

```text
2.5D/
|-- benchmark/             generation, planning, analysis, tests, environment
|-- dataset_5010_v1/       validated maps, images, metadata, and documentation
|-- .gitignore
`-- README.md
```

Use `benchmark/generate_25d_dataset.py` to create preview or final datasets,
`benchmark/benchmark_pathplanning.py` to run the standardized planners, and
`benchmark/analyze_25d_benchmark_results.py` to audit and summarize results.
Exact environment setup and release checks are documented in
`benchmark/REPRODUCIBILITY.md`.

The full map and image payload is intentionally excluded from ordinary Git
history. The compact documentation, summaries, checksums, benchmark figures,
and contact sheets remain versioned. The complete release is archived on
Zenodo at <https://doi.org/10.5281/zenodo.22074838>.
