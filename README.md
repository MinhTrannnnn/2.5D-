# Paired multi-difficulty 2.5D terrain dataset

This repository contains the generation, validation, and path-planning
benchmark code for **A paired multi-difficulty 2.5D terrain dataset for
ground-robot path planning**.

The complete dataset is archived on Zenodo:
[https://doi.org/10.5281/zenodo.22074838](https://doi.org/10.5281/zenodo.22074838).
Large map and image payloads are intentionally excluded from Git history.

## Dataset

- 5,010 procedurally generated 2.5D height-field maps;
- 1,670 matched Easy--Medium--Hard triplets;
- five terrain families and ten shared Start--Goal tasks per triplet;
- 50,100 map--task records;
- six reference planners and 90,180 fixed-protocol benchmark trials;
- deterministic seeds, validation reports, and SHA-256 checksums.

## Repository structure

```text
2.5D/
|-- benchmark/          generation, planning, validation, tests, environment
|-- dataset_5010_v1/    compact release metadata, examples, and documentation
|-- paper/              English and Vietnamese manuscript sources
`-- README.md           implementation overview
```

The full data payload is distributed through Zenodo as one compact core
archive and five terrain-family archives. The `2D/` maze project is not part of
this publication or repository release.

## Quick start

Create the pinned Python environment:

```bash
cd 2.5D
conda env create -f benchmark/environment.yml
conda activate paired-25d
```

Run the regression tests:

```bash
cd benchmark
python -m unittest discover -s tests -v
```

Validate a downloaded dataset:

```bash
python validate_dataset.py --dataset ../dataset_5010_v1
```

Detailed generation and reproduction instructions are provided in
[`2.5D/benchmark/REPRODUCIBILITY.md`](2.5D/benchmark/REPRODUCIBILITY.md).

## Licences

- Dataset: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Source code: MIT License.

See the licence files in `2.5D/dataset_5010_v1/` and `2.5D/benchmark/`.

## Citation

```text
Tran, Q. M. A paired multi-difficulty 2.5D terrain dataset for ground-robot
path planning. Zenodo (2026). https://doi.org/10.5281/zenodo.22074838
```
