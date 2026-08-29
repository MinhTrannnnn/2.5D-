# Reproducible software environment

The release was validated with Python 3.9.6 and the exact package versions in
`requirements-lock.txt`. The shorter `requirements.txt` states the supported
direct dependencies; use the lock file when reproducing published results.

## Option A: Conda

Run from this source directory:

```bash
conda env create -f environment.yml
conda activate paired-25d
python -m unittest discover -s tests -v
```

## Option B: Python virtual environment

Use a Python 3.9.6 interpreter:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==21.2.4
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

The reference package set is:

- NumPy 2.0.2
- SciPy 1.13.1
- Matplotlib 3.9.4
- Pillow 11.3.0

Generation, task sampling, stochastic planner seeds, and numerical outputs are
deterministic under the recorded protocol. Runtime is not deterministic across
machines or concurrent worker counts. The released benchmark timing was
collected in a three-worker throughput run and must not be presented as
isolated single-core latency. Use `--workers 1` for a new latency experiment and
record the CPU model, RAM, operating system, Python version, and package lock
alongside the results.

## Reproduce the release checks

Assuming the final dataset is at `../../dataset_5010_v1` relative to this
source directory:

```bash
python validate_dataset.py --dataset ../../dataset_5010_v1
python analyze_25d_benchmark_results.py \
  --dataset ../../dataset_5010_v1 \
  --benchmark-prefix pathfinding_benchmark_v1
python prepare_25d_release.py \
  --dataset ../../dataset_5010_v1 \
  --benchmark-prefix pathfinding_benchmark_v1
```

The last command hashes the published map JSON files directly. Generation and
benchmark checkpoint files are deliberately excluded from the public release.
