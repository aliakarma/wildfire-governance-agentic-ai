# Contributing Guide

Thank you for your interest in contributing to `wildfire-governance-agentic-ai`.

## Development Setup

```bash
# Bash (Linux/macOS)
git clone https://github.com/akarma-iu/wildfire-governance-agentic-ai.git
cd wildfire-governance-agentic-ai
conda env create -f environment.yml
conda activate wildfire-gov
pip install -e ".[dev]"
pre-commit install
make test-smoke
```

```powershell
# PowerShell (Windows)
git clone https://github.com/akarma-iu/wildfire-governance-agentic-ai.git
Set-Location wildfire-governance-agentic-ai
conda env create -f environment.yml
conda activate wildfire-gov
pip install -e ".[dev]"
pre-commit install
python -m pytest tests/smoke/ -v --no-cov
```

## Branch Naming

- `feature/<description>` — new functionality
- `bugfix/<description>` — bug fixes
- `experiment/<name>` — new experiment scripts
- `docs/<description>` — documentation only

## Commit Message Format (Conventional Commits)

```
feat: add ERA5 weather downloader
fix: correct Bayesian update denominator
docs: update dataset setup instructions
test: add Theorem 2 boundary cases
refactor: extract consensus delay model
```

## Pull Request Requirements

1. All CI checks must pass: lint, type-check, unit tests, smoke experiment
2. Test coverage must not decrease below 75%
3. Every new public function must have a Google-style docstring
4. Every new module must have at least one unit test
5. New experiments must include a corresponding `results/paper/*.csv` entry

## Adding a New Experiment

1. Create `experiments/NN_<name>.py` following the pattern in `experiments/01_main_comparison.py`
2. Add a YAML config in `configs/experiments/<name>.yaml`
3. Add the experiment to `experiments/run_all.sh`
4. Commit result CSV to `results/paper/`
5. Update `README.md` results tables

## Adding a New Dataset Adapter

1. Create `data/scripts/download_<name>.py` with `--region`, `--start_date`, `--end_date` CLI args
2. Create `data/scripts/preprocess_<name>.py` outputting `.npz` to `data/processed/`
3. Add checksum to `data/scripts/validate_datasets.py`
4. Document in `data/README.md`
5. Add a `make download-<name>` target to the `Makefile`

## Code Style

- Line length: 88 (black)
- Imports: isort with black profile
- All functions: type-annotated
- All public APIs: Google-style docstrings
- No `assert` for runtime validation — use `raise ValueError`
- No `random` — use `np.random.default_rng(seed)`
- All paths via `pathlib.Path`, never string concatenation
