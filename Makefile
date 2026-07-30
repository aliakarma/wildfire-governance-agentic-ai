# ============================================================
# Makefile — wildfire-governance-agentic-ai
# Usage: make <target>
# ============================================================

.PHONY: install install-dev test test-smoke test-unit test-integration \
        lint format type-check docs docs-serve \
	download-data download-viirs download-checkpoint generate-synthetic \
        train-ppo eval-ppo \
        reproduce figures \
        adversarial stress-test \
        verify-theorem1 verify-theorem2 verify-paper check-repro \
        docker-build docker-run \
        clean help

# ---- Installation ---------------------------------------------------

install:
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e ".[dev]"
	pre-commit install

# ---- Testing --------------------------------------------------------

test:
	pytest tests/ -v --timeout=120

test-smoke:
	pytest tests/smoke/ -v --timeout=60 --no-cov

test-unit:
	pytest tests/unit/ -v --timeout=60

test-integration:
	pytest tests/integration/ -v --timeout=300

# ---- Code quality ---------------------------------------------------

lint:
	black --check src/ tests/ experiments/
	isort --check-only src/ tests/ experiments/
	flake8 src/ tests/ experiments/

format:
	black src/ tests/ experiments/
	isort src/ tests/ experiments/

type-check:
	mypy src/wildfire_governance/ --ignore-missing-imports

# ---- Documentation --------------------------------------------------

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# ---- Data -----------------------------------------------------------

download-data: download-viirs
	python data/scripts/download_nifc.py --years 2020 --states CA
	python data/scripts/download_goes16.py --region california --start_datetime 2020-08-01T00:00:00 --end_datetime 2020-08-07T00:00:00
	python data/scripts/validate_datasets.py

# Dates are each event's PEAK-activity window (VIIRS_SNPP_SP archive, 10-day pull)
# so the grids carry rich fire pixels. Year in the date sets the canonical grid
# filename (viirs_grid_<region>_<year>.npz), which the experiments load.
download-viirs:
	python data/scripts/download_viirs.py --region california --start_date 2020-09-07 --end_date 2020-09-17
	python data/scripts/download_viirs.py --region mediterranean --start_date 2021-08-03 --end_date 2021-08-13
	python data/scripts/download_viirs.py --region australia --start_date 2019-12-30 --end_date 2020-01-09

download-checkpoint:
	python scripts/download_checkpoint.py

generate-synthetic:
	python data/scripts/generate_synthetic.py

# ---- RL Training ---------------------------------------------------

train-ppo:
	python experiments/11_ppo_training.py \
	  --config configs/experiments/ppo_training.yaml

# Needs a trained checkpoint; `make train-ppo` produces one (see TRAINING.md).
# 11b_rl_comparison.py stops with instructions if none is present.
eval-ppo:
	python experiments/11b_rl_comparison.py \
	  --config configs/experiments/paper_main_results.yaml \
	  --use_pretrained

# ---- Theorem verification ------------------------------------------

verify-theorem1:
	pytest tests/integration/test_gomdp_policy_agnostic.py -v --timeout=300

verify-theorem2:
	pytest tests/unit/test_breach_probability.py -v

# Submission gate: every committed result cell vs Paper/AAAI/Wildfire.tex
verify-paper:
	python scripts/verify_paper_alignment.py

# Diff a fresh run under results/runs/ against the committed CSVs
check-repro:
	python scripts/check_reproducibility.py

# ---- Experiments ---------------------------------------------------

adversarial:
	python experiments/09_adversarial_robustness.py \
	  --config configs/experiments/adversarial_robustness.yaml

stress-test:
	python experiments/10_stress_testing.py \
	  --config configs/experiments/stress_testing.yaml

# ---- Full reproduction (all paper results) -------------------------

# The RL stages need a trained checkpoint (see TRAINING.md); the closed-form and
# blockchain stages do not. Stages that require one stop with instructions.
reproduce:
	python data/scripts/generate_synthetic.py
	bash experiments/run_all.sh

reproduce-smoke:
	bash experiments/run_all.sh --smoke

figures:
	bash scripts/generate_paper_figures.sh

# ---- Docker --------------------------------------------------------

docker-build:
	docker build -t wildfire-gov:latest .

docker-run:
	docker-compose up wildfire-gov

# ---- Cleanup -------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	find . -name "coverage.xml" -delete
	find . -name ".coverage" -delete

# ---- Help ----------------------------------------------------------

help:
	@echo "Available targets:"
	@echo "  install          Install package (production)"
	@echo "  install-dev      Install with dev dependencies + pre-commit"
	@echo "  test             Run full test suite with coverage"
	@echo "  test-smoke       Quick smoke test (< 60s)"
	@echo "  lint             Check code style (black + isort + flake8)"
	@echo "  format           Auto-format code"
	@echo "  download-data    Download all real-world datasets"
	@echo "  train-ppo        Train PPO-GOMDP from scratch"
	@echo "  eval-ppo         Evaluate pre-trained PPO-GOMDP"
	@echo "  verify-theorem1  Empirically verify Theorem 1 (Policy-Agnostic Safety)"
	@echo "  verify-theorem2  Numerically verify Theorem 2 (Adversarial Robustness)"
	@echo "  adversarial      Run adversarial robustness evaluation"
	@echo "  stress-test      Run stress testing suite"
	@echo "  reproduce        Reproduce all paper results (2-4 hrs)"
	@echo "  reproduce-smoke  Smoke reproduction (< 5 min)"
	@echo "  figures          Regenerate all paper figures"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run via Docker Compose"
	@echo "  clean            Remove caches and build artefacts"
