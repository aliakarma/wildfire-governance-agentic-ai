# Installation

## Requirements

- Python 3.10 or 3.11
- 4 GB RAM minimum (16 GB recommended for full experiments)
- No GPU required (CPU-only by default)

## Option A — Conda (Recommended)

```bash
# Bash (Linux/macOS)
git clone https://github.com/akarma-iu/wildfire-governance-agentic-ai.git
cd wildfire-governance-agentic-ai
conda env create -f environment.yml
conda activate wildfire-gov
pip install -e ".[dev]"
```

```powershell
# PowerShell (Windows)
git clone https://github.com/akarma-iu/wildfire-governance-agentic-ai.git
Set-Location wildfire-governance-agentic-ai
conda env create -f environment.yml
conda activate wildfire-gov
pip install -e ".[dev]"
```

## Option B — pip + venv

```bash
# Bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e ".[dev]"
export PYTHONPATH=src
```

```powershell
# PowerShell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e ".[dev]"
$env:PYTHONPATH = "src"
```

## Option C — Docker

```bash
docker-compose up wildfire-gov
```

## Verify Installation

```bash
# Bash / PowerShell
python -m pytest tests/smoke/ -v --timeout=60
# Expected: 7 passed in < 10 seconds
```
