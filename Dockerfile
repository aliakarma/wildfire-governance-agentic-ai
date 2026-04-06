FROM python:3.10-slim

# Metadata
LABEL org.opencontainers.image.title="Wildfire Governance Agentic AI"
LABEL org.opencontainers.image.description="GOMDP: Blockchain-Enforced Human Oversight for Wildfire Monitoring"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/akarma-iu/wildfire-governance-agentic-ai"

# System dependencies for geospatial and scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libhdf5-dev \
    libnetcdf-dev \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements first (layer-cache optimisation)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY setup.py pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy configs, experiments, synthetic data, and pre-committed results
COPY configs/ configs/
COPY data/synthetic/ data/synthetic/
COPY data/scripts/ data/scripts/
COPY experiments/ experiments/
COPY results/paper/ results/paper/
COPY scripts/ scripts/

# Create output directories
RUN mkdir -p data/raw data/processed results/runs

# Non-root user for security
RUN useradd -m -u 1000 researcher && \
    chown -R researcher:researcher /workspace
USER researcher

# Default: run smoke experiment
CMD ["python", "experiments/01_main_comparison.py", \
     "--config", "configs/experiments/paper_main_results.yaml", \
     "--smoke"]
