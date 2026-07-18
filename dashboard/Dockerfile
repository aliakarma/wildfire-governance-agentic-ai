# syntax=docker/dockerfile:1
# Build context must be the REPO ROOT (see docker-compose.dashboard.yml):
#   docker build -f dashboard/Dockerfile -t wildfire-dashboard .

# --- Stage 1: build the static frontend ---------------------------------------
FROM node:20-slim AS frontend
WORKDIR /fe
COPY dashboard/frontend/package.json dashboard/frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY dashboard/frontend/ ./
RUN npm run build          # → /fe/out (static export)

# --- Stage 2: python runtime that serves UI + API -----------------------------
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app/src:/app PYTHONUNBUFFERED=1

COPY dashboard/backend/requirements-runtime.txt dashboard/backend/requirements.txt dashboard/backend/
RUN pip install --no-cache-dir -r dashboard/backend/requirements-runtime.txt \
 && pip install --no-cache-dir -r dashboard/backend/requirements.txt

# Application source (the simulation lives in src/wildfire_governance).
COPY src/ src/
COPY experiments/ experiments/
COPY results/paper/ results/paper/
COPY dashboard/backend/ dashboard/backend/
COPY --from=frontend /fe/out dashboard/frontend/out

EXPOSE 8123
CMD ["python", "-m", "uvicorn", "dashboard.backend.main:app", "--host", "0.0.0.0", "--port", "8123"]
