# Data Directory

## Quick Setup (No Downloads Required for Testing)

Synthetic data in `data/synthetic/` is sufficient for all unit tests and smoke tests.
No API keys or downloads are needed to run `make test-smoke`.

To reproduce all paper results including real-world VIIRS experiments (Table VI),
follow the download instructions below.

---

## Synthetic Data (Committed to Repository)

| File | Size | Used For |
|------|------|----------|
| `data/synthetic/grid_10x10_seed42.npz` | ~5 KB | Unit tests |
| `data/synthetic/grid_100x100_seed0.npz` | ~300 KB | Smoke tests |

Regenerate with:
```bash
# Bash
python data/synthetic/generate_synthetic.py --size 10 --seed 42 --output data/synthetic/grid_10x10_seed42.npz
python data/synthetic/generate_synthetic.py --size 100 --seed 0 --output data/synthetic/grid_100x100_seed0.npz

# PowerShell
python data/synthetic/generate_synthetic.py --size 10 --seed 42 --output data/synthetic/grid_10x10_seed42.npz
python data/synthetic/generate_synthetic.py --size 100 --seed 0 --output data/synthetic/grid_100x100_seed0.npz
```

---

## Real-World Datasets (Optional — Required for Table VI)

### Dataset 1 — NASA FIRMS VIIRS 375m Active Fire

**Provider:** NASA Fire Information for Resource Management System (FIRMS)
**URL:** https://firms.modaps.eosdis.nasa.gov/
**Licence:** NASA Open Data Policy (public domain)
**Format:** CSV with latitude, longitude, bright_ti4, bright_ti5, frp, confidence, acq_date

**Step 1: Register and obtain API key**
1. Register at https://urs.earthdata.nasa.gov/
2. Request a FIRMS MAP_KEY at https://firms.modaps.eosdis.nasa.gov/usfs/api/area/
3. Set the environment variable:
   ```bash
   # Bash (Linux/macOS)
   export NASA_FIRMS_KEY="your_map_key_here"
   # Add to ~/.bashrc or ~/.zshrc for persistence

   # PowerShell (Windows)
   $env:NASA_FIRMS_KEY = "your_map_key_here"
   # Add to $PROFILE for persistence
   ```

**Step 2: Download**
```bash
# Bash — all three regions used in Table VI
make download-viirs

# PowerShell — equivalent commands
python data/scripts/download_viirs.py --region california --start_date 2020-08-01 --end_date 2020-10-01
python data/scripts/download_viirs.py --region mediterranean --start_date 2021-08-01 --end_date 2021-09-30
python data/scripts/download_viirs.py --region australia --start_date 2019-11-01 --end_date 2020-02-28
```

Each download now **auto-preprocesses** to the canonical simulation grid the
experiment scripts expect — `data/processed/viirs_grid_<region>_<year>.npz`
(e.g. `viirs_grid_california_2020.npz`) — so `experiments/08_viirs_california.py`,
`08b`, and `08c` run the real simulation on the VIIRS grid rather than the CSV
fallback. Pass `--no-preprocess` to download the raw CSV only. These grids are
git-ignored.

**Without API key:** the download script writes a small synthetic VIIRS fallback
CSV and preprocesses it to the same canonical grid names, so the whole
download → preprocess → experiment pipeline is exercised end-to-end. The grid is
labeled SYNTHETIC in the console output; set `NASA_FIRMS_KEY` and rerun for real
VIIRS validation (the only step that needs the key — everything else is wired).

---

### Dataset 2 — NIFC Historical Fire Perimeters (US)

**Provider:** National Interagency Fire Center
**URL:** https://data-nifc.opendata.arcgis.com/
**Licence:** US Federal Government Open Data (public domain)
**Format:** GeoJSON via ArcGIS REST API

```bash
# Bash / PowerShell
python data/scripts/download_nifc.py --years 2020 --states CA
```

---

### Dataset 3 — NOAA GOES-16 Fire Detection (Optional)

**Provider:** NOAA via AWS Open Data (no authentication required)
**URL:** s3://noaa-goes16/ABI-L2-FDCF/
**Licence:** NOAA Open Data

```bash
# Bash / PowerShell
python data/scripts/download_goes16.py --region california \
    --start_datetime 2020-08-01T00:00:00 \
    --end_datetime 2020-08-07T00:00:00
```

---

## Dataset Summary Table

| Dataset | Provider | Size (regional) | Required for Paper | Script |
|---------|----------|-----------------|-------------------|--------|
| VIIRS 375m | NASA FIRMS | ~50–200 MB | Table VI | `download_viirs.py` |
| NIFC Perimeters | NIFC | ~10–50 MB | Table VI (Fp labels) | `download_nifc.py` |
| GOES-16 | NOAA AWS | ~100 MB | Optional | `download_goes16.py` |

---

## Verifying Downloads

```bash
# Bash / PowerShell
python data/scripts/validate_datasets.py
```

---

## Directory Layout After Download

```
data/
├── raw/
│   ├── viirs/          VIIRS CSV files (gitignored)
│   ├── nifc/           NIFC GeoJSON files (gitignored)
│   └── goes16/         GOES-16 NetCDF files (gitignored)
├── processed/          Simulation-ready .npz files (gitignored)
├── synthetic/          Small committed synthetic grids
└── scripts/            Download and preprocessing scripts
```

All `data/raw/` and `data/processed/` contents are gitignored.
Only `data/synthetic/` and `data/scripts/` are committed.
