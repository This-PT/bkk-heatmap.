# Bangkok Heat Map Prototype — Build Plan

## Goal
Answer: **"Which Bangkok districts are hottest, why, and who is most exposed?"**
Deliverable: an interactive Streamlit map with 3 layers — land surface temperature (LST), heat drivers, and a district heat-risk score for all 50 districts (เขต).

## Rules for the agent
- Work phase by phase. Do not start a phase until the previous phase's acceptance check passes.
- Every number shown in the app or README must come from running the code. Never hardcode or estimate values.
- Cache all fetched data under `data/raw/` and processed outputs under `data/processed/`. Scripts must be re-runnable and skip work already cached.
- Ask the human before: running `earthengine authenticate`, creating cloud projects, or downloading anything over 500 MB.
- Keep each script small and runnable on its own: `python -m src.<module>`.

## Stack
- Python 3.11+, managed with `uv` (or `venv` + `pip`)
- `earthengine-api`, `geemap` — satellite processing in Google Earth Engine (GEE)
- `geopandas`, `shapely`, `rasterio`, `rasterstats` — vector/raster ops
- `osmnx` (or `requests` against the Overpass API) — OpenStreetMap data
- `requests` — Open-Meteo historical API
- `pandas`, `numpy`, `scipy`, `matplotlib`
- `streamlit`, `leafmap` (or `folium` + `streamlit-folium`) — app

## Repo layout
```
bkk-heatmap/
├── PLAN.md
├── README.md
├── pyproject.toml
├── config.py              # AOI, date ranges, GEE project id, paths
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── boundaries.py      # Phase 0
│   ├── lst.py             # Phase 1
│   ├── drivers.py         # Phase 2
│   ├── validate.py        # Phase 3
│   ├── score.py           # Phase 4
│   └── utils.py
├── app/
│   └── streamlit_app.py   # Phase 5
└── outputs/
    └── figures/
```

## Config defaults (`config.py`)
- Season: March–May (hot, dry season), years 2023, 2024, 2025
- CRS: EPSG:4326 for storage, EPSG:32647 (UTM 47N) for area/distance calculations
- GEE project id: read from env var `GEE_PROJECT`

---

## Phase 0 — Setup and boundaries
**Tasks**
1. Scaffold the repo layout above, `pyproject.toml`, and `.gitignore` (ignore `data/`, `.env`).
2. Human step: `earthengine authenticate` and set `GEE_PROJECT`. Stop and ask if not done.
3. `src/boundaries.py`: get the 50 Bangkok district polygons.
   - Primary source: data.bangkok.go.th district boundary dataset.
   - Fallback: GADM level 2 for Thailand, filtered to Bangkok.
   - Save as `data/processed/districts.gpkg` with columns `district_th`, `district_en`, `geometry`.

**Acceptance**
- `districts.gpkg` has exactly 50 rows with valid geometries.
- A quick plot of the districts saved to `outputs/figures/districts.png`.

## Phase 1 — Heat layer (GEE)
**Tasks** (`src/lst.py`)
1. Load Landsat 8 and 9 Collection 2 Level-2 (`LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`).
2. Filter by the Bangkok bounds and Mar–May of each config year.
3. Cloud-mask with `QA_PIXEL` (cloud, cloud shadow, cirrus bits).
4. Convert `ST_B10` to °C: `ST_B10 * 0.00341802 + 149.0 - 273.15`.
5. Median composite, clipped to Bangkok.
6. Export a GeoTIFF at 30 m to `data/processed/lst_landsat.tif`. Use a Drive export if the direct download is too large, and tell the human.
7. Optional: MODIS `MODIS/061/MOD11A1` day and night LST (1 km), same season and years → `lst_modis_day.tif`, `lst_modis_night.tif`.

**Acceptance**
- The GeoTIFF opens, and its values fall in a plausible range. Print min, median, and max in °C.
- Print the fraction of masked (no-data) pixels inside Bangkok.
- Map saved to `outputs/figures/lst.png`.

## Phase 2 — Heat drivers
**Tasks** (`src/drivers.py`)
1. NDVI from the same Landsat composite (SR_B5, SR_B4 with C2 scaling) → `ndvi.tif`. Optionally use Sentinel-2 at 10 m.
2. OpenStreetMap via Overpass/osmnx, within the Bangkok boundary:
   - parks and green space (`leisure=park`, `landuse=grass|forest|recreation_ground`)
   - water and canals (`natural=water`, `waterway=canal|river`)
   - building footprints (`building=*`)
   - Save each as a `.gpkg` in `data/raw/osm/`.
3. Per district, compute:
   - mean LST, mean NDVI
   - green-space % of district area
   - building footprint % of district area
   - mean distance to water
   - Save to `data/processed/district_features.csv`.

**Acceptance**
- The features CSV has 50 rows and no unexplained NaNs.
- Print the OSM building count per district. Flag districts whose count looks suspiciously low, since OSM coverage may be incomplete there.
- Scatter plot of LST vs NDVI with a Pearson r value → `outputs/figures/lst_vs_ndvi.png`.

## Phase 3 — Validation
**Tasks** (`src/validate.py`)
1. Pick 5–10 points: TMD station locations in Bangkok (list their coordinates in `config.py`).
2. Pull hourly 2 m air temperature for the same season and years from the Open-Meteo historical API.
3. Compare station or point air temperature with LST at the same pixels.
4. Report the correlation and mean difference. Note in the README that LST is surface temperature from a ~10:30 am overpass, not air temperature.

**Acceptance**
- Validation table → `outputs/validation.csv`, plot → `outputs/figures/validation.png`.

## Phase 4 — District heat-risk score
**Tasks** (`src/score.py`)
1. Optional exposure data: district population and elderly share from data.bangkok.go.th or data.go.th → `data/raw/population.csv`.
2. Z-score each feature, then compute:
   `heat_risk = z(LST) + z(building %) - z(NDVI) + z(elderly %)`
   (drop the elderly term if that data is unavailable, and say so in the README).
3. Rank the districts and save `data/processed/district_scores.gpkg`.

**Acceptance**
- Print the top 10 and bottom 5 districts with their scores and component values.
- Choropleth map → `outputs/figures/heat_risk.png`.

## Phase 5 — Streamlit app
**Tasks** (`app/streamlit_app.py`)
1. Load only local processed files. The app must not call GEE at runtime.
2. Layers (toggleable): LST raster, NDVI raster, parks/water, district heat-risk choropleth.
3. Sidebar: a district selector showing its stats and rank, plus a top-10 hottest table.
4. Run with: `streamlit run app/streamlit_app.py`.

**Acceptance**
- The app starts with no errors, and all layers render.
- Clicking or selecting a district shows the correct values from `district_scores.gpkg`.

## Phase 6 — README / story
**Tasks**
1. A README with: question, data sources, method, key results (generated numbers only), and limitations.
2. Limitations to state:
   - Landsat thermal is 100 m native.
   - The overpass is around 10:30 am local time.
   - The composite is seasonal.
   - OSM coverage varies by district.
   - The score is a simple index and has not been validated against health outcomes.
3. Context section, written by the human:
   - World Bank CCKP projections for Thailand
   - the BMA climate action plan
   - UddC, C40, and UN-Habitat intervention examples

## Run order
```
python -m src.boundaries
python -m src.lst
python -m src.drivers
python -m src.validate
python -m src.score
streamlit run app/streamlit_app.py
```
