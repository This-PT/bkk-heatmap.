"""Project-wide settings: AOI, date ranges, GEE project id, paths."""
import os
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"

SEASON_MONTHS = (3, 5)  # March-May
YEARS = (2023, 2024, 2025)

CRS_STORAGE = "EPSG:4326"
CRS_METRIC = "EPSG:32647"  # UTM 47N, for area/distance

GEE_PROJECT = os.environ.get("GEE_PROJECT")

# Phase 3 validation points: TMD stations (WMO ids), coordinates from NOAA ISD station list
# (data/raw/isd-history.csv, https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv)
STATIONS = {
    "Bangkok Metropolis (48455)": (13.733, 100.567),
    "Bangkok Port (48454)": (13.700, 100.567),
    "Bang Na Agromet (48453)": (13.667, 100.617),
    "Don Mueang (48456)": (13.913, 100.607),
    "Suvarnabhumi (48429)": (13.683, 100.767),
}
OVERPASS_HOUR = 10.5  # Landsat 8/9 crosses Bangkok at about 10:30 local time
