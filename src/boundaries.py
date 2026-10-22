"""Phase 0: the 50 Bangkok district polygons -> data/processed/districts.gpkg.

Primary: data.bangkok.go.th shapefile. Fallback: GADM 4.1 level 2 (Thailand), Bangkok only.
Run: python -m src.boundaries
"""
import zipfile

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from config import CRS_STORAGE, FIGURES, PROCESSED, RAW
from src.utils import ensure_dirs

BMA_URL = (
    "https://data.bangkok.go.th/dataset/e537025b-1cf6-4c5b-8e46-c2e976f13283/resource/"
    "d7be7e84-9d84-4595-bf8f-79b0bc01f1ae/download/district-2.zip"
)
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_THA_2.json.zip"
OUT = PROCESSED / "districts.gpkg"


def _download(url, dest):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def from_bma():
    z = _download(BMA_URL, RAW / "boundaries" / "district-2.zip")
    # Thai attribute text is TIS-620/cp874 encoded
    g = gpd.read_file(f"zip://{z}", encoding="cp874")
    g = g.rename(columns={"dname": "district_th", "dname_e": "district_en"})
    return g[["district_th", "district_en", "geometry"]]


def from_gadm():
    z = _download(GADM_URL, RAW / "boundaries" / "gadm41_THA_2.json.zip")
    with zipfile.ZipFile(z) as zf:
        name = zf.namelist()[0]
        zf.extract(name, z.parent)
    g = gpd.read_file(z.parent / name)
    g = g[g["NAME_1"] == "Bangkok Metropolis"]
    g = g.rename(columns={"NAME_2": "district_en"})
    g["district_th"] = g["NL_NAME_2"] if "NL_NAME_2" in g else None
    return g[["district_th", "district_en", "geometry"]]


def build():
    try:
        g = from_bma()
        src = "BMA"
        assert len(g) == 50
    except Exception as e:  # noqa: BLE001
        print(f"BMA source failed ({e}); falling back to GADM")
        g = from_gadm()
        src = "GADM"
    g = g.to_crs(CRS_STORAGE)
    g["geometry"] = g.geometry.make_valid()
    g = g.sort_values("district_en").reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    g.to_file(OUT, driver="GPKG")
    return g, src


def plot(g):
    FIGURES.mkdir(parents=True, exist_ok=True)
    ax = g.plot(figsize=(8, 9), edgecolor="k", facecolor="#e8e8e8", linewidth=0.5)
    for _, r in g.iterrows():
        p = r.geometry.representative_point()
        ax.annotate(r.district_en, (p.x, p.y), fontsize=5, ha="center")
    ax.set_title("Bangkok districts (50)")
    ax.set_axis_off()
    plt.savefig(FIGURES / "districts.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    ensure_dirs()
    g, src = build()
    plot(g)
    print(f"source={src} rows={len(g)} all_valid={g.is_valid.all()} -> {OUT}")
    print(g[["district_th", "district_en"]].head(10).to_string())
