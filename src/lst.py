"""Phase 1: Landsat 8/9 land surface temperature composite (Mar-May, 2023-2025).

Outputs (cached, skipped if present):
  data/processed/lst_landsat.tif   median LST in deg C, 30 m
  data/processed/landsat_sr.tif    median SR_B4 / SR_B5 reflectance (used by Phase 2 for NDVI)
  outputs/figures/lst.png
Run: python -m src.lst
"""
import ee
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from config import FIGURES, GEE_PROJECT, PROCESSED, SEASON_MONTHS, YEARS
from src.utils import ensure_dirs

LST_TIF = PROCESSED / "lst_landsat.tif"
SR_TIF = PROCESSED / "landsat_sr.tif"
SCALE = 30


def init():
    if not GEE_PROJECT:
        raise SystemExit("Set GEE_PROJECT, e.g. export GEE_PROJECT=bkk-heatmap")
    ee.Initialize(project=GEE_PROJECT)


def mask_clouds(img):
    qa = img.select("QA_PIXEL")
    # bit 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow
    bad = (qa.bitwiseAnd(1 << 1).neq(0)
           .Or(qa.bitwiseAnd(1 << 2).neq(0))
           .Or(qa.bitwiseAnd(1 << 3).neq(0))
           .Or(qa.bitwiseAnd(1 << 4).neq(0)))
    return img.updateMask(bad.Not())


def prep(img):
    lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")
    b4 = img.select("SR_B4").multiply(0.0000275).add(-0.2).rename("SR_B4")
    b5 = img.select("SR_B5").multiply(0.0000275).add(-0.2).rename("SR_B5")
    return lst.addBands([b4, b5]).copyProperties(img, ["system:time_start"])


def composite(region):
    cols = []
    for coll in ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"):
        for y in YEARS:
            c = (ee.ImageCollection(coll).filterBounds(region)
                 .filterDate(f"{y}-{SEASON_MONTHS[0]:02d}-01", f"{y}-{SEASON_MONTHS[1] + 1:02d}-01")
                 .map(mask_clouds).map(prep))
            cols.append(c)
    merged = cols[0]
    for c in cols[1:]:
        merged = merged.merge(c)
    return merged.median(), merged.size()


def download(img, region, path):
    img = img.toFloat()  # float32 keeps the request under GEE's 50 MB direct-download limit
    url = img.getDownloadURL({"region": region, "scale": SCALE, "crs": "EPSG:4326",
                              "format": "GEO_TIFF"})
    import requests
    r = requests.get(url, timeout=900)
    r.raise_for_status()
    path.write_bytes(r.content)


def report_and_plot():
    with rasterio.open(LST_TIF) as src:
        a = src.read(1, masked=True)
        bounds = src.bounds
    d = gpd.read_file(PROCESSED / "districts.gpkg")
    # restrict stats to pixels inside Bangkok
    from rasterio.features import geometry_mask
    with rasterio.open(LST_TIF) as src:
        inside = ~geometry_mask(d.geometry, out_shape=src.shape, transform=src.transform)
    vals = a[inside]
    valid = vals.compressed()
    print(f"LST degC  min={valid.min():.1f}  median={np.median(valid):.1f}  max={valid.max():.1f}")
    print(f"masked (no-data) fraction inside Bangkok: {vals.mask.mean():.4f}")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.where(inside, a, np.nan), cmap="inferno",
                   extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
    d.boundary.plot(ax=ax, color="white", linewidth=0.3)
    plt.colorbar(im, ax=ax, label="LST (°C), Mar–May 2023–25 median")
    ax.set_axis_off()
    plt.savefig(FIGURES / "lst.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()
    if not (LST_TIF.exists() and SR_TIF.exists()):
        init()
        d = gpd.read_file(PROCESSED / "districts.gpkg")
        region = ee.Geometry.Rectangle(list(d.total_bounds))
        img, n = composite(region)
        print("scenes in composite:", n.getInfo())
        img = img.clip(ee.Geometry.Rectangle(list(d.total_bounds)))
        if not LST_TIF.exists():
            download(img.select("LST"), region, LST_TIF)
        if not SR_TIF.exists():
            download(img.select(["SR_B4", "SR_B5"]), region, SR_TIF)
    report_and_plot()


if __name__ == "__main__":
    main()
