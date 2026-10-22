"""Phase 2: heat drivers -> data/processed/ndvi.tif, data/processed/district_features.csv.

NDVI from the Landsat composite; parks/water/buildings from OSM (cached per district under
data/raw/osm/cache/); per-district features. Run: python -m src.drivers
"""
import os
import time

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pyogrio
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterstats import zonal_stats
from scipy.stats import pearsonr

from config import CRS_METRIC, CRS_STORAGE, FIGURES, PROCESSED, RAW
from src.lst import LST_TIF, SR_TIF
from src.utils import ensure_dirs

NDVI_TIF = PROCESSED / "ndvi.tif"
OSM = RAW / "osm"
CACHE = OSM / "cache"
LAYERS = {
    "parks": {"leisure": ["park", "garden", "nature_reserve", "golf_course"],
              "landuse": ["grass", "forest", "recreation_ground", "village_green", "meadow"],
              "natural": ["wood", "grassland"]},
    "water": {"natural": "water", "waterway": ["canal", "river"]},
    "buildings": {"building": True},
}
ox.settings.timeout = 300
ox.settings.use_cache = True
ox.settings.cache_folder = str(RAW / "osmnx_http_cache")


def make_ndvi():
    if NDVI_TIF.exists():
        return
    with rasterio.open(SR_TIF) as src:
        red, nir = src.read(1).astype("float32"), src.read(2).astype("float32")
        prof = src.profile
    ndvi = (nir - red) / (nir + red)
    prof.update(count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(NDVI_TIF, "w", **prof) as dst:
        dst.write(ndvi.astype("float32"), 1)


KEEP = {"parks": ["name", "leisure", "landuse", "natural"]}  # tags kept to split parks from gardens
EAST_PARKS = os.environ.get("EAST_PARKS") == "1"  # opt in: slow, Overpass often times out
PBF = OSM / "Bangkok.osm.pbf"
PBF_URL = "https://download.bbbike.org/osm/bbbike/Bangkok/Bangkok.osm.pbf"
PBF_EAST = 100.85  # the BBBike extract stops at ~100.858E; the strip east of this comes from Overpass
OVERPASS_TAGS = LAYERS
ox.settings.requests_headers = {"User-Agent": "bkk-heatmap-student-project/0.1"}
GREEN_LEISURE = "'park','garden','nature_reserve','golf_course'"
GREEN_LANDUSE = "'grass','forest','recreation_ground','village_green','meadow'"
SQL = {
    "parks": ("multipolygons", f"leisure IN ({GREEN_LEISURE}) OR landuse IN ({GREEN_LANDUSE}) OR \"natural\" IN ('wood','grassland')"),
    "water": ("multipolygons", "\"natural\" = 'water'"),
    "water_lines": ("lines", "waterway IN ('canal','river')"),
    "buildings": ("multipolygons", "building IS NOT NULL"),
}


def download_pbf():
    if PBF.exists():
        return
    import requests
    OSM.mkdir(parents=True, exist_ok=True)
    r = requests.get(PBF_URL, timeout=600)
    r.raise_for_status()
    PBF.write_bytes(r.content)


def read_pbf(name, bbox):
    layer, where = SQL[name]
    g = pyogrio.read_dataframe(PBF, layer=layer, where=where, bbox=tuple(bbox), use_arrow=False)
    return g[[c for c in KEEP.get(name, []) if c in g.columns] + ["geometry"]]


def fetch_east_strip(layer, bbox, tiles=6, patient=False):
    """Overpass fallback for the strip the extract misses; cached per tile."""
    parts = []
    lats = np.linspace(bbox[1], bbox[3], tiles + 1)
    for i in range(tiles):
        f = CACHE / f"east_{layer}_{i}.gpkg"
        if f.exists():
            parts.append(gpd.read_file(f) if f.stat().st_size else None)
            continue
        box = (bbox[0], lats[i], bbox[2], lats[i + 1])
        attempts = 10 if patient else 3
        for t in range(attempts):
            try:
                g = ox.features_from_bbox(box, LAYERS[layer])
                break
            except ox._errors.InsufficientResponseError:
                g = gpd.GeoDataFrame(geometry=[], crs=CRS_STORAGE)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry east {layer}[{i}] ({e.__class__.__name__})")
                time.sleep((45 if patient else 10) * (t + 1))
        else:
            # Overpass is flaky: skip this tile (not cached, so a rerun retries it) rather than block the pipeline
            print(f"WARNING: east strip {layer} tile {i + 1}/{tiles} skipped (Overpass unavailable)")
            parts.append(None)
            continue
        if len(g):
            g = g.reset_index(drop=True)
            g = g[[c for c in KEEP.get(layer, []) if c in g.columns] + ["geometry"]]
        CACHE.mkdir(parents=True, exist_ok=True)
        g.to_file(f, driver="GPKG") if len(g) else f.write_bytes(b"")
        parts.append(g if len(g) else None)
        print(f"OSM east strip {layer} tile {i + 1}/{tiles}: {len(g)} features")
    parts = [p for p in parts if p is not None]
    return pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame(geometry=[], crs=CRS_STORAGE)


def fetch_osm(districts):
    download_pbf()
    bbox = districts.total_bounds
    east = [PBF_EAST, bbox[1], bbox[2], bbox[3]]
    for layer in ("parks", "water", "buildings"):
        out = OSM / f"{layer}.gpkg"
        if out.exists():
            continue
        g = read_pbf(layer, bbox)
        if layer == "water":
            g = pd.concat([g, read_pbf("water_lines", bbox)], ignore_index=True)
        # Overpass kept failing for the eastern strip of parks/gardens, so that strip is skipped here; the satellite
        # open-green layer still covers it. Water and buildings for the strip are cached from earlier runs.
        e = gpd.GeoDataFrame(geometry=[], crs=CRS_STORAGE) if (layer == "parks" and not EAST_PARKS) else fetch_east_strip(layer, east, patient=True)
        g = pd.concat([g, e], ignore_index=True)
        g = gpd.GeoDataFrame(g, crs=CRS_STORAGE)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        if layer == "parks":
            g["kind"] = np.where(g.leisure.isin(["park", "nature_reserve", "golf_course"]) | (g.landuse == "recreation_ground"),
                                 "park", "garden")
        # assign each feature to the district holding its representative point
        pts = g.copy()
        pts["geometry"] = g.geometry.representative_point()
        j = gpd.sjoin(pts, districts[["district_en", "geometry"]], how="inner", predicate="within")
        g = g.loc[j.index.unique()].copy()
        g["district_en"] = j.loc[~j.index.duplicated(), "district_en"].reindex(g.index).values
        g.to_file(out, driver="GPKG")
        print(f"{layer}: {len(g)} features saved")


def area_share(d_m, feat_m):
    """% of each district covered by the (dissolved) features, computed in metric CRS."""
    out = []
    for _, r in d_m.iterrows():
        f = feat_m[feat_m.district_en == r.district_en]
        f = f[f.geom_type.isin(["Polygon", "MultiPolygon"])]
        if f.empty:
            out.append(0.0)
            continue
        u = f.geometry.make_valid().union_all().intersection(r.geometry)
        out.append(100 * u.area / r.geometry.area)
    return out


def mean_dist_to_water(d_m, water_m, step=200):
    water_u = water_m.geometry.make_valid().union_all()
    res = []
    for _, r in d_m.iterrows():
        minx, miny, maxx, maxy = r.geometry.bounds
        xs, ys = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
        pts = gpd.GeoSeries(gpd.points_from_xy(xs.ravel(), ys.ravel()), crs=CRS_METRIC)
        pts = pts[pts.within(r.geometry)]
        if pts.empty:
            pts = gpd.GeoSeries([r.geometry.representative_point()], crs=CRS_METRIC)
        res.append(float(pts.distance(water_u).mean()))
    return res


BUILTUP_CSV = PROCESSED / "builtup.csv"


def builtup_fraction(districts):
    """% of each district classed 'built-up' (class 50) in ESA WorldCover v200 (10 m, 2021).

    Used instead of OSM building share for the score, because OSM building coverage is patchy.
    """
    if BUILTUP_CSV.exists():
        return pd.read_csv(BUILTUP_CSV)
    import ee

    from config import GEE_PROJECT
    ee.Initialize(project=GEE_PROJECT)
    built = ee.Image("ESA/WorldCover/v200/2021").select("Map").eq(50).rename("builtup")
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry(r.geometry.__geo_interface__), {"district_en": r.district_en})
        for _, r in districts.iterrows()])
    out = built.reduceRegions(fc, ee.Reducer.mean(), scale=10).getInfo()["features"]
    df = pd.DataFrame([{"district_en": f["properties"]["district_en"],
                        "builtup_pct": 100 * f["properties"]["mean"]} for f in out])
    df.to_csv(BUILTUP_CSV, index=False)
    return df


def main():
    ensure_dirs()
    make_ndvi()
    d = gpd.read_file(PROCESSED / "districts.gpkg")
    fetch_osm(d)
    parks, water, bld = (gpd.read_file(OSM / f"{k}.gpkg") for k in ("parks", "water", "buildings"))

    feats = d[["district_th", "district_en"]].copy()
    # Land-only means: open water (OSM water polygons, or NDVI < 0) is cool and has very low NDVI,
    # which otherwise masks the LST-vs-NDVI relationship on land.
    with rasterio.open(LST_TIF) as L, rasterio.open(NDVI_TIF) as N:
        lst, ndvi, tf = L.read(1), N.read(1), L.transform
        wp = water[water.geom_type.isin(["Polygon", "MultiPolygon"])]
        wmask = rasterize(((g, 1) for g in wp.geometry.make_valid()), out_shape=lst.shape,
                          transform=tf, fill=0, dtype="uint8").astype(bool)
    land = ~wmask & np.isfinite(lst) & np.isfinite(ndvi) & (ndvi >= 0)
    print(f"land pixels kept after water mask: {land.sum() / np.isfinite(lst).sum():.3f}")

    def zmean(arr):
        return [z["mean"] for z in zonal_stats(d, arr, affine=tf, nodata=np.nan, stats=["mean"])]

    feats["mean_lst"] = zmean(np.where(land, lst, np.nan))
    feats["mean_ndvi"] = zmean(np.where(land, ndvi, np.nan))
    feats["mean_lst_all"] = zmean(lst)
    feats["mean_ndvi_all"] = zmean(ndvi)
    d_m = d.to_crs(CRS_METRIC)
    feats["green_pct"] = area_share(d_m, parks.to_crs(CRS_METRIC))
    feats["building_pct"] = area_share(d_m, bld.to_crs(CRS_METRIC))
    feats["mean_dist_water_m"] = mean_dist_to_water(d_m, water.to_crs(CRS_METRIC))
    feats = feats.merge(builtup_fraction(d), on="district_en", how="left")
    feats["building_count"] = bld.groupby("district_en").size().reindex(feats.district_en).fillna(0).astype(int).values
    feats["area_km2"] = d_m.geometry.area.values / 1e6
    feats["buildings_per_km2"] = feats.building_count / feats.area_km2
    feats.to_csv(PROCESSED / "district_features.csv", index=False)

    print(feats.drop(columns="district_th").round(2).to_string())
    print("NaNs per column:\n", feats.isna().sum()[feats.isna().sum() > 0])
    low = feats[feats.buildings_per_km2 < feats.buildings_per_km2.median() * 0.25]
    print("\nOSM building coverage possibly incomplete (<25% of median density):")
    print(low[["district_en", "building_count", "buildings_per_km2"]].round(1).to_string(index=False))

    ok = feats.dropna(subset=["mean_lst", "mean_ndvi"])
    r, p = pearsonr(ok.mean_ndvi, ok.mean_lst)
    print(f"\nPearson r (district mean LST vs NDVI) = {r:.3f} (p={p:.2g}, n={len(ok)})")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ok.mean_ndvi, ok.mean_lst)
    ax.set_xlabel("mean NDVI"); ax.set_ylabel("mean LST (°C)"); ax.set_title(f"LST vs NDVI by district (land only), r = {r:.2f}")
    plt.savefig(FIGURES / "lst_vs_ndvi.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
