"""Phase 7: sub-district analysis. Where in each district should green space be added, and where is it already?

Steps (all cached):
  1. GEE rasters: WorldPop population (100 m), ESA WorldCover class map (30 m), built-up fraction (30 m)
  2. existing green zones = OSM parks + open satellite green (>= 2 ha, NDVI >= 0.45, < 40% built-up around it);
     tree canopy inside housing is kept separately and NOT counted
  3. 250 m grid: LST, NDVI, built-up, population, distance to nearest green zone -> priority score
  4. hotspots = connected clusters of high-priority cells that are far from green space, ranked per district
Outputs: data/processed/{pop,landcover,builtup}.tif, green_zones.gpkg, grid_cells.gpkg, hotspots.gpkg, priority.tif
Run: python -m src.hotspots
"""
import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import requests
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import box, shape

from config import CRS_METRIC, CRS_STORAGE, GEE_PROJECT, PROCESSED, RAW
from src.lst import LST_TIF
from src.drivers import NDVI_TIF, OSM, PBF
from src.utils import ensure_dirs

POP_TIF, LC_TIF, BU_TIF = PROCESSED / "pop.tif", PROCESSED / "landcover.tif", PROCESSED / "builtup.tif"
CELL = 250  # m
MIN_GREEN_HA = 1.0
OPEN_GREEN_HA = 2.0       # satellite patches smaller than this are not counted as green zones
MIN_OSM_GREEN_M2 = 1000   # ignore tiny OSM gardens
RING_M = 60
MAX_RING_BUILTUP = 40    # % built-up in a 60 m ring around a patch; above this it is canopy inside housing
FAR_FROM_GREEN_M = 400   # roughly a 5-minute walk
TOP_QUANTILE = 0.85      # priority threshold within the city
WHO_M2_PER_PERSON = 9    # commonly cited minimum green space per person (WHO)


def gee_rasters(bounds):
    ee.Initialize(project=GEE_PROJECT)
    region = ee.Geometry.Rectangle(list(bounds))
    wc = ee.Image("ESA/WorldCover/v200/2021").select("Map")
    jobs = {
        POP_TIF: ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("country", "THA"))
                   .filter(ee.Filter.eq("year", 2020)).mosaic().rename("pop").toFloat(),
        LC_TIF: wc.toInt16(),
        BU_TIF: wc.eq(50).toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=1024)
                  .reproject("EPSG:4326", None, 30).rename("builtup"),
    }
    scale = {POP_TIF: 100, LC_TIF: 30, BU_TIF: 30}
    for path, img in jobs.items():
        if path.exists():
            continue
        url = img.getDownloadURL({"region": region, "scale": scale[path], "crs": "EPSG:4326", "format": "GEO_TIFF"})
        r = requests.get(url, timeout=900)
        r.raise_for_status()
        path.write_bytes(r.content)
        print("saved", path.name, f"{path.stat().st_size / 1e6:.1f} MB")


POP30_TIF = PROCESSED / "pop30.tif"


def clean_population():
    """WorldPop 100 m -> NDVI 30 m grid; -inf/nodata -> 0; counts split by pixel-area ratio so sums are conserved."""
    if POP30_TIF.exists():
        return
    from rasterio.warp import Resampling, reproject
    with rasterio.open(POP_TIF) as p, rasterio.open(NDVI_TIF) as n:
        a = p.read(1).astype("float64")
        a[~np.isfinite(a)] = 0
        dst = np.zeros(n.shape, dtype="float32")
        reproject(a, dst, src_transform=p.transform, src_crs=p.crs, dst_transform=n.transform,
                  dst_crs=n.crs, resampling=Resampling.nearest)
        ratio = (abs(p.transform.a * p.transform.e)) / (abs(n.transform.a * n.transform.e))
        prof = n.profile
    prof.update(dtype="float32", nodata=None)
    with rasterio.open(POP30_TIF, "w", **prof) as w:
        w.write(dst / ratio, 1)


def existing_green(districts):
    out = PROCESSED / "green_zones.gpkg"
    if out.exists():
        return gpd.read_file(out)
    with rasterio.open(LC_TIF) as lc_src, rasterio.open(NDVI_TIF) as nd_src:
        ndvi = nd_src.read(1)
        tf = nd_src.transform
        lc = lc_src.read(1)
        if lc.shape != ndvi.shape:  # landcover exported on its own grid: resample by nearest
            from rasterio.warp import Resampling, reproject
            dst = np.zeros(ndvi.shape, dtype="int16")
            reproject(lc, dst, src_transform=lc_src.transform, src_crs=lc_src.crs, dst_transform=tf,
                      dst_crs=nd_src.crs, resampling=Resampling.nearest)
            lc = dst
    veg = np.isin(lc, [10, 20, 30]) & (ndvi >= 0.4)  # tree, shrub, grassland
    lab, n = ndimage.label(veg)
    px_ha = abs(tf.a * tf.e) * (111_320 ** 2) * np.cos(np.radians(13.75)) / 1e4
    sizes = ndimage.sum(veg, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.where(sizes * px_ha >= MIN_GREEN_HA)[0])
    geoms = [shape(g) for g, v in shapes(keep.astype("uint8"), mask=keep, transform=tf) if v == 1]
    sat = gpd.GeoDataFrame(geometry=geoms, crs=CRS_STORAGE)
    # Judge each satellite patch by its surroundings: tree canopy inside housing is not a usable green zone
    from rasterstats import zonal_stats
    sm = sat.to_crs(CRS_METRIC)
    ring = sm.copy()
    ring["geometry"] = sm.buffer(RING_M).difference(sm.geometry)
    sat["ring_builtup"] = [100 * (r["mean"] or 0) for r in zonal_stats(ring.to_crs(CRS_STORAGE), str(BU_TIF), stats=["mean"], nodata=-999)]
    sat["patch_ndvi"] = [r["mean"] for r in zonal_stats(sat, str(NDVI_TIF), stats=["mean"], nodata=np.nan)]
    sat["patch_ha"] = sm.area / 1e4
    open_green = (sat.patch_ha >= OPEN_GREEN_HA) & (sat.ring_builtup < MAX_RING_BUILTUP) & (sat.patch_ndvi >= 0.45)
    sat["category"] = np.where(open_green, "open green (satellite)", "neighbourhood tree canopy")
    sat["source"] = "satellite vegetation"
    parks = gpd.read_file(OSM / "parks.gpkg")[["name", "kind", "geometry"]]
    parks = parks[parks.geom_type.isin(["Polygon", "MultiPolygon"])]
    parks = parks[parks.to_crs(CRS_METRIC).area >= MIN_OSM_GREEN_M2].assign(source="OSM", category=lambda x: np.where(x.kind == "park", "park", "garden / other green"))
    g = pd.concat([sat[["source", "category", "geometry"]], parks[["source", "category", "name", "geometry"]]], ignore_index=True)
    g = gpd.GeoDataFrame(g, crs=CRS_STORAGE)
    g["geometry"] = g.geometry.make_valid().simplify(0.00005)
    g = gpd.overlay(g, districts[["district_en", "geometry"]], how="intersection", keep_geom_type=True)
    g["area_m2"] = g.to_crs(CRS_METRIC).area
    g.to_file(out, driver="GPKG")
    return g


def build_grid(districts):
    d = districts.to_crs(CRS_METRIC)
    minx, miny, maxx, maxy = d.total_bounds
    xs = np.arange(np.floor(minx / CELL) * CELL, maxx, CELL)
    ys = np.arange(np.floor(miny / CELL) * CELL, maxy, CELL)
    cells = gpd.GeoDataFrame(geometry=[box(x, y, x + CELL, y + CELL) for x in xs for y in ys], crs=CRS_METRIC)
    cents = cells.copy()
    cents["geometry"] = cells.geometry.centroid
    j = gpd.sjoin(cents, d[["district_en", "geometry"]], how="inner", predicate="within")
    cells = cells.loc[j.index].assign(district_en=j.district_en.values).reset_index(drop=True)
    return cells


def sample(cells_ll, path, name, stat="mean"):
    from rasterstats import zonal_stats
    z = zonal_stats(cells_ll, str(path), stats=[stat], nodata=np.nan)
    return np.array([r[stat] if r[stat] is not None else np.nan for r in z], dtype=float)


def zscore(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


def main():
    ensure_dirs()
    d = gpd.read_file(PROCESSED / "districts.gpkg")
    scores = gpd.read_file(PROCESSED / "district_scores.gpkg").drop(columns="geometry")
    gee_rasters(d.total_bounds)
    green = existing_green(d)
    clean_population()

    cells = build_grid(d)
    cells_ll = cells.to_crs(CRS_STORAGE)
    cells["lst"] = sample(cells_ll, LST_TIF, "lst")
    cells["ndvi"] = sample(cells_ll, NDVI_TIF, "ndvi")
    cells["builtup"] = 100 * sample(cells_ll, BU_TIF, "builtup")
    cells["pop"] = sample(cells_ll, POP30_TIF, "pop", stat="sum")
    cells = cells.merge(scores[["district_en", "elderly_pct"]], on="district_en", how="left")
    cells["pop_density"] = cells["pop"] / (CELL * CELL / 1e6)  # people per km2

    # distance from each cell centre to the nearest existing green zone
    counted = green[green.category != "neighbourhood tree canopy"]
    gu = counted.to_crs(CRS_METRIC).geometry.union_all()
    cells["dist_green_m"] = cells.geometry.centroid.distance(gu)
    gm = counted.to_crs(CRS_METRIC).dissolve(by="district_en").geometry.area
    canopy = green[green.category == "neighbourhood tree canopy"].to_crs(CRS_METRIC).dissolve(by="district_en").geometry.area
    cells["green_in_cell_pct"] = np.nan  # filled below for transparency
    cells["green_in_cell_pct"] = [100 * c.intersection(gu).area / c.area for c in cells.geometry]

    ok = cells[["lst", "ndvi", "builtup", "pop_density"]].notna().all(axis=1)
    cells = cells[ok].reset_index(drop=True)
    cells["priority"] = (zscore(cells.lst) + zscore(cells.builtup) - zscore(cells.ndvi)
                         + zscore(np.log1p(cells.pop_density)))
    thr = cells.priority.quantile(TOP_QUANTILE)
    cells["is_hot"] = (cells.priority >= thr) & (cells.dist_green_m > FAR_FROM_GREEN_M)
    print(f"cells={len(cells)} hot-and-far-from-green={cells.is_hot.sum()} priority threshold={thr:.2f}")

    # hotspot clusters: connected hot cells (8-neighbour) within the same district
    cells["ix"] = (cells.geometry.centroid.x // CELL).astype(int)
    cells["iy"] = (cells.geometry.centroid.y // CELL).astype(int)
    hot = cells[cells.is_hot]
    grid = {(r.ix, r.iy): i for i, r in zip(hot.index, hot.itertuples())}
    parent = {k: k for k in grid}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for (x, y) in grid:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = (x + dx, y + dy)
                if n in grid and cells.at[grid[n], "district_en"] == cells.at[grid[(x, y)], "district_en"]:
                    parent[find(n)] = find((x, y))
    cells["cluster"] = -1
    for k, i in grid.items():
        cells.at[i, "cluster"] = hash(find(k))
    h = cells[cells.cluster != -1]
    hs = h.dissolve(by="cluster", aggfunc={"district_en": "first", "lst": "mean", "ndvi": "mean", "builtup": "mean",
                                           "pop": "sum", "priority": "mean", "dist_green_m": "mean"})
    hs["n_cells"] = h.groupby("cluster").size()
    hs["area_ha"] = hs.geometry.area / 1e4
    hs = hs[hs.n_cells >= 2].reset_index(drop=True)  # ignore single isolated cells
    hs["pop"] = hs["pop"].round(0)
    # a hotspot's need = priority weighted by how many people live in it
    hs["need"] = hs.priority * np.log1p(hs["pop"])
    hs["rank_in_district"] = hs.groupby("district_en").need.rank(ascending=False, method="first").astype(int)
    hs["rank_city"] = hs.need.rank(ascending=False, method="first").astype(int)
    c = hs.geometry.representative_point().to_crs(CRS_STORAGE)
    hs["lat"], hs["lon"] = c.y.values, c.x.values
    hs["hotspot_id"] = [f"{r.district_en} #{r.rank_in_district}" for r in hs.itertuples()]

    # nearest named place from the OSM extract, so each hotspot has a human-readable anchor
    pts = pyogrio.read_dataframe(PBF, layer="points", where="name IS NOT NULL AND place IS NOT NULL",
                                 use_arrow=False)[["name", "geometry"]].to_crs(CRS_METRIC)
    near = gpd.sjoin_nearest(hs.assign(geometry=hs.geometry.representative_point()), pts, how="left", distance_col="near_m")
    near = near[~near.index.duplicated()]
    hs["near_place"] = near["name"].values
    hs["near_place_m"] = near["near_m"].round(0).values

    hs.to_crs(CRS_STORAGE).to_file(PROCESSED / "hotspots.gpkg", driver="GPKG")
    cells.drop(columns=["cluster", "ix", "iy"]).to_crs(CRS_STORAGE).to_file(PROCESSED / "grid_cells.gpkg", driver="GPKG")

    # district-level green accounting
    acc = cells.groupby("district_en").agg(pop=("pop", "sum"), hot_cells=("is_hot", "sum"), cells=("is_hot", "size"))
    acc["green_m2"] = gm.reindex(acc.index).fillna(0)
    acc["canopy_m2"] = canopy.reindex(acc.index).fillna(0)
    acc["green_m2_per_person"] = acc.green_m2 / acc["pop"]
    acc["hot_share_pct"] = 100 * acc.hot_cells / acc.cells
    acc["hotspots"] = hs.groupby("district_en").size().reindex(acc.index).fillna(0).astype(int)
    acc.reset_index().to_csv(PROCESSED / "district_green.csv", index=False)
    print(acc.sort_values("green_m2_per_person").round(1).to_string())
    print("\nTop 10 hotspots citywide")
    print(hs.sort_values("rank_city").head(10)[["hotspot_id", "district_en", "lst", "ndvi", "builtup", "pop", "area_ha",
                                                  "dist_green_m", "near_place"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
