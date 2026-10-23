"""Simple focus method: where is it HOT and short of GREEN?

Uses only the 250 m grid already built by src.hotspots (no new downloads).
  heat          = percentile rank of LST                          (0-1, 1 = hottest)
  green deficit = mean of percentile ranks of low NDVI and distance to nearest green zone (0-1, 1 = least green)
  focus score   = heat * green deficit   (high only if BOTH are high)
Focus cells = heat >= FOCUS_Q and green deficit >= FOCUS_Q; connected focus cells form focus areas.
Outputs: data/processed/focus_cells.gpkg, focus_areas.gpkg, outputs/focus_by_district.csv
Run: python -m src.focus
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage

from config import CRS_METRIC, CRS_STORAGE, PROCESSED, ROOT

FOCUS_Q = 0.80   # a cell must be in the hottest 20% AND the least-green 20% of the city
MIN_CELLS = 2    # ignore isolated single cells
CELL = 250


def main():
    cells = gpd.read_file(PROCESSED / "grid_cells.gpkg").to_crs(CRS_METRIC)
    cells["heat"] = cells.lst.rank(pct=True)
    cells["green_deficit"] = (
        (-cells.ndvi).rank(pct=True) + cells.dist_green_m.rank(pct=True)
    ) / 2
    cells["focus_score"] = cells.heat * cells.green_deficit
    cells["is_focus"] = (cells.heat >= FOCUS_Q) & (cells.green_deficit >= FOCUS_Q)

    # connected focus cells (8-neighbour) on the grid -> focus areas
    cx, cy = cells.geometry.centroid.x, cells.geometry.centroid.y
    ix, iy = ((cx - cx.min()) // CELL).astype(int), ((cy - cy.min()) // CELL).astype(int)
    mat = np.zeros((ix.max() + 1, iy.max() + 1), dtype=bool)
    mat[ix[cells.is_focus], iy[cells.is_focus]] = True
    lab, _ = ndimage.label(mat, structure=np.ones((3, 3)))
    cells["area_id"] = np.where(cells.is_focus, lab[ix, iy], 0)

    f = cells[cells.area_id > 0]
    areas = f.dissolve(by="area_id", aggfunc={"district_en": lambda s: s.mode().iat[0], "lst": "mean", "ndvi": "mean",
                                              "dist_green_m": "mean", "focus_score": "mean", "pop": "sum"})
    areas["n_cells"] = f.groupby("area_id").size()
    areas = areas[areas.n_cells >= MIN_CELLS].copy()
    areas["area_ha"] = areas.geometry.area / 1e4
    areas["rank"] = areas.focus_score.mul(areas.area_ha).rank(ascending=False, method="first").astype(int)
    areas = areas.sort_values("rank").reset_index()

    cells.to_crs(CRS_STORAGE).drop(columns="area_id").to_file(PROCESSED / "focus_cells.gpkg", driver="GPKG")
    areas.to_crs(CRS_STORAGE).to_file(PROCESSED / "focus_areas.gpkg", driver="GPKG")

    by_d = (cells.groupby("district_en")
            .agg(cells=("is_focus", "size"), focus_cells=("is_focus", "sum"), mean_lst=("lst", "mean"),
                 mean_ndvi=("ndvi", "mean"))
            .assign(focus_share_pct=lambda d: 100 * d.focus_cells / d.cells)
            .sort_values("focus_share_pct", ascending=False).round(2))
    out = ROOT / "outputs" / "focus_by_district.csv"
    by_d.to_csv(out)

    print(f"focus cells: {cells.is_focus.sum()} of {len(cells)} ({100 * cells.is_focus.mean():.1f}%), "
          f"{len(areas)} focus areas (>= {MIN_CELLS} cells), {areas.area_ha.sum():,.0f} ha")
    print("\nDistricts with the largest share of focus cells")
    print(by_d.head(10).to_string())
    print("\nTop 10 focus areas")
    print(areas.head(10)[["rank", "district_en", "lst", "ndvi", "dist_green_m", "area_ha", "pop"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
