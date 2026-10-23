"""Phase 5: Bangkok heat map. Reads only local processed files (no Earth Engine at runtime).

Run: streamlit run app/streamlit_app.py
"""
import base64
import io
import sys
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely
import rasterio
from affine import Affine
from rasterio.features import geometry_mask
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import PROCESSED, RAW  # noqa: E402

SCORES = PROCESSED / "district_scores.gpkg"
LST_TIF, NDVI_TIF = PROCESSED / "lst_landsat.tif", PROCESSED / "ndvi.tif"
OSM = RAW / "osm"

st.set_page_config(page_title="Bangkok heat map", layout="wide")


@st.cache_data
def load_scores():
    return gpd.read_file(SCORES)


@st.cache_data
def raster_overlay(path, cmap, _geom, vmin_q=2, vmax_q=98, step=3):
    """Colour a raster to a PNG data-URI for a folium ImageOverlay, clipped to Bangkok (_geom).
    Returns (uri, bounds, vmin, vmax)."""
    with rasterio.open(path) as src:
        a = src.read(1)[::step, ::step].astype("float32")
        b = src.bounds
        inside = ~geometry_mask([_geom], out_shape=a.shape, transform=src.transform * Affine.scale(step))
    a[~np.isfinite(a)] = np.nan
    a[~inside] = np.nan  # nothing is drawn outside the Bangkok border
    vmin, vmax = np.nanpercentile(a, [vmin_q, vmax_q])
    rgba = matplotlib.colormaps[cmap](np.clip((a - vmin) / (vmax - vmin), 0, 1))
    rgba[np.isnan(a), 3] = 0
    buf = io.BytesIO()
    Image.fromarray((rgba * 255).astype("uint8")).save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return uri, [[b.bottom, b.left], [b.top, b.right]], float(vmin), float(vmax)


@st.cache_data
def load_osm(name):
    g = gpd.read_file(OSM / f"{name}.gpkg")[["geometry"]]
    g["geometry"] = g.geometry.simplify(0.0001)
    return g


@st.cache_data
def load_hotspots():
    return gpd.read_file(PROCESSED / "hotspots.gpkg")


@st.cache_data
def load_focus():
    """Output of src.focus (hot AND short of green); empty if it has not been run."""
    p = PROCESSED / "focus_areas.gpkg"
    return gpd.read_file(p) if p.exists() else gpd.GeoDataFrame(geometry=[], crs=4326)


@st.cache_data
def load_green():
    g = gpd.read_file(PROCESSED / "green_zones.gpkg")[["district_en", "category", "name", "area_m2", "geometry"]]
    g["geometry"] = shapely.set_precision(g.geometry.values, 1e-5)  # ~1 m, keeps citywide GeoJSON small
    return g[g.area_m2 >= 1000]


@st.cache_data
def load_cells():
    return gpd.read_file(PROCESSED / "grid_cells.gpkg")


@st.cache_data
def load_green_stats():
    return pd.read_csv(PROCESSED / "district_green.csv").set_index("district_en")


WHO = 9  # m2 of green space per person, commonly cited minimum (WHO)
scores = load_scores()
bkk_geom = scores.geometry.union_all()
hotspots, green, gstats = load_hotspots(), load_green(), load_green_stats()
names = scores.sort_values("district_en").district_en.tolist()

# ---- sidebar -------------------------------------------------------------
st.sidebar.title("Bangkok heat risk")
if "district" not in st.session_state:
    st.session_state.district = names[0]
sel = st.sidebar.selectbox("District", names, key="district")
r = scores[scores.district_en == sel].iloc[0]
st.sidebar.subheader(f"{r.district_en}  ({r.district_th})")
st.sidebar.metric("Heat-risk rank", f"{int(r['rank'])} of {len(scores)}", f"score {r.heat_risk:.2f}", delta_color="off")
c1, c2 = st.sidebar.columns(2)
c1.metric("Mean LST", f"{r.mean_lst:.1f} °C")
c2.metric("Mean NDVI", f"{r.mean_ndvi:.2f}")
c1.metric("Built-up", f"{r.builtup_pct:.0f}%")
c2.metric("Elderly", f"{r.elderly_pct:.1f}%")
c1.metric("Green space", f"{r.green_pct:.1f}%")
c2.metric("Dist. to water", f"{r.mean_dist_water_m:.0f} m")
st.sidebar.caption(f"OSM buildings: {int(r.building_count):,} (OSM coverage is incomplete)")

st.sidebar.subheader("Top-10 hottest (heat risk)")
top = scores.sort_values("rank").head(10)[["rank", "district_en", "heat_risk", "mean_lst"]]
st.sidebar.dataframe(top.rename(columns={"district_en": "district", "heat_risk": "score", "mean_lst": "LST °C"})
                     .round(2), hide_index=True, width="stretch")

st.sidebar.subheader("Layers")
show_lst = st.sidebar.checkbox("Land surface temperature", True)
show_ndvi = st.sidebar.checkbox("NDVI (greenness)", False)
show_osm = st.sidebar.checkbox("Parks and water (OSM)", False)
show_risk = st.sidebar.checkbox("District choropleth", True)
COLOUR_BY = {  # label -> (column, colormap, legend)
    "Heat-risk score (heat + exposure)": ("heat_risk", "RdYlBu_r", "Heat-risk score"),
    "Surface temperature (LST only)": ("mean_lst", "YlOrRd", "Mean LST (°C)"),
    "Built-up share": ("builtup_pct", "YlOrRd", "Built-up (%)"),
    "Greenness (NDVI)": ("mean_ndvi", "YlGn", "Mean NDVI"),
    "Elderly share": ("elderly_pct", "PuRd", "Elderly (%)"),
}
colour_by = st.sidebar.selectbox("Colour districts by", list(COLOUR_BY), disabled=not show_risk)
show_green = st.sidebar.checkbox("Existing green zones (highlighted)", True)
shown_types = st.sidebar.multiselect(
    "Green zone types to show", ["park", "garden / other green", "open green (satellite)"],
    default=["park", "garden / other green", "open green (satellite)"],
    help="Dark green = parks, light green = gardens and other OSM green, pale mint = open green seen by satellite")
show_canopy = st.sidebar.checkbox("Tree canopy in neighbourhoods (not counted as green zone)", False)
show_hot = st.sidebar.checkbox("Where to add green space (hotspots)", True)
show_focus = st.sidebar.checkbox("Focus areas: hot + short of green (src.focus)", True,
                                 help="Hottest 20% AND least-green 20% of 250 m cells; orange outlines, ranked by score x area")
satellite = st.sidebar.checkbox("Satellite imagery base map", False, help="Use it to check whether a focus area is really housing, or an airport / factory / empty land")
zoom = st.sidebar.checkbox("Zoom to selected district", True, help="Only moves the view; layers are always drawn for the whole city")
opacity = st.sidebar.slider("Raster opacity", 0.2, 1.0, 0.7)

# ---- map -----------------------------------------------------------------
st.title("Which Bangkok districts are hottest, why, and who is most exposed?")
st.caption("LST: Landsat 8/9, Mar–May 2023–25 median (~10:30 am overpass, surface not air temperature). "
           "Heat risk = z(LST) + z(built-up %) − z(NDVI) + z(elderly %).")
m = folium.Map(location=[13.75, 100.6], zoom_start=10, tiles="OpenStreetMap",
               zoomAnimation=False, fadeAnimation=False, markerZoomAnimation=False)  # no animated blink on zoom
if satellite:
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Satellite").add_to(m)
if show_lst:
    uri, bnds, lo, hi = raster_overlay(str(LST_TIF), "inferno", bkk_geom)
    folium.raster_layers.ImageOverlay(uri, bnds, opacity=opacity, name=f"LST {lo:.0f}–{hi:.0f} °C").add_to(m)
if show_ndvi:
    uri, bnds, lo, hi = raster_overlay(str(NDVI_TIF), "RdYlGn", bkk_geom)
    folium.raster_layers.ImageOverlay(uri, bnds, opacity=opacity, name=f"NDVI {lo:.2f}–{hi:.2f}").add_to(m)
if show_osm:
    folium.GeoJson(load_osm("parks").to_json(), name="Parks",
                   style_function=lambda f: {"color": "#1a7f37", "fillColor": "#1a7f37", "weight": 0.5, "fillOpacity": 0.6}).add_to(m)
    folium.GeoJson(load_osm("water").to_json(), name="Water",
                   style_function=lambda f: {"color": "#1f77b4", "fillColor": "#1f77b4", "weight": 1, "fillOpacity": 0.6}).add_to(m)

geo = scores[["district_en", "district_th", "rank", "heat_risk", "mean_lst", "geometry"]]
if show_risk:
    col, cmap, legend = COLOUR_BY[colour_by]
    folium.Choropleth(geo_data=geo.to_json(), data=scores, columns=["district_en", col],
                      key_on="feature.properties.district_en", fill_color=cmap, fill_opacity=0.35,
                      line_opacity=0.4, legend_name=legend).add_to(m)
folium.GeoJson(
    geo.to_json(), name="Districts",
    style_function=lambda f: {"color": "#000" if f["properties"]["district_en"] == sel else "#444",
                              "weight": 3 if f["properties"]["district_en"] == sel else 0.5, "fillOpacity": 0},
    tooltip=folium.GeoJsonTooltip(fields=["district_en", "rank", "heat_risk", "mean_lst"],
                                  aliases=["District", "Rank", "Score", "LST °C"], localize=True)).add_to(m)
counted_cats = ["park", "garden / other green", "open green (satellite)"]
gv = green  # every layer is drawn citywide; the zoom option only moves the view
STYLE = {"park": ("#004d1f", "#00963c", 0.9),                       # dark green
         "garden / other green": ("#5fae6b", "#a5e8a0", 0.8),        # light green
         "open green (satellite)": ("#7fb8a4", "#c9f0e2", 0.6)}      # pale mint
if show_green:
    for cat in [c for c in counted_cats if c in shown_types]:
        part = gv[gv.category == cat]
        if len(part):
            ln, fl, op = STYLE[cat]
            part = part.assign(name=part.name.fillna("(unnamed)"))
            folium.GeoJson(part[["category", "name", "geometry"]].to_json(), name=cat,
                           style_function=lambda f, ln=ln, fl=fl, op=op: {"color": ln, "fillColor": fl, "weight": 1.2, "fillOpacity": op},
                           tooltip=folium.GeoJsonTooltip(fields=["category", "name"], aliases=["Type", "Name"])).add_to(m)
if show_canopy:
    part = gv[gv.category == "neighbourhood tree canopy"]
    if len(part):
        folium.GeoJson(part[["category", "geometry"]].to_json(), name="Tree canopy (not counted)",
                       style_function=lambda f: {"color": "#8d9c6b", "fillColor": "#c5e1a5", "weight": 0.5, "fillOpacity": 0.35,
                                                 "dashArray": "3"},
                       tooltip=folium.GeoJsonTooltip(fields=["category"], aliases=["Not counted as green zone"])).add_to(m)
if show_hot:
    hv = hotspots  # citywide
if show_hot and len(hv):
    folium.GeoJson(hv[["hotspot_id", "district_en", "lst", "pop", "area_ha", "near_place", "geometry"]].to_json(), name="Hotspots",
                   style_function=lambda f: {"color": "#d50000", "fillColor": "#ff1744",
                                             "weight": 3 if f["properties"]["district_en"] == sel else 1.5,
                                             "fillOpacity": 0.5 if f["properties"]["district_en"] == sel else 0.3,
                                             "dashArray": "4"},
                   tooltip=folium.GeoJsonTooltip(fields=["hotspot_id", "lst", "pop", "area_ha", "near_place"],
                                                 aliases=["Hotspot", "LST °C", "People", "Area (ha)", "Near"],
                                                 localize=True)).add_to(m)
focus = load_focus()
if show_focus and len(focus):
    folium.GeoJson(focus[["rank", "district_en", "lst", "ndvi", "dist_green_m", "area_ha", "pop", "geometry"]].to_json(),
                   name="Focus areas",
                   style_function=lambda f: {"color": "#e65100", "fillColor": "#ff9800", "weight": 2, "fillOpacity": 0.35},
                   tooltip=folium.GeoJsonTooltip(fields=["rank", "district_en", "lst", "ndvi", "dist_green_m", "area_ha", "pop"],
                                                 aliases=["Rank", "District", "LST °C", "NDVI", "Dist. to green (m)", "Area (ha)", "People"],
                                                 localize=True)).add_to(m)
elif show_focus:
    st.sidebar.warning("No focus areas found. Run: python -m src.focus")
if zoom:
    x0, y0, x1, y1 = scores[scores.district_en == sel].total_bounds
    m.fit_bounds([[y0, x0], [y1, x1]])
folium.LayerControl().add_to(m)

out = st_folium(m, key="bkk_map", height=650, width="stretch", returned_objects=["last_active_drawing"])
clicked = (out or {}).get("last_active_drawing")
if clicked:
    name = clicked.get("properties", {}).get("district_en")
    if name in names and name != st.session_state.district:
        st.session_state.district = name
        st.rerun()


# ---- district analysis ----------------------------------------------------
st.divider()
st.header(f"{sel}: what is going on and where to add green space")
gs = gstats.loc[sel]
hs = hotspots[hotspots.district_en == sel].sort_values("rank_in_district")
per = gs.green_m2_per_person
vs = "below" if per < WHO else "above"
heat_word = "very high" if r["rank"] <= 10 else "high" if r["rank"] <= 20 else "moderate" if r["rank"] <= 35 else "lower"
lines = [
    f"**Heat.** {sel} ranks **{int(r['rank'])} of {len(scores)}** for heat risk ({heat_word}). Its land is on average "
    f"**{r.mean_lst:.1f} °C** at the ~10:30 am satellite pass, with **{r.builtup_pct:.0f}%** built-up land and mean NDVI "
    f"**{r.mean_ndvi:.2f}**. About **{r.elderly_pct:.1f}%** of registered residents receive the elderly allowance.",
    f"**Existing green.** About **{gs.green_m2 / 1e4:,.0f} ha** of counted green zones (OSM parks and gardens plus satellite open green of ≥ 2 ha "
    f"away from dense housing), roughly **{per:.1f} m² per person** ({vs} the commonly cited WHO minimum of {WHO} m²/person). "
    f"A further **{gs.canopy_m2 / 1e4:,.0f} ha** is tree canopy inside neighbourhoods; it cools streets but is not counted "
    f"as usable green space (toggle it in the sidebar). **Dark green = parks, light green = gardens and other OSM green, pale mint = open green seen by satellite.**",
]
if len(hs):
    pop_hot, ha_hot = int(hs["pop"].sum()), hs.area_ha.sum()
    top1 = hs.iloc[0]
    near = f", near **{top1.near_place}**" if isinstance(top1.near_place, str) else ""
    lines.append(
        f"**Where to add green space.** **{len(hs)}** hotspot(s) cover **{ha_hot:,.0f} ha** where about **{pop_hot:,}** "
        f"people live: hot, almost fully built-up, and more than ~400 m from any green zone. The top priority is "
        f"**{top1.hotspot_id}**{near} (LST {top1.lst:.1f} °C, {top1.builtup:.0f}% built-up, about {int(top1['pop']):,} people, "
        f"{top1.area_ha:.0f} ha). Hotspots are drawn as red dashed outlines.")
else:
    why = (f"almost every part of the district is within ~400 m of some green patch, but the total is only "
           f"**{per:.1f} m² per person**, so the issue is how little green there is, not how far away it is"
           if per < WHO else "the district already has enough green space per person and no hot, built-up area is "
           "far from it")
    lines.append(f"**Where to add green space.** No hotspot met the strict criteria (top 15% priority and > 400 m from a "
                 f"green zone): {why}. The hottest 250 m cells are listed below as candidates for pocket parks, street "
                 f"trees or green roofs.")
st.markdown("\n\n".join(lines))

if len(hs):
    tab = hs[["hotspot_id", "near_place", "lst", "builtup", "ndvi", "pop", "area_ha", "dist_green_m", "lat", "lon"]].copy()
    tab["map"] = [f"https://www.google.com/maps?q={a:.5f},{b:.5f}" for a, b in zip(tab.lat, tab.lon)]
    tab = tab.drop(columns=["lat", "lon"]).rename(columns={
        "hotspot_id": "Hotspot", "near_place": "Near", "lst": "LST °C", "builtup": "Built-up %", "ndvi": "NDVI",
        "pop": "People", "area_ha": "Area (ha)", "dist_green_m": "Dist. to green (m)", "map": "Open map"})
    st.dataframe(tab.round(1), hide_index=True, width="stretch",
                 column_config={"Open map": st.column_config.LinkColumn("Open map", display_text="Google Maps")})
if not len(hs):
    cc = load_cells()
    cc = cc[cc.district_en == sel].sort_values("priority", ascending=False).head(5).copy()
    c = cc.geometry.to_crs(32647).centroid.to_crs(4326)  # centroid in a metric CRS, then back to lat/lon
    cc["lat"], cc["lon"] = c.y, c.x
    cc["Open map"] = [f"https://www.google.com/maps?q={a:.5f},{b:.5f}" for a, b in zip(cc.lat, cc.lon)]
    st.subheader("Hottest 250 m cells (fallback candidates)")
    st.dataframe(cc[["priority", "lst", "builtup", "ndvi", "pop", "dist_green_m", "Open map"]].rename(columns={
        "priority": "Priority", "lst": "LST °C", "builtup": "Built-up %", "ndvi": "NDVI", "pop": "People",
        "dist_green_m": "Dist. to green (m)"}).round(1), hide_index=True, width="stretch",
        column_config={"Open map": st.column_config.LinkColumn("Open map", display_text="Google Maps")})
st.caption("Hotspot method: 250 m grid; priority = z(LST) + z(built-up %) − z(NDVI) + z(log population density); "
           "hotspots are connected clusters of top-15% priority cells more than 400 m from existing green zones. "
           "Population is WorldPop 2020. Counted green zones are OSM parks plus satellite patches of ≥ 2 ha "
           "(WorldCover tree/shrub/grass, NDVI ≥ 0.45) with < 40% built-up land around them; smaller patches and canopy "
           "inside housing are not counted. OSM parks are incomplete, so green per person is a lower bound. "
           "Planning aid only; check sites on the ground.")
