"""Phase 3: compare Landsat LST with Open-Meteo 2 m air temperature at TMD station points.

LST is a seasonal median of ~10:30 am surface temperature, so the matching air temperature is the
Mar-May median of the 10:00/11:00 hourly values (interpolated to 10:30).
Outputs: outputs/validation.csv, outputs/figures/validation.png. Run: python -m src.validate
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from scipy.stats import pearsonr

from config import FIGURES, OVERPASS_HOUR, RAW, ROOT, SEASON_MONTHS, STATIONS, YEARS
from src.lst import LST_TIF
from src.utils import ensure_dirs

API = "https://archive-api.open-meteo.com/v1/archive"


def air_temp(lat, lon):
    cache = RAW / "openmeteo" / f"{lat}_{lon}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["time"])
    frames = []
    for y in YEARS:
        r = requests.get(API, params={
            "latitude": lat, "longitude": lon, "hourly": "temperature_2m", "timezone": "Asia/Bangkok",
            "start_date": f"{y}-{SEASON_MONTHS[0]:02d}-01", "end_date": f"{y}-{SEASON_MONTHS[1]:02d}-31"},
            timeout=60)
        r.raise_for_status()
        h = r.json()["hourly"]
        frames.append(pd.DataFrame({"time": pd.to_datetime(h["time"]), "t2m": h["temperature_2m"]}))
    df = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def air_temp_at_overpass(df):
    h0, h1 = int(OVERPASS_HOUR), int(OVERPASS_HOUR) + 1
    w = OVERPASS_HOUR - h0
    d = df.assign(date=df.time.dt.date, hour=df.time.dt.hour)
    p = d[d.hour.isin([h0, h1])].pivot(index="date", columns="hour", values="t2m").dropna()
    return float(((1 - w) * p[h0] + w * p[h1]).median())


def lst_at(src, lat, lon, half=1):
    r, c = src.index(lon, lat)
    win = src.read(1, window=((r - half, r + half + 1), (c - half, c + half + 1)))
    return float(np.nanmean(win)), float(src.read(1)[r, c])


def main():
    ensure_dirs()
    rows = []
    with rasterio.open(LST_TIF) as src:
        for name, (lat, lon) in STATIONS.items():
            lst3, lst1 = lst_at(src, lat, lon)
            ta = air_temp_at_overpass(air_temp(lat, lon))
            rows.append({"station": name, "lat": lat, "lon": lon, "lst_3x3_mean_c": lst3,
                         "lst_pixel_c": lst1, "air_temp_1030_c": ta, "lst_minus_air_c": lst3 - ta})
    v = pd.DataFrame(rows).round(2)
    v.to_csv(ROOT / "outputs" / "validation.csv", index=False)
    print(v.to_string(index=False))
    r, p = pearsonr(v.air_temp_1030_c, v.lst_3x3_mean_c)
    print(f"\nPearson r (n={len(v)}) = {r:.2f} (p={p:.2f});  mean LST - air = {v.lst_minus_air_c.mean():.2f} degC")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(v.air_temp_1030_c, v.lst_3x3_mean_c)
    for _, s in v.iterrows():
        ax.annotate(s.station.split(" (")[0], (s.air_temp_1030_c, s.lst_3x3_mean_c), fontsize=7)
    lim = [min(v.air_temp_1030_c.min(), v.lst_3x3_mean_c.min()) - 1, max(v.air_temp_1030_c.max(), v.lst_3x3_mean_c.max()) + 1]
    ax.plot(lim, lim, "k--", lw=0.5, label="1:1")
    ax.set_xlabel("Open-Meteo 2 m air temp at 10:30 (°C, Mar–May median)")
    ax.set_ylabel("Landsat LST, 3×3 px mean (°C)")
    ax.set_title(f"LST vs air temperature at stations (n={len(v)}, r={r:.2f})")
    ax.legend()
    plt.savefig(FIGURES / "validation.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
