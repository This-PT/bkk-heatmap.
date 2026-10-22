"""Phase 4: district heat-risk score -> data/processed/district_scores.gpkg, outputs/figures/heat_risk.png.

heat_risk = z(LST) + z(built-up %) - z(NDVI) + z(elderly %)
  * built-up % is ESA WorldCover (OSM building share is too incomplete; see README)
  * elderly % = BMA elderly-allowance recipients (Sep 2566 = Sep 2023) / registered population
Run: python -m src.score
"""
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES, PROCESSED, RAW
from src.utils import ensure_dirs

ELDERLY_CSV = RAW / "elderly_recipients_2566.csv"
POP_CSV = RAW / "population.csv"
DISTRICT_ZIP = RAW / "boundaries" / "district-2.zip"
COMPONENTS = {"mean_lst": +1, "builtup_pct": +1, "mean_ndvi": -1, "elderly_pct": +1}


def build_population():
    """Registered population (male+female) per district from the BMA boundary dataset + elderly counts."""
    if POP_CSV.exists():
        return pd.read_csv(POP_CSV)
    g = gpd.read_file(f"zip://{DISTRICT_ZIP}", encoding="cp874")
    pop = pd.DataFrame({"district_th": g.dname, "population": g.num_male + g.num_female})
    e = pd.read_csv(ELDERLY_CSV)
    e = e[(e.calendar_year == 2566) & (e.monthly == "กันยายน")]
    e["district_th"] = e.district.str.strip().str.replace("ฎ", "ฏ")  # boundary file spells ราษฏร์บูรณะ with ฏ
    e = e.drop_duplicates("district_th")[["district_th", "number_of_elder"]]
    pop["key"] = pop.district_th.str.replace("^เขต", "", regex=True).str.strip()
    out = pop.merge(e.rename(columns={"district_th": "key"}), on="key", how="left").drop(columns="key")
    out.to_csv(POP_CSV, index=False)
    return out


def main():
    ensure_dirs()
    d = gpd.read_file(PROCESSED / "districts.gpkg")
    f = pd.read_csv(PROCESSED / "district_features.csv")
    pop = build_population()
    pop["district_th"] = pop.district_th.str.strip()
    df = f.merge(pop, on="district_th", how="left", validate="1:1")
    assert df.number_of_elder.notna().all(), df[df.number_of_elder.isna()].district_en.tolist()
    df["elderly_pct"] = 100 * df.number_of_elder / df.population
    for c, sign in COMPONENTS.items():
        df[f"z_{c}"] = sign * (df[c] - df[c].mean()) / df[c].std(ddof=0)
    df["heat_risk"] = df[[f"z_{c}" for c in COMPONENTS]].sum(axis=1)
    df["rank"] = df.heat_risk.rank(ascending=False).astype(int)
    df = df.sort_values("rank")
    cols = ["rank", "district_en", "heat_risk", "mean_lst", "builtup_pct", "mean_ndvi", "elderly_pct"]
    print("TOP 10 (highest heat risk)\n", df[cols].head(10).round(2).to_string(index=False))
    print("\nBOTTOM 5\n", df[cols].tail(5).round(2).to_string(index=False))
    print(f"\nelderly_pct: min {df.elderly_pct.min():.1f}  median {df.elderly_pct.median():.1f}  max {df.elderly_pct.max():.1f}")
    print(df[list(COMPONENTS)].corr().round(2).to_string())

    out = d.merge(df.drop(columns=["district_en"]), on="district_th", validate="1:1")
    out.to_file(PROCESSED / "district_scores.gpkg", driver="GPKG")
    ax = out.plot(column="heat_risk", cmap="RdYlBu_r", legend=True, figsize=(9, 8), edgecolor="k",
                  linewidth=0.3, legend_kwds={"label": "heat-risk score (sum of z-scores)"})
    for _, r in out[out["rank"] <= 10].iterrows():
        p = r.geometry.representative_point()
        ax.annotate(r.district_en, (p.x, p.y), fontsize=5, ha="center")
    ax.set_title("Bangkok district heat risk (top 10 labelled)")
    ax.set_axis_off()
    plt.savefig(FIGURES / "heat_risk.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
