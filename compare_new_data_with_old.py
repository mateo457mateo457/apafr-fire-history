# compare_new_data_with_old.py

"""
Here we compare the recent fire data to the old fire data that was used in
several papers.

The old data was originally in two SAS files (prescribedfires.sas7bdat and
wildfires.sas7bdat), which were compiled into a single CSV file
(2010_fire_data/old_fire_data.csv) for easier use. Area burned by source in
the CSV matches the SAS files exactly for Oct 1996 - Sep 2009 (FY1997-FY2009).

The new data is in a geopackage (fire_data.gpkg) produced by
process_fire_data.py.

The papers that used the old data include:

    Slocum, M. G., B. Beckage, W. J. Platt, S. L. Orzell, and W. Taylor. 2010.
    Effect of climate on wildfire size: a cross-scale analysis. Ecosystems
    13:828-840. https://doi.org/10.1007/s10021-010-9357-y

    Slocum, M. G., W. J. Platt, B. Beckage, S. L. Orzell, and W. Taylor. 2010.
    Accurate quantification of seasonal rainfall and associated
    climate-wildfire relationships. Journal of Applied Meteorology and
    Climatology 49:2559-2573. https://doi.org/10.1175/2010JAMC2532.1

    Platt WJ, Orzell SL, Slocum MG (2015) Seasonality of Fire Weather Strongly
    Influences Fire Regimes in South Florida Savanna-Grassland Landscapes.
    PLoS ONE 10(1): e0116952. doi:10.1371/journal.pone.0116952

    Slocum MG, Orzell SL (2013) A Structural Equation Model Analysis of
    Relationships among ENSO, Seasonal Descriptors and Wildfires. PLoS ONE
    8(9): e75946. doi:10.1371/journal.pone.0075946

Conclusion: over the common record period (FY1997-FY2009), total area burned
in the new data is within 3% of the old data (+4,551 ha; Prescribed +3,366 ha,
Military +1,286 ha, Lightning -101 ha). The difference is mainly due to fires
of unknown origin: the old data dropped them, while the new data assigns them
a predicted ignition source (see process_fire_data.py).
"""

# Packages
from pathlib import Path
import pandas as pd
import geopandas as gpd
import calendar
import numpy as np
from matplotlib import pyplot as plt

# Options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

# Folders
data_folder = Path.cwd() / "data"
raw_data_folder = data_folder / "raw" / "fire"
processed_data_folder = data_folder / "processed" / "fire"

# Read in old fire data (2010 compilation)
_path = raw_data_folder / "2010_fire_data"
_file = "old_fire_data.csv"
old = pd.read_csv(_path / _file)
print(f"Read {old.shape[0]:,} rows x {old.shape[1]} columns from {_path / _file}")

# Read in new fire data (processed output)
_path = processed_data_folder
_file = "fire_data.gpkg"
new = gpd.read_file(_path / _file, layer="fire_data")
print(f"Read {new.shape[0]:,} rows x {new.shape[1]} columns from {_path / _file}")

# --- Dates and fiscal years (FY starts in October) ---
old["DATE"] = pd.to_datetime(old["DATE"])
old["FY"] = old["DATE"].dt.year + (old["DATE"].dt.month >= 10).astype(int)

new["date"] = pd.to_datetime(new["date"])
new["FY"] = new["date"].dt.year + (new["date"].dt.month >= 10).astype(int)

# --- Common fiscal years: the old record period ---
common_fy = list(range(1997, 2010))  # FY1997-FY2009 (Oct 1996 - Sep 2009)

# --- Old: area burned per FY by source ---
old_fy = (
    old[old["FY"].isin(common_fy)]
    .groupby("FY")[["ha_prescribed", "ha_ordnance", "ha_lightning"]].sum()
    .rename(columns={"ha_prescribed": "Prescribed",
                     "ha_ordnance": "Military",
                     "ha_lightning": "Lightning"})
    .reindex(common_fy, fill_value=0)
)

# --- New: area burned per FY by source ---
new_fy = (
    new[new["FY"].isin(common_fy)]
    .pivot_table(index="FY", columns="source", values="ha", aggfunc="sum", fill_value=0)
    .reindex(index=common_fy, columns=["Prescribed", "Military", "Lightning"], fill_value=0)
)

# --- Overall totals ---
totals = pd.DataFrame({"old_ha": old_fy.sum(), "new_ha": new_fy.sum()})
totals.loc["Total"] = totals.sum()
totals["diff_ha"] = totals["new_ha"] - totals["old_ha"]
totals["diff_pct"] = 100 * totals["diff_ha"] / totals["old_ha"]
print(f"Area burned (ha), FY{common_fy[0]}-FY{common_fy[-1]}")
print(totals.round(1))

# --- Per-FY differences (new - old) ---
print("\nDifference by FY (new - old, ha)")
print((new_fy - old_fy).round(0))

# ============================================================
# Figures: old vs new
# ============================================================
checks_folder = processed_data_folder / "data_checks"
checks_folder.mkdir(parents=True, exist_ok=True)

sources = ["Prescribed", "Military", "Lightning"]
colors = {"Prescribed": "tab:green", "Military": "tab:orange", "Lightning": "yellow"}  # same as process_fire_data.py


def plot_old_vs_new(old_df, new_df, labels, xlabel, title, out_name):
    """Three stacked panels (one per source), old and new bars side by side."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    x = np.arange(len(labels))
    w = 0.4
    for ax, s in zip(axes, sources):
        ax.bar(x - w / 2, old_df[s], w, color="lightgray", edgecolor="black", linewidth=0.3, label="Old")
        ax.bar(x + w / 2, new_df[s], w, color=colors[s], edgecolor="black", linewidth=0.3, label="New")
        ax.set_ylabel(f"{s} (ha)")
        ax.legend(loc="upper right", frameon=False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel(xlabel)
    axes[0].set_title(title)
    plt.tight_layout()

    _out = checks_folder / out_name
    fig.savefig(_out, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Wrote figure to {_out}")


# --- Figure 1: area burned by month (FY order, Oct-Sep), old vs new ---
month_order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
month_labels = [calendar.month_abbr[m] for m in month_order]

old_c = old[old["FY"].isin(common_fy)]
new_c = new[new["FY"].isin(common_fy)]

old_month = (
    old_c.groupby(old_c["DATE"].dt.month)[["ha_prescribed", "ha_ordnance", "ha_lightning"]].sum()
    .rename(columns={"ha_prescribed": "Prescribed",
                     "ha_ordnance": "Military",
                     "ha_lightning": "Lightning"})
    .reindex(month_order, fill_value=0)
)

new_month = (
    new_c.groupby([new_c["date"].dt.month, "source"])["ha"].sum()
    .unstack(fill_value=0)
    .reindex(index=month_order, columns=sources, fill_value=0)
)

plot_old_vs_new(old_month, new_month, month_labels, "Month",
                f"Area burned by month, FY{common_fy[0]}-FY{common_fy[-1]}: old vs new",
                "compare_area_by_month.png")

# --- Figure 2: area burned by fiscal year, old vs new ---
plot_old_vs_new(old_fy, new_fy, [str(fy) for fy in common_fy], "Fiscal year",
                f"Area burned by fiscal year, FY{common_fy[0]}-FY{common_fy[-1]}: old vs new",
                "compare_area_by_fy.png")

# --- Figure 3: old vs new by fiscal year, with 1:1 line ---
# Each source gets its own panel and scale, since Prescribed is ~10x the others.
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, s in zip(axes, sources):
    ax.scatter(old_fy[s], new_fy[s], color=colors[s], edgecolor="black", s=45, zorder=3)
    lim = max(old_fy[s].max(), new_fy[s].max()) * 1.08
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, zorder=2)
    for fy in common_fy:
        ax.annotate(str(fy)[2:], (old_fy.loc[fy, s], new_fy.loc[fy, s]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_title(s)
    ax.set_xlabel("Old (ha)")
axes[0].set_ylabel("New (ha)")
fig.suptitle(f"Area burned per fiscal year, old vs new (labels = FY; dashed = 1:1)")
plt.tight_layout()

_out = checks_folder / "compare_scatter_by_fy.png"
fig.savefig(_out, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"Wrote figure to {_out}")

# --- Figure 4: share of total area burned by source, old vs new ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (name, df) in zip(axes, [("Old", old_fy), ("New", new_fy)]):
    tot = df[sources].sum()
    ax.pie(tot, labels=sources, colors=[colors[s] for s in sources],
           autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="black", linewidth=0.5))
    ax.set_title(f"{name}: {tot.sum():,.0f} ha")
fig.suptitle(f"Share of area burned by source, FY{common_fy[0]}-FY{common_fy[-1]}")
plt.tight_layout()

_out = checks_folder / "compare_pie_by_source.png"
fig.savefig(_out, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"Wrote figure to {_out}")