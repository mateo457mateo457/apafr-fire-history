# _explore_1.py

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

colors = {"prescribed": "green", "military": "orange", "lightning": "yellow", "wild": "red"}

boundary = gpd.read_file(r"C:\Users\mateo\projects\piFire\FIRE\data\APAFR boundary\boundary.shp")

# ---- NEW file: May 2006
new = gpd.read_file(r"C:\Users\mateo\projects\piFire\FIRE\data\Burn History 1977_2026\2006_2026\APAFR_Burn_Histroy_2006_2026.shp")
new["date"] = pd.to_datetime(new["fireStartD"])
new = new[(new["date"].dt.year == 2006) & (new["date"].dt.month == 5)].copy()
new["cat"] = new["wildland_1"].map(lambda t: "prescribed" if t == "prescribed" else "wild")

# ---- OLD file: May 2006
old = gpd.read_file(r"C:\Users\mateo\projects\Grindstone\APAFR\APAFR 2005\Task 6. Fire\Data\Burn shape files\burn06_final.shp")
old = old[old["Month06"] == 5].copy()

def cat(r):
    if r["Type06"] == "Prescribed": return "prescribed"
    if r["Ignition06"] == "Lightning": return "lightning"
    if r["Type06"] == "Mission" or r["Ignition06"] in ("Ordnance", "Ordinance"): return "military"
    return "wild"

old["cat"] = old.apply(cat, axis=1)

# ---- tables
print("NEW FILE, May 2006")
print(new[["fireId", "date", "wildland_1", "areaSize"]].to_string(index=False))
print("\nOLD FILE, May 2006")
print(old[["FIRENUMBER", "Month06", "Day06", "Type06", "Ignition06", "Acres"]].to_string(index=False))
print("\nOLD FILE acres by category")
print(old.groupby("cat")["Acres"].agg(["size", "sum"]))

# ---- maps, side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, d, idcol, title in [(axes[0], new, "fireId", "New file (2006)"),
                            (axes[1], old, "FIRENUMBER", "Old file (burn06_final)")]:
    boundary.to_crs(d.crs).plot(ax=ax, color="none", edgecolor="gray")
    for c, g in d.groupby("cat"):
        g.plot(ax=ax, color=colors[c], edgecolor="black")
    for _, r in d.iterrows():
        ax.annotate(str(r[idcol]), (r.geometry.centroid.x, r.geometry.centroid.y), fontsize=7)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in colors],
              labels=list(colors), loc="lower left")
    ax.set_title(title)
    ax.set_axis_off()

plt.tight_layout()
plt.show()

# What is the last year of fire records for the old data?
from pathlib import Path

p = Path(r"C:\Users\mateo\projects\Grindstone\APAFR\APAFR 2005\Task 6. Fire\Data\Burn shape files")

for f in sorted(p.rglob("*.shp")):
    print(f.relative_to(p))

# This file is the last year of fire records for the old data; how recent is it?
f08 = gpd.read_file(p / "APAFR_Fires_2008.shp")

print(f08.columns.tolist())
print(len(f08))

print(f08.sort_values(["Month_", "day_", "Year_", "Ignition"]).tail(10))