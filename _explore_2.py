
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

#
pd.set_option("display.max_columns", None)   # show all columns
pd.set_option("display.max_rows", None)      # show all rows
pd.set_option("display.width", None)         # don't wrap wide tables
pd.set_option("display.max_colwidth", None)  # don't truncate long cell values

# Read in 1977–2005 burn history (Excel export of the shapefile attribute table)
_path = Path(r"C:\Users\mateo\projects\piFire\FIRE\data\Burn History 1977_2026\1977_2005")
_file = "Burn History 1977-2005 (excel).xls"

# CUT OFF YEAR TO 1996 AND BEYOND
old_df = pd.read_excel(_path / _file, sheet_name=0)
old_df = old_df[old_df["Year_"] >= 1996].reset_index(drop=True)
print(f"Filtered to 1996+: {old_df.shape[0]:,} rows x {old_df.shape[1]} columns")
print(old_df.head(25))

# Export for inspection
# _out = Path.home() / "Desktop" / "old_df_1996_2005.csv"
# old_df.to_csv(_out, index=False)
# print(f"\nWrote {old_df.shape[0]:,} rows x {old_df.shape[1]} columns to {_out}")

# Make a graph sho
# Classify ignition source
def classify(row):
    if row["Type"] == "Prescribed":
        return "Prescribed"
    ig = str(row["Ignition"]).strip()
    if ig == "Lightning":
        return "Lightning"
    if ig in ["Ordnance", "Ordinance", "Flare", "Smokey Sam", "Smoke Grenade",
              "Simulator", "Pyrotechs", "Surfacing EOD"]:
        return "Military"
    return "Unknown"

old_df["source"] = old_df.apply(classify, axis=1)
print(old_df["source"].value_counts())

# Fires per month by source
counts = old_df.groupby(["Month_", "source"]).size().unstack(fill_value=0)
counts = counts.reindex(columns=["Prescribed", "Military", "Lightning", "Unknown"], fill_value=0)

# Area burned per month by source
area = old_df.groupby(["Month_", "source"])["Acres"].sum().unstack(fill_value=0)
area = area.reindex(columns=["Prescribed", "Military", "Lightning", "Unknown"], fill_value=0)

colors = {"Prescribed": "tab:green", "Military": "tab:orange", "Lightning": "yellow", "Unknown": "tab:gray"}

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

counts.plot(kind="bar", stacked=True, ax=axes[0], color=[colors[c] for c in counts.columns])
axes[0].set_ylabel("Number of fires")
axes[0].set_title("Fires per month by ignition source (1996–2005)")

area.plot(kind="bar", stacked=True, ax=axes[1], color=[colors[c] for c in area.columns])
axes[1].set_ylabel("Acres burned")
axes[1].set_xlabel("Month")

plt.tight_layout()
plt.show()



_file = "APAFR_Burn_History_1977_2005.shp"
old_gdf = gpd.read_file(_path / _file)
print(f"Read {old_gdf.shape[0]:,} rows x {old_gdf.shape[1]} columns from {_path / _file}")
print(f"CRS: {old_gdf.crs}")
old_gdf.plot()
plt.show()