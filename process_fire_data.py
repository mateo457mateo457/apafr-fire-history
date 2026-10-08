# process_fire_data.py

"""
Builds the APAFR fire history: one row per fire event (not per raw
polygon), with a date, size (ha), and ignition source, for 1996-2025.

Inputs (data/raw/fire/):
    master files/1977_2005/APAFR_Burn_History_1977_2005.shp   (1996-2005 used)
    master files/2006_2026/APAFR_Burn_Histroy_2006_2026.shp   (2006-2025 used)
    yearly files/2006-2009/*.shp   (old-style schema, used to recover lost
        ignition-source detail for 2006-2009 Unknowns in the master file)
data/raw/boundary/boundary.shp  (range boundary + named target/impact areas,
    used to compute distance-to-target for the Unknown-resolution model)

Ignition source is classified into four categories: Prescribed, Military,
Lightning, Unknown.
    - 1996-2005: from Type/Ignition fields.
    - 2006-2025: from wildland_1/wildfireCause fields, with narrative text
      used to catch unknown-cause fires tied to a live-fire mission.
    - "Jump" fires (escaped from another fire, source not recorded) are
      resolved by matching to an adjacent fire burning the same day or the
      day before and inheriting its source; left Unknown if no match. One
      exception: one jumped fire on 1999-06-01 (aka the "fenceline fire")
      is manually set to Prescribed per work done by Slocum (2005, Task 6),
      which documents it as an escaped prescribed burn; day-matching alone
      can't find this because the burn it escaped from wasn't recorded as
      a separate same-day event.

For wildfires, raw burn-unit polygons are grouped into fire events using
spatial adjacency (10 m gap tolerance) and a date window (+/-2 days). For
prescribed fires, one polygon = one event. This is a known limitation for
Prescribed fires: a real prescribed burn is a planned operation that can
span non-adjacent units (confirmed from 2006+ fireId groupings), which
can't be reconstructed from the 1996-2005 data -- it has no Fire Zone/
operation field. One-polygon-per-event is used as the best available
stand-in, not a claim that it matches true burn operations.

2006-2009 Unknowns in the master file are reclassified where possible
using the old-style yearly shapefiles for those years, matched by
year/month and spatial overlap. These files have information on ignition
source. No yearly files were found for 2010-2011.

Remaining Unknown wildfires (anything not resolved above) are assigned a
probabilistic source using a logistic regression fit on known
Lightning/Military fires (features: day-of-year, distance to nearest
target area, polygon compactness, log-area). The model reliably separates
out Military fires -- precision there is high, consistent with Military
dominating this class split. It is weaker for Lightning: cross-validated
precision of a "Lightning" call holds in a ~0.75-0.85 band across
P(Military) thresholds from 0.05 to 0.50, so a stricter cutoff doesn't buy
more confidence, just fewer Lightning calls at the same error rate. In
practice this affects a very small amount of area burned.

Diagnostic maps (by month, known fires by source / Unknown fires by
predicted source with P(Military) labeled) and check graphs (ignition
source by month and by year, area and count) are written to
data/processed/fire/maps/ and data/processed/fire/data_checks/
respectively, for visual QA.

Output: data/processed/fire/fire_data.gpkg
    One row per fire event. Columns: event_id, date, source, ha, notes,
    era, geometry.
"""

# Packages
from pathlib import Path
import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from matplotlib import pyplot as plt

# Options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

# Folders
data_folder = Path.cwd() / "data"
raw_folder = "raw"
processed_folder = "processed"
fire_folder = "fire"

raw_data_folder = data_folder / raw_folder / fire_folder
processed_data_folder = data_folder / processed_folder / fire_folder

processed_data_folder.mkdir(parents=True, exist_ok=True)
print(f"Processed output will be written to: {processed_data_folder}")

# GLOBALS
start_year = 1996  # Note: before 1996 wildfire ignition source is not provided
end_year = 2025

# Read in master files: 1977-2005
_path = raw_data_folder / "master files" / "1977_2005"
_file = "APAFR_Burn_History_1977_2005.shp"
old_gdf = gpd.read_file(_path / _file)
print(f"Read {old_gdf.shape[0]:,} rows x {old_gdf.shape[1]} columns from {_path / _file}")
print(f"CRS: {old_gdf.crs}")

# Read in master files: 2006-2026
_path = raw_data_folder / "master files" / "2006_2026"
_file = "APAFR_Burn_Histroy_2006_2026.shp"
new_gdf = gpd.read_file(_path / _file)
print(f"Read {new_gdf.shape[0]:,} rows x {new_gdf.shape[1]} columns from {_path / _file}")
print(f"CRS: {new_gdf.crs}")

# Yearly folder
yearly_folder = raw_data_folder / "yearly files"

# --- Reproject to match old_gdf's CRS (meters) before any distance/adjacency work ---
# old_gdf's CRS also matches the APAFR boundary/target shapefile we'll use later.
if new_gdf.crs != old_gdf.crs:
    print(f"Reprojecting new_gdf from {new_gdf.crs} to {old_gdf.crs}")
    new_gdf = new_gdf.to_crs(old_gdf.crs)
else:
    print(f"new_gdf already in {new_gdf.crs}, no reprojection needed")

# Cut old_gdf to start_year and beyond
old_gdf = old_gdf[old_gdf["Year_"] >= start_year].reset_index(drop=True)
print(f"old_gdf filtered to {start_year}+: {old_gdf.shape[0]:,} rows x {old_gdf.shape[1]} columns")

# Cut new_gdf to end_year and earlier
new_gdf = new_gdf[new_gdf["fireYear"] <= end_year].reset_index(drop=True)
print(f"new_gdf filtered to {end_year}-: {new_gdf.shape[0]:,} rows x {new_gdf.shape[1]} columns")

# ==============================
# Prcessing 1995-2005 (old_gdf)
# ==============================

# Classify ignition source
def classify_old(row):
    if row["Type"] == "Prescribed":
        return "Prescribed"
    ig = str(row["Ignition"]).strip()
    if ig == "Jump":
        return "Jump"
    if ig == "Lightning":
        return "Lightning"
    if ig in ["Ordnance", "Ordinance", "Flare", "Smokey Sam", "Smoke Grenade",
              "Simulator", "Pyrotechs", "Surfacing EOD"]:
        return "Military"
    if ig in ["Catalytic Conv", "Heavy Equip"]:
        return "Exclude"
    return "Unknown"

old_gdf["source"] = old_gdf.apply(classify_old, axis=1)
print("Ignition source classification for old_gdf (number of rows):")
print(old_gdf["source"].value_counts())

# Build a date column from Year_/Month_/day_
old_gdf["date"] = pd.to_datetime(
    dict(year=old_gdf["Year_"], month=old_gdf["Month_"], day=old_gdf["day_"]),
    errors="coerce",
)
n_bad = old_gdf["date"].isna().sum()
print(f"{n_bad} rows have an unparseable date (e.g. day_ == 99)")
print(old_gdf.loc[old_gdf["date"].isna(), ["Year_", "Month_", "day_", "source", "Acres"]])

# Where day is invalid (99, 45, or any value outside 1-31), use day 15 as a mid-month placeholder
old_gdf.loc[~old_gdf["day_"].between(1, 31), "day_"] = 15

old_gdf["date"] = pd.to_datetime(
    dict(year=old_gdf["Year_"], month=old_gdf["Month_"], day=old_gdf["day_"]),
    errors="coerce",
)
n_bad = old_gdf["date"].isna().sum()
print(f"{n_bad} rows still have an unparseable date")
print(old_gdf.loc[old_gdf["date"].isna(), ["Year_", "Month_", "day_", "source", "Acres"]])

# Where month is 99 (unknown), drop the row entirely
n_month99 = (old_gdf["Month_"] == 99).sum()
print(f"Dropping {n_month99} rows with Month_ == 99")
old_gdf = old_gdf[old_gdf["Month_"] != 99].reset_index(drop=True)

# --- Group fires into events ---
# Wildfires (Military/Lightning/Unknown/Jump): adjacent polygons within +/-2 days
# are treated as one fire, since a wildfire can spread across several burn
# units over more than one day.
# Prescribed: each polygon is kept as its own event. A "real" prescribed fire
# event is better defined as one day's planned burn operation across a Fire
# Zone, which the 2006+ data (fireId) shows can span units kilometers apart --
# not something adjacency can detect. We don't have the field (Fire Zone/
# operation ID) needed to reconstruct that for this period, so one polygon =
# one event is used here as a known-limited stand-in: it's the best we can do
# with the data provided.
# Jump is kept as its own category for now; it gets resolved to Prescribed,
# Military, Lightning, or Unknown in a later step.

GAP_TOLERANCE = 10  # meters
DAY_WINDOW = 2       # days, wildfires only

old_gdf["event_id"] = -1
next_id = 0

for source, group in old_gdf.groupby("source"):
    day_window = 0 if source == "Prescribed" else DAY_WINDOW
    idx = group.index.tolist()
    if len(idx) == 1:
        old_gdf.loc[idx, "event_id"] = next_id
        next_id += 1
        continue
    buffered = group.geometry.buffer(GAP_TOLERANCE / 2)
    dates = group["date"]
    G = nx.Graph()
    G.add_nodes_from(idx)
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if not buffered.iloc[i].intersects(buffered.iloc[j]):
                continue
            di, dj = dates.iloc[i], dates.iloc[j]
            if pd.notna(di) and pd.notna(dj) and abs((di - dj).days) <= day_window:
                G.add_edge(idx[i], idx[j])
    for component in nx.connected_components(G):
        old_gdf.loc[list(component), "event_id"] = next_id
        next_id += 1

print(f"{old_gdf.shape[0]:,} polygons grouped into {old_gdf['event_id'].nunique():,} fire events")
print(old_gdf.groupby("source")["event_id"].nunique())

# --- Dissolve polygons into one row per fire event ---
# Note: "geometry" is the column holding each row's shape (the fire's footprint
# on the map); dissolve() merges the shapes of every polygon sharing an
# event_id into one combined shape for that fire.
# Wildfire events with multiple polygons are merged into one row with a single
# merged geometry and summed acreage. Prescribed events pass through unchanged,
# since each Prescribed polygon was already its own event going in.
old_fires = old_gdf.sort_values("date").dissolve(by="event_id", aggfunc={
    "source": "first", "Year_": "first", "Month_": "first", "day_": "first",
    "Acres": "sum", "date": "first",
}).reset_index()

n_before_drop = old_fires.shape[0]
old_fires = old_fires[old_fires["source"] != "Exclude"]
n_excluded = n_before_drop - old_fires.shape[0]

print(f"old_gdf had {old_gdf.shape[0]:,} rows (one per polygon)")
print(f"old_fires now has {old_fires.shape[0]:,} rows (one per fire event, after dropping {n_excluded} Exclude)")

# --- Resolve Jump events: inherit source from nearest same-day or day-before neighbor ---
# A "Jump" fire is a wildfire that escaped from another fire; the record never
# says which one. We infer it by finding whatever fire was burning adjacent to
# it on the same day or the day before, and taking that fire's source. If no
# such neighbor exists, the Jump event falls back to Unknown -- except for one
# documented case below.
jump_mask = old_fires["source"] == "Jump"
other_mask = old_fires["source"].isin(["Prescribed", "Military", "Lightning", "Unknown"])

jump_events = old_fires[jump_mask]
other_events = old_fires[other_mask]

resolved_source = {}
match_notes = {}

for idx, jrow in jump_events.iterrows():
    buffered = jrow.geometry.buffer(GAP_TOLERANCE / 2)
    candidates = other_events[
        (other_events["date"] == jrow["date"]) | (other_events["date"] == jrow["date"] - pd.Timedelta(days=1))
    ]
    match = candidates[candidates.geometry.buffer(GAP_TOLERANCE / 2).intersects(buffered)]

    if len(match) == 0:
        resolved_source[idx] = "Unknown"
        match_notes[idx] = "no same-day/prior-day neighbor found"
    else:
        sources_found = match["source"].unique()
        resolved_source[idx] = sources_found[0]
        match_notes[idx] = f"matched {len(match)} neighbor(s), source(s): {list(sources_found)}"

old_fires["jump_note"] = old_fires.index.map(match_notes).astype("object")
old_fires.loc[jump_mask, "source"] = old_fires.loc[jump_mask].index.map(resolved_source)

# --- Documented exception: the 1999-06-01 fenceline fire ---
# Slocum (2005 report, Task 6; SAS "Fire data construction") identifies this
# specific event as an escaped prescribed fire from a fenceline burn. Day-
# matching above finds no neighbor for it, because the prescribed fire it
# escaped from wasn't recorded as a separate same-day event in this data.
_mask_1999 = (
    (old_fires["Year_"] == 1999) & (old_fires["Month_"] == 6) & (old_fires["day_"] == 1)
    & (old_fires["source"] == "Unknown")
)
old_fires.loc[_mask_1999, "source"] = "Prescribed"
old_fires.loc[_mask_1999, "jump_note"] = "manual override: Slocum (2005) documents as escaped prescribed fire (fenceline burn)"

# --- Summary: how Jump fires were resolved, by count and by acreage ---
resolved_df = old_fires.loc[jump_mask, ["source", "Acres"]].copy()
summary = resolved_df.groupby("source").agg(n_events=("source", "size"), acres=("Acres", "sum"))
print(f"{jump_mask.sum()} Jump events resolved ({resolved_df['Acres'].sum():,.1f} acres total):")
print(summary)
print(f"\n{(resolved_df['source'] == 'Unknown').sum()} event(s) / "
      f"{resolved_df.loc[resolved_df['source']=='Unknown', 'Acres'].sum():,.1f} acres remain Unknown. "
      f"Separately, 1 event (the 1999-06-01 fenceline fire) was overridden to Prescribed rather than "
      f"left Unknown, per Slocum (2005).")

# --- Finalize old_fires: friendlier column names, area in hectares ---
# Note: "notes" is a column available for additional information, but for this analysis it only has a note about jumping.
old_fires["ha"] = old_fires["Acres"] * 0.404686
old_fires = old_fires.rename(columns={"jump_note": "notes"})
old_fires = old_fires[["event_id", "date", "source", "ha", "notes", "geometry"]]

print(f"old_fires: {old_fires.shape[0]:,} fire events, columns: {list(old_fires.columns)}")
print(old_fires.head())

# ============================================================
# 2006-2026 (new_gdf)
# ============================================================

# Classify ignition source from the master file's schema.
# wildland_1 == "prescribed" identifies planned burns. For everything else,
# wildfireCa gives a specific cause where recorded; an "unknown" cause whose
# narrative mentions a live fire mission is still Military, just without a
# coded cause. Arson/offPostFire/previousFire are neither Lightning nor
# Military, so they're excluded rather than forced into either category.
def classify_new(row):
    if row["wildland_1"] == "prescribed":
        return "Prescribed"
    cause = str(row["wildfireCa"]).strip()
    if cause == "lightningStrike":
        return "Lightning"
    if cause in ["milLiveMissionMunition", "milLiveMissionIncendiary"]:
        return "Military"
    if cause == "unknown":
        narrative = str(row["narrative"]).lower()
        return "Military" if "live fire" in narrative else "Unknown"
    return "Exclude"  # arson, offPostFire, previousFire

new_gdf["source"] = new_gdf.apply(classify_new, axis=1)
print("Ignition source classification for new_gdf (number of rows), before yearly-file augmentation:")
print(new_gdf["source"].value_counts())

# --- Where are the Unknowns concentrated? ---
new_gdf["fireYear_check"] = new_gdf["fireYear"]
print("Unknown polygons by year:")
print(new_gdf.loc[new_gdf["source"] == "Unknown", "fireYear_check"].value_counts().sort_index())

# --- Unknown wildfires in new_gdf: the 2006-2011 gap ---
# The master file (new_gdf) is missing ignition-source detail for wildfires
# from 2006 to 2011: these fires are recorded as "wild"/"unknown" rather than
# "lightning" or "military" or the like.
#
# For 2006-2009, we have the original per-year shapefiles (old-style schema,
# Type/Ignition fields, same as the 1995-2005 data) with the ignition source
# data. Here we read these in and match them against new_gdf's Unknown
# polygons by date and location (there's no shared ID between the two
# schemas), and overwrite source wherever a match resolves one.
#
# For 2010-2011, no yearly files have been found. Below we will
# attempt to identify these unknowns (as well as other unknowns) using a
# statistical model.

# --- Read and normalize the 2006-2009 yearly shapefiles ---
# Column names shift year to year (e.g. 2007 uses Type07/Ignition07/...), so
# we find each by keyword rather than assuming a fixed name, same approach
# used for the 1996-2005 yearly-file comparison.

yearly_2006_09 = []

for year in range(2006, 2010):
    _path = yearly_folder / str(year)
    _candidates = list(_path.glob("*.shp")) if _path.exists() else []
    if not _candidates:
        print(f"{year}: no shapefile found in {_path}, skipping")
        continue

    _file = _candidates[0]
    yr_gdf = gpd.read_file(_file)

    type_col = [c for c in yr_gdf.columns if "TYPE" in c.upper()][0]
    ig_col = [c for c in yr_gdf.columns if "IGNIT" in c.upper()][0]
    month_col = [c for c in yr_gdf.columns if "MONTH" in c.upper()][0]
    day_col = [c for c in yr_gdf.columns if c.upper().startswith("DAY")][0]
    year_col = [c for c in yr_gdf.columns if "YEAR" in c.upper()][0]
    acres_col = [c for c in yr_gdf.columns if "ACRE" in c.upper()][0]

    df = yr_gdf[[year_col, month_col, day_col, type_col, ig_col, acres_col, "geometry"]].copy()
    df.columns = ["year", "month", "day", "type", "ignition", "acres", "geometry"]
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=yr_gdf.crs)
    df["source_file"] = _file.name

    yearly_2006_09.append(df)
    print(f"{year}: read {df.shape[0]} rows from {_file.name}")

yearly_2006_09 = pd.concat(yearly_2006_09, ignore_index=True)
yearly_2006_09 = gpd.GeoDataFrame(yearly_2006_09, geometry="geometry", crs=yearly_2006_09.crs)
print(f"\nCombined 2006-2009 yearly data: {yearly_2006_09.shape[0]:,} rows")

# Note: yearly_2006_09 is already in EPSG:26917 (same as old_gdf/new_gdf), consistent
# with every yearly file checked so far -- no reprojection needed.

# --- Classify the 2006-2009 yearly data using the same logic as classify_old ---
def classify_yearly_old_style(row):
    if row["type"] == "Prescribed":
        return "Prescribed"
    ig = str(row["ignition"]).strip()
    if ig == "Jump":
        return "Jump"
    if ig.lower() == "lightning":
        return "Lightning"
    if ig in ["Ordnance", "Ordinance", "Flare", "Smokey Sam", "Smoke Grenade",
              "Simulator", "Pyrotechs", "Surfacing EOD", "Mission"]:
        return "Military"
    if ig in ["Catalytic Conv", "Heavy Equip"]:
        return "Exclude"
    return "Unknown"

yearly_2006_09["source"] = yearly_2006_09.apply(classify_yearly_old_style, axis=1)
print("2006-2009 yearly data, source classification:")
print(yearly_2006_09["source"].value_counts())
# Result: we have only 11 unknowns in the yearly files from 2006-2009 compared to 68
# unknowns in the master file for the same period.

# Where day is invalid (0, 99, or any value outside 1-31), use day 15 as a mid-month placeholder
print(yearly_2006_09.loc[~yearly_2006_09["day"].between(1, 31), ["year", "month", "day", "source", "acres"]])

yearly_2006_09.loc[~yearly_2006_09["day"].between(1, 31), "day"] = 15

yearly_2006_09["date"] = pd.to_datetime(
    dict(year=yearly_2006_09["year"], month=yearly_2006_09["month"], day=yearly_2006_09["day"]),
    errors="coerce",
)
n_bad = yearly_2006_09["date"].isna().sum()
print(f"{n_bad} rows still have an unparseable date")

# --- Match new_gdf's 2006-2009 Unknown polygons against yearly_2006_09 by year/month + overlap ---
# Exact date won't work: yearly_2006_09 had day=0/99 placeholders set to day=15,
# so its date no longer reflects the true day for those rows. Matching on
# year/month plus spatial overlap avoids relying on day where it may be wrong.
unknown_mask = (new_gdf["source"] == "Unknown") & new_gdf["fireYear"].between(2006, 2009)
unknown_rows = new_gdf[unknown_mask]
print(f"Attempting to match {unknown_rows.shape[0]} Unknown polygons from new_gdf (2006-2009)")

match_results = {}

for idx, row in unknown_rows.iterrows():
    same_month = yearly_2006_09[
        (yearly_2006_09["year"] == row["fireStartD"].year)
        & (yearly_2006_09["month"] == row["fireStartD"].month)
    ]
    overlap = same_month[same_month.geometry.intersects(row.geometry)]

    if len(overlap) == 0:
        match_results[idx] = ("no match", None)
    else:
        sources_found = overlap["source"].unique()
        if len(sources_found) == 1:
            match_results[idx] = ("matched", sources_found[0])
        else:
            match_results[idx] = ("ambiguous", list(sources_found))

outcomes = pd.Series([v[0] for v in match_results.values()]).value_counts()
print("\nMatch outcomes:")
print(outcomes)

# --- Apply the clean matches; surface the rest for review ---
n_applied = 0
for idx, (outcome, value) in match_results.items():
    if outcome == "matched":
        new_gdf.loc[idx, "source"] = value
        n_applied += 1

print(f"Applied {n_applied} resolved sources to new_gdf")
print(f"\nNew source counts for new_gdf (2006-2009 Unknowns augmented):")
print(new_gdf["source"].value_counts())

# --- Ambiguous and no-match cases, for review ---
review_idx = [idx for idx, (outcome, _) in match_results.items() if outcome in ("ambiguous", "no match")]
review = new_gdf.loc[review_idx, ["fireStartD", "fireYear", "Acres" if "Acres" in new_gdf.columns else "areaSize"]].copy()
review["outcome"] = [match_results[idx][0] for idx in review_idx]
review["sources_found"] = [match_results[idx][1] for idx in review_idx]
print("\nAmbiguous / no-match cases:")
print(review.to_string())

# At this point we are done patching the 2006-2009 Unknowns. The remaining Unknowns (2010-2011 and any others) will be
# addressed with a statistical model later in the script.

# --- Build date column ---
new_gdf["date"] = pd.to_datetime(new_gdf["fireStartD"], errors="coerce")
n_bad = new_gdf["date"].isna().sum()
print(f"{n_bad} rows have an unparseable date")

# --- Group fires into events (same rule as old_gdf) ---
# Wildfires: adjacent polygons within +/-2 days are one fire.
# Prescribed: each polygon is kept as its own event, for the same reason
# documented in the old_gdf section (no Fire Zone/operation field available
# to reconstruct APAFR's actual operation-level grouping for this data).
# Note: this is the same logic as used for old_gdf, so the two datasets will
# be consistent in how they define fire events.
new_gdf["event_id"] = -1
next_id = 0

for source, group in new_gdf.groupby("source"):
    day_window = 0 if source == "Prescribed" else DAY_WINDOW
    idx = group.index.tolist()
    if len(idx) == 1:
        new_gdf.loc[idx, "event_id"] = next_id
        next_id += 1
        continue
    buffered = group.geometry.buffer(GAP_TOLERANCE / 2)
    dates = group["date"]
    G = nx.Graph()
    G.add_nodes_from(idx)
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if not buffered.iloc[i].intersects(buffered.iloc[j]):
                continue
            di, dj = dates.iloc[i], dates.iloc[j]
            if pd.notna(di) and pd.notna(dj) and abs((di - dj).days) <= day_window:
                G.add_edge(idx[i], idx[j])
    for component in nx.connected_components(G):
        new_gdf.loc[list(component), "event_id"] = next_id
        next_id += 1

print(f"{new_gdf.shape[0]:,} polygons grouped into {new_gdf['event_id'].nunique():,} fire events")
print(new_gdf.groupby("source")["event_id"].nunique())

# --- Dissolve polygons into one row per fire event ---
new_fires = new_gdf.sort_values("date").dissolve(by="event_id", aggfunc={
    "source": "first", "date": "first", "areaSize": "sum",
}).reset_index()

new_fires = new_fires.rename(columns={"areaSize": "Acres"})

n_before_drop = new_fires.shape[0]
new_fires = new_fires[new_fires["source"] != "Exclude"]
n_excluded = n_before_drop - new_fires.shape[0]

print(f"new_gdf had {new_gdf.shape[0]:,} rows (one per polygon)")
print(f"new_fires now has {new_fires.shape[0]:,} rows (one per fire event, after dropping {n_excluded} Exclude)")

# --- Finalize new_fires: match old_fires's column shape, area in hectares ---
# No "notes" content exists for this period (no Jump-style resolution needed),
# but the column is kept, empty, so the two periods can be combined cleanly.
new_fires["ha"] = new_fires["Acres"] * 0.404686
new_fires["notes"] = pd.NA
new_fires = new_fires[["event_id", "date", "source", "ha", "notes", "geometry"]]

print(f"new_fires: {new_fires.shape[0]:,} fire events, columns: {list(new_fires.columns)}")
print(new_fires.head())

# Flatten to 2D -- old_fires has no Z coordinate, so this keeps geometry
# consistent across both periods for anything downstream that compares or
# combines them (distance, area, the eventual concat).
new_fires["geometry"] = gpd.GeoSeries(new_fires.geometry).force_2d()
print(f"Geometry types after flattening: {new_fires.geom_type.unique()}")

# ============================================================
# Combine 1996-2005 and 2006-2025 into one fire history
# ============================================================
old_fires["era"] = "1996-2005"
new_fires["era"] = "2006-2025"

fires = pd.concat([old_fires, new_fires], ignore_index=True)
fires = gpd.GeoDataFrame(fires, geometry="geometry", crs=old_fires.crs)

print(f"Combined: {fires.shape[0]:,} fire events "
      f"({old_fires.shape[0]:,} from {old_fires['era'].iloc[0]}, {new_fires.shape[0]:,} from {new_fires['era'].iloc[0]})")
print(fires.groupby(["era", "source"]).size())

# ============================================================
# Resolve Unknown fires: Lightning vs Military probability model
# ============================================================

# --- Boundary and target areas (needed for distance-to-target feature) ---
_boundary_path = raw_data_folder / "boundary" / "boundary.shp"
boundary = gpd.read_file(_boundary_path)
if boundary.crs != fires.crs:
    boundary = boundary.to_crs(fires.crs)

targets = boundary[boundary["Name"].notna()].copy()
target_union = targets.geometry.union_all()

print(f"Target areas: {targets.shape[0]}")
print(targets[["Name", "Range", "Acres"]])

# --- Build features: day-of-year, distance to target, compactness, size ---
fires["dist_to_target"] = fires.geometry.distance(target_union)
fires["doy"] = fires["date"].dt.dayofyear
fires["doy_sin"] = np.sin(2 * np.pi * fires["doy"] / 365)
fires["doy_cos"] = np.cos(2 * np.pi * fires["doy"] / 365)
fires["compactness"] = (4 * np.pi * fires.geometry.area) / (fires.geometry.length ** 2)
fires["log_ha"] = np.log1p(fires["ha"])

print(fires[["dist_to_target", "compactness", "log_ha"]].describe())

# --- Split known (Lightning/Military) vs Unknown ---
features = ["doy_sin", "doy_cos", "dist_to_target", "compactness", "log_ha"]

known = fires[fires["source"].isin(["Lightning", "Military"])].copy()
known["y"] = (known["source"] == "Military").astype(int)

unknown = fires[fires["source"] == "Unknown"].copy()

print(f"Known: n={len(known)} ({known['y'].sum()} Military, {(known['y']==0).sum()} Lightning)")
print(f"Unknown: n={len(unknown)}")
# Note: this is fairly unbalanced

# --- Cross-validated logistic regression on known fires ---
X = known[features].values
y = known["y"].values

clf = make_pipeline(StandardScaler(), LogisticRegression())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
preds = (probs >= 0.5).astype(int)

print("Confusion matrix (rows=true, cols=pred; 0=Lightning, 1=Military):")
print(confusion_matrix(y, preds))
print(classification_report(y, preds, target_names=["Lightning", "Military"]))
print(f"AUC: {roc_auc_score(y, probs):.3f}")

# --- Precision of a "Lightning" call at different thresholds on P(Military) ---
for threshold in [0.50, 0.40, 0.35, 0.25, 0.15, 0.10, 0.05]:
    pred_lightning = probs < threshold
    actual_lightning = (y == 0)

    n_called = pred_lightning.sum()
    n_correct = (pred_lightning & actual_lightning).sum()
    precision = n_correct / n_called if n_called else float("nan")

    print(f"p < {threshold:.2f}: {n_called:3d} fires called Lightning, precision={precision:.2f}")
# Conclusion: if we use stricter cutoffs, it does not help us much.

# --- Fit on full known data, score the Unknowns, sum area by predicted label ---
clf.fit(X, y)
unknown["p_military"] = clf.predict_proba(unknown[features].values)[:, 1]
unknown["pred_label"] = np.where(unknown["p_military"] < 0.5, "Lightning", "Military")

print(unknown.groupby("pred_label")["ha"].agg(["count", "sum"]))
print(f"\nTotal Unknown area: {unknown['ha'].sum():,.1f} ha")
print(f"Total fire area, all sources, 1996-2025: {fires['ha'].sum():,.1f} ha")
print(f"Potential misclassification error (~20% of 794 ha): {0.2 * 794:,.1f} ha")
print(f"That error as a share of total: {0.2 * 794 / fires['ha'].sum() * 100:.2f}%")
# Conclusion: the model is not a great fit, but it does provide a reasonable prediction as to
# the ignition source of the unknowns, and any misclassification is a very small share of the total
# area burned.

# --- Output folders for known/unknown fire maps, one map per month ---
# Note: this is for visual verification
maps_folder = processed_data_folder / "maps"
known_maps_folder = maps_folder / "known"
unknown_maps_folder = maps_folder / "unknown"
known_maps_folder.mkdir(parents=True, exist_ok=True)
unknown_maps_folder.mkdir(parents=True, exist_ok=True)

print(f"Known fire maps will be written to: {known_maps_folder}")
print(f"Unknown fire maps will be written to: {unknown_maps_folder}")

# --- Known fires: one map per month, colored by source ---
colors = {"Prescribed": "tab:green", "Military": "tab:orange", "Lightning": "yellow", "Unknown": "tab:gray"}

fires["year"] = fires["date"].dt.year
fires["month"] = fires["date"].dt.month

months_with_fires = fires[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
print(f"{len(months_with_fires)} year-month combinations have at least one fire")

n_written = 0
for _, ym in months_with_fires.iterrows():
    yr, mo = ym["year"], ym["month"]
    sub = fires[(fires["year"] == yr) & (fires["month"] == mo) & (fires["source"] != "Unknown")]
    if sub.empty:
        continue

    fig, ax = plt.subplots(figsize=(10, 10))
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.5)
    sub.plot(ax=ax, color=[colors[s] for s in sub["source"]], edgecolor="black", linewidth=0.3)
    ax.set_title(f"{yr}-{mo:02d}: known fires by source (n={len(sub)})")
    plt.tight_layout()

    _out = known_maps_folder / f"{yr}-{mo:02d}.png"
    fig.savefig(_out, dpi=150)
    plt.close(fig)
    n_written += 1

print(f"Wrote {n_written} known-fire maps to {known_maps_folder}")

# --- Unknown fires: one map per month, colored by predicted label, P(Military) labeled ---
unknown["year"] = unknown["date"].dt.year
unknown["month"] = unknown["date"].dt.month

months_with_unknown = unknown[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
print(f"{len(months_with_unknown)} year-month combinations have at least one Unknown fire")

n_written = 0
for _, ym in months_with_unknown.iterrows():
    yr, mo = ym["year"], ym["month"]
    sub = unknown[(unknown["year"] == yr) & (unknown["month"] == mo)]
    if sub.empty:
        continue

    fig, ax = plt.subplots(figsize=(10, 10))
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.5)
    targets.plot(ax=ax, color="lightgray", edgecolor="black", alpha=0.6)
    sub.plot(ax=ax, color=[colors[c] for c in sub["pred_label"]], edgecolor="black", linewidth=0.4)

    for _, row in sub.iterrows():
        c = row.geometry.centroid
        ax.annotate(f"{row.p_military:.2f}", (c.x, c.y), fontsize=10, ha="center", fontweight="bold")

    ax.set_title(f"{yr}-{mo:02d}: Unknown fires, predicted source (n={len(sub)})")
    plt.tight_layout()

    _out = unknown_maps_folder / f"{yr}-{mo:02d}.png"
    fig.savefig(_out, dpi=150)
    plt.close(fig)
    n_written += 1

print(f"Wrote {n_written} Unknown-fire maps to {unknown_maps_folder}")

# Conclusion: looking through the maps, if there is a fire that is questionable, it is small, so the overall error is
# likely small.

# --- Merge resolved Unknowns back into fires ---
# Overwrite source with the model's call, and leave a note recording that
# this was a model prediction (not a directly recorded cause) along with the
# probability it was based on.
for idx, row in unknown.iterrows():
    fires.loc[idx, "source"] = row["pred_label"]
    fires.loc[idx, "notes"] = f"previous unknown; model prediction (military) = {row['p_military']:.2f}"

print("fires source counts after merging model predictions:")
print(fires["source"].value_counts())
print(f"\n{len(unknown)} rows updated with a model-based source and note")
print(fires.loc[unknown.index, ["date", "source", "notes"]].head())


# --- Check output --- #

# Graph #1: fire activity per ignition source per month
fires["month"] = fires["date"].dt.month

area_by_month = fires.groupby(["month", "source"])["ha"].sum().unstack(fill_value=0)
area_by_month = area_by_month.reindex(columns=["Prescribed", "Military", "Lightning"], fill_value=0)

count_by_month = fires.groupby(["month", "source"]).size().unstack(fill_value=0)
count_by_month = count_by_month.reindex(columns=["Prescribed", "Military", "Lightning"], fill_value=0)

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
area_by_month.plot(kind="bar", stacked=True, ax=axes[0], color=[colors[c] for c in area_by_month.columns])
axes[0].set_ylabel("Area burned (ha)")
axes[0].set_title("Fires by ignition source, 1996-2025 combined, by month")

count_by_month.plot(kind="bar", stacked=True, ax=axes[1], color=[colors[c] for c in count_by_month.columns])
axes[1].set_ylabel("Number of fires")
axes[1].set_xlabel("Month")

plt.tight_layout()
plt.show()

checks_folder = processed_data_folder / "data_checks"
checks_folder.mkdir(parents=True, exist_ok=True)

_out = checks_folder / "fire_type_by_month.png"
fig.savefig(_out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Wrote check figure to {_out}")

# graph #2: fire activity per ignition source per year
fires["year"] = fires["date"].dt.year

area_by_year = fires.groupby(["year", "source"])["ha"].sum().unstack(fill_value=0)
area_by_year = area_by_year.reindex(columns=["Prescribed", "Military", "Lightning"], fill_value=0)

count_by_year = fires.groupby(["year", "source"]).size().unstack(fill_value=0)
count_by_year = count_by_year.reindex(columns=["Prescribed", "Military", "Lightning"], fill_value=0)

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
area_by_year.plot(kind="bar", stacked=True, ax=axes[0], color=[colors[c] for c in area_by_year.columns])
axes[0].set_ylabel("Area burned (ha)")
axes[0].set_title("Fires by ignition source, 1996-2025 combined, by year")

count_by_year.plot(kind="bar", stacked=True, ax=axes[1], color=[colors[c] for c in count_by_year.columns])
axes[1].set_ylabel("Number of fires")
axes[1].set_xlabel("Year")

plt.tight_layout()

_out = checks_folder / "fire_type_by_year.png"
fig.savefig(_out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Wrote check figure to {_out}")

# --- Finalize fires: clean output ---
final_cols = ["event_id", "date", "source", "ha", "notes", "era", "geometry"]
fires = fires[final_cols]
fires = fires.sort_values("date").reset_index(drop=True)

print(f"fires: {fires.shape[0]:,} rows, columns: {list(fires.columns)}")
print(fires.head())

# --- Write processed output ---
_out = processed_data_folder / "fire_data.gpkg"
fires.to_file(_out, driver="GPKG")
print(f"Wrote {fires.shape[0]:,} rows to {_out}")