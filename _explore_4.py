# _explore_4.py
# Combine known wildfires from both periods to train one model, then score
# Unknown wildfires from both periods.

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

GAP_TOLERANCE = 10  # meters
DAY_WINDOW = 2       # days
colors = {"Prescribed": "tab:green", "Military": "tab:orange", "Lightning": "yellow", "Unknown": "tab:gray"}
features = ["doy_sin", "doy_cos", "dist_to_target", "compactness", "log_ha"]

# --- Boundary and target areas ---
_boundary_path = Path(r"C:\Users\mateo\projects\piFire\FIRE\data\APAFR boundary\boundary.shp")
boundary = gpd.read_file(_boundary_path)
targets = boundary[boundary["Name"].notna()].copy()
target_union = targets.geometry.union_all()


def add_features(gdf, date_col):
    """Add doy_sin/cos, dist_to_target, compactness, log_ha. Drops rows with unparseable date."""
    gdf = gdf.copy()
    gdf["ha"] = gdf["Acres"] * 0.404686
    gdf["dist_to_target"] = gdf.geometry.distance(target_union)
    gdf["doy"] = gdf[date_col].dt.dayofyear
    n_bad = gdf["doy"].isna().sum()
    print(f"  dropping {n_bad} row(s) with unparseable date")
    gdf = gdf.dropna(subset=["doy"])
    gdf["doy_sin"] = np.sin(2 * np.pi * gdf["doy"] / 365)
    gdf["doy_cos"] = np.cos(2 * np.pi * gdf["doy"] / 365)
    gdf["compactness"] = (4 * np.pi * gdf.geometry.area) / (gdf.geometry.length ** 2)
    gdf["log_ha"] = np.log1p(gdf["ha"])
    return gdf


# ============================================================
# 1996–2005
# ============================================================
_path = Path(r"C:\Users\mateo\projects\piFire\FIRE\data\Burn History 1977_2026\1977_2005")
old_gdf = gpd.read_file(_path / "APAFR_Burn_History_1977_2005.shp")
old_gdf = old_gdf[old_gdf["Year_"] >= 1996].reset_index(drop=True)
print(f"1996-2005: {old_gdf.shape[0]:,} polygons")


def classify_old(row):
    if row["Type"] == "Prescribed":
        return "Prescribed"
    ig = str(row["Ignition"]).strip()
    if ig == "Jump":
        return "Prescribed"
    if ig == "Lightning":
        return "Lightning"
    if ig in ["Ordnance", "Ordinance", "Flare", "Smokey Sam", "Smoke Grenade",
              "Simulator", "Pyrotechs", "Surfacing EOD"]:
        return "Military"
    if ig in ["Catalytic Conv", "Heavy Equip"]:
        return "Exclude"
    return "Unknown"


old_gdf["source"] = old_gdf.apply(classify_old, axis=1)
old_gdf["date"] = pd.to_datetime(
    dict(year=old_gdf["Year_"], month=old_gdf["Month_"], day=old_gdf["day_"]), errors="coerce"
)

# Event grouping: Prescribed = 1 polygon = 1 event; wildfires = adjacency within DAY_WINDOW
old_gdf["event_id"] = -1
next_id = 0
presc_idx = old_gdf[old_gdf["source"] == "Prescribed"].index
old_gdf.loc[presc_idx, "event_id"] = range(next_id, next_id + len(presc_idx))
next_id += len(presc_idx)

wildfire_mask = ~old_gdf["source"].isin(["Prescribed"])
for source, group in old_gdf[wildfire_mask].groupby("source"):
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
            if pd.notna(di) and pd.notna(dj) and abs((di - dj).days) <= DAY_WINDOW:
                G.add_edge(idx[i], idx[j])
    for component in nx.connected_components(G):
        old_gdf.loc[list(component), "event_id"] = next_id
        next_id += 1

old_fires = old_gdf.sort_values("date").dissolve(by="event_id", aggfunc={
    "source": "first", "Year_": "first", "Month_": "first", "day_": "first",
    "Acres": "sum", "date": "first",
}).reset_index()
old_fires = old_fires[old_fires["source"] != "Exclude"]
print(old_fires["source"].value_counts())

print("\n1996-2005 known wildfires:")
old_known = add_features(old_fires[old_fires["source"].isin(["Lightning", "Military"])], "date")
old_known["y"] = (old_known["source"] == "Military").astype(int)
old_known["era"] = "1996-2005"

print("1996-2005 Unknown wildfires:")
old_unknown = add_features(old_fires[old_fires["source"] == "Unknown"], "date")
old_unknown["era"] = "1996-2005"
old_unknown["polygon_source_gdf"] = "old_gdf"  # for later mapping

# ============================================================
# 2006–2026
# ============================================================
_path2 = Path(r"C:\Users\mateo\projects\piFire\MWB\Analysis\data\raw\fire\2006 - 2026")
_file2 = "APAFR_Fire_Data_2006_2026.shp"
new_gdf = gpd.read_file(_path2 / _file2)
print(f"\n2006-2026: {new_gdf.shape[0]:,} polygons, CRS: {new_gdf.crs}")
if new_gdf.crs != old_gdf.crs:
    new_gdf = new_gdf.to_crs(old_gdf.crs)

new_gdf["fireStartD"] = pd.to_datetime(new_gdf["fireStartD"], errors="coerce")

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
print(new_gdf["source"].value_counts())

# Official grouping: dissolve by APAFR's own fireId
new_fires = new_gdf.dissolve(by="fireId", aggfunc={
    "source": "first", "fireStartD": "first", "areaSize": "sum",
}).reset_index()
new_fires = new_fires.rename(columns={"areaSize": "Acres", "fireStartD": "date"})
new_fires = new_fires[new_fires["source"] != "Exclude"]

# --- Validation: does our adjacency rule agree with APAFR's fireId grouping? ---
wf_mask = new_gdf["source"].isin(["Lightning", "Military", "Unknown"])
check = new_gdf[wf_mask].copy()
agree = 0
total = 0
for fid, group in check.groupby("fireId"):
    if len(group) < 2:
        continue
    total += 1
    buffered = group.geometry.buffer(GAP_TOLERANCE / 2)
    touches_all = all(
        buffered.iloc[i].intersects(buffered.iloc[j])
        for i in range(len(group)) for j in range(i + 1, len(group))
    )
    if touches_all:
        agree += 1
print(f"\nMulti-polygon fireId events where all polygons are mutually adjacent (<= {GAP_TOLERANCE}m): "
      f"{agree}/{total}")

print("\n2006-2026 known wildfires:")
new_known = add_features(new_fires[new_fires["source"].isin(["Lightning", "Military"])], "date")
new_known["y"] = (new_known["source"] == "Military").astype(int)
new_known["era"] = "2006-2026"

print("2006-2026 Unknown wildfires:")
new_unknown = add_features(new_fires[new_fires["source"] == "Unknown"], "date")
new_unknown["era"] = "2006-2026"
new_unknown["polygon_source_gdf"] = "new_gdf"

# ============================================================
# Combine and model
# ============================================================
known = pd.concat([old_known, new_known], ignore_index=True)
unknown = pd.concat([old_unknown, new_unknown], ignore_index=True)
print(f"\nCombined training set: n={len(known)} ({known['y'].sum()} Military, {(known['y']==0).sum()} Lightning)")
print(known.groupby(["era", "source"]).size())

X = known[features].values
y = known["y"].values

clf = make_pipeline(StandardScaler(), LogisticRegression())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
preds = (probs >= 0.5).astype(int)

print("\nConfusion matrix:")
print(confusion_matrix(y, preds))
print(classification_report(y, preds, target_names=["Lightning", "Military"]))
print(f"AUC: {roc_auc_score(y, probs):.3f}")

clf.fit(X, y)
coefs = pd.Series(clf.named_steps["logisticregression"].coef_[0], index=features)
print("\nStandardized coefficients:")
print(coefs.sort_values())

# --- Predict on combined Unknowns ---
unknown["p_military"] = clf.predict_proba(unknown[features].values)[:, 1]
unknown = unknown.sort_values("ha", ascending=False)
print(f"\n{len(unknown)} Unknown wildfire events scored")
print(unknown[["era", "date", "ha", "p_military"]].to_string())

# --- Map ---
unknown["pred_label"] = np.where(unknown["p_military"] >= 0.5, "Military", "Lightning")

fig, ax = plt.subplots(figsize=(10, 10))
boundary.boundary.plot(ax=ax, color="black", linewidth=0.5)
targets.plot(ax=ax, color="lightgray", edgecolor="black", alpha=0.6)
unknown.plot(ax=ax, color=[colors[c] for c in unknown["pred_label"]], edgecolor="black", linewidth=0.4)
for _, row in unknown.iterrows():
    c = row.geometry.centroid
    ax.annotate(f"{row.p_military:.2f}", (c.x, c.y), fontsize=6, ha="center")
ax.set_title(f"Unknown wildfires 1996-2026 (n={len(unknown)}): predicted source and P(Military)")
plt.tight_layout()
plt.show()