# _explore_3.py

from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


GAP_TOLERANCE = 10  # meters; max gap between polygons to treat as one fire (wildfires only)
DAY_WINDOW = 2       # days; max date gap to treat as one fire (wildfires only)

colors = {"Prescribed": "tab:green", "Military": "tab:orange", "Lightning": "yellow", "Unknown": "tab:gray"}

# Read in 1977–2005 burn history shapefile
_path = Path(r"C:\Users\mateo\projects\piFire\FIRE\data\Burn History 1977_2026\1977_2005")
_file = "APAFR_Burn_History_1977_2005.shp"
old_gdf = gpd.read_file(_path / _file)
print(f"Read {old_gdf.shape[0]:,} rows x {old_gdf.shape[1]} columns from {_path / _file}")
print(f"CRS: {old_gdf.crs}")

# Cut off year to 1996 and beyond
old_gdf = old_gdf[old_gdf["Year_"] >= 1996].reset_index(drop=True)
print(f"Filtered to 1996+: {old_gdf.shape[0]:,} rows x {old_gdf.shape[1]} columns")


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


old_gdf["source"] = old_gdf.apply(classify, axis=1)
print(old_gdf["source"].value_counts())

# --- Polygon-level graph: fires per month by source ---
counts = old_gdf.groupby(["Month_", "source"]).size().unstack(fill_value=0)
counts = counts.reindex(columns=["Prescribed", "Military", "Lightning", "Unknown"], fill_value=0)

area = old_gdf.groupby(["Month_", "source"])["Acres"].sum().unstack(fill_value=0)
area = area.reindex(columns=["Prescribed", "Military", "Lightning", "Unknown"], fill_value=0)

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
counts.plot(kind="bar", stacked=True, ax=axes[0], color=[colors[c] for c in counts.columns])
axes[0].set_ylabel("Number of fires")
axes[0].set_title("Fires per month by ignition source (1996–2005, polygon-level)")
area.plot(kind="bar", stacked=True, ax=axes[1], color=[colors[c] for c in area.columns])
axes[1].set_ylabel("Acres burned")
axes[1].set_xlabel("Month")
plt.tight_layout()
plt.show()

# --- Build a date column for day-window checks ---
old_gdf["date"] = pd.to_datetime(
    dict(year=old_gdf["Year_"], month=old_gdf["Month_"], day=old_gdf["day_"]),
    errors="coerce",
)
n_bad_date = old_gdf["date"].isna().sum()
print(f"\n{n_bad_date} rows have an unparseable date and can't be matched by date window")

# --- Group fires into events ---
# Prescribed: each polygon is its own event (unit boundaries, not inferred events)
# Wildfires (Military/Lightning/Unknown): adjacent polygons within DAY_WINDOW days
old_gdf["event_id"] = -1
next_id = 0

prescribed_idx = old_gdf[old_gdf["source"] == "Prescribed"].index
old_gdf.loc[prescribed_idx, "event_id"] = range(next_id, next_id + len(prescribed_idx))
next_id += len(prescribed_idx)

wildfire_mask = old_gdf["source"] != "Prescribed"
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

print(f"\n{old_gdf.shape[0]:,} polygons grouped into {old_gdf['event_id'].nunique():,} fire events")
print(f"(Prescribed: 1 polygon = 1 event. Wildfires: GAP_TOLERANCE={GAP_TOLERANCE}m, DAY_WINDOW={DAY_WINDOW} days)")
print(old_gdf.groupby("source")["event_id"].nunique())

event_sizes = old_gdf.groupby("event_id").size()
print(event_sizes.value_counts().sort_index())

# --- Dissolve polygons into one row per fire event ---
# For wildfire events spanning >1 day, use the earliest date/Month_/day_ as the event's date
fires_gdf = old_gdf.sort_values("date").dissolve(by="event_id", aggfunc={
    "source": "first",
    "Year_": "first",
    "Month_": "first",
    "day_": "first",
    "Acres": "sum",
}).reset_index()

print(f"\n{old_gdf.shape[0]:,} polygons dissolved into {fires_gdf.shape[0]:,} fire events")
print(fires_gdf["source"].value_counts())

# --- Boundary for context ---
_boundary_path = Path(r"C:\Users\mateo\projects\piFire\FIRE\data\APAFR boundary\boundary.shp")
boundary = gpd.read_file(_boundary_path)
print(f"Boundary CRS: {boundary.crs}")
if boundary.crs != old_gdf.crs:
    boundary = boundary.to_crs(old_gdf.crs)

# --- Days with at least 10 ha of wildfire activity ---
wildfire_gdf = fires_gdf[fires_gdf["source"] != "Prescribed"].copy()
wildfire_gdf["ha"] = wildfire_gdf["Acres"] * 0.404686

# --- Days with at least 10 ha of wildfire activity ---
busy = (
    wildfire_gdf
    .groupby(["Year_", "Month_", "day_"])["ha"]
    .sum()
    .sort_values(ascending=False)
)
busy = busy[busy >= 10]
print(f"{len(busy)} days with >= 10 ha of wildfire activity")
print(busy.head(15))

# --- Output folder ---
_fig_folder = Path.home() / "Desktop" / "FIRES"
_fig_folder.mkdir(parents=True, exist_ok=True)


def plot_day(year, month, day):
    sub = old_gdf[(old_gdf["Year_"] == year) & (old_gdf["Month_"] == month) & (old_gdf["day_"] == day)]
    if sub.empty:
        print(f"No fires on {year}-{month}-{day}")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.8)
    sub.plot(ax=ax, color=[colors[s] for s in sub["source"]], edgecolor="black", linewidth=0.3)

    for _, row in sub.iterrows():
        c = row.geometry.centroid
        ax.annotate(str(row["event_id"]), (c.x, c.y), fontsize=6, ha="center")

    n_events = sub["event_id"].nunique()
    ax.set_title(f"{year}-{month:02d}-{day:02d}: {len(sub)} polygons, {n_events} events")
    plt.tight_layout()

    _out = _fig_folder / f"{year}-{month:02d}-{day:02d}.png"
    fig.savefig(_out, dpi=150)
    plt.close(fig)
    print(f"Wrote {_out}")


for (year, month, day), n in busy.items():
    plot_day(year, month, day)

# --- Event-level graph: fires per month by source ---
counts_events = fires_gdf.groupby(["Month_", "source"]).size().unstack(fill_value=0)
counts_events = counts_events.reindex(columns=["Prescribed", "Military", "Lightning", "Unknown"], fill_value=0)

area_events = fires_gdf.groupby(["Month_", "source"])["Acres"].sum().unstack(fill_value=0)
area_events = area_events.reindex(columns=["Prescribed", "Military", "Lightning", "Unknown"], fill_value=0)

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
counts_events.plot(kind="bar", stacked=True, ax=axes[0], color=[colors[c] for c in counts_events.columns])
axes[0].set_ylabel("Number of fires")
axes[0].set_title("Fires per month by ignition source (1996–2005, event-level)")
area_events.plot(kind="bar", stacked=True, ax=axes[1], color=[colors[c] for c in area_events.columns])
axes[1].set_ylabel("Acres burned")
axes[1].set_xlabel("Month")
plt.tight_layout()
plt.show()

# --- Target areas (named ranges only, drop the "out" catch-all) ---
targets = boundary[boundary["Name"].notna()].copy()
print(targets[["Name", "Range", "Acres"]])

# --- Distance from each wildfire event to nearest target (0 if inside one) ---
target_union = targets.geometry.union_all()

wildfire_gdf = fires_gdf[fires_gdf["source"].isin(["Lightning", "Military"])].copy()
wildfire_gdf["ha"] = wildfire_gdf["Acres"] * 0.404686
wildfire_gdf["dist_to_target"] = wildfire_gdf.geometry.distance(target_union)

print(wildfire_gdf.groupby("source")["dist_to_target"].describe())

# --- Map: known wildfires colored by source, targets shown for context ---
fig, ax = plt.subplots(figsize=(8, 8))
boundary.boundary.plot(ax=ax, color="black", linewidth=0.5)
targets.plot(ax=ax, color="lightgray", edgecolor="black", alpha=0.6)
wildfire_gdf.plot(ax=ax, color=[colors[s] for s in wildfire_gdf["source"]], edgecolor="black", linewidth=0.4)
ax.set_title("Known wildfires vs. target areas (1996–2005)")
plt.tight_layout()
plt.show()



# --- Features on known wildfires ---
model_df = wildfire_gdf.copy()

# Day of year (cyclical)
model_df["doy"] = pd.to_datetime(
    dict(year=model_df["Year_"], month=model_df["Month_"], day=model_df["day_"]), errors="coerce"
).dt.dayofyear
n_bad = model_df["doy"].isna().sum()
print(f"{n_bad} events dropped for unparseable date")
model_df = model_df.dropna(subset=["doy"])

model_df["doy_sin"] = np.sin(2 * np.pi * model_df["doy"] / 365)
model_df["doy_cos"] = np.cos(2 * np.pi * model_df["doy"] / 365)

# Shape: Polsby-Popper compactness (1 = circle, lower = more jagged/elongated)
model_df["compactness"] = (4 * np.pi * model_df.geometry.area) / (model_df.geometry.length ** 2)

# Size
model_df["log_ha"] = np.log1p(model_df["ha"])

# Target: 1 = Military, 0 = Lightning
model_df["y"] = (model_df["source"] == "Military").astype(int)

features = ["doy_sin", "doy_cos", "dist_to_target", "compactness", "log_ha"]
# features = ["dist_to_target", "compactness", "log_ha"]
X = model_df[features].values
y = model_df["y"].values
print(f"n = {len(model_df)} ({(y==1).sum()} Military, {(y==0).sum()} Lightning)")

# --- Cross-validated logistic regression ---
clf = make_pipeline(StandardScaler(), LogisticRegression())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]  # P(Military)
preds = (probs >= 0.5).astype(int)

print("\nConfusion matrix (rows=true, cols=pred; 0=Lightning, 1=Military):")
print(confusion_matrix(y, preds))
print()
print(classification_report(y, preds, target_names=["Lightning", "Military"]))
print(f"AUC: {roc_auc_score(y, probs):.3f}")

# --- Fit on full data to inspect coefficients ---
clf.fit(X, y)
coefs = pd.Series(clf.named_steps["logisticregression"].coef_[0], index=features)
print("\nStandardized coefficients (positive = pushes toward Military):")
print(coefs.sort_values())

# --- Exclude Unknowns with a known cause that isn't Lightning/Military ---
exclude_ignitions = ["Catalytic Conv", "Heavy Equip"]
exclude_ids = set(old_gdf.loc[old_gdf["Ignition"].isin(exclude_ignitions), "event_id"])
print(f"Excluding {len(exclude_ids)} event(s): {exclude_ignitions}")

# --- Refit full model (with date) on known wildfires ---
features = ["doy_sin", "doy_cos", "dist_to_target", "compactness", "log_ha"]
X = model_df[features].values
y = model_df["y"].values

clf_full = make_pipeline(StandardScaler(), LogisticRegression())
clf_full.fit(X, y)

# --- Build same features for Unknown wildfire events ---
unknown_gdf = fires_gdf[(fires_gdf["source"] == "Unknown") & (~fires_gdf["event_id"].isin(exclude_ids))].copy()

unknown_gdf["ha"] = unknown_gdf["Acres"] * 0.404686
unknown_gdf["dist_to_target"] = unknown_gdf.geometry.distance(target_union)

unknown_gdf["doy"] = pd.to_datetime(
    dict(year=unknown_gdf["Year_"], month=unknown_gdf["Month_"], day=unknown_gdf["day_"]), errors="coerce"
).dt.dayofyear
n_bad = unknown_gdf["doy"].isna().sum()
print(f"{n_bad} Unknown event(s) dropped: unparseable date")
unknown_gdf = unknown_gdf.dropna(subset=["doy"])

unknown_gdf["doy_sin"] = np.sin(2 * np.pi * unknown_gdf["doy"] / 365)
unknown_gdf["doy_cos"] = np.cos(2 * np.pi * unknown_gdf["doy"] / 365)
unknown_gdf["compactness"] = (4 * np.pi * unknown_gdf.geometry.area) / (unknown_gdf.geometry.length ** 2)
unknown_gdf["log_ha"] = np.log1p(unknown_gdf["ha"])

unknown_gdf["p_military"] = clf_full.predict_proba(unknown_gdf[features].values)[:, 1]
unknown_gdf = unknown_gdf.sort_values("ha", ascending=False)

print(unknown_gdf[["Year_", "Month_", "day_", "ha", "p_military"]].to_string())

# --- Single map: all Unknown events, colored by predicted class, labeled with P(Military) ---
unknown_gdf["pred_label"] = np.where(unknown_gdf["p_military"] >= 0.5, "Military", "Lightning")

fig, ax = plt.subplots(figsize=(10, 10))
boundary.boundary.plot(ax=ax, color="black", linewidth=0.5)
targets.plot(ax=ax, color="lightgray", edgecolor="black", alpha=0.6)

unknown_gdf.plot(ax=ax, color=[colors[c] for c in unknown_gdf["pred_label"]], edgecolor="black", linewidth=0.4)

for _, row in unknown_gdf.iterrows():
    c = row.geometry.centroid
    ax.annotate(f"{row.p_military:.2f}", (c.x, c.y), fontsize=7, ha="center")

ax.set_title("Unknown wildfires (1996–2005): predicted source and P(Military)")
plt.tight_layout()
plt.show()