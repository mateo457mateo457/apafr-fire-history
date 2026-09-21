import pandas as pd
import matplotlib.pyplot as plt


# First, the old data
df = pd.read_csv(r"C:\Users\mateo\projects\piFire\MWB\Analysis\data\processed\fire\fire_data.csv")
df["month"] = pd.to_datetime(df["DATE"]).dt.month

t = df.groupby("month")[["ha_lightning", "ha_ordnance", "ha_prescribed"]].sum().reindex(range(1, 13)).fillna(0)
t.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
t.columns = ["Lightning", "Ordnance", "Prescribed"]
print(t.round(0))

t.plot(kind="bar", color=["gold", "orange", "green"], edgecolor="black", figsize=(10, 5))
plt.ylabel("Area burned (ha)")
plt.xlabel("")
plt.xticks(rotation=0)
plt.title("Monthly area burned by ignition source, 1997–2009")
plt.tight_layout()
plt.show()

# The new data
AC2HA = 0.404686
MIL = ["Ordnance", "Ordinance", "Smokey Sam", "Smoke Grenade", "seek and destroy",
       "Surfacing EOD", "Simulator", "Pyrotechs"]

# ---- 1977-2005 spreadsheet
a = pd.read_excel(r"C:\Users\mateo\projects\piFire\FIRE\data\Burn History 1977_2026\1977_2005\Burn History 1977-2005 (excel).xls")
a = a[a["Month_"].between(1, 12)].copy()
a["Type"] = a["Type"].astype(str).str.strip()
a["ig"] = a["Ignition"].astype(str).str.strip()
a["source"] = "Unknown"
a.loc[a["Type"] == "Prescribed", "source"] = "Prescribed"
a.loc[(a["Type"] == "Wildfire") & (a["ig"] == "Lightning"), "source"] = "Lightning"
a.loc[(a["Type"] == "Wildfire") & a["ig"].isin(MIL), "source"] = "Ordnance"
a = a[a["Type"].isin(["Prescribed", "Wildfire"])]
a = pd.DataFrame({"year": a["Year_"], "month": a["Month_"], "source": a["source"], "ha": a["Acres"] * AC2HA})

# ---- 2006-2026 layer
b = pd.read_excel(r"C:\Users\mateo\projects\piFire\FIRE\data\Burn History 1977_2026\2006_2026\Burn History 2006-2026 (excel).xls")
d = pd.to_datetime(b["fireStartDate"])
live = b["narrative"].astype(str).str.contains("live fire", case=False, na=False)
src = pd.Series("Unknown", index=b.index)
src[b["wildlandFireType"] == "prescribed"] = "Prescribed"
src[b["wildfireCauseType"] == "lightningStrike"] = "Lightning"
src[b["wildfireCauseType"].astype(str).str.startswith("milLive") | live] = "Ordnance"
b = pd.DataFrame({"year": d.dt.year, "month": d.dt.month, "source": src, "ha": b["areaSize"] * AC2HA})

# ---- same window as the old data: Oct 1996 - Sep 2009
new = pd.concat([a, b])
ym = new["year"] * 100 + new["month"]
new = new[(ym >= 199610) & (ym <= 200909)]

t = new.pivot_table(index="month", columns="source", values="ha", aggfunc="sum").reindex(range(1, 13)).fillna(0)
t = t[["Lightning", "Ordnance", "Prescribed", "Unknown"]]
t.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
print(t.round(0))

t.plot(kind="bar", color=["gold", "orange", "green", "gray"], edgecolor="black", figsize=(10, 5))
plt.ylabel("Area burned (ha)")
plt.xlabel("")
plt.xticks(rotation=0)
plt.title("Monthly area burned by ignition source, 1997–2009 (new data)")
plt.tight_layout()
plt.show()

# Now plot the new data for 2011 onwards
b = pd.read_excel(r"C:\Users\mateo\projects\piFire\FIRE\data\Burn History 1977_2026\2006_2026\Burn History 2006-2026 (excel).xls")
d = pd.to_datetime(b["fireStartDate"])
live = b["narrative"].astype(str).str.contains("live fire", case=False, na=False)

src = pd.Series("Unknown", index=b.index)
src[b["wildlandFireType"] == "prescribed"] = "Prescribed"
src[b["wildfireCauseType"] == "lightningStrike"] = "Lightning"
src[b["wildfireCauseType"].astype(str).str.startswith("milLive") | live] = "Ordnance"

new = pd.DataFrame({"year": d.dt.year, "month": d.dt.month, "source": src, "ha": b["areaSize"] * AC2HA})
new = new[new["year"] >= 2011]

t = new.pivot_table(index="month", columns="source", values="ha", aggfunc="sum").reindex(range(1, 13)).fillna(0)
t = t[["Lightning", "Ordnance", "Prescribed", "Unknown"]]
t.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
print(t.round(0))

t.plot(kind="bar", color=["gold", "orange", "green", "gray"], edgecolor="black", figsize=(10, 5))
plt.ylabel("Area burned (ha)")
plt.xlabel("")
plt.xticks(rotation=0)
plt.title("Monthly area burned by ignition source, 2011–2026")
plt.tight_layout()
plt.show()