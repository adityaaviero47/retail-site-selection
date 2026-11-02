"""Charts for README from results/: white-space map and model / competition summary."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm

R = json.load(open("results/results.json"))
ws = pd.read_csv("results/county_white_space.csv", dtype={"fips": str}).set_index("fips")
geo = json.load(open("data/us_counties.geojson"))
polys, vals = [], []
for f in geo["features"]:
    if f["id"] not in ws.index: continue
    r = ws.loc[f["id"]]
    v = np.clip(r.gap / r.population * 1e5, -4, 4)                 # gap in stores per 100k residents
    g = f["geometry"]
    for p in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
        polys.append(np.asarray(p[0])); vals.append(v)
fig, ax = plt.subplots(figsize=(11, 6.2))
pc = PolyCollection(polys, array=np.array(vals), cmap="RdBu_r", norm=TwoSlopeNorm(0, -4, 4), edgecolors="white", linewidths=0.08)
ax.add_collection(pc); ax.set_xlim(-125, -66.5); ax.set_ylim(24, 50); ax.set_aspect(1.25); ax.axis("off")
cb = fig.colorbar(pc, ax=ax, shrink=0.6, pad=0.01); cb.set_label("Expected minus actual stores per 100k residents")
ax.set_title("Starbucks white space by county (red = under-served, blue = saturated)", fontsize=13)
top = ws[ws.uncontested].head(10)
cty = pd.read_csv("data/us_counties.csv", dtype={"fips": str}); cty["fips"] = cty.fips.str.zfill(5); cty = cty.set_index("fips")
for f_, r in top.iterrows():
    ax.plot(cty.loc[f_, "longitude (deg)"], cty.loc[f_, "latitude (deg)"], "o", ms=6, mfc="none", mec="black", mew=1.2)
plt.tight_layout(); plt.savefig("results/white_space_map.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
v = R["validation"]; vals = [v["baseline"]["mae"], v["model"]["mae"]]
ax[0].bar(["Stores per resident\n(national rate)", "Gradient-boosted\ndemand model"], vals, color=["#9aa5ad", "#0052CC"])
for i, x in enumerate(vals): ax[0].text(i, x + 0.04, f"{x:.2f}", ha="center")
ax[0].set_ylabel("Mean abs. error (stores per county)"); ax[0].set_title("Error on states the model never saw")
b = pd.DataFrame(R["competition"]["penetration_by_dunkin_per_100k"])
ax[1].bar(b.dunkin_band, b.penetration * 100, color="#0052CC"); ax[1].axhline(100, color="#555", lw=0.8, ls="--")
for i, x in enumerate(b.penetration * 100): ax[1].text(i, x + 2, f"{x:.0f}%", ha="center")
ax[1].set_xlabel("Dunkin' stores per 100k residents"); ax[1].set_ylabel("Starbucks actual / expected (%)")
ax[1].set_title("Penetration falls where the rival is strong")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("results/summary.png", dpi=150)
