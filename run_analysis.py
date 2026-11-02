"""Retail site selection and white-space analysis: where should a store chain open next?

1. Maps every US Starbucks to its county (point-in-polygon on county boundaries).
2. Learns how many stores a county's demographics and job base can support (stores per 100k
   residents, Poisson gradient boosting), validated on states the model never saw.
3. White space = expected minus actual stores, using out-of-state predictions only.
4. Tests whether the gap is explained by a rival chain (Dunkin') and ranks the counties
   with demand but little competition.
Reproduces every number in README.md:  python run_analysis.py"""
import json, numpy as np, pandas as pd
from matplotlib.path import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

R = {}
# ---------------- 1. Stores -> counties (point in polygon) ----------------
st = pd.read_csv("data/starbucks_us.csv")
geo = json.load(open("data/us_counties.geojson"))
pts = st[["longitude", "latitude"]].to_numpy()
fips = np.array([None] * len(st), dtype=object)
for f in geo["features"]:
    g = f["geometry"]
    for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
        ring = np.asarray(poly[0]); lo, hi = ring.min(0), ring.max(0)
        box = (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]) & (fips == None)
        if box.any():
            idx = np.where(box)[0]
            fips[idx[Path(ring).contains_points(pts[idx])]] = f["id"]
st["fips"] = fips
dk = pd.read_csv("data/dunkin_us.csv", dtype={"fips": str}); dk["fips"] = dk.fips.str.zfill(5)
R["stores"] = dict(starbucks_us=int(len(st)), starbucks_mapped_to_county=int(st.fips.notna().sum()), dunkin_us=int(len(dk)))

# ---------------- 2. County features (demographics and jobs only: no brand or location terms) ----------------
c = pd.read_csv("data/us_counties.csv", dtype={"fips": str})
c["fips"] = c.fips.str.zfill(5)
c = c[~c.state.isin(["AK", "HI", "PR"])].reset_index(drop=True)      # contiguous US
pop = c["population/2017"].to_numpy(float)
X = pd.DataFrame({
    "log_population": np.log(pop),
    "log_density": np.log(pop / c["land_area (km^2)"]),
    "avg_income": c["avg_income"],
    "bachelors_plus": c["edu/bachelors+"],
    "poverty_rate": c["poverty-rate"],
    "age_20_39": c[["age/20-24", "age/25-29", "age/30-34", "age/35-39"]].sum(axis=1),
    "age_65_plus": c[["age/65-69", "age/70-74", "age/75-79", "age/80-84", "age/85+"]].sum(axis=1),
    "unemployment": c["bls/2016/unemployed"] / c["bls/2016/labor_force"],
    "housing_cost": c["cost-of-living/housing_costs"]})
for k, col in [("jobs_professional", "Professional, scientific, and technical services"), ("jobs_finance", "Finance and insurance"),
               ("jobs_information", "Information"), ("jobs_retail", "Retail trade"), ("jobs_food_service", "Accommodation and food services")]:
    X[k] = c[f"industry/{col}/employees"].fillna(0) / pop
y = c.fips.map(st.groupby("fips").size()).fillna(0).to_numpy()
dunkin = c.fips.map(dk.groupby("fips").size()).fillna(0).to_numpy()
R["counties"] = dict(counties=int(len(c)), states=int(c.state.nunique()), starbucks_in_counties=int(y.sum()),
                     counties_with_starbucks=int((y > 0).sum()),
                     top_10pct_counties_store_share=float(np.sort(y)[::-1][: len(y) // 10].sum() / y.sum()))

# ---------------- 3. Expected stores per county, validated on unseen states ----------------
def model():
    return HistGradientBoostingRegressor(loss="poisson", max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=7)
# target = stores per 100k residents, weighted by population, then scaled back to a store count,
# so expected stores grow with county size instead of being capped by the biggest training counties
rate, w = y / pop * 1e5, pop / pop.mean()
oof, base = np.zeros(len(c)), np.zeros(len(c))
for trn, tst in GroupKFold(n_splits=5).split(X, y, c.state):
    oof[tst] = model().fit(X.iloc[trn], rate[trn], sample_weight=w[trn]).predict(X.iloc[tst]) * pop[tst] / 1e5
    base[tst] = pop[tst] * y[trn].sum() / pop[trn].sum()          # baseline: national stores per resident

def poisson_dev(p):
    p = np.clip(p, 1e-9, None); t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / p), 0)
    return 2 * np.sum(t - (y - p))
null = poisson_dev(np.full(len(y), y.mean()))
def scores(p):
    k = len(p) // 10
    return dict(deviance_explained=1 - poisson_dev(p) / null, mae=float(np.mean(np.abs(y - p))),
                top_decile_capture=float(y[np.argsort(-p)[:k]].sum() / y.sum()), has_store_auc=float(roc_auc_score(y > 0, p)))
R["validation"] = dict(method="5-fold cross-validation grouped by state: every county is predicted by a model that never saw its state",
                       baseline=scores(base), model=scores(oof))
R["validation"]["mae_reduction"] = 1 - R["validation"]["model"]["mae"] / R["validation"]["baseline"]["mae"]

full = model().fit(X, rate, sample_weight=w)
pi = permutation_importance(full, X, rate, sample_weight=w, scoring="neg_mean_poisson_deviance", n_repeats=5, random_state=7)
imp = pd.Series(pi.importances_mean, index=X.columns).clip(lower=0)
R["drivers"] = (imp / imp.sum()).sort_values(ascending=False).round(4).to_dict()

# ---------------- 4. White space and the competitor test ----------------
c = c.assign(population=pop, starbucks=y, expected=oof, gap=oof - y, dunkin=dunkin,
             dunkin_per_100k=dunkin / pop * 1e5)
big = c[c.expected >= 5].copy()                                     # counties large enough to judge penetration
big["penetration"] = big.starbucks / big.expected                   # actual / expected stores
big["dunkin_band"] = pd.cut(big.dunkin_per_100k, [-0.01, 0, 2, 5, 1e9], labels=["none", "0-2", "2-5", "5+"])
def by_band(d):
    return d.groupby("dunkin_band", observed=True).apply(
        lambda g: pd.Series(dict(counties=len(g), penetration=g.starbucks.sum() / g.expected.sum())), include_groups=False)
band = by_band(big)
east = by_band(big[c.loc[big.index, "longitude (deg)"] > -90])   # robustness: east of ~Mississippi only,
R["competition"] = dict(                                          # so the effect is not just West vs East
    counties_judged=int(len(big)),
    penetration_by_dunkin_per_100k=band.round(3).reset_index().to_dict("records"),
    east_only_penetration_by_dunkin_per_100k=east.round(3).reset_index().to_dict("records"),
    spearman_gap_vs_dunkin=float(big[["penetration", "dunkin_per_100k"]].corr("spearman").iloc[0, 1]))

# uncontested white space: demand the model expects, not yet served, and little rival presence
c["uncontested"] = (c.gap >= 3) & (c.dunkin_per_100k < 1)
top = c[c.uncontested].sort_values("gap", ascending=False)
R["white_space"] = dict(
    total_gap_stores=float(c.gap.clip(lower=0).sum()),
    counties_gap_ge_3=int((c.gap >= 3).sum()),
    uncontested_counties=int(c.uncontested.sum()),
    uncontested_gap_stores=float(top.gap.sum()),
    uncontested_population=float(top.population.sum()),
    zero_store_counties_expected_ge_1=int(((c.starbucks == 0) & (c.expected >= 1)).sum()),
    top10_uncontested=top.head(10)[["name", "state", "population", "starbucks", "expected", "gap", "dunkin"]]
        .round(1).to_dict("records"),
    most_oversupplied=c.sort_values("gap").head(5)[["name", "state", "starbucks", "expected"]].round(1).to_dict("records"))
c[["fips", "name", "state", "population", "starbucks", "expected", "gap", "dunkin", "dunkin_per_100k", "uncontested"]] \
    .sort_values("gap", ascending=False).round(2).to_csv("results/county_white_space.csv", index=False)
json.dump(R, open("results/results.json", "w"), indent=2, default=float)
print(json.dumps(R, indent=2, default=float))
