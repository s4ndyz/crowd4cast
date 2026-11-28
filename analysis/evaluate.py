#!/usr/bin/env python3
"""Train and evaluate all models.

Stages
  dev    Development data only (default 10/28-11/20). Rolling-origin folds: train on every
         day before the validation week, validate on that week. Used to pick hyperparameters
         and to run the feature ablation. Writes results/dev_*.csv and chosen_params.json.
  test   Final, run ONCE after the data are frozen (TA: keep the last week untouched until
         every choice is final). Uses chosen_params.json from dev; for each test day, refits
         on all data before that day (as a nightly-retrained app would) and predicts the day.

Models (one per horizon): curve (site's typical curve), persistence (now = later),
ridge / rf (predict occupancy), ridge_resid / rf_resid (predict actual - typical, add it back).

Significance: one MAE per test day; paired Wilcoxon signed-rank (and paired t) on daily MAE,
Holm-corrected within each horizon; bootstrap 95% CI over days for the MAE difference.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import dataset as D

RES = D.HERE / "results"
SEED = 441
DEV = ("2026-10-28", "2026-11-20")
DEV_FOLDS = [("2026-11-04", "2026-11-10"), ("2026-11-11", "2026-11-17"), ("2026-11-18", "2026-11-20")]
TEST = ("2026-11-30", "2026-12-09")
GRID = {
    "ridge": [{"alpha": a} for a in (0.1, 1, 10, 100, 1000)],
    "rf": [dict(max_depth=d, min_samples_leaf=l, max_features=f)
           for d, l, f in itertools.product((None, 6, 12), (1, 5, 20), (1.0, 0.5, "sqrt"))],
}
ALL_GROUPS = list(D.FEATURE_GROUPS)
ML = ["ridge", "ridge_resid", "rf", "rf_resid"]
BASELINES = ["curve", "persistence"]


def features(groups):
    return [c for g in groups for c in D.FEATURE_GROUPS[g]]


def make_model(kind, params, cols):
    if kind == "ridge":
        cat = [c for c in ("dow",) if c in cols]
        num = [c for c in cols if c not in cat]
        ct = ColumnTransformer([("num", StandardScaler(), num)] +
                               ([("cat", OneHotEncoder(handle_unknown="ignore"), cat)] if cat else []))
        # hour enters ridge as one-hot too (occupancy vs hour is not a line)
        if "hour" in cols:
            ct = ColumnTransformer([("num", StandardScaler(), [c for c in num if c != "hour"]),
                                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat + ["hour"])])
        return make_pipeline(ct, Ridge(**params))
    return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, **params)


def fit_predict(name, params, train, test, groups=ALL_GROUPS):
    """Return predictions for `test` from model `name` trained on `train`."""
    if name == "curve":
        return test["typ_target"].to_numpy()
    if name == "persistence":
        return test["persistence"].to_numpy()
    kind, resid = name.split("_")[0], name.endswith("_resid")
    cols = features(groups)
    m = make_model(kind, params, cols)
    target = train["y"] - train["typ_target"] if resid else train["y"]
    m.fit(train[cols], target)
    p = m.predict(test[cols])
    return p + test["typ_target"].to_numpy() if resid else p


def daily_scores(df, pred_col):
    g = df.assign(ae=(df[pred_col] - df.y).abs(), se=(df[pred_col] - df.y) ** 2).groupby("date")
    return g.ae.mean()


def summarize(df, models):
    rows = []
    for m in models:
        e = df[m] - df.y
        yb, pb = D.band(df.y), D.band(df[m])
        daily = daily_scores(df, m)
        rows.append({"model": m, "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                     "bias": e.mean(), "band_acc": (yb == pb).mean(),
                     "band_macroF1": f1_score(yb, pb, labels=[0, 1, 2], average="macro", zero_division=0),
                     "daily_MAE_mean": daily.mean(), "n_days": len(daily), "n": len(df)})
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_tests(df, best, others, rng):
    rows = []
    a = daily_scores(df, best)
    for o in others:
        b = daily_scores(df, o)
        d = (b - a).to_numpy()                       # >0 means `best` is better
        boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        rows.append({"model": best, "vs": o, "daily_MAE_diff": d.mean(),
                     "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                     "wins": int((d > 0).sum()), "days": len(d),
                     "p_wilcoxon": stats.wilcoxon(d).pvalue if np.any(d) else 1.0,
                     "p_ttest": stats.ttest_rel(b, a).pvalue})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_wilcoxon)
    return out


def usable(df):
    return df[df.typ_target.notna()].reset_index(drop=True)


def split(df, lo, hi):
    return df[(df.date >= lo) & (df.date <= hi)]


# ---- stages ---------------------------------------------------------------------------------
def run_dev(tables):
    RES.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    tuning, chosen, preds_all, ablation, tests = [], {}, [], [], []
    for h, df in tables.items():
        df = usable(split(df, *DEV))
        folds = [(df[df.date < lo], split(df, lo, hi)) for lo, hi in DEV_FOLDS]

        def cv(name, params, groups=ALL_GROUPS):
            parts = []
            for tr, va in folds:
                parts.append(va.assign(pred=fit_predict(name, params, tr, va, groups)))
            va = pd.concat(parts)
            return va, daily_scores(va, "pred").mean()

        chosen[h] = {}
        for name in ML:
            best = None
            for params in GRID[name.split("_")[0]]:
                _, score = cv(name, params)
                tuning.append({"horizon": h, "model": name, "params": json.dumps(params), "daily_MAE": score})
                if best is None or score < best[1]:
                    best = (params, score)
            chosen[h][name] = best[0]
            print(f"[{h}] {name:12s} best {best[0]} daily MAE {best[1]:.2f}")

        # validation predictions of every model with its chosen hyperparameters
        va = pd.concat([v for _, v in folds]).copy()
        for name in BASELINES + ML:
            va[name] = cv(name, chosen[h].get(name, {}))[0]["pred"].to_numpy()
        va["horizon"] = h
        preds_all.append(va)
        best_ml = min(ML, key=lambda m: daily_scores(va, m).mean())
        chosen[h]["_best"] = best_ml
        tests.append(paired_tests(va, best_ml, [m for m in BASELINES + ML if m != best_ml], rng).assign(horizon=h))

        # feature ablation: cumulative groups (rf) and leave-one-group-out (best model)
        for k in range(1, len(ALL_GROUPS) + 1):
            g = ALL_GROUPS[:k]
            _, s = cv("rf", chosen[h]["rf"], g)
            ablation.append({"horizon": h, "model": "rf", "kind": "cumulative", "groups": "+".join(g), "daily_MAE": s})
        for drop in ALL_GROUPS:
            g = [x for x in ALL_GROUPS if x != drop]
            _, s = cv(best_ml, chosen[h][best_ml], g)
            ablation.append({"horizon": h, "model": best_ml, "kind": "drop_one", "groups": "-" + drop, "daily_MAE": s})

    preds = pd.concat(preds_all)
    pd.DataFrame(tuning).to_csv(RES / "dev_tuning.csv", index=False)
    preds.to_csv(RES / "dev_predictions.csv", index=False)
    pd.concat([summarize(p, BASELINES + ML).assign(horizon=h) for h, p in preds.groupby("horizon", sort=False)]
              ).to_csv(RES / "dev_summary.csv", index=False)
    pd.concat(tests).to_csv(RES / "dev_tests.csv", index=False)
    pd.DataFrame(ablation).to_csv(RES / "dev_ablation.csv", index=False)
    (RES / "chosen_params.json").write_text(json.dumps(chosen, indent=2))


def run_test(tables, force=False):
    out = RES / "test_summary.csv"
    if out.exists() and not force:
        raise SystemExit("Final test already run (results/test_summary.csv exists). Use --force only on purpose.")
    chosen = json.loads((RES / "chosen_params.json").read_text())
    rng = np.random.default_rng(SEED)
    preds_all, tests = [], []
    for h, df in tables.items():
        df = usable(df)
        days = sorted(split(df, *TEST).date.unique())
        if not days:
            raise SystemExit(f"No test-period data for {h}. Has the data been collected/pulled?")
        parts = []
        for day in days:                             # nightly refit on everything before `day`
            tr, te = df[df.date < day], df[df.date == day].copy()
            for name in BASELINES + ML:
                te[name] = fit_predict(name, chosen[h].get(name, {}), tr, te)
            parts.append(te)
        te = pd.concat(parts).assign(horizon=h)
        preds_all.append(te)
        best = chosen[h]["_best"]                    # fixed in dev, not re-picked on test
        tests.append(paired_tests(te, best, [m for m in BASELINES + ML if m != best], rng).assign(horizon=h))
    preds = pd.concat(preds_all)
    preds.to_csv(RES / "test_predictions.csv", index=False)
    pd.concat(tests).to_csv(RES / "test_tests.csv", index=False)
    pd.concat([summarize(p, BASELINES + ML).assign(horizon=h) for h, p in preds.groupby("horizon", sort=False)]
              ).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["dev", "test"])
    ap.add_argument("--data-dir", default=str(D.DATA_DIR))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    tables = D.build(a.data_dir)
    run_dev(tables) if a.stage == "dev" else run_test(tables, a.force)
    pd.set_option("display.width", 200)
    for f in (["dev_summary", "dev_tests"] if a.stage == "dev" else ["test_summary", "test_tests"]):
        print("\n==", f)
        print(pd.read_csv(RES / f"{f}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
