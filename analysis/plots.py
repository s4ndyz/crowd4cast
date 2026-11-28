#!/usr/bin/env python3
"""Report figures from results/*.csv (static PNGs, light mode).  python plots.py [dev|test]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dataset as D
import evaluate as E

STAGE = sys.argv[1] if len(sys.argv) > 1 else "dev"
FIG = D.HERE / "figures"
TEXT, TEXT2, GRID, SURF = "#191919", "#666666", "#e6e6e6", "#ffffff"
S1, S2, S3 = "#257de0", "#f47230", "#00b894"          # categorical slots 1-3 (validated all-pairs)
BASE = "#9c9c9c"                                      # baselines: neutral gray
HORIZONS = ["1h", "3h", "24h"]
LABEL = {"curve": "Typical curve (site)", "persistence": "Persistence", "ridge": "Ridge",
         "ridge_resid": "Ridge (residual)", "rf": "Random forest", "rf_resid": "RF (residual)"}

plt.rcParams.update({
    "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
    "ytick.color": TEXT2, "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def mae_by_horizon(pred):
    """Daily MAE per model with 95% bootstrap CI over days, one panel per horizon."""
    rng = np.random.default_rng(E.SEED)
    models = E.BASELINES + E.ML
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True)
    for ax, h in zip(axes, HORIZONS):
        p = pred[pred.horizon == h]
        rows = []
        for m in models:
            d = E.daily_scores(p, m).to_numpy()
            b = [rng.choice(d, len(d)).mean() for _ in range(3000)]
            rows.append((m, d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)))
        ypos = np.arange(len(rows))[::-1]
        for y, (m, mu, lo, hi) in zip(ypos, rows):
            c = BASE if m in E.BASELINES else S1
            ax.barh(y, mu, height=0.55, color=c, edgecolor=SURF, linewidth=2)
            ax.plot([lo, hi], [y, y], color=TEXT, lw=1.2)
            ax.text(hi + 0.2, y, f"{mu:.2f}", va="center", fontsize=8.5, color=TEXT)
        curve = rows[0][1]
        ax.axvline(curve, color=TEXT2, lw=1, ls=(0, (3, 3)))
        ax.set_yticks(ypos, [LABEL[r[0]] for r in rows] if ax is axes[0] else [])
        ax.set_title(f"{h} ahead", loc="left", fontsize=10, color=TEXT)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Daily MAE (pct. points)")
    fig.suptitle(f"Mean daily MAE with 95% bootstrap CI over days ({STAGE}; dashed = typical curve)",
                 x=0.01, y=1.04, ha="left", fontsize=10.5, color=TEXT)
    fig.savefig(FIG / f"{STAGE}_mae_by_horizon.png")


def ablation():
    a = pd.read_csv(D.HERE / "results" / "dev_ablation.csv")
    a = a[a.kind == "cumulative"]
    steps = a.groups.unique()
    names = ["time", "+ recent", "+ typical", "+ weather", "+ calendar"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for h, c in zip(HORIZONS, (S1, S2, S3)):
        s = a[a.horizon == h].set_index("groups").loc[steps, "daily_MAE"].to_numpy()
        ax.plot(range(len(s)), s, color=c, lw=2, marker="o", ms=6, mec=SURF, mew=2, label=h)
        ax.text(len(s) - 1 + 0.12, s[-1], f"{h}", va="center", fontsize=9, color=TEXT)
    ax.set_xticks(range(len(steps)), names)
    ax.set_xlim(-0.2, len(steps) - 0.5)
    ax.set_ylabel("Daily MAE (pct. points)")
    ax.legend(frameon=False, loc="upper right", ncol=3)
    ax.set_title("Feature ablation, random forest (dev): groups added left to right",
                 loc="left", fontsize=10.5, color=TEXT)
    fig.savefig(FIG / "dev_ablation.png")


def error_by_hour(pred, h="1h"):
    chosen = E.json.loads((D.HERE / "results" / "chosen_params.json").read_text())
    best = chosen[h]["_best"]
    p = pred[pred.horizon == h].copy()
    p["hr"] = pd.to_datetime(p.target).dt.hour
    g = p.groupby("hr")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for m, c in (("curve", BASE), (best, S1)):
        s = g.apply(lambda x: (x[m] - x.y).abs().mean(), include_groups=False)
        ax.plot(s.index, s.to_numpy(), color=c, lw=2, marker="o", ms=5, mec=SURF, mew=1.5, label=LABEL[m])
    n = g.date.nunique()
    ax.set_xticks(range(7, 24, 2))
    ax.set_xlabel("Hour of day (target)")
    ax.set_ylabel("MAE (pct. points)")
    ax.legend(frameon=False, loc="upper left")
    thin = n[n < 3]
    if len(thin):
        ax.axvspan(thin.index.min() - 0.5, thin.index.max() + 0.5, color=GRID, alpha=0.6, lw=0)
        ax.text(thin.index.min() - 0.4, ax.get_ylim()[1] * 0.92, "< 3 days of data", fontsize=8, color=TEXT2)
    ax.set_title(f"Error by hour, {h} ahead ({STAGE})", loc="left", fontsize=10.5, color=TEXT)
    fig.savefig(FIG / f"{STAGE}_error_by_hour_{h}.png")


def example_day(pred, day=None, h="1h"):
    chosen = E.json.loads((D.HERE / "results" / "chosen_params.json").read_text())
    best = chosen[h]["_best"]
    p = pred[pred.horizon == h].copy()
    p["target"] = pd.to_datetime(p.target)
    if day is None:                                       # the day with the longest evening coverage
        tod = p.target.dt.hour * 60 + p.target.dt.minute
        day = tod.groupby(p.date).max().idxmax()
    d = p[p.date == day].sort_values("target")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = d.target.dt.hour + d.target.dt.minute / 60
    ax.plot(x, d.y, color=TEXT, lw=2, label="Actual")
    ax.plot(x, d.curve, color=BASE, lw=2, ls=(0, (4, 3)), label=LABEL["curve"])
    ax.plot(x, d[best], color=S1, lw=2, label=f"{LABEL[best]}, {h} ahead")
    for lvl in D.BANDS:
        ax.axhline(lvl, color=TEXT2, lw=0.8, ls=":")
        ax.text(x.min(), lvl + 1.5, f"{lvl}% (site 'busy' threshold)" if lvl == 60 else f"{lvl}%",
                fontsize=8, color=TEXT2)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Occupancy (% of capacity)")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_title(f"ARC on {pd.Timestamp(day):%a %b %-d}", loc="left", fontsize=10.5, color=TEXT)
    fig.savefig(FIG / f"{STAGE}_example_day.png")


def main():
    FIG.mkdir(exist_ok=True)
    pred = pd.read_csv(D.HERE / "results" / f"{STAGE}_predictions.csv", parse_dates=["date"])
    mae_by_horizon(pred)
    error_by_hour(pred)
    example_day(pred)
    if STAGE == "dev":
        ablation()
    print("wrote", sorted(f.name for f in FIG.glob(f"*.png")))


if __name__ == "__main__":
    main()
