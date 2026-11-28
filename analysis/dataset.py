"""Build the modelling tables from the raw collector output.

Pipeline (all local time, America/Chicago):
  1. day.csv (ARC)  -> one reading per 30-min interval (label = interval END time).
     The day feed returns the whole day so far, so a day is complete up to the last
     successful poll; intervals after that are MISSING (never forward-filled: that would
     invent the evening peak).
  2. Open-hours grid: Mon-Thu 06:30-23:00, Fri 06:30-22:00, Sat/Sun 09:30-22:00
     (interval ends). Closed dates come from external/calendar.csv.
  3. Typical curve (week.csv): the site's hourly value for hour h best matches the interval
     ending at h:30 (MAE 5.28, bias +0.09 vs 5.49 / 5.75 for h:00 / h+1:00), so it is
     placed at h:30 and averaged for :00 ends. Each week's curve is only used once it had
     been fetched (first fetched_at <= forecast origin); otherwise the same weekday/hour
     from the previous week is used, which is what the site would have shown then.
  4. One sample per (target interval with an observed reading, horizon). The forecast
     origin is target - horizon; every feature is computed from data available at origin.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "crowd4cast-data"   # clone of the `data` branch
EXT = HERE / "external"
ARC_ID = "02e5e50d-71c6-4122-aa70-7c9d4e61c27a"
HORIZONS = {"1h": pd.Timedelta("1h"), "3h": pd.Timedelta("3h"), "24h": pd.Timedelta("24h")}
STEP = pd.Timedelta("30min")
BANDS = (60, 90)                                     # site thresholds: not busy / busy / very busy
CAL_FLAGS = ["football_home", "pre_break", "post_break", "last_day_instruction", "reading_day"]
WEATHER_VARS = ["temperature_2m", "apparent_temperature", "precipitation", "snowfall",
                "wind_speed_10m", "cloud_cover"]

FEATURE_GROUPS = {
    "time": ["hour", "dow", "is_weekend", "academic_week"],
    "recent": ["y_origin", "y_lag30", "y_lag60", "resid_origin", "resid_day_mean", "has_origin"],
    "typical": ["typ_target", "typ_origin"],
    "weather": ["w_" + v for v in WEATHER_VARS] + ["w_precip_prev3h"],
    "calendar": CAL_FLAGS,
}


def band(x):
    return np.digitize(x, BANDS)                     # 0 = <60, 1 = 60-90, 2 = >=90


# ---- raw inputs ---------------------------------------------------------------------------
def load_readings(data_dir=DATA_DIR):
    d = pd.read_csv(Path(data_dir) / "day.csv", parse_dates=["interval_end_local"])
    d = d[d.facility_id == ARC_ID].drop_duplicates("interval_end_local", keep="last")
    return d.set_index("interval_end_local")["count"].astype(float).sort_index()


def load_calendar():
    c = pd.read_csv(EXT / "calendar.csv", parse_dates=["date"])
    return c


def open_grid(start, end, closed):
    """All interval-end times the ARC is open, for dates start..end (inclusive)."""
    out = []
    for day in pd.date_range(start, end, freq="D"):
        if day in closed:
            continue
        dow = day.day_of_week                          # Mon=0
        first = "09:30" if dow >= 5 else "06:30"
        last = "23:00" if dow <= 3 else "22:00"
        out.append(pd.date_range(f"{day:%Y-%m-%d} {first}", f"{day:%Y-%m-%d} {last}", freq=STEP))
    return pd.DatetimeIndex(np.concatenate(out)) if out else pd.DatetimeIndex([])


class TypicalCurve:
    """The site's hourly typical curve, as it was available at a given moment."""

    def __init__(self, data_dir=DATA_DIR):
        w = pd.read_csv(Path(data_dir) / "week.csv", parse_dates=["date_local", "fetched_at_utc"])
        w = w[w.facility_id == ARC_ID].sort_values("fetched_at_utc")
        w["avail"] = (w.fetched_at_utc.dt.tz_convert("America/Chicago").dt.tz_localize(None))
        first = w.drop_duplicates("date_local", keep="first").set_index("date_local")
        self.val = first["occupancy_scaled"].astype(float)
        self.avail = first["avail"]
        self.week = first["academic_week"].astype(int)

    def hourly(self, hour_ts, origin):
        """Value for the hour starting at hour_ts, using only curves fetched by `origin`."""
        for back in range(0, 4):                     # this week, else same slot 1..3 weeks back
            t = hour_ts - pd.Timedelta(days=7 * back)
            if t in self.val.index and self.avail[t] <= origin:
                return self.val[t]
        return np.nan

    def at(self, t, origin):
        """Value for the 30-min interval ending at t (h:30 -> hour h; h:00 -> mean of h-1, h)."""
        h = t.floor("h")
        if t.minute == 30:
            return self.hourly(h, origin)
        vals = [self.hourly(h - pd.Timedelta("1h"), origin), self.hourly(h, origin)]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    def academic_week(self, t):
        d = t.normalize()
        for back in range(0, 3):
            x = self.week[(self.week.index.normalize() == d - pd.Timedelta(days=7 * back))]
            if len(x):
                return int(x.iloc[0]) + back
        return np.nan


def load_weather(kind):
    w = pd.read_csv(EXT / f"weather_{kind}.csv", parse_dates=["time"]).set_index("time")
    w["precip_prev3h"] = w["precipitation"].rolling(3, min_periods=1).sum().shift(1)
    return w


# ---- samples ------------------------------------------------------------------------------
def build(data_dir=DATA_DIR, start=None, end=None):
    """Return {horizon: DataFrame}. Columns: features, y, typ_target (baseline), persistence."""
    y = load_readings(data_dir)
    cal = load_calendar()
    closed = set(cal.loc[cal.flag == "closed", "date"])
    start = pd.Timestamp(start or y.index.min().normalize())
    end = pd.Timestamp(end or y.index.max().normalize())
    grid = open_grid(start, end, closed)
    y = y.reindex(grid)                              # NaN = open but not captured
    typ = TypicalCurve(data_dir)
    wx = {"hist_forecast": load_weather("hist_forecast"), "prev_day1": load_weather("prev_day1")}
    flags = {f: set(cal.loc[cal.flag == f, "date"]) for f in CAL_FLAGS}

    # typical curve as seen at the start of each target's own day (for residuals of readings)
    typ_same_day = pd.Series([typ.at(t, t) for t in grid], index=grid)
    resid = y - typ_same_day

    out = {}
    for hname, H in HORIZONS.items():
        rows = []
        for t in grid[y.notna().values]:
            o = t - H
            day_o = o.normalize()
            yo = y.get(o, np.nan)
            # same-day observed readings up to origin (empty if origin is before opening)
            so_far = resid[(resid.index >= day_o) & (resid.index <= o)].dropna()
            if hname != "24h" and np.isnan(yo):
                continue                             # 1h/3h need a live reading at origin
            w = wx["prev_day1" if hname == "24h" else "hist_forecast"]
            wt = w.loc[t.floor("h")] if t.floor("h") in w.index else None
            r = {
                "target": t, "origin": o, "date": t.normalize(), "y": y[t],
                "hour": t.hour + t.minute / 60, "dow": t.day_of_week, "is_weekend": int(t.day_of_week >= 5),
                "academic_week": typ.academic_week(t),
                "has_origin": int(not np.isnan(yo)),
                "y_origin": yo if not np.isnan(yo) else np.nan,
                "y_lag30": y.get(o - STEP, np.nan), "y_lag60": y.get(o - 2 * STEP, np.nan),
                "resid_origin": resid.get(o, np.nan),
                "resid_day_mean": so_far.mean() if len(so_far) else 0.0,
                "typ_target": typ.at(t, o), "typ_origin": typ.at(o, o),
            }
            for v in WEATHER_VARS:
                r["w_" + v] = wt[v] if wt is not None else np.nan
            r["w_precip_prev3h"] = wt["precip_prev3h"] if wt is not None else np.nan
            for f in CAL_FLAGS:
                r[f] = int(t.normalize() in flags[f])
            rows.append(r)
        df = pd.DataFrame(rows)
        # fallbacks so every model sees complete inputs (documented in the report):
        #   no reading at origin -> assume "on the typical curve" (residual 0)
        df["resid_origin"] = df["resid_origin"].fillna(0.0)
        df["typ_origin"] = df["typ_origin"].fillna(df["typ_target"])
        df["y_origin"] = df["y_origin"].fillna(df["typ_origin"])
        df["y_lag30"] = df["y_lag30"].fillna(df["y_origin"])
        df["y_lag60"] = df["y_lag60"].fillna(df["y_lag30"])
        df["persistence"] = df["y_origin"]           # "it will be like it is now"
        out[hname] = df
    return out


if __name__ == "__main__":
    tables = build()
    for h, df in tables.items():
        print(h, len(df), "samples,", df.date.nunique(), "days, missing typ:",
              df.typ_target.isna().sum(), "| has_origin:", df.has_origin.mean().round(2))
        print("   bands:", np.bincount(band(df.y), minlength=3))
