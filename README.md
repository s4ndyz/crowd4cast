# Crowd4Cast: how busy will the ARC be?

CS 441 (Fall 2026) final project. Forecasts occupancy of the ARC (UIUC's main campus gym)
1 hour, 3 hours and 24 hours ahead, as % of capacity and as the site's own busy bands
(< 60% / 60-90% / >= 90%), and compares ML models against the "typical" curve the Campus
Recreation site already publishes.

## Repository layout

| Path | What |
|---|---|
| `collect.py` | Collector (standard library only). Polls the public Campus Rec occupancy API and appends a row only when a value changes. |
| `.github/workflows/collect.yml` | Runs the collector on GitHub Actions and commits the data to the **`data` branch**. |
| `analysis/dataset.py` | Raw data -> 30-min grid -> one sample per (target time, horizon); features use only data available at forecast time. |
| `analysis/weather.py` | Downloads hourly weather forecasts from Open-Meteo. |
| `analysis/evaluate.py` | Hyperparameter search, rolling-origin validation, feature ablation, paired significance tests, final test. |
| `analysis/plots.py` | Figures for the report. |
| `analysis/external/` | Weather CSVs and the hand-coded academic calendar. |
| `analysis/results/`, `analysis/figures/` | Output tables and figures. |

## Data

All evaluation data were collected by this project; no ML dataset is used.

- **Occupancy**: [`data` branch](https://github.com/s4ndyz/crowd4cast/tree/data), collected since 2026-10-28
  from `https://apps.campusrec.illinois.edu/FacilityInfo/api/occupancy/` (public, no login):
  - `day.csv`: the day's full 30-min series for the ARC/CRCE (the modelling target)
  - `week.csv`: the site's hourly "typical" curve, kept with the time each value was first seen
  - `building.csv`, `zones.csv`: live snapshots; `polls.csv`: one row per request (status, errors)
- **Weather**: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0): the historical-forecast API for
  1h/3h, and the previous-runs API (`*_previous_day1`, the forecast issued a day earlier) for 24h.
- **Calendar**: `analysis/external/calendar.csv`, hand-coded from the university calendar.

Known gaps: GitHub ran the 5-minute schedule only ~3-5 times a day, so until 2026-11-27 most
days were captured only up to the last poll (usually 19:00-21:00). Missing evening
intervals are treated as missing, never filled. Two closing-time runs were added on
2026-11-27. The ARC was closed for fall break 2026-11-21 to 11-29.

## Reproduce

```bash
git clone https://github.com/s4ndyz/crowd4cast.git
git clone -b data --single-branch https://github.com/s4ndyz/crowd4cast.git crowd4cast-data
pip install -r crowd4cast/analysis/requirements.txt
cd crowd4cast/analysis
python weather.py --end 2026-12-10        # optional: CSVs are committed
python evaluate.py dev                    # development folds, ~3 min
python evaluate.py test                   # final held-out test (run once)
python plots.py dev && python plots.py test
```

`dataset.py` expects the data clone at `../../crowd4cast-data` (or pass `--data-dir`).

## Collector

```bash
python collect.py                               # one poll (auto: building+zones; day+week hourly)
python collect.py --sources all
python collect.py --report --data-dir ../crowd4cast-data   # health, update cadence, curve revisions
```
