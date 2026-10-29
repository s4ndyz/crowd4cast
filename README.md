# ARC / CRCE crowd data collector (CS 441 final project)

Polls the public Campus Recreation facility-counts feed and logs every change.

## Files
- `collect.py` — the collector (standard library only)
- `.github/workflows/collect.yml` — runs it every 5 min on GitHub Actions
- `data/counts.csv` — one row per change in a location's reading
- `data/polls.csv` — one row per poll attempt (use it to find gaps in collection)

## Setup (GitHub Actions)
1. Create a **public** repo (public repos get unlimited free Actions minutes; private ones would run out) and push this folder.
2. Settings → Secrets and variables → Actions → **Variables** → New repository variable:
   `COUNTS_URL` = the endpoint URL you found in the browser's Network tab.
3. Settings → Actions → General → Workflow permissions → **Read and write permissions**.
4. Actions tab → `collect` → **Run workflow** once to test. Check that `data/` appears.

Note: GitHub disables scheduled workflows after 60 days without repo activity; the
data commits count as activity, so this is not an issue during the project.

## Local use
```
COUNTS_URL="<endpoint>" python collect.py      # one poll
python collect.py --report                     # health + how often each zone really updates
python collect.py --file sample.json           # offline test
```

## Reconstructing a regular time series
`counts.csv` only stores changes. For a 5-minute grid, forward-fill each
`location_id` (pandas: `pivot` then `reindex(...).ffill()`), restricted to times when
`polls.csv` shows a successful poll.

Timestamps in `last_updated_local` are the feed's own (no timezone given; assumed
campus local time). `fetched_at_utc` is when we polled.
