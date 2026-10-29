#!/usr/bin/env python3
"""Poll the Campus Rec occupancy API (ARC / CRCE) and log every change.

Sources (public, no login; the same calls the facility page makes):
  building  occupancy/building      latest whole-building % per 30-min interval
  zones     occupancy/sensors       per-zone % (Gym 1, Gym 4 = Strength & Conditioning)
  day       occupancy/day/<id>      today's full 30-min series so far (backfills missed polls)
  week      occupancy/week/<id>     the site's hourly "typical" curve (occupancyScaled), kept
                                    so we can check whether it changes (= baseline vs real)

Files under --data-dir (default data/): building.csv, zones.csv, day.csv, week.csv, polls.csv.
A row is appended only when something changed; polls.csv logs every attempt, so the
two together are lossless (forward-fill to get a regular grid).

Usage:
  python collect.py                         # auto: building+zones always; day+week if >=55 min since last
  python collect.py --sources all           # everything now
  python collect.py --sources day,week
  python collect.py --report                # health + real update cadence
  python collect.py --sources day --file day:ARC=a.json --file day:CRCE=c.json   # offline test

Standard library only.
"""
import argparse
import csv
import json
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://apps.campusrec.illinois.edu/FacilityInfo/api/occupancy/"
ARC_ID = "02e5e50d-71c6-4122-aa70-7c9d4e61c27a"
CRCE_ID = "3b4297b0-703a-44bb-a02a-20fcb4bc5c78"
# Zone ids from the sensors feed. Untested with day/ (try in the browser first, then uncomment):
#   "ARC Gym 4 (S&C)": "3a9daeeb-94ec-4b4f-acf0-ce182291e189",
#   "ARC Gym 1":       "f7f41ee5-29c8-4409-86d3-5b24a3ea2a72",

# source -> list of (label, facility_id, url)
TARGETS = {
    "building": [(None, None, BASE + "building")],
    "zones": [(None, None, BASE + "sensors")],
    "day": [("ARC", ARC_ID, BASE + "day/" + ARC_ID),
            ("CRCE", CRCE_ID, BASE + "day/" + CRCE_ID)],
    "week": [("ARC", ARC_ID, BASE + "week/" + ARC_ID)],
}
HOURLY_SOURCES = ("day", "week")      # in auto mode, polled at most once per ~hour
USER_AGENT = "CS441-course-project (UIUC student; low-rate polling of public occupancy data)"
POLL_COLS = ["fetched_at_utc", "source", "status", "n_records", "n_new_rows", "error"]


def s(x):
    """Stringify consistently so new rows compare equal to rows read back from CSV."""
    return "" if x is None else str(x)


def pct(x):
    try:
        return str(float(x))
    except (TypeError, ValueError):
        return ""


# ---- per-source parsers: (raw record, facility_id, label) -> flat dict of strings -----
def parse_building(r, fid, label):
    return {
        "facility": s(r["facilityName"]).strip(),
        "interval_end_local": s(r["dateIntervalEnd"]),
        "interval_end_offset": s(r.get("dateIntervalEndOffset")),
        "capacity_pct": pct(r.get("capacityPct")),
        "building_is_open": s(int(bool(r.get("buildingIsOpen")))),
        "record_id": s(r.get("id")),
    }


def parse_zone(r, fid, label):
    return {
        "facility_id": s(r["facilityId"]),
        "name": " ".join(s(r["name"]).split()),
        "description": " ".join(s(r.get("description")).split()),
        "last_updated_local": s(r["lastUpdated"]),
        "capacity_pct": pct(r.get("capacityPct")),
    }


def parse_day(r, fid, label):
    # "count" at 21:30 equalled the building feed's capacityPct (79), so it looks like the same %.
    return {
        "facility": label, "facility_id": fid,
        "interval_end_local": s(r["dateIntervalEnd"]),
        "count": s(r["count"]),
    }


def parse_week(r, fid, label):
    return {
        "facility": label, "facility_id": fid,
        "academic_week": s(r.get("academicWeek")),
        "day_of_week": s(r.get("dayOfWeek")),
        "date_local": s(r["date"]),
        "occupancy_scaled": s(r["occupancyScaled"]),
    }


# group = which series a row belongs to; key = fields whose change means "new row"
SPECS = {
    "building": dict(
        file="building.csv", parse=parse_building,
        cols=["fetched_at_utc", "facility", "interval_end_local", "interval_end_offset",
              "capacity_pct", "building_is_open", "record_id"],
        group=["facility"], key=["interval_end_local", "capacity_pct", "building_is_open"]),
    "zones": dict(
        file="zones.csv", parse=parse_zone,
        cols=["fetched_at_utc", "facility_id", "name", "description",
              "last_updated_local", "capacity_pct"],
        group=["facility_id"], key=["last_updated_local", "capacity_pct"]),
    "day": dict(
        file="day.csv", parse=parse_day,
        cols=["fetched_at_utc", "facility", "facility_id", "interval_end_local", "count"],
        group=["facility_id", "interval_end_local"], key=["count"]),
    "week": dict(
        file="week.csv", parse=parse_week,
        cols=["fetched_at_utc", "facility", "facility_id", "academic_week", "day_of_week",
              "date_local", "occupancy_scaled"],
        group=["facility_id", "date_local"], key=["occupancy_scaled", "academic_week"]),
}


# ---- helpers ----------------------------------------------------------------------------
def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(url, timeout=20, retries=1):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if attempt == retries:
                raise
            time.sleep(3)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(path, cols, rows):
    if not rows:
        return
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        w.writerows(rows)


def tup(row, fields):
    return tuple(row.get(f, "") for f in fields)


def run_target(name, label, fid, url, data, fetched_at, files):
    """Poll one target of one source. Logs to polls.csv; never raises."""
    spec = SPECS[name]
    tag = name + (":" + label if label else "")
    poll = {"fetched_at_utc": fetched_at, "source": tag, "status": "ok",
            "n_records": 0, "n_new_rows": 0, "error": ""}
    try:
        local = files.get(tag) or files.get(name)
        raw = json.loads(Path(local).read_text(encoding="utf-8")) if local else fetch(url)
        if not isinstance(raw, list):
            raise ValueError("expected a JSON list, got %s" % type(raw).__name__)
        rows = []
        for rec in raw:
            try:
                row = spec["parse"](rec, fid, label)
                row["fetched_at_utc"] = fetched_at
                rows.append(row)
            except (KeyError, TypeError, ValueError, AttributeError):
                continue                      # skip malformed records
        if not rows:
            raise ValueError("0 usable records (format changed?)")
        path = data / spec["file"]
        last = {tup(r, spec["group"]): tup(r, spec["key"]) for r in read_csv(path)}
        new = [r for r in rows if last.get(tup(r, spec["group"])) != tup(r, spec["key"])]
        append_rows(path, spec["cols"], new)
        poll["n_records"], poll["n_new_rows"] = len(rows), len(new)
    except Exception as e:
        poll["status"], poll["error"] = "error", str(e)[:200]
        print("ERROR [%s]: %s" % (tag, e), file=sys.stderr)
    append_rows(data / "polls.csv", POLL_COLS, [poll])
    print("%s: %s, %d records, %d new" % (tag, poll["status"], poll["n_records"], poll["n_new_rows"]))


def due(src, data, min_gap_minutes=55):
    """True if `src` has not had a successful poll in the last ~hour."""
    last = None
    for p in read_csv(data / "polls.csv"):
        if p["status"] == "ok" and p["source"].split(":")[0] == src:
            last = p["fetched_at_utc"]
    if not last:
        return True
    t = datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds() >= min_gap_minutes * 60


def pick_sources(arg, data):
    if arg == "auto":
        return ["building", "zones"] + [x for x in HOURLY_SOURCES if due(x, data)]
    if arg == "all":
        return list(TARGETS)
    return [x for x in arg.split(",") if x in TARGETS]


def poll_all(args):
    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    files = dict(f.split("=", 1) for f in (args.file or []) if "=" in f)
    fetched_at = now_utc()
    for name in pick_sources(args.sources, data):
        for label, fid, url in TARGETS[name]:
            run_target(name, label, fid, url, data, fetched_at, files)
    return 0


def median_gap_minutes(times):
    ts = []
    for t in times:
        try:
            ts.append(datetime.fromisoformat(t))
        except ValueError:
            pass
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(ts, ts[1:]) if b > a]
    return "%.0f" % statistics.median(gaps) if gaps else "-"


def report(args):
    data = Path(args.data_dir)
    polls = read_csv(data / "polls.csv")
    if not polls:
        print("No polls yet.")
        return 0
    by_src = {}
    for p in polls:
        by_src.setdefault(p["source"], []).append(p)
    print("poll health (per source):")
    for src, ps in sorted(by_src.items()):
        ok = sum(p["status"] == "ok" for p in ps)
        print("  %-14s polls=%-5d ok=%-5d errors=%-4d %s -> %s" % (
            src, len(ps), ok, len(ps) - ok, ps[0]["fetched_at_utc"], ps[-1]["fetched_at_utc"]))
        for p in [p for p in ps if p["status"] != "ok"][-2:]:
            print("      recent error:", p["fetched_at_utc"], p["error"])

    print("\nbuilding: rows and median minutes between new intervals")
    by = {}
    for r in read_csv(data / "building.csv"):
        by.setdefault(r["facility"], []).append(r["interval_end_local"])
    for fac, t in sorted(by.items()):
        print("  %-6s rows=%-5d median_gap_min=%s" % (fac, len(t), median_gap_minutes(t)))

    print("\nzones: rows and median minutes between updates")
    by = {}
    for r in read_csv(data / "zones.csv"):
        by.setdefault(r["name"] + " | " + r["description"], []).append(r["last_updated_local"])
    for k, t in sorted(by.items()):
        print("  %-48s rows=%-5d median_gap_min=%s" % (k[:48], len(t), median_gap_minutes(t)))

    print("\nday series: rows, distinct days, range")
    by = {}
    for r in read_csv(data / "day.csv"):
        by.setdefault(r["facility"], []).append(r["interval_end_local"])
    for fac, t in sorted(by.items()):
        days = {x[:10] for x in t}
        print("  %-6s rows=%-6d days=%-3d %s -> %s" % (fac, len(t), len(days), min(t), max(t)))

    wk = read_csv(data / "week.csv")
    if wk:
        # same (facility, date) appearing with different values = the curve is revised over time
        seen = {}
        for r in wk:
            seen.setdefault((r["facility_id"], r["date_local"]), set()).add(r["occupancy_scaled"])
        revised = sum(len(v) > 1 for v in seen.values())
        print("\nweek curve: %d (date,hour) slots seen, %d revised after first seen" % (len(seen), revised))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="auto",
                    help="auto | all | comma list of building,zones,day,week (default: auto)")
    ap.add_argument("--file", action="append",
                    help="offline test: SRC=path or SRC:LABEL=path (repeatable)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--report", action="store_true", help="print health + update cadence")
    args = ap.parse_args()
    sys.exit(report(args) if args.report else poll_all(args))


if __name__ == "__main__":
    main()
