#!/usr/bin/env python3
"""Poll the Campus Rec facility-counts endpoint and log it.

Two files are written under --data-dir (default: data/):

  counts.csv  one row per *change* of a location's reading (long format).
              A new row is written only when (last_updated, last_count,
              is_closed) differs from the previous row for that location.
  polls.csv   one row per poll attempt (time, ok/error, rows seen, rows new).

Together they are lossless: to know the reading at any poll time, take the
latest counts.csv row for that location at or before the poll time. polls.csv
tells you whether you were actually polling then (vs. the script not running).

Usage:
  COUNTS_URL=<endpoint> python collect.py          # normal poll
  python collect.py --file sample.json             # offline test
  python collect.py --report                       # health + update cadence

Only the standard library is used.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FACILITIES = {"ARC", "CRCE"}          # facilities to keep
USER_AGENT = "CS441-course-project (UIUC student; low-rate polling of public counts)"

COUNT_COLS = [
    "fetched_at_utc", "facility", "location_id", "location",
    "last_count", "total_capacity", "pct", "last_updated_local", "is_closed",
]
POLL_COLS = ["fetched_at_utc", "status", "n_locations", "n_new_rows", "error"]


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


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


def parse(records, fetched_at):
    """Turn raw API records into normalized rows. Skips malformed records."""
    rows = []
    for rec in records:
        try:
            facility = str(rec["FacilityName"]).strip()
            if facility not in FACILITIES:
                continue
            cap = rec.get("TotalCapacity")
            cnt = rec.get("LastCount")
            # NOTE: the API's PercetageCapacity / CountOfParticipants fields were
            # all 0 in the sample, so we compute the percentage ourselves.
            pct = round(100.0 * cnt / cap, 1) if cap else ""
            rows.append({
                "fetched_at_utc": fetched_at,
                "facility": facility,
                "location_id": int(rec["LocationId"]),
                "location": " ".join(str(rec["LocationName"]).split()),  # tidy spaces
                "last_count": cnt,
                "total_capacity": cap,
                "pct": pct,
                "last_updated_local": rec.get("LastUpdatedDateAndTime", ""),
                "is_closed": int(bool(rec.get("IsClosed"))),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def last_seen(counts_rows):
    """Latest (last_updated, last_count, is_closed) per location_id."""
    seen = {}
    for r in counts_rows:                      # file is chronological
        seen[r["location_id"]] = (r["last_updated_local"], r["last_count"], r["is_closed"])
    return seen


def poll(args):
    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    fetched_at = now_utc()
    try:
        if args.file:
            records = json.loads(Path(args.file).read_text(encoding="utf-8"))
        else:
            url = args.url or os.environ.get("COUNTS_URL", "")
            if not url:
                raise RuntimeError("No endpoint: set COUNTS_URL or pass --url")
            records = fetch(url)
        if not isinstance(records, list):
            raise ValueError("expected a JSON list, got %s" % type(records).__name__)
        rows = parse(records, fetched_at)
        if not rows:
            raise ValueError("0 usable ARC/CRCE records (format changed?)")
    except Exception as e:                      # log failure, don't crash the job
        append_rows(data / "polls.csv", POLL_COLS, [{
            "fetched_at_utc": fetched_at, "status": "error",
            "n_locations": 0, "n_new_rows": 0, "error": str(e)[:200]}])
        print("ERROR:", e, file=sys.stderr)
        return 0

    counts_path = data / "counts.csv"
    seen = last_seen(
        [{"location_id": r["location_id"], "last_updated_local": r["last_updated_local"],
          "last_count": r["last_count"], "is_closed": r["is_closed"]}
         for r in read_csv(counts_path)])
    new = []
    for r in rows:
        key = (r["last_updated_local"], str(r["last_count"]), str(r["is_closed"]))
        if seen.get(str(r["location_id"])) != key:
            new.append(r)
    append_rows(counts_path, COUNT_COLS, new)
    append_rows(data / "polls.csv", POLL_COLS, [{
        "fetched_at_utc": fetched_at, "status": "ok",
        "n_locations": len(rows), "n_new_rows": len(new), "error": ""}])
    print("ok: %d locations, %d new rows" % (len(rows), len(new)))
    return 0


def report(args):
    data = Path(args.data_dir)
    polls = read_csv(data / "polls.csv")
    counts = read_csv(data / "counts.csv")
    if not polls:
        print("No polls yet.")
        return 0
    ok = sum(1 for p in polls if p["status"] == "ok")
    print("polls: %d  ok: %d  errors: %d  first: %s  last: %s" % (
        len(polls), ok, len(polls) - ok, polls[0]["fetched_at_utc"], polls[-1]["fetched_at_utc"]))
    errs = [p for p in polls if p["status"] != "ok"][-3:]
    for p in errs:
        print("  recent error:", p["fetched_at_utc"], p["error"])

    # How often does each (open) location's reading actually change?
    by_loc = {}
    for r in counts:
        by_loc.setdefault((r["facility"], r["location"]), []).append(r)
    print("\nlocation (rows, median minutes between updates):")
    for (fac, loc), rs in sorted(by_loc.items()):
        ts = []
        for r in rs:
            try:
                ts.append(datetime.fromisoformat(r["last_updated_local"]))
            except ValueError:
                pass
        gaps = [(b - a).total_seconds() / 60 for a, b in zip(ts, ts[1:]) if b > a]
        med = "%.0f" % statistics.median(gaps) if gaps else "-"
        print("  %-5s %-34s rows=%-5d median_gap_min=%s" % (fac, loc[:34], len(rs), med))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="endpoint URL (default: $COUNTS_URL)")
    ap.add_argument("--file", help="read JSON from a local file instead (testing)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--report", action="store_true", help="print health + update cadence")
    args = ap.parse_args()
    sys.exit(report(args) if args.report else poll(args))


if __name__ == "__main__":
    main()
