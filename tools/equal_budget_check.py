#!/usr/bin/env python3
"""Equal-memory-budget control: many small models vs one big model.

Why this exists
---------------
Every result PandoSI has produced so far compares the swarm to *itself* or to
a worse swarm. The claim the project actually makes - that a committee of
small models is competitive with one larger model that fits the same memory -
has never been measured, because the fleet never held a big model and a
matched budget of small ones side by side. Without it, "the group works" and
"we happened to run several models" are the same observation.

This tool runs the smallest version of that control that can still contradict
the claim. It is deliberately blunt about what it is not:

- **It is not a capability benchmark.** The rows come from the project's own
  generation pipeline. This is internally consistent and externally unanchored
  (see docs/EXPERIMENT-equal-budget.md).
- **It does not measure "resident memory" honestly by default.** On a box that
  is already swapping, the OS decides what stays resident, so measured
  footprint and weight bytes disagree wildly. Both numbers are reported; the
  budget is stated in *weight bytes* because those are stable and auditable.
- **It does not tune either arm.** Same rows, same prompt path (each member's
  own stem /decide), same parse. No per-arm prompt edits, no split selection
  after the fact.

Decision rule, fixed before the run
-----------------------------------
Let d = (rows where arm B is right and arm A is wrong) - (rows where arm A is
right and arm B is wrong), on rows both arms answered.

- d <= 0: the "many small models win at equal budget" claim is *not supported*
  by this pilot. If arm A also has the lower p95 latency, the claim is
  weakened, and that is reported straight.
- d > 0 and |d| > 1.96*SE(d): arm B is ahead beyond noise on these rows.
- otherwise: inconclusive at this n; no claim either way.

The aggregate for arm B uses pando.common.aggregate - the same gate/combiner
the live root uses - so arm B is scored as the swarm actually decides, not as
a hand-rolled majority.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.common import Vote, aggregate  # noqa: E402


def load_suite(spec: str, limit: int | None) -> list[dict]:
    rows = []
    for name in [p for p in spec.split(",") if p.strip()]:
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / name
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_suite"] = name
                rows.append(row)
    return rows[:limit] if limit else rows


def ask(url: str, row: dict, timeout: float) -> dict:
    """One decision from one member, plus the wall time it took."""
    body = json.dumps({"state": row["state"], "question": row["question"],
                       "candidates": row["candidates"],
                       "request_id": f"budget-{row['task_id']}"}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/decide", data=body,
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            vote = json.loads(resp.read())["vote"]
        return {"ok": True, "ms": (time.monotonic() - started) * 1000.0,
                "choice": vote["choice"], "probability": float(vote["probability"]),
                "neutral": bool(vote.get("neutral")),
                "checkpoint": vote.get("checkpoint")}
    except (urllib.error.URLError, KeyError, ValueError) as exc:
        return {"ok": False, "ms": (time.monotonic() - started) * 1000.0,
                "error": f"{type(exc).__name__}: {exc}"}


def collect(urls: list[str], rows: list[dict], timeout: float,
            label: str) -> dict:
    """Ask every member every row. Sequential: this is a measurement, not a service."""
    out = {}
    for url in urls:
        votes = []
        for i, row in enumerate(rows):
            voted = ask(url, row, timeout)
            voted["expected"] = row["expected"]
            voted["task_id"] = row["task_id"]
            votes.append(voted)
            if (i + 1) % 40 == 0:
                print(f"  [{label}] {url} {i + 1}/{len(rows)}", file=sys.stderr,
                      flush=True)
        out[url] = votes
    return out


def accuracy(votes: list[dict]) -> float:
    scored = [v for v in votes if v.get("ok")]
    if not scored:
        return float("nan")
    return sum(v["choice"] == v["expected"] for v in scored) / len(scored)


def latency(votes: list[dict]) -> dict:
    ms = sorted(v["ms"] for v in votes if v.get("ok"))
    if not ms:
        return {"n": 0}
    def pct(q):
        i = min(len(ms) - 1, int(round(q * (len(ms) - 1))))
        return ms[i]
    return {"n": len(ms), "p50_ms": pct(0.5), "p95_ms": pct(0.95),
            "mean_ms": statistics.fmean(ms)}


def arm_b_decision(rows: list[dict], per_member: dict, urls: list[str],
                   use_calibration: bool) -> list[dict]:
    """Score arm B the way the live root does: aggregate() over its members."""
    from pando.calibrate import apply_temperature, load_calibration

    decisions = []
    for i, row in enumerate(rows):
        votes = []
        for url in urls:
            v = per_member[url][i]
            if not v.get("ok"):
                continue
            probability, neutral = v["probability"], v["neutral"]
            if use_calibration and not neutral:
                info = load_calibration(v.get("checkpoint") or "")
                if info.get("calibrated"):
                    probability = apply_temperature(
                        probability, float(info["temperature"]))
            votes.append(Vote(url, v.get("checkpoint") or url, v["choice"],
                              probability, neutral=neutral))
        result = aggregate(votes, min_quorum=len(votes) if len(votes) < 3 else 3)
        decisions.append({
            "task_id": row["task_id"], "expected": row["expected"],
            "choice": result.choice, "tier": result.tier,
            "agreement": result.agreement,
            "mean_probability": result.mean_probability,
            "n_votes": len(votes)})
    return decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="comma-separated suites")
    parser.add_argument("--big", required=True,
                        help="arm A: one stem url (the big model, alone)")
    parser.add_argument("--small", required=True,
                        help="arm B: comma-separated small member stem urls")
    parser.add_argument("--big-checkpoint", default="")
    parser.add_argument("--big-weight-gb", type=float, default=None)
    parser.add_argument("--small-weight-gb", type=float, default=None,
                        help="total weight bytes of arm B, as measured")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", default="")
    parser.add_argument("--votes", default="",
                        help="path for the raw per-row harvest; reused if it "
                             "already exists and its suite/row count match")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore a cached --votes file and re-harvest")
    args = parser.parse_args()

    rows = load_suite(args.suite, args.limit)
    small_urls = [u for u in args.small.split(",") if u.strip()]
    print(f"{len(rows)} rows; arm A = {args.big}; arm B = {len(small_urls)} members")

    cache = Path(args.votes) if args.votes else None
    if cache is not None and not cache.is_absolute():
        cache = ROOT / cache
    reuse = False
    if cache is not None and cache.exists() and not args.fresh:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if (saved.get("suite") == args.suite
                and saved.get("n_rows") == len(rows)
                and saved.get("arm_a_url") == args.big
                and saved.get("arm_b_urls") == small_urls):
            big, per_member = saved["arm_a"], saved["arm_b"]
            reuse = True
            print(f"reusing the raw harvest in {cache} (--fresh to redo)")
        else:
            print(f"{cache} is for a different run; re-harvesting")
    if not reuse:
        print("harvesting arm A (big)")
        big = collect([args.big], rows, args.timeout, "A")
        print("harvesting arm B (small)")
        per_member = collect(small_urls, rows, args.timeout, "B")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(
                {"suite": args.suite, "n_rows": len(rows),
                 "arm_a_url": args.big, "arm_b_urls": small_urls,
                 "arm_a": big, "arm_b": per_member}, indent=2) + "\n",
                encoding="utf-8")
            print(f"wrote raw votes to {cache}")

    big_votes = big[args.big]
    big_decisions = [{"task_id": v["task_id"], "expected": v["expected"],
                      "choice": v.get("choice"), "ok": bool(v.get("ok"))}
                     for v in big_votes]

    report = {"suite": args.suite, "n_rows": len(rows),
              "arm_a": {"stems": [args.big],
                        "checkpoint": args.big_checkpoint or None,
                        "weight_gb": args.big_weight_gb,
                        "accuracy": accuracy(big_votes),
                        "latency": latency(big_votes),
                        "per_checkpoint": {}},
              "arm_b": {"stems": small_urls, "weight_gb": args.small_weight_gb,
                        "per_member": {}, "accuracy_calibrated": None,
                        "accuracy_raw": None, "latency": {}}}

    for url in small_urls:
        report["arm_b"]["per_member"][url] = {
            "checkpoint": next((v.get("checkpoint") for v in per_member[url]
                                if v.get("ok")), None),
            "accuracy": accuracy(per_member[url]),
            "latency": latency(per_member[url]),
            "neutral": all(v.get("neutral") for v in per_member[url]
                           if v.get("ok")) if any(v.get("ok") for v in per_member[url]) else None}

    for label, flag in (("calibrated", True), ("raw", False)):
        decisions = arm_b_decision(rows, per_member, small_urls, flag)
        scored = [d for d in decisions if d["choice"] is not None]
        right = sum(d["choice"] == d["expected"] for d in scored)
        report["arm_b"][f"accuracy_{label}"] = (
            right / len(scored) if scored else None)
        report["arm_b"][f"coverage_{label}"] = len(scored) / len(rows)
        # A committee that abstains must not be scored only on the rows it
        # chose to answer: that denominator is picked by the arm being judged.
        report["arm_b"][f"accuracy_all_rows_{label}"] = right / len(rows)
        report[f"decisions_{label}"] = decisions
        if label == "calibrated":
            report["arm_b"]["latency"] = {
                "max_member_mean_ms": max(
                    (m["latency"].get("mean_ms") or 0)
                    for m in report["arm_b"]["per_member"].values()),
                "members_mean_ms": {
                    url: m["latency"].get("mean_ms")
                    for url, m in report["arm_b"]["per_member"].items()}}

    # The paired comparison: only rows both arms answered, and B is reported
    # calibrated here because that is how the swarm is actually configured.
    b_dec = {d["task_id"]: d for d in report["decisions_calibrated"]}
    a_right_b_wrong = b_right_a_wrong = both = 0
    for v in big_votes:
        if not v.get("ok"):
            continue
        d = b_dec.get(v["task_id"])
        if not d or d["choice"] is None:
            continue
        both += 1
        a_ok = v["choice"] == v["expected"]
        b_ok = d["choice"] == d["expected"]
        if a_ok and not b_ok:
            a_right_b_wrong += 1
        elif b_ok and not a_ok:
            b_right_a_wrong += 1
    n = a_right_b_wrong + b_right_a_wrong
    se = (n ** 0.5 / 2.0) if n else 0.0  # SE of the paired difference
    delta = b_right_a_wrong - a_right_b_wrong
    report["paired"] = {
        "rows_scored_by_both": both,
        "a_right_b_wrong": a_right_b_wrong,
        "b_right_a_wrong": b_right_a_wrong,
        "delta_b_minus_a": delta, "se_delta": se,
        "delta_in_se": (delta / se) if se else None}
    coverage = report["arm_b"]["coverage_calibrated"]
    # Denominator = every row B chose to answer, NOT only the rows A also
    # answered. Otherwise an arm-A abstention on a B-decided row silently
    # leaves the denominator, which would flatter A. (A answered all rows in
    # the pilot, so this is currently a no-op there; it matters in general.)
    b_decided = [v for v in big_votes
                 if (b_dec.get(v["task_id"]) or {}).get("choice") is not None]
    report["arm_a"]["rows_b_decided"] = len(b_decided)
    report["arm_a"]["accuracy_on_b_decided_rows"] = (
        (sum(1 for v in b_decided
             if v.get("ok") and v["choice"] == v["expected"])
         / len(b_decided)) if b_decided else None)
    if n == 0:
        verdict = "inconclusive: no discordant rows"
    elif delta <= 0:
        verdict = "one-big is at least as good (claim NOT supported)"
    elif abs(delta) > 1.96 * se:
        verdict = ("many-small ahead beyond noise ON THE ROWS IT ANSWERED - "
                   "see the coverage caveat below")
    else:
        verdict = f"inconclusive at n={both} ({delta}/{n} discordant rows)"
    # Coverage is not a footnote. A committee whose members disagree abstains,
    # so it picks its own denominator; restricting the comparison to that
    # denominator hands the arm the easier rows and can manufacture a win.
    caveat = (
        f"arm B answered {int(round(coverage * len(rows)))}/{len(rows)} rows "
        f"({coverage:.3f}); scored over ALL rows with an abstention counted as "
        f"no answer it is {report['arm_b']['accuracy_all_rows_calibrated']:.4f} "
        f"vs arm A's {report['arm_a']['accuracy']:.4f}. A win on the answered "
        f"subset is conditional on a selection arm A never got, so it is NOT a "
        f"win over the arm as a whole."
        if coverage < 0.9 else
        f"arm B answered {coverage:.3f} of rows; the paired comparison covers "
        f"{both}/{len(rows)} rows.")
    # Per-suite split: the two suites are different task families, and a
    # single averaged number has already hidden a sign flip in this project
    # once (round 174). Report them separately.
    def _suite(name):
        idx = [i for i, r in enumerate(rows)
               if str(r["task_id"]).startswith(name)]
        a = [big_votes[i] for i in idx if big_votes[i].get("ok")]
        b = [b_dec.get(rows[i]["task_id"]) for i in idx]
        b = [d for d in b if d and d["choice"] is not None]
        return {
            "rows": len(idx),
            "arm_a_accuracy": (sum(v["choice"] == v["expected"] for v in a)
                               / len(a)) if a else None,
            "arm_b_accuracy_on_answered": (
                sum(d["choice"] == d["expected"] for d in b) / len(b))
            if b else None,
            "arm_b_answered": len(b),
        }
    report["per_suite"] = {name: _suite(name)
                           for name in sorted({str(r["task_id"]).split("-")[0]
                                               for r in rows})}
    report["coverage_caveat"] = caveat
    report["verdict_raw_rule"] = verdict
    if n == 0:
        report["verdict"] = "inconclusive: no discordant rows"
    elif coverage < 0.9:
        report["verdict"] = (
            f"many-small NOT supported: arm B answered only "
            f"{coverage:.1%} of rows, so any edge it shows is conditional on a "
            f"selection it chose for itself")
    elif delta <= 0:
        report["verdict"] = "one-big is at least as good (claim NOT supported)"
    elif abs(delta) > 1.96 * se:
        report["verdict"] = "many-small ahead beyond noise on these rows"
    else:
        report["verdict"] = f"inconclusive at n={both} ({delta}/{n} discordant)"

    print(f"\narm A  acc={report['arm_a']['accuracy']:.4f}  "
          f"p95={report['arm_a']['latency'].get('p95_ms', 0):.0f}ms")
    for url, m in report["arm_b"]["per_member"].items():
        print(f"  member {url} acc={m['accuracy']:.4f} "
              f"neutral={m['neutral']} p95={m['latency'].get('p95_ms', 0):.0f}ms")
    print(f"arm B  acc(cal)={report['arm_b']['accuracy_calibrated']} "
          f"acc(raw)={report['arm_b']['accuracy_raw']} "
          f"coverage={report['arm_b']['coverage_calibrated']:.3f}")
    print(f"paired: A right/B wrong={a_right_b_wrong} "
          f"B right/A wrong={b_right_a_wrong} on {both} rows "
          f"(delta={delta}, 1SE={se:.2f})")
    print(f"verdict(rule as pre-registered): {verdict}")
    print(f"verdict(after coverage): {report['verdict']}")
    print(f"  {caveat}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(report)
        payload["raw_votes"] = {"arm_a": big, "arm_b": per_member}
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
