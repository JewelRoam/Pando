"""Refit every lineage's temperature on the labels this session produced.

The calibration tables the swarm runs on were fitted on tiny samples - 14
items for `english`, 14 for `multilingual`, 12 for `typed-decisions` - because
that is all the labelled data that existed when they were written. PROJECT-
STATE has carried "rootd calibration from a held-out set" as *blocked for want
of labelled data* since round 100. There are now 720 labelled rows across
three wording splits.

No model has to be re-run to fix that. A recorded report stores calibrated
confidences, and the calibration is `sigmoid(logit(p) / T)`, which inverts
exactly: `logit(p_raw) = T * logit(p_calibrated)`. So the raw probabilities
are recoverable from the old tables, a new temperature can be fitted on 240
rows, and the effect can be measured on a split neither fit has seen.

Usage:
    python tools/refit_calibration_check.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.calibrate import apply_temperature, load_calibration  # noqa: E402
from pando.common import Vote, aggregate  # noqa: E402

TEMPERATURES = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
TIER_ORDER = {"auto": 0, "verify": 1, "escalate": 2}


def logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1 - 1e-6)
    return math.log(clipped / (1 - clipped))


def uncalibrate(calibrated: float, temperature: float) -> float:
    """Invert sigmoid(logit(p)/T) back to the raw probability."""
    return 1.0 / (1.0 + math.exp(-logit(calibrated) * temperature))


def log_loss(entries: list[tuple[float, bool]], temperature: float) -> float:
    total = 0.0
    for probability, correct in entries:
        calibrated = min(max(apply_temperature(probability, temperature),
                             1e-7), 1 - 1e-7)
        total -= math.log(calibrated if correct else 1 - calibrated)
    return total / max(1, len(entries))


def expectation(rows: list[dict], members: list[str]) -> dict:
    """Gate behaviour when every member's confidence is already calibrated."""
    auto, hits, auto_hits = 0, 0, 0
    for row in rows:
        votes = [Vote(name, name, row["per_lineage"][name],
                      row["confidences"][name]) for name in members]
        result = aggregate(votes, min_quorum=min(3, len(votes)))
        hit = result.choice == row["expected"]
        hits += hit
        if result.tier == "auto":
            auto += 1
            auto_hits += hit
    return {"full": round(hits / len(rows), 4), "auto": auto,
            "auto_accuracy": round(auto_hits / max(1, auto), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default=str(ROOT / "metrics"
                                             / "ensemble-checks"
                                             / "ensemble-1792062150.json"))
    parser.add_argument("--heldout", default=str(ROOT / "metrics"
                                                 / "ensemble-checks"
                                                 / "ensemble-1792076803.json"))
    args = parser.parse_args()

    dev = json.loads(Path(args.dev).read_text(encoding="utf-8"))
    held = json.loads(Path(args.heldout).read_text(encoding="utf-8"))
    members = sorted(dev["rows"][0]["per_lineage"])

    old = {name: load_calibration(name)["temperature"] for name in members}
    print(f"current temperatures (fitted on 12-22 items): "
          + ", ".join(f"{name} {value}" for name, value in old.items()))

    raw = {}
    for label, report in (("dev", dev), ("held", held)):
        for row in report["rows"]:
            for name in members:
                raw[(label, row["task_id"], name)] = uncalibrate(
                    row["confidences"][name], old[name])

    fitted = {}
    for name in members:
        entries = [(raw[("dev", row["task_id"], name)],
                    row["per_lineage"][name] == row["expected"])
                   for row in dev["rows"]]
        scored = [(log_loss(entries, temperature), temperature)
                  for temperature in TEMPERATURES]
        best_loss, temperature = min(scored)
        fitted[name] = temperature
        print(f"  {name:18s} 240-row refit: T={temperature:<5} "
              f"log-loss {log_loss(entries, 1.0):.4f} -> {best_loss:.4f} "
              f"(was T={old[name]}: {log_loss(entries, old[name]):.4f})")

    def with_temperatures(rows: list[dict], table: dict) -> list[dict]:
        out = []
        for row in rows:
            out.append({
                "expected": row["expected"],
                "per_lineage": dict(row["per_lineage"]),
                "confidences": {
                    name: round(apply_temperature(
                        raw[("held", row["task_id"], name)], table[name]), 4)
                    for name in members},
            })
        return out

    print(f"\n  on the third wording split (240 rows, unseen):")
    before = expectation(with_temperatures(held["rows"], old), members)
    after = expectation(with_temperatures(held["rows"], fitted), members)
    print(f"    shipped tables   full {before['full']:.4f}  "
          f"auto {before['auto']} at {before['auto_accuracy']:.4f}")
    print(f"    240-row refit    full {after['full']:.4f}  "
          f"auto {after['auto']} at {after['auto_accuracy']:.4f}")

    # The honest test of a calibration: is it better on rows it was not fitted
    # on? If it is not, the refit has learned dev's confidence scale rather
    # than the model's.
    print(f"\n  calibration quality (log-loss, lower is better):")
    losses = {}
    for label, table in (("shipped", old), ("refit", fitted)):
        total, count = 0.0, 0
        for row in held["rows"]:
            for name in members:
                probability = apply_temperature(
                    raw[("held", row["task_id"], name)], table[name])
                correct = row["per_lineage"][name] == row["expected"]
                probability = min(max(probability, 1e-7), 1 - 1e-7)
                total -= math.log(probability if correct else 1 - probability)
                count += 1
        losses[label] = round(total / count, 4)
        print(f"    {label:8s} {losses[label]}")

    # The ceiling: what a temperature fitted *on the deployment's own wording*
    # would buy. It needs labels from that distribution, which is the honest
    # requirement the shipped table cannot substitute for.
    in_place = {}
    for name in members:
        entries = [(raw[("held", row["task_id"], name)],
                    row["per_lineage"][name] == row["expected"])
                   for row in held["rows"]]
        in_place[name] = min(TEMPERATURES,
                             key=lambda temperature: log_loss(entries,
                                                              temperature))
    total, count = 0.0, 0
    for row in held["rows"]:
        for name in members:
            probability = apply_temperature(raw[("held", row["task_id"], name)],
                                            in_place[name])
            correct = row["per_lineage"][name] == row["expected"]
            probability = min(max(probability, 1e-7), 1 - 1e-7)
            total -= math.log(probability if correct else 1 - probability)
            count += 1
    print(f"    fitted on the deployment's own wording: {round(total / count, 4)}"
          f"  (temperatures {in_place})")
    losses["in_place"] = round(total / count, 4)
    losses["in_place_temperatures"] = in_place
    summary = {"old": old, "refit": fitted, "before": before, "after": after,
               "heldout_log_loss": losses}

    out = ROOT / "metrics" / "ensemble-checks" / "calibration-refit.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
