"""Tuning the gate on a deployment's own rows, with a learning curve.

Round 148 found that calibration does not transfer across wording
distributions (a 240-row refit was better on dev and slightly worse on an
unseen split) but that fitting on the deployment's own rows would be worth 9%
of log-loss - and that a temperature change is a *threshold* change, because
sharper confidences satisfy `auto_min_probability` more often and the shipped
thresholds are tuned to the shipped tables' conservatism.

So recalibration and gate tuning are one procedure, and this measures it: the
rows are split into a tuning half and an evaluation half, the temperatures and
the two gate thresholds are chosen on the tuning half alone, and the resulting
coverage and accuracy are reported on the evaluation half - against the
shipped tables and thresholds doing the same job.

The point of the curve is the practical question a deployment asks: how many
of its own labelled decisions does it need before tuning beats shipping?

Usage:
    python tools/tune_gate_check.py [--report <ensemble-*.json>]
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

TEMPERATURES = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
AGREEMENTS = [0.5, 0.6, 0.67, 0.75]
PROBABILITIES = [0.35, 0.45, 0.5, 0.6, 0.7, 0.8]
TUNING_SIZES = [0, 25, 50, 100]


def logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1 - 1e-6)
    return math.log(clipped / (1 - clipped))


def uncalibrate(calibrated: float, temperature: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit(calibrated) * temperature))


def fit_temperature(entries: list[tuple[float, bool]]) -> float:
    def loss(temperature: float) -> float:
        total = 0.0
        for probability, correct in entries:
            value = min(max(apply_temperature(probability, temperature),
                            1e-7), 1 - 1e-7)
            total -= math.log(value if correct else 1 - value)
        return total / max(1, len(entries))
    return min(TEMPERATURES, key=loss)


def judge(rows: list[dict], members: list[str], table: dict,
          agreement: float, probability: float) -> tuple[int, int]:
    auto, hits = 0, 0
    for row in rows:
        votes = [Vote(name, name, row["per_lineage"][name],
                      apply_temperature(row["raw"][name], table[name]))
                 for name in members]
        result = aggregate(votes, min_quorum=min(3, len(votes)),
                           auto_min_agreement=agreement,
                           auto_min_probability=probability)
        if result.tier == "auto":
            auto += 1
            hits += result.choice == row["expected"]
    return auto, hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(ROOT / "metrics"
                                                / "ensemble-checks"
                                                / "ensemble-1792076803.json"))
    parser.add_argument("--accuracy-floor", type=float, default=0.9)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = report["rows"]
    members = sorted(rows[0]["per_lineage"])
    shipped = {name: load_calibration(name)["temperature"] for name in members}
    for row in rows:
        row["raw"] = {name: uncalibrate(row["confidences"][name],
                                        shipped[name]) for name in members}

    half = len(rows) // 2
    tuning_pool, evaluation = rows[:half], rows[half:]
    shipped_auto, shipped_hits = judge(evaluation, members, shipped, 0.6, 0.6)
    print(f"{len(rows)} rows: {len(tuning_pool)} to tune on, "
          f"{len(evaluation)} to evaluate")
    print(f"  shipped tables+thresholds: auto {shipped_auto} at "
          f"{shipped_hits / max(1, shipped_auto):.4f} "
          f"({shipped_hits}/{shipped_auto})")
    print(f"  {'own rows':>9s} {'T':>28s} {'agr':>5s} {'prob':>5s} "
          f"{'auto':>5s} {'accuracy':>9s}")

    curve = {}
    for size in TUNING_SIZES:
        subset = tuning_pool[:size] if size else []
        if subset:
            table = {}
            for name in members:
                entries = [(row["raw"][name],
                            row["per_lineage"][name] == row["expected"])
                           for row in subset]
                table[name] = fit_temperature(entries)
            best = None
            for agreement in AGREEMENTS:
                for probability in PROBABILITIES:
                    auto, hits = judge(subset, members, table, agreement,
                                       probability)
                    if not auto:
                        continue
                    if hits / auto < args.accuracy_floor:
                        continue
                    score = (auto, hits / auto)
                    if best is None or score > best[0]:
                        best = (score, agreement, probability)
            if best is None:
                print(f"  {size:>9d} no threshold pair met the floor on the "
                      f"tuning half")
                curve[size] = None
                continue
            (_, _), agreement, probability = best
        else:
            table, agreement, probability = shipped, 0.6, 0.6
        auto, hits = judge(evaluation, members, table, agreement, probability)
        accuracy = hits / max(1, auto)
        curve[size] = {"temperatures": table, "agreement": agreement,
                       "probability": probability, "auto": auto,
                       "accuracy": round(accuracy, 4)}
        print(f"  {size:>9d} "
              f"{json.dumps({k: v for k, v in table.items()})[:28]:>28s} "
              f"{agreement:>5} {probability:>5} {auto:>5} {accuracy:>9.4f}")

    print(f"\n  on the evaluation half, at an accuracy floor of "
          f"{args.accuracy_floor:.0%}:")
    print(f"    shipped:   {shipped_auto} rows at "
          f"{shipped_hits / max(1, shipped_auto):.4f}")
    for size, entry in curve.items():
        if entry and entry["accuracy"] >= args.accuracy_floor:
            print(f"    tuned on {size:3d} of its own rows: {entry['auto']} "
                  f"rows at {entry['accuracy']:.4f}")

    # The upper bound: the best threshold pair *chosen on the evaluation half
    # itself*. If the shipped defaults are close to this, no tuning procedure
    # could do much better and the negative result is about the system, not
    # about the tuning being naive.
    best_in_sample = None
    for agreement in AGREEMENTS:
        for probability in PROBABILITIES:
            auto, hits = judge(evaluation, members, shipped, agreement,
                               probability)
            if not auto or hits / auto < args.accuracy_floor:
                continue
            score = (auto, hits / auto)
            if best_in_sample is None or score > best_in_sample[0]:
                best_in_sample = (score, agreement, probability)
    if best_in_sample:
        (auto, accuracy), agreement, probability = best_in_sample
        print(f"    in-sample ceiling (thresholds chosen on the evaluation "
              f"half): {auto} rows at {accuracy:.4f} "
              f"(agreement {agreement}, probability {probability})")
        curve["in_sample_ceiling"] = {"auto": auto,
                                      "accuracy": round(accuracy, 4),
                                      "agreement": agreement,
                                      "probability": probability}

    out = ROOT / "metrics" / "ensemble-checks" / "gate-tuning.json"
    out.write_text(json.dumps(curve, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
