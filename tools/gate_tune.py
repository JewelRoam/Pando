"""Pick gate thresholds from recorded decisions, for a target accuracy.

Reads the rows written by tools/gating_check.py - which are calibrated
exactly as rootd calibrates a live vote - and sweeps the two gating
thresholds the playbook exposes:

    auto_min_agreement
    auto_min_probability

For each pair it reports coverage (fraction of decisions the gate would
execute) and accuracy on that covered set, then recommends the pair with
the most coverage that still meets a target accuracy.

This is offline tuning on recorded evidence, not a live run: it cannot see
decisions the gate never produced. See tools/gate_e2e_check.py for the
live confirmation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_REPORT_DIR = ROOT / "metrics" / "gating-checks"


def newest_report(directory: Path) -> Path:
    reports = sorted(directory.glob("gating-*.json"),
                     key=lambda path: path.stat().st_mtime)
    if not reports:
        raise SystemExit(f"no gating reports in {directory}; run "
                         f"tools/gating_check.py first")
    return reports[-1]


def evaluate(rows: list[dict], min_agreement: float,
             min_probability: float) -> dict:
    # A three-way split is a tie: aggregate() returns no choice, so lowering
    # the agreement threshold can never make such a decision executable. Only
    # rows with an actual majority are candidates for the gate.
    decidable = [row for row in rows if row.get("choice") is not None]
    covered = [row for row in decidable
               if row["agreement"] >= min_agreement
               and row["mean_probability"] >= min_probability]
    correct = sum(row["correct"] for row in covered)
    return {
        "min_agreement": min_agreement,
        "min_probability": min_probability,
        "coverage": round(len(covered) / len(rows), 4) if rows else 0.0,
        "accuracy": round(correct / len(covered), 4) if covered else None,
        "covered": len(covered),
        "correct": correct,
        "ties_excluded": len(rows) - len(decidable),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=None)
    parser.add_argument("--target", type=float, default=0.9,
                        help="minimum accuracy for the recommended gate")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report_path = Path(args.report) if args.report else newest_report(
        DEFAULT_REPORT_DIR)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["rows"]
    decidable = [row for row in rows if row.get("choice") is not None]
    agreements = sorted({row["agreement"] for row in decidable})
    print(f"{len(rows)} recorded decisions from {report_path.name} "
          f"({len(rows) - len(decidable)} three-way ties carry no choice); "
          f"observed majority agreements {agreements}")

    grid = []
    for min_agreement in [round(a, 4) for a in agreements] + [1.0]:
        for step in range(0, 11):
            min_probability = round(0.40 + step * 0.05, 2)
            grid.append(evaluate(rows, min_agreement, min_probability))

    viable = [entry for entry in grid
              if entry["accuracy"] is not None
              and entry["accuracy"] >= args.target]
    # Prefer the most coverage, then the most accuracy, then the smallest
    # thresholds (fewer ties broken by luck).
    viable.sort(key=lambda entry: (-entry["coverage"], -entry["accuracy"],
                                   entry["min_agreement"],
                                   entry["min_probability"]))
    recommendation = viable[0] if viable else None

    summary = {
        "report": str(report_path),
        "decisions": len(rows),
        "ties": len(rows) - len(decidable),
        "target_accuracy": args.target,
        "baseline_accuracy": report["overall_accuracy"],
        "current_defaults": evaluate(rows, 0.6, 0.6),
        "recommended": recommendation,
        "grid": [entry for entry in grid if entry["accuracy"] is not None],
    }

    if recommendation is None:
        print(f"no threshold pair reaches {args.target:.0%} on this suite")
    else:
        print(json.dumps(summary, indent=2) if args.json else
              f"baseline accuracy {summary['baseline_accuracy']:.1%}\n"
              f"current 0.60/0.60 -> coverage "
              f"{summary['current_defaults']['coverage']:.1%} at "
              f"{summary['current_defaults']['accuracy']:.1%}\n"
              f"recommended {recommendation['min_agreement']}/"
              f"{recommendation['min_probability']} -> coverage "
              f"{recommendation['coverage']:.1%} at "
              f"{recommendation['accuracy']:.1%} "
              f"({recommendation['correct']}/{recommendation['covered']})")
    out = ROOT / "metrics" / "gating-checks" / (
        f"tune-{int(report_path.stat().st_mtime)}.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
