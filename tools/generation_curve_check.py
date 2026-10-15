"""The training loop's own curve, with the spread that decides whether it is one.

Round 150 showed the project can build the member it was missing: one
fine-tune of `english` on 720 of its own labels scored 0.9125 on an unseen
wording split against its parent's 0.7333, and took the fleet from a 0.7708
gate to 0.8833 (round 151: 0.9083 decided, with the advisory combiner).

That was a single step. This measures the step after it - the same recipe on
960 rows, the 720 plus the third split - on a fourth split that none of them
has seen, and it repeats each recipe with different seeds, because the first
two runs of the 960-row recipe landed thirteen points apart.

Usage:
    python tools/generation_curve_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summarise(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"samples": len(values), "mean": round(sum(values) / len(values), 4),
            "low": ordered[0], "high": ordered[-1],
            "spread": round(ordered[-1] - ordered[0], 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=",".join(
        str(ROOT / "artifacts" / name / "rlcd-cuda-report.json")
        for name in ("rlcd-fi-v4", "rlcd-fi-v4-s1", "rlcd-fi-v3",
                     "rlcd-fi-v3-s1", "rlcd-fi-v3-s2")))
    args = parser.parse_args()

    by_corpus: dict[int, list[float]] = {}
    parent = None
    runs = {}
    for name in (item.strip() for item in args.runs.split(",") if item.strip()):
        report = json.loads(Path(name).read_text(encoding="utf-8"))
        trained_on = report["training"]["train_items"] \
            if report.get("training") else 0
        accuracy = report["after"]["accuracy"]
        parent = report["before"]["accuracy"] if parent is None else parent
        runs[Path(name).parent.name] = {
            "trained_on": trained_on, "accuracy_on_D": accuracy,
            "decisions_changed": report["comparison"]["tasks"]
            - report["comparison"]["matches"]}
        by_corpus.setdefault(trained_on, []).append(accuracy)
        print(f"  {Path(name).parent.name:16s} corpus {trained_on:4d}  "
              f"accuracy on D {accuracy:.4f}  "
              f"changed {runs[Path(name).parent.name]['decisions_changed']}"
              f"/240")
    print(f"\n  parent (0 rows trained): {parent}")
    curve = {"parent": parent, "runs": runs}
    for corpus in sorted(by_corpus):
        entry = summarise(by_corpus[corpus])
        curve[f"corpus_{corpus}"] = entry
        print(f"  corpus {corpus:4d}: {entry['samples']} sample(s), mean "
              f"{entry['mean']:.4f}, range {entry['low']:.4f}-{entry['high']:.4f}"
              f" (spread {entry['spread']:.4f})")

    out = ROOT / "metrics" / "ensemble-checks" / "generation-curve.json"
    out.write_text(json.dumps(curve, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
