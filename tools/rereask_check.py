"""When the trio disagrees, does a narrower question resolve it?

Round 118 measured where the swarm's headroom is: on the 157 of 480 tasks
where the three lineages disagree, the correct answer is present somewhere
in the trio 91.7% of the time, while the best resolution rule - the
confidence-weighted vote - captures 74.5%. Seventeen points sit in the
disagreements, and no reweighting of the same three votes reaches them.

A combiner can only reweight votes. This measures the other move: when the
trio disagrees, ask again with only the disputed candidates on the ballot.
That is a new decision rather than a new weighting of the old one, and it is
what the playbook's verify tier would have to do in production.

Both ballots are scored on the same rows, so the comparison is paired.

Usage:
    python tools/rereask_check.py --report <ensemble-*.json> [--suite ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from combiner_check import choice_for  # noqa: E402
from ensemble_value_check import (  # noqa: E402
    DEFAULT_LINEAGES,
    load_backends,
    run_task,
)
from pandosi.calibrate import Calibrator  # noqa: E402


def plain_majority(row: dict) -> str | None:
    top = Counter(row["per_lineage"].values()).most_common()
    if not top:
        return None
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def score(rows: list[dict], rule: str) -> float | None:
    if not rows:
        return None
    if rule == "majority":
        hits = sum(1 for row in rows
                   if plain_majority(row) == row["expected"])
    elif rule.startswith("member:"):
        name = rule.split(":", 1)[1]
        hits = sum(1 for row in rows
                   if row["per_lineage"].get(name) == row["expected"])
    else:
        hits = sum(1 for row in rows
                   if choice_for(row, rule, "") == row["expected"])
    return round(hits / len(rows), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--suite", default=str(ROOT / "assets"
                                               / "big_dev.jsonl"))
    parser.add_argument("--lineages", default=DEFAULT_LINEAGES)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in
             (json.loads(line) for line
              in Path(args.suite).read_text(encoding="utf-8").splitlines()
              if line.strip())}
    splits = [row for row in report["rows"] if row["distinct"] > 1]
    recoverable = [row for row in splits
                   if row["expected"] in set(row["choices"])]
    print(f"{len(report['rows'])} rows, {len(splits)} disagreements, "
          f"{len(recoverable)} of them recoverable (answer in the trio)")

    names = [name.strip() for name in args.lineages.split(",") if name.strip()]
    backends = load_backends(names)
    calibrator = Calibrator()
    rereasks = []
    for row in splits:
        task = tasks[row["task_id"]]
        disputed = sorted(set(row["choices"]))
        rereasks.append(run_task(backends, calibrator,
                                 dict(task, candidates=disputed)))
    pairs = list(zip(splits, rereasks))
    wide = sum(len(tasks[row["task_id"]]["candidates"])
               for row in splits) / len(splits)
    narrow = sum(len(set(row["choices"])) for row in splits) / len(splits)
    print(f"  the re-ask narrowed the ballot from {wide:.1f} to "
          f"{narrow:.1f} candidates and produced a unanimous answer on "
          f"{sum(1 for row in rereasks if row['distinct'] == 1)}"
          f"/{len(rereasks)}")

    # The routing question: after the re-ask, does agreement mean anything?
    # If it does, the verify tier has a rule; if it does not, the re-ask is
    # only worth its small accuracy gain.
    settled = [pair for pair in pairs if pair[1]["distinct"] == 1]
    unsettled = [pair for pair in pairs if pair[1]["distinct"] > 1]
    breakdown = {}
    for label, subset, rule in (("settled", settled, "majority"),
                                ("unsettled", unsettled, "margin")):
        first = score([pair[0] for pair in subset], rule)
        second = score([pair[1] for pair in subset], rule)
        breakdown[label] = {"rows": len(subset), "first_ballot": first,
                            "re-ask": second}
        print(f"  {label:10s} {len(subset):3d} rows  first ballot "
              f"{first}  ->  re-ask {second}")

    rules = ["majority", "margin", "agreement", "max_conf", "min_conf"] + \
            [f"member:{name}" for name in names]
    before = {rule: score(splits, rule) for rule in rules}
    after = {rule: score(rereasks, rule) for rule in rules}
    print(f"  {'rule':22s} {'first ballot':>12s} {'re-ask':>8s} "
          f"{'change':>8s}")
    for rule in rules:
        delta = (None if before[rule] is None or after[rule] is None
                 else round(after[rule] - before[rule], 4))
        print(f"  {rule:22s} {before[rule]:12.3f} {after[rule]:8.3f} "
              f"{delta:+8.3f}")

    summary = {
        "report": str(args.report),
        "suite": Path(args.suite).name,
        "lineages": names,
        "disagreements": len(splits),
        "recoverable": len(recoverable),
        "breakdown": breakdown,
        "first_ballot": before,
        "rereask": after,
        "rows": rereasks,
    }
    out = ROOT / "metrics" / "ensemble-checks" / (
        f"rereask-{int(time.time())}.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
