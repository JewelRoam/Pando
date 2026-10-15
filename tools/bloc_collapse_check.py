"""Does collapsing a bloc recover what a correlated family costs the swarm?

Round 128 measured the cliff and the mechanism: four cheap members, each
decorrelated from the neural trio but correlated with each other (phi up to
+0.898), override a unanimous trio on 37 rows and are wrong on 32 of them,
leaving the swarm 11.6 points worse than the trio alone. Two reweightings of
the choice rule did not help, because the problem is not how the votes are
weighed - it is that a near-copy gets a vote of its own at all.

Round 121 answered that for replicas by collapsing votes that share a
checkpoint name. This measures the generalisation: collapse the group, and
see whether the swarm keeps the family's information without letting the
family outvote anyone.

Usage:
    python tools/bloc_collapse_check.py --report <ensemble-*.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from member_portfolio_check import (  # noqa: E402
    TIER_ORDER,
    build_members,
    build_rows,
    paired_delta,
)
from pandosi.common import Vote, aggregate  # noqa: E402


def judge(rows: list[dict], members: list[str],
          blocs: dict | None) -> dict:
    hits, auto_hits, ranked = [], [], []
    answered = auto = 0
    for row in rows:
        votes = [Vote(name, name, row["per_lineage"][name],
                      row["confidences"][name]) for name in members]
        result = aggregate(votes, min_quorum=min(3, len(
            {name for name in members} if blocs is None else
            {(blocs or {}).get(name, name) for name in members})),
            blocs=blocs)
        hit = result.choice == row["expected"]
        hits.append(hit)
        answered += result.choice is not None
        if result.tier == "auto":
            auto += 1
            auto_hits.append(hit)
        ranked.append((TIER_ORDER[result.tier], -result.mean_probability, hit))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return {
        "full": round(sum(hits) / len(rows), 4),
        "answered": answered,
        "auto": auto,
        "accuracy_where_auto": (round(sum(auto_hits) / auto, 4) if auto
                                else None),
        "hits": hits,
        "ranked_hits": ranked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train", default=str(ROOT / "assets"
                                              / "big_dev.jsonl"))
    parser.add_argument("--test", default=str(ROOT / "assets"
                                              / "heldout_test.jsonl"))
    parser.add_argument("--holdout", type=int, default=48)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    def load(path):
        return [json.loads(line) for line
                in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()]

    train = load(args.train)
    tasks = {task["task_id"]: task for task in load(args.test)}
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    trio = sorted(report["rows"][0]["per_lineage"])
    members = build_members(train[:-args.holdout], train[-args.holdout:])
    rows = build_rows(report, tasks, members)

    cheap = sorted(members)
    # The operator declares the family; the swarm cannot estimate correlation
    # offline. Every cheap member maps to one bloc id.
    bloc_map = {name: cheap[0] for name in cheap}
    scenarios = [
        ("trio alone", trio, None),
        ("trio + centroid only", trio + ["centroid"], None),
        (f"trio + all {len(cheap)} (names)",
         trio + cheap, None),
        (f"trio + all {len(cheap)} (one bloc)",
         trio + cheap, bloc_map),
    ]
    base = judge(rows, trio, None)
    print(f"{len(rows)} held-out rows; trio baseline {base['full']:.4f} "
          f"(auto {base['auto']} at {base['accuracy_where_auto']:.4f})")
    print(f"  {'scenario':40s} {'full':>7s} {'auto':>5s} {'auto acc':>9s} "
          f"{'vs trio':>18s} {'better':>7s}")
    summary = {"report": str(args.report), "cheap": cheap,
               "baseline": {k: v for k, v in base.items()
                            if k not in ("hits", "ranked_hits")}}
    entries = {}
    for label, names, blocs in scenarios:
        entry = judge(rows, names, blocs)
        entries[label] = entry
        delta = ({"mean": 0.0, "low": 0.0, "high": 0.0,
                  "share_positive": 0.0} if names == trio else
                 paired_delta(entry["hits"], base["hits"], args.trials,
                              args.seed))
        summary[label] = {k: v for k, v in entry.items()
                          if k not in ("hits", "ranked_hits")}
        summary[label]["delta_vs_trio"] = delta
        print(f"  {label:40s} {entry['full']:7.4f} {entry['auto']:5d} "
              f"{entry['accuracy_where_auto']:9.4f} "
              f"[{delta['low']:+.4f},{delta['high']:+.4f}] "
              f"{delta['share_positive']:6.1%}")

    # How a bloc should speak is a design question: a mass-weighted consensus
    # of near-copies is not obviously better than delegating to the member
    # with the best probe accuracy.
    best_member = "trio + centroid only"
    collapsed = f"trio + all {len(cheap)} (one bloc)"
    delta = paired_delta(entries[collapsed]["hits"],
                         entries[best_member]["hits"], args.trials, args.seed)
    print(f"\n  the bloc's consensus against its best member "
          f"(+centroid only): {delta['mean']:+.4f} "
          f"[{delta['low']:+.4f},{delta['high']:+.4f}], "
          f"ahead in {delta['share_positive']:.1%} of resamples")
    summary["bloc_vs_best_member"] = delta

    out = ROOT / "metrics" / "ensemble-checks" / "bloc-collapse.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
