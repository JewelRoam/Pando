"""Does calibrating the new member lift the system?

Round 153 measured the tension: `ft-v3` decides 0.9750 alone on split D, yet
the system with it seated decides 0.8792 - and every live row it votes on
carries `uncalibrated_checkpoints: [ft-v3]`. Its confidences are saturated
(228 of 240 rows at exactly 1.0 on split C), so the gate's probability
channel is inert for the strongest member the fleet has.

This fits one temperature for `ft-v3` from its recorded votes on split C -
a split it never trained on and that is not the evaluator - then applies it
exactly where the live root applies calibration: to every vote before
aggregation, and to the combiner's inputs. Split D stays the evaluator.

Configurations measured on D (duplicate `english` entries mirror the live
CPU-free fleet, whose iGPU and NPU both run the english lineage):

  trio                     the seated english/multilingual/typed-decisions
  trio + ft-v3             round 153's fleet, raw vs fully calibrated
  english x2 + ft-v3       the live fleet's membership: "live today" is the
                           path as shipped (english T=3.2, ft-v3 raw) and
                           reproduces round 153's live 0.8292; "calibrated"
                           adds the fitted ft-v3 temperature.

Usage:
    python tools/calibrated_gate_check.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from improved_fleet_check import decide  # noqa: E402
from pandosi.calibrate import apply_temperature, load_calibration  # noqa: E402
from pandosi.common import Vote, aggregate, collapse_lineages  # noqa: E402
from pandosi.combiner import Combiner  # noqa: E402

TRIO = ["english", "multilingual", "typed-decisions"]
NEW = "ft-v3"


def ece(rows: list[dict], temperature: float, bins: int = 5) -> float:
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for row in rows:
        p = min(max(apply_temperature(row["probability"], temperature),
                    1e-7), 1.0 - 1e-7)
        buckets[min(int(p * bins), bins - 1)].append(
            (p, int(row["choice"] == row["expected"])))
    total = sum(len(bucket) for bucket in buckets) or 1
    error = 0.0
    for bucket in buckets:
        if bucket:
            mean_p = sum(p for p, _ in bucket) / len(bucket)
            accuracy = sum(ok for _, ok in bucket) / len(bucket)
            error += len(bucket) / total * abs(mean_p - accuracy)
    return error


def fit_temperature(rows: list[dict]) -> dict:
    """The NLL-optimal temperature for a checkpoint's recorded votes."""
    def nll(temperature: float) -> float:
        total = 0.0
        for row in rows:
            p = min(max(apply_temperature(row["probability"], temperature),
                        1e-7), 1.0 - 1e-7)
            total += -math.log(p if row["choice"] == row["expected"]
                               else 1.0 - p)
        return total

    best = {"temperature": 1.0, "nll": nll(1.0)}
    checks = int((10.0 - 0.2) / 0.01) + 1
    for index in range(checks):
        temperature = round(0.2 + index * 0.01, 2)
        value = nll(temperature)
        if value < best["nll"]:
            best = {"temperature": temperature, "nll": value}
    best["nll"] = round(best["nll"], 3)
    best["nll_at_1.0"] = round(nll(1.0), 3)
    best["ece"] = round(ece(rows, best["temperature"]), 4)
    best["ece_at_1.0"] = round(ece(rows, 1.0), 4)
    return best


def transform(rows: list[dict], temperatures: dict[str, float]) -> list[dict]:
    """Apply temperature scaling per member, as the live root does."""
    out = []
    for row in rows:
        confidences = {name: apply_temperature(
            p, temperatures.get(name, 1.0))
            for name, p in row["confidences"].items()}
        out.append({**row, "confidences": confidences})
    return out


def decide_lineage(rows: list[dict], stem_members: list[str],
                   temperatures: dict[str, float],
                   combiner: Combiner | None,
                   combiner_sees_opinions: bool) -> dict:
    """Mirror rootd's lineage-counted path exactly.

    The quorum counts opinions (same-lineage stems collapse to one vote),
    but rootd hands the *combiner* either everything it heard (today's
    behaviour, replicated lineages included) or the collapsed opinion set -
    the difference this function exists to measure.
    """
    auto, auto_hits, decided = 0, 0, 0
    for row in transform(rows, temperatures):
        votes = [Vote(name, name, row["choices"][name],
                      row["confidences"][name]) for name in stem_members]
        opinions = collapse_lineages(votes, None)
        result = aggregate(opinions, min_quorum=min(3, len(opinions)))
        hit = result.choice == row["expected"]
        answer = result.choice if result.tier == "auto" else None
        if result.tier == "auto":
            auto += 1
            auto_hits += hit
        if answer is None and combiner is not None:
            advice = opinions if combiner_sees_opinions else votes
            suggestion = combiner.suggest(
                [(vote.checkpoint, vote.choice, vote.probability)
                 for vote in advice], row["candidates"],
                state=row["state"], question=row["question"])
            if suggestion:
                answer = suggestion["choice"]
        decided += answer == row["expected"]
    return {"auto": auto,
            "auto_accuracy": round(auto_hits / max(1, auto), 4),
            "decided": round(decided / len(rows), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-votes", default=str(
        ROOT / "metrics" / "ensemble-checks" / "finetuned-votes.json"),
        help="ft-v3's recorded votes on split C (the fit split)")
    parser.add_argument("--suite", default=str(
        ROOT / "assets" / "heldout3_test.jsonl"))
    parser.add_argument("--votes-dir", default=str(
        ROOT / "metrics" / "ensemble-checks"))
    parser.add_argument("--combiner", default=str(
        ROOT / "assets" / "combiner.json"))
    parser.add_argument("--cross-report", default=str(
        ROOT / "metrics" / "ensemble-checks" / "ensemble-1792076803.json"),
        help="trio votes on split C, for a second membership check")
    parser.add_argument("--cross-suite", default=str(
        ROOT / "assets" / "heldout2_test.jsonl"))
    args = parser.parse_args()

    fit_rows = json.loads(Path(args.fit_votes).read_text(
        encoding="utf-8"))["rows"]
    fitted = fit_temperature(fit_rows)
    saturated = sum(1 for row in fit_rows if row["probability"] >= 0.999)
    print(f"ft-v3 temperature fitted on split C ({len(fit_rows)} rows, "
          f"{saturated} saturated at 1.0):")
    print(f"  T={fitted['temperature']}  NLL {fitted['nll_at_1.0']} -> "
          f"{fitted['nll']}  ECE {fitted['ece_at_1.0']} -> {fitted['ece']}")

    tasks = {task["task_id"]: task for task in
             (json.loads(line) for line
              in Path(args.suite).read_text(encoding="utf-8").splitlines()
              if line.strip())}
    directory = Path(args.votes_dir)

    def load(name: str) -> dict:
        payload = json.loads((directory / f"votes-d-{name}.json")
                             .read_text(encoding="utf-8"))
        return {row["task_id"]: row for row in payload["rows"]}

    seated = {name: load("seated-" + ("typed" if name == "typed-decisions"
                                      else name)) for name in TRIO}
    new_votes = load(NEW)
    rows = []
    for task_id in sorted(tasks):
        rows.append({
            "task_id": task_id,
            "expected": seated[TRIO[0]][task_id]["expected"],
            "candidates": tasks[task_id]["candidates"],
            "state": tasks[task_id]["state"],
            "question": tasks[task_id]["question"],
            "choices": {name: seated[name][task_id]["choice"]
                        for name in TRIO} | {NEW: new_votes[task_id]["choice"]},
            "confidences": {name: seated[name][task_id]["probability"]
                            for name in TRIO}
            | {NEW: new_votes[task_id]["probability"]},
        })

    tables = {name: load_calibration(name)["temperature"] for name in TRIO}
    shipped_tables = {name: tables[name] for name in TRIO}
    calibrated_tables = dict(shipped_tables, **{NEW: fitted["temperature"]})
    combiner = Combiner.load(args.combiner)

    live_today = {"english": tables["english"]}
    live_calibrated = {"english": tables["english"],
                       NEW: fitted["temperature"]}
    configs = [
        ("trio (raw)", TRIO, {}),
        ("trio (shipped tables)", TRIO, shipped_tables),
        ("trio + ft-v3 (raw)", TRIO + [NEW], {}),
        ("trio + ft-v3 (calibrated)", TRIO + [NEW], calibrated_tables),
        ("english x2 + ft-v3 (live today)", ["english", "english", NEW],
         live_today),
        ("english x2 + ft-v3 (calibrated)", ["english", "english", NEW],
         live_calibrated),
        # Same three devices, quorum counted over opinions instead of stems
        # (rootd --lineage-quorum): the duplicated english lineage becomes
        # one vote, so the distinct member can actually move the answer.
        ("english x2 + ft-v3, lineage-counted", ["english", NEW], {}),
        ("english x2 + ft-v3, lineage-counted (cal)", ["english", NEW],
         live_calibrated),
    ]
    print(f"\n{len(rows)} rows on split D, seen by nothing that was fitted")
    print(f"  {'configuration':46s} {'full':>7s} {'auto':>5s} "
          f"{'auto acc':>9s} {'decided':>8s}")
    results = {}
    for label, members, temperatures in configs:
        entry = decide(transform(rows, temperatures), members, combiner)
        results[label] = {"full": entry["full"], "auto": entry["auto"],
                          "auto_accuracy": entry["auto_accuracy"],
                          "decided": entry["decided"],
                          "temperatures": temperatures}
        print(f"  {label:46s} {entry['full']:>7.4f} {entry['auto']:>5d} "
              f"{entry['auto_accuracy']:>9.4f} {entry['decided']:>8.4f}")
    lineage_configs = [
        ("lineage-counted, combiner sees stems (live path)",
         ["english", "english", NEW], live_calibrated, False),
        ("lineage-counted, combiner sees opinions", ["english", "english",
                                                     NEW], live_calibrated,
         True),
    ]
    for label, members, temperatures, sees_opinions in lineage_configs:
        entry = decide_lineage(rows, members, temperatures, combiner,
                               sees_opinions)
        results[label] = {"auto": entry["auto"],
                          "auto_accuracy": entry["auto_accuracy"],
                          "decided": entry["decided"]}
        print(f"  {label:46s} {'-':>7s} {entry['auto']:>5d} "
              f"{entry['auto_accuracy']:>9.4f} {entry['decided']:>8.4f}")

    out = directory / "calibrated-gate.json"
    cross = None
    if args.cross_report and Path(args.cross_report).exists():
        report = json.loads(Path(args.cross_report).read_text(
            encoding="utf-8"))
        cross_tasks = {task["task_id"]: task for task in
                       (json.loads(line) for line
                        in Path(args.cross_suite).read_text(
                            encoding="utf-8").splitlines() if line.strip())}
        fit_lookup = {row["task_id"]: row for row in fit_rows}
        cross_rows = []
        for record in report["rows"]:
            task_id = record["task_id"]
            if task_id not in fit_lookup or task_id not in cross_tasks:
                continue
            cross_rows.append({
                "task_id": task_id,
                "expected": record["expected"],
                "candidates": cross_tasks[task_id]["candidates"],
                "state": cross_tasks[task_id]["state"],
                "question": cross_tasks[task_id]["question"],
                "choices": {"english": record["per_lineage"]["english"],
                            NEW: fit_lookup[task_id]["choice"]},
                "confidences": {"english": record["confidences"]["english"],
                                NEW: fit_lookup[task_id]["probability"]},
            })
        print(f"\ncross-check on {report.get('suite', Path(args.cross_report).name)}"
              f" ({len(cross_rows)} rows; temperature was fitted here, so the"
              f" calibrated rows are in-sample for T only)")
        print(f"  {'configuration':46s} {'full':>7s} {'auto':>5s} "
              f"{'auto acc':>9s} {'decided':>8s}")
        cross = {}
        for label, members, temperatures in [
                ("english x2 + ft-v3 (live membership)", ["english", "english",
                                                          NEW], live_today),
                ("english x2 + ft-v3 (calibrated)", ["english", "english",
                                                     NEW], live_calibrated),
                ("english x2 + ft-v3, lineage-counted", ["english", NEW],
                 live_today),
                ("english x2 + ft-v3, lineage-counted (cal)", ["english", NEW],
                 live_calibrated)]:
            entry = decide(transform(cross_rows, temperatures), members,
                           combiner)
            cross[label] = {"full": entry["full"], "auto": entry["auto"],
                            "auto_accuracy": entry["auto_accuracy"],
                            "decided": entry["decided"]}
            print(f"  {label:46s} {entry['full']:>7.4f} {entry['auto']:>5d} "
                  f"{entry['auto_accuracy']:>9.4f} {entry['decided']:>8.4f}")
        for label, sees_opinions in [
                ("lineage-counted, combiner sees stems (live path)", False),
                ("lineage-counted, combiner sees opinions", True)]:
            entry = decide_lineage(cross_rows, ["english", "english", NEW],
                                   live_calibrated, combiner, sees_opinions)
            cross[label] = {"auto": entry["auto"],
                            "auto_accuracy": entry["auto_accuracy"],
                            "decided": entry["decided"]}
            print(f"  {label:46s} {'-':>7s} {entry['auto']:>5d} "
                  f"{entry['auto_accuracy']:>9.4f} {entry['decided']:>8.4f}")

    out.write_text(json.dumps(
        {"fitted": fitted, "fit_rows": len(fit_rows),
         "fit_saturated_at_1": saturated, "configs": results,
         "cross_check": cross}, indent=2),
        encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
