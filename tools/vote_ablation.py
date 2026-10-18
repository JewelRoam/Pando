#!/usr/bin/env python3
"""Ask the questions the headline hides, using a harvest that already exists.

``tools/equal_budget_check.py`` answers "who won". This answers the three
follow-ups that decide what the answer *means*, from the same raw votes --
no model is started, no network is touched:

  1. **Per-member, per-suite.** An aggregate can hide a member that is dead on
     one distribution and fine on another. Reported as accuracy per suite so a
     collapse is visible as a collapse, not as a slightly lower average.
  2. **Aggregation variants.** The same votes re-aggregated over subsets. This
     is exploratory by construction -- the subsets were chosen after seeing the
     result -- so every row carries its own z and the caller is told which
     Bonferroni threshold applies to the number of variants printed.
  3. **Is the weakest member weak, or just badly labelled?** A member whose
     answers are a permuted version of the truth would score near chance while
     carrying real information. Tested as explicit single label swaps (
     praise<->complaint etc.), because a *free* permutation fitted on the same
     rows can always reach 1.0 and therefore proves nothing.

Distinct from a free-permutation upper bound, which is uninformative here: with
14 labels and ~26 rows per label the assignment problem can always be solved to
perfection, so the fitted number is 1.000 for every member, including the one
at 0.358. That is why the swap test uses named pairs and a Bonferroni
threshold rather than a fitted mapping.

Usage:
    .venv/bin/python tools/vote_ablation.py
    .venv/bin/python tools/vote_ablation.py --votes assets/votes/x.raw.json \
        --json /tmp/x.ablation.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_VOTES = "assets/votes/equal-budget-002.raw.json"

#: Two-sided z a single test must clear at 5% for the number of hypotheses
#: printed below. Computed in code so the doc and the number cannot drift.
def bonferroni_z(hypotheses: int) -> float:
    """The two-sided normal quantile for alpha = 0.05 / hypotheses."""
    alpha = 0.05 / max(1, hypotheses)
    # Acklam-free: bisect the normal CDF built from erf, which is all we need.
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < 1.0 - alpha / 2.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def load_votes(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ok_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("ok")]


def per_member(raw: dict) -> dict:
    """Accuracy, coverage and per-suite accuracy for every member of both arms."""
    out: dict[str, dict] = {}
    arms = {"A": raw["arm_a"], "B": raw["arm_b"]}
    for arm, members in arms.items():
        for url, rows in members.items():
            good = ok_rows(rows)
            by_suite: dict[str, list[bool]] = collections.defaultdict(list)
            for r in good:
                suite = str(r.get("task_id", "")).split("-")[0]
                by_suite[suite].append(r.get("choice") == r.get("expected"))
            out[f"{arm} {url}"] = {
                "url": url,
                "arm": arm,
                "n": len(rows),
                "ok": len(good),
                "coverage": len(good) / len(rows) if rows else float("nan"),
                "accuracy": (
                    sum(r.get("choice") == r.get("expected") for r in good) / len(good)
                    if good else float("nan")),
                "per_suite": {
                    k: {"n": len(v), "accuracy": sum(v) / len(v)}
                    for k, v in sorted(by_suite.items())},
            }
    return out


def rows_for(raw: dict, member: dict) -> list[dict]:
    """The raw rows behind one ``per_member`` entry.

    ``per_member`` names members ``"A <url>"`` / ``"B <url>"`` for a reader,
    but the raw harvest keys its arms in lower case. Going through the member's
    own ``arm``/``url`` fields keeps the two spellings from ever meeting.
    """
    return raw["arm_" + str(member["arm"]).lower()][member["url"]]


def variants(raw: dict) -> dict[str, dict]:
    """Re-aggregate arm B over subsets and McNemar each against arm A."""
    from tools.equal_budget_check import arm_b_decision, load_suite

    rows = load_suite(raw["suite"], None)
    per_member_votes = raw["arm_b"]
    urls = list(per_member_votes)
    arm_a = raw["arm_a"][raw["arm_a_url"]]

    # Named subsets have to be expressed as positions, because the harvest keys
    # members by URL and a URL means nothing to a reader. Order is the order
    # the harvest used.
    named = {
        "B4 pre-registered (all members)": urls,
        "B3 drop-text": [u for u in urls if "8766" not in u] or urls,
        "B2 two-LLM": [u for u in urls if "8769" not in u and "8766" not in u] or urls,
        "B3 drop-phi": [u for u in urls if "8770" not in u] or urls,
    }

    out: dict[str, dict] = {}
    for name, picked in named.items():
        if not picked:
            continue
        decisions = arm_b_decision(rows, per_member_votes, picked, False)
        answered = [d for d in decisions if d.get("choice") is not None]
        a_wrong = b_wrong = n = 0
        for i, d in enumerate(decisions):
            a = arm_a[i]
            if not a.get("ok") or d.get("choice") is None:
                continue
            n += 1
            a_ok = a["choice"] == a["expected"]
            b_ok = d["choice"] == d["expected"]
            if a_ok and not b_ok:
                a_wrong += 1
            if b_ok and not a_ok:
                b_wrong += 1
        delta = b_wrong - a_wrong
        se = math.sqrt(a_wrong + b_wrong) / 2 if (a_wrong + b_wrong) else float("nan")
        out[name] = {
            "members": picked,
            "n_paired": n,
            "accuracy": (sum(d["choice"] == d["expected"] for d in answered)
                         / len(answered)) if answered else float("nan"),
            "coverage": len(answered) / len(decisions) if decisions else float("nan"),
            "a_right_b_wrong": a_wrong,
            "b_right_a_wrong": b_wrong,
            "delta": delta,
            "se_delta": se,
            "z": (delta / se) if se and not math.isnan(se) else float("nan"),
        }
    return out


def swap_test(rows: list[dict], pairs: int = 6) -> dict:
    """Would relabelling two answers rescue this member?

    The pairs tested are the largest off-diagonal confusion cells, so the test
    is aimed where a name-mapping bug would show. A free permutation is *not*
    used: an assignment fitted on the same rows can always reach 1.0 (14 labels,
    ~26 rows each) and would be reported as a fix that does not exist.
    """
    good = ok_rows(rows)
    if not good:
        return {}
    n = len(good)
    base_hits = sum(r["choice"] == r["expected"] for r in good)
    base = base_hits / n
    se = math.sqrt(base * (1 - base) / n)

    confusion = collections.Counter(
        (r["expected"], r["choice"]) for r in good if r["choice"] != r["expected"])
    labels: list[tuple[str, str]] = []
    for (expected, choice), _count in confusion.most_common():
        key = tuple(sorted((expected, choice)))
        if key not in labels:
            labels.append(key)
        if len(labels) >= pairs:
            break

    trials = []
    for a, b in labels:
        hits = 0
        for r in good:
            choice = {a: b, b: a}.get(r["choice"], r["choice"])
            hits += choice == r["expected"]
        trials.append({
            "swap": [a, b],
            "accuracy": hits / n,
            "delta": (hits - base_hits) / n,
            "z": ((hits - base_hits) / n) / se if se else float("nan"),
        })
    threshold = bonferroni_z(len(trials) + 1)
    return {
        "n": n,
        "base_accuracy": base,
        "se": se,
        "chance": 1.0 / len({r["expected"] for r in good} | {r["choice"] for r in good}),
        "z_vs_chance": (base - (1.0 / len({r["expected"] for r in good}
                                            | {r["choice"] for r in good}))) / se,
        "trials": trials,
        "bonferroni_z": threshold,
        # Split by sign: a swap that clears the threshold with a *negative*
        # delta is significant evidence against the member, not a repair, and
        # reporting the two together is how a reader mistakes harm for a fix.
        "rescued": [t["swap"] for t in trials if t["z"] > threshold],
        "worsened": [t["swap"] for t in trials if -t["z"] > threshold],
    }


def per_class(rows: list[dict]) -> dict:
    good = ok_rows(rows)
    recall: dict[str, dict] = {}
    for r in good:
        cell = recall.setdefault(r["expected"], {"n": 0, "right": 0})
        cell["n"] += 1
        cell["right"] += r["choice"] == r["expected"]
    for v in recall.values():
        v["recall"] = v["right"] / v["n"] if v["n"] else float("nan")
    return dict(sorted(recall.items(), key=lambda kv: kv[1]["recall"]))


def weakest(per: dict) -> str:
    scored = {k: v for k, v in per.items() if v["accuracy"] == v["accuracy"]}
    return min(scored, key=lambda k: scored[k]["accuracy"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--votes", default=DEFAULT_VOTES)
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    path = Path(args.votes)
    if not path.is_absolute():
        path = ROOT / path
    raw = load_votes(path)

    per = per_member(raw)
    print(f"votes: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print(f"suite: {raw['suite']}   rows: {raw['n_rows']}\n")

    print("member                              n   ok   cov    acc   per-suite")
    for name, m in per.items():
        suites = "  ".join(f"{k}={v['accuracy']:.4f}(n={v['n']})"
                           for k, v in m["per_suite"].items())
        print(f"{name:34s} {m['n']:3d} {m['ok']:4d} {m['coverage']:.3f} "
              f"{m['accuracy']:.4f}   {suites}")

    var = variants(raw)
    print(f"\naggregation variants ({len(var)} hypotheses, "
          f"Bonferroni needs |z| > {bonferroni_z(len(var)):.2f})")
    print(f"{'variant':34s} {'acc':>7s} {'cov':>6s} {'aR/bW':>6s} {'bR/aW':>6s} "
          f"{'d':>4s} {'SE':>5s} {'z':>6s}")
    for name, v in var.items():
        print(f"{name:34s} {v['accuracy']:7.4f} {v['coverage']:6.3f} "
              f"{v['a_right_b_wrong']:6d} {v['b_right_a_wrong']:6d} "
              f"{v['delta']:4d} {v['se_delta']:5.2f} {v['z']:6.2f}")

    key = weakest(per)
    member = per[key]
    print(f"\nweakest member: {key}")
    print("  per-class recall (worst first):")
    for label, cell in list(per_class(rows_for(raw, member)).items())[:6]:
        print(f"    {label:22s} {cell['right']:3d}/{cell['n']:3d} = {cell['recall']:.2f}")

    swap = swap_test(rows_for(raw, member))
    print(f"  is it a label-mapping artifact? chance={swap['chance']:.4f} "
          f"z-vs-chance={swap['z_vs_chance']:+.2f}")
    for t in swap["trials"]:
        print(f"    swap {t['swap'][0]:16s}<->{t['swap'][1]:16s} "
              f"acc={t['accuracy']:.4f} delta={t['delta']:+.4f} z={t['z']:+.2f}")
    print(f"    Bonferroni over {len(swap['trials']) + 1} hypotheses needs "
          f"|z| > {swap['bonferroni_z']:.2f}; rescued: "
          f"{swap['rescued'] or 'none'}; significantly worse: "
          f"{swap['worsened'] or 'none'}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"per_member": per, "variants": var, "weakest": key,
             "swap_test": swap,
             "per_class": per_class(rows_for(raw, member))},
            indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
