#!/usr/bin/env python3
"""Fit a temperature table for a *remote* stem, from the wire.

Why this exists
---------------
``tools/calibrate_checkpoint.py`` fits Laya locally and needs torch. The
checkpoint that actually votes in the live group does not live here: it is an
int8 ONNX file on a Raspberry Pi 5, reachable only over LAN. So the confidence
scale that the root's gate depends on was, until this tool, estimated nowhere
and asserted nowhere -- ``uncalibrated_checkpoints: ["laya-en"]`` was a true
statement about the system and an unfalsifiable one about the model.

This tool does the smallest honest thing: it asks the running member the
questions the suites already have labels for, and fits one parameter.

What is and is not claimed
--------------------------
- Temperature scaling fits *confidence*, not *accuracy*. Accuracy is invariant
  under it. If the member is wrong, a calibrated member is a wrong member that
  knows how unsure it should be.
- A single temperature is a single curve. It is fitted on states that must be
  disjoint from whatever the checkpoint was trained on, and it is only valid on
  the distribution it was fitted on. ``--holdout`` rows from another domain are
  reported separately and are expected to get *worse*; see step 4 below.
- Disjointness is checked by state string against ``--training``. Overlap is
  reported and those rows are dropped, because fitting on rows the model has
  seen measures memorisation, not calibration.

Usage
-----
  # fit, report, and eyeball the holdout before writing anything
  python tools/fit_stem_temperature.py \
      --stem-url http://10.1.0.22:35575 \
      --fit assets/heldout_test.jsonl,assets/heldout2_test.jsonl,assets/heldout3_test.jsonl \
      --holdout assets/heldout4_test.jsonl,assets/heldout6_test.jsonl \
      --shift assets/heldout5_test.jsonl \
      --training rollouts/distill_ft5_train_laya.jsonl \
      --checkpoint laya-en

  # write assets/calibration/<checkpoint>.json once the numbers say so
  ... --write

A stem URL is ephemeral: stems bind an ephemeral port and re-register on every
restart. Read the current one from the root's /nodes, do not cache it.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_suites(spec: str) -> list[dict]:
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
    return rows


def harvest(rows: list[dict], stem_url: str, timeout: float) -> list[dict]:
    """Ask the member each labelled row, one at a time, and keep its answer."""
    out = []
    url = stem_url.rstrip("/") + "/decide"
    for i, row in enumerate(rows):
        request = {"state": row["state"], "question": row["question"],
                   "candidates": row["candidates"],
                   "request_id": f"cal-{i:05d}"}
        body = json.dumps(request).encode("utf-8")
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type":
                                              "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                vote = json.loads(resp.read())["vote"]
            out.append({"suite": row["_suite"], "state": row["state"],
                        "expected": row["expected"], "choice": vote["choice"],
                        "p": float(vote["probability"]),
                        "checkpoint": vote.get("checkpoint"),
                        "neutral": vote.get("neutral")})
        except (urllib.error.URLError, KeyError, ValueError) as exc:
            out.append({"suite": row["_suite"], "error": str(exc)})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}", file=sys.stderr, flush=True)
    return out


# --- the arithmetic, in numpy so the grid search is not hand-rolled ---------

def _arrays(rows: list[dict]):
    import numpy as np
    ok = [r for r in rows if "error" not in r]
    p = np.clip(np.array([r["p"] for r in ok], dtype=float), 1e-6, 1 - 1e-6)
    y = np.array([1.0 if r["choice"] == r["expected"] else 0.0 for r in ok])
    return p, y


def _apply(p, temperature):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / temperature))


def _nll(p, y):
    import numpy as np
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ece(p, y, bins=10):
    import numpy as np
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum():
            total += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(total)


def fit_temperature(p, y) -> float:
    """Grid-search the single temperature that minimises NLL."""
    import numpy as np
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 4000))
    losses = [_nll(_apply(p, t), y) for t in grid]
    return float(grid[int(np.argmin(losses))])


def report(name, rows, temperature=None):
    p, y = _arrays(rows)
    line = (f"{name:<10} n={len(p):<5} accuracy={y.mean():.4f} "
            f"mean_p={p.mean():.4f} ece={_ece(p, y):.4f} nll={_nll(p, y):.4f}")
    if temperature is not None:
        pc = _apply(p, temperature)
        line += (f"  |  calibrated: mean_p={pc.mean():.4f} "
                 f"ece={_ece(pc, y):.4f} nll={_nll(pc, y):.4f}")
    # The sample size is stated next to the number on purpose: a 240-row suite
    # has a 1-SE accuracy of about 2.7pp, so differences below that are noise.
    import numpy as np
    se = float(np.sqrt(y.mean() * (1 - y.mean()) / len(y)))
    print(line + f"   (1se_acc={se:.4f})")
    return {"n": int(len(p)), "accuracy": float(y.mean()),
            "mean_p_raw": float(p.mean()), "ece_raw": _ece(p, y),
            "nll_raw": _nll(p, y),
            "mean_p_calibrated": (float(_apply(p, temperature).mean())
                                  if temperature is not None else None),
            "ece_calibrated": (_ece(_apply(p, temperature), y)
                               if temperature is not None else None),
            "nll_calibrated": (_nll(_apply(p, temperature), y)
                               if temperature is not None else None)}


def split_by_suite(rows: list[dict], temperature):
    """Per-suite numbers.

    An aggregate hides which domain went wrong: ``--shift`` above was one
    number for two domains with opposite behaviour, and the average made a
    domain that got worse look like one that got better. Report each suite.
    """
    out = {}
    for suite in dict.fromkeys(r["suite"] for r in rows):
        group = [r for r in rows if r["suite"] == suite]
        if not group or any("error" in r for r in group):
            out[suite] = {"n": len(group), "error": "harvest incomplete"}
            continue
        out[suite] = report(f"  {Path(suite).name}", group, temperature)
    return out


def worse_domains(by_suite: dict) -> list[str]:
    """Suites where this single temperature makes calibration *worse*."""
    return sorted(
        name for name, stat in (by_suite or {}).items()
        if stat.get("ece_raw") is not None
        and stat.get("ece_calibrated") is not None
        and stat["ece_calibrated"] > stat["ece_raw"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem-url", required=True,
                        help="e.g. http://10.1.0.22:35575 (from /nodes)")
    parser.add_argument("--fit", required=True, help="comma-separated suites")
    parser.add_argument("--holdout", default="")
    parser.add_argument("--shift", default="",
                        help="rows from another domain: expected to get worse")
    parser.add_argument("--training", default="",
                        help="suite the checkpoint trained on; overlaps are "
                             "dropped from the fit")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--write", action="store_true",
                        help="write assets/calibration/<checkpoint>.json")
    args = parser.parse_args()

    fit_rows = load_suites(args.fit)
    if args.training:
        seen = {json.loads(line)["state"]
                for line in (ROOT / args.training).read_text(
                    encoding="utf-8").splitlines() if line.strip()}
        before = len(fit_rows)
        fit_rows = [r for r in fit_rows if r["state"] not in seen]
        print(f"training-overlap: dropped {before - len(fit_rows)}/"
              f"{before} fit rows the checkpoint has already seen")

    print(f"harvesting {len(fit_rows)} fit rows from {args.stem_url}")
    fit_votes = harvest(fit_rows, args.stem_url, args.timeout)
    errors = sum(1 for r in fit_votes if "error" in r)
    if errors:
        print(f"WARNING: {errors}/{len(fit_votes)} decisions failed; "
              f"a fit on a partial harvest is a fit on a biased sample")

    p, y = _arrays(fit_votes)
    temperature = fit_temperature(p, y)
    print(f"\nfitted temperature = {temperature:.4f}")
    summary = {"fit": report("fit", fit_votes, temperature),
               "fit_by_suite": split_by_suite(fit_votes, temperature)}

    for label, spec in (("holdout", args.holdout), ("shift", args.shift)):
        if not spec:
            continue
        rows = load_suites(spec)
        print(f"harvesting {len(rows)} {label} rows")
        votes = harvest(rows, args.stem_url, args.timeout)
        summary[label] = report(label, votes, temperature)
        summary[f"{label}_by_suite"] = split_by_suite(votes, temperature)

    # One temperature is one curve. Any domain the fit makes worse must be
    # named in the artifact, not averaged into the shift line (round 033).
    out_of_scope = worse_domains(summary.get("shift_by_suite"))
    if out_of_scope:
        print("  ^ out-of-scope: this table makes these domains WORSE: "
              + ", ".join(out_of_scope)
              + "  (named in the artifact; scoped, not hidden)")

    if args.write:
        if not args.checkpoint:
            print("--write needs --checkpoint", file=sys.stderr)
            return 2
        out = ROOT / "assets" / "calibration" / f"{args.checkpoint}.json"
        payload = {
            "checkpoint": args.checkpoint,
            "temperature": round(temperature, 4),
            "source": ("tools/fit_stem_temperature.py - harvested from "
                       f"{args.stem_url}, NLL fit on "
                       f"{[s for s in args.fit.split(',') if s]}"),
            "n_tasks": summary["fit"]["n"],
            "accuracy": round(summary["fit"]["accuracy"], 4),
            "raw_ece": round(summary["fit"]["ece_raw"], 4),
            "calibrated_ece": round(summary["fit"]["ece_calibrated"], 4),
            "raw_log_loss": round(summary["fit"]["nll_raw"], 4),
            "calibrated_log_loss": round(summary["fit"]["nll_calibrated"], 4),
            "fit_suites": [s for s in args.fit.split(",") if s],
            "holdout_suites": [s for s in args.holdout.split(",") if s],
            "shift_suites": [s for s in args.shift.split(",") if s],
            "holdout": summary.get("holdout"),
            "holdout_by_suite": summary.get("holdout_by_suite"),
            "shift": summary.get("shift"),
            "shift_by_suite": summary.get("shift_by_suite"),
            # Same strings as fit_suites / shift_by_suite keys, so a reader
            # can look a domain up in the tables instead of guessing at a
            # basename that may not be unique across suites.
            "fitted_domains": [s for s in args.fit.split(",") if s],
            "out_of_scope_domains": out_of_scope,
            "note": ("temperature scaling fits confidence, not accuracy: "
                     "accuracy is invariant under it, so this makes a wrong "
                     "member a wrong member that knows how unsure to be. It "
                     "does not by itself open any auto tier; read "
                     "out_of_scope_domains before using the table offline. "
                     "Do not read this file as '<checkpoint> is calibrated' - "
                     "read it as 'calibrated on fitted_domains'."),
        }
        if out.exists():
            print(f"refusing to overwrite {out}", file=sys.stderr)
            return 2
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
