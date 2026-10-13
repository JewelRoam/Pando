"""Render docs/dashboard.html from metrics/decisions.jsonl.

Usage: python tools/dashboard.py
No server, no telemetry - a static local file, per docs/METRICS.md.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics" / "decisions.jsonl"
OUT = ROOT / "docs" / "dashboard.html"


def main() -> None:
    entries = []
    if METRICS.exists():
        for line in METRICS.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(entries)
    tiers = Counter(e["tier"] for e in entries)
    choices = Counter(e["choice"] for e in entries)
    agreements = [e["agreement"] for e in entries if e["agreement"]]
    mean_agreement = round(sum(agreements) / len(agreements), 3) if agreements else 0.0

    rows = "".join(
        f"<tr><td>{e['ts']}</td><td>{e['choice']}</td><td>{e['tier']}</td>"
        f"<td>{e['agreement']}</td><td>{e['responded']}/{e['quorum_size']}</td></tr>"
        for e in reversed(entries[-50:]))

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>PandoSI dashboard</title>
<style>
body{{font-family:ui-monospace,monospace;margin:2rem;background:#101512;color:#e8efe9}}
h1{{letter-spacing:-0.03em}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;max-width:900px}}
.card{{background:#1a2420;padding:18px;border-top:3px solid #5ed6c7}}
.card b{{font-size:28px;display:block}}
table{{border-collapse:collapse;margin-top:2rem;width:100%;max-width:900px}}
td,th{{border:1px solid #2c3733;padding:6px 10px;text-align:left;font-size:13px}}
</style></head><body>
<h1>PandoSI decisions</h1>
<div class="cards">
<div class="card"><span>total decisions</span><b>{total}</b></div>
<div class="card"><span>tiers</span><b>{dict(tiers)}</b></div>
<div class="card"><span>mean agreement</span><b>{mean_agreement}</b></div>
<div class="card"><span>choices</span><b>{dict(choices)}</b></div>
</div>
<table><tr><th>time</th><th>choice</th><th>tier</th><th>agreement</th><th>responded/quorum</th></tr>
{rows}</table>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({total} decisions)")


if __name__ == "__main__":
    main()
