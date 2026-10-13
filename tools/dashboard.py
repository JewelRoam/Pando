"""Render docs/dashboard.html from metrics + rollouts (local only).

Sections: decision stats, per-lineage vote stats, playbook attribution,
disagreement log (vote matrix for verify/escalate), recent decisions.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics" / "decisions.jsonl"
ROLLOUTS = ROOT / "rollouts" / "rollouts.jsonl"
OUT = ROOT / "docs" / "dashboard.html"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def main() -> None:
    metrics = []
    if METRICS.exists():
        for line in METRICS.read_text(encoding="utf-8").splitlines():
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rollouts = []
    if ROLLOUTS.exists():
        for line in ROLLOUTS.read_text(encoding="utf-8").splitlines():
            try:
                rollouts.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(metrics)
    tiers = Counter(e["tier"] for e in metrics)
    mean_agreement = (round(sum(e["agreement"] for e in metrics
                                if e["agreement"]) / total, 3)
                      if total else 0.0)

    # per-lineage vote stats
    lineage_votes: Counter = Counter()
    lineage_correct_if_known = Counter()
    for r in rollouts:
        for v in r.get("votes", []):
            key = v.get("checkpoint") or v.get("node_id") or "?"
            lineage_votes[key] += 1
    pb_hashes = Counter(r.get("playbook_hash", "?")[:16]
                        for r in rollouts if r.get("playbook_hash"))

    # disagreement log: verify/escalate with full matrix
    disagree_rows = []
    for r in rollouts:
        if r.get("tier") in ("verify", "escalate"):
            matrix = " | ".join(
                f"{esc(v.get('checkpoint') or v.get('node_id', '?'))}: "
                f"{esc(v.get('choice'))}@{v.get('probability')}"
                for v in r.get("votes", []))
            disagree_rows.append(
                f"<tr><td>{esc(r['ts'])}</td><td>{esc(r['state'][:60])}</td>"
                f"<td>{esc(r.get('tier'))}</td><td>{matrix}</td></tr>")

    recent_rows = "".join(
        f"<tr><td>{esc(e['ts'])}</td><td>{esc(e['choice'])}</td>"
        f"<td>{esc(e['tier'])}</td><td>{e['agreement']}</td>"
        f"<td>{e['responded']}/{e['quorum_size']}</td></tr>"
        for e in reversed(metrics[-50:]))

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>PandoSI dashboard</title>
<style>
body{{font-family:ui-monospace,monospace;margin:2rem;background:#101512;color:#e8efe9}}
h1,h2{{letter-spacing:-0.03em}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;max-width:1000px}}
.card{{background:#1a2420;padding:18px;border-top:3px solid #5ed6c7}}
.card b{{font-size:26px;display:block;word-break:break-all}}
table{{border-collapse:collapse;margin:1.5rem 0 2.5rem;width:100%;max-width:1100px}}
td,th{{border:1px solid #2c3733;padding:6px 10px;text-align:left;font-size:13px}}
th{{background:#1a2420}}
</style></head><body>
<h1>PandoSI dashboard</h1>
<div class="cards">
<div class="card"><span>decisions</span><b>{total}</b></div>
<div class="card"><span>tiers</span><b>{esc(dict(tiers))}</b></div>
<div class="card"><span>mean agreement</span><b>{mean_agreement}</b></div>
<div class="card"><span>playbook versions</span><b>{len(pb_hashes)}</b></div>
</div>
<h2>Per-lineage votes</h2>
<table><tr><th>lineage / checkpoint</th><th>votes cast</th></tr>
{''.join(f'<tr><td>{esc(k)}</td><td>{v}</td></tr>' for k, v in lineage_votes.most_common())}
</table>
<h2>Playbook attribution</h2>
<table><tr><th>playbook hash</th><th>decisions</th></tr>
{''.join(f'<tr><td>{esc(k)}</td><td>{v}</td></tr>' for k, v in pb_hashes.most_common())}
</table>
<h2>Disagreement log (verify / escalate)</h2>
<table><tr><th>time</th><th>state</th><th>tier</th><th>vote matrix</th></tr>
{''.join(disagree_rows[-50:]) or '<tr><td colspan=4>no disagreements recorded</td></tr>'}
</table>
<h2>Recent decisions</h2>
<table><tr><th>time</th><th>choice</th><th>tier</th><th>agreement</th><th>responded</th></tr>
{recent_rows}
</table>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} | decisions={total} rollouts={len(rollouts)} "
          f"disagreements={len(disagree_rows)}")


if __name__ == "__main__":
    main()
