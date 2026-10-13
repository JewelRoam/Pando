"""Expand the graft suite: 16 -> 48 tasks (24 added, 8 per lineage).

All tasks use deterministic, human-obvious labels (explicit scenario
phrasing); no ambiguous cases. Distribution: english 16, multilingual 16
(8 languages), typed-decisions 16 (workflow states). RISK is the shared
5-level score criteria list.

Usage: python tools/expand_suite.py   (idempotent: refuses to double-add)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "assets" / "graft_suite.jsonl"
RISK = ["negligible", "low", "moderate", "high", "critical"]

NEW = [
    # --- english additions (8) ---
    ("en07", "english", "noul", "Fire! Everyone get out of the building now!", "urgent", "true"),
    ("en08", "english", "noul", "The quarterly report was emailed to the team", "urgent", "false"),
    ("en09", "english", "choice", "This thing broke after two days, unacceptable", "intent",
     ["complaint", "refund", "greeting"], "complaint"),
    ("en10", "english", "choice", "Hi! Just wanted to say thanks for the great service", "intent",
     ["complaint", "refund", "greeting"], "greeting"),
    ("en11", "english", "score", "Smoke is coming out of the charging cable", "risk", "critical"),
    ("en12", "english", "score", "Customer asks about opening hours", "risk", "negligible"),
    ("en13", "english", "noul", "My card was charged twice for the same order", "billing_issue", "true"),
    ("en14", "english", "choice", "Where is my refund? It has been three weeks", "intent",
     ["refund", "greeting", "login_issue"], "refund"),
    # --- multilingual additions (8, one per language) ---
    ("ml05b", "multilingual", "noul", "\u5927\u697c\u7740\u706b\u4e86\uff0c\u5feb\u9003\u751f\uff01", "urgent", "true"),
    ("ml06b", "multilingual", "noul", "Das Meeting wurde auf Montag verschoben", "urgent", "false"),
    ("ml07b", "multilingual", "choice", "Quiero mi dinero de vuelta, el producto lleg\u00f3 roto", "intent",
     ["refund", "complaint", "greeting"], "refund"),
    ("ml08b", "multilingual", "choice", "Bonjour, je voulais juste vous dire merci", "intent",
     ["complaint", "refund", "greeting"], "greeting"),
    ("ml09b", "multilingual", "score", "\uc5f0\uae30\uc5d0\uc11c \uc5f0\uae30\uac00 \ub098\uace0 \uc788\uc5b4\uc694", "risk", "critical"),
    ("ml10b", "multilingual", "score", "\u0633\u0624\u0627\u0644 \u062d\u0648\u0644 \u0645\u0648\u0627\u0639\u064a\u062f \u0627\u0644\u0639\u0645\u0644", "risk", "negligible"),
    ("ml11b", "multilingual", "noul", "मेरा ऑर्डर अभी तक नहीं आया", "order_missing", "true"),
    ("ml12b", "multilingual", "choice", "Il prodotto \u00e8 guasto dopo due giorni", "intent",
     ["complaint", "refund", "greeting"], "complaint"),
    # --- typed-decisions additions (8) ---
    ("td05", "typed-decisions", "choice", "State: user reports smoke from the device", "route",
     ["emergency_stop", "standard_ticket", "discount"], "emergency_stop"),
    ("td06", "typed-decisions", "choice", "State: user asks about store hours", "route",
     ["escalate", "faq_reply"], "faq_reply"),
    ("td07", "typed-decisions", "noul", "State: payment succeeded but order not confirmed", "needs_followup", "true"),
    ("td08", "typed-decisions", "noul", "State: user confirmed shipping address unchanged", "needs_followup", "false"),
    ("td09", "typed-decisions", "score", "State: 50 failed logins from one IP in 60 seconds", "risk", "critical"),
    ("td10", "typed-decisions", "score", "State: user prefers email contact", "risk", "negligible"),
    ("td11", "typed-decisions", "choice", "State: refund approved, awaiting method", "route",
     ["original_payment", "store_credit", "bank_transfer"], "original_payment"),
    ("td12", "typed-decisions", "choice", "State: VIP user, outage exceeding SLA", "route",
     ["compensation", "ignore", "standard_reply"], "compensation"),
]


def main() -> None:
    tasks = [json.loads(l) for l in SUITE.read_text(encoding="utf-8").splitlines()]
    existing_ids = {t["task_id"] for t in tasks}
    added = 0
    with SUITE.open("a", encoding="utf-8") as fh:
        for tid, ckpt, qtype, state, question, a, *rest in NEW:
            if tid in existing_ids:
                continue
            row = {"task_id": tid, "checkpoint": ckpt, "qtype": qtype,
                   "state": state, "question": question}
            if qtype == "choice":
                row["candidates"] = a
                row["expected"] = rest[0]
            elif qtype == "score":
                row["criteria"] = RISK
                row["expected"] = a
            else:  # noul
                row["expected"] = a
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            added += 1
    print(f"added {added} tasks; suite now has {len(tasks) + added}")


if __name__ == "__main__":
    main()
