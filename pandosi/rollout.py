"""RLCD rollout recording: the data layer of the evolution loop (Phase 5).

Every consensus decision is recorded with its full vote matrix. Later,
export_rollouts() converts these into RLCD-training JSONL rows (compatible
with laya-evals / laya-rs train_jsonl): the quorum majority becomes the
preferred answer; dissenting lineages become the rejected signal.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class RolloutRecorder:
    def __init__(self, rollout_dir: Path | str = "rollouts"):
        self.dir = Path(rollout_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "rollouts.jsonl"

    def record(self, request_id: str, state: str, question: str,
               candidates: list[str], votes: list[dict],
               choice: str | None, tier: str,
               playbook_hash: str | None = None) -> dict:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "request_id": request_id, "state": state,
                 "question": question, "candidates": candidates,
                 "consensus": choice, "tier": tier,
                 "playbook_hash": playbook_hash,
                 "votes": votes}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry


def export_training_rows(rollout_path: Path, out_path: Path,
                         min_votes: int = 3) -> int:
    """Convert consensus rollouts to RLCD training rows.

    Row format (laya-evals compatible):
      {state, question, candidates, preferred, rejected?}
    preferred = the quorum majority choice. For lines where at least one
    lineage dissented, rejected = the top dissenting choice (the RLCD
    contrast pair). Escalated/tie lines are skipped (no trusted signal).
    """
    rows = 0
    with open(rollout_path, encoding="utf-8") as fh, \
            open(out_path, "a", encoding="utf-8") as out:
        for line in fh:
            e = json.loads(line)
            votes = e.get("votes") or []
            if len(votes) < min_votes or not e.get("consensus"):
                continue
            counts: dict[str, int] = {}
            for v in votes:
                c = v.get("choice")
                if c:
                    counts[c] = counts.get(c, 0) + 1
            if len(counts) < 2:
                continue  # unanimous: no contrast pair (still usable for SFT)
            sorted_c = sorted(counts.items(), key=lambda kv: -kv[1])
            preferred, rejected = sorted_c[0][0], sorted_c[1][0]
            out.write(json.dumps({
                "state": e["state"], "question": e["question"],
                "candidates": e.get("candidates") or sorted(counts),
                "preferred": preferred, "rejected": rejected,
                "playbook_hash": e.get("playbook_hash"),
                "votes": {v.get("checkpoint", v.get("node_id", "?")):
                          v.get("choice") for v in votes},
            }, ensure_ascii=False) + "\n")
            rows += 1
    return rows
