"""Decide whether a proposal may be executed. Do not touch the world.

The colony can already propose and vote. It cannot yet do the thing it voted
for, and wiring that straight through would run almost every proposal: the
decision models currently answer 赞同 far more often than they refuse.

So this module only judges. An action runs only when three things are true:
the action is in the action space, enough other decision models adopted it,
and a tested actuator for that exact action was passed in. The default
actuator set is empty. A judgment of "adopted, not executed" is a hold on
the world, recorded as such, not a success.

Nothing here starts itself. A dry run reads the mailbox and prints counts.
It does not write an outcome, restart a process, or copy a seed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .genesis import ACTION_SPACE, Proposal
from .proposal_mail import fetch_proposals, is_reaction_task

ADOPT = "赞同"
HOLD = "搁置"
DECLINE = "反对"
STANCES = (ADOPT, HOLD, DECLINE)


def tally(proposal: Proposal, reactions: list[dict]) -> dict[str, list[str]]:
    """One vote per decision model. The author does not count."""
    buckets = {ADOPT: [], HOLD: [], DECLINE: []}
    seen: set[str] = set()
    for reaction in reactions:
        if str(reaction.get("target_id") or "") != proposal.proposal_id:
            continue
        node = str(reaction.get("node_id") or "").strip()
        stance = str(reaction.get("stance") or "").strip()
        if not node or node == proposal.author or node in seen:
            continue
        if stance not in buckets:
            continue
        seen.add(node)
        buckets[stance].append(node)
    return buckets


def judge(proposal: Proposal, reactions: list[dict], *,
          min_adopt: int = 2, veto: bool = True,
          actuators: dict | None = None) -> dict:
    """Return what should happen. Does not perform the action by itself.

    ``actuators`` maps an action name to ``(Proposal) -> bool``. Missing
    means the colony may have adopted the proposal and still must not act.
    """
    votes = tally(proposal, reactions)
    result = {
        "proposal_id": proposal.proposal_id,
        "action": proposal.action,
        "author": proposal.author,
        "adopt": list(votes[ADOPT]),
        "hold": list(votes[HOLD]),
        "decline": list(votes[DECLINE]),
        "adopted": False,
        "executed": False,
        "success": None,
        "detail": "",
    }
    if proposal.action not in ACTION_SPACE:
        result["detail"] = "held: action is not in the action space"
        return result
    if veto and votes[DECLINE]:
        result["detail"] = "held: a decision model declined"
        return result
    if len(votes[ADOPT]) < min_adopt:
        result["detail"] = (f"held: {len(votes[ADOPT])} adopt "
                            f"< {min_adopt}")
        return result
    result["adopted"] = True
    actuator = (actuators or {}).get(proposal.action)
    if actuator is None:
        result["detail"] = "adopted, not executed: no actuator"
        return result
    try:
        ok = bool(actuator(proposal))
    except Exception as exc:
        result["detail"] = f"actuator failed before completion: {exc}"
        return result
    result["executed"] = True
    result["success"] = ok
    result["detail"] = "actuator reported " + ("success" if ok else "failure")
    return result


def summarize(results: list[dict]) -> dict:
    counts = Counter(row["detail"].split(":")[0] if row["detail"] else "?"
                     for row in results)
    # Stable buckets a person can read without parsing every detail line.
    return {
        "considered": len(results),
        "held_not_in_space": sum(row["detail"].startswith("held: action")
                                 for row in results),
        "held_veto": sum(row["detail"].startswith("held: a decision")
                         for row in results),
        "held_quorum": sum(row["detail"].startswith("held: ")
                           and "adopt <" in row["detail"]
                           for row in results),
        "adopted_not_executed": sum(
            row["adopted"] and not row["executed"] for row in results),
        "executed": sum(row["executed"] for row in results),
        "details": dict(counts),
    }


def dry_run(proposals: list[Proposal], reactions: list[dict], **policy) -> dict:
    """Judge every proposal. No actuator, no write."""
    results = [judge(proposal, reactions, actuators={}, **policy)
               for proposal in proposals]
    return {"summary": summarize(results), "results": results}


def _load_reactions(url: str, token: str | None, limit: int) -> list[dict]:
    from .proposal_mail import _request
    from .genesis import verify_reaction
    listing = _request(url, "/tasks?status=all", token, "GET")
    tasks = listing.get("tasks") or []
    tasks = tasks[-limit:]
    found = []
    for task in tasks:
        if not isinstance(task, dict) or not is_reaction_task(str(task.get("title") or "")):
            continue
        full = _request(url, f"/task/{task.get('id')}", token, "GET")
        body = (full.get("task") or {}).get("body")
        if not isinstance(body, str):
            continue
        try:
            record = json.loads(body)
        except json.JSONDecodeError:
            continue
        view = verify_reaction(record)
        if view is not None:
            found.append(view)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="judge proposals; does not execute them")
    parser.add_argument("--mailbox", default="http://127.0.0.1:8776")
    parser.add_argument("--token-file", default="~/.pandosi/mailbox.token")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        print("refusing to run without --dry-run; no actuator is shipped",
              flush=True)
        return 2
    from pathlib import Path
    token = Path(args.token_file).expanduser().read_text(encoding="utf-8").strip()
    proposals = fetch_proposals(args.mailbox, token, status="all",
                                max_items=args.limit)
    reactions = _load_reactions(args.mailbox, token, max(args.limit * 4, args.limit))
    report = dry_run(proposals, reactions)
    summary = report["summary"]
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    shown = 0
    for row in report["results"]:
        if not row["adopted"]:
            continue
        print(f"{row['proposal_id']} {row['action']} "
              f"adopt={len(row['adopt'])} {row['detail']}", flush=True)
        shown += 1
        if shown >= 5:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
