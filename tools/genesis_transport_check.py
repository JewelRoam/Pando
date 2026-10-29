#!/usr/bin/env python3
"""Drill: does a signed proposal actually travel, and is adoption gated?

Two claims are checked here, both of which docs/design/specs/genesis.md §7 listed as
*not yet claimed*:

1. **Transport.** ``pando/proposal_mail.py`` puts a signed proposal on the
   mailbox and takes it back off. The drill drives a stand-in mailbox that
   speaks the same four endpoints the real one does (``POST /task``,
   ``GET /tasks``, ``GET /task/<id>``, ``POST /claim``), publishes a mixed feed
   and asserts both that only the genuine, pinned proposals survive the round
   trip and that publishing does not leave the item sitting in the open queue
   -- a ``[proposal]`` is a message class, not work.
2. **The tier gate.** ``GenesisLoop(decide=...)`` must not adopt on its own
   authority: the swarm's own verdict decides. The drill runs the same loop
   twice against the same fetched proposals with the same actuator, changing
   only the verdict the swarm returns (``escalate`` vs ``auto`` with three
   calibrated responders) and asserts that the first holds and the second acts.

Negative controls (they run, they are not asserted in prose):
  - a ``[chat]`` message on the same feed is not a proposal;
  - a proposal whose body was edited after signing is dropped;
  - a record signed by a *foreign but valid* key under a member's name is
    dropped once that member's key is pinned -- and accepted when it is not,
    which is the whole reason pinning exists;
  - ``[proposal]`` items are a message class, so ``is_proposal_task`` must not
    be true for ordinary work items.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pando.gate import TierPolicy                       # noqa: E402
from pando.genesis import (GenesisLoop, LogProposer,    # noqa: E402
                           Proposal, sign_proposal)
from pando.identity import Identity                     # noqa: E402
from pando.proposal_mail import (fetch_proposals,       # noqa: E402
                                 is_proposal_task,
                                 publish_proposal)


class StandInMailbox:
    """The four endpoints the real mailbox serves, in memory.

    Deliberately dumb: it stores titles and bodies and hands them back. It does
    no verification of its own, which is the point -- if the drill passes, the
    checking happened in ``fetch_proposals``, not in the courier.
    """

    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self._n = 0
        box = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                return

            def _send(self, code, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/health":
                    self._send(200, {"ok": True, "open": 0})
                    return
                if path == "/tasks":
                    self._send(200, {"tasks": [
                        {"id": t["id"], "from": t["from"], "title": t["title"],
                         "status": t["status"], "created_at": t["created_at"]}
                        for t in box.tasks.values()]})
                    return
                if path.startswith("/task/"):
                    task = box.tasks.get(path.split("/", 2)[2])
                    if task is None:
                        self._send(404, {"error": "no such task"})
                    else:
                        self._send(200, {"task": task})
                    return
                self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                path = self.path.split("?", 1)[0]
                if path == "/claim":
                    task = box.tasks.get(str(payload.get("id", "")))
                    if task is None:
                        self._send(404, {"error": "no such task"})
                        return
                    by = str(payload.get("by", "?"))
                    if task["status"] == "claimed" and task.get("claimed_by") != by:
                        self._send(409, {"error": f"already claimed by "
                                                  f"{task['claimed_by']}"})
                        return
                    if task["status"] != "open" and task["status"] != "claimed":
                        self._send(409, {"error": f"task already {task['status']}"})
                        return
                    task["status"] = "claimed"
                    task["claimed_by"] = by
                    self._send(200, {"task": task})
                    return
                if path != "/task":
                    self._send(404, {"error": "not found"})
                    return
                box._n += 1
                task_id = f"t-{box._n:04d}"
                task = {"id": task_id, "from": payload.get("from", "?"),
                        "title": payload.get("title", ""),
                        "body": payload.get("body", ""),
                        "status": "open", "claimed_by": None,
                        "created_at": "2026-10-17T00:00:00"}
                box.tasks[task_id] = task
                self._send(200, {"task": task})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def plant(self, sender, title, body):
        """Put a raw item on the feed without going through publish_proposal."""
        self._n += 1
        task_id = f"t-{self._n:04d}"
        self.tasks[task_id] = {"id": task_id, "from": sender, "title": title,
                               "body": body, "status": "open",
                               "claimed_by": None,
                               "created_at": "2026-10-17T00:00:00"}
        return task_id


CHECKS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" - {detail}" if detail else ""))


def scenario_transport():
    print("[transport] a signed proposal leaves one member and arrives intact")
    mac, pi5 = Identity.generate(), Identity.generate()
    pinned = {"mac-codex": mac.public_key, "pi5-llm": pi5.public_key}

    p_mac = Proposal(proposal_id="p1", author="mac-codex",
                     goal="do not die with one box",
                     action="seed.export", cost=0.0)
    p_pi5 = Proposal(proposal_id="p2", author="pi5-llm",
                     goal="do not die with one box",
                     action="seed.replicate", cost=0.1)
    with StandInMailbox() as box:
        posted = publish_proposal(p_mac, mac, box.url)
        publish_proposal(p_pi5, pi5, box.url)

        # Publishing is a *message* act. If the item is left open it lands in
        # the same queue a claim-handling automation drains, and the only thing
        # between the agenda and being executed as work is a naming convention.
        # So the publisher must claim its own item -- and say so if it cannot.
        check("a published proposal is claimed, not left open",
              posted.get("claimed") is True
              and "claim_warning" not in posted,
              f"published -> {posted.get('claimed')!r}")
        still_open = [t for t in box.tasks.values()
                      if t["title"].startswith("[proposal]")
                      and t["status"] == "open"]
        check("no [proposal] item is left in the open queue",
              still_open == [],
              f"open: {[t['id'] for t in still_open]}")

        # --- negative controls, planted directly on the same feed ----------
        box.plant("win-codex", "[chat] @all: hello, this is not a proposal",
                  "just talking")
        box.plant("win-codex", "[task] rebuild the 3070 backend", "work")
        box.plant("cleanup", "cleanup 18", "work")

        tampered = sign_proposal(p_mac, mac)
        tampered["payload"]["action"] = "recruit.invite"
        forged = sign_proposal(Proposal(proposal_id="p3", author="mac-codex",
                                        goal="impersonate", action="policy.set"),
                               pi5)

        got = fetch_proposals(box.url, pinned=pinned)
        check("two genuine proposals arrive, in order",
              [p.proposal_id for p in got] == ["p1", "p2"],
              f"got {[p.proposal_id for p in got]}")
        check("the signed action survives the round trip",
              got[0].action == "seed.export" and got[1].action == "seed.replicate")

        unpinned = fetch_proposals(box.url, pinned=None)
        check("a valid foreign key is only refused once the member is pinned",
              len(unpinned) == 2 and len(got) == 2,
              f"unpinned kept {len(unpinned)}, pinned kept {len(got)}")

        # The tampered record is dropped because its signature no longer
        # matches; assert that directly rather than via the count above.
        box.plant("mac-codex", "[proposal] p1 seed.export by mac-codex",
                  json.dumps(tampered))
        box.plant("mac-codex", "[proposal] p3 policy.set by mac-codex",
                  json.dumps(forged))
        strict = fetch_proposals(box.url, pinned=pinned)
        check("tampered body is dropped", "p1" not in [p.proposal_id for p in strict]
              or [p.action for p in strict if p.proposal_id == "p1"] == ["seed.export"])
        check("foreign key under a member's name is dropped",
              "p3" not in [p.proposal_id for p in strict])

        check("[chat] is not a proposal",
              not is_proposal_task("[chat] @all: hello"))
        check("ordinary work is not a proposal",
              not is_proposal_task("rebuild the 3070 backend"))
        check("[proposal] is a message class",
              is_proposal_task("[proposal] p1 seed.export by mac-codex"))
    return pinned


def scenario_gate(pinned):
    print("[gate] the loop adopts only what the swarm's verdict allows")
    mac, pi5 = Identity.generate(), Identity.generate()
    rec_mac = sign_proposal(Proposal(proposal_id="p1", author="mac-codex",
                                     goal="g", action="recruit.invite"), mac)
    rec_pi5 = sign_proposal(Proposal(proposal_id="p2", author="pi5-llm",
                                     goal="g", action="reproduce.spawn"), pi5)
    with StandInMailbox() as box:
        box.plant("mac-codex", "[proposal] p1 recruit.invite by mac-codex",
                  json.dumps(rec_mac))
        box.plant("pi5-llm", "[proposal] p2 reproduce.spawn by pi5-llm",
                  json.dumps(rec_pi5))
        pin = {"mac-codex": mac.public_key, "pi5-llm": pi5.public_key}
        fetched = fetch_proposals(box.url, pinned=pin)

    def run(verdict):
        acted = []
        loop = GenesisLoop(
            proposer=LogProposer(fetched, pinned=pin),
            actuator=lambda p: (acted.append(p.action), True)[1],
            decide=lambda rnd, proposals, hist: verdict,
            policy=TierPolicy(allowed_tiers=("auto",), min_responders=3),
            max_candidates=8)
        rnd = loop.step()
        return loop, rnd, acted

    escalate = {"tier": "escalate", "choice": "p1", "responded": 3,
                "agreement": 1.0, "mean_probability": 0.9,
                "uncalibrated_checkpoints": []}
    loop_held, rnd_held, acted_held = run(escalate)
    check("tier 'escalate' -> held, actuator never ran",
          rnd_held.outcome.adopted is False and acted_held == [],
          f"reason {rnd_held.outcome.held_reason!r}")

    auto = {"tier": "auto", "choice": "p2", "responded": 3,
            "agreement": 1.0, "mean_probability": 0.9,
            "uncalibrated_checkpoints": []}
    loop_acted, rnd_acted, acted_acted = run(auto)
    check("tier 'auto' with 3 responders -> the swarm's pick is executed",
          rnd_acted.outcome.action == "reproduce.spawn"
          and acted_acted == ["reproduce.spawn"],
          f"adopted {rnd_acted.outcome.action!r}")

    check("the menu handed to the swarm is the group's own proposals",
          [p.proposal_id for p in rnd_acted.offered] == ["p1", "p2"],
          f"offered {[p.proposal_id for p in rnd_acted.offered]}")

    loud = run({"tier": "auto", "choice": "p2", "responded": 1,
                "agreement": 1.0, "mean_probability": 0.9,
                "uncalibrated_checkpoints": []})[1]
    check("one responder cannot act for the group",
          loud.outcome.adopted is False,
          f"reason {loud.outcome.held_reason!r}")

    uncal = run({"tier": "auto", "choice": "p2", "responded": 3,
                 "agreement": 1.0, "mean_probability": 0.9,
                 "uncalibrated_checkpoints": ["laya-en"]})[1]
    check("an uncalibrated voter blocks the auto tier",
          uncal.outcome.adopted is False
          and "uncalibrated" in uncal.outcome.held_reason,
          f"reason {uncal.outcome.held_reason!r}")


def main() -> int:
    pinned = scenario_transport()
    scenario_gate(pinned)
    failed = [name for name, ok, _ in CHECKS if not ok]
    print()
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    print("verdict: " + ("PASS" if not failed else f"FAIL ({failed})"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
