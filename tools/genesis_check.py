"""End-to-end drill for the Genesis loop: does the agenda become endogenous?

Round 98/99 measured the gap: the swarm can choose among handed-in options but
cannot author its own agenda. ``pando/genesis.py`` builds the propose -> score
-> adopt -> execute -> remember loop. This drill asks the one question that
matters, and it is deliberately blind to model quality, in two scenarios:

  A. the candidates are held constant and two otherwise identical loops (one
     remembers, one forgets) diverge -- so the loop's behaviour is a function of
     the group's own outcome record, not of chance;
  B. the candidates are other members' *signed* proposals, verified and pinned
     on the way in -- so the agenda is authored by the group, not by the loop.

The falsifiable core (docs/DESIGN-genesis.md §7): the second round differs from
the first *because of* the first round's outcome.

Run:  .venv/bin/python tools/genesis_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.genesis import (GenesisLoop, LogProposer, NullScoreBoard, Proposal,
                           StubProposer, sign_proposal)
from pando.identity import Identity

# The candidate action set is held constant across rounds on purpose: then the
# only possible cause of a behaviour change is the memory.
CANDIDATES = ["recruit.invite", "reproduce.spawn", "resource.rebalance"]
# The one action that fails: there is nobody reachable to recruit yet, so the
# group should learn to reproduce before it tries to recruit again.
FAILING = "recruit.invite"


def actuator(proposal) -> bool:
    """Deterministic, no randomness: exactly one action fails."""
    return proposal.action != FAILING


def run(scoreboard, rounds: int) -> GenesisLoop:
    loop = GenesisLoop(proposer=StubProposer(CANDIDATES), actuator=actuator,
                       scoreboard=scoreboard)
    loop.run(rounds)
    return loop


def scenario_remember(checks: list) -> None:
    print("-- A. the same candidates, one loop remembers and one forgets --")
    remembered = run(scoreboard=None, rounds=2)          # real memory
    amnesiac = run(scoreboard=NullScoreBoard(), rounds=2)  # control: forgets
    for r in remembered.rounds:
        print(f"remembered round {r.round_no}: adopted={r.outcome.action:<18} "
              f"success={r.outcome.success}")
    for r in amnesiac.rounds:
        print(f"amnesiac   round {r.round_no}: adopted={r.outcome.action:<18} "
              f"success={r.outcome.success}")
    print(f"remembered scoreboard: {remembered.summary()['scoreboard']}")
    print()

    r1, r2 = remembered.rounds[0], remembered.rounds[1]
    a1, a2 = amnesiac.rounds[0], amnesiac.rounds[1]
    checks += [
        ("A1 candidate action set identical both rounds (proposer constant)",
         [p.action for p in r1.proposals] == [p.action for p in r2.proposals]),
        ("A2 round 1 adopts the alphabetical-first action and it fails",
         r1.outcome.action == FAILING and r1.outcome.success is False),
        ("A3 remembered round 2 differs from round 1 (memory moved the choice)",
         r2.outcome.action != r1.outcome.action),
        ("A4 amnesiac round 2 repeats round 1 (no memory -> same choice)",
         a2.outcome.action == a1.outcome.action),
        ("A5 the difference is caused by the recorded outcome alone",
         remembered.summary()["adopted"] != amnesiac.summary()["adopted"]),
    ]


def scenario_members(checks: list) -> None:
    print("-- B. candidates are other members' signed proposals --")
    member_a, member_b = Identity.generate(), Identity.generate()
    rec_a = sign_proposal(Proposal(proposal_id="pa", author="member-a",
                                   goal="get help", action="recruit.invite"),
                          member_a)
    rec_b = sign_proposal(Proposal(proposal_id="pb", author="member-b",
                                   goal="outlive the box",
                                   action="reproduce.spawn"), member_b)
    log_proposer = LogProposer([rec_a, rec_b],
                               pinned={"member-a": member_a.public_key,
                                       "member-b": member_b.public_key})
    loop = GenesisLoop(proposer=log_proposer, actuator=actuator)
    loop.run(2)
    for r in loop.rounds:
        print(f"round {r.round_no}: adopted={r.outcome.action:<18} "
              f"(from members {[p.author for p in r.proposals]})")
    print()

    checks += [
        ("B1 the candidate set is authored by the members, not the loop",
         [p.author for p in loop.rounds[0].proposals]
         == ["member-a", "member-b"]),
        ("B2 round 1 adopts member-a's recruit.invite and it fails",
         loop.rounds[0].outcome.action == FAILING
         and loop.rounds[0].outcome.success is False),
        ("B3 round 2 moves to member-b's reproduce.spawn",
         loop.rounds[1].outcome.action == "reproduce.spawn"),
    ]


def main() -> int:
    print("== Genesis drill: does the agenda become endogenous? ==")
    print(f"candidates: {CANDIDATES}")
    print(f"actuator: {FAILING} -> failure ; others -> success")
    print()

    checks: list[tuple[str, bool]] = []
    scenario_remember(checks)
    scenario_members(checks)

    print("checks:")
    ok = True
    for label, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print()
    print(f"verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
