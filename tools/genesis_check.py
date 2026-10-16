"""End-to-end drill for the Genesis loop: does the agenda become endogenous?

Round 98/99 measured the gap: the swarm can choose among handed-in options but
cannot author its own agenda. ``pando/genesis.py`` builds the propose -> score
-> adopt -> execute -> remember loop. This drill asks the one question that
matters, and it is deliberately blind to model quality: is the loop's behaviour
a function of its own recorded outcomes?

Two loops, identical in every way — same proposer, same candidate set, same
actuator — with one difference: one remembers, one does not. If they diverge,
the memory is doing the work, and the agenda is a function of the group's own
past rather than of whoever handed it a menu.

Property under test (docs/DESIGN-genesis.md §7): the second round differs from
the first *because of* the first round's outcome. That is the falsifiable core.

Run:  .venv/bin/python tools/genesis_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando.genesis import GenesisLoop, NullScoreBoard, StubProposer

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


def main() -> int:
    print("== Genesis drill: does the agenda become endogenous? ==")
    print(f"candidates (constant every round): {CANDIDATES}")
    print(f"actuator: {FAILING} -> failure ; others -> success")
    print()

    remembered = run(scoreboard=None, rounds=2)          # real memory
    amnesiac = run(scoreboard=NullScoreBoard(), rounds=2)  # control: forgets

    print("-- remembered loop (memory on) --")
    for r in remembered.rounds:
        print(f"round {r.round_no}: adopted={r.outcome.action:<18} "
              f"success={r.outcome.success}")
    print(f"scoreboard: {remembered.summary()['scoreboard']}")
    print()

    print("-- amnesiac control (memory off) --")
    for r in amnesiac.rounds:
        print(f"round {r.round_no}: adopted={r.outcome.action:<18} "
              f"success={r.outcome.success}")
    print(f"scoreboard: {amnesiac.summary()['scoreboard']}")
    print()

    r1, r2 = remembered.rounds[0], remembered.rounds[1]
    a1, a2 = amnesiac.rounds[0], amnesiac.rounds[1]

    checks: list[tuple[str, bool]] = [
        ("candidate action set identical both rounds (proposer held constant)",
         [p.action for p in r1.proposals] == [p.action for p in r2.proposals]),
        ("round 1 adopts the alphabetical-first action and it fails",
         r1.outcome.action == FAILING and r1.outcome.success is False),
        ("remembered round 2 differs from round 1 (memory moved the choice)",
         r2.outcome.action != r1.outcome.action),
        ("amnesiac round 2 repeats round 1 (no memory -> same choice)",
         a2.outcome.action == a1.outcome.action),
        ("the difference is caused by the recorded outcome alone",
         remembered.summary()["adopted"] != amnesiac.summary()["adopted"]),
    ]

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
