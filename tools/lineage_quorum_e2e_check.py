"""Does the lineage rule hold over the wire?

Round 121 added `count_lineages` and verified it by replaying recorded
votes. Replay cannot show that the flag reaches the coordinator, that the
quorum picks the stems it thinks it picked, or that the answer tells an
operator which opinions stood behind it. This starts real roots on loopback,
registers real HTTP stems into them, and asks the same question of each:

  flag off, three stems (english, english, typed-decisions), 2:1 split
      -> a stem majority, tier auto
  flag on, the same three stems
      -> a 1:1 lineage tie: tier verify, never auto (round 127 lets a tied
         quorum return the confidence-ranked choice rather than silence, so
         the property under test is the tier, not the absence of a choice)
  flag on, three english stems that agree
      -> the vacuity this rule exposes: agreement 1.0 from one lineage
  flag on + [gating] corroboration_floor = 2, three english stems
      -> the policy answer: tier verify, because one opinion corroborated
         nothing
  blocs, two differently-named stems declared one opinion
      -> the round-129 rule: the group votes once, so a 2:1 stem majority
         between the group and a distinct lineage is a 1:1 opinion tie

Usage:
    python tools/lineage_quorum_e2e_check.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.backend import MockBackend  # noqa: E402
from pandosi.common import DecisionRequest  # noqa: E402
from pandosi.rootd import Coordinator, Root  # noqa: E402
from pandosi.stemd import Stem  # noqa: E402

QUESTION = "intent"
CANDIDATES = ["approve", "reject"]


def ask(root: Root, state: str) -> dict:
    request = DecisionRequest(state=state, question=QUESTION,
                              candidates=CANDIDATES)
    with urllib.request.urlopen(
            urllib.request.Request(f"{root.url()}/decide",
                                   data=request.to_json(),
                                   headers={"Content-Type": "application/json"}),
            timeout=15) as response:
        return json.loads(response.read())


def start_stem(name: str, checkpoint: str, choice: str) -> Stem:
    stem = Stem(name, MockBackend(name, checkpoint, correct_choice=choice,
                                  agree_rate=1.0))
    stem.start()
    return stem


def scenario(label: str, members: list[Stem], count_lineages: bool,
             state: str, gating: dict | None = None,
             blocs: dict | None = None) -> dict:
    root = Root(coordinator=Coordinator(count_lineages=count_lineages,
                                        gating=gating, blocs=blocs))
    root.start()
    try:
        for stem in members:
            root.coordinator.register(stem.node_id, stem.url(),
                                      stem.backend.checkpoint)
        result = ask(root, state)
    finally:
        root.stop()
    print(f"  {label:34s} tier={result['tier']:8s} "
          f"choice={str(result['choice']):8s} "
          f"agreement={result['agreement']:.2f} "
          f"stems={result['quorum_size']} "
          f"lineages={result.get('lineages', '-')}")
    return result


def main() -> int:
    stems = [
        start_stem("en-1", "english", "approve"),
        start_stem("en-2", "english", "approve"),
        start_stem("en-3", "english", "approve"),
        start_stem("td-1", "typed-decisions", "reject"),
    ]
    twin = start_stem("en-twin", "en-copy", "approve")
    mixed = [stems[0], stems[1], stems[3]]
    replicas = [stems[0], stems[1], stems[2]]
    try:
        print("paired scenarios over real HTTP:")
        plain = scenario("stems, flag off (2:1 split)", mixed, False, "s-a")
        lineage = scenario("stems, flag on (same quorum)", mixed, True, "s-a")
        single = scenario("replicas, flag on (3x english)", replicas, True,
                          "s-b")
        corroborated = scenario(
            "replicas, flag on, floor 2", replicas, True, "s-b",
            {"corroboration_floor": 2})
        bloc = scenario("english + en-copy, declared one bloc",
                        [stems[0], twin, stems[3]], False, "s-a",
                        None, {"en-copy": "english"})
    finally:
        for stem in stems + [twin]:
            stem.stop()

    failures = []
    if plain["tier"] != "auto" or plain["choice"] != "approve":
        failures.append("flag off should follow the 2:1 stem majority")
    if lineage["tier"] != "verify":
        failures.append("a lineage tie must not auto-execute")
    if lineage.get("lineages") != 2 or plain.get("lineages") is not None:
        failures.append("only the flagged root should report lineages")
    if single["tier"] != "auto" or single.get("lineages") != 1:
        failures.append("a single-lineage quorum should be visible as such")
    if corroborated["tier"] != "verify":
        failures.append("corroboration_floor=2 should stop a lone lineage "
                        "from auto-executing")
    if bloc["tier"] != "verify" or bloc.get("blocs") != 2:
        failures.append("two names declared one bloc must vote once, and a "
                        "group against a distinct lineage is a tie")

    print(f"\none lineage still decides by default (tier={single['tier']} "
          f"with lineages={single.get('lineages')}), because a single device "
          f"has to work; a playbook that wants corroboration sets "
          f"corroboration_floor and the same quorum lands on "
          f"{corroborated['tier']}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("LINEAGE QUORUM E2E OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
