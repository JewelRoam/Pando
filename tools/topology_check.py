"""Scenario checks for pandosi.topology against this project's measurements.

Every scenario encodes a measurement from docs/research so the planner
cannot silently drift away from the evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pandosi.topology import Device, Link, plan_topology  # noqa: E402


def show(title: str, plan, expect: str) -> None:
    assert plan.mode == expect, f"{title}: {plan.mode} != {expect} ({plan.reason})"
    print(f"{title}\n  mode={plan.mode} latency={plan.latency_ms} "
          f"throughput={plan.throughput_rps}\n  {plan.reason}")


def main() -> int:
    # Round 94 cross-device: one 3070 (13.7 ms whole model) plus an Intel CPU
    # (106 ms). The model fits everywhere, so replication wins and staging is
    # explicitly rejected.
    fast = Device("rtx3070", full_model_ms=13.7, memory_gb=8)
    slow = Device("intel-cpu", full_model_ms=106.0, memory_gb=64)
    loopback = Link("rtx3070", "intel-cpu", rtt_ms=1.2)
    plan = plan_topology([fast, slow], [loopback], model_gb=2.0,
                         activation_mb=0.4, quorum=1)
    show("one fast GPU + slow CPU, quorum 1 (round 94)", plan, "replicate")
    assert plan.voters == ["rtx3070"], plan.voters
    assert plan.rejected, "staging rejection should be recorded"

    # Requiring a quorum the fast device cannot supply pulls the slow one in.
    # Replicating then wastes the fast device (it waits for the slow voter),
    # so the planner gives each device a slice proportional to its speed
    # instead. This is the fix for round 94's 50/50 node split, which gave
    # the slow device far more work than its speed justified.
    plan = plan_topology([fast, slow], [loopback], model_gb=2.0,
                         activation_mb=0.4, quorum=2)
    show("fast GPU + slow CPU, quorum 2 (speed-proportional split)",
         plan, "stage")
    shares = {stage["device"]: stage["share"] for stage in plan.stages}
    assert shares["rtx3070"] > 0.8 > shares["intel-cpu"], shares
    stage_times = {stage["stage_ms"] for stage in plan.stages}
    assert len(stage_times) == 1, plan.stages  # both stages finish together
    assert plan.latency_ms == round(2 * stage_times.pop() + 1.2, 3), plan

    # Two comparable, disjoint GPUs: a 50/50 split halves the critical path,
    # so staging doubles quorum throughput. (Predicted, not yet measured on
    # this host - it has one GPU. Round 94 measured the shared-CPU case.)
    twin_a = Device("gpu-a", full_model_ms=20.0, memory_gb=8)
    twin_b = Device("gpu-b", full_model_ms=20.0, memory_gb=8)
    twin_link = Link("gpu-a", "gpu-b", rtt_ms=1.0)
    plan = plan_topology([twin_a, twin_b], [twin_link], model_gb=2.0,
                         activation_mb=0.4)
    show("two equal disjoint GPUs", plan, "stage")
    assert plan.throughput_rps == 100.0, plan.throughput_rps

    # Model does not fit on one device: the pipeline becomes the only option,
    # and its shares follow measured device speed. 6 GB of model, 4 GB + 3 GB
    # of memory, so the 0.67 share fits the faster box exactly.
    small_a = Device("box-a", full_model_ms=200.0, memory_gb=4)
    small_b = Device("box-b", full_model_ms=400.0, memory_gb=3)
    big_link = Link("box-a", "box-b", rtt_ms=1.0)
    plan = plan_topology([small_a, small_b], [big_link], model_gb=6.0,
                         activation_mb=0.5)
    show("model fits no single device", plan, "stage")
    shares = {stage["device"]: stage["share"] for stage in plan.stages}
    assert shares["box-a"] > shares["box-b"], shares
    assert abs(sum(shares.values()) - 1.0) < 1e-6, shares

    # Same model on two 1 GB boxes: no share fits, so the pipeline is refused
    # for memory before the link is even considered.
    tiny_a = Device("box-a", full_model_ms=200.0, memory_gb=1)
    tiny_b = Device("box-b", full_model_ms=400.0, memory_gb=1)
    plan = plan_topology([tiny_a, tiny_b], [big_link], model_gb=6.0,
                         activation_mb=0.5)
    show("model fits no stage either", plan, "infeasible")
    assert any("GB" in problem for problem in plan.rejected), plan.rejected
    assert len(plan.rejected) == 2, plan.rejected

    # Same model, but the only link is a slow one: L < T fails, so the
    # planner refuses rather than building a pipeline that would stall.
    wan = Link("box-a", "box-b", rtt_ms=200.0)
    plan = plan_topology([small_a, small_b], [wan], model_gb=6.0,
                         activation_mb=0.5)
    show("model too big, slow link", plan, "infeasible")
    assert plan.rejected

    print(json.dumps({"scenarios": 6, "status": "OK"}))
    print("TOPOLOGY PLANNER OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
