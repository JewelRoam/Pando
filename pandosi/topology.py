"""Topology planning: replicate or stage, from measured device and link data.

The rules here are the ones this project measured, not assumptions:

- round 93: a pipeline pays only while the inter-stage link is cheaper than
  a stage's compute (``L < T``);
- round 94: that is necessary but not sufficient - on one shared CPU
  staging bought nothing (0.95-1.13x), and offloading half the model to a
  slower device measured 0.47x. Stages must also occupy disjoint compute
  resources, and the work must be split in proportion to device speed;
- round 95: the runtime does not change a vote (27/27 agreement), so
  replicas on different ecosystems are interchangeable.

The planner therefore defaults to replicate-and-vote and only chooses a
pipeline when replication cannot cover the model or when a pipeline
actually beats replication on the measured numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Device:
    """One measured worker."""

    node_id: str
    full_model_ms: float          # measured whole-model forward latency
    memory_gb: float
    throughput_scale: float = 1.0  # relative speed for splitting work


@dataclass
class Link:
    """One measured transport between two devices."""

    a: str
    b: str
    rtt_ms: float
    mbps: float = 1000.0

    def connects(self, first: str, second: str) -> bool:
        return {self.a, self.b} == {first, second}


@dataclass
class Plan:
    mode: str                     # replicate | stage | infeasible
    reason: str
    voters: list[str] = field(default_factory=list)
    stages: list[dict] = field(default_factory=list)
    latency_ms: float | None = None
    throughput_rps: float | None = None
    rejected: list[str] = field(default_factory=list)


def _rtt(links: list[Link], first: str, second: str) -> float:
    if first == second:
        return 0.0
    for link in links:
        if link.connects(first, second):
            return link.rtt_ms
    return float("inf")            # unknown link: assume it cannot be used


def _replicate_plan(devices: list[Device], links: list[Link],
                    quorum: int) -> Plan:
    """Each voter runs the whole model; the slowest voter bounds the round."""
    usable = sorted(devices, key=lambda d: d.full_model_ms)
    voters = usable[:max(1, quorum)]
    slowest = max(voters, key=lambda d: d.full_model_ms)
    return Plan(
        mode="replicate",
        reason=(f"{len(voters)} devices run the whole model; the slowest "
                f"({slowest.node_id}) bounds the round. Runtime differences "
                f"do not change votes (round 95)"),
        voters=[device.node_id for device in voters],
        latency_ms=round(slowest.full_model_ms, 3),
        throughput_rps=round(1000.0 / slowest.full_model_ms, 3),
    )


def _stage_plan(devices: list[Device], links: list[Link],
                activation_mb: float, max_stages: int,
                model_gb: float) -> Plan:
    """Split work across devices in proportion to their measured speed."""
    ranked = sorted(devices, key=lambda d: d.full_model_ms)
    chain = ranked[:max(2, max_stages)]
    total_scale = sum(1.0 / device.full_model_ms for device in chain)

    # A node's share of the model is its share of the cluster's speed, so
    # that every stage finishes at about the same time. A device's own
    # whole-model latency scales by its share, which is what makes the
    # split proportional to measured speed rather than to node count.
    stages = []
    for device in chain:
        share = (1.0 / device.full_model_ms) / total_scale
        stages.append({"device": device.node_id, "share": round(share, 4),
                       "stage_ms": round(device.full_model_ms * share, 3),
                       "_share": share})

    problems = []
    for stage in stages:
        # Use the exact share: a rounded 0.6667 must not fail a 4.00 GB fit.
        needed = stage["_share"] * model_gb
        device = next(d for d in chain if d.node_id == stage["device"])
        if needed > device.memory_gb + 1e-9:
            problems.append(
                f"{device.node_id} would hold {needed:.2f} GB "
                f"(share {stage['_share']:.2f}) but has "
                f"{device.memory_gb} GB")
    for index in range(len(stages) - 1):
        first, second = stages[index], stages[index + 1]
        rtt = _rtt(links, first["device"], second["device"])
        transfer_ms = 0.0
        for link in links:
            if link.connects(first["device"], second["device"]):
                transfer_ms = activation_mb * 8.0 / max(link.mbps, 0.001)
        budget = min(first["stage_ms"], second["stage_ms"])
        if rtt + transfer_ms >= budget:
            problems.append(
                f"link {first['device']}->{second['device']} costs "
                f"{rtt + transfer_ms:.2f} ms >= the {budget:.2f} ms stage "
                f"budget (L < T fails)")
    if problems:
        return Plan(mode="infeasible",
                    reason="no pipeline: " + "; ".join(problems),
                    stages=stages, rejected=problems)

    for stage in stages:
        stage.pop("_share")

    bottleneck = max(stage["stage_ms"] for stage in stages)
    latency = sum(stage["stage_ms"] for stage in stages) + sum(
        _rtt(links, stages[index]["device"], stages[index + 1]["device"])
        for index in range(len(stages) - 1))
    return Plan(mode="stage",
                reason=("split in proportion to device speed; balance "
                        f"{min(s['stage_ms'] for s in stages) / bottleneck:.2f}"),
                stages=stages,
                latency_ms=round(latency, 3),
                throughput_rps=round(1000.0 / bottleneck, 3))


def plan_topology(devices: list[Device], links: list[Link],
                  model_gb: float, activation_mb: float = 1.0,
                  quorum: int = 3, max_stages: int = 3,
                  min_gain: float = 1.15) -> Plan:
    """Choose replicate or stage from measured numbers.

    Replication is the default because it needs no assumption about links
    and never makes one device the bottleneck for another. A pipeline is
    only chosen when no single device can hold the model, or when a split
    beats replication by ``min_gain`` on the measured throughput.
    """
    if not devices:
        return Plan(mode="infeasible", reason="no devices")

    holds_model = [device for device in devices
                   if device.memory_gb >= model_gb]
    replicate = _replicate_plan(holds_model or devices, links, quorum)
    if not holds_model:
        staged = _stage_plan(devices, links, activation_mb, max_stages, model_gb)
        if staged.mode == "stage":
            staged.reason = ("no device holds the model "
                             f"({model_gb} GB); " + staged.reason)
            return staged
        return Plan(mode="infeasible",
                    reason=("no device holds the model and no feasible "
                            "pipeline exists: " + staged.reason),
                    rejected=staged.rejected)

    if len(devices) < 2:
        return replicate
    staged = _stage_plan(devices, links, activation_mb, max_stages, model_gb)
    if staged.mode != "stage":
        replicate.rejected = staged.rejected
        return replicate
    gain = staged.throughput_rps / max(replicate.throughput_rps, 1e-9)
    if gain < min_gain:
        replicate.rejected = [
            f"staging predicted {staged.throughput_rps} req/s vs "
            f"{replicate.throughput_rps} req/s replicated "
            f"({gain:.2f}x < {min_gain:.2f}x)"]
        return replicate
    staged.reason += (f"; beats replication {gain:.2f}x "
                      f"({staged.throughput_rps} vs "
                      f"{replicate.throughput_rps} req/s)")
    return staged
