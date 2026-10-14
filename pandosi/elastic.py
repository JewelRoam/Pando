"""Elasticity: growing, shrinking, and deciding which.

Reproduction grows a swarm (rounds 105-108) and the adaptive quorum decides
how much of it to use per decision (rounds 109-110). Nothing removed a
member: a slow, idle, expensive device stayed a voter forever, and because a
stem re-registers every few seconds, deleting it from the registry would
last five seconds.

So retirement is a *decision* plus a *durable* effect: the root blocks the
node, and refusing its re-registration is what makes it stick. Removing a
device is sensitive, so it goes through the same grant ledger and journal as
reproduction.

`ElasticSupervisor` closes the loop: it reads the root's own signals and says
when to grow, when to retire, and - more often - when to leave the swarm
alone, because a slow decision at the quorum floor is a capacity problem
while a slow decision with quorum left to spend is a policy question.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .autonomy import Outcome, Journal
from .common import bypass_proxy_for_lan
from .grants import require_grant


@dataclass
class Member:
    """What the root knows about one member when retirement is considered."""

    node_id: str
    checkpoint: str
    latency_ewma_ms: float | None
    decisions: int = 0


def choose_retirement(members: list[Member], keep: int,
                      max_latency_ms: float | None = None,
                      min_decisions: int = 0) -> tuple[str | None,
                                                       list[str]]:
    """Pick one member to retire, or explain why none should be.

    Two rules, in order: never go below ``keep`` members, and never retire a
    member that has not been exercised. A device that is slow but is the only
    one answering is the swarm, not a problem; a device that is slow while
    others are healthy is capacity the swarm is paying for and not using.
    """
    refusals: list[str] = []
    if len(members) <= keep:
        refusals.append(f"{len(members)} members, keep={keep}: nothing to "
                        "spare")
        return None, refusals
    if max_latency_ms is None:
        refusals.append("no latency threshold configured")
        return None, refusals

    candidates = [member for member in members
                  if member.latency_ewma_ms is not None
                  and member.latency_ewma_ms > max_latency_ms]
    if not candidates:
        refusals.append(f"no member above {max_latency_ms:.0f} ms")
        return None, refusals
    candidates = [member for member in candidates
                  if member.decisions >= min_decisions]
    if not candidates:
        refusals.append(f"no slow member has served {min_decisions} "
                        "decisions yet")
        return None, refusals
    slowest = max(candidates, key=lambda member: member.latency_ewma_ms)
    return slowest.node_id, refusals


class Retirer:
    """Ask a live root to retire a member, through the grant ledger."""

    def __init__(self, root_url: str, pairing_token: str | None = None,
                 grants_path=None, journal: Journal | None = None,
                 timeout: float = 30):
        bypass_proxy_for_lan()
        urllib.request.install_opener(
            urllib.request.build_opener(urllib.request.ProxyHandler({})))
        self.root_url = root_url.rstrip("/")
        self.pairing_token = pairing_token
        self.grants_path = grants_path
        self.journal = journal
        self.timeout = timeout

    def members(self) -> list[Member]:
        headers = {"Authorization": f"Bearer {self.pairing_token}"} \
            if self.pairing_token else {}
        request = urllib.request.Request(f"{self.root_url}/nodes",
                                         headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        return [Member(node_id, info.get("checkpoint", "?"),
                       info.get("stats", {}).get("latency_ewma_ms"),
                       int(info.get("stats", {}).get("successes", 0)))
                for node_id, info in payload.get("nodes", {}).items()]

    def retire(self, node_id: str, reason: str) -> Outcome:
        outcome = Outcome(observation_id=node_id, tier="retire",
                          acted=False, choice=None, agreement=0.0,
                          mean_probability=0.0, responded=0, quorum_size=0,
                          reason=reason)
        try:
            require_grant("retire", path=self.grants_path)
        except PermissionError as exc:
            outcome.reason = str(exc)
            if self.journal is not None:
                self.journal.append(outcome)
            return outcome

        body = json.dumps({"node_id": node_id, "reason": reason}).encode()
        headers = {"Content-Type": "application/json"}
        if self.pairing_token:
            headers["Authorization"] = f"Bearer {self.pairing_token}"
        request = urllib.request.Request(f"{self.root_url}/retire", data=body,
                                         headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
            outcome.acted = bool(payload.get("ok"))
            outcome.reason = (f"retired {node_id}: {reason}; "
                              f"blocked={payload.get('blocked')}")
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            outcome.reason = f"retire failed: {exc}"
        if self.journal is not None:
            self.journal.append(outcome)
        return outcome

    def sweep(self, keep: int, max_latency_ms: float,
              min_decisions: int = 0) -> tuple[list[Outcome], list[str]]:
        """Retire at most one member, and say why nothing else was retired."""
        node_id, refusals = choose_retirement(
            self.members(), keep=keep, max_latency_ms=max_latency_ms,
            min_decisions=min_decisions)
        if node_id is None:
            return [], refusals
        reason = (f"slowest voter above {max_latency_ms:.0f} ms while "
                  f"{len(self.members())} members are live")
        return [self.retire(node_id, reason)], refusals
@dataclass
class Supervision:
    """What the supervisor decided, and why."""

    action: str                      # grow | retire | hold
    reason: str
    target: str | None = None
    latency_ms: float | None = None
    quorum_k: int | None = None
    members: int = 0


class ElasticSupervisor:
    """Decide whether the swarm should grow, shrink, or leave itself alone.

    The loop this closes: reproduction can add a device (rounds 105-108), the
    adaptive quorum can spend checking power for throughput (round 110), and
    retirement can remove a device (round 111) - but nothing decided *when*.

    Growth is only proposed when the swarm is out of room on the other two
    dials: decisions are slow, the quorum is already at its floor, and there
    are members to spare below the ceiling. Anything else is a request to
    spend checking power, which is a policy question, not a capacity one.
    """

    def __init__(self, root_url: str, pairing_token: str | None = None,
                 max_members: int = 5, keep: int = 2,
                 max_member_latency_ms: float = 250.0,
                 pressure_factor: float = 1.1, timeout: float = 30):
        self.root_url = root_url.rstrip("/")
        self.pairing_token = pairing_token
        self.max_members = max_members
        self.keep = keep
        self.max_member_latency_ms = max_member_latency_ms
        self.pressure_factor = pressure_factor
        self.timeout = timeout

    def _get(self, path: str) -> dict:
        headers = {"Authorization": f"Bearer {self.pairing_token}"} \
            if self.pairing_token else {}
        request = urllib.request.Request(f"{self.root_url}{path}",
                                         headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def status(self) -> dict:
        return self._get("/status")

    def decide(self, status: dict | None = None,
               retired: list[str] | None = None) -> Supervision:
        """One decision. ``status`` is injectable so the policy is testable."""
        status = status or self.status()
        members = int(status.get("members", 0))
        quorum_k = status.get("quorum_k")
        floor = status.get("quorum_floor")
        target = status.get("latency_target_ms")
        latency = status.get("decision_ewma_ms")
        adaptive = bool(status.get("adaptive"))

        if retired:
            return Supervision("retire", retired[0], target=retired[1],
                               latency_ms=latency, quorum_k=quorum_k,
                               members=members)
        if latency is None or target is None:
            return Supervision("hold", "no latency signal yet",
                               latency_ms=latency, quorum_k=quorum_k,
                               members=members)
        if latency <= target * self.pressure_factor:
            return Supervision("hold",
                               f"{latency:.0f} ms within {target:.0f} ms "
                               f"target (x{self.pressure_factor})",
                               latency_ms=latency, quorum_k=quorum_k,
                               members=members)
        if adaptive and floor is not None and quorum_k is not None \
                and quorum_k > floor:
            return Supervision(
                "hold",
                f"{latency:.0f} ms is slow but the quorum can still shrink "
                f"({quorum_k} > floor {floor})",
                latency_ms=latency, quorum_k=quorum_k, members=members)
        if members >= self.max_members:
            return Supervision(
                "hold",
                f"{latency:.0f} ms and at the member ceiling "
                f"({members}/{self.max_members}); this is a "
                "per-decision-cost problem, not a capacity one",
                latency_ms=latency, quorum_k=quorum_k, members=members)
        return Supervision(
            "grow",
            f"{latency:.0f} ms > {target * self.pressure_factor:.0f} ms and "
            f"the quorum is at its floor ({quorum_k}) with "
            f"{members}/{self.max_members} members",
            latency_ms=latency, quorum_k=quorum_k, members=members)

    def step(self, grow=None, retire=None,
             journal: Journal | None = None) -> Supervision:
        """Decide, then execute. ``grow``/``retire`` are caller callbacks."""
        members = []
        try:
            members = Retirer(self.root_url, self.pairing_token,
                              timeout=self.timeout).members()
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        slow = [member for member in members
                if member.latency_ewma_ms is not None
                and member.latency_ewma_ms > self.max_member_latency_ms]
        if len(members) > self.keep and slow:
            slowest = max(slow, key=lambda member: member.latency_ewma_ms)
            supervision = Supervision(
                "retire",
                f"{slowest.node_id} at {slowest.latency_ewma_ms:.0f} ms "
                f"while {len(members)} members are live",
                target=slowest.node_id,
                latency_ms=slowest.latency_ewma_ms, members=len(members))
        else:
            supervision = self.decide()
        if supervision.action == "grow" and grow is not None:
            supervision.reason += f"; grow -> {grow()}"
        elif supervision.action == "retire" and retire is not None:
            outcome = retire(supervision.target, supervision.reason)
            supervision.reason = (outcome.reason if outcome else
                                  supervision.reason)
        if journal is not None:
            journal.append(Outcome(
                observation_id=supervision.target or "swarm",
                tier=supervision.action, acted=supervision.action != "hold",
                choice=None, agreement=0.0, mean_probability=0.0,
                responded=supervision.members,
                quorum_size=supervision.quorum_k or 0,
                reason=supervision.reason))
        return supervision
