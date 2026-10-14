"""Retirement: the half of elasticity that removes a device.

Reproduction grows a swarm (rounds 105-108) and the adaptive quorum decides
how much of it to use per decision (rounds 109-110). Nothing removed a
member: a slow, idle, expensive device stayed a voter forever, and because a
stem re-registers every few seconds, deleting it from the registry would
last five seconds.

So retirement is a *decision* plus a *durable* effect: the root blocks the
node, and refusing its re-registration is what makes it stick. Removing a
device is sensitive, so it goes through the same grant ledger and journal as
reproduction.
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
