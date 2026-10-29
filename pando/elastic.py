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
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .autonomy import Outcome, Journal
from .grants import require_grant
from .net import open_url


@dataclass
class Member:
    """What the root knows about one member when retirement is considered."""

    node_id: str
    checkpoint: str
    latency_ewma_ms: float | None
    decisions: int = 0
    host: str = "?"


def co_located(members: list[Member], host: str | None,
               checkpoint: str | None) -> list[Member]:
    """Members that would share a machine *and* a lineage with a candidate.

    A replica of the same checkpoint on the same host adds compute
    contention and no information: round 95 measured that the runtime and
    replica do not change a vote, and round 114 measured that a second
    same-lineage worker on one host took p50 from 138.7 ms to 981 ms. So this
    is the signal growth has to consult before it acts.
    """
    if not host or not checkpoint:
        return []
    return [member for member in members
            if member.host == host and member.checkpoint == checkpoint]


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
        with open_url(request, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        return [Member(node_id, info.get("checkpoint", "?"),
                       info.get("stats", {}).get("latency_ewma_ms"),
                       int(info.get("stats", {}).get("successes", 0)),
                       info.get("host", "?"))
                for node_id, info in payload.get("nodes", {}).items()]

    def retire(self, node_id: str, reason: str) -> Outcome:
        return self.retire_with_grant(node_id, reason, "retire")

    def retire_with_grant(self, node_id: str, reason: str,
                          grant_action: str = "retire") -> Outcome:
        """Retire a node under a named grant.

        A rollback passes ``reproduce``: the grant that let the swarm grow is
        the grant that lets it undo that same growth, so the ledger entry
        names the authority instead of a fresh approval being invented
        (round 115).
        """
        outcome = Outcome(observation_id=node_id, tier="retire",
                          acted=False, choice=None, agreement=0.0,
                          mean_probability=0.0, responded=0, quorum_size=0,
                          reason=reason)
        try:
            require_grant(grant_action, path=self.grants_path)
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
            with open_url(request, timeout=self.timeout) as resp:
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


def rollback_allowed(target: str, members: list[Member],
                     keep: int) -> tuple[bool, str]:
    """Is undoing a growth safe, or has the new member become load-bearing?

    Rolling back is the point of the bet, so this does not ask whether the
    node is fast - a harmful growth can add a *fast* member that makes
    everyone else slower. It asks whether the swarm can still serve without
    it: not below ``keep``, and not removing the only member that has
    answered anything (round 116).
    """
    remaining = [member for member in members if member.node_id != target]
    if len(remaining) < keep:
        return False, (f"only {len(remaining)} member(s) would remain, below "
                       f"keep={keep}")
    node = next((member for member in members
                 if member.node_id == target), None)
    if node is None:
        return False, f"{target} is no longer a member"
    serving = [member for member in remaining if member.decisions > 0]
    if node.decisions > 0 and not serving:
        return False, (f"{target} has served {node.decisions} decisions and "
                       "no other member has served any")
    return True, "the swarm can serve without this member"


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
                 max_per_host: int = 1,
                 pressure_factor: float = 1.1, cooldown_s: float = 30.0,
                 rollback: bool = True, rollback_factor: float = 1.2,
                 rollback_samples: int = 3, rollback_window_s: float = 60.0,
                 bad_candidate_ttl_s: float = 1800.0,
                 clock=None, timeout: float = 30):
        self.root_url = root_url.rstrip("/")
        self.pairing_token = pairing_token
        self.max_members = max_members
        self.keep = keep
        self.max_member_latency_ms = max_member_latency_ms
        self.max_per_host = max(1, int(max_per_host))
        self.pressure_factor = pressure_factor
        self.timeout = timeout
        # A change takes time to have an effect - a reproduced node has to
        # transfer an artifact, start and register before it can vote - so
        # deciding again immediately would stack changes on top of a change
        # whose result is not in yet (round 114).
        self.cooldown_s = cooldown_s
        self._clock = clock or time.monotonic
        self.last_action: Supervision | None = None
        self.last_action_at: float | None = None
        self.actions: list[tuple[float, str, str | None]] = []
        # A growth is a bet. If it loses, the loop should undo its own change
        # rather than wait for an operator to notice (round 115).
        self.rollback = rollback
        self.rollback_factor = rollback_factor
        self.rollback_samples = max(1, int(rollback_samples))
        self.rollback_window_s = rollback_window_s
        # A rollback is evidence. Without remembering it, the loop re-proposes
        # the change that just failed as soon as the trigger fires again - it
        # oscillated grow/hurt/rollback/grow in one run (round 116).
        self.bad_candidate_ttl_s = bad_candidate_ttl_s
        self.rolled_back: dict[tuple[str, str], dict] = {}
        self._growth_bet: dict | None = None

    def cooling_down(self) -> float:
        """Seconds left before the last change is allowed to have landed."""
        if self.last_action is None or self.last_action_at is None:
            return 0.0
        if self.last_action.action == "hold":
            return 0.0
        return max(0.0, self.cooldown_s
                   - (self._clock() - self.last_action_at))

    def _get(self, path: str) -> dict:
        headers = {"Authorization": f"Bearer {self.pairing_token}"} \
            if self.pairing_token else {}
        request = urllib.request.Request(f"{self.root_url}{path}",
                                         headers=headers)
        with open_url(request, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def status(self) -> dict:
        return self._get("/status")

    def decide(self, status: dict | None = None,
               retired: list[str] | None = None,
               co_located_members: int = 0,
               candidate: str | None = None,
               failed_before: dict | None = None) -> Supervision:
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
        if co_located_members >= self.max_per_host:
            return Supervision(
                "hold",
                f"{latency:.0f} ms is slow, but {candidate or 'the candidate'} "
                f"already carries {co_located_members} member(s) of this "
                "lineage: another would share its compute and add no "
                "information (rounds 95, 114)",
                latency_ms=latency, quorum_k=quorum_k, members=members)
        if failed_before is not None:
            return Supervision(
                "hold",
                f"{latency:.0f} ms is slow, but growing onto "
                f"{candidate or 'this candidate'} was rolled back earlier "
                f"(median {failed_before.get('observed', 0):.0f} ms against "
                f"{failed_before.get('before', 0):.0f} ms): the same change "
                "was measured and rejected",
                latency_ms=latency, quorum_k=quorum_k, members=members)
        return Supervision(
            "grow",
            f"{latency:.0f} ms > {target * self.pressure_factor:.0f} ms and "
            f"the quorum is at its floor ({quorum_k}) with "
            f"{members}/{self.max_members} members",
            latency_ms=latency, quorum_k=quorum_k, members=members)

    def _failed_before(self, host: str | None,
                       checkpoint: str | None) -> dict | None:
        """A rollback of this exact change, if it is still recent."""
        if not host or not checkpoint:
            return None
        entry = self.rolled_back.get((host, checkpoint))
        if entry is None:
            return None
        if self._clock() - entry["at"] > self.bad_candidate_ttl_s:
            return None
        return entry

    def _record(self, supervision: Supervision,
                journal: Journal | None) -> Supervision:
        """Journal a decision that was not executed (holds and proposals)."""
        if journal is not None:
            journal.append(Outcome(
                observation_id=supervision.target or "swarm",
                tier=supervision.action, acted=False, choice=None,
                agreement=0.0, mean_probability=0.0,
                responded=supervision.members,
                quorum_size=supervision.quorum_k or 0,
                reason=supervision.reason))
        return supervision

    def step(self, grow=None, retire=None,
             journal: Journal | None = None,
             candidate_host: str | None = None,
             candidate_checkpoint: str | None = None,
             status: dict | None = None) -> Supervision:
        """Decide, then execute. ``grow``/``retire`` are caller callbacks.

        ``status`` is injectable for tests; the live path reads ``/status``.
        """
        members: list[Member] = []
        try:
            members = Retirer(self.root_url, self.pairing_token,
                              timeout=self.timeout).members()
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        if status is None:
            try:
                status = self.status()
            except (urllib.error.URLError, urllib.error.HTTPError):
                status = {}
        remaining = self.cooling_down()
        if (remaining <= 0 and self._growth_bet is not None
                and self._clock() >= self._growth_bet["due"]):
            bet = self._growth_bet
            sample = status.get("decision_ewma_ms")
            if sample is not None:
                bet["samples"].append(float(sample))
            ready = (len(bet["samples"]) >= self.rollback_samples
                     or self._clock() >= bet["due"] + self.rollback_window_s)
            if not ready:
                # One sample after one cooldown cannot tell a slow start from
                # a bad change, so the bet is judged on a median (round 116).
                return self._record(
                    Supervision(
                        "hold",
                        f"evaluating growth {bet['node_id']}: sample "
                        f"{len(bet['samples'])}/{self.rollback_samples}",
                        latency_ms=sample, members=len(members)),
                    journal)
            self._growth_bet = None
            observed = statistics.median(bet["samples"]) if bet["samples"] \
                else None
            before = bet.get("latency_before")
            if (retire is None or not observed or not before
                    or observed <= before * self.rollback_factor):
                return self._record(
                    Supervision(
                        "hold",
                        f"growth {bet['node_id']} paid off: median "
                        f"{observed:.0f} ms vs {before:.0f} ms before it"
                        if observed and before
                        else f"growth {bet['node_id']}: no usable sample",
                        latency_ms=observed, members=len(members)),
                    journal)
            allowed, why = rollback_allowed(bet["node_id"], members, self.keep)
            if not allowed:
                return self._record(
                    Supervision(
                        "hold",
                        f"growth {bet['node_id']} looks bad "
                        f"({observed:.0f} ms vs {before:.0f} ms) but {why}",
                        target=bet["node_id"], latency_ms=observed,
                        members=len(members)),
                    journal)
            reason = (f"rollback: median {observed:.0f} ms after growing "
                      f"{bet['node_id']} against {before:.0f} ms before it "
                      f"({len(bet['samples'])} samples); authorised by the "
                      "reproduce grant")
            outcome = retire(bet["node_id"], reason, "reproduce")
            supervision = Supervision(
                "retire", outcome.reason, target=bet["node_id"],
                latency_ms=observed, members=len(members))
            if outcome.acted:
                key = (bet.get("host"), bet.get("checkpoint"))
                if key != (None, None):
                    self.rolled_back[key] = {
                        "at": self._clock(), "observed": observed,
                        "before": before, "node_id": bet["node_id"]}
            if journal is not None:
                journal.append(Outcome(
                    observation_id=bet["node_id"], tier="rollback",
                    acted=bool(outcome.acted), choice=None, agreement=0.0,
                    mean_probability=0.0, responded=len(members),
                    quorum_size=0, reason=supervision.reason))
            if outcome.acted:
                self.last_action = supervision
                self.last_action_at = self._clock()
                self.actions.append((self.last_action_at, "rollback",
                                     bet["node_id"]))
            return supervision
        if remaining > 0 and self.last_action is not None:
            return self._record(
                Supervision(
                    "hold",
                    f"cooling down {remaining:.0f}s after "
                    f"{self.last_action.action} "
                    f"({self.last_action.target or 'swarm'})",
                    latency_ms=self.last_action.latency_ms,
                    members=self.last_action.members),
                journal)
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
            supervision = self.decide(
                co_located_members=len(co_located(
                    members, candidate_host, candidate_checkpoint)),
                candidate=(f"{candidate_host}/{candidate_checkpoint}"
                           if candidate_host else None),
                failed_before=self._failed_before(candidate_host,
                                                   candidate_checkpoint))
        # A decision nobody executed must not start a cooldown or be recorded
        # as an action: a supervisor with no retire executor would otherwise
        # report the same retirement every cooldown cycle, having never
        # performed one (found in round 114).
        executed = False
        if supervision.action == "grow":
            if grow is not None:
                result = grow()
                supervision.reason += f"; grow -> {result}"
                executed = True
                if self.rollback:
                    self._growth_bet = {
                        "node_id": str(result).split()[0],
                        "latency_before": supervision.latency_ms,
                        "samples": [],
                        "due": self._clock() + self.cooldown_s,
                        "host": candidate_host,
                        "checkpoint": candidate_checkpoint,
                    }
            else:
                supervision.reason += "; no grow executor configured"
        elif supervision.action == "retire":
            if retire is not None:
                outcome = retire(supervision.target, supervision.reason)
                supervision.reason = (outcome.reason if outcome
                                      else supervision.reason)
                executed = bool(outcome and outcome.acted)
            else:
                supervision.reason += "; no retire executor configured"
        if journal is not None:
            journal.append(Outcome(
                observation_id=supervision.target or "swarm",
                tier=supervision.action, acted=supervision.action != "hold",
                choice=None, agreement=0.0, mean_probability=0.0,
                responded=supervision.members,
                quorum_size=supervision.quorum_k or 0,
                reason=supervision.reason))
        if executed:
            self.last_action = supervision
            self.last_action_at = self._clock()
            self.actions.append((self.last_action_at, supervision.action,
                                 supervision.target))
        return supervision
