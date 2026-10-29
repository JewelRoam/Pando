"""coordinator: the PandoRSI consensus coordinator.

Maintains a registry of stems (lazy expiry), fans decisions out concurrently
to a checkpoint-diverse quorum, and aggregates votes via common.aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .chat import normalize_chat


class _PandoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

from .common import (DECIDING_ROLE, DecisionRequest, DecisionResult,
                     PROPOSING_ROLE, Vote, aggregate, collapse_lineages)
from .spore import build_manifest, generate_signing_key
from .rollout import RolloutRecorder
from .calibrate import Calibrator
from .identity import verify as verify_identity

HEARTBEAT_TTL = 15.0
QUORUM_K = 5
CALL_TIMEOUT = 30.0
INITIAL_STEM_TIMEOUT = 5.0
MIN_STEM_TIMEOUT = 0.5
STEM_TIMEOUT_MULTIPLIER = 4.0
STEM_TIMEOUT_MARGIN = 0.25
STEM_EWMA_ALPHA = 0.3
FAILURE_BACKOFF_BASE = 1.0
FAILURE_BACKOFF_MAX = 30.0
# Voters whose latencies fall in the same band are treated as interchangeable
# when a quorum is selected, so the rotation can spread load across them.
LATENCY_BAND_MS = 25.0

# The measured-accuracy registry is deliberately data, not a model: a
# checkpoint is eligible for auto-assent only when a recorded evaluation
# names it. Unknown members are ignored rather than assigned a prior.
ASSENT_REGISTRY_ENV = "PANDOSI_ASSENT_REGISTRY"
ASSENT_REGISTRY_DEFAULT = ("registry", "accuracies.json")


DEFAULT_ASSENT_FAMILY = "known"


def _parse_accuracy_map(accuracy: dict) -> dict[str, float]:
    """Return finite numeric split accuracies from one registry record."""
    parsed: dict[str, float] = {}
    for split, raw in accuracy.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if math.isfinite(value):
            parsed[str(split)] = value
    return parsed


def load_accuracy_registry(
        source: str | Path | dict | None = None
) -> dict[str, dict[str, dict[str, float]]]:
    """Load per-checkpoint, per-family measured accuracy.

    The returned shape is ``checkpoint -> family -> split -> accuracy``.
    Schema v1 records are migrated in memory by wrapping their ``accuracy``
    mapping in the ``known`` family, so an old registry remains readable.

    ``source`` may be a registry path, an already-parsed payload, or ``None``
    for the checked-in ``assets/registry/accuracies.json`` (overridable with
    ``PANDOSI_ASSENT_REGISTRY``). Missing, malformed, and empty registries
    all become an empty mapping so callers can fall back to no-assent
    behavior without treating bad data as a reason to guess.
    """
    if isinstance(source, dict):
        payload = source
    else:
        if source is None:
            configured = os.environ.get(ASSENT_REGISTRY_ENV)
            if configured:
                path = Path(configured).expanduser()
            else:
                from .resources import asset_path
                path = asset_path(*ASSENT_REGISTRY_DEFAULT)
        else:
            path = Path(source).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    if not isinstance(payload, dict):
        return {}
    members = payload.get("members")
    if members is None and "accuracy" not in payload:
        # Also accept a compact direct mapping:
        # {"english": {"D": 0.7292}, ...}.
        members = payload
    if not isinstance(members, dict):
        return {}

    registry: dict[str, dict[str, dict[str, float]]] = {}
    for checkpoint, record in members.items():
        if not isinstance(checkpoint, str) or not isinstance(record, dict):
            continue
        families: dict[str, dict[str, float]] = {}
        raw_families = record.get("families")
        if isinstance(raw_families, dict):
            for family, family_record in raw_families.items():
                if (not isinstance(family, str)
                        or not isinstance(family_record, dict)):
                    continue
                accuracy = family_record.get("accuracy", family_record)
                if not isinstance(accuracy, dict):
                    continue
                parsed = _parse_accuracy_map(accuracy)
                if parsed:
                    families[family] = parsed
        if not families:
            # Schema v1 (and compact v1-shaped payloads) have no family
            # dimension. Their measured splits belong to the known family.
            accuracy = record.get("accuracy", record)
            if isinstance(accuracy, dict):
                parsed = _parse_accuracy_map(accuracy)
                if parsed:
                    families[DEFAULT_ASSENT_FAMILY] = parsed
        if families:
            registry[checkpoint] = families
    return registry


def _host_of(url: str) -> str:
    """The machine a member reports itself on, best effort."""
    try:
        return urlparse(url).hostname or url
    except ValueError:
        return url


def _empty_stem_stats() -> dict:
    return {
        "successes": 0,
        "failures": 0,
        "last_latency_ms": None,
        "latency_ewma_ms": None,
        "last_error": None,
        "consecutive_failures": 0,
        "unhealthy_until": 0.0,
    }


def http_transport(url: str, payload: bytes, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class Coordinator:
    def __init__(self, transport=http_transport, quorum_k: int = QUORUM_K,
                 gating: dict | None = None,
                 call_timeout: float = CALL_TIMEOUT,
                 initial_stem_timeout: float = INITIAL_STEM_TIMEOUT,
                 min_stem_timeout: float = MIN_STEM_TIMEOUT,
                 timeout_multiplier: float = STEM_TIMEOUT_MULTIPLIER,
                 timeout_margin: float = STEM_TIMEOUT_MARGIN,
                 failure_backoff_base: float = FAILURE_BACKOFF_BASE,
                 failure_backoff_max: float = FAILURE_BACKOFF_MAX,
                 adaptive_quorum: bool = False,
                 quorum_floor: int = 1,
                 latency_target_ms: float | None = None,
                 count_lineages: bool = False,
                 blocs: dict | None = None,
                 combiner=None,
                 rollout_dir: str | Path = "rollouts",
                 assent_checkpoint: str | None = None,
                 assent_fallback: str = "advisor",
                 assent_registry: str | Path | dict | None = None,
                 assent_registry_path: str | Path | None = None,
                 identity_store: str | Path | None = None):
        self.nodes: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._pick_rotation = 0
        self._inflight: dict[str, int] = {}
        # Retired nodes stay out: a stem re-registers every few seconds, so
        # simply deleting it from the registry would last five seconds.
        self.blocked: dict[str, str] = {}
        self._transport = transport
        self.quorum_k = quorum_k
        # Round 119: a replica is a correlated voter, so counting stems lets
        # a replicated lineage decide alone. With this on, the quorum is
        # counted over lineages and repeated stems collapse to one vote.
        self.count_lineages = count_lineages
        # Round 128: a *group* of lineages whose errors move together is one
        # opinion, declared by the playbook. Empty means names are the only
        # grouping, which is the round-121 behaviour.
        self.blocs = dict(blocs or {})
        # An advisory fitted combiner (rounds 139/140): it answers where the
        # hand-written gate declines, and it can never change a tier or an
        # auto-executed choice.
        self.combiner = combiner
        # Adaptive quorum (round 110): a larger k means more independent
        # voters per decision and fewer decisions in parallel, so the size is
        # traded against latency under load - down to a floor, never below it.
        self.adaptive_quorum = adaptive_quorum
        self.quorum_floor = max(1, int(quorum_floor))
        self.quorum_ceiling = max(self.quorum_floor, int(quorum_k))
        self.latency_target_ms = latency_target_ms
        self._decision_ewma_ms: float | None = None
        self.gating = gating or {}
        # Opt-in policy (round 157): a named checkpoint may veto auto tier
        # when its (collapsed) vote disagrees with the aggregate choice.
        # Round 158 made the answer source an explicit deployment policy:
        # "advisor" keeps the aggregate choice at verify, while "veto"
        # returns the assent member's choice without fabricating agreement.
        # "auto" resolves that name from measured accuracy at decision time.
        if assent_fallback not in {"advisor", "veto"}:
            raise ValueError("assent_fallback must be 'advisor' or 'veto'")
        if assent_registry is not None and assent_registry_path is not None:
            raise ValueError("assent_registry and assent_registry_path are "
                             "mutually exclusive")
        if assent_registry_path is not None:
            assent_registry = assent_registry_path
        self.assent_checkpoint = assent_checkpoint
        self.assent_fallback = assent_fallback
        self.assent_registry = load_accuracy_registry(assent_registry)
        # Keep a readable alias for callers that inspect the loaded data.
        self.assent_accuracies = self.assent_registry
        self.assent_resolved: str | None = None
        self._assent_warning_emitted = False
        self._assent_family_warnings: set[str] = set()
        self.call_timeout = call_timeout
        self.initial_stem_timeout = initial_stem_timeout
        self.min_stem_timeout = min_stem_timeout
        self.timeout_multiplier = timeout_multiplier
        self.timeout_margin = timeout_margin
        self.failure_backoff_base = failure_backoff_base
        self.failure_backoff_max = failure_backoff_max
        self.playbook_hash: str | None = None
        self._calibrator = Calibrator()
        self._uncalibrated_warned: set[str] = set()
        self.known_questions: set[str] | None = None  # None = accept any
        self.question_categories: dict[str, str] = {}
        self._recorder = RolloutRecorder(rollout_dir)
        # Durable node-id -> public-key bindings (identity layer). Unlike the
        # TTL'd registry entry this survives heartbeat expiry, so a token
        # holder cannot re-claim a member's id with a fresh key. With an
        # identity_store file it also survives root restarts (round 033).
        self.identity_bindings: dict[str, str] = {}
        self.identity_store: Path | None = (
            Path(identity_store).expanduser() if identity_store else None)
        if self.identity_store and self.identity_store.exists():
            try:
                loaded = json.loads(
                    self.identity_store.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.identity_bindings.update(
                        {str(k): str(v) for k, v in loaded.items()})
            except (OSError, json.JSONDecodeError):
                pass

    # -- identity -------------------------------------------------------
    def _save_identity_bindings(self) -> None:
        """Best-effort write-through; durability must never block a join."""
        if not self.identity_store:
            return
        try:
            self.identity_store.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.identity_store.with_name(
                self.identity_store.name + ".tmp")
            tmp.write_text(json.dumps(self.identity_bindings, indent=2),
                           encoding="utf-8")
            os.replace(tmp, self.identity_store)
        except OSError:
            pass

    def bind_identity(self, node_id: str, pubkey: str) -> bool:
        """Bind a node id to a public key. False when another key owns it."""
        with self._lock:
            bound = self.identity_bindings.get(node_id)
            if bound is None:
                self.identity_bindings[node_id] = pubkey
                self._save_identity_bindings()
                return True
            return bound == pubkey

    def identity_of(self, node_id: str) -> str | None:
        with self._lock:
            return self.identity_bindings.get(node_id)

    # -- registry -------------------------------------------------------
    def register(self, node_id: str, url: str, checkpoint: str,
                 cap: dict | None = None,
                 artifact_sha256: str | None = None,
                 chat: dict | None = None,
                 role: str | None = None) -> bool:
        """Add or refresh a node. Returns False for a retired node."""
        with self._lock:
            if node_id in self.blocked:
                return False
        self._maybe_warn_assent_registry()
        # Calibration is looked up by checkpoint string, so a misnamed stem
        # silently changes the gate. Say so once, at join time, rather than
        # letting it alter every later decision in silence (round 100).
        _, is_calibrated = self._calibrator.calibrate(checkpoint, 0.5)
        if not is_calibrated and checkpoint not in self._uncalibrated_warned:
            self._uncalibrated_warned.add(checkpoint)
            print(f"[calibration] checkpoint {checkpoint!r} has no "
                  f"temperature table; its votes cannot reach tier 'auto'",
                  flush=True)
        with self._lock:
            stats = self.nodes.get(node_id, {}).get("stats")
            record = {
                "url": url,
                # Which machine this member lives on. Two members of the same
                # lineage on one host share its compute and add no
                # information, so growth decisions need to see it (round 115).
                "host": _host_of(url),
                "checkpoint": checkpoint,
                **({"artifact_sha256": artifact_sha256}
                   if artifact_sha256 else {}),
                "capability": cap or {},
                "last_seen": time.monotonic(),
                "stats": stats or _empty_stem_stats(),
            }
            kept = normalize_chat(chat)
            if kept:
                record["chat"] = kept
            # What this member is for travels in its own field, so a reader
            # never has to infer it from the backend again. A stem that
            # names nothing is read off what it can do: one that holds a
            # conversation proposes, everything else decides.
            record["role"] = str(
                role or (PROPOSING_ROLE if kept else DECIDING_ROLE))
            self.nodes[node_id] = record
        return True

    def retire(self, node_id: str, reason: str = "retired") -> bool:
        """Drop a node from the registry and refuse its re-registration."""
        with self._lock:
            existed = node_id in self.nodes
            self.nodes.pop(node_id, None)
            self._inflight.pop(node_id, None)
            self.blocked[node_id] = reason
        return existed

    def live_nodes(self) -> dict[str, dict]:
        now = time.monotonic()
        with self._lock:
            live = {}
            for node_id, info in self.nodes.items():
                if now - info["last_seen"] > HEARTBEAT_TTL:
                    continue
                copy = dict(info)
                bound = self.identity_bindings.get(node_id)
                if bound:
                    copy["pubkey"] = bound
                stats = dict(info.get("stats", _empty_stem_stats()))
                stats["cooldown_remaining_s"] = round(
                    max(0.0, stats.get("unhealthy_until", 0.0) - now), 3)
                copy["stats"] = stats
                live[node_id] = copy
            return live

    def _collapses(self) -> bool:
        """Whether the quorum is counted over opinions rather than stems."""
        return bool(self.blocs) or self.count_lineages

    def _bloc_of(self, checkpoint: str) -> str:
        return self.blocs.get(checkpoint, checkpoint)

    @staticmethod
    def _cost_bucket(info: dict) -> int:
        """Coarse host-cost band for a member, cheapest first.

        Ten-fold buckets: anything within an order of magnitude counts as the
        same price, so a fleet spreads across peers of similar cost and
        ignores the one that is orders of magnitude more expensive.
        """
        cost = info.get("capability", {}).get("host_ms_per_decision")
        if cost is None:
            return 9
        return int(math.log10(max(float(cost), 1.0)))

    def _maybe_warn_assent_registry(self) -> None:
        """Warn once when auto-assent has no usable measured registry."""
        if (self.assent_checkpoint != "auto" or self.assent_registry
                or self._assent_warning_emitted):
            return
        self._assent_warning_emitted = True
        print("[assent] measured accuracy registry is missing or empty; "
              "auto assent disabled", flush=True)

    def _registry_families(self) -> set[str]:
        """Every evidence family named by any member in the registry."""
        return {
            family
            for member in self.assent_registry.values()
            for family in member
        }

    def _assent_family_for(self, req: DecisionRequest) -> str:
        """Resolve the evidence family for this request.

        An explicit request field wins. Otherwise a playbook category is
        honoured only when it names a family present in the registry; current
        categories such as ``ordinary`` and ``security`` are gating policy,
        not evidence families, so they fall through to ``known``.
        """
        if req.family is not None:
            requested = str(req.family).strip()
            if requested:
                return requested
        declared = (self.question_categories or {}).get(req.question)
        if declared in self._registry_families():
            return declared
        return DEFAULT_ASSENT_FAMILY

    def _warn_missing_assent_family(self, family: str) -> None:
        """Warn once when no present member has data for this family."""
        if not self.assent_registry or family in self._assent_family_warnings:
            return
        self._assent_family_warnings.add(family)
        print(f"[assent] no present member has measured accuracy for family "
              f"{family!r}; auto assent disabled for this decision",
              flush=True)

    def _resolve_assent_checkpoint(self, votes: list[Vote],
                                   family: str = DEFAULT_ASSENT_FAMILY
                                   ) -> str | None:
        """Choose the best mean accuracy for one family among present votes.

        Only checkpoints with measured data in this family are eligible;
        members measured in another family do not participate. Ties resolve
        by checkpoint name so a decision is reproducible across runs.
        """
        ranked: list[tuple[float, str]] = []
        for vote in votes:
            families = self.assent_registry.get(vote.checkpoint) or {}
            accuracies = families.get(family)
            if not accuracies:
                continue
            ranked.append((sum(accuracies.values()) / len(accuracies),
                           vote.checkpoint))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][1]

    def _selectable_nodes(self) -> dict[str, dict]:
        live = self.live_nodes()
        now = time.monotonic()
        healthy = {
            node_id: info for node_id, info in live.items()
            if info["stats"].get("unhealthy_until", 0.0) <= now
        }
        return healthy or live

    def _group_rank_key(self, inflight: dict):
        """Order members inside a checkpoint group (rounds 109/134/156).

        Members of one group are interchangeable in information, so their
        order is a spending decision. Two signals price a member: what it
        costs the machine it lives on, in coarse ten-fold buckets (round
        134 measured a 500-1000x spread between a CPU stem and an NPU stem
        on one host), and a coarse latency band (25 ms) so two fast units
        are treated as equals and share load (rounds 109-110). Least-loaded
        members still lead, so concurrent decisions spread.

        Which signal leads depends on what the operator asked for. With no
        latency target, cost leads: a cheap member is not woken ahead of an
        expensive one. With a target, latency leads and cost breaks ties -
        round 156 measured the alternative live: the cost-first order kept
        the 110 ms NPU seated at the quorum floor while the 40 ms iGPU
        idled, 115 ms p50 against the fleet's own 100 ms target.
        """
        under_target = self.latency_target_ms is not None

        def key(pair):
            index, (node_id, info) = pair
            stats = info.get("stats", {})
            load = inflight.get(node_id, 0)
            band = int((stats.get("latency_ewma_ms") or 0.0)
                       // LATENCY_BAND_MS)
            cost = self._cost_bucket(info)
            if under_target:
                return (load, band, cost, index)
            return (cost, load, band, index)

        return key

    def pick_quorum(self, k: int) -> list[tuple[str, dict]]:
        """Pick k live nodes maximizing checkpoint diversity (round-robin
        across checkpoint groups). Falls back to any live nodes.

        Candidates are ordered by how many calls they are already serving,
        then by latency, with a per-decision rotation breaking ties among
        equally idle members. Round 109 showed that without spreading,
        concurrent decisions used the same voters and extra replicas bought no
        capacity; round 110 showed blind rotation still queued behind a busy
        voter, so load comes first.
        """
        nodes = self._selectable_nodes()
        with self._lock:
            inflight = dict(self._inflight)
        groups: dict[str, list[tuple[str, dict]]] = {}
        for nid, info in nodes.items():
            groups.setdefault(info["checkpoint"], []).append((nid, info))
        for members in groups.values():
            members.sort(key=lambda item: item[0])
        with self._lock:
            rotation = self._pick_rotation
            self._pick_rotation = (rotation + 1) % 1_000_003
        for checkpoint, members in groups.items():
            if len(members) > 1:
                shift = rotation % len(members)
                members = members[shift:] + members[:shift]
            ranked = sorted(enumerate(members),
                            key=self._group_rank_key(inflight))
            groups[checkpoint] = [item for _, item in ranked]
        picked: list[tuple[str, dict]] = []
        group_iter = iter(sorted(groups))
        while len(picked) < k and groups:
            try:
                ckpt = next(group_iter)
            except StopIteration:
                group_iter = iter(sorted(groups))
                ckpt = next(group_iter)
            if groups[ckpt]:
                picked.append(groups[ckpt].pop(0))
            else:
                del groups[ckpt]
        return picked[:k]

    # -- decision -------------------------------------------------------
    def _log_decision(self, req, result, quorum_size: int,
                      fanout_ms: float) -> None:
        """Append one JSON line per decision (local metrics only)."""
        try:
            metrics_dir = Path("metrics")
            metrics_dir.mkdir(exist_ok=True)
            entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "request_id": req.request_id,
                     "choice": result.choice,
                     "tier": result.tier,
                     "agreement": result.agreement,
                     "mean_probability": result.mean_probability,
                     "responded": result.responded if hasattr(result, "responded") else len(result.votes),
                     "quorum_size": quorum_size,
                     "fanout_ms": fanout_ms}
            with (metrics_dir / "decisions.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # metrics must never break decisions

    def _record_stem_call(self, node_id: str, ok: bool,
                          latency_ms: float, error: str | None = None) -> None:
        with self._lock:
            info = self.nodes.get(node_id)
            if info is None:
                return
            stats = info.setdefault("stats", _empty_stem_stats())
            stats["last_latency_ms"] = round(latency_ms, 3)
            if ok:
                stats["successes"] += 1
                stats["consecutive_failures"] = 0
                stats["unhealthy_until"] = 0.0
                previous = stats.get("latency_ewma_ms")
                stats["latency_ewma_ms"] = round(
                    latency_ms if previous is None else
                    STEM_EWMA_ALPHA * latency_ms
                    + (1 - STEM_EWMA_ALPHA) * previous,
                    3)
                stats["last_error"] = None
            else:
                stats["failures"] += 1
                failures = int(stats.get("consecutive_failures", 0)) + 1
                stats["consecutive_failures"] = failures
                backoff = min(
                    self.failure_backoff_max,
                    self.failure_backoff_base
                    * (2 ** min(failures - 1, 50)))
                stats["unhealthy_until"] = time.monotonic() + backoff
                stats["last_error"] = error

    def _category_floor(self, question: str) -> int:
        """The fewest voters acceptable for this question's category.

        The floor is the same knob as "how much agreement am I willing to
        give up", and a security question is where giving it up is least
        acceptable, so a playbook can set a higher floor for that category
        (round 111).

        The category is *declared* by the playbook question (round 112): a
        question named ``risk_check`` used to fall through to the ordinary
        floor because the keyword scan did not match. Now a declared category
        decides, the keywords remain only as a conservative cross-check, and
        a question that a categorised playbook does not classify is treated
        as the most demanding category rather than the least.
        """
        gating = self.gating or {}
        categories = self.question_categories or {}
        declared = categories.get(question)
        keyword_security = any(
            keyword in (question or "").lower()
            for keyword in ("security", "suspicious", "phishing", "threat"))

        def floor_for(category: str | None) -> int:
            if category and (gating.get(category) or {}).get("quorum_floor"):
                return int(gating[category]["quorum_floor"])
            if category == "security":
                return int(gating.get("quorum_floor", self.quorum_floor))
            return int(gating.get("quorum_floor", self.quorum_floor))

        if declared:
            floor = floor_for(declared)
        elif categories:
            # The playbook categorises its questions, so an unclassified one
            # is a playbook defect: take the most demanding declared floor
            # rather than the default one.
            floors = [int((gating.get(category) or {}).get("quorum_floor", 0))
                      for category in set(categories.values())]
            floors.append(int(gating.get("quorum_floor", self.quorum_floor)))
            floor = max(floors)
        else:
            floor = floor_for("security" if keyword_security else None)
        if keyword_security:
            floor = max(floor, floor_for("security"))
        return max(1, floor)

    def _adapt_quorum(self, question: str) -> None:
        """Trade voters per decision against decisions in parallel.

        Called once per decision. With a latency target, the quorum shrinks
        while loaded decisions run slow and recovers once they do not; the
        floor is per question category, so a security decision keeps its
        voters while an ordinary one may give them up for throughput.

        When the quorum counts opinions, the ceiling is the number of
        *opinions*: a k beyond that wakes a second stem of some lineage -
        latency with no information. Round 156 measured the unfixed dial
        live: a two-opinion fleet oscillated k 2<->3 forever, paying ~40 ms
        on every second decision for a vote its lineage already cast.
        """
        target = self.latency_target_ms
        if target is None or self._decision_ewma_ms is None:
            return
        floor = self._category_floor(question)
        selectable = self._selectable_nodes()
        available = len(selectable)
        ceiling = min(self.quorum_ceiling, max(available, 1))
        if self._collapses():
            opinions = len({self._bloc_of(info["checkpoint"])
                            for info in selectable.values()})
            ceiling = min(ceiling, max(1, opinions))
        if self.quorum_k < floor:
            self.quorum_k = floor
            return
        if self._decision_ewma_ms > target and self.quorum_k > floor:
            self.quorum_k -= 1
            print(f"[quorum] {self._decision_ewma_ms:.0f} ms > target "
                  f"{target:.0f} ms -> k={self.quorum_k} (floor {floor})",
                  flush=True)
        elif (self._decision_ewma_ms < target * 0.6
              and self.quorum_k < ceiling):
            self.quorum_k += 1
            print(f"[quorum] {self._decision_ewma_ms:.0f} ms < target "
                  f"{target * 0.6:.0f} ms -> k={self.quorum_k}", flush=True)

    def _timeout_for(self, node_id: str) -> float:
        with self._lock:
            info = self.nodes.get(node_id)
            stats = info.get("stats", {}) if info else {}
            ewma_ms = stats.get("latency_ewma_ms")
        if ewma_ms is None:
            return self.initial_stem_timeout
        learned_s = (float(ewma_ms) / 1000.0) * self.timeout_multiplier \
            + self.timeout_margin
        return min(self.call_timeout,
                   max(self.min_stem_timeout, learned_s))

    def _call_stem(self, node_id: str, info: dict,
                   payload: bytes) -> Vote | None:
        timeout = self._timeout_for(node_id)
        with self._lock:
            self._inflight[node_id] = self._inflight.get(node_id, 0) + 1
        started = time.perf_counter()
        try:
            raw = self._transport(f"{info['url']}/decide", payload,
                                  timeout)
            data = json.loads(raw)
            wire_vote = data["vote"]
            vote = Vote(node_id, info["checkpoint"],
                        wire_vote["choice"],
                        float(wire_vote["probability"]),
                        bool(wire_vote.get("neutral", False)))
            self._record_stem_call(
                node_id, True, (time.perf_counter() - started) * 1000)
            return vote
        except Exception as exc:
            self._record_stem_call(
                node_id, False, (time.perf_counter() - started) * 1000,
                f"{type(exc).__name__}: {exc}")
            return None  # dead node: shrink quorum, never block the vote
        finally:
            with self._lock:
                self._inflight[node_id] = max(
                    0, self._inflight.get(node_id, 1) - 1)


    def decide(self, req: DecisionRequest) -> dict:
        self._maybe_warn_assent_registry()
        if self.known_questions and req.question not in self.known_questions:
            return {"error": f"question {req.question!r} not in playbook; "
                    f"known: {sorted(self.known_questions)}",
                    "responded": 0, "quorum_size": 0}
        if self.adaptive_quorum:
            self._adapt_quorum(req.question)
        quorum = self.pick_quorum(self.quorum_k)
        payload = req.to_json()
        fanout_started = time.perf_counter()
        if quorum:
            with ThreadPoolExecutor(
                    max_workers=len(quorum),
                    thread_name_prefix="pando-quorum") as pool:
                results = list(pool.map(
                    lambda item: self._call_stem(item[0], item[1], payload),
                    quorum))
            votes = [vote for vote in results if vote is not None]
            unanswered = [node_id for (node_id, _), vote in zip(quorum,
                                                               results)
                          if vote is None]
        else:
            votes = []
            unanswered = []
        fanout_ms = round((time.perf_counter() - fanout_started) * 1000, 3)
        # Track the end-to-end decision time, not just the fan-out, so the
        # adaptive quorum reacts to queueing rather than to one slow voter.
        with self._lock:
            if self._decision_ewma_ms is None:
                self._decision_ewma_ms = fanout_ms
            else:
                self._decision_ewma_ms = (
                    STEM_EWMA_ALPHA * fanout_ms
                    + (1 - STEM_EWMA_ALPHA) * self._decision_ewma_ms)

        # Calibrate probabilities before aggregation (blueprint 3.2.1)
        calibrated_votes = []
        uncalibrated: set[str] = set()
        for v in votes:
            if v.neutral:
                # No confidence was reported, so there is nothing to
                # temperature-scale and no calibration warning to raise.
                calibrated_votes.append(Vote(
                    v.node_id, v.checkpoint, v.choice, v.probability,
                    neutral=True))
                continue
            cp, is_calibrated = self._calibrator.calibrate(
                v.checkpoint, v.probability)
            if not is_calibrated:
                uncalibrated.add(v.checkpoint)
            calibrated_votes.append(Vote(v.node_id, v.checkpoint, v.choice, cp))

        # Security questions get stricter gating (lower thresholds)
        is_security = any(
            kw in req.question.lower()
            for kw in ("security", "suspicious", "phishing", "threat"))
        eff_gating = dict(self.gating)
        if is_security and self.gating.get("security"):
            sec = self.gating["security"]
            eff_gating["auto_min_agreement"] = sec.get(
                "auto_min_agreement", 0.5)
            eff_gating["auto_min_probability"] = sec.get(
                "auto_min_probability", 0.4)
        result = aggregate(
            calibrated_votes,
            min_quorum=min(3, len({self._bloc_of(info["checkpoint"])
                                   for _, info in quorum})
                           if self._collapses() else len(quorum)),
            auto_min_agreement=eff_gating.get("auto_min_agreement", 0.6),
            auto_min_probability=eff_gating.get("auto_min_probability", 0.6),
            count_lineages=self.count_lineages,
            blocs=self.blocs or None)
        lineages = len({info["checkpoint"] for _, info in quorum})
        opinions = len({self._bloc_of(info["checkpoint"])
                        for _, info in quorum}) if self._collapses() else lineages
        # Corroboration is a policy, not a default: a lone lineage is still
        # allowed to decide (a single device has to work), but a playbook can
        # require more opinions than one. Round 119 measured what the
        # alternative costs - replicas are one opinion, so three copies of one
        # lineage agreeing is agreement of 1.0 over nothing. This is
        # independent of count_lineages: a playbook may want two opinions
        # without wanting same-lineage votes collapsed.
        corroboration_floor = max(1, int(
            eff_gating.get("corroboration_floor", 1)))
        veto_checkpoint = None
        if result.tier == "auto" and opinions < corroboration_floor:
            result = DecisionResult(result.choice, result.agreement,
                                    result.mean_probability, "verify",
                                    result.votes)
        resolved_assent = None
        assent_family: str | None = None
        collapsed_assent_votes: list[Vote] = []
        if self.assent_checkpoint is not None:
            collapsed_assent_votes = (
                collapse_lineages(calibrated_votes, self.blocs or None)
                if self._collapses() else calibrated_votes)
            if self.assent_checkpoint == "auto":
                assent_family = self._assent_family_for(req)
                resolved_assent = self._resolve_assent_checkpoint(
                    collapsed_assent_votes, assent_family)
                if resolved_assent is None:
                    self._warn_missing_assent_family(assent_family)
                self.assent_resolved = resolved_assent
            else:
                # An explicit witness is known up front; whether it becomes
                # the answer on a declined row is decided below.
                resolved_assent = self.assent_checkpoint
            if result.tier == "auto" and resolved_assent is not None:
                assent_vote = next(
                    (vote for vote in collapsed_assent_votes
                     if vote.checkpoint == resolved_assent), None)
                if (assent_vote is not None
                        and assent_vote.choice != result.choice):
                    choice = result.choice
                    if self.assent_fallback == "veto":
                        choice = assent_vote.choice
                        veto_checkpoint = resolved_assent
                    result = DecisionResult(choice, result.agreement,
                                            result.mean_probability, "verify",
                                            result.votes)
            # Round 165: `veto` answers *every* declined row, not only the
            # rows where the witness disagreed with an auto verdict. When a
            # witness is resolved and actually present, its choice becomes
            # the answer on any non-auto tier. Agreement, mean_probability
            # and the tier itself are left untouched, so nothing is
            # fabricated; the witness is named in `veto` for audit.
            # `advisor` (the default) never reaches this branch, so its
            # behaviour is unchanged.
            if (self.assent_fallback == "veto"
                    and veto_checkpoint is None
                    and resolved_assent is not None
                    and result.tier != "auto"):
                witness_vote = next(
                    (vote for vote in collapsed_assent_votes
                     if vote.checkpoint == resolved_assent), None)
                if witness_vote is not None:
                    veto_checkpoint = resolved_assent
                    result = DecisionResult(
                        witness_vote.choice, result.agreement,
                        result.mean_probability, result.tier, result.votes)
        self._log_decision(req, result, len(quorum), fanout_ms)
        try:
            self._recorder.record(req.request_id, req.state, req.question,
                                  req.candidates,
                                  [v.to_json() for v in votes],
                                  result.choice, result.tier,
                                  playbook_hash=self.playbook_hash)
        except OSError:
            pass
        out = result.__dict__.copy()
        out["quorum_size"] = len(quorum)
        if veto_checkpoint is not None:
            out["veto"] = veto_checkpoint
        # Agreement over a single lineage is vacuous, so an operator needs to
        # see how many opinions stood behind the vote (round 119: a quorum of
        # replicas is one model with extra steps).
        if self._collapses() or corroboration_floor > 1:
            out["lineages"] = lineages
            out["opinions"] = opinions
        if self.blocs:
            out["blocs"] = opinions
        if corroboration_floor > 1:
            out["corroboration_floor"] = corroboration_floor
        out["responded"] = len(votes)
        # "Escalate" without a reason is unreadable: a member that was picked
        # and said nothing looks exactly like a member that was never asked.
        # The list names the ones that dropped out (round 135).
        if unanswered:
            out["unanswered"] = unanswered
        # Where the gate has nothing to say, offer the fitted rule's answer -
        # as a suggestion, at the tier the gate already chose. Round 139
        # measured 54.4% for this rule on the declined rows against 41.2% for
        # the best member, so it improves the prior without touching safety.
        if self.combiner is not None and result.tier != "auto" and votes:
            # The advice must see what the quorum counted. When the quorum is
            # counted over opinions, hand the combiner the collapsed set: it
            # was fitted on distinct memberships, and feeding it the stems
            # lets a duplicated lineage swamp the rows the gate declined.
            # Round 154, under --lineage-quorum, on the declined rows
            # (english x2 + ft-v3): advisor accuracy 0.704 -> 0.926 (split D,
            # 81 rows) and 0.685 -> 0.904 (split C, 73 rows); the system goes
            # 0.9000 -> 0.9750 and 0.8583 -> 0.9250 respectively.
            advice = (collapse_lineages(calibrated_votes, self.blocs or None)
                      if self._collapses() else calibrated_votes)
            suggestion = self.combiner.suggest(
                [(vote.checkpoint, vote.choice, vote.probability)
                 for vote in advice], req.candidates,
                state=req.state, question=req.question)
            if suggestion:
                out["suggested_choice"] = suggestion["choice"]
                out["suggested_probability"] = suggestion["probability"]
        if self.assent_checkpoint == "auto":
            out["assent_family"] = assent_family
            if resolved_assent is not None:
                out["assent"] = resolved_assent
        out["fanout_ms"] = fanout_ms
        # Report uncalibrated voters rather than acting on them silently. The
        # tier is left alone so a synthetic test harness is unaffected; the
        # policy belongs to whatever acts on the answer (see AutonomyLoop).
        if uncalibrated:
            out["uncalibrated_checkpoints"] = sorted(uncalibrated)
        return out


def build_handler(coordinator: Coordinator, root=None):
    class RootHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, payload) -> None:
            body = (payload if isinstance(payload, bytes)
                    else json.dumps(payload).encode("utf-8"))
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _pairing_authorized(self) -> bool:
            expected = getattr(root, "pairing_token", None) if root else None
            if not expected:
                return True
            supplied = self.headers.get("X-Pairing-Token", "")
            authorization = self.headers.get("Authorization", "")
            if authorization.startswith("Bearer "):
                supplied = authorization[7:]
            return hmac.compare_digest(
                supplied.encode("utf-8"), expected.encode("utf-8"))

        def do_GET(self):  # noqa: N802
            if self.path == "/pairing":
                required = bool(root and root.pairing_token)
                self._send(200, {"required": required})
                return
            if not self._pairing_authorized():
                self._send(403, {"error": "invalid or missing pairing token"})
                return
            if self.path == "/nodes":
                self._send(200, {"nodes": coordinator.live_nodes()})
            elif self.path == "/status":
                # What an elastic supervisor needs to decide, in one read
                # (round 113): how much checking is left to spend, and how
                # slow decisions actually are.
                self._send(200, {
                    "members": len(coordinator.live_nodes()),
                    "blocked": sorted(coordinator.blocked),
                    "quorum_k": coordinator.quorum_k,
                    "quorum_floor": coordinator.quorum_floor,
                    "quorum_ceiling": coordinator.quorum_ceiling,
                    "adaptive": coordinator.adaptive_quorum,
                    "latency_target_ms": coordinator.latency_target_ms,
                    "decision_ewma_ms": coordinator._decision_ewma_ms,
                    "question_categories": coordinator.question_categories,
                    "assent_checkpoint": coordinator.assent_checkpoint,
                    "assent_fallback": coordinator.assent_fallback,
                    "assent_resolved": coordinator.assent_resolved,
                })
            elif self.path == "/playbook" and root is not None \
                    and getattr(root, "playbook", None) is not None:
                self._send(200, {
                    "name": root.playbook.name,
                    "version": root.playbook.version,
                    "hash": root.playbook.hash(),
                    "canonical": json.loads(root.playbook.canonical_json()),
                    "gating": root.playbook.gating,
                    "questions": [
                        {"id": q["id"], "type": q["type"],
                         "instructions": q.get("instructions", ""),
                         "criteria": q.get("criteria", []),
                         "informational": q.get("informational", False)}
                        for q in root.playbook.questions
                    ],
                })
            elif (self.path == "/spore/pubkey"
                    and root is not None
                    and getattr(root, chr(115)+chr(112)+chr(111)+chr(114)+chr(101)+chr(95)+chr(109)+chr(97)+chr(110)+chr(105)+chr(102)+chr(101)+chr(115)+chr(116), None) is not None):
                self._send(200, {"pubkey": root.spore_pubkey})
            elif (self.path == "/spore/manifest"
                    and root is not None
                    and getattr(root, chr(115)+chr(112)+chr(111)+chr(114)+chr(101)+chr(95)+chr(109)+chr(97)+chr(110)+chr(105)+chr(102)+chr(101)+chr(115)+chr(116), None) is not None):
                self._send(200, root.spore_manifest)
            elif (self.path.startswith("/spore/manifests/")
                    and root is not None):
                digest = self.path.rsplit("/", 1)[-1]
                cached = getattr(root, "spore_history", {}).get(digest)
                if cached is None:
                    self._send(404, {"error": "unknown manifest"})
                else:
                    self._send(200, cached)
            elif (self.path.startswith("/spore/chunk/")
                    and root is not None
                    and getattr(root, chr(115)+chr(112)+chr(111)+chr(114)+chr(101)+chr(95)+chr(109)+chr(97)+chr(110)+chr(105)+chr(102)+chr(101)+chr(115)+chr(116), None) is not None):
                parts = self.path.split("/")
                name = "/".join(parts[3:-1])
                index = int(parts[-1])
                chunk = root.spore_chunks.get((name, index))
                if chunk is None:
                    self._send(404, b"{}")
                else:
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "application/octet-stream")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
            elif (self.path.startswith("/spore/chunks/")
                    and root is not None
                    and getattr(root, chr(115)+chr(112)+chr(111)+chr(114)+chr(101)+chr(95)+chr(109)+chr(97)+chr(110)+chr(105)+chr(102)+chr(101)+chr(115)+chr(116), None) is not None):
                # Batched transfer, same as SporeServer (round 106).
                parts = self.path.split("/")
                name = "/".join(parts[3:-2])
                start, count = int(parts[-2]), int(parts[-1])
                present = all((name, index) in root.spore_chunks
                              for index in range(start, start + count))
                if not present or count <= 0:
                    self._send(404, b"{}")
                else:
                    body = b"".join(root.spore_chunks[(name, index)]
                                    for index in range(start, start + count))
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "application/octet-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            if self.path != "/register" and not self._pairing_authorized():
                self._send(403, {"error": "invalid or missing pairing token"})
                return
            if self.path == "/register":
                data = json.loads(raw)
                expected = getattr(root, "pairing_token", None) if root else None
                supplied = data.get("pairing_token", "")
                if expected and not hmac.compare_digest(
                        str(supplied).encode("utf-8"),
                        str(expected).encode("utf-8")):
                    self._send(403, {"error": "invalid or missing pairing token"})
                    return
                admitted = getattr(root, "admitted_lineages", None)
                if admitted is not None and data.get("checkpoint") not in admitted:
                    self._send(403, {
                        "error": f"lineage {data.get('checkpoint')!r} is not "
                                 f"declared by this playbook",
                        "admitted": sorted(admitted)})
                    return
                expected_artifact = getattr(
                    root, "lineage_artifacts", {}).get(data.get("checkpoint"))
                if expected_artifact and \
                        data.get("artifact_sha256") != expected_artifact:
                    self._send(403, {
                        "error": f"lineage {data.get('checkpoint')!r} requires "
                                 "artifact attestation",
                        "expected_sha256": expected_artifact,
                        "reported_sha256": data.get("artifact_sha256")})
                    return
                identity = data.get("identity") or {}
                pubkey = str(identity.get("pubkey", "")).strip()
                signature = str(identity.get("sig", "")).strip()
                if getattr(root, "require_identity", False) and not (
                        pubkey and signature):
                    self._send(403, {"error": "this swarm requires a "
                                              "signed node identity"})
                    return
                if pubkey or signature:
                    signed = dict(data)
                    signed.pop("identity", None)
                    if not (pubkey and signature and verify_identity(
                            pubkey, signed, signature)):
                        self._send(403, {"error": "invalid node identity "
                                                  "signature"})
                        return
                    if not coordinator.bind_identity(data["node_id"], pubkey):
                        self._send(403, {
                            "error": "node id is bound to a different "
                                     "identity",
                            "node_id": data["node_id"]})
                        return
                if not coordinator.register(data["node_id"], data["url"],
                                            data["checkpoint"],
                                            data.get("capability"),
                                            data.get("artifact_sha256"),
                                            data.get("chat"),
                                            data.get("role")):
                    self._send(403, {
                        "error": f"node {data['node_id']!r} was retired by "
                                 "this swarm",
                        "reason": coordinator.blocked.get(data["node_id"])})
                    return
                self._send(200, {"ok": True})
            elif self.path == "/retire":
                # Durable removal: the registry entry goes and the node is
                # blocked, or a stem's own heartbeats undo it in five seconds.
                data = json.loads(raw)
                node_id = data.get("node_id")
                if not node_id:
                    self._send(400, {"error": "node_id is required"})
                    return
                existed = coordinator.retire(node_id,
                                             data.get("reason", "retired"))
                print(f"[retire] {node_id} ({data.get('reason')})"
                      f"{'' if existed else ' (was not registered)'}",
                      flush=True)
                self._send(200, {"ok": True, "blocked": sorted(
                    coordinator.blocked)})
            elif self.path == "/spore/blobs":
                # Delta pulls: return only the chunks a replica asked for and
                # does not already hold (round 107).
                wanted = json.loads(raw).get("hashes", [])
                blobs = getattr(root, "spore_blobs", {}) if root else {}
                missing = [digest for digest in wanted if digest not in blobs]
                if missing:
                    self._send(404, {"error": "unknown chunks",
                                     "missing": missing[:8]})
                    return
                body = b"".join(blobs[digest] for digest in wanted)
                self.send_response(200)
                self.send_header("Content-Type",
                                 "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/decide":
                req = DecisionRequest.from_json(raw)
                try:
                    self._send(200, coordinator.decide(req))
                except Exception as exc:
                    self._send(503, {"error": f"decide failed: {exc}"})
            elif self.path == "/shutdown":
                data = json.loads(raw)
                if root and root._auth_token and \
                        data.get("token") == root._auth_token:
                    print("SHUTDOWN requested via /shutdown (token valid)",
                          flush=True)
                    self._send(200, {"ok": True, "message": "shutting down"})
                    threading.Thread(target=root.stop, daemon=True).start()
                else:
                    self._send(403, {"error": "invalid or missing token"})
            else:
                self._send(404, {"error": "not found"})

    return RootHandler


class Root:
    def __init__(self, coordinator: Coordinator | None = None,
                 host: str = "127.0.0.1", port: int = 0,
                 spore_files: dict | None = None,
                 chunking: str = "fixed"):
        self.coordinator = coordinator or Coordinator()
        self.spore = None
        if spore_files:
            # Persist the spore keypair: an ephemeral key would trip the
            # joiners' TOFU pinning on every root restart (round 16).
            import nacl.signing
            from pathlib import Path as _P
            key_file = _P("spore-cache") / "root-spore-key.hex"
            key_file.parent.mkdir(parents=True, exist_ok=True)
            if key_file.exists():
                key = nacl.signing.SigningKey(
                    bytes.fromhex(key_file.read_text(encoding="utf-8").strip()))
                print("spore key loaded from", key_file, flush=True)
            else:
                key = generate_signing_key()
                key_file.write_text(key.encode().hex(), encoding="utf-8")
                print("spore key generated and saved to", key_file, flush=True)
            self.spore = {"key": key}
            self.spore_pubkey = key.verify_key.encode().hex()
            pb_meta = {"role": "root"}
            if getattr(self, "playbook", None) is not None:
                pb_meta["playbook_hash"] = self.playbook.hash()
                pb_meta["playbook_name"] = self.playbook.name
                pb_meta["playbook_version"] = self.playbook.version
            from .spore import build_manifest_cdc
            if chunking == "cdc":
                # Content-defined chunking lets a joiner that already holds a
                # related lineage fetch only the chunks it lacks (round 107).
                self.spore_manifest = build_manifest_cdc(
                    spore_files, key, metadata=pb_meta)
            else:
                self.spore_manifest = build_manifest(spore_files, key,
                                                     metadata=pb_meta)
            # Signed manifests this root can serve by content hash, so a
            # joiner that already holds a descendant can fetch its parent.
            from .spore import manifest_hash
            self.spore_history: dict[str, dict] = {
                manifest_hash(self.spore_manifest): self.spore_manifest}
            self.spore_chunks = {}
            self.spore_blobs: dict[str, bytes] = {}
            if chunking == "cdc":
                from .spore import chunk_spans
                avg = int(self.spore_manifest["chunking"]["avg_size"])
                for name, path in spore_files.items():
                    data = path.read_bytes()
                    for start, end in chunk_spans(data, avg):
                        chunk = data[start:end]
                        self.spore_blobs[
                            hashlib.sha256(chunk).hexdigest()] = chunk
            else:
                for name, path in spore_files.items():
                    data = path.read_bytes()
                    for i in range(0, len(data), 64 * 1024):
                        self.spore_chunks[(name, i // (64 * 1024))] = \
                            data[i:i + 64 * 1024]
        self.shutdown_event = threading.Event()
        self._auth_token = None
        self._pairing_token = None
        self._require_identity = False
        self._httpd = _PandoHTTPServer(
            (host, port), build_handler(self.coordinator, root=self))
        self.port = self._httpd.server_address[1]
        self._thread: threading.Thread | None = None

    def set_auth_token(self, token: str) -> None:
        self._auth_token = token

    @property
    def pairing_token(self) -> str | None:
        return self._pairing_token

    def set_pairing_token(self, token: str) -> None:
        """Require this secret on every HTTP /register request."""
        if not token:
            raise ValueError("pairing token must not be empty")
        self._pairing_token = token

    @property
    def require_identity(self) -> bool:
        return self._require_identity

    def set_require_identity(self, required: bool = True) -> None:
        """Reject /register without a valid signed node identity."""
        self._require_identity = bool(required)

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="PandoRSI root coordinator")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--serve-spore", default=None,
                        help="payload file to serve for spore pulls")
    parser.add_argument("--chunking", default="fixed",
                        choices=["fixed", "cdc"],
                        help="fixed = v1 manifests; cdc = content-defined "
                             "chunking so an existing replica pulls a delta")
    parser.add_argument("--playbook", default=None,
                        help="playbook TOML to load and serve")
    parser.add_argument("--playbook-key", default=None,
                        help="pinned Ed25519 pubkey (hex) for the playbook; "
                             "with it, an unsigned or invalid playbook is "
                             "refused")
    parser.add_argument("--playbook-signature", default=None,
                        help="detached signature path "
                             "(default: <playbook>.sig)")
    parser.add_argument("--history", default=None,
                        help="playbook history chain to verify against")
    parser.add_argument("--allow-non-head", action="store_true")
    parser.add_argument("--auth-token", default=None,
                        help="enable /shutdown with this token")
    parser.add_argument("--pairing-token", default=None,
                        help="fixed stem admission token; generated when omitted")
    parser.add_argument("--pairing-token-file", default=None,
                        help="read the fixed admission token from a user-owned file")
    parser.add_argument("--rollout-dir", default="rollouts",
                        help="directory for local rollout evidence")
    parser.add_argument("--stem-call-timeout", type=float, default=CALL_TIMEOUT,
                        help="maximum per-stem call timeout")
    parser.add_argument("--quorum-k", type=int, default=QUORUM_K,
                        help="voters per decision; keep it fixed while adding "
                             "replicas or every decision uses every replica")
    parser.add_argument("--adaptive-quorum", action="store_true",
                        help="let the quorum size fall under latency pressure "
                             "and recover when idle (requires "
                             "--latency-target-ms)")
    parser.add_argument("--quorum-floor", type=int, default=1,
                        help="fewest voters the adaptive policy may use")
    parser.add_argument("--latency-target-ms", type=float, default=None,
                        help="decision latency the adaptive quorum aims for")
    parser.add_argument("--lineage-quorum", action="store_true",
                        help="count the quorum over lineages, not stems: "
                             "replicated members collapse to one vote "
                             "(round 119)")
    parser.add_argument("--combiner", default=None,
                        help="a fitted combiner artifact (tools/"
                             "train_combiner.py); its answer is attached as "
                             "suggested_choice wherever the gate does not "
                             "auto-execute (round 140)")
    parser.add_argument("--assent-checkpoint", default=None,
                        help="a named checkpoint whose (collapsed) vote may "
                             "veto an auto tier when it disagrees with the "
                             "aggregate choice; 'auto' picks the present "
                             "checkpoint with the highest mean measured "
                             "accuracy in assets/registry/accuracies.json")
    parser.add_argument("--assent-fallback", default="advisor",
                        choices=["advisor", "veto"],
                        help="what to return after an assent demotion: "
                             "advisor keeps the aggregate choice; veto "
                             "returns the assent member's choice")
    parser.add_argument("--initial-stem-timeout", type=float,
                        default=INITIAL_STEM_TIMEOUT,
                        help="discovery timeout before latency is learned")
    parser.add_argument("--lan", action="store_true",
                        help="bind 0.0.0.0; requires an active lan_listen grant")
    parser.add_argument("--reveal-pairing-token", action="store_true",
                        help="print the full token even when stdout is not a TTY")
    parser.add_argument("--require-identity", action="store_true",
                        help="reject /register without a valid signed node "
                             "identity (pando/identity.py)")
    args = parser.parse_args()
    if args.pairing_token and args.pairing_token_file:
        raise SystemExit("--pairing-token and --pairing-token-file are mutually exclusive")
    if args.pairing_token_file:
        from .common import read_secret_file
        try:
            args.pairing_token = read_secret_file(args.pairing_token_file)
        except ValueError as exc:
            raise SystemExit(str(exc))
    from .common import bypass_proxy_for_lan
    bypass_proxy_for_lan()
    spore_files = None
    if args.serve_spore:
        from pathlib import Path as _P
        from .spore import artifact_files
        payload = _P(args.serve_spore)
        # An exported graph may be a small protobuf plus a multi-gigabyte
        # external weight blob that the protobuf references *by name*, so
        # every file is served under its original name or the replica cannot
        # resolve its weights (round 105).
        spore_files = {file.name: file for file in artifact_files(payload)}
        if args.playbook:
            spore_files["playbook.toml"] = _P(args.playbook)
    listen_host = "127.0.0.1"
    if args.lan:
        from .grants import require_grant
        try:
            require_grant("lan_listen")
        except PermissionError as exc:
            raise SystemExit(f"[locked] {exc}")
        listen_host = "0.0.0.0"
    combiner = None
    if args.combiner:
        from .combiner import Combiner
        try:
            combiner = Combiner.load(args.combiner)
        except (OSError, ValueError, KeyError) as exc:
            raise SystemExit(f"[combiner] {args.combiner}: {exc}")
        print(f"[combiner] {args.combiner} loaded: {combiner.meta}",
              flush=True)
    coordinator = Coordinator(
        call_timeout=args.stem_call_timeout,
        initial_stem_timeout=args.initial_stem_timeout,
        quorum_k=args.quorum_k,
        adaptive_quorum=args.adaptive_quorum,
        quorum_floor=args.quorum_floor,
        latency_target_ms=args.latency_target_ms,
        count_lineages=args.lineage_quorum,
        combiner=combiner,
        rollout_dir=args.rollout_dir,
        assent_checkpoint=args.assent_checkpoint,
        assent_fallback=args.assent_fallback,
        identity_store=Path(args.rollout_dir) / "identity-bindings.json")
    root = Root(coordinator=coordinator, host=listen_host, port=args.port,
                spore_files=spore_files, chunking=args.chunking)
    import secrets
    root.set_pairing_token(args.pairing_token or secrets.token_urlsafe(18))
    if args.require_identity:
        root.set_require_identity(True)
        print("signed node identity is required for /register", flush=True)
    token_digest = hashlib.sha256(
        root.pairing_token.encode("utf-8")).hexdigest()[:16]
    if args.reveal_pairing_token or sys.stdout.isatty():
        print(f"stem pairing token: {root.pairing_token}", flush=True)
    else:
        print(f"stem pairing token fingerprint: sha256:{token_digest}",
              flush=True)
    if args.auth_token:
        root.set_auth_token(args.auth_token)
        print(f"kill-switch enabled (token set)", flush=True)
    if args.playbook:
        from .pb_history import check_integrity
        from .playbook import (load_verified_playbook,
                               resolve_admitted_lineages, resolve_blocs,
                               resolve_combiner)
        try:
            root.playbook, signature_status = load_verified_playbook(
                Path(args.playbook), args.playbook_key,
                args.playbook_signature)
        except ValueError as exc:
            raise SystemExit(f"[playbook] {exc}")
        if not args.playbook_key:
            print(f"[playbook] not pinned: {signature_status} - anyone who "
                  f"can write this file chooses what this swarm admits",
                  flush=True)
        else:
            print(f"[playbook] {signature_status}", flush=True)
        admitted, unusable = resolve_admitted_lineages(root.playbook)
        blocs = resolve_blocs(root.playbook)
        if root.playbook and root.playbook.combiner and combiner is None:
            try:
                combiner = resolve_combiner(root.playbook)
            except (OSError, ValueError, KeyError) as exc:
                raise SystemExit(str(exc))
            root.coordinator.combiner = combiner
            print(f"[combiner] {root.playbook.combiner['path']} loaded from "
                  f"the playbook (pinned sha256 verified)", flush=True)
        if blocs:
            root.coordinator.blocs = blocs
            print(f"[lineages] blocs: "
                  + ", ".join(sorted({f"{name}->{bloc}"
                                      for name, bloc in blocs.items()})),
                  flush=True)
        if root.playbook.lineages:
            if not admitted:
                raise SystemExit(
                    "playbook declares no usable lineage: "
                    f"{sorted(unusable)} have no calibration table, so no "
                    "stem could register")
            root.admitted_lineages = admitted
            artifacts = root.playbook.lineages.get("artifacts") or {}
            root.lineage_artifacts = artifacts
            for name in unusable:
                print(f"[lineages] declared {name!r} has no calibration "
                      f"table and is NOT admitted", flush=True)
            print(f"[lineages] admitting {sorted(admitted)}", flush=True)
            if artifacts:
                attested = sorted(artifacts)
                print(f"[lineages] requiring artifact attestation for "
                      f"{attested}", flush=True)
        pb_hash = root.playbook.hash()
        if args.history:
            ok, problems = check_integrity(Path(args.history))
            if not ok:
                raise SystemExit(
                    f"SECURITY: playbook history chain FAILED verification: {problems}")
            chain = [json.loads(l) for l in
                     Path(args.history).read_text(encoding="utf-8")
                     .splitlines() if l.strip()]
            heads = [chain[-1]["playbook_hash"]] if chain else []
            if args.allow_non_head:
                known = {e["playbook_hash"] for e in chain}
                if pb_hash not in known:
                    raise SystemExit(
                        f"SECURITY: playbook hash {pb_hash[:16]} is not in the history chain - refusing (audited forks must be appended "
                        "to the chain first)")
            elif heads and pb_hash != heads[0]:
                raise SystemExit(
                    f"SECURITY: playbook hash {pb_hash[:16]} is not the chain head ({heads[0][:16]}) - refusing. Use --allow-non-head only "
                    "for audited forks.")
            print(f"playbook hash verified against history chain "
                  f"({len(chain)} entries)", flush=True)
        root.coordinator.gating = root.playbook.gating
        root.coordinator.playbook_hash = root.playbook.hash()
        root.coordinator.known_questions = {
            q["id"] for q in root.playbook.questions}
        root.coordinator.question_categories = (
            root.playbook.question_categories or {})
        print(f"playbook loaded: {root.playbook.name} v{root.playbook.version} "
              f"hash={pb_hash[:16]} "
              f"gating={root.playbook.gating}", flush=True)
    root.start()
    print(f"coordinator listening on {root.url()}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        root.stop()


if __name__ == "__main__":
    main()
