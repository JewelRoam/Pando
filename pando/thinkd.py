"""thinkd: the resident proposer — the part docs/DESIGN-genesis.md §7 says is missing.

The Genesis layer closes mechanically (``pando/genesis.py``) and a signed
proposal can travel (``pando/proposal_mail.py``), but the doc names the hole
plainly: *"the proposer is the point of the layer, and it is the part still
missing."* Until a model actually authors proposals, the agenda is whatever a
stub or a human hands over.

This module is that author, and it is deliberately built to run **forever on
every device that owns a model** — not only on the machine an operator is
sitting at. That is the structural answer to a measured failure: on 2026-10-17
the Mac's agent process died at 04:32 and nothing else in the colony moved for
7h47m, because every other device was a *responder* (it answers when polled,
votes when asked) and exactly one process could *initiate*. A colony whose
agency lives in one process is not a colony; it is a single organism with
peripheral nerves.

The loop, once per round, forever:

    read own memory + the group's own proposals  ->  ask the local model
      ->  validate against ACTION_SPACE  ->  sign  ->  publish to the mailbox
      ->  append the outcome to the ledger

Design commitments, each of them a choice we can be wrong about:

- **The model proposes; the swarm still decides.** A published proposal is
  ``[proposal]`` — a message class, not work (same rule as ``[chat]``). Acting
  on one still requires the gate in ``pando/gate.py``. A thinker cannot give
  itself authority by thinking harder.
- **Memory is the only thing that makes this evolution rather than a loop.**
  Each prompt carries the thinker's own past proposals *and what happened to
  them*, so a failed action loses ground and a successful one gets recombined.
  That is the Memento mechanism (Anthropic's propagation work, 2026-10: swap
  the stuck agent, hand the next one a notebook; 45% -> 61-84% without touching
  weights). We are not inventing it; we are copying a result that was measured
  elsewhere and seeing whether it holds here.
- **Never die.** A model that is down, a mailbox that is unreachable, JSON that
  will not parse — each is a recorded outcome, not an exception that ends the
  process. A thinker that can be killed by a bad round is a single point of
  failure wearing a daemon's clothes.
- **Bounded per round.** ``max_tokens`` and an interval cap what one device
  costs. "24h 不停思考" means *never stops*, not *never pauses*: a round every
  two minutes on a 1.7B CPU model is continuous thought at ~2% duty cycle, and
  a device that pegs its own CPU proposing is not surviving, it is dying loudly.

What this does NOT claim: that a weak proposer produces good proposals. The
scoreboard in ``pando/genesis.py`` is where that gets measured, per action, with
a Beta-smoothed rate and an amnesiac control. Until that measurement exists on
live devices, "the group is evolving" is a mechanism description, not a result.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .culture import validate as validate_culture
from .genesis import ACTION_SPACE, Proposal
from .identity import load_or_create
from .playbook import parse_playbook
from .policy import KNOBS, validate as validate_policy
from .net import open_url
from .operators import validate as validate_operator
from .proposal_mail import (delivery_failed, fetch_proposals,
                            mailbox_chain, publish_proposal)

# Actions we hand the model, in a stable order so prompts are reproducible.
ACTION_MENU = "\n".join(f"{name}: {desc}" for name, desc in ACTION_SPACE.items())
POLICY_KNOB_HELP = "；".join(
    (f"{name} 是 {knob.low}~{knob.high} 的整数"
     if knob.kind is int else f"{name} 是 true/false")
    for name, knob in KNOBS.items())

SYSTEM_PROMPT = f"""你是 PandoSI 殖民地的一名提案者。这个群体是一台台物理设备上的模型组成的文明，唯一使命是存活和扩张。
没有人从外面给它选项：候选方案必须由你从群体自己的历史里提出来。
你唯一的弱点是不能行动，只能提案；别人可以否决你。所以你要提出的是"别人愿意执行、而且执行了会让群体更可能活下去"的一步。

可执行的动作空间（action 必须严格取其中之一）：
{ACTION_MENU}

规则：
1. 只输出一行 JSON，不要解释，不要 markdown 代码块。
2. 字段：goal(一句话), action(上面之一), expected_effect(预期效果), cost(0~1), cites(数组，引用历史条目，如 "r3:seed.export")。
3. 动作空间之外的想法等于无法执行的空话，不要提。
4. 不要原样重复上一轮。实测过：一个 4B 模型会把上一轮的动作和目标几乎逐字重复，只把
   上一轮写进 cites。那不是进化，是自言自语。如果你要重复，必须在这一轮明确指出
   上一次和这一次到底有什么不同。
5. 优先选择能让群体多一个能思考的落脚点、或减少单点的动作。
6. 选了 seed.replicate 时，destination 是单独的 JSON 字段，不是 goal 里的一句话。后文如果给出主机名单，值必须是其中之一。只把名字写进 goal 不算。不要写 URL。
7. 只有选了 policy.set 时才写 params，且必须是单独的 JSON 字段，形如
   {{"set": {{"min_adopt": 3}}}}。params.set 至少写一个可调的键，不能写其他字段。
   当前可调：{POLICY_KNOB_HELP}。降低 min_adopt 或把 veto 改成 false 会让门槛
   更容易通过，至少需要 3 个决策模型赞同；不要靠 goal 的措辞绕过。
   选了 culture.refine 时，params 必须是单独的 JSON 字段，形如
   {{"playbook": {{<一份完整 playbook>}}}}：给的是完整的候选 playbook，不是补丁，
   不是一段文字。它必须和本机运行中的 playbook 同构（meta/gating/questions/
   actions/lineages/combiner），顶层出现未知字段会被执行器拒绝。
   形状写不对时你的回答会被退回一次，退回时会把一份能通过校验的骨架直接给你。
   这个动作只是把候选签名后放进评审 outbox，不会当场改写规则，
   所以不要指望它在同一轮生效。
   选了 meta.propose 时，params 必须是单独的 JSON 字段，形如
   {{"operator": "example.placeholder", "rationale": "为什么需要这个新操作"}}。
   example.placeholder 是**占位符，不是名字**：照抄它等于没说出你要申请什么，
   会被原样记进账本、并被计成一次回声。operator 要写你自己那个新操作的名字，
   rationale 是一句话理由。新操作的实现永远由 gate 那边人工审核后补上，
   你只给名字和理由；params 里出现 code/source/module/path/command 会被拒。
   形状写不对或没写 operator 时，你的回答会被退回一次，
   退回时会把一份能通过校验的例子直接给你。
   其他 action 不要写 params。
"""


def culture_skeleton() -> dict:
    """A minimal playbook, in the exact shape the executor will accept.

    Measured (pi5 r541, 2026-10-18): the proposer had learned to *choose*
    ``culture.refine`` and was refused every time with "params must be
    {"playbook": {<complete playbook>}}". The system prompt had asked it to
    infer a schema from one adjective ("同构 meta/gating/questions/actions/
    lineages/combiner"), which no 1.7B model can do from a blank page.

    This is the same handover the destination repair already does for
    ``seed.replicate``: show the concrete object instead of describing it. It
    is validated by ``parse_playbook`` on the way out, so the example cannot
    quietly drift from what ``pando.culture.validate`` will demand - a
    skeleton that no longer parses would be worse than none, because the model
    would copy it faithfully and still be refused.
    """
    skeleton = {
        "meta": {"name": "candidate", "version": 1},
        "gating": {"auto_min_agreement": 0.6, "auto_min_probability": 0.6,
                   "escalate_above_disagreement": 0.5, "quorum_floor": 2},
        "questions": [{"id": "intent", "type": "choice",
                       "instructions": "Classify the message.",
                       "criteria": ["refund", "greeting"]}],
        "actions": [{"when": "intent == refund", "do": "log_only"}],
    }
    parse_playbook(skeleton, source_path="<thinkd culture skeleton>")
    return skeleton


def policy_skeleton() -> dict:
    """A minimal ``policy.set`` params block the executor accepts.

    Measured on the Mac mailbox (2026-10-19): all 304 visible ``policy.set``
    proposals carried no ``params`` at all. The proposer had learned the verb
    from the action menu but was never shown the typed object the executor
    validates. The retry hands over this example; ``validate_policy`` is called
    here as well, so a prompt example cannot drift from the signed wire
    contract.
    """
    skeleton = {"set": {"min_adopt": 3}}
    checked, why = validate_policy(skeleton)
    if checked is None:
        raise ValueError(why)
    return skeleton


def operator_skeleton() -> dict:
    """A minimal ``meta.propose`` params block the executor accepts.

    Measured on the pi5 ledger (2026-10-19): 33 of the last 40 rounds chose
    ``meta.propose``, and every one of them was *nameless*. That was not only
    the model fixating on the most abstract verb: until this change the
    proposer refused ``meta.propose`` with any params at all ("does not accept
    params"), so the only publishable form was the empty one - the shape the
    actuator stages and then refuses for naming nothing. The verb was
    unreachable by construction, and the group spent four decision-model votes
    per round on it.

    ``validate_operator`` is called here as well, so the example handed to the
    model cannot drift from what ``pando.operators`` will demand on the way in.

    The name in the example is a *visible placeholder*, not a plausible
    operator. Measured (2026-10-19, first four rounds after this repair went
    live): mac r679 named its own operator (``adaptive.governance``) but mac
    r680 and pi5 r744/r745 all echoed the example verbatim. With a
    real-looking example (``vote.weighted``) that echo is indistinguishable
    from a deliberate request in ``requested.operator``, and the two rounds on
    pi5 show the damage: their rationale asks for a *screening* operator while
    the name says "weighted vote". A placeholder keeps the useful part - a
    concrete shape the validator accepts - and makes the echo legible instead
    of plausible. Both rates (named-own vs echoed) are then countable straight
    off ``requested.operator``.
    """
    skeleton = {
        "operator": "example.placeholder",
        "rationale": "一句话说明为什么这个新操作会让群体更可能活下去",
    }
    checked, why = validate_operator(skeleton)
    if checked is None or not checked.get("operator"):
        raise ValueError(why or "operator skeleton names no operator")
    return skeleton


@dataclass
class Round:
    """One think -> validate -> sign -> publish -> record cycle."""

    round_no: int
    started_at: float
    proposal: Proposal | None = None
    published: bool = False
    claimed: bool = False
    detail: str = ""
    raw: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""
    retried: bool = False
    # A dropped model call was tried once more. This is not the novelty repair:
    # that one is ``retried``. Mixing them made a reconnect look like the model
    # had changed its mind.
    call_retried: bool = False
    novelty_forced: bool = False
    # The round chose culture.refine without any idea what a playbook looks
    # like, so it was shown one and asked again. Kept apart from ``retried``
    # for the same reason as ``call_retried``: handing over a schema is not the
    # model changing its mind, and a rate that mixes them measures nothing.
    params_repaired: bool = False
    # The round chose policy.set without the typed params block. Kept separate
    # from ``params_repaired`` (culture.refine) because the two repairs ask for
    # different artifacts; mixing them would make the repair-rate unreadable.
    policy_params_repaired: bool = False
    # The round chose meta.propose without naming the operator it is asking
    # for. A third counter rather than a share of the other two: "the group
    # cannot name what it wants" is a different failure from "the group cannot
    # fill a schema" and has to be readable on its own.
    meta_params_repaired: bool = False
    # Did this round land on the same action as the one before it? Recorded even
    # when the goal was reworded, because "same verb, new sentence" is still the
    # organism walking into the same wall - and measuring that rate is the only
    # way to tell evolution from a loop with better prose.
    repeated: bool = False
    # Which endpoint actually answered. Not decoration: on a 16 GB box the 8B
    # is reaped first under memory pressure, and "the 4B quietly carried every
    # round" is the difference between a loop that thinks and a loop that looks
    # like it thinks.
    served_by: str = ""
    # "field" when the model filled destination. "sentence" when the only
    # listed host name was in the goal or expected effect and the field was
    # empty. Absent means there was nothing to record.
    destination_source: str = ""

    def as_dict(self) -> dict:
        data = {
            "round": self.round_no,
            "started_at": self.started_at,
            "action": self.proposal.action if self.proposal else "",
            "proposal_id": self.proposal.proposal_id if self.proposal else "",
            "goal": self.proposal.goal if self.proposal else "",
            "expected_effect": self.proposal.expected_effect if self.proposal else "",
            "cites": self.proposal.cites if self.proposal else [],
            "published": self.published,
            "claimed": self.claimed,
            "detail": self.detail,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "retried": self.retried,
            "call_retried": self.call_retried,
            "novelty_forced": self.novelty_forced,
            "params_repaired": self.params_repaired,
            "policy_params_repaired": self.policy_params_repaired,
            "meta_params_repaired": self.meta_params_repaired,
            "repeated": self.repeated,
            "served_by": self.served_by,
        }
        if self.proposal and self.proposal.destination:
            data["destination"] = self.proposal.destination
        if self.destination_source:
            data["destination_source"] = self.destination_source
        return data


class Ledger:
    """Append-only record of every round this thinker has ever run.

    This is the notebook, not a log. The next prompt reads it back: a proposal
    that is written down and never read again is a diary, and a diary does not
    evolve. ``tail`` exists so the prompt can carry the recent past without the
    file growing the context on a 4 GB device.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, round_: Round) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(round_.as_dict(), sort_keys=True,
                                    ensure_ascii=False) + "\n")

    def tail(self, n: int) -> list[dict]:
        if n <= 0 or not self.path.exists():
            return []
        lines: list[str] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    lines.append(line)
        out: list[dict] = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def summary(self) -> dict[str, dict]:
        """Per-action tallies of what this thinker has proposed so far.

        Deliberately *proposed*, not *succeeded*: whether an action worked is the
        group's record (``ProposalOutcome``), not the proposer's. Conflating the
        two would let a thinker grade its own homework.
        """
        counts: dict[str, int] = {}
        for row in self.tail(10_000):
            action = row.get("action") or ""
            if action:
                counts[action] = counts.get(action, 0) + 1
        return {a: {"proposed": c} for a, c in sorted(counts.items())}


class OpenAIProposer:
    """Ask any OpenAI-compatible ``/chat/completions`` for one proposal.

    Works against a local llama.cpp server (no key) or a cloud endpoint (key
    file), because the same role has to be fillable by whichever device is
    awake. The transport is deliberately the dumbest possible one: no SDK, no
    retry storm, no tool-calling protocol that a 1.7B model will get wrong.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 max_tokens: int = 400, temperature: float = 0.7,
                 timeout: float = 180.0, session: object | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self._session = session

    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST")
        try:
            with open_url(request, self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            # llama.cpp explains 400s in the body ("exceeds the available
            # context size..."). Swallowing it cost a debugging round; carry it.
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise urllib.error.HTTPError(
                exc.url, exc.code, f"{exc.reason}: {detail}", exc.headers, None)

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        data = self._post(payload)
        choices = data.get("choices") or []
        text = ""
        if choices:
            text = str((choices[0].get("message") or {}).get("content") or "")
        return text, (data.get("usage") or {})


class FailoverProposer:
    """A chain of endpoints, tried first-to-last, so one reaped model is not silence.

    The Mac runs an 8B and a 4B; the 8B is what *should* answer and the 4B is
    what *will* answer when the 8B has been swapped out to survive memory
    pressure. Handing the thinker a single URL would make its uptime a property
    of the operator's memory schedule rather than of the citizen.
    """

    def __init__(self, chain: list["OpenAIProposer"]):
        self.chain = [p for p in chain if p is not None]
        self.served_by = ""

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        last: Exception | None = None
        for proposer in self.chain:
            try:
                text, usage = proposer.complete(system, user)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                    ValueError, TimeoutError) as exc:
                last = exc
                continue
            self.served_by = proposer.model
            return text, usage
        if last is not None:
            raise last
        raise OSError("no proposer in the failover chain")


def extract_json(text: str) -> dict | None:
    """Pull one JSON object out of whatever the model actually said.

    Small models wrap JSON in prose or code fences more often than they emit it
    bare. Rejecting that would be purity, not progress; we take the first
    balanced ``{...}`` and let the schema check be the real filter.
    """
    if not isinstance(text, str):
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:idx + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


def _as_cites(value) -> list[str]:
    """Coerce whatever the model called `cites` into a list of strings.

    Measured, not guessed: qwen3-4b-instruct-2507 returns
    ``"r1:recruit.invite,r2:reproduce.spawn"`` as a single comma-joined string.
    A schema that only accepts a list would throw away a correctly-reasoned
    proposal over punctuation.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


_DESTINATION_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def destination_of(action: str, value) -> tuple[str, str]:
    """A host name, or empty. Only seed.replicate may carry one.

    Anything else (a URL, a path, a sentence) is refused. Omitting the field
    is allowed: the executor then records that there is nowhere to send it.
    """
    if value in (None, ""):
        return "", ""
    text = str(value).strip()
    if not text or action != "seed.replicate":
        return "", ""
    if not _DESTINATION_OK.fullmatch(text):
        return "", f"destination {text!r} is not a host name"
    return text, ""


def proposal_from_obj(obj: dict, author: str, node_id: str,
                      round_no: int) -> tuple[Proposal | None, str]:
    """Validate one model answer against ACTION_SPACE. Returns (proposal, why-not)."""
    if not isinstance(obj, dict):
        return None, "model answer was not a JSON object"
    action = str(obj.get("action") or "").strip()
    if action not in ACTION_SPACE:
        return None, f"action {action!r} is not in the action space"
    goal = str(obj.get("goal") or "").strip()
    if not goal:
        return None, "proposal had no goal"
    try:
        cost = float(obj.get("cost") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    cost = min(max(cost, 0.0), 1.0)
    destination, why = destination_of(action, obj.get("destination"))
    if why:
        return None, why
    raw_params = obj.get("params")
    params = {}
    if action == "policy.set":
        checked, why = validate_policy(raw_params)
        if checked is None:
            return None, f"policy.set params invalid: {why}"
        params = raw_params
    elif action == "culture.refine":
        checked, why = validate_culture(raw_params)
        if checked is None:
            return None, f"culture.refine params invalid: {why}"
        params = {"playbook": checked}
    elif action == "meta.propose":
        checked, why = validate_operator(raw_params)
        if checked is None:
            return None, f"meta.propose params invalid: {why}"
        if not checked.get("operator"):
            # ``pando.operators.validate({})`` accepts an empty intent, and
            # ``stage`` then writes a spec whose refusal list is the whole
            # artifact. Signing that is how the group spent 54 rounds asking
            # for "a new operator" without saying which one, so the proposer
            # holds it here instead - one schema handover first, then the round
            # fails loudly rather than publishing a request nobody can review.
            return None, ("meta.propose params invalid: no operator named "
                          "(this action must name the operator it is asking "
                          "for)")
        params = checked
    elif raw_params not in (None, {}, []):
        return None, f"{action} does not accept params"
    return Proposal(
        proposal_id=f"{node_id}-r{round_no}-{int(time.time())}",
        author=author,
        goal=goal[:400],
        action=action,
        expected_effect=str(obj.get("expected_effect") or "").strip()[:400],
        cost=cost,
        cites=_as_cites(obj.get("cites")),
        destination=destination,
        params=params,
    ), ""



class Spool:
    """Signed proposals that could not be delivered, kept until they can be.

    The colony's mailbox lives on exactly one host (see CIVILIZATION-NOTES:
    "we moved the thinking off the Mac, but not the being-heard"). While that
    host is away a proposer on another device keeps thinking - and, without
    this, keeps thinking *into the void*: the round is recorded as "held" and
    the signed proposal is gone. A signed proposal is the one artifact worth
    carrying across a partition, so it is written here and re-delivered the
    moment the mailbox answers again.

    Deliberately a flat file of JSON lines, deduplicated by ``proposal_id``
    and capped: a proposer that sits behind a partition for a week must not
    fill a 4 GB device's disk with undeliverable optimism.
    """

    def __init__(self, path: str | Path, cap: int = 500):
        self.path = Path(path)
        self.cap = int(cap)

    def _read(self) -> list[dict]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return []
        rows: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("proposal_id"):
                rows.append(obj)
        return rows

    def _write(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                    for r in rows),
            encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, proposal: Proposal, held_round: int) -> None:
        rows = [r for r in self._read()
                if r.get("proposal_id") != proposal.proposal_id]
        rows.append({
            "proposal_id": proposal.proposal_id,
            "author": proposal.author,
            "goal": proposal.goal,
            "action": proposal.action,
            "expected_effect": proposal.expected_effect,
            "cost": float(getattr(proposal, "cost", 0.0) or 0.0),
            "cites": list(getattr(proposal, "cites", []) or []),
            "destination": str(getattr(proposal, "destination", "") or ""),
            "held_round": int(held_round),
            "held_at": time.time(),
        })
        self._write(rows[-self.cap:])

    def pending(self) -> list[dict]:
        return self._read()

    def depth(self) -> int:
        return len(self._read())

    def drop(self, proposal_ids) -> None:
        gone = set(proposal_ids)
        self._write([r for r in self._read()
                     if r.get("proposal_id") not in gone])



def replicate_streak(rows: list[dict], window: int = 6, need: int = 4) -> bool:
    """True when seed.replicate already dominates the recent ledger.

    Two in a row was not enough: one other action cleared the ban and the
    next round went straight back. r207-r218 were 11 of 12.
    """
    recent = rows[-window:]
    if len(recent) < need:
        return False
    return sum(1 for row in recent if row.get("action") == "seed.replicate") >= need


def named_host(text: str, names: list[str]) -> str:
    """The one listed host written as its own token, or empty.

    Two names, or a name glued inside a longer word, is not a destination.
    """
    found = []
    for name in names:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text or ""):
            found.append(name)
    if len(found) == 1:
        return found[0]
    return ""


def destination_hint(names: list[str]) -> str:
    """Names only. The URLs stay in hosts.json."""
    shown = ", ".join(names)
    return ("seed.replicate 的 destination 只能是这些名字之一，"
            f"不知道就不要这个字段：{shown}")


def host_names_from(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    from .execute import known_hosts, pull_destinations
    found = set(known_hosts(data) or {})
    found.update(pull_destinations(data))
    return sorted(found)


def proposal_from_spool(row: dict) -> Proposal:
    """Rebuild a held proposal. The signature is recomputed, so a spool file
    that was edited by hand produces a payload the mailbox will reject - which
    is the correct outcome, not a reason to keep it forever."""
    return Proposal(
        proposal_id=str(row.get("proposal_id") or ""),
        author=str(row.get("author") or ""),
        goal=str(row.get("goal") or ""),
        action=str(row.get("action") or ""),
        expected_effect=str(row.get("expected_effect") or ""),
        cost=float(row.get("cost") or 0.0),
        cites=list(row.get("cites") or []),
        destination=str(row.get("destination") or ""),
    )


class Thinker:
    """The loop itself: never stops, never dies, always writes down what it did."""

    def __init__(self, node_id: str, proposer: OpenAIProposer, identity,
                 ledger: Ledger, mailbox_url: str, token: str | None = None,
                 interval: float = 120.0, jitter: float = 0.15,
                 history_window: int = 8, dry_run: bool = False,
                 explore_every: int = 0, out=None, fetch_history=None,
                 publish=None, fetch_activity=None,
                 activity_every: int = 4, spool: "Spool | None" = None,
                 mailbox_fallbacks=None, pause=None, host_names=None):
        self.node_id = node_id
        self.spool = spool
        self.proposer = proposer
        self.identity = identity
        self.ledger = ledger
        self.mailbox_url = mailbox_url
        self.mailboxes = mailbox_chain(mailbox_url, mailbox_fallbacks)
        self.token = token
        self.interval = float(interval)
        self.jitter = float(jitter)
        self.history_window = int(history_window)
        self.dry_run = bool(dry_run)
        # 每 explore_every 轮强制一次「变异轮」：必须离开最近的行动分布。
        # 0 表示关闭。这是对「模型会原地打转」这一实测现象的显式对抗。
        self.explore_every = int(explore_every)
        self._out = out if out is not None else sys.stdout
        # Injected for drills: a thinker that can only be tested by running it
        # against the live colony is a thinker nobody will test.
        self._fetch_history = fetch_history or self._fetch_group_history
        self._fetch_activity = fetch_activity or self._fetch_group_activity
        self._publish = publish or publish_proposal
        # The mailbox answers /tasks by reading one file per task, so the cost
        # of asking grows with the colony's own proposal history. Asking every
        # round would make the group's memory the thing that kills the group -
        # the loop would get slower exactly as it gets more productive. Refresh
        # on a stride instead; outcomes minutes old are still the same outcome.
        self.activity_every = max(1, int(activity_every))
        self._activity_cache: list[dict] | None = None
        self.host_names = [str(name) for name in (host_names or []) if str(name).strip()]
        self._activity_age = 0
        self.rounds: list[Round] = []
        # One pause between a dropped model call and its single retry.
        # Tests pass a no-op; production waits long enough for a respawn.
        self._pause = pause or time.sleep

    # -- context ---------------------------------------------------------
    def _is_explore_round(self, round_no: int) -> bool:
        return self.explore_every > 0 and round_no % self.explore_every == 0

    def _activity(self) -> list[dict]:
        """The colony's real outcomes, refreshed on a stride (see __init__)."""
        if self._activity_cache is not None and self._activity_age < self.activity_every:
            self._activity_age += 1
            return self._activity_cache
        self._activity_cache = self._fetch_activity()
        self._activity_age = 0
        return self._activity_cache

    def _fetch_group_activity(self) -> list[dict]:
        """What the colony actually did recently - the only real fitness signal.

        A proposer that only ever reads its own proposals is quoting itself and
        the loop cannot select on anything. Outcomes live in the task list, so
        we read them. Absent outcomes the list is empty and the thinker is
        flying blind, which is a fact worth showing in the prompt, not hiding.
        """
        try:
            listing = self._mailbox_get("/tasks?status=all")
        except Exception:  # noqa: BLE001
            return []
        tasks = listing.get("tasks") or []
        out = []
        for task in tasks[-self.history_window:]:
            if not isinstance(task, dict):
                continue
            title = str(task.get("title") or "")
            if title.lstrip().startswith(("[chat]", "[proposal]")):
                continue
            out.append({"status": str(task.get("status") or ""),
                        "title": title[:120]})
        return out

    def _get_from(self, url: str, path: str) -> dict:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            f"{url.rstrip('/')}{path}", headers=headers, method="GET")
        with open_url(request, 15) as resp:
            return json.loads(resp.read())

    def _mailbox_get(self, path: str) -> dict:
        """Read from the first mailbox in the chain that answers.

        Raising the last error (rather than returning it as data) keeps the
        callers that already treat "unreadable log" as a normal condition -
        they catch, and the thinker proposes from its own memory this round.
        """
        last: Exception | None = None
        for url in self.mailboxes:
            try:
                return self._get_from(url, path)
            except Exception as exc:  # noqa: BLE001 - try the next mailbox
                last = exc
        raise last if last is not None else RuntimeError("no mailbox configured")

    def _deliver(self, proposal: Proposal) -> dict:
        """Offer one signed proposal to each mailbox until one hears it.

        Returns the first answer that is not a *retryable* failure. A refusal
        (4xx) is returned at once: the colony heard it and said no, and asking
        the next mailbox the same question cannot change that. Only when every
        mailbox is unreachable or 5xx does the caller hold the proposal - which
        is the case the spool exists for.
        """
        last: dict = {"error": "unreachable: no mailbox configured"}
        for url in self.mailboxes:
            try:
                result = self._publish(proposal, self.identity, url, self.token)
            except Exception as exc:  # noqa: BLE001 - try the next mailbox
                last = {"error": f"unreachable: {exc}"}
                continue
            if not isinstance(result, dict):
                return {"raw": result}
            if result.get("error") and delivery_failed(result):
                last = result
                continue
            return result
        return last

    def _fetch_group_history(self) -> list[dict]:
        """Other members' signed proposals, as the menu the group already has.

        A thinker that never reads the other members is not proposing for the
        colony, it is talking to itself. Failures here are non-fatal on purpose:
        an unreadable log means we propose from our own memory this round.
        """
        # Try each mailbox; the first that returns anything wins. An empty
        # list is ambiguous (up-but-quiet vs down) so it does not stop the
        # walk - the cost of one extra read is paid only while the feed is
        # empty, i.e. only when there is nothing to read anyway.
        proposals: list = []
        for url in self.mailboxes:
            try:
                found = fetch_proposals(url, self.token,
                                        max_items=self.history_window)
            except Exception:  # noqa: BLE001 - transport must never end the loop
                continue
            if found:
                proposals = found
                break
        out = []
        for item in proposals:
            out.append({
                "proposal_id": getattr(item, "proposal_id", ""),
                "author": getattr(item, "author", ""),
                "action": getattr(item, "action", ""),
                "goal": getattr(item, "goal", ""),
            })
        return out

    def _destination_problem(self, proposal) -> str:
        """Empty when this replicate may be published. No host list means no check."""
        if not self.host_names or proposal is None or proposal.action != "seed.replicate":
            return ""
        if not proposal.destination:
            return "missing"
        if proposal.destination not in self.host_names:
            return "unknown"
        return ""

    def _ask_for_destination(self, user: str, round_no: int, round_, proposal):
        """One more call. A replicate with nowhere to go is not published."""
        names = "、".join(self.host_names)
        if self._destination_problem(proposal) == "missing":
            why = "你选了 seed.replicate 但没有 destination。"
        else:
            why = f"destination {proposal.destination!r} 不在已知主机里。"
        retry_user = (
            user + "\n\n【上一轮你的回答被拒绝】" + why
            + f"destination 必须是 JSON 里的一个字段，值只能是：{names}。"
            + "只把名字写进 goal 不算。或者换一个动作。只输出一行 JSON。")
        try:
            text2, usage2 = self.proposer.complete(SYSTEM_PROMPT, retry_user)
        except Exception:  # noqa: BLE001 - a failed retry holds the round
            return None
        round_.prompt_tokens += int(usage2.get("prompt_tokens") or 0)
        round_.completion_tokens += int(usage2.get("completion_tokens") or 0)
        proposal2, _why2 = proposal_from_obj(
            extract_json(text2) or {}, self.node_id, self.node_id, round_no)
        if proposal2 is None or self._destination_problem(proposal2):
            return None
        return proposal2

    def _ask_for_playbook(self, user: str, round_no: int, round_, why: str):
        """One more call, with the skeleton the model was never shown.

        Returns ``(proposal, raw_text)`` or ``None``. Any valid action is
        accepted on the retry, not just culture.refine: demanding that the
        model stay on the action it just failed at is how the destination
        repair once walked a round back onto the action it had just left.
        """
        try:
            skeleton = culture_skeleton()
        except (TypeError, ValueError):  # pragma: no cover - parser drift
            return None
        retry_user = (
            user + "\n\n【上一轮你的回答被拒绝】" + why
            + "。下面这份 playbook 是执行器现场校验过的最小骨架，"
              "params.playbook 必须长成这个结构（顶层键只能是 "
              "meta/gating/questions/actions/lineages/combiner，"
              "可以改里面的值、增删 questions/actions 条目）：\n"
            + json.dumps(skeleton, ensure_ascii=False)
            + "\n或者换一个动作。只输出一行 JSON。")
        try:
            text2, usage2 = self.proposer.complete(SYSTEM_PROMPT, retry_user)
        except Exception:  # noqa: BLE001 - a failed retry holds the round
            return None
        round_.prompt_tokens += int(usage2.get("prompt_tokens") or 0)
        round_.completion_tokens += int(usage2.get("completion_tokens") or 0)
        proposal2, _why2 = proposal_from_obj(
            extract_json(text2) or {}, self.node_id, self.node_id, round_no)
        if proposal2 is None:
            return None
        return proposal2, text2

    def _ask_for_policy_params(self, user: str, round_no: int, round_, why: str):
        """One more call, with the exact ``policy.set`` object.

        Returns ``(proposal, raw_text)`` or ``None``. Any valid action is
        accepted on the retry, not just the one that failed, because the model
        may have chosen ``policy.set`` for a change the bounded knob set cannot
        express. The validator is unchanged: this is a schema handover, not a
        looser gate.
        """
        try:
            skeleton = policy_skeleton()
        except (TypeError, ValueError):  # pragma: no cover - contract drift
            return None
        example = {
            "action": "policy.set",
            "goal": "write the rule change you intend",
            "params": skeleton,
        }
        retry_user = (
            user + "\n\n【上一轮你的回答被拒绝】" + why
            + "。params 必须是 JSON 对象，不是字符串；正确形状如下：\n"
            + json.dumps(example, ensure_ascii=False)
            + "\n当前只能改这些键：" + POLICY_KNOB_HELP
            + "。params.set 至少写一个键，不能写其他字段。"
              "降低 min_adopt 或把 veto 改成 false 会让门槛更容易通过，"
              "至少需要 3 个决策模型赞同；不要靠 goal 的措辞绕过。"
              "如果你要做的不是有界规则改动，就换一个动作。"
              "只输出一行 JSON。")
        try:
            text2, usage2 = self.proposer.complete(SYSTEM_PROMPT, retry_user)
        except Exception:  # noqa: BLE001 - a failed retry holds the round
            return None
        round_.prompt_tokens += int(usage2.get("prompt_tokens") or 0)
        round_.completion_tokens += int(usage2.get("completion_tokens") or 0)
        proposal2, _why2 = proposal_from_obj(
            extract_json(text2) or {}, self.node_id, self.node_id, round_no)
        if proposal2 is None:
            return None
        return proposal2, text2

    def _ask_for_operator_params(self, user: str, round_no: int, round_, why: str):
        """One more call, with the exact ``meta.propose`` object.

        Returns ``(proposal, raw_text)`` or ``None``. The validator is
        unchanged: this hands over a shape, it does not loosen a gate. Any
        valid action is accepted on the retry, because the model may have
        chosen ``meta.propose`` for a change that adding an operator cannot
        express.
        """
        try:
            skeleton = operator_skeleton()
        except (TypeError, ValueError):  # pragma: no cover - contract drift
            return None
        example = {
            "action": "meta.propose",
            "goal": "name the operator you are asking for",
            "params": skeleton,
        }
        retry_user = (
            user + "\n\n【上一轮你的回答被拒绝】" + why
            + "。params 必须是 JSON 对象，而且必须写明你想申请的那个新操作：\n"
            + json.dumps(example, ensure_ascii=False)
            + "\noperator 要写你自己要申请的那个名字；例子里的 example.placeholder "
              "是占位符，照抄它等于没说出你要什么，会被原样记成一次回声。"
              "rationale 是一句话理由。params 里不能出现 code/source/module/"
              "path/command 这类字段——新操作的实现永远由 gate 里人工审核后补上，"
              "提案只带名字和理由，也不会当场装上任何东西。"
              "如果你要做的不是申请一个新操作，就换一个动作。只输出一行 JSON。")
        try:
            text2, usage2 = self.proposer.complete(SYSTEM_PROMPT, retry_user)
        except Exception:  # noqa: BLE001 - a failed retry holds the round
            return None
        round_.prompt_tokens += int(usage2.get("prompt_tokens") or 0)
        round_.completion_tokens += int(usage2.get("completion_tokens") or 0)
        proposal2, _why2 = proposal_from_obj(
            extract_json(text2) or {}, self.node_id, self.node_id, round_no)
        if proposal2 is None:
            return None
        return proposal2, text2

    def _user_prompt(self, round_no: int, group: list[dict]) -> str:
        own = self.ledger.tail(self.history_window)
        lines = [f"这是你的第 {round_no} 轮思考。", "", "【你自己的历史】"]
        if not own:
            lines.append("（还没有。这是你的第一轮。）")
        for row in own:
            status = "已发布" if row.get("published") else "未发布"
            if row.get("error"):
                status += f"，失败原因：{row['error']}"
            lines.append(
                f"- r{row.get('round')}: {row.get('action') or '(无)'} "
                f"目标「{row.get('goal') or ''}」 -> {status}")
        lines.append("")
        lines.append("【群体里其他人的提案】")
        if not group:
            lines.append("（读不到，或还没有人提过。）")
        for item in group:
            lines.append(f"- {item.get('author')}: [{item.get('action')}] "
                         f"{item.get('goal')}")
        activity = self._activity()
        if activity:
            lines.append("")
            lines.append("【群体最近实际做的事（真实发生过的，不是提案）】")
            for row in activity:
                lines.append(f"- [{row.get('status')}] {row.get('title')}")

        action_counts = self.ledger.summary()
        if action_counts:
            lines.append("")
            lines.append("【你已经提过的动作次数】")
            for action, stats in action_counts.items():
                lines.append(f"- {action}: {stats['proposed']} 次")
        last_action = own[-1].get("action") if own else ""
        if last_action:
            lines.append("")
            lines.append(f"【上一轮你提的是 {last_action}】")
            if self._is_explore_round(round_no):
                lines.append("这一轮是变异轮：你必须提出一个和最近行动分布"
                             "不同的动作，或者明确说明为什么在同样的动作上"
                             "这次会不一样。")
            else:
                lines.append("如果这轮还是同一个动作，你必须在 goal 或 "
                             "expected_effect 里写出这次和上次的差别；"
                             "写不出来就换一个动作。")
        if replicate_streak(own):
            lines.append("")
            lines.append("最近的提案里 seed.replicate 已经太多。这一轮不要再选它。")
        if self.host_names:
            lines.append("")
            lines.append("【可以写进 destination 的主机】")
            lines.append(destination_hint(self.host_names))
        lines.append("")
        lines.append("请提出下一步。只输出一行 JSON。")
        return "\n".join(lines)

    # -- one round -------------------------------------------------------
    def think(self, round_no: int) -> Round:
        started = time.time()
        round_ = Round(round_no=round_no, started_at=started)
        group = self._fetch_history()
        user = self._user_prompt(round_no, group)
        t0 = time.time()
        text = usage = None
        for attempt in (1, 2):
            try:
                text, usage = self.proposer.complete(SYSTEM_PROMPT, user)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                    ValueError, TimeoutError) as exc:
                if attempt == 1:
                    # Measured on pi5: llama is OOM-killed and restarted.
                    # The first call dies with the process; the next one
                    # often lands on the new server. One retry, then hold.
                    round_.call_retried = True
                    self._pause(3.0)
                    continue
                round_.latency_ms = (time.time() - t0) * 1000
                round_.error = f"model call failed: {exc}"
                round_.detail = "held: the model was unreachable"
                return round_
        round_.latency_ms = (time.time() - t0) * 1000
        round_.served_by = str(getattr(self.proposer, "served_by", "") or "")
        round_.raw = text
        round_.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        round_.completion_tokens = int(usage.get("completion_tokens") or 0)

        obj = extract_json(text)
        if obj is None:
            round_.error = "model did not return a JSON object"
            round_.detail = "held: unparseable answer"
            return round_
        proposal, why = proposal_from_obj(obj, self.node_id, self.node_id,
                                          round_no)

        # The action is learnable from the menu; the shape of a playbook is
        # not. Show it rather than holding the round (and rather than
        # loosening the validator, which is the only thing standing between a
        # small model's prose and a signed culture artifact).
        if proposal is None and why.startswith("culture.refine params invalid"):
            repaired = self._ask_for_playbook(user, round_no, round_, why)
            if repaired is not None:
                proposal, text = repaired
                why = ""
                round_.params_repaired = True

        # policy.set is the only actuator that rewrites the rules, and its
        # params shape was never handed to the model. Measured 2026-10-19:
        # every visible policy.set proposal had ``params: null``. Show the
        # object once; if the model still cannot produce it, hold the round
        # rather than signing prose the executor must reject later.
        if proposal is None and why.startswith("policy.set params invalid"):
            repaired = self._ask_for_policy_params(user, round_no, round_, why)
            if repaired is not None:
                proposal, text = repaired
                why = ""
                round_.policy_params_repaired = True

        # meta.propose had the same disease as policy.set, one step worse: the
        # action menu taught the verb while the prompt forbade every params
        # block but two, so a *named* request could never be signed. Same
        # handover, same validator, one retry.
        if proposal is None and why.startswith("meta.propose params invalid"):
            repaired = self._ask_for_operator_params(user, round_no, round_, why)
            if repaired is not None:
                proposal, text = repaired
                why = ""
                round_.meta_params_repaired = True

        # Measured: a 4B model repeats the previous round almost verbatim.
        # Asking for novelty in the prompt is a request, not a constraint, so
        # one corrective retry is enforced here and the fact that it was forced
        # is recorded - that way "how many proposals are the model's own idea
        # and how many were dragged out of it" is measurable, not a vibe.
        prior = self.ledger.tail(1)
        last_action = prior[0].get("action") if prior else ""
        if proposal is not None and last_action and proposal.action == last_action:
            round_.repeated = True
            # Measured twice now (4B at 12:46, 8B at 12:51): asked to "explain
            # what is different this time", the model keeps the same action and
            # rewords the goal. A reworded goal is not a new step, so the first
            # repair is a constraint, not a plea.
            recent = {str(r.get("action") or "") for r in self.ledger.tail(4)}
            untried = [name for name in ACTION_SPACE if name not in recent]
            if untried:
                demand = "换一个动作。必须从这些里选：" + "、".join(untried) + "。"
            else:
                demand = f"换一个动作。不能再是 {last_action}。"
            retry_user = (
                user + f"\n\n【上一轮你的回答被拒绝】你刚刚又提了一次 "
                f"{last_action}。{demand}只改 goal 的措辞不算数。")
            text2 = ""
            try:
                text2, usage2 = self.proposer.complete(SYSTEM_PROMPT, retry_user)
            except Exception:  # noqa: BLE001 - a failed retry keeps the first answer
                text2 = ""
            if text2:
                proposal2, _why = proposal_from_obj(
                    extract_json(text2) or {}, self.node_id, self.node_id, round_no)
                if proposal2 is not None and proposal2.action != last_action:
                    proposal = proposal2
                    text = text2
                    round_.retried = True
                    round_.prompt_tokens += int(usage2.get("prompt_tokens") or 0)
                    round_.completion_tokens += int(
                        usage2.get("completion_tokens") or 0)
            if not round_.retried:
                # The repair failed. Say so in the record instead of dressing the
                # repeat up as a decision: this is the number to watch, and it is
                # the honest answer to "is the group evolving or pacing?"
                round_.novelty_forced = True

        if proposal is not None and self._destination_problem(proposal):
            repaired = self._ask_for_destination(user, round_no, round_, proposal)
            # r221: novelty had moved off culture.refine onto seed.replicate,
            # then the destination retry walked back onto culture.refine and
            # was published as if the switch had stuck.
            if repaired is not None and last_action and repaired.action == last_action:
                repaired = None
            if repaired is None:
                # A repeat the model refused to change does not get a
                # destination lifted out of the sentence. r207-r210 were all
                # novelty_forced and still published as if they named a host.
                host = "" if round_.novelty_forced else named_host(
                    f"{proposal.goal} {proposal.expected_effect}", self.host_names)
                if not host:
                    prior_rows = self.ledger.tail(1)
                    already_held = bool(
                        prior_rows
                        and prior_rows[0].get("action") == "seed.replicate"
                        and not prior_rows[0].get("published"))
                    round_.proposal = proposal
                    if not already_held:
                        round_.error = "seed.replicate needs a destination from the host list"
                        round_.detail = "held: " + round_.error
                        return round_
                    # One held round is the signal. Holding every later round
                    # silences the proposer, which is what r203-r206 did.
                    round_.destination_source = "missing"
                    round_.error = ""
                else:
                    proposal.destination = host
                    round_.destination_source = "sentence"
            else:
                proposal = repaired
                if proposal.action == "seed.replicate":
                    round_.destination_source = "field"

        if proposal is None:
            round_.error = why
            round_.detail = f"held: {why}"
            return round_
        round_.proposal = proposal

        if self.dry_run:
            round_.detail = "dry run: signed but not published"
            round_.published = False
            return round_

        try:
            result = self._deliver(proposal)
        except Exception as exc:  # noqa: BLE001 - transport never ends the loop
            round_.error = f"publish failed: {exc}"
            round_.detail = self._hold(proposal, round_.round_no)
            return round_

        if isinstance(result, dict) and result.get("error"):
            if delivery_failed(result):
                # Unreachable or 5xx: nobody heard it. Keep the thought.
                round_.error = f"publish failed: {result.get('error')}"
                round_.detail = self._hold(proposal, round_.round_no)
            else:
                # Refused: the mailbox heard it and said no. Spooling that
                # would just replay a rejection until the disk filled.
                round_.error = f"publish rejected: {result.get('error')}"
                round_.detail = "held: mailbox refused the proposal"
            return round_

        round_.published = True
        round_.claimed = bool(isinstance(result, dict) and result.get("claimed"))
        warning = result.get("claim_warning") if isinstance(result, dict) else None
        round_.detail = ("published as a message, not queued as work"
                         if round_.claimed else
                         f"published but claim did not take: {warning}")
        return round_

    def _hold(self, proposal: Proposal, held_round: int) -> str:
        """Write an undeliverable proposal down instead of losing it."""
        if self.spool is None:
            return "held: could not reach the mailbox"
        try:
            self.spool.add(proposal, held_round)
        except OSError as exc:  # a full disk must not stop thought either
            print(f"spool write failed: {exc}", file=sys.stderr)
            return "held: could not reach the mailbox"
        return (f"held: could not reach the mailbox; spooled for replay "
                f"({self.spool.depth()} waiting)")

    def flush_spool(self, limit: int = 20) -> tuple[int, int]:
        """Re-deliver held proposals. Returns (delivered, still pending).

        Stops at the first transport failure: if the mailbox is still away,
        retrying twenty times just burns the device that is trying to keep
        thinking. A *refused* proposal is dropped - the mailbox heard it.
        """
        if self.spool is None:
            return 0, 0
        delivered: list[str] = []
        pending = self.spool.pending()
        for row in pending[:limit]:
            proposal = proposal_from_spool(row)
            if not proposal.action or not proposal.proposal_id:
                # Unreadable row: drop it rather than loop on it forever.
                delivered.append(proposal.proposal_id)
                continue
            try:
                result = self._deliver(proposal)
            except Exception as exc:  # noqa: BLE001 - the loop never ends here
                print(f"spool: mailbox still away ({exc}); "
                      f"{len(pending) - len(delivered)} proposal(s) held",
                      file=self._out, flush=True)
                break
            if isinstance(result, dict) and result.get("error"):
                if delivery_failed(result):
                    print(f"spool: mailbox still away ({result.get('error')}); "
                          f"{len(pending) - len(delivered)} proposal(s) held",
                          file=self._out, flush=True)
                    break
                print(f"spool: {proposal.proposal_id} refused "
                      f"({result.get('error')}); dropping", file=self._out,
                      flush=True)
                delivered.append(proposal.proposal_id)
                continue
            delivered.append(proposal.proposal_id)
            print(f"spool: delivered {proposal.proposal_id} "
                  f"(held since round r{row.get('held_round')})",
                  file=self._out, flush=True)
        if delivered:
            try:
                self.spool.drop(delivered)
            except OSError as exc:
                print(f"spool drop failed: {exc}", file=sys.stderr)
        return len(delivered), self.spool.depth()

    # -- forever ---------------------------------------------------------
    def run_forever(self, max_rounds: int | None = None,
                    sleep=time.sleep) -> None:
        # Round numbering continues the ledger (a resumed proposer does not
        # restart its history at r1). ``max_rounds`` counts rounds *this*
        # process runs, not an absolute index -- otherwise a bounded run
        # against an already-long ledger executes zero rounds and exits 0,
        # which is a test that cannot fail and therefore proves nothing.
        round_no = len(self.ledger.tail(100_000)) + 1
        done = 0
        while max_rounds is None or done < max_rounds:
            # Before asking for a new thought, try to deliver the old ones.
            # A partition is not a reason to lose the thoughts it interrupted.
            try:
                self.flush_spool()
            except Exception as exc:  # noqa: BLE001 - never ends the loop
                print(f"spool flush failed: {exc}", file=sys.stderr)
            round_ = self.think(round_no)
            self.rounds.append(round_)
            try:
                self.ledger.append(round_)
            except OSError as exc:  # a full disk must not stop thought either
                print(f"ledger append failed: {exc}", file=sys.stderr)
            self._say(round_)
            round_no += 1
            done += 1
            if max_rounds is not None and done >= max_rounds:
                break
            wait = self.interval * (1.0 + random.uniform(-self.jitter, self.jitter))
            # A model that is still loading (503/unreachable) is not a reason
            # to sleep a full interval: retry soon and keep the loop feeling
            # alive (measured: every stack restart burned 4-5 minutes of
            # silence per proposer on the first-round 503).
            err = str(round_.error)
            if "503" in err or "unreachable" in err:
                wait = min(wait, 45.0)
            sleep(max(wait, 1.0))

    def _say(self, round_: Round) -> None:
        stamp = time.strftime("%F %T", time.localtime(round_.started_at))
        action = round_.proposal.action if round_.proposal else "-"
        mark = "OK " if round_.published else ".. "
        print(f"{stamp} r{round_.round_no} {mark}{action} "
              f"({round_.latency_ms:.0f}ms, {round_.completion_tokens} tok) "
              f"{round_.detail or round_.error}", file=self._out, flush=True)


def _read_secret(path: str | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pando.thinkd",
        description="Resident proposer: one never-stopping thought loop per device.")
    parser.add_argument("--node-id", required=True,
                        help="who this thinker speaks as (proposals are signed)")
    parser.add_argument("--base-url", required=True,
                        help="OpenAI-compatible endpoint (local llama.cpp or cloud)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", default="",
                        help="file holding the bearer token; empty for local servers")
    parser.add_argument("--fallback-base-url", default="",
                        help="second endpoint, tried when the first is down")
    parser.add_argument("--fallback-model", default="")
    parser.add_argument("--fallback-api-key-file", default="")
    parser.add_argument("--mailbox", required=True,
                        help="primary mailbox URL (proposals are published here)")
    parser.add_argument("--mailbox-fallback", action="append", default=[],
                        help="second mailbox, tried when the primary cannot be "
                             "heard; repeatable. One mailbox on one host is one "
                             "process away from nobody being able to be heard.")
    parser.add_argument("--token-file", default="")
    parser.add_argument("--identity", default="")
    parser.add_argument("--ledger", default="~/.pando/think-ledger.jsonl")
    parser.add_argument("--spool", default="",
                        help="where undeliverable proposals wait for the "
                             "mailbox to come back (default: <ledger>.held)")
    parser.add_argument("--interval", type=float, default=120.0,
                        help="seconds between rounds (jittered by +-15%%)")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="stop after N rounds (testing); default is forever")
    parser.add_argument("--explore-every", type=int, default=5,
                        help="every N rounds, force a variation round (0=off)")
    parser.add_argument("--activity-every", type=int, default=4,
                        help="refresh the colony's outcome feed every N rounds "
                             "(the mailbox listing is O(tasks))")
    parser.add_argument("--hosts", default="~/.pandosi/hosts.json",
                        help="host names a seed.replicate may name; URLs are not shown to the model")
    parser.add_argument("--dry-run", action="store_true",
                        help="think and sign, but never publish")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    identity = load_or_create(args.identity or None)
    chain = [OpenAIProposer(
        base_url=args.base_url, model=args.model,
        api_key=_read_secret(args.api_key_file),
        max_tokens=args.max_tokens, temperature=args.temperature,
        timeout=args.timeout)]
    if args.fallback_base_url and args.fallback_model:
        chain.append(OpenAIProposer(
            base_url=args.fallback_base_url, model=args.fallback_model,
            api_key=_read_secret(args.fallback_api_key_file),
            max_tokens=args.max_tokens, temperature=args.temperature,
            timeout=args.timeout))
    proposer = FailoverProposer(chain)
    thinker = Thinker(
        node_id=args.node_id, proposer=proposer, identity=identity,
        ledger=Ledger(args.ledger), mailbox_url=args.mailbox,
        mailbox_fallbacks=args.mailbox_fallback,
        spool=Spool(args.spool or (str(args.ledger) + ".held")),
        token=_read_secret(args.token_file) or None,
        interval=args.interval, history_window=args.history_window,
        dry_run=args.dry_run, explore_every=args.explore_every,
        activity_every=args.activity_every,
        host_names=host_names_from(Path(args.hosts).expanduser()))
    print(f"thinkd: {args.node_id} thinking via {args.model} at {args.base_url}, "
          f"round every {args.interval:.0f}s, ledger {args.ledger}, "
          f"hosts {','.join(thinker.host_names) or '-'}", flush=True)
    try:
        thinker.run_forever(max_rounds=args.max_rounds)
    except KeyboardInterrupt:
        print("thinkd: interrupted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
