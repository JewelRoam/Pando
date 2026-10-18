"""The bounded self-modification store: which knobs may move, and how far.

``policy.set`` is the one action in ``ACTION_SPACE`` that changes the rules the
colony runs under — *"change a bounded voting rule or gate threshold"*. The
action name advertises only the knobs that exist; a "role grant" is not yet one
of them. That makes policy.set the most dangerous verb in the space, so its
backing store is the most conservative thing in the repo. Five decisions, each
of which we can be wrong about:

- **The knob set is closed.** ``KNOBS`` is the whitelist. An unknown knob does
  not get ignored, it fails the whole proposal: a proposal whose author and
  reader disagree about what it does must not run.
- **Out-of-range is refused, never clamped.** A silent clamp would make the
  stored value differ from the value the author signed. The refusal is recorded
  in the ledger with the reason, so a rejected rule change is visible.
- **Values come from the signed ``params`` block, never from the prose.** The
  ``goal`` and ``expected_effect`` fields are free text written by a 1.7B model.
  Prose cannot move a threshold; only a signature over a typed value can.
- **Weakening the gate costs more than tightening it.** A change that makes the
  gate easier to pass (lower ``min_adopt``, or ``veto`` true -> false) needs
  ``WEAKEN_ADOPTERS`` decision models instead of the ordinary majority. A
  system that can vote away its own veto with a bare majority is not a safety
  system; this is the smallest rule that stops the obvious version of that.
- **All-or-nothing, and atomic on disk.** Every knob in one proposal is checked
  before anything is written, the file is replaced in one ``os.replace``, and
  the previous value of each knob is returned for the audit row.

The knobs are named after the keyword arguments of ``pando.execute.judge`` on
purpose: the store's contents *are* that function's policy. Nothing here
decides whether a proposal may run — that stays in ``pando/execute.py``, which
only reaches this module after the gate has already adopted the proposal.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

DEFAULT_PATH = "~/.pandosi/policy.json"

#: How many decision models must adopt a change that makes the gate easier to
#: pass. One more than the default ``min_adopt``, and it is a floor, not a
#: suggestion: ``min_adopt`` cannot be set below it by a lesser vote.
WEAKEN_ADOPTERS = 3


class PolicyError(ValueError):
    """A requested change was refused. The message is the audit reason."""


@dataclasses.dataclass(frozen=True)
class Knob:
    """One settable knob: its type, its legal range, and what it means."""

    name: str
    kind: type
    default: object
    low: int | None
    high: int | None
    meaning: str


KNOBS: dict[str, Knob] = {
    "min_adopt": Knob(
        name="min_adopt", kind=int, default=2, low=1, high=5,
        meaning="how many other decision models must adopt before an action runs"),
    "veto": Knob(
        name="veto", kind=bool, default=True, low=None, high=None,
        meaning="whether a single decline holds an otherwise adopted action"),
}


def defaults() -> dict:
    """The knobs as a fresh dict. Always the fallback, never a shared object."""
    return {name: knob.default for name, knob in KNOBS.items()}


def check(knob: Knob, value) -> tuple[bool, str]:
    """Is ``value`` legal for ``knob``? Returns (ok, why-not)."""
    if knob.kind is bool:
        if not isinstance(value, bool):
            return False, f"{knob.name} must be a boolean, got {value!r}"
        return True, ""
    if knob.kind is int:
        # bool is an int subclass in Python; True is not min_adopt=1.
        if isinstance(value, bool) or not isinstance(value, int):
            return False, f"{knob.name} must be an integer, got {value!r}"
        if value < knob.low or value > knob.high:
            return False, (f"{knob.name}={value} is outside "
                           f"{knob.low}..{knob.high}")
        return True, ""
    return False, f"{knob.name} has an unsupported type {knob.kind!r}"


def coalesce(data) -> dict:
    """A legal knob dict from whatever was on disk. Unknown keys are dropped.

    A missing file and a corrupt one are the same case: fall back to the
    defaults. Defaults are the *known* strict values, so a file we cannot read
    can never make the gate weaker than shipping code.
    """
    values = defaults()
    if not isinstance(data, dict):
        return values
    for name, knob in KNOBS.items():
        if name not in data:
            continue
        ok, _ = check(knob, data[name])
        if ok:
            values[name] = data[name]
    return values


def load(path: str | Path = DEFAULT_PATH) -> dict:
    """Read the store, falling back to the defaults on any problem."""
    try:
        raw = Path(path).expanduser().read_text(encoding="utf-8")
    except OSError:
        return defaults()
    try:
        return coalesce(json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        return defaults()


def weakens(name: str, old, new) -> bool:
    """Does this change make the gate easier to pass?

    Explicit per knob rather than a generic comparison, because "lower is
    weaker" is a property of the knob, not of the number. A knob with no rule
    here raises, so adding a knob forces a decision about it.
    """
    if name == "veto":
        return bool(old) and not bool(new)
    if name == "min_adopt":
        return int(new) < int(old)
    raise PolicyError(f"no weakening rule for knob {name!r}")


def validate(params) -> tuple[dict | None, str]:
    """Is this a well-formed ``policy.set`` intent? Returns (knobs, why-not).

    This is the *only* place the shape of the intent is defined. The proposer
    asks it before it signs an intent and the executor asks it before it
    writes one, so the two cannot drift: a proposal that passed here composes,
    and one that failed here is refused rather than signed and dropped later.

    The weakening brake is deliberately not applied. Whether a change is
    allowed depends on how many adopters carried it, and a proposal that has
    not been voted on yet has none. This answers "is this a well-formed
    request", not "may it proceed".
    """
    if not isinstance(params, dict) or set(params) != {"set"}:
        return None, 'params must be {"set": {"<knob>": <value>}}'
    wanted = params["set"]
    if not isinstance(wanted, dict) or not wanted:
        return None, "params.set must name at least one knob"
    for name in sorted(wanted):
        knob = KNOBS.get(name)
        if knob is None:
            return None, (f"unknown knob {name!r}; settable: "
                          + ", ".join(sorted(KNOBS)))
        ok, why = check(knob, wanted[name])
        if not ok:
            return None, why
    return {name: wanted[name] for name in sorted(wanted)}, ""


def set_knobs(values: dict, params, *, adopters: int = 0) -> tuple[dict, dict]:
    """Validate one signed ``params`` block against ``values``. Pure.

    Returns the new store and a per-knob ``{from, to}`` record. Raises
    ``PolicyError`` with the reason on the first problem — nothing is applied,
    so a proposal that names one good knob and one impossible one changes
    neither. All knobs are checked before any is written for the same reason.
    """
    wanted, why = validate(params)
    if wanted is None:
        raise PolicyError(why)
    new = dict(values)
    applied: dict[str, dict] = {}
    for name in sorted(wanted):
        knob = KNOBS[name]
        value = wanted[name]
        old = new.get(name, knob.default)
        if weakens(name, old, value) and int(adopters) < WEAKEN_ADOPTERS:
            raise PolicyError(
                f"{name} {old} -> {value} makes the gate easier to pass; "
                f"needs {WEAKEN_ADOPTERS} adopters, saw {int(adopters)}")
        applied[name] = {"from": old, "to": value}
        new[name] = value
    return new, applied


def write_atomic(path: str | Path, values: dict) -> Path:
    """Replace the store in one move. A reader never sees a half-written file."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".tmp")
    staging.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    os.replace(staging, target)
    return target


def apply(proposal, path: str | Path = DEFAULT_PATH, *,
          adopters: int = 0) -> dict:
    """Carry out one adopted ``policy.set``. The only write in this module.

    ``proposal.params`` is the signed block; ``goal`` is not read. Raises
    ``PolicyError`` without touching the disk when the change is refused.
    """
    current = load(path)
    new, applied = set_knobs(current, getattr(proposal, "params", None),
                             adopters=adopters)
    write_atomic(path, new)
    return {"applied": applied, "policy": new}
