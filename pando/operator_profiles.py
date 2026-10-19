"""The fixed operator profiles an executor may deploy without per-item review.

The catalogue is code, not proposal input.  A proposal may ask for one of
these exact names; it cannot supply an implementation, parameters, a module,
a path or a command.  The executor resolves the name here and installs only
the fixed binding below.
"""

from __future__ import annotations

import copy
import json

from .culture import validate as validate_culture
from .invite import validate as validate_invite
from .operators import BINDABLE, validate as validate_operator
from .rebalance import validate as validate_rebalance
from .spawn import validate as validate_spawn

#: The actuator validators available to the fixed catalogue and the manual
#: installer.  The catalogue imports this module rather than ``tools`` so a
#: resident executor does not depend on a command-line helper.
VALIDATORS = {
    "culture.refine": validate_culture,
    "recruit.invite": validate_invite,
    "reproduce.spawn": validate_spawn,
    "resource.rebalance": validate_rebalance,
    "meta.propose": validate_operator,
}

if set(VALIDATORS) != set(BINDABLE):
    raise RuntimeError("operator profile validators and BINDABLE have drifted")


class OperatorProfileError(ValueError):
    """A proposed operator name is not in the fixed deployment catalogue."""


# Parameters are stored as canonical JSON so later callers cannot mutate the
# catalogue's copy.  Resolution always returns a fresh object.
_PROFILES: dict[str, dict[str, str]] = {
    "adaptive.governance": {
        "binds_to": "culture.refine",
        "params_json": json.dumps({
            "playbook": {
                "actions": [{"do": "log_only", "when": "intent == refund"}],
                "gating": {
                    "auto_min_agreement": 0.6,
                    "auto_min_probability": 0.6,
                    "escalate_above_disagreement": 0.5,
                    "quorum_floor": 2,
                },
                "meta": {"name": "candidate", "version": 1},
                "questions": [{
                    "criteria": ["refund", "greeting"],
                    "id": "intent",
                    "instructions": "Classify the message.",
                    "type": "choice",
                }],
            },
        }, sort_keys=True, separators=(",", ":")),
    },
    "dynamic.threshold.adjustment": {
        "binds_to": "resource.rebalance",
        "params_json": json.dumps({
            "node_id": "mac-llm-8b-local",
            "target_host": "win",
        }, sort_keys=True, separators=(",", ":")),
    },
}


def names() -> tuple[str, ...]:
    """The exact names an automatic deployment may request."""
    return tuple(sorted(_PROFILES))


def resolve(name: object) -> dict:
    """Return a fresh ``{name, binds_to, params, reviewer}`` profile.

    Matching is exact.  There is deliberately no normalization or fuzzy
    mapping: an old proposal spelling such as ``adaptive_governance_protocol``
    is not a licence to run a similarly named profile.
    """
    if not isinstance(name, str) or name not in _PROFILES:
        raise OperatorProfileError(
            f"operator {name!r} is not in the fixed profile catalogue; "
            "allowed names are " + ", ".join(names()))
    definition = _PROFILES[name]
    params = json.loads(definition["params_json"])
    binds_to = definition["binds_to"]
    checked, why = VALIDATORS[binds_to](params)
    if checked is None:
        raise OperatorProfileError(
            f"catalogue profile {name} has invalid fixed params: {why}")
    # ``culture.validate`` returns the playbook mapping itself, while the
    # actuator's params contract is the one-key {"playbook": ...} wrapper.
    # Keep the catalogue's stored value in the actuator's shape so the same
    # validator can check exactly what install will later replay.
    if binds_to == "culture.refine":
        checked = {"playbook": checked}
    return {
        "name": name,
        "binds_to": binds_to,
        "params": copy.deepcopy(checked),
        "reviewer": "auto-profile-catalogue/v1",
    }
