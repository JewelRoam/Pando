"""Per-action selectors: the oldest adopted local proposal not yet done.

One ``next_*`` pair per actuator action.  ``next_*`` picks this host's own
oldest adopted, unsettled proposal for its action; the matching
``foreign_*`` names the same action adopted on a peer's signature, so the
ledger records a real group outcome even when this host must not carry it
out.
"""
from ..genesis import Proposal

from .judge import adopted

def next_policy_set(proposals: list[Proposal], reactions: list[dict],
                    done: set[str], signer: str | None = None,
                    knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted ``policy.set`` this host authored and not yet applied.

    One per round, like the seed export, and for the same two reasons: a round
    should not rewrite every knob at once, and a rule change applied halfway
    through a round would leave the ledger describing a rule that never held.
    A change applied here is in force from the next round.
    """
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "policy.set":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def next_culture_refine(proposals: list[Proposal], reactions: list[dict],
                        done: set[str], signer: str | None = None,
                        knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted ``culture.refine`` this host authored and not yet staged.

    Staging is one-way: once a candidate is written to the outbox the ledger
    row marks it done, so a later round will not restage the same proposal.
    Only the signer's own proposals are staged — a foreign culture change is
    recorded, never written locally, for the same reason a foreign seed is not
    sealed here.
    """
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "culture.refine":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def foreign_culture_refines(proposals: list[Proposal], reactions: list[dict],
                            done: set[str], signer: str,
                            knobs: dict | None = None) -> list[str]:
    """Adopted ``culture.refine`` ids another host authored.

    Recorded rather than dropped: a foreign culture candidate that is neither
    staged here nor named here would leave the ledger silent about a culture
    change the group adopted. It is not staged locally — the authoring host
    owns its own outbox.
    """
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "culture.refine":
            continue
        if proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def next_recruit_invite(proposals: list[Proposal], reactions: list[dict],
                        done: set[str], signer: str | None = None,
                        knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted ``recruit.invite`` this host authored and not yet staged.

    Only the signer's own invitation is minted: an invitation is issued with
    the issuer's key and into the issuer's outbox, so a foreign invitation is
    named, never written here.
    """
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "recruit.invite":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def foreign_recruit_invites(proposals: list[Proposal], reactions: list[dict],
                            done: set[str], signer: str,
                            knobs: dict | None = None) -> list[str]:
    """Adopted ``recruit.invite`` ids another host authored.

    Recorded rather than dropped: an invitation the group adopted is a real
    outcome even though this host must not mint it.
    """
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "recruit.invite":
            continue
        if proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def next_reproduce_spawn(proposals: list[Proposal], reactions: list[dict],
                         done: set[str], signer: str | None = None,
                         knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted ``reproduce.spawn`` this host authored and not yet drafted.

    Only the signer's own proposal is drafted here. The plan is written into the
    authoring host's outbox, so a foreign reproduction stays foreign: it is
    named in the ledger and never written locally.
    """
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "reproduce.spawn":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def foreign_reproduce_spawns(proposals: list[Proposal], reactions: list[dict],
                             done: set[str], signer: str,
                             knobs: dict | None = None) -> list[str]:
    """Adopted ``reproduce.spawn`` ids another host authored.

    Recorded rather than dropped: a reproduction the group adopted is a real
    outcome even though this host must not draft it. The authoring host owns
    its own outbox.
    """
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "reproduce.spawn":
            continue
        if proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def next_rebalance(proposals: list[Proposal], reactions: list[dict],
                   done: set[str], signer: str | None = None,
                   knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted ``resource.rebalance`` this host authored, not yet drafted.

    Only the signer's own proposal is drafted: the plan is written into the
    authoring host's outbox, and a foreign move is named, never re-hosted here.
    """
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "resource.rebalance":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def foreign_rebalances(proposals: list[Proposal], reactions: list[dict],
                       done: set[str], signer: str,
                       knobs: dict | None = None) -> list[str]:
    """Adopted ``resource.rebalance`` ids another host authored."""
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "resource.rebalance":
            continue
        if proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def next_meta_propose(proposals: list[Proposal], reactions: list[dict],
                      done: set[str], signer: str | None = None,
                      knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted ``meta.propose`` this host authored, not yet drafted."""
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "meta.propose":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None


def foreign_meta_proposes(proposals: list[Proposal], reactions: list[dict],
                          done: set[str], signer: str,
                          knobs: dict | None = None) -> list[str]:
    """Adopted ``meta.propose`` ids another host authored."""
    found = []
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "meta.propose":
            continue
        if proposal.pubkey == signer:
            continue
        if adopted(proposal, reactions, knobs):
            found.append(proposal.proposal_id)
    return found


def next_seed_export(proposals: list[Proposal], reactions: list[dict],
                     done: set[str], signer: str | None = None,
                     knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted seed.export this host is allowed to seal.

    The bundle is this machine's identity. A proposal signed by another
    host must be sealed there. ``signer`` is this machine's public key;
    when it is set, a proposal with any other key is left alone.
    """
    for proposal in proposals:
        if proposal.proposal_id in done or proposal.action != "seed.export":
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if adopted(proposal, reactions, knobs):
            return proposal
    return None

def next_installed(proposals: list[Proposal], reactions: list[dict],
                   done: set[str], signer: str | None,
                   installed_operators: dict,
                   knobs: dict | None = None) -> Proposal | None:
    """Oldest adopted proposal naming an installed verb this host authored.

    One per round, like every other actuator here: a round that staged every
    installed verb at once would put several unrelated artifacts in flight on
    the strength of one vote each, and the ledger would stop being a
    description of what happened in order.
    """
    for proposal in proposals:
        if proposal.proposal_id in done:
            continue
        if signer is not None and proposal.pubkey != signer:
            continue
        if proposal.action not in installed_operators:
            continue
        if adopted(proposal, reactions, knobs, installed_operators):
            return proposal
    return None

