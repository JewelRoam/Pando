"""Choosing which mailbox rows one execution round may look at.

Three layers: the live window, every unsettled ``seed.export`` /
``seed.replicate`` (the only actions with a peer on the other end), and —
when the caller names its authors — the oldest unsettled proposal of each
actuator action this host signs under.  Layer 3 is what keeps an adopted
culture.refine / recruit.invite / ... older than the window from becoming
invisible.
"""
import json
from collections import Counter
from pathlib import Path

from ..proposal_mail import mailbox_chain

from .consts import ACTUATORS

#: Path arguments that live inside the state dir, as argument name -> (leaf,
#: what the leaf is for). ``--identity`` is deliberately absent: both hosts
#: keep the identity in ``~/.pando``, which predates this flag and must not be
#: silently relocated by it.
STATE_PATHS: dict[str, tuple[str, str]] = {
    "token_file": ("mailbox.token", "the mailbox token this host reads"),
    "passphrase_file": ("seed.passphrase", "the passphrase that seals a seed"),
    "out_root": ("executions", "where a sealed seed is written"),
    "ledger": ("execute-ledger.jsonl", "what this host has already acted on"),
    "policy": ("policy.json", "the bounded self-modification store"),
    "culture_outbox": ("culture-outbox", "staged culture.refine candidates"),
    "invite_outbox": ("invites", "staged recruit.invite tokens"),
    "spawn_outbox": ("reproduce-outbox", "drafted reproduce.spawn plans"),
    "rebalance_outbox": ("rebalance-outbox", "drafted resource.rebalance plans"),
    "operator_outbox": ("operator-outbox", "drafted meta.propose specs"),
    # Not an outbox: this is the set the gate will *load* and run, so it sits
    # beside the state root rather than inside the draft tree. Defaulting it
    # here as well as in ``pando.operators`` keeps the two spellings of the
    # same directory from drifting apart on a host that relocated its state.
    "operators_dir": ("operators", "installed operator bindings this host runs"),
    "grants": ("grants.toml", "expiring capability grants"),
    "operator_install_ledger": (
        "operator-install-ledger.jsonl",
        "installed operator lifecycle ledger"),
    "operator_deploy_ledger": (
        "operator-deploy-ledger.jsonl",
        "automatic profile deployment ledger"),
}

def _titled(title: str, prefix: str) -> tuple[str, str] | None:
    """Second and third tokens of a ``[proposal]`` or ``[reaction]`` title.

    The mailbox listing has titles, not bodies. The id is the second token
    and the action or stance is the third, followed by ``by``.
    """
    parts = str(title).split()
    if len(parts) < 5 or parts[0] != prefix or parts[3] != "by":
        return None
    if not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]
DEFAULT_PER_ACTION = 40

#: Where a host keeps its executor state. The Mac has used ``~/.pando`` since
#: the first round; a node brought up under the shorter ``~/.pando`` should pass
#: ``--state-dir`` once rather than spell out ten paths -- spelling them out per
#: unit is how one host ends up with its ledger in one tree and its outboxes in
#: another, and how a seal writes a rollout home nobody reads.
DEFAULT_STATE_DIR = "~/.pando"
def _author_of(title: str, prefix: str = "[proposal]") -> str:
    """The author token of a ``[proposal]`` title, or ``""``.

    ``publish_proposal`` writes ``"[proposal] <id> <action> by <author>"`` and
    an author is a single token, so the fifth token is the whole name. It is
    read without fetching the body, which is what lets retention tell this
    host's drafts from a peer's before paying for a GET.
    """
    parts = str(title).split()
    if len(parts) < 5 or parts[0] != prefix or parts[3] != "by":
        return ""
    return parts[4]


def rows_for_execution(tasks: list, limit: int,
                       per_action: int = DEFAULT_PER_ACTION,
                       settled: frozenset | set = frozenset(),
                       authors: set | None = None) -> list:
    """The rows one round may act on, in listing order.

    Three layers:

    1. the last ``limit`` rows -- the live window the dry run and the ledger
       describe;
    2. every unsettled ``seed.export`` and ``seed.replicate``. Those two are
       the only actions with a second host on the other end, so the send/receive
       bookkeeping has to see a peer's offer as well as this host's: an
       addressed replication pushed out of the window would be offered twice.
       A proposal already named by ``settled`` needs neither its own body nor
       its votes again;
    3. with ``authors`` set, the oldest ``per_action`` proposals of each
       actuator action that this host signs under and that no ledger row has
       settled yet (``settled``).

    Layer 3 is the one that is easy to leave out and hard to notice missing.
    Without it an adopted ``culture.refine`` / ``recruit.invite`` /
    ``reproduce.spawn`` / ``resource.rebalance`` / ``meta.propose`` /
    ``policy.set`` older than ``limit`` rows is invisible, so the only actions
    that ever reach an actuator are the two seeds. Oldest-first on purpose: a
    backlog is drained from the front, and newest-first would strand every
    proposal behind a window that keeps sliding away from it. It keys on the
    author name in the title rather than the pubkey because the body has not
    been fetched yet; ``next_*`` re-checks the pubkey, so a name collision
    buys a wasted fetch, never a wrong execution.

    Reactions naming any kept proposal ride along. A proposal judged on half
    its votes reads as "not adopted", which is indistinguishable in the ledger
    from a real refusal.
    """
    rows = [row for row in tasks if isinstance(row, dict)]
    window = rows[-limit:] if limit > 0 else []
    keep = {row.get("id") for row in window}
    names = {str(name) for name in (authors or ()) if name}
    seed_ids = set()
    act_ids = set()
    taken: Counter = Counter()
    for row in rows:
        parsed = _titled(str(row.get("title") or ""), "[proposal]")
        if parsed is None:
            continue
        proposal_id, action = parsed
        if action in ("seed.export", "seed.replicate"):
            if proposal_id in settled:
                continue
            seed_ids.add(proposal_id)
            keep.add(row.get("id"))
            continue
        if not names or per_action <= 0 or action not in ACTUATORS:
            continue
        if taken[action] >= per_action or proposal_id in settled:
            continue
        if _author_of(str(row.get("title") or "")) not in names:
            continue
        taken[action] += 1
        act_ids.add(proposal_id)
        keep.add(row.get("id"))
    for row in rows:
        parsed = _titled(str(row.get("title") or ""), "[reaction]")
        if parsed and (parsed[0] in seed_ids or parsed[0] in act_ids):
            keep.add(row.get("id"))
    return [row for row in rows if row.get("id") in keep]


def _hosts_table(path: str = "~/.pando/hosts.json") -> dict:
    """This host's ``hosts.json``, or ``{}`` when it is missing or corrupt.

    The executor already reads the same file for its mailbox and pull
    destinations; reading the author name from it too keeps one source of
    truth for "who is this host".
    """
    try:
        return json.loads(
            Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


#: The localhost conventions, in the order the connectivity doctrine lists
#: them: the Mac's mailbox port first, the win node's second. They are used
#: only when neither the command line nor this host's table names a mailbox,
#: so a node with no config still looks at its own door instead of dialling a
#: peer's address that this checkout is not allowed to know.
DEFAULT_MAILBOXES = ("http://127.0.0.1:8776", "http://127.0.0.1:8778")


def host_mailboxes(table: dict | None) -> list[str]:
    """The mailbox URLs a ``hosts.json`` names, in order. Blank -> ``[]``."""
    urls = table.get("mailboxes") if isinstance(table, dict) else None
    if not isinstance(urls, list):
        return []
    return [str(url).strip() for url in urls if str(url or "").strip()]


def executor_mailboxes(mailbox: str, table: dict | None, fallbacks=()) -> list[str]:
    """The ordered shelf list one round may read.

    Command line first, then *every* mailbox this host's table names, then
    ``--mailbox-fallback``. The table goes in whole rather than as
    ``table[1:]``: when the command line names a primary, the table's own
    first entry is still a shelf this host is meant to read, and dropping it
    is how a node polls a peer's replica while its own mailbox fills up
    unread. Duplicates collapse in ``mailbox_chain``, so the caller may pass
    the same address in all three places -- which is exactly what the pi5 and
    Windows units do, so that a moved address is reused everywhere a person
    was already writing it.
    """
    named = host_mailboxes(table)
    primary = str(mailbox or "").strip() or (named[0] if named else "")
    chain = mailbox_chain(primary, named + list(fallbacks or []))
    return chain or list(DEFAULT_MAILBOXES)


def local_authors(table: dict | None, extra=()) -> set[str]:
    """The author names this host may drain in layer 3 of the row selection.

    ``hosts.json`` already carries this host's ``proposer_id``. ``extra`` is
    for a second local name (a host can sign under more than one). An empty
    result disables layer 3 instead of guessing: a wrong name here fails
    silently -- the host would drain nothing beyond the seed pair -- while an
    empty set is merely the behaviour that preceded this function.
    """
    names = {str(name) for name in (extra or ()) if name}
    if isinstance(table, dict) and table.get("proposer_id"):
        names.add(str(table["proposer_id"]))
    return names


def execution_selectors(authors: set[str] | None) -> list[dict]:
    """Server-side rows whose omission would change an execution round.

    Seeds are selected across every author because the other end of the offer
    is a peer.  Other actuators are selected only for names this host signs
    under; those are the only proposals layer 3 of ``rows_for_execution`` may
    drain.  The mailbox adds every reaction that targets a selected proposal,
    which is what keeps an old proposal's late vote visible.
    """
    selectors = [{"actions": ["seed.export", "seed.replicate"]}]
    names = sorted(str(name) for name in (authors or ()) if name)
    if names:
        actions = sorted(action for action in ACTUATORS
                         if action not in ("seed.export", "seed.replicate"))
        selectors.append({"actions": actions, "authors": names})
    return selectors
