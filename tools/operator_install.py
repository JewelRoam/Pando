"""Clear the ``meta.propose`` hold: install or retire one operator binding.

``pando.operators.stage`` writes a signed *draft*: a verb's name, the prose that
asks for it, and a list of what is still owed. This tool is the gate's half of
that exchange, and it is deliberately a command a person runs on the host whose
executor will carry the verb out. No proposal reaches it, no mailbox path leads
here, and it writes no code.

What it writes is a *binding*: which actuator the new verb reuses, and the fixed
params that actuator will see. ``pando/execute.py`` reads the record once per
round and resolves the verb to that actuator, so a name nobody wrote code for
can still have an effect -- bounded by an actuator the gate already ran.

Usage:
  python tools/operator_install.py list
  python tools/operator_install.py install --spec ~/.pandosi/operator-outbox/<id> \
      --binds-to recruit.invite --param candidate=node-a --reviewer mac-codex
  python tools/operator_install.py retire <verb> --reason "the group stopped"

A ``--param`` value is parsed as JSON when it parses and kept as a string when
it does not, so ``--param expires_in_hours=72`` is the integer 72 and
``--param candidate=node-a`` is the label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pando.culture import validate as validate_culture  # noqa: E402
from pando.genesis import ACTION_SPACE  # noqa: E402
from pando.identity import Identity  # noqa: E402
from pando.invite import validate as validate_invite  # noqa: E402
from pando.operators import DEFAULT_INSTALL_LEDGER  # noqa: E402
from pando.operators import DEFAULT_INSTALLED  # noqa: E402
from pando.operators import OperatorError  # noqa: E402
from pando.operators import install, installed, uninstall  # noqa: E402
from pando.operators import validate as validate_operator  # noqa: E402
from pando.rebalance import validate as validate_rebalance  # noqa: E402
from pando.spawn import validate as validate_spawn  # noqa: E402

DEFAULT_IDENTITY = "~/.pando/identity.json"

#: The validator of the actuator a verb binds to. The bound actuator has the
#: last word on its own params: without this a reviewer could install a verb
#: whose every round fails inside the stager, which is the empty shell with
#: extra steps. Kept equal to ``pando.operators.BINDABLE`` by a test, so a
#: bindable verb can never be installable without a way to check its params.
VALIDATORS = {
    "culture.refine": validate_culture,
    "recruit.invite": validate_invite,
    "reproduce.spawn": validate_spawn,
    "resource.rebalance": validate_rebalance,
    "meta.propose": validate_operator,
}


def _param(text: str) -> tuple[str, object]:
    """``key=value``, with the value read as JSON when it is JSON."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"a param looks like key=value, not {text!r}")
    key, _, raw = text.partition("=")
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"a param needs a key, not {text!r}")
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def _parser() -> argparse.ArgumentParser:
    # The path flags live on the subcommands rather than on the program: a
    # ``--installed-dir`` written after ``install`` has to mean the directory
    # this record lands in, not be ignored as a leftover of the wrong parser.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--installed-dir", default=DEFAULT_INSTALLED,
                        help=f"where installed records live "
                             f"(default: {DEFAULT_INSTALLED})")
    common.add_argument("--ledger", default=DEFAULT_INSTALL_LEDGER,
                        help=f"append-only install/retire ledger "
                             f"(default: {DEFAULT_INSTALL_LEDGER})")
    parser = argparse.ArgumentParser(
        description="install or retire a reviewed operator binding")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", parents=[common],
                   help="print the verbs this gate will run")

    add = sub.add_parser("install", parents=[common],
                         help="bind one verified draft to an actuator")
    add.add_argument("--spec", required=True,
                     help="the staged spec directory from meta.propose")
    add.add_argument("--binds-to", required=True,
                     help="the actuator the new verb reuses")
    add.add_argument("--param", action="append", default=[], type=_param,
                     help="key=value; repeatable. JSON values are parsed")
    add.add_argument("--reviewer", default="",
                     help="who reviewed the binding; recorded in the record")
    add.add_argument("--identity", default=DEFAULT_IDENTITY,
                     help=f"the host key that signs the record "
                          f"(default: {DEFAULT_IDENTITY})")

    drop = sub.add_parser("retire", parents=[common],
                          help="stop loading a verb, keeping the trace")
    drop.add_argument("verb")
    drop.add_argument("--reason", default="",
                      help="why it went; recorded in the ledger")
    return parser


def _render(record: dict) -> str:
    name = str(record.get("operator") or "?")
    return (f"{name}\tbinds_to={record.get('binds_to')}"
            f"\treviewer={record.get('reviewer') or '-'}"
            f"\tproposal={record.get('proposal_id') or '-'}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        try:
            found = installed(args.installed_dir)
        except OperatorError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        if not found:
            print("no installed operators on this host", flush=True)
            return 0
        for name in sorted(found):
            print(_render(found[name]), flush=True)
        return 0

    if args.command == "retire":
        try:
            result = uninstall(args.verb, reason=args.reason,
                               installed_dir=args.installed_dir,
                               ledger=args.ledger)
        except OperatorError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                         indent=2), flush=True)
        return 0

    params: dict = {}
    for key, value in args.param:
        if key in params:
            print(f"refused: --param {key} given twice", file=sys.stderr)
            return 2
        params[key] = value
    if args.binds_to not in VALIDATORS:
        print(f"refused: {args.binds_to} is not an actuator this executor "
              "stages into an outbox; a verb may only bind to "
              + ", ".join(sorted(VALIDATORS)), file=sys.stderr)
        return 2
    path = Path(args.identity).expanduser()
    if not path.is_file():
        print(f"refused: no identity at {path}; an installer signs its record",
              file=sys.stderr)
        return 2
    identity = Identity.from_json(json.loads(path.read_text(encoding="utf-8")))
    try:
        record = install(
            args.spec, identity=identity, action_space=ACTION_SPACE,
            implementation={"binds_to": args.binds_to, "params": params},
            reviewer=args.reviewer,
            validate_params=VALIDATORS[args.binds_to],
            installed_dir=args.installed_dir, ledger=args.ledger)
    except OperatorError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
