"""Unified command-line entry point for PandoSI."""

from __future__ import annotations

import argparse
import importlib
import sys

from . import __version__

COMMANDS = {
    "doctor": ("pando.doctor", "read-only capability and onboarding report"),
    "onboard": ("pando.onboard", "safe dry-run onboarding plan"),
    "grants": ("pando.grants", "expiring operator capability ledger"),
    "join": ("pando.join", "start or join a local/LAN swarm"),
    "root": ("pando.rootd", "run the consensus coordinator"),
    "stem": ("pando.stemd", "run one decision worker"),
    "bridge": ("pando.bridge", "expose a remote group as one stem"),
    "service": ("pando.service", "generate explicit service plans"),
    "lab": ("pando.lab", "sandbox-gated replication lab"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pando",
        description="Elastic heterogeneous unified-intelligence tooling")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    for name, (_, description) in COMMANDS.items():
        subparsers.add_parser(name, add_help=False, help=description)
    args, remainder = parser.parse_known_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    module_name, _ = COMMANDS[args.command]
    module = importlib.import_module(module_name)
    original_argv = sys.argv
    sys.argv = [f"pando {args.command}", *remainder]
    try:
        module.main()
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = original_argv
