#!/usr/bin/env python3
"""Render this host's service units from its untracked host config.

    python tools/render_units.py                 # print what would render
    python tools/render_units.py --install       # write into render.outdir

The units land in the platform's service directory and are never committed:
the checkout carries the catalogue (``pando.deploy.SERVICES``), the host
carries the wiring (``host.toml``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pando import deploy, hostconfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                        help="host TOML (default: $PANDO_HOST_CONFIG or "
                             "~/.pando/host.toml)")
    parser.add_argument("--platform", default="auto",
                        choices=["auto", "linux", "darwin"])
    parser.add_argument("--install", action="store_true",
                        help="write into [render].outdir instead of printing")
    args = parser.parse_args()

    config = hostconfig.load(args.config)
    if args.install:
        render = config.get("render") or {}
        outdir = render.get("outdir")
        if not outdir:
            parser.error("host config render.outdir is required with --install")
        result = deploy.install_units(config, outdir, platform=args.platform)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    units = deploy.render_units(config, args.platform)
    for filename, content in sorted(units.items()):
        print(f"# ----- {filename} -----")
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
