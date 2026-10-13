"""Safe onboarding planner.

This module deliberately does not install, elevate, or mutate the system.
Its contract is: inspect doctor rows, classify each next action, and show the
least-privileged command that moves the user forward.
"""

from __future__ import annotations

import argparse
import json
import re

PLAN_FIELDS = ("ready", "nonprivileged", "consent_required", "sandbox_required")
CONSENT_RE = re.compile(
    r"driver|admin|administrator|sudo|uac|firewall|autostart|service|"
    r"network permission|device permission", re.I)


def normalize_rows(rows: list[dict] | list[tuple[str, str, str]]) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append({"status": row.get("status", "miss"),
                               "item": row.get("item", ""),
                               "fix": row.get("fix", "")})
        else:
            status, item, fix = row
            normalized.append({"status": status, "item": item, "fix": fix})
    return normalized


def classify(row: dict) -> str:
    """Classify one row without pretending an elevated action is safe."""
    if row["status"] == "ok":
        return "ready"
    haystack = f'{row["item"]} {row["fix"]}'
    if row["status"] == "locked" and "lab sandbox" in row["item"]:
        return "sandbox_required"
    if not row["fix"] or CONSENT_RE.search(haystack):
        return "consent_required"
    return "nonprivileged"


def build_plan(rows: list[dict] | list[tuple[str, str, str]]) -> dict:
    """Turn doctor output into a least-privilege action plan."""
    normalized = normalize_rows(rows)
    groups: dict[str, list[dict]] = {field: [] for field in PLAN_FIELDS}
    for row in normalized:
        row = dict(row)
        row["kind"] = classify(row)
        groups[row["kind"]].append(row)

    blocking = [row for row in normalized if row["status"] == "miss"]
    # A missing sandbox only blocks Ring 2. It must not make the product look
    # unusable on a machine that can already run local consensus.
    if blocking:
        next_command = "python -m pandosi.onboard --dry-run"
    else:
        next_command = "python tools/demo.py"

    return {
        "ring0_ready": not blocking,
        "sandbox_ready": not groups["sandbox_required"],
        "next_command": next_command,
        "counts": {field: len(rows) for field, rows in groups.items()},
        "steps": groups,
    }


def render_plan(plan: dict) -> str:
    labels = {
        "ready": "Ready",
        "nonprivileged": "Safe, user-scoped actions",
        "consent_required": "Requires explicit consent (not run by onboard)",
        "sandbox_required": "Requires disposable sandbox",
    }
    lines = ["PandoSI onboarding plan (dry run)", "",
             f"Ring 0 ready: {plan['ring0_ready']}",
             f"Sandbox ready: {plan['sandbox_ready']}",
             f"Next command: {plan['next_command']}", ""]
    for field in PLAN_FIELDS:
        steps = plan["steps"][field]
        lines.append(labels[field])
        if not steps:
            lines.append("  [none]")
        for step in steps:
            marker = "x" if field == "ready" else " "
            lines.append(f"  [{marker}] {step['item']}")
            if step["fix"]:
                lines.append(f"      -> {step['fix']}")
        lines.append("")
    lines.append("onboard never installs, elevates, writes grants, or opens ports.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a safe onboarding plan")
    parser.add_argument("--dry-run", action="store_true",
                        help="accepted for explicitness; dry run is the only mode")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from .doctor import checks
    plan = build_plan(checks())
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(render_plan(plan))


if __name__ == "__main__":
    main()
