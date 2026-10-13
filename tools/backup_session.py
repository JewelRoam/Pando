"""Backup and sanitize a Codex session transcript into the project docs.

Usage:
    python tools/backup_session.py [--thread <id>] [--out <dir>]

Finds all rollout JSONL files for the thread under %USERPROFILE%/.codex/sessions,
extracts user/assistant messages and tool-call names, sanitizes sensitive data,
and writes one timestamped Markdown file. Sanitization is mechanical and
conservative: when in doubt, redact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

DEFAULT_THREAD = "01a135c7-8380-7033-bc33-5ccefddde30d"
CODEX_SESSIONS = Path.home() / ".codex" / "sessions"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
TOKEN_RE = re.compile(r"(sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{8,}|Bearer\s+\S+)", re.I)
PROXY_RE = re.compile(r"https?://127\.0\.0\.1:\d+")
# Project-local redaction list: known identifiers for this workspace.
EXTRA_REDACTIONS = [
    (re.compile(r"\b27525\b"), "[USER]"),
    (re.compile(r"\b54056\b"), "[PROXY_PORT]"),
]
USER_PATH_RE = re.compile(r"[A-Z]:\\Users\\[^\\\s]+")


def sanitize(text: str) -> str:
    text = TOKEN_RE.sub("[TOKEN]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PROXY_RE.sub("[PROXY]", text)
    text = USER_PATH_RE.sub("~", text)
    for pattern, replacement in EXTRA_REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def content_text(payload: dict) -> str:
    parts = []
    for item in payload.get("content", []):
        if isinstance(item, dict) and item.get("text"):
            parts.append(item["text"])
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default=DEFAULT_THREAD)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rollouts = sorted(
        (p for p in CODEX_SESSIONS.rglob("*.jsonl") if args.thread in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    if not rollouts:
        raise SystemExit(f"no rollout files found for thread {args.thread}")

    turns: list[tuple[str, str, str]] = []  # (timestamp, kind, text)
    tool_calls = 0
    for path in rollouts:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = record.get("timestamp", "")[:19].replace("T", " ")
                payload = record.get("payload", {})
                if record.get("type") != "response_item":
                    continue
                kind = payload.get("type")
                if kind == "message":
                    role = payload.get("role", "?")
                    text = content_text(payload).strip()
                    if text:
                        turns.append((ts, role, text))
                elif kind == "function_call":
                    tool_calls += 1
                    name = payload.get("name", "?")
                    turns.append((ts, "tool", f"`{name}` (call #{tool_calls})"))

    now = dt.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "docs" / "session-logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{now}-session-backup.md"

    lines = [
        f"# Session backup {now}",
        "",
        f"- thread: `{args.thread}`",
        f"- rollout files: {len(rollouts)}",
        f"- extracted: {sum(1 for _, k, _ in turns if k == chr(117)+chr(115)+chr(101)+chr(114))} user / "
        f"{sum(1 for _, k, _ in turns if k == 'assistant')} assistant messages, {tool_calls} tool calls",
        "- sanitization: emails, tokens, local proxy URLs, and user paths redacted",
        "",
        "## Summary",
        "",
        "<!-- curated summary appended by the agent on each run -->",
        "",
        "## Transcript",
        "",
    ]
    for ts, kind, text in turns:
        label = {"user": "USER", "assistant": "ASSISTANT", "tool": "tool"}.get(kind, kind)
        lines.append(f"### [{label}] {ts}")
        lines.append("")
        lines.append(sanitize(text))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path} ({len(turns)} entries from {len(rollouts)} rollout files)")


if __name__ == "__main__":
    main()
