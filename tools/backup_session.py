"""Backup and summarize the current Codex session into auditable project docs.

The transcript never leaves the operator's machine. This tool extracts only
user/assistant messages and tool-call names, applies conservative redaction,
and creates a timestamped snapshot only when the source transcript changed.
The heartbeat agent should pass --note with a short factual summary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

DEFAULT_THREAD = "01a135c7-8380-7033-bc33-5ccefddde30d"
CODEX_SESSIONS = Path.home() / ".codex" / "sessions"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "docs" / "session-logs"
STATE_PATH = DEFAULT_OUT / ".backup-state.json"
INDEX_PATH = DEFAULT_OUT / "INDEX.md"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
TOKEN_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,}"
    r"|xox[baprs]-[A-Za-z0-9-]{8,}|AIza[A-Za-z0-9_-]{8,}"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"
    r"|Bearer\s+\S+)", re.I)
CREDENTIAL_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@", re.I)
PRIVATE_URL_RE = re.compile(
    r"https?://(?:127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+"
    r"|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+)(?::\d+)?")
TUNNEL_URL_RE = re.compile(r"https?://[a-z0-9-]+\.trycloudflare\.com", re.I)
USER_PATH_RE = re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+")
# Project-local redaction list for known machine-specific identifiers.
EXTRA_REDACTIONS = [
    (re.compile(r"\b27525\b"), "[USER]"),
    (re.compile(r"\b54056\b"), "[PROXY_PORT]"),
]

# Live token values are redacted by literal match, loaded from the local
# secret files at write time (never embedded in the repo). This is the class
# of leak earlier backups missed: a bare 36-char token on its own line
# matches none of the token regexes.
SECRET_FILES = (
    PROJECT_ROOT / "metrics" / "lan-check" / "mac-mailbox.token",
    PROJECT_ROOT / "artifacts" / "lan-experiment" / "root-pairing.token",
)


def _literal_secrets() -> list[str]:
    values = []
    for path in SECRET_FILES:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if len(value) >= 16:
            values.append(value)
    return values


LITERAL_SECRETS = _literal_secrets()


def sanitize(text: str) -> str:
    """Redact common secrets and identifiers without interpreting content."""
    text = CREDENTIAL_URL_RE.sub("[CREDENTIAL_URL]@", text)
    text = TOKEN_RE.sub("[TOKEN]", text)
    for value in LITERAL_SECRETS:
        text = text.replace(value, "[TOKEN]")
    text = TUNNEL_URL_RE.sub("[TUNNEL_URL]", text)
    text = PRIVATE_URL_RE.sub("[LOCAL_URL]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
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


def find_rollouts(thread: str) -> list[Path]:
    rollouts = sorted(
        (p for p in CODEX_SESSIONS.rglob("*.jsonl") if thread in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    if not rollouts:
        raise SystemExit(f"no rollout files found for thread {thread}")
    return rollouts


def load_entries(paths: list[Path]) -> tuple[list[tuple[str, str, str]], int]:
    """Return (timestamp, kind, sanitized-later-text) entries and tool count."""
    entries: list[tuple[str, str, str]] = []
    tool_calls = 0
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") != "response_item":
                    continue
                ts = record.get("timestamp", "")[:19].replace("T", " ")
                payload = record.get("payload", {})
                kind = payload.get("type")
                if kind == "message":
                    role = payload.get("role", "?")
                    text = content_text(payload).strip()
                    # Developer/system context is not a chat turn and can leak
                    # desktop plumbing into a public project backup.
                    if role in ("user", "assistant") and text:
                        entries.append((ts, role, text))
                elif kind == "function_call":
                    tool_calls += 1
                    entries.append((ts, "tool", payload.get("name", "?")))
    return entries, tool_calls


def source_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(str(path.relative_to(CODEX_SESSIONS)).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def auto_summary(entries: list[tuple[str, str, str]]) -> list[str]:
    """Create a mechanical summary; an agent can improve it via --note."""
    lines = []
    # Internal continuation envelopes are not the human's visible request.
    users = [e for e in entries if e[1] == "user"
             and not e[2].lstrip().startswith("<codex_internal_context")]
    assistants = [e for e in entries if e[1] == "assistant"]
    if users:
        latest = sanitize(users[-1][2]).replace("\n", " ")
        lines.append(f"- latest user request: {latest[:700]}")
    if assistants:
        latest = sanitize(assistants[-1][2]).replace("\n", " ")
        lines.append(f"- latest assistant position: {latest[:700]}")
    if not lines:
        lines.append("- no user/assistant prose was present in this slice")
    return lines


def read_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def append_index(now: str, output: Path, state: dict, note: str) -> None:
    rel_output = output.relative_to(PROJECT_ROOT).as_posix()
    line = (f"| {now} | `{rel_output}` | {state['source_hash'][:12]} "
            f"| {state['user_messages']}/{state['assistant_messages']}"
            f"/{state['tool_calls']} | {note.replace('|', chr(92)+'|')} |")
    if not INDEX_PATH.exists():
        header = [
            "# Session backup index",
            "",
            "Timestamped, sanitized snapshots of this PandoSI working session.",
            "",
            "| time | snapshot | source | user/assistant/tool | note |",
            "|---|---|---|---|---|",
        ]
        INDEX_PATH.write_text("\n".join(header) + "\n" + line + "\n",
                              encoding="utf-8")
    else:
        with INDEX_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def create_backup(thread: str, out_dir: Path, note: str,
                  force: bool) -> Path | None:
    rollouts = find_rollouts(thread)
    source_hash = source_fingerprint(rollouts)
    state = read_state()
    output_rel = state.get("output")
    previous_output = PROJECT_ROOT / output_rel if output_rel else None
    if (not force and state.get("source_hash") == source_hash
            and previous_output and previous_output.exists()):
        print(f"unchanged; latest backup: {previous_output}")
        return None

    entries, tool_calls = load_entries(rollouts)
    now = dt.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{now}-session-backup.md"
    user_count = sum(1 for _, kind, _ in entries if kind == "user")
    assistant_count = sum(1 for _, kind, _ in entries if kind == "assistant")
    note = sanitize(note.strip() or "incremental transcript snapshot")

    summary = [f"- {note}", ""] + auto_summary(entries)
    lines = [
        f"# Session backup {now}",
        "",
        f"- thread: `{thread}`",
        f"- rollout files: {len(rollouts)}",
        f"- extracted: {user_count} user / {assistant_count} assistant messages, "
        f"{tool_calls} tool calls",
        "- sanitization: credentials, tokens, private URLs, emails, and user paths "
        "are redacted; tool arguments are intentionally not copied",
        "",
        "## Summary",
        "",
        *summary,
        "",
        "## Transcript",
        "",
    ]
    for ts, kind, text in entries:
        label = {"user": "USER", "assistant": "ASSISTANT", "tool": "tool"}.get(
            kind, kind)
        lines.append(f"### [{label}] {ts}")
        lines.append("")
        lines.append(sanitize(text) if kind != "tool" else f"`{text}`")
        lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")
    new_state = {
        "thread": thread,
        "source_hash": source_hash,
        "rollout_files": len(rollouts),
        "output": output.relative_to(PROJECT_ROOT).as_posix(),
        "updated_at": now,
        "user_messages": user_count,
        "assistant_messages": assistant_count,
        "tool_calls": tool_calls,
    }
    write_state(new_state)
    append_index(now, output, new_state, note)
    print(f"wrote {output} ({user_count}/{assistant_count}/{tool_calls} "
          f"user/assistant/tool from {len(rollouts)} rollout files)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread", default=DEFAULT_THREAD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--note", default="",
                        help="short factual summary appended to the snapshot")
    parser.add_argument("--force", action="store_true",
                        help="create a snapshot even if the source did not change")
    args = parser.parse_args()
    create_backup(args.thread, args.out, args.note, args.force)


if __name__ == "__main__":
    main()
