#!/usr/bin/env python3
"""Serve sealed seed bundles that are already on disk.

The executor copies a bundle into the outbox. A peer pulls it. Nothing
here unwraps a key, and a directory that does not pass ``judge_bundle``
is not listed and not packed.
"""
from __future__ import annotations

import argparse
import hmac
import io
import json
import os
import subprocess
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_OFFER_BYTES = 2 * 1024 * 1024

# One cache path can be requested by several mailbox threads at once.  The
# unique temporary files below already protect the destination from torn
# writes; this lock additionally avoids doing the same expensive git bundle
# build once per concurrent waiter in this process.
_BUNDLE_LOCKS: dict[str, threading.Lock] = {}
_BUNDLE_LOCKS_GUARD = threading.Lock()


def _bundle_lock(dest: Path) -> threading.Lock:
    key = str(dest.resolve())
    with _BUNDLE_LOCKS_GUARD:
        lock = _BUNDLE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _BUNDLE_LOCKS[key] = lock
        return lock


def _seed():
    import importlib.util
    tool = Path(__file__).resolve().parent / "seed.py"
    spec = importlib.util.spec_from_file_location("pandosi_seed_outbox_tool", tool)
    if spec is None or spec.loader is None:
        raise RuntimeError("seed tool is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def offer_dirs(outbox: Path) -> list[Path]:
    outbox = Path(outbox).expanduser()
    if not outbox.is_dir():
        return []
    seed = _seed()
    found = []
    for child in sorted(outbox.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if seed.judge_bundle(child).get("accepted"):
            found.append(child)
    return found


def pack_offer(src: Path) -> bytes | None:
    """Tar of the manifest and the files it names. None when it must not leave."""
    seed = _seed()
    loaded = seed.read_bundle(src)
    if loaded is None or not seed.judge_bundle(src).get("accepted"):
        return None
    manifest, src = loaded
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        archive.add(src / seed.MANIFEST, arcname=seed.MANIFEST)
        for entry in manifest["files"]:
            archive.add(src / entry["path"], arcname=entry["path"])
    packed = buf.getvalue()
    if len(packed) > MAX_OFFER_BYTES:
        return None
    return packed


def write_ack(outbox: Path, proposal_id: str, body: dict) -> bool:
    seed = _seed()
    if seed._INBOX_NAME.fullmatch(proposal_id) is None:
        return False
    src = Path(outbox).expanduser() / proposal_id
    if not seed.judge_bundle(src).get("accepted"):
        return False
    payload = {"proposal_id": proposal_id, "received": True,
               "by": str(body.get("by") or "")[:80]}
    path = Path(outbox).expanduser() / f"{proposal_id}.ack.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return True



def _git_env() -> dict[str, str]:
    """Prevent an invoking git hook from redirecting bundle operations."""
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        env.pop(key, None)
    return env


def _repo_head(repo: Path, env: dict[str, str]) -> str:
    """The commit the bundle would be built from, or "" if git cannot say."""
    try:
        got = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, env=env, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    return got.stdout.strip() if got.returncode == 0 else ""


def code_bundle(repo: Path, dest: Path, max_age: float = 600) -> Path | None:
    """A git bundle of ``main``. Rebuilt when HEAD moves, or when it is old.

    This is the same history as origin, served only when a peer cannot
    fetch GitHub. It is not a second branch.

    The cache is keyed on HEAD, not only on age. A peer that has no route to
    GitHub reads this file as its *only* source of new code, so a commit that
    landed a minute ago must not sit unserved for the remaining nine minutes
    of an age-based window.
    """
    repo = Path(repo)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _bundle_lock(dest):
        stamp = dest.parent / (dest.name + ".head")
        env = _git_env()
        head = _repo_head(repo, env)
        if dest.is_file() and time.time() - dest.stat().st_mtime < max_age:
            try:
                served = stamp.read_text(encoding="utf-8").strip()
            except OSError:
                served = ""
            if head and served == head:
                return dest
            if not head and served == "":
                # git could not answer; fall back to the age window rather
                # than rebuilding a bundle on every single request.
                return dest

        # A unique temporary file means two independent server processes
        # cannot overwrite each other's half-written bundle.  os.replace is
        # atomic on both POSIX and Windows, so a reader sees old or new bytes,
        # never a truncated file.
        partial = dest.with_name(
            f".{dest.name}.{os.getpid()}.{threading.get_ident()}."
            f"{time.time_ns()}.partial")
        try:
            got = subprocess.run(
                ["git", "-C", str(repo), "bundle", "create",
                 str(partial), "HEAD"],
                capture_output=True, text=True, env=env, timeout=60)
            if got.returncode != 0 or not partial.is_file():
                return None
            if partial.stat().st_size > 30 * 1024 * 1024:
                return None
            partial.replace(dest)
        finally:
            partial.unlink(missing_ok=True)
        if head:
            try:
                stamp.write_text(head + "\n", encoding="utf-8")
            except OSError:
                pass
        return dest


def make_handler(outbox: Path, token: str, repo: Path | None = None):
    outbox = Path(outbox).expanduser()
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _authed(self) -> bool:
            header = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            if hmac.compare_digest(header, expected):
                return True
            self._send(403, {"error": "missing or wrong token"})
            return False

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._send(200, {"ok": True, "service": "seed-outbox"})
                return
            if not self._authed():
                return
            if path == "/code.bundle":
                built = code_bundle(repo, Path.home() / ".pando" / "code.bundle")
                if built is None:
                    self._send(404, {"error": "no bundle"})
                    return
                raw = built.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if path == "/offers":
                rows = []
                for child in offer_dirs(outbox):
                    verdict = _seed().judge_bundle(child)
                    rows.append({"proposal_id": child.name,
                                 "fingerprint": verdict.get("fingerprint") or ""})
                self._send(200, {"offers": rows})
                return
            prefix = "/offer/"
            if path.startswith(prefix) and path.endswith(".tar"):
                name = path[len(prefix):-4]
                packed = pack_offer(outbox / name)
                if packed is None:
                    self._send(404, {"error": "no sealed offer"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-tar")
                self.send_header("Content-Length", str(len(packed)))
                self.end_headers()
                self.wfile.write(packed)
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if not self._authed():
                return
            path = self.path.split("?", 1)[0]
            prefix = "/offer/"
            if not path.startswith(prefix) or not path.endswith("/ack"):
                self._send(404, {"error": "not found"})
                return
            name = path[len(prefix):-len("/ack")]
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(min(length, 4096)) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = {}
            if not isinstance(body, dict) or not write_ack(outbox, name, body):
                self._send(404, {"error": "no sealed offer"})
                return
            self._send(200, {"acked": name})

    return Handler


def serve(outbox: Path, token: str, host: str, port: int, repo: Path | None = None) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(outbox, token, repo))
    print(f"seed-outbox on {host}:{port}", flush=True)
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outbox", default="~/.pando/outbox")
    parser.add_argument("--token-file", default="~/.pando/mailbox.token")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--repo", default="")
    args = parser.parse_args(argv)
    token = Path(args.token_file).expanduser().read_text(encoding="utf-8").strip()
    if not token:
        print("refusing to serve without a token", flush=True)
        return 2
    repo = Path(args.repo).expanduser() if args.repo else None
    serve(Path(args.outbox), token, args.bind, args.port, repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
