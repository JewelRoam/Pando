"""Probe llama-server context-guard behavior on the local GPU node.

Evidence for the claim "long context cannot crash the 3070 node":
VRAM is pre-allocated at load, the context window never grows at runtime,
and an over-context request is rejected with a clean HTTP 400 while the
server keeps serving. Run against the resident server:

    python tools/ctx_guard_probe.py [--host 127.0.0.1] [--port 8086]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request

FILLER = "The quick brown fox jumps over the lazy dog. "  # ~10 tokens per rep


def vram_used() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10)
    return int(out.stdout.strip().splitlines()[0])


class VramSampler:
    """Poll VRAM in the background while a request is in flight."""

    def __init__(self) -> None:
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(vram_used())
            except Exception:  # noqa: BLE001 - sampling must never break the probe
                pass
            time.sleep(0.25)

    def __enter__(self) -> "VramSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def peak(self) -> int:
        return max(self.samples) if self.samples else 0


def post(base: str, content: str, max_tokens: int = 64) -> tuple[int, float, dict | str]:
    body = json.dumps({"messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, time.time() - start, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, time.time() - start, exc.read().decode("utf-8", "replace")


def health(base: str) -> str:
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as resp:
            return json.loads(resp.read()).get("status", "?")
    except Exception as exc:  # noqa: BLE001
        return f"DOWN ({exc})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8086)
    args = parser.parse_args()
    base = f"http://{args.host}:{args.port}"

    print(f"health before: {health(base)} | vram baseline {vram_used()} MiB")

    in_bounds = FILLER * 300 + "\nReply with exactly: OK"
    with VramSampler() as sampler:
        code, dt, data = post(base, in_bounds)
    if isinstance(data, dict):
        usage = data.get("usage", {})
        print(f"in-bounds: HTTP {code} in {dt:.1f}s "
              f"prompt_tokens={usage.get('prompt_tokens')} "
              f"completion_tokens={usage.get('completion_tokens')} "
              f"vram_peak={sampler.peak} MiB")
    else:
        print(f"in-bounds: HTTP {code} in {dt:.1f}s "
              f"body={data[:200]!r} vram_peak={sampler.peak} MiB")
    print(f"health after in-bounds: {health(base)}")

    over = FILLER * 600 + "\nReply with exactly: OK"
    code, dt, data = post(base, over)
    detail = data if isinstance(data, str) else json.dumps(data)[:160]
    print(f"over-context: HTTP {code} in {dt:.1f}s -> {detail!r}")
    status_after = health(base)
    print(f"health after over-context: {status_after} | vram now {vram_used()} MiB")

    passed = code == 400 and status_after == "ok"
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
