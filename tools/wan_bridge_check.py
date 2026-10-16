"""First WAN bridge: Windows parent root <- bridge <- island root (tunnel).

The island URL is read from the mac-endpoint control plane and cross-checked
against the Task 023 report; the island's pairing bearer comes from a local
secret file (the user-provisioned root token). Six decisions (three from the
old family, three from the new intent family) go through the parent root.

Usage: python tools/wan_bridge_check.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = ROOT / "metrics" / "lan-check"


def load_island_url() -> str:
    subprocess.run(["git", "fetch", "origin", "mac-endpoint"],
                   capture_output=True)
    blob = subprocess.run(
        ["git", "show", "origin/mac-endpoint:endpoint/mac.json"],
        capture_output=True).stdout
    endpoint = json.loads(blob)
    island = endpoint.get("island_root_url")
    report = (WORK / "task-023-output.md").read_text(encoding="utf-8")
    if not island or island not in report:
        raise SystemExit("island URL missing or not cross-confirmed by the "
                         "Task 023 report")
    return island.rstrip("/")


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from pando.common import DecisionRequest
    from pando.rootd import Coordinator, Root

    island = load_island_url()
    token = (WORK / "mac-root.token").read_text(encoding="ascii").strip()
    parent = Root(coordinator=Coordinator(quorum_k=1, call_timeout=60.0,
                                          initial_stem_timeout=120.0))
    parent.start()
    bridge = subprocess.Popen(
        [sys.executable, "-m", "pando.bridge",
         "--node-id", "mac-island",
         "--upstream-root", island,
         "--upstream-pairing-token", token,
         "--local-root", parent.url(),
         "--checkpoint", "mac-island", "--host", "127.0.0.1"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT, creationflags=0x08000000)
    rows = []
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and \
                "mac-island" not in parent.coordinator.live_nodes():
            time.sleep(1.0)
        if "mac-island" not in parent.coordinator.live_nodes():
            raise SystemExit("bridge did not register with the parent root")
        print("bridge registered; parent /nodes:", list(
            parent.coordinator.live_nodes()))
        tasks = []
        for suite, count, family in (("assets/heldout3_test.jsonl", 3, None),
                                     ("assets/heldout5_test.jsonl", 3,
                                      "general")):
            data = [json.loads(line) for line in
                    (ROOT / suite).read_text(encoding="utf-8").splitlines()
                    if line.strip()][:count]
            tasks.extend((task, family) for task in data)
        for task, family in tasks:
            request = DecisionRequest(state=task["state"],
                                      question=task["question"],
                                      candidates=task["candidates"],
                                      family=family)
            started = time.perf_counter()
            with urllib.request.urlopen(
                    urllib.request.Request(
                        f"{parent.url()}/decide", data=request.to_json(),
                        headers={"Content-Type": "application/json"}),
                    timeout=120) as response:
                body = json.loads(response.read())
            latency = (time.perf_counter() - started) * 1000
            rows.append({"task_id": task["task_id"], "expected":
                         task["expected"], "choice": body.get("choice"),
                         "tier": body.get("tier"), "family": family,
                         "responded": body.get("responded"),
                         "ms": round(latency, 1)})
            print(f"  {task['task_id']} family={family} -> "
                  f"choice={body.get('choice')} tier={body.get('tier')} "
                  f"({latency:.0f} ms)")
    finally:
        bridge.terminate()
        try:
            bridge.wait(timeout=15)
        except subprocess.TimeoutExpired:
            bridge.kill()
        parent.stop()
    hits = sum(1 for row in rows if row["choice"] == row["expected"])
    report = {"island": island, "rows": rows,
              "accuracy": round(hits / len(rows), 4)}
    out = ROOT / "metrics" / "ensemble-checks" / "wan-bridge.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"decided {hits}/{len(rows)} = {hits / len(rows):.4f}; report {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
