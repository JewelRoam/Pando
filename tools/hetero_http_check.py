"""Launch a heterogeneous HTTP swarm with Intel EUs plus an NVIDIA CUDA stem.

The root and Intel stems use the host Python environment. The RTX 3070 stem
uses a separate Python environment with ``onnxruntime-gpu``. Every worker
speaks the same tokenized HTTP protocol; the root aggregates their votes
without knowing which accelerator produced them.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(url: str, *, method: str = "GET", payload: dict | None = None,
            token: str | None = None, timeout: float = 90) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read())
        except Exception:
            detail = {"error": str(exc)}
        return exc.code, detail


def wait_root(url: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    last = "not attempted"
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/pairing", timeout=2)
        if code == 200 and payload.get("required") is True:
            return
        last = f"{code}: {payload}"
        time.sleep(0.25)
    raise RuntimeError(f"root not ready: {last}")


def wait_nodes(url: str, token: str, expected: set[str],
               timeout: float = 150) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        code, payload = request(f"{url}/nodes", token=token, timeout=3)
        if code == 200:
            last = payload
            if expected <= set(payload.get("nodes", {})):
                return payload
        time.sleep(0.5)
    raise RuntimeError(f"workers did not register: expected={expected}, "
                       f"last={last}")


def python_for_cuda(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    candidates = [
        ROOT / ".venv-cuda" / "Scripts" / "python.exe",
        ROOT / ".venv-cuda" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "CUDA Python not found; create .venv-cuda and install "
        "onnxruntime-gpu, or pass --cuda-python")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-port", type=int, default=0)
    parser.add_argument("--cuda-python", default=None)
    parser.add_argument("--skip-igpu", action="store_true")
    parser.add_argument("--skip-npu", action="store_true")
    parser.add_argument("--hotplug-cuda", action="store_true",
                        help="start Intel workers first, then join CUDA live")
    parser.add_argument("--torch-stem", action="store_true",
                        help="also join a native PyTorch (CUDA) Laya stem, a "
                             "non-OpenVINO ecosystem, into the same quorum")
    parser.add_argument("--torch-model", default="convaiinnovations/laya")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        parser.error("this starts real model workers; pass --yes after review")

    host_python = Path(sys.executable)
    cuda_python = python_for_cuda(args.cuda_python)
    root_port = args.root_port or free_port()
    root_url = f"http://127.0.0.1:{root_port}"
    en_dynamic = ROOT / "artifacts" / "laya-en-fixed.onnx"
    igpu_graph = ROOT / "artifacts" / "laya-en-static-fixed.onnx"
    npu_graph = ROOT / "artifacts" / "laya-en-static-nonisnan.onnx"
    if not en_dynamic.exists():
        raise SystemExit(f"missing ONNX graph: {en_dynamic}")

    hf_home = os.environ.get("PANDOSI_HF_HOME", str(ROOT / ".hf-cache"))
    base_env = {
        **os.environ,
        "HF_HOME": hf_home,
        "PYTHONPATH": str(ROOT) + os.pathsep +
        os.environ.get("PYTHONPATH", ""),
        "PYTHONIOENCODING": "utf-8",
    }
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
        if sys.platform == "win32" else 0

    workers = [
        {
            "node_id": "intel-cpu-ort",
            "python": host_python,
            "args": ["--backend", "onnx", "--onnx-path", str(en_dynamic),
                     "--device", "CPU"],
        },
        {
            "node_id": "nvidia-3070-cuda",
            "python": cuda_python,
            "args": ["--backend", "onnx", "--onnx-path", str(en_dynamic),
                     "--device", "CUDA"],
        },
    ]
    if not args.skip_igpu:
        if not igpu_graph.exists():
            raise SystemExit(f"missing iGPU graph: {igpu_graph}")
        workers.append({
            "node_id": "intel-igpu-openvino",
            "python": host_python,
            "args": ["--backend", "onnx", "--onnx-path", str(igpu_graph),
                     "--device", "GPU", "--pad-to", "256"],
        })
    if not args.skip_npu:
        if not npu_graph.exists():
            raise SystemExit(f"missing NPU graph: {npu_graph}")
        workers.append({
            "node_id": "intel-npu-openvino",
            "python": host_python,
            "args": ["--backend", "onnx", "--onnx-path", str(npu_graph),
                     "--device", "NPU", "--pad-to", "256"],
        })
    if args.torch_stem:
        # Same Laya model, different ecosystem: native PyTorch on CUDA rather
        # than ONNX Runtime. Nothing about the HTTP contract changes.
        workers.append({
            "node_id": "nvidia-3070-torch",
            "python": cuda_python,
            "args": ["--backend", "laya", "--device", "CUDA",
                     "--model", args.torch_model],
        })

    summary: dict = {
        "root_url": root_url,
        "workers": [worker["node_id"] for worker in workers],
        "cuda_python": str(cuda_python),
        "steps": [],
    }

    with tempfile.TemporaryDirectory(prefix="pandosi-hetero-") as td:
        work = Path(td)
        token = secrets.token_urlsafe(24)
        token_file = work / "pairing.token"
        token_file.write_text(token, encoding="utf-8")
        root_out = work / "rootd.out"
        root_err = work / "rootd.err"
        procs: list[tuple[str, subprocess.Popen, Path, Path]] = []
        root_proc: subprocess.Popen | None = None

        def cleanup() -> None:
            for _, proc, _, _ in procs:
                if proc.poll() is None:
                    proc.terminate()
            if root_proc and root_proc.poll() is None:
                root_proc.terminate()
            for proc in [*(entry[1] for entry in procs), root_proc]:
                if proc is None:
                    continue
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        try:
            with root_out.open("wb") as out, root_err.open("wb") as err:
                root_proc = subprocess.Popen(
                    [str(host_python), "-m", "pandosi.rootd",
                     "--port", str(root_port),
                     "--pairing-token-file", str(token_file)],
                    cwd=ROOT, stdout=out, stderr=err, env=base_env,
                    creationflags=creationflags)
            wait_root(root_url)
            code, payload = request(
                f"{root_url}/register", method="POST",
                payload={"node_id": "unauthorized",
                         "url": "http://127.0.0.1:1",
                         "checkpoint": "mock"})
            summary["steps"].append({"step": "reject_missing_token",
                                     "code": code})
            if code != 403:
                raise RuntimeError(
                    f"unauthenticated registration was not rejected: {payload}")

            def start_worker(worker):
                out_path = work / f"{worker['node_id']}.out"
                err_path = work / f"{worker['node_id']}.err"
                with out_path.open("wb") as out, err_path.open("wb") as err:
                    proc = subprocess.Popen(
                        [str(worker["python"]), "-m", "pandosi.stemd",
                         "--node-id", worker["node_id"],
                         "--root", root_url,
                         "--pairing-token-file", str(token_file),
                         *worker["args"]],
                        cwd=ROOT, stdout=out, stderr=err, env=base_env,
                        creationflags=creationflags)
                procs.append((worker["node_id"], proc, out_path, err_path))

            cuda_worker = next(
                worker for worker in workers
                if worker["node_id"] == "nvidia-3070-cuda")
            if args.hotplug_cuda:
                initial_workers = [
                    worker for worker in workers
                    if worker["node_id"] != "nvidia-3070-cuda"]
                late_workers = [cuda_worker]
            else:
                initial_workers = workers
                late_workers = []

            for worker in initial_workers:
                start_worker(worker)
            nodes = wait_nodes(root_url, token,
                               {worker["node_id"]
                                for worker in initial_workers})
            summary["registered"] = sorted(nodes["nodes"])
            summary["steps"].append({"step": "register_initial", "code": 0,
                                     "count": len(summary["registered"])})

            def run_decision(state):
                started = time.perf_counter()
                code, result = request(
                    f"{root_url}/decide", method="POST",
                    payload={"state": state,
                             "question": "approve?",
                             "candidates": ["approve", "reject"]},
                    token=token, timeout=120)
                elapsed_ms = (time.perf_counter() - started) * 1000
                if code != 200:
                    raise RuntimeError(f"decision failed ({code}): {result}")
                votes = result.get("votes", [])
                return {
                    "choice": result.get("choice"),
                    "tier": result.get("tier"),
                    "responded": result.get("responded"),
                    "latency_ms": round(elapsed_ms, 1),
                    "root_fanout_ms": result.get("fanout_ms"),
                    "checkpoints": sorted(
                        vote["checkpoint"] for vote in votes),
                }

            summary["baseline_decisions"] = [
                run_decision(f"heterogeneous baseline vote {index}")
                for index in range(2)
            ]
            if any(decision["responded"] != len(initial_workers)
                   for decision in summary["baseline_decisions"]):
                raise RuntimeError(
                    "baseline responded mismatch: "
                    f"{summary['baseline_decisions']}")
            summary["steps"].append({"step": "baseline_decisions", "code": 0})

            if late_workers:
                hotplug_started = time.perf_counter()
                for worker in late_workers:
                    start_worker(worker)
                nodes = wait_nodes(root_url, token,
                                   {worker["node_id"] for worker in workers})
                hotplug_ms = (time.perf_counter() - hotplug_started) * 1000
                summary["hotplug"] = {
                    "workers": [worker["node_id"]
                                for worker in late_workers],
                    "registered": sorted(nodes["nodes"]),
                    "elapsed_ms": round(hotplug_ms, 1),
                }
                summary["steps"].append({"step": "hotplug_cuda", "code": 0,
                                         "elapsed_ms": round(hotplug_ms, 1)})
            else:
                summary["hotplug"] = None

            latencies = []
            decisions = []
            for index in range(5):
                decision = run_decision(f"heterogeneous HTTP vote {index}")
                latencies.append(decision["latency_ms"])
                decisions.append(decision)

            summary["wire_latency_p50_ms"] = round(
                statistics.median(latencies), 1)
            summary["root_fanout_p50_ms"] = round(statistics.median(
                decision["root_fanout_ms"] for decision in decisions
                if decision["root_fanout_ms"] is not None), 1)
            code, nodes_after = request(
                f"{root_url}/nodes", token=token, timeout=3)
            if code == 200:
                summary["node_stats"] = {
                    node_id: info.get("stats", {})
                    for node_id, info in nodes_after.get("nodes", {}).items()
                }
            summary["decisions"] = decisions
            summary["steps"].append({"step": "decide", "code": 0, "count": 5})
            expected_responded = len(workers)
            for decision in decisions:
                if decision["responded"] != expected_responded:
                    raise RuntimeError(
                        f"responded={decision['responded']}, "
                        f"expected={expected_responded}: {decision}")
            expected_checkpoints = {
                "onnx", "onnx-cuda",
                *({"onnx-gpu"} if not args.skip_igpu else set()),
                *({"onnx-npu"} if not args.skip_npu else set()),
                *({"torch-cuda"} if args.torch_stem else set()),
            }
            expected_initial = {
                checkpoint
                for checkpoint in expected_checkpoints
                if not (args.hotplug_cuda and checkpoint == "onnx-cuda")
            }
            for decision in summary["baseline_decisions"]:
                if set(decision["checkpoints"]) != expected_initial:
                    raise RuntimeError(
                        "baseline checkpoint mismatch: "
                        f"{decision['checkpoints']} != "
                        f"{sorted(expected_initial)}")
            if set(decisions[-1]["checkpoints"]) != expected_checkpoints:
                raise RuntimeError(
                    "checkpoint diversity mismatch: "
                    f"{decisions[-1]['checkpoints']} != "
                    f"{sorted(expected_checkpoints)}")
            summary["status"] = "OK"
        except Exception as exc:
            summary["status"] = "FAILED"
            summary["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            cleanup()
            summary["root_stdout_tail"] = root_out.read_text(
                encoding="utf-8", errors="replace").splitlines()[-12:]
            summary["root_stderr_tail"] = root_err.read_text(
                encoding="utf-8", errors="replace").splitlines()[-12:]
            summary["worker_logs"] = {
                node_id: {
                    "stdout": out_path.read_text(
                        encoding="utf-8", errors="replace").splitlines()[-12:],
                    "stderr": err_path.read_text(
                        encoding="utf-8", errors="replace").splitlines()[-12:],
                }
                for node_id, _, out_path, err_path in procs
            }

    report_dir = ROOT / "metrics" / "hetero-checks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"hetero-{int(time.time())}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
