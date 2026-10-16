"""Measure the Round 93 pipeline rule on one host.

Chains the two stages produced by tools/stage_split.py over real HTTP and
compares the two-stage latency against the single-device forward, using a
real tokenized decision feed captured from the Laya ONNX agent.

Reports T_a (stage A), T_b (stage B compute), L (activation transfer round
trip minus server compute) and the verdict on the rule
``L < per_stage_compute => stage``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import struct
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

from tools.stage_split import split_graph  # noqa: E402

from pando.common import bypass_proxy_for_lan  # noqa: E402

bypass_proxy_for_lan()  # Windows registry proxies black-hole loopback

# Never route the stage hop through a system/env proxy.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
urllib.request.install_opener(OPENER)


def encode(header: dict, arrays: list) -> bytes:
    head = json.dumps(header).encode("utf-8")
    return (struct.pack(">I", len(head)) + head
            + b"".join(np.ascontiguousarray(a).tobytes() for a in arrays))


def decode(body: bytes) -> tuple:
    (head_len,) = struct.unpack(">I", body[:4])
    header = json.loads(body[4:4 + head_len])
    offset = 4 + head_len
    arrays = []
    for meta in header["tensors"]:
        count = int(np.prod(meta["shape"])) if meta["shape"] else 1
        itemsize = np.dtype(meta["dtype"]).itemsize
        raw = body[offset:offset + count * itemsize]
        offset += count * itemsize
        arrays.append(np.frombuffer(raw, dtype=meta["dtype"])
                      .reshape(meta["shape"]).copy())
    return header, arrays


class StageServer:
    """Serves one pipeline stage over HTTP: POST /forward -> activations."""

    def __init__(self, session, output_names: list):
        self.session = session
        self.output_names = output_names
        self.last_compute_ms = 0.0
        server = self

        class Handler(BaseHTTPRequestHandler):
            # Without this, Nagle + delayed ACK turn a 374 KiB loopback
            # activation transfer into tens of milliseconds of stalls.
            disable_nagle_algorithm = True

            def log_message(self, fmt, *args):
                pass

            def do_POST(self):  # noqa: N802
                if self.path not in ("/forward", "/ping"):
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                if self.path == "/ping":
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "application/octet-stream")
                    self.send_header("Content-Length", "8")
                    self.end_headers()
                    self.wfile.write(b"\x00" * 8)
                    return
                header, arrays = decode(body)
                feed = dict(zip(header["names"], arrays))
                started = time.perf_counter()
                outputs = server.session.run(server.output_names, feed)
                server.last_compute_ms = (time.perf_counter() - started) * 1000
                meta = [{"shape": list(o.shape), "dtype": str(o.dtype)}
                        for o in outputs]
                body = encode({"tensors": meta}, outputs)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("X-Compute-Ms",
                                 f"{server.last_compute_ms:.4f}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._httpd.shutdown()

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def capture_feed(onnx_path: Path) -> dict:
    """Capture one real tokenized input feed from the Laya ONNX agent."""
    from laya.onnx_agent import ONNXAgent

    captured = {}
    agent = ONNXAgent("convaiinnovations/laya", onnx_path=str(onnx_path))
    real = agent.session

    class Recorder:
        def run(self, output_names, input_feed, *args, **kwargs):
            if not captured:
                captured.update({k: np.asarray(v) for k, v in
                                 input_feed.items()})
            return real.run(output_names, input_feed, *args, **kwargs)

    agent.session = Recorder()
    agent.predict("The nightly build failed twice with a network timeout.",
                  {"q1": {"type": "choice",
                          "instructions": "Classify: choose the best action.",
                          "criteria": ["retry", "escalate", "abort"]}})
    if not captured:
        raise SystemExit("failed to capture an input feed")
    return captured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph",
                        default=str(ROOT / "artifacts" / "laya-en.onnx"))
    parser.add_argument("--cut", type=int, default=900)
    parser.add_argument("--device-a", default="CPU",
                        help="CPU | CUDA | GPU | NPU (ORT provider for stage A)")
    parser.add_argument("--device-b", default="CPU",
                        help="ORT provider for stage B")
    parser.add_argument("--device-full", default=None,
                        help="single-device baseline device (default: the "
                             "faster of A and B, i.e. 'CUDA' if requested)")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--threads", type=int, default=4,
                        help="intra-op threads per ORT session (dominates "
                             "the CPU-contention noise in this tool)")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="allowed |Δ| between the chained and single "
                             "device outputs; raise it for cross-provider "
                             "chains, where fp32 rounding differs")
    parser.add_argument("--tasks", type=int, default=8,
                        help="decisions per throughput measurement")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    import onnxruntime as ort

    graph = Path(args.graph)
    report = split_graph(graph, graph.parent, args.cut,
                         names=("stage-a.onnx", "stage-b.onnx"))
    interface = report["interface"]

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = args.threads

    def providers(device: str) -> list:
        device = device.strip().upper()
        if device == "CUDA":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    full_device = args.device_full or ("CUDA" if "CUDA" in
                                       (args.device_a.upper(),
                                        args.device_b.upper()) else "CPU")
    full = ort.InferenceSession(str(graph), so,
                                providers=providers(full_device))
    stage_a = ort.InferenceSession(report["stage_a"], so,
                                   providers=providers(args.device_a))
    stage_b = ort.InferenceSession(report["stage_b"], so,
                                   providers=providers(args.device_b))
    if "CUDA" in (args.device_a.upper(), args.device_b.upper(),
                  full_device.upper()):
        wanted = {"CUDA" if "cuda" in d.upper() else "cpu"
                  for d in (args.device_a, args.device_b, full_device)}
        got = {s.get_providers()[0] for s in (full, stage_a, stage_b)}
        if "CUDA" in wanted and "CUDAExecutionProvider" not in got:
            raise SystemExit("CUDA requested but not selected: " +
                             ", ".join(sorted(got)))
    full_outputs = [out.name for out in full.get_outputs()]
    a_outputs = [out.name for out in stage_a.get_outputs()]
    b_outputs = [out.name for out in stage_b.get_outputs()]
    if list(stage_b.get_inputs())[0].name not in interface:
        pass  # interface order is authoritative for the wire format

    feed = capture_feed(graph)
    full_inputs = {i.name for i in full.get_inputs()}
    feed = {k: v for k, v in feed.items() if k in full_inputs}
    a_inputs = {i.name for i in stage_a.get_inputs()}
    feed_a = {k: v for k, v in feed.items() if k in a_inputs}
    if set(a_inputs) - set(feed_a):
        raise SystemExit(f"stage A needs inputs not in the captured feed: "
                         f"{sorted(set(a_inputs) - set(feed_a))}")

    # Correctness: chained stages must reproduce the single-device outputs.
    reference = full.run(full_outputs, feed)
    activation = stage_a.run(a_outputs, feed_a)
    # Stage B also needs the small original inputs its nodes still read
    # (marker_pos/marker_mask/qtype); they travel with the activations.
    b_input_names = [value.name for value in stage_b.get_inputs()]
    extra_b = {name: feed[name] for name in b_input_names
               if name in feed and name not in a_outputs}
    chained = stage_b.run(
        b_outputs, {**dict(zip(a_outputs, activation)), **extra_b})
    max_delta = 0.0
    deltas = {}
    for name, ref, out in zip(full_outputs, reference, chained):
        delta = float(np.max(np.abs(ref.astype(np.float64)
                                    - out.astype(np.float64))))
        deltas[name] = round(delta, 8)
        max_delta = max(max_delta, float(np.max(np.abs(
            ref.astype(np.float64) - out.astype(np.float64)))))
        # Masked marker logits are filled with very large values, so an
        # absolute-only test is meaningless there; allclose's relative term
        # is what a real pipeline would compare on.
        if not np.allclose(ref, out, atol=args.atol, rtol=1e-4):
            raise SystemExit(
                f"stage split changed output {name!r} beyond atol="
                f"{args.atol}: max|Δ|={delta:.3e}")

    server = StageServer(stage_b, b_outputs)
    server.start()
    interface_bytes = sum(int(np.prod(t.shape)) * t.dtype.itemsize
                          for t in activation) + sum(
        int(np.prod(t.shape)) * t.dtype.itemsize for t in extra_b.values())

    def activation_body(arrays: list) -> bytes:
        names = a_outputs + list(extra_b)
        payload = arrays + list(extra_b.values())
        metas = [{"shape": list(a.shape), "dtype": str(a.dtype)}
                 for a in payload]
        return encode({"names": names, "tensors": metas}, payload)

    try:
        request = urllib.request.Request(
            f"{server.url()}/forward",
            data=activation_body(activation),
            headers={"Content-Type": "application/octet-stream"})
        ping = urllib.request.Request(
            f"{server.url()}/ping",
            data=activation_body(activation),
            headers={"Content-Type": "application/octet-stream"})

        full_ms, a_ms, round_ms, server_ms, link_rtt_ms = [], [], [], [], []
        for _ in range(args.repeats):
            started = time.perf_counter()
            full.run(full_outputs, feed)
            full_ms.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            activation = stage_a.run(a_outputs, feed_a)
            a_ms.append((time.perf_counter() - started) * 1000)

            request.data = activation_body(activation)
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=60) as resp:
                resp.read()
                server_ms.append(float(resp.headers["X-Compute-Ms"]))
            round_ms.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            with urllib.request.urlopen(ping, timeout=60) as resp:
                resp.read()
            link_rtt_ms.append((time.perf_counter() - started) * 1000)

        # The rule is about steady-state throughput, so measure it with the
        # same number of requests in flight on both sides: N decisions on the
        # single device, then N through the two-stage chain (stage A and
        # stage B overlap when more than one request is in flight).
        def single_worker(_):
            full.run(full_outputs, feed)

        def staged_worker(_):
            act = stage_a.run(a_outputs, feed_a)
            post = urllib.request.Request(
                f"{server.url()}/forward", data=activation_body(act),
                headers={"Content-Type": "application/octet-stream"})
            with urllib.request.urlopen(post, timeout=120) as resp:
                resp.read()

        baseline = {}
        throughput = {}
        for concurrency in (1, 2, 4):
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(single_worker, range(args.tasks)))
            baseline[concurrency] = round(
                args.tasks / (time.perf_counter() - started), 3)
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(staged_worker, range(args.tasks)))
            throughput[concurrency] = round(
                args.tasks / (time.perf_counter() - started), 3)
    finally:
        server.stop()

    link_ms = [r - s for r, s in zip(round_ms, server_ms)]
    med = statistics.median
    gains = {c: round(throughput[c] / baseline[c], 3) for c in throughput}
    best_concurrency = max(gains, key=gains.get)
    single_rps = baseline[best_concurrency]
    pipeline_rps = throughput[best_concurrency]

    summary = {
        "cut": args.cut,
        "nodes_a": report["nodes_a"],
        "nodes_b": report["nodes_b"],
        "interface_tensors": len(interface),
        "interface_bytes": interface_bytes,
        "ort_threads": args.threads,
        "device_a": args.device_a,
        "device_b": args.device_b,
        "device_full": full_device,
        "max_output_delta": round(max_delta, 10),
        "output_deltas": deltas,
        "atol": args.atol,
        "t_full_ms": round(med(full_ms), 3),
        "t_a_ms": round(med(a_ms), 3),
        "t_b_ms": round(med(server_ms), 3),
        "link_rtt_ms": round(med(link_rtt_ms), 3),
        "link_rtt_min_ms": round(min(link_rtt_ms), 3),
        "link_net_ms": round(med(link_ms), 3),
        "round_ms": round(med(round_ms), 3),
        "sequential_stage_ms": round(med(a_ms) + med(link_rtt_ms)
                                     + med(server_ms), 3),
        "single_rps": round(single_rps, 3),
        "baseline_by_concurrency": baseline,
        "throughput_by_concurrency": throughput,
        "gain_by_concurrency": gains,
        "best_concurrency": best_concurrency,
        "pipeline_rps": round(pipeline_rps, 3),
        "throughput_gain": gains[best_concurrency],
        # 1.0 = the two stages cost the same, 0.0 = one stage is everything.
        "stage_balance": round(min(med(a_ms), med(server_ms))
                               / max(med(a_ms), med(server_ms)), 3),
        "repeats": args.repeats,
    }
    stage_worth_it = (med(link_rtt_ms) < max(med(a_ms), med(server_ms))
                      and pipeline_rps > single_rps)
    summary["verdict"] = "stage" if stage_worth_it else "replicate"
    summary["latency_penalty_ms"] = round(
        med(a_ms) + med(link_rtt_ms) + med(server_ms) - med(full_ms), 3)

    out_dir = ROOT / "metrics" / "stage-checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"stage-{args.cut}-{int(time.time())}.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report_path)

    print(json.dumps(summary, indent=2) if args.json else
          f"interface: {summary['interface_tensors']} tensors, "
          f"{summary['interface_bytes']} bytes\n"
          f"T_full {summary['t_full_ms']} ms | T_a {summary['t_a_ms']} ms | "
          f"T_b {summary['t_b_ms']} ms | L(rtt) {summary['link_rtt_ms']} ms\n"
          f"throughput {summary['single_rps']} -> "
          f"{summary['pipeline_rps']} req/s "
          f"(x{summary['throughput_gain']}), balance "
          f"{summary['stage_balance']}\n"
          f"sequential {summary['sequential_stage_ms']} ms vs single "
          f"{summary['t_full_ms']} ms "
          f"(+{summary['latency_penalty_ms']} ms) -> {summary['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
