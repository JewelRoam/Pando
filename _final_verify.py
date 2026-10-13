# -*- coding: utf-8 -*-
import sys
import time

sys.path.insert(0, r"D:\PandoSI")
from pandosi.backend import OnnxLayaBackend
from pandosi.common import DecisionRequest

STATE = "User message: I want a refund, the order never arrived"

gpu = OnnxLayaBackend("gpu-en",
                      r"D:\PandoSI\artifacts\laya-en-static-fixed.onnx",
                      device="GPU", pad_to=256)
print("GPU backend loaded (static surgery graph, warmup done)", flush=True)

req = DecisionRequest(state=STATE, question="intent",
                      candidates=["refund", "complaint", "greeting"])
times = []
for i in range(6):
    t0 = time.perf_counter()
    v = gpu.decide(req)
    times.append((time.perf_counter() - t0) * 1000)
    print(f"run{i}: {v.choice} @ {v.probability} ({times[-1]:.0f} ms)", flush=True)
import statistics
print(f"GPU decision p50: {statistics.median(times):.0f} ms", flush=True)
assert v.choice == "refund" and 0 < v.probability <= 1
print("STATIC IGPU END-TO-END OK", flush=True)
