# -*- coding: utf-8 -*-
# Diagnose the multi-EU registration flake: run with flushed output and
# longer wait to see exactly where it hangs.
import json
import os
import subprocess
import sys
import time
import urllib.request

env = {**os.environ, "HF_ENDPOINT": "https://hf-mirror.com",
       "HF_HOME": r"D:\PandoSI\.hf-cache",
       "PYTHONIOENCODING": "utf-8",
       "HTTP_PROXY": "http://127.0.0.1:54056",
       "HTTPS_PROXY": "http://127.0.0.1:54056"}

root_log = open(r"D:\PandoSI\mu3_root_log.txt", "w", encoding="utf-8")
root = subprocess.Popen([sys.executable, "-m", "pandosi.rootd", "--port", "8098",
                         "--playbook", r"assets\playbook.example.toml"],
                        cwd=r"D:\PandoSI", env=env,
                        stdout=root_log, stderr=subprocess.STDOUT)
time.sleep(3)
join_log = open(r"D:\PandoSI\mu3_join_log.txt", "w", encoding="utf-8")
joiner = subprocess.Popen([sys.executable, "-u", "-m", "pandosi.join",
                           "--node-id", "aipc", "--multi-eu",
                           "--backend", "onnx",
                           "--onnx-path", r"artifacts\laya-en-static-fixed.onnx",
                           "--root", "http://127.0.0.1:8098"],
                          cwd=r"D:\PandoSI", env=env,
                          stdout=join_log, stderr=subprocess.STDOUT)
try:
    registered = []
    for attempt in range(36):
        time.sleep(5)
        alive = joiner.poll() is None
        try:
            nodes = json.loads(urllib.request.urlopen(
                "http://127.0.0.1:8098/nodes", timeout=5).read())
            registered = sorted(nodes["nodes"].keys())
        except Exception:
            pass
        print(f"t={(attempt+1)*5}s alive={alive} reg={registered}", flush=True)
        if not alive or (len(registered) >= 2 and attempt >= 14):
            break
    # Decide AFTER both registered
    if len(registered) >= 2:
        time.sleep(10)  # let warmup settle
        body = json.dumps({
            "state": "User message: I want a refund, the order never arrived",
            "question": "intent",
            "candidates": ["refund", "complaint", "greeting"],
        }).encode()
        t0 = time.perf_counter()
        req = urllib.request.Request("http://127.0.0.1:8098/decide", data=body,
                                     headers={"Content-Type": "application/json"})
        result = json.loads(urllib.request.urlopen(req, timeout=120).read())
        dt = time.perf_counter() - t0
        print(f"\ndecide: {dt*1000:.0f} ms | responded={result['responded']}",
              flush=True)
        for v in result["votes"]:
            print(f"  {v['node_id']:>12} ({v['checkpoint']}): "
                  f"{v['choice']} @ {v['probability']}", flush=True)
        print(f"  -> {result['choice']} tier={result['tier']} "
              f"agreement={result['agreement']}", flush=True)
        ckpts = {v["checkpoint"] for v in result["votes"]}
        verdict = "OK" if result["responded"] == 2 and len(ckpts) == 2 else "PARTIAL"
    else:
        verdict = "FAIL (not all registered)"
finally:
    join_log.close()
    print("\n=== JOINER LOG ===", flush=True)
    print(open(r"D:\PandoSI\mu3_join_log.txt",
               encoding="utf-8").read()[-800:], flush=True)
    root.terminate()
    joiner.terminate()
print(f"MULTI-EU SMOKE {verdict}", flush=True)
