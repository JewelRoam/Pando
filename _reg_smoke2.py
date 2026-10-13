# -*- coding: utf-8 -*-
import json
import subprocess
import sys
import time
import urllib.request

procs = [subprocess.Popen([sys.executable, "-m", "pandosi.rootd", "--port", "8088"], cwd=".")]
join_log = open("reg_join_log.txt", "w", encoding="utf-8")
procs.append(subprocess.Popen([sys.executable, "-m", "pandosi.join", "--node-id", "sprout1",
                               "--onnx-path", "spore-cache/sprout1/laya-en.int8.onnx",
                               "--root", "http://127.0.0.1:8088"], cwd=".",
                              stdout=join_log, stderr=subprocess.STDOUT))
try:
    registered = []
    for _ in range(30):
        time.sleep(5)
        try:
            nodes = json.loads(urllib.request.urlopen(
                "http://127.0.0.1:8088/nodes", timeout=5).read())
            registered = sorted(nodes["nodes"].keys())
        except Exception:
            pass
        if "sprout1" in registered:
            break
    print("registered:", registered, flush=True)
    print("REG-AFTER-SPORE", "OK" if "sprout1" in registered else "FAIL", flush=True)
finally:
    join_log.close()
    print("--- joiner log ---", flush=True)
    print(pathlib.Path("reg_join_log.txt").read_text(encoding="utf-8")[-500:], flush=True)
    for p in procs:
        p.terminate()
