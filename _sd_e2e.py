# -*- coding: utf-8 -*-
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request

env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
token = secrets.token_hex(16)

def get_nodes():
    try:
        return sorted(json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8099/nodes", timeout=5).read())["nodes"].keys())
    except Exception:
        return []

# boot root
root = subprocess.Popen([sys.executable, "-m", "pandosi.rootd",
                         "--port", "8099", "--auth-token", token],
                        cwd=r"D:\PandoSI", env=env)
time.sleep(3)
stem = subprocess.Popen([sys.executable, "-m", "pandosi.stemd",
                         "--node-id", "s0", "--root", "http://127.0.0.1:8099"],
                        cwd=r"D:\PandoSI", env=env)
time.sleep(8)
print("before:", get_nodes(), flush=True)

# shutdown
req = urllib.request.Request("http://127.0.0.1:8099/shutdown",
                             data=json.dumps({"token": token}).encode(),
                             headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=5)
time.sleep(2)
print("root alive after shutdown:", root.poll() is None, flush=True)

# restart root
root2 = subprocess.Popen([sys.executable, "-m", "pandosi.rootd",
                          "--port", "8099", "--auth-token", token],
                         cwd=r"D:\PandoSI", env=env)
# stem heartbeat retries every 5s, give it 20s
time.sleep(20)
print("after restart:", get_nodes(), flush=True)
ok = "s0" in get_nodes()
print("SHUTDOWN E2E", "OK" if ok else "PARTIAL", flush=True)
root2.terminate()
stem.terminate()
