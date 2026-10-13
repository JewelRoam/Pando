# -*- coding: utf-8 -*-
import json
import os
import sys

sys.path.insert(0, r"D:\PandoSI")
os.chdir(r"D:\PandoSI")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = r"D:\PandoSI\.hf-cache"
from laya.onnx_agent import ONNXAgent

SUITES = {
    "english": ("convaiinnovations/laya",
                r"D:\PandoSI\artifacts\laya-en.onnx"),
    "multilingual": ("convaiinnovations/laya-multilingual",
                     r"D:\PandoSI\artifacts\laya-ml.onnx"),
    "typed-decisions": ("convaiinnovations/laya-typed-decisions",
                        r"D:\PandoSI\artifacts\laya-td.onnx"),
}
tasks = [json.loads(l) for l in open(r"D:\PandoSI\assets\graft_suite.jsonl",
                                     encoding="utf-8")]
score_tasks = [t for t in tasks if t["qtype"] == "score"]
RISK = ["negligible", "low", "moderate", "high", "critical"]

for ckpt, (repo, onnx) in SUITES.items():
    agent = ONNXAgent(repo, onnx_path=onnx)
    print(f"[{ckpt}]", flush=True)
    for t in score_tasks:
        q = {"risk": {"type": "score", "instructions": t["question"],
                      "criteria": t["criteria"]}}
        result = agent.predict(t["state"], q)
        ans = result["answers"]["risk"]
        score_val = ans.get("score")
        probs = ans.get("probabilities") or {}
        top = max(probs, key=lambda k: probs[k]) if probs else None
        top_label = (ans.get("legend") or {}).get(top, top)
        expected_idx = RISK.index(t["expected"]) if t["expected"] in RISK else -1
        dist = abs(score_val - expected_idx) if isinstance(score_val, (int, float)) else None
        verdict = "PASS" if (dist is not None and round(score_val) == expected_idx) else "FAIL"
        print(f"  {t['task_id']}: expected {t['expected']} (idx {expected_idx}) | "
              f"score={score_val:.2f} top={top_label} | "
              f"{verdict}{' dist=' + format(dist, '.2f') if verdict == 'FAIL' else ''}",
              flush=True)
