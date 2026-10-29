# Reproducing PandoSI results

Every number in this repository traces to a reproducible experiment.
This guide walks through each one, from simplest to most complex.

## Prerequisites

```bash
# Python 3.11+, then:
pip install pynacl zeroconf                                   # core
pip install laya onnxruntime onnx onnxruntime-directml        # real model
pip install openvino                                           # NPU/iGPU
```

## 1. One-command demo (no model weights needed)

```bash
python tools/demo.py
# Expected: 3 stems register, 3 votes, dashboard rendered
# Time: <30 seconds
```

## 2. Capability report

```bash
python -m pando.doctor
# Expected: 10-12 checks, all [ok] on a capable machine
```

## 3. Real Laya decisions (CPU)

```bash
# Download weights (~800MB, one time):
python -c "
from laya.onnx_agent import ONNXAgent
a = ONNXAgent('convaiinnovations/laya', onnx_path='artifacts/laya-en.onnx')
print('loaded')
"
# Then:
python tools/laya_smoke.py
# Expected: refund @ ~0.98, 50-160ms depending on CPU
```

## 4. ONNX export (fp32 + int8)

```bash
python tools/vendor/export_onnx.py --model convaiinnovations/laya \
    --output artifacts/laya-en.onnx --quantize
# Expected: ~1.6GB fp32 + ~571MB int8
```

## 5. Static export for GPU/NPU

```bash
python tools/export_static.py --seq 256 --markers 8
python tools/fix_gathernd.py --in artifacts/laya-en-static.onnx \
    --out artifacts/laya-en-static-fixed.onnx
python tools/postprocess_onnx.py --in artifacts/laya-en-static-fixed.onnx \
    --out artifacts/laya-en-static-nonisnan.onnx
# Expected: 3 files, IsNaN=0, GatherND=0 in the final output
```

## 6. Multi-EU consensus (CPU + iGPU + NPU)

```bash
# Terminal 1: root with playbook
python -m pando.coordinator --port 8098 --playbook assets/playbook.example.toml

# Terminal 2: join with multi-EU
python -m pando.join --node-id aipc --multi-eu --backend onnx \
    --onnx-path artifacts/laya-ml.onnx --root http://127.0.0.1:8098

# Terminal 3: send a decision
python -c "
import json, urllib.request
body = json.dumps({'state': 'test', 'question': 'intent',
                   'candidates': ['refund', 'complaint', 'greeting']}).encode()
req = urllib.request.Request('http://127.0.0.1:8098/decide', data=body,
                             headers={'Content-Type': 'application/json'})
print(json.loads(urllib.request.urlopen(req, timeout=30).read()))
"
# Expected: 2-3 votes from different checkpoints/backends
```

## 7. Graft suite (40 tasks x 3 lineages)

```bash
python tools/graft_eval.py
# Expected: accuracy + ECE + latency per lineage
# Reference: english 78%, multilingual 79%, td 58% (round 44)
```

## 8. Cross-lineage divergence

```bash
# See docs/research/lineage-divergence.json for the reference data
# Expected: en-ml 32.5% divergence, en-td 10%, ml-td 27.5%
# 14 contrast pairs (35% split rate)
```

## 9. RLCD training

```bash
python -c "
from laya.train import TrainConfig, dry_run
config = TrainConfig(max_len=128, head_max_len=32)
result = dry_run('rollouts/train_rlcd.jsonl', 'convaiinnovations/laya', config)
print(result)
"
# Expected: rows_read > 0, valid_items > 0, skipped = {}
```

## 10. NPU verification (Intel AI PC only)

```bash
python -c "
import openvino as ov
core = ov.Core()
print('devices:', core.available_devices)
# Expected: ['CPU', 'GPU', 'NPU'] on Core Ultra
"
```

## Troubleshooting

- **Proxy issues**: set `NO_PROXY=localhost,127.0.0.1` before running
- **HF download fails**: set `HF_ENDPOINT=https://hf-mirror.com`
- **ONNX load fails**: ensure `onnxruntime` is installed
- **NPU not detected**: install Intel NPU driver (separate from GPU driver)
- **Port conflict**: use `--port` to change the default port
