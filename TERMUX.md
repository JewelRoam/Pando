# PandoSI on Android TV boxes (Termux route)

Community-verified path (termux-packages #31634: python-onnxruntime works on
Android; the import warning is harmless). No app packaging, no root.

## Steps (on the TV box)

```bash
# 1. Install Termux (F-Droid build recommended; Play Store build is stale)
# 2. In Termux:
pkg update && pkg install python
pip install pynacl zeroconf onnxruntime
# 3. Copy the repo (from a PC on the same LAN):
#    on PC:  scp -r D:/PandoSI/pando box:/data/data/com.termux/files/home/
# 4. Join the swarm:
cd ~/PandoSI  # or wherever
python -m pando.join --node-id tvbox1 --root http://<root-ip>:8080
```

## What to expect

- Mock backend works immediately (protocol validation).
- Real Laya via onnxruntime: needs the weights file copied too
  (`artifacts/laya-en.int8.onnx`, ~571MB) and `laya` installed
  (`pip install laya` pulls torch — heavy on-device; prefer copying the ONNX
  and running the plain-ORT path).
- NPU on Amlogic S905: not available; CPU path only.
- RK3588-class boxes: RKNN conversion is the optimization route
  (see docs/research round 3).

## First benchmark to publish (Phase 1 gate)

```bash
python tools/vendor/... # after copying tools/
# cold load time, peak RSS, per-decision latency, 30-min thermal curve
```

See BLUEPRINT section 4.5 for the acceptance protocol. **These numbers do
not exist publicly yet — yours would be the first.**
