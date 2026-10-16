# Onboarding a Raspberry Pi 5 into the fabric (consent-based)

Goal: the Pi joins a LAN fabric as a real, honest member with one visible command. The device consents
(scoped, expiring grant), the root authorizes (pairing token), and the experiment leaves evidence.
The fabric never scans or commandeers anything: enrollment is an invitation the device accepts.

## 1. Flash
Recommended image: **Raspberry Pi OS Lite (64-bit, Bookworm)** - headless, systemd, Python 3.11, apt.
- Alternatives: Ubuntu Server 24.04 LTS (arm64) - Python 3.12, familiar tooling; DietPi - ultra-light,
  good for an always-on seed. 64-bit is required (arm64 wheels; 32-bit caps per-process memory).
- Raspberry Pi Imager advanced settings: hostname `pandosi-pi5`; enable SSH (public key); user and
  password; WiFi/Ethernet; locale. Prefer Ethernet for latency-sensitive runs; WiFi is a second axis
  (network jitter) to test later.
- Storage: a 16 GB+ SD card is enough; use a USB SSD if the Pi will act as a seed.

## 2. First boot
```
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-venv git
python3 -V        # expect 3.11+ (Bookworm) or 3.12 (Ubuntu 24.04)
```

## 3. Install PandoSI (no root needed)
```
python3 -m venv ~/pando
~/pando/bin/pip install --upgrade pip
# path A: the released wheel (pure python; works on arm64)
~/pando/bin/pip install https://github.com/JewelRoam/PandoSI/releases/download/v0.2.0/pandosi-0.2.0-py3-none-any.whl
# path B: from source
#   git clone https://github.com/JewelRoam/PandoSI ~/PandoSI
#   ~/pando/bin/pip install -e ~/PandoSI
~/pando/bin/pandosi --help
```

## 4. Join the LAN fabric (the consent step)
The root side supplies: root URL (or mDNS discovery) + pairing token.
On the Pi:
```
~/pando/bin/python -m pando.grants grant lan_listen \
  --resource 0.0.0.0:8200 --scope 192.168.31.0/24 --hours 24 --yes

~/pando/bin/pandosi join \
  --root http://<WINDOWS_IP>:<ROOT_PORT> \
  --pairing-token <TOKEN> \
  --backend text \
  --checkpoint text-nb-v1 \
  --lan
```
- `--backend text` is the 19 KB naive-Bayes member shipped inside the package: no downloads, real votes,
  and it is an independent lineage (the decorrelation anchor; see docs/POWER-AND-COMPETENCE.md).
- `--lan` refuses to bind without the active `lan_listen` grant (scope + expiry enforced).
- mDNS discovery: omit `--root` and the device searches the local network for a root (zeroconf).
- Later, wrap it in `systemd --user` (Step 5 of docs/ONBOARDING.zh-CN.md) once the run is repeatable.

## 5. Verify from the root (evidence, not vibes)
- `GET /nodes` shows the Pi's node id, checkpoint, and heartbeat.
- Run one decision; confirm the Pi's vote is counted (responded count); compare against a Pi-down run.
- Then the interesting run: kill the Pi mid-stream (or unplug its network) and watch the quorum shrink and
  recover. That is the firekeeper-L2 subject (docs/SPEC-firekeeper-L2.md); the bounded-substitution patch
  (docs/DESIGN-fallback.md) is the pending prerequisite for it.

## 6. What this experiment must produce
- One copy-paste enrollment block, boot-to-joined timing, and the /nodes evidence.
- A short record (docs/ or metrics/ brief): what worked, what needed a human, which consent prompts the
  device displayed, and what the root measured.
- Standing rule: enrollment is consent-based on devices we own or have explicit permission to use.

## 7. A real neural member (Laya, int8 CPU) - measured 2026-10-17
The pi5 runs the project's native decision model (Laya ONNX int8, 571 MB) alongside the text member:
- `pip install onnxruntime`, then `pip install --no-deps laya` and `pip install transformers`
  (tokenizer only). The laya package imports torch in `laya.common` even on the ONNX path, so:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu` (aarch64 CPU wheel, ~80 MB;
  never the CUDA stack). PyPI's default index did not serve an aarch64 torch wheel - the CPU index did.
- Ship `laya-en.int8.onnx` (sha256 854abee1...) plus the small repo files
  (`rl_agent_config.json`, `tokenizer/{tokenizer.json,tokenizer_config.json}`, `encoder/config.json`)
  into `~/models/laya-repo/`, and pass `--repo ~/models/laya-repo` so nothing touches HuggingFace.
- Start:
  `python -m pando.stemd --node-id <host>-laya --backend onnx --onnx-path ~/models/laya-en.int8.onnx`
  ` --repo ~/models/laya-repo --checkpoint laya-en --root http://<root>:8790`
  ` --pairing-token-file ~/.pando/pairing.token --host 0.0.0.0 --advertise-host <ip>`
- Measured on the pi5: model load ~40 s; RAM after load ~1.2 GB used / 2.8 GB available;
  one quorum-2 decision (text + Laya on the same device, one device key signing both members):
  `responded = 2`, Laya `199 ms` with p=0.9606, text member `49 ms`; total fanout 200 ms.
- Honest flags: `laya-en` has no temperature table yet (the decision output lists it under
  `uncalibrated_checkpoints`; its votes cannot reach tier `auto`), and the attested artifact digest
  (`2df81c3e...`) differs from the raw file hash (`854abee1...`) - resolve what is being attested
  before digests are treated as provenance.

## 8. A real LLM member (Qwen3-1.7B via llama.cpp) - measured 2026-10-17
- Binaries: llama.cpp release tarball (note: `.tar.gz`, not `.zip`):
  `llama-b11700-bin-ubuntu-arm64.tar.gz` (~14 MB), extracted to `~/llama/llama-b11700/`; run with
  `LD_LIBRARY_PATH` pointing at that directory.
- Weights: ModelScope `unsloth/Qwen3-1.7B-GGUF` (community quants; the official
  `Qwen/Qwen3-1.7B-GGUF` ships Q8_0 only) - `Qwen3-1.7B-Q4_K_M.gguf`, 1.03 GiB.
- Serve (launch script in artifacts/lan-experiment/launch_llama.sh):
  `llama-server -m ~/models/Qwen3-1.7B-Q4_K_M.gguf --host 127.0.0.1 --port 8085 -c 2048 -t 4
  --jinja --chat-template-kwargs '{"enable_thinking": false}'`
  (upstream now deprecates the kwargs path in favor of `--reasoning off`). Load ~13 s from SD;
  `/v1/models` OK; bind loopback only - the stem is the network-facing part.
- Member:
  `stemd --backend openai --openai-url http://127.0.0.1:8085/v1 --openai-model qwen3-1.7b
  --openai-key-file '' --checkpoint qwen3-1.7b-q4km ...`
  An *empty string argument* disables auth; pointing at an empty *file* is rejected
  (`secret file is empty`).
- Measured: three members co-resident on the 4 GB board (text 19 KB + Laya int8 571 MB +
  Qwen3-1.7B 1.03 GB; 2.5 GB used, ~1.5 GB available); one quorum-3 decision `responded=3` in
  1.57 s; the LLM's vote arrives `neutral=true` (choice-only) - counted in agreement, excluded
  from mean_probability, exactly the neutral semantics.
