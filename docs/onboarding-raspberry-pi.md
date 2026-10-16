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
