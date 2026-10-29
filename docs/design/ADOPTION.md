# PandoRSI Adoption Design — how an open-source swarm project gets used

## Design philosophy

PyTorch won because `pip install torch` + 3 lines of code worked on day one.
PandoRSI's adoption follows the same law: **the first useful experience must
require zero privilege, zero drivers, and under five minutes.**

## The three-ring capability model

```
Ring 0  零特权    pip install pando; python -m pando.join     <- default, always works
Ring 1  建议授权  accelerator drivers, firewall exceptions, autostart
Ring 2  沙箱实验室  Spore replication experiments (Docker/podman gated)
```

### Ring 0 — the five-minute path (no root, ever)

1. `pip install pando` (pure wheels; PyNaCl is the only hard dep).
2. `python -m pando.join` — runs the capability probe, joins the LAN swarm
   via mDNS, starts serving decisions. CPU is always enough for Ring 0.
3. Open `http://localhost:8080` on any peer — see the vote stream and dashboard.

Why this is enough to be useful: even a single AI PC runs consensus across its
own ExecutionUnits (CPU/iGPU/NPU — blueprint section 2.5). The product has
value on one machine before it has value across machines.

### Ring 1 — opt-in acceleration with per-action consent

Sensitive or machine-specific operations are never bundled into install.
Instead, `pando doctor` prints a capability report:

```
$ pando doctor
  [ok]      python 3.11+, pynacl, LAN mDNS
  [ok]      EU probe: CPU, GPU.0 (Arc 140T), NPU (AI Boost)
  [miss]    openvino runtime        -> pip install openvino   (enables NPU/iGPU)
  [miss]    NPU driver              -> see https://... (vendor install, needs admin)
  [locked]  autostart service       -> `pando enable-autostart` (needs admin, ask per boot method)
```

Rules:
- **Detect, explain, then ask.** Installing a driver or opening a firewall port
  is one explicit `pando ...` command the user runs deliberately — never an
  auto-elevation buried in the installer.
- **Per-OS guidance, not per-OS implementation.** We cannot maintain silent
  installers for every distro/Android build. Each elevated action shows the
  exact command for the detected platform and asks before running it.
- Consent is recorded locally as a signed capability token (reuse Spore's
  Ed25519 manifest format) so automation can prove what was authorized.

### Ring 2 — sandbox-gated lab mode

Self-replication (Spore) experiments require the target to be disposable by
construction:

- `pando lab` refuses to start unless a container runtime (Docker/podman) or
  WSL2 distro is detected; all Spore targets must be containers or VMs.
- The join path (Ring 0) never contains Spore code — replication is a separate
  optional package (`pando-lab`), so the default install is protocol-only.
- This converts "is it safe?" into "is it sandboxed?" — a checkable property.

## Platform matrix (honest about what we cannot auto-adapt)

| Platform | Ring 0 path | Notes |
|---|---|---|
| Windows x86 (AI PC) | native | richest: OpenVINO EU enumeration, NPU |
| Linux x86 / SBC | native | syworker autostart is Ring 1 |
| macOS (Apple Silicon) | native | Metal EU; no NPU abstraction yet |
| Android TV box | Termux guide | `pkg install python` + pip; guided, not silent |

## Survival metrics (what we hold ourselves to)

- **<5 min, <3 commands** from clone to first LAN vote on any Ring 0 platform.
- Every failure mode prints the exact next command.
- `pando doctor` never says "it failed" without saying what would fix it.
- No feature ships if it breaks Ring 0 zero-privilege cleanliness.

## Relationship to prior art

- exo proved one-command LAN cluster join resonates (`exo run`); we inherit the
  pattern but add consent tiers and EU-level granularity.
- Termux is the proven Android path (community-verified onnxruntime); we ship a
  guide + `pando doctor` detection rather than an APK for v0.
- Petals' lesson inverted: public swarms die without easy private value first —
  hence Ring 0 must be valuable on a single device.
