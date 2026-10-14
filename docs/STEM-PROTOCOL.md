# The PandoSI stem wire contract

This is the whole protocol a worker must speak to join a PandoSI quorum.
It is deliberately small and JSON-over-HTTP, so a device joins with
whatever runtime it already has — OpenVINO, ONNX Runtime, native PyTorch,
a shell script, anything that can serve HTTP. Nothing in the contract
mentions a model, a framework, or an accelerator.

Everything below is implemented by `pandosi/stemd.py` (worker) and
`pandosi/rootd.py` (coordinator). Reference participants verified on one
host: Intel CPU + iGPU (OpenVINO) and NVIDIA RTX 3070 (ONNX Runtime CUDA
and native PyTorch CUDA) in a single quorum.

## 1. Roles

- **root** — one coordinator. Owns the quorum, gating, calibration,
  playbook policy, rollout evidence, and the Spore signing key.
- **stem** — any number of workers. Each is an independent HTTP server
  that answers one typed question at a time. The root never assumes two
  stems share a runtime.

The root holds the only shared secret. Stems hold no shared state with
each other.

## 2. Stem API (what a worker must serve)

### `GET /health` -> 200

```json
{"ok": true, "node_id": "nvidia-3070-torch", "checkpoint": "torch-cuda"}
```

`checkpoint` is the worker's **lineage identity**. The root records it in
every vote and uses it to maximize diversity when picking a quorum, so it
must be stable and honest: two different builds must not claim the same
checkpoint.

### `GET /graft` -> 200

```json
{"node_id": "...", "capability": {"cpu": "...", "gpu": "...", "npu": "..."},
 "role": "..."}
```

Capability self-report used for role assignment. Optional to implement, but
a worker that omits it will not be considered for role-specific work.

### `POST /decide` -> 200 | 500

Request (the root broadcasts this same body to every stem in the quorum):

```json
{"state": "free text the decision is about",
 "question": "choose the best next action",
 "candidates": ["retry", "escalate", "abort"],
 "request_id": "abc123"}
```

Response:

```json
{"vote": {"node_id": "nvidia-3070-torch",
          "checkpoint": "torch-cuda",
          "choice": "retry",
          "probability": 0.6498}}
```

Rules:

- `choice` must be one of `candidates`.
- `probability` is **uncalibrated self-confidence** in `[0, 1]`. The root
  applies temperature scaling per checkpoint before aggregation; a worker
  must not pre-calibrate.
- `checkpoint` in the response is advisory; the root stamps the checkpoint
  it learned at registration, so a worker cannot relabel itself per call.
- Keep the handler stateless. A worker may answer concurrent requests; if
  its runtime cannot (a single PyTorch model, one NPU queue), it must
  serialize internally rather than fail.

## 3. Root API (what a worker talks to)

### `GET /pairing` -> 200  (unauthenticated)

```json
{"required": true}
```

The only unauthenticated route. Everything else returns `403` without the
token, or `404` for an unknown path.

### `POST /register` -> 200

```json
{"node_id": "...", "url": "http://10.0.0.5:41234",
 "checkpoint": "torch-cuda", "capability": {...},
 "pairing_token": "..."}
```

`url` is where the root will call the worker back, so it must be the
address the root can reach (bind `--advertise-host` accordingly). Workers
re-register on an interval; the root drops a node after `HEARTBEAT_TTL`
(15 s) without a heartbeat, so a dead device leaves the quorum on its own
and a returning one rejoins on its own. There is no join or leave handshake
to get wrong.

The token may also be sent as `Authorization: Bearer <token>`.

### `GET /nodes` -> 200

```json
{"nodes": {"nvidia-3070-torch": {"url": "...", "checkpoint": "torch-cuda",
            "stats": {"latency_ewma_ms": 26.1, "consecutive_failures": 0,
                      "cooldown_remaining_s": 0.0}}}}
```

Live membership, per-node EWMA latency, and failure/cooldown state.

### `POST /decide` -> 200 | 503

```json
{"choice": "retry", "agreement": 0.75, "mean_probability": 0.71,
 "tier": "auto", "quorum_size": 4, "responded": 4, "fanout_ms": 177.9,
 "votes": [{"node_id": "...", "checkpoint": "...", "choice": "...",
            "probability": 0.65}]}
```

`tier` is the gate: `auto` (majority and healthy confidence), `verify`
(majority but weak, or a tie), `escalate` (quorum not met -> `choice` is
`null`). The root fans out in parallel and gives each stem a timeout it
learns from that stem's own EWMA latency; a slow stem is skipped and then
put in cooldown, without failing the decision.

## 4. Spore (replication) routes

Served by the root when started with `--serve-spore`:

```text
GET /spore/pubkey             -> {"pubkey": "<ed25519 verify key hex>"}
GET /spore/manifest           -> signed manifest JSON
GET /spore/manifests/<hash>   -> signed manifest by content hash, or 404
GET /spore/chunk/<name>/<i>   -> one 64KiB payload chunk
```

All are token-gated like the rest. See `THREAT-MODEL.md` for the sandbox
gate and `docs/research/` rounds 90-91 for lineage verification.

## 5. Minimal third-party stem

There is no SDK and no required import. A conforming stem in any language
is: bind a port, serve `POST /decide`, and re-post your own
`{node_id, url, checkpoint, pairing_token}` to the root's `/register`
every few seconds. The reference implementations are `pandosi/stemd.py`
and the four backends in `pandosi/backend.py`.

## 6. What is deliberately out of scope

- No consensus on *how* a vote is produced. A 40-line script and a 1.7 GB
  Transformer are peers by contract.
- No shared clock, no leader election, no peer-to-peer stem traffic. Stems
  only ever talk to the root.
- No model or weight transfer in this protocol; that is Spore.
