# The PandoSI stem wire contract

This is the whole protocol a worker must speak to join a PandoSI quorum.
It is deliberately small and JSON-over-HTTP, so a device joins with
whatever runtime it already has — OpenVINO, ONNX Runtime, native PyTorch,
a shell script, anything that can serve HTTP. Nothing in the contract
mentions a model, a framework, or an accelerator.

Everything below is implemented by `pando/worker.py` (worker) and
`pando/coordinator.py` (coordinator). Every field and semantic is cited as
`file:line` against the commit this document was written from; when the
two disagree, the code wins and this document is the bug.

Reference participants verified on one host: Intel CPU + iGPU (OpenVINO)
and NVIDIA RTX 3070 (ONNX Runtime CUDA and native PyTorch CUDA) in a
single quorum.

## 1. Roles

- **root** — one coordinator. Owns the quorum, gating, calibration,
  playbook policy, rollout evidence, and the Spore signing key
  (`pando/coordinator.py:170-256`).
- **stem** — any number of workers. Each is an independent HTTP server
  that answers one typed question at a time. The root never assumes two
  stems share a runtime (`pando/worker.py:101-126`).

The root holds the only shared secret. Stems hold no shared state with
each other.

## 2. Stem API (what a worker must serve)

All three routes are served unauthenticated on the stem's own port; the
root is the only caller (`pando/worker.py:48-97`).

### `GET /health` -> 200

```json
{"ok": true,
 "node_id": "mac-llm-8b",
 "checkpoint": "mac-llm-8b",
 "decisions": 240,
 "host_ms_per_decision": 175.4}
```

| Field | Type | Meaning | Source |
| --- | --- | --- | --- |
| `ok` | bool | Always `true` on a live stem. | `pando/worker.py:60` |
| `node_id` | str | Stable id this stem registered under. | `pando/worker.py:60` |
| `checkpoint` | str | **Lineage identity** — the string the root records in every vote and uses for quorum diversity. Two different builds must not claim the same checkpoint. | `pando/worker.py:61` |
| `decisions` | int | Decisions this process has actually served (round 156: stem-side is the only place "who served" is answerable under a lineage-counted quorum). | `pando/worker.py:67-68` |
| `host_ms_per_decision` | float\|null | Host CPU the whole process spent per served decision, `1000 * (process_time - started) / decisions`, `null` before any decision. | `pando/worker.py:31-36`, `:69-70` |

### `GET /graft` -> 200

```json
{"node_id": "mac-llm-8b",
 "capability": {"arch": "arm64", "platform": "macOS-...",
                "python": "3.11.9", "cpu_count": 10, "ram_gb": 16.0},
 "role": "worker"}
```

| Field | Type | Meaning | Source |
| --- | --- | --- | --- |
| `node_id` | str | Same id as `/health`. | `pando/worker.py:73` |
| `capability` | object | Best-effort probe: `arch`, `platform`, `python`, `cpu_count`, `ram_gb`. Not a GPU/NPU claim. | `pando/graft.py:38-47`, `worker.py:72-74` |
| `role` | str | `worker` if `ram_gb >= 3.5` and `cpu_count >= 2`, else `observer`. | `pando/graft.py:50-56`, `worker.py:73-74` |

Optional to implement, but a worker that omits it will not be considered
for role-specific work.

### `POST /decide` -> 200 | 500

Request (the root broadcasts this same body to every stem in the quorum):

```json
{"state": "free text the decision is about",
 "question": "choose the best next action",
 "candidates": ["retry", "escalate", "abort"],
 "request_id": "abc123",
 "family": "known"}
```

| Field | Type | Required | Meaning | Source |
| --- | --- | --- | --- | --- |
| `state` | str | yes | Free text the decision is about. | `pando/common.py:30` |
| `question` | str | yes | The typed question name. | `pando/common.py:31` |
| `candidates` | list[str] | no | The option set; `choice` must be one of these. Defaults to `[]`. | `pando/common.py:32` |
| `request_id` | str | no | Correlation id; generated `uuid4().hex[:12]` when absent. | `pando/common.py:33` |
| `family` | str\|null | no | Evidence family for assent resolution. Defaults to `null` (root falls back to the playbook category, then `known`). | `pando/common.py:34`, `:46` |

Response:

```json
{"vote": {"node_id": "mac-llm-8b",
          "checkpoint": "mac-llm-8b",
          "choice": "retry",
          "probability": 0.6498,
          "neutral": false}}
```

| Field | Type | Meaning | Source |
| --- | --- | --- | --- |
| `node_id` | str | Voter id. | `pando/common.py:63` |
| `checkpoint` | str | Lineage the vote came from. In the response it is advisory; the root stamps the checkpoint it learned at registration. | `pando/common.py:64` |
| `choice` | str | One of the request's candidates. | `pando/common.py:65` |
| `probability` | float | **Uncalibrated self-confidence** in `[0, 1]`. The root temperature-scales it per checkpoint before aggregation; a worker must not pre-calibrate. | `pando/common.py:66`, `:80-81` |
| `neutral` | bool | When `true` the vote is **choice-only**: its choice still counts toward agreement and quorum, but its probability is excluded from the confidence mean and from calibration. Defaults `false`. | `pando/common.py:66-67`, `:114-128` |

Rules:

- `choice` must be one of `candidates`.
- The handler is stateless per call. A worker **serializes by default**
  (one decision at a time, `--serialize` is the default and is now a
  no-op flag); `--parallel` opts into concurrency only for a backend
  proven thread-safe. Round 135 measured the OpenVINO path failing about
  half its calls at four concurrent clients while the ONNX CPU path
  never failed (`pando/worker.py:102-126`).
- A backend exception becomes HTTP 500 with `{"error": ...}`; the root
  treats that as an unanswered stem, not a vote (`pando/worker.py:90-97`).

### Heartbeat / registration payload (stem -> root)

Every 5 s the stem re-posts this to the root's `/register`
(`pando/worker.py:144-181`):

```json
{"node_id": "mac-llm-8b",
 "url": "http://10.0.0.5:41234",
 "checkpoint": "mac-llm-8b",
 "capability": {"arch": "arm64", "...": "...",
                "host_ms_per_decision": 175.4},
 "artifact_sha256": "…",
 "pairing_token": "…"}
```

| Field | Type | Required | Meaning | Source |
| --- | --- | --- | --- | --- |
| `node_id` | str | yes | Voter id. | `pando/worker.py:146-147` |
| `url` | str | yes | Where the root calls the stem back. Must be reachable from the root. | `pando/worker.py:146-147`, `:117` |
| `checkpoint` | str | yes | Lineage identity. | `pando/worker.py:148` |
| `capability` | object | no | Base probe plus `host_ms_per_decision`, refreshed every beat so a root can prefer a cheap member (round 134). | `pando/worker.py:149`, `:165-172` |
| `artifact_sha256` | str | no | Present only when the loaded artifact has a digest; lets a root that declares an expected artifact refuse a mislabeled stem (round 102). | `pando/worker.py:150-156` |
| `pairing_token` | str | no | Admission secret when the root has one. Also mirrored into `Authorization: Bearer`. | `pando/worker.py:157-160` |

The token may also be sent as `Authorization: Bearer <token>`.

## 3. Root API (what a worker talks to)

### `GET /pairing` -> 200  (unauthenticated)

```json
{"required": true}
```

The only unauthenticated route. Everything else returns `403` without the
token, or `404` for an unknown path
(`pando/coordinator.py:888-893`).

### `POST /register` -> 200 | 400 | 403

```json
{"node_id": "nvidia-3070-torch", "url": "http://10.0.0.5:41234",
 "checkpoint": "torch-cuda", "capability": {...},
 "artifact_sha256": "…", "pairing_token": "…"}
```

Admission order (`pando/coordinator.py:993-1027`):

1. `pairing_token` is compared with `hmac.compare_digest`; mismatch -> 403.
2. `checkpoint` must be in the playbook's admitted lineages when the
   playbook declares any; otherwise 403 with the admitted set.
3. `artifact_sha256` must equal the declared digest for that lineage when
   the playbook pins one; otherwise 403 with expected vs reported.
4. A node id previously retired by the swarm is refused (403) — deleting
   it from the registry alone would only last until the next heartbeat.

`url` is what the root calls the worker back, so it must be the address
the root can reach (bind/advertise accordingly). Workers re-register on
an interval; the root drops a node after `HEARTBEAT_TTL` (15 s) without a
heartbeat, so a dead device leaves the quorum on its own and a returning
one rejoins on its own. There is no join or leave handshake to get wrong
(`pando/coordinator.py:297-310`, `HEARTBEAT_TTL` at `:35`).

### `GET /nodes` -> 200

```json
{"nodes": {"nvidia-3070-torch": {
    "url": "http://10.0.0.5:41234",
    "checkpoint": "torch-cuda",
    "capability": {"arch": "x86_64", "host_ms_per_decision": 26.1},
    "stats": {"successes": 12, "failures": 0,
              "last_latency_ms": 25.4, "latency_ewma_ms": 26.1,
              "last_error": null, "consecutive_failures": 0,
              "unhealthy_until": 0.0, "cooldown_remaining_s": 0.0}}}}
```

Live membership plus per-node call statistics. `stats` starts from
`_empty_stem_stats()` (`pando/coordinator.py:151-161`) and gains
`cooldown_remaining_s` on read (`pando/coordinator.py:302-306`). Only nodes
seen within `HEARTBEAT_TTL` appear.

### `GET /status` -> 200

One read for an elastic supervisor: how much checking is left to spend
and how slow decisions actually are (`pando/coordinator.py:898-915`).

```json
{"members": 4, "blocked": ["retired-node"],
 "quorum_k": 3, "quorum_floor": 1, "quorum_ceiling": 3,
 "adaptive": false, "latency_target_ms": null,
 "decision_ewma_ms": 231.7,
 "question_categories": {"ordinary": "known"},
 "assent_checkpoint": "auto", "assent_fallback": "advisor",
 "assent_resolved": "mac-llm-8b"}
```

| Field | Meaning | Source |
| --- | --- | --- |
| `members` | Live node count. | `pando/coordinator.py:899` |
| `blocked` | Sorted retired node ids. | `pando/coordinator.py:900` |
| `quorum_k`, `quorum_floor`, `quorum_ceiling` | Configured quorum bounds. | `pando/coordinator.py:901-903` |
| `adaptive` | Whether adaptive quorum is on. | `pando/coordinator.py:904` |
| `latency_target_ms` | Adaptive target, or `null`. | `pando/coordinator.py:905` |
| `decision_ewma_ms` | End-to-end decision EWMA, or `null` before the first decision. | `pando/coordinator.py:906` |
| `question_categories` | Playbook question -> category map. | `pando/coordinator.py:907` |
| `assent_checkpoint`, `assent_fallback` | Configured assent policy. | `pando/coordinator.py:908-909` |
| `assent_resolved` | Last checkpoint auto-resolution chose, or `null`. Not a per-request value. | `pando/coordinator.py:910`, `:761` |

### `POST /decide` -> 200 | 400 | 403 | 503

The request body is the `DecisionRequest` above. The result starts as
`DecisionResult.__dict__` (`choice`, `agreement`, `mean_probability`,
`tier`, `votes`) and gains the fields the coordinator actually computed
(`pando/coordinator.py:828-860`).

```json
{"choice": "retry", "agreement": 0.75, "mean_probability": 0.71,
 "tier": "auto", "quorum_size": 4, "responded": 4,
 "fanout_ms": 177.9, "votes": [
   {"node_id": "a", "checkpoint": "english", "choice": "retry",
    "probability": 0.65, "neutral": false},
   {"node_id": "b", "checkpoint": "mac-llm-8b", "choice": "retry",
    "probability": 0.54, "neutral": false},
   {"node_id": "c", "checkpoint": "text-nb-v1", "choice": "retry",
    "probability": 0.5, "neutral": true},
   {"node_id": "d", "checkpoint": "ft-v3", "choice": "abort",
    "probability": 0.61, "neutral": false}]}
```

| Field | Type | Present when | Meaning | Source |
| --- | --- | --- | --- | --- |
| `choice` | str\|null | always | Aggregated choice, `null` on a true tie or sub-quorum. | `pando/common.py:76` |
| `agreement` | float | always | Count share of the top choice, **including neutral votes**. | `pando/common.py:116-123` |
| `mean_probability` | float | always | Mean probability of the **backing non-neutral** votes, `0.0` if a choice is backed only by neutral votes. | `pando/common.py:122-128` |
| `tier` | str | always | `auto` (majority + healthy confidence), `verify` (majority but weak, tie, or demoted), `escalate` (quorum not met -> `choice` is `null`). | `pando/common.py:89-92`, `:155-161` |
| `votes` | list | always | The votes aggregation saw — the collapsed set under lineage/bloc counting. | `pando/common.py:76`, `:180-200` |
| `quorum_size` | int | always | Stems actually selected, not stems that answered. | `pando/coordinator.py:809` |
| `veto` | str | only when a veto fired | The checkpoint whose choice replaced the answer. | `pando/coordinator.py:810-811`, `:766-798` |
| `lineages` | int | only when collapsing or flooring | Distinct checkpoints among selected stems. | `pando/coordinator.py:814-818`, `:731` |
| `opinions` | int | only when collapsing or flooring | Distinct opinions (blocs when declared). | `pando/coordinator.py:814-818`, `:732-733` |
| `blocs` | int | only when blocs declared | Opinion count under bloc mapping. | `pando/coordinator.py:819-820` |
| `corroboration_floor` | int | only when `> 1` | Policy floor that can demote a lone opinion from `auto`. | `pando/coordinator.py:821-822` |
| `responded` | int | always | Votes actually received. | `pando/coordinator.py:823` |
| `unanswered` | list[str] | only when non-empty | Selected nodes that returned nothing (timeout/error), so an escalate is readable. | `pando/coordinator.py:824-827` |
| `suggested_choice` | str | combiner + non-auto + votes | Fitted-combiner advice where the gate declined; never changes the tier or an auto choice. | `pando/coordinator.py:828-849` |
| `suggested_probability` | float | same | The combiner's confidence. | `pando/coordinator.py:849` |
| `assent_family` | str\|null | only when `--assent-checkpoint auto` | Evidence family used to resolve the witness. | `pando/coordinator.py:850-853` |
| `assent` | str | only when a witness resolved | The resolved witness checkpoint. | `pando/coordinator.py:852-853` |
| `fanout_ms` | float | always | Wall time of the parallel fan-out to the selected stems. | `pando/coordinator.py:854` |
| `uncalibrated_checkpoints` | list[str] | only when non-empty | Voters whose probability had no calibration entry; reported, not silently acted on. | `pando/coordinator.py:855-859` |

An unknown question against a loaded playbook returns early without
touching the quorum: `{"error": ..., "responded": 0, "quorum_size": 0}`
(`pando/coordinator.py:658-663`).

The root fans out in parallel and gives each stem a timeout learned from
that stem's own EWMA latency; a slow stem is skipped, put in cooldown, and
named in `unanswered`, without failing the decision
(`pando/coordinator.py:629-656`, `:658-697`).

## 4. Semantics: assent, veto, advisor, neutral

### Neutral votes

A `neutral` vote carries a choice but no confidence
(`pando/common.py:54-67`). Consequences in `aggregate`
(`pando/common.py:82-161`):

- It **counts** toward the choice tally and therefore toward `agreement`
  and toward meeting quorum (`common.py:116-121`).
- It is **excluded** from the probability mean of its choice
  (`common.py:122-128`).
- A choice backed only by neutral votes gets `mean_probability = 0.0` and
  can therefore never reach `auto` — the gate refuses to invent a
  confidence (`common.py:122-128`, comment `:124-126`).
- On a tie, neutral votes are ignored when ranking by probability mass;
  if the non-neutral mass also ties the decision stays `choice = null` at
  `verify` (`common.py:139-153`).
- Under lineage/bloc collapsing, a group whose members are all neutral
  collapses to one `neutral=True` vote at `p = 0.5`
  (`pando/common.py:186-189`).

The OpenAI-compatible backend declares itself neutral: the upstream chat
API returns a choice, not a calibrated probability, so the wire
probability is a constant and the vote is marked `neutral=True`
(`pando/worker.py:268-280`, `pando/backend.py:278`, `:365`; default `--openai-probability 0.5`).

### Assent, advisor, veto

Assent is an opt-in second opinion (`--assent-checkpoint`). A named
checkpoint — or, with `auto`, the present checkpoint with the highest mean
measured accuracy in `assets/registry/accuracies.json` — is the **witness**
(`pando/coordinator.py:375-395`, `:751-761`).

- If the vote is `auto` and the witness disagrees, the result is demoted
  to `verify` (`pando/coordinator.py:776-782`).
- `--assent-fallback advisor` (default) keeps the aggregated choice.
- `--assent-fallback veto` returns the witness's choice instead, and names
  the witness in the `veto` field (`pando/coordinator.py:777-782`).
- Round 165 extended veto to answer **every non-auto row**, not only the
  rows where the witness disagreed with an auto verdict. When a witness is
  resolved and present, its choice becomes the answer on any non-auto
  tier; `agreement`, `mean_probability` and `tier` are left untouched, so
  nothing is fabricated and the witness is still named in `veto` for audit
  (`pando/coordinator.py:783-798`, comment `:784-793`).
- Family resolution: an explicit `family` on the request wins; otherwise a
  playbook category is honoured only when it names a family actually
  present in the registry; otherwise `known`
  (`pando/coordinator.py:341-364`, default `:56`).

### Round 165 / 166 identities (project-recorded measurements)

These are measured results recorded in `docs/evidence/FINDINGS.md`, not wire fields.
Cited so the numbers are not confused with the contract above.

- **Round 165** — excluding choice-only (`p=0.5`) votes from the mean:
  auto coverage D +2 / E +2 / F +5, precision flat; neutral votes shipped
  as `Vote.neutral` plus the count/mean split, all-neutral can never auto.
  Neutral live on F: auto 94 -> 99 @ 0.9899, decided 0.9833 unchanged
  (`docs/design/ARCHITECTURE.zh-CN.md`).
- **Round 166** — under assent+veto, decided == the witness's own accuracy:
  every row resolves to the witness's choice. `recommend_fallback` shipped
  as an offline advisor-vs-veto recommender (`docs/design/ARCHITECTURE.zh-CN.md`).

## 5. Data formats

### `assets/registry/accuracies.json` — schema v2

Per-checkpoint, per-family measured accuracy. The resolver averages the
splits inside the family it is asked for; no value is inferred
(`pando/coordinator.py:71-140`).

```json
{
  "schema_version": 2,
  "provenance": "Each accuracy value is the measured member accuracy …",
  "members": {
    "english": {
      "updated": "2026-10-16",
      "families": {
        "known": {
          "accuracy": {"D": 0.7292, "E": 0.8375},
          "source": {"D": "votes-d-seated-english.json",
                     "E": "metrics/lan-check/e-votes-english.json"}
        },
        "general": {
          "accuracy": {"F": 0.8417},
          "source": {"F": "metrics/lan-check/f-votes-english.json"}
        }
      }
    }
  }
}
```

Shape: `checkpoint -> family -> split -> accuracy`
(`pando/coordinator.py:74-80`). Only finite numeric accuracies survive the
parse; booleans are rejected (`pando/coordinator.py:59-68`). A v1 record (or a
compact `{"english": {"D": 0.7292}}` mapping) with no `families` key is
migrated **in memory** into the `known` family so an old registry stays
readable (`pando/coordinator.py:117-133`). Missing, malformed, and empty
registries all become an empty mapping rather than a guess
(`pando/coordinator.py:104-107`). The default path is
`assets/registry/accuracies.json`, overridable with the
`PANDOSI_ASSENT_REGISTRY` environment variable
(`pando/coordinator.py:52-53`, `:90-98`).

Current members: `english`, `ft-v3`, `mac-llm`, `mac-llm-8b`, each with
`known` and `general` families (`assets/registry/accuracies.json`).

### `assets/calibration/<checkpoint>.json`

Temperature scaling data, loaded per checkpoint string
(`pando/calibrate.py:22-66`).

```json
{
  "checkpoint": "mac-llm",
  "temperature": 4.09,
  "source": "fit_temperature on mac-llm votes, split C (Task 003 via the Mac)",
  "n_tasks": 240,
  "accuracy": 0.6542,
  "raw_ece": 0.3033,
  "calibrated_ece": 0.0336
}
```

Required: `temperature`, which must be a finite float in `[0.1, 10.0]`;
outside that band the file is ignored and the checkpoint is uncalibrated
(`pando/calibrate.py:54-60`). All other keys are documentation. The
scaling is `sigmoid(logit(p) / T)` with `p` clamped to `[1e-7, 1-1e-7]`;
`T > 1` softens, `T < 1` sharpens (`pando/calibrate.py:69-80`). Without a
usable entry the calibrator returns identity `T = 1.0` **and** flags the
checkpoint uncalibrated, which the root surfaces as
`uncalibrated_checkpoints` (`pando/calibrate.py:62-66`, `:83-96`).

Lookup order: a checkpoint directory's `rl_agent_config.json` /
`config.json` (`temperature` or `temperatures`), then
`assets/calibration/<checkpoint>.json`, then a `*.json` whose
`checkpoint` field matches by name (`pando/calibrate.py:26-60`).

## 6. CLI

### `coordinator` key flags (`pando/coordinator.py:1174-1243`)

| Flag | Default | Purpose | Line |
| --- | --- | --- | --- |
| `--port` | `8080` | Listen port. | `1176` |
| `--serve-spore` | `None` | Payload file to serve for Spore pulls. | `1177-1178` |
| `--chunking {fixed,cdc}` | `fixed` | v1 fixed manifests vs content-defined-chunking delta pulls. | `1179-1182` |
| `--playbook` | `None` | Playbook TOML to load and serve. | `1183-1184` |
| `--playbook-key` | `None` | Pinned Ed25519 pubkey (hex); with it an unsigned or invalid playbook is refused. | `1185-1187` |
| `--playbook-signature` | `None` | Detached signature path, default `<playbook>.sig`. | `1188-1190` |
| `--history` | `None` | Playbook history chain to verify against. | `1191-1192` |
| `--allow-non-head` | off | Accept a playbook that is not the history head. | `1193` |
| `--auth-token` | `None` | Enable `/shutdown` with this token. | `1194-1195` |
| `--pairing-token` / `--pairing-token-file` | generated | Fixed stem admission token, or read it from a user-owned file. Mutually exclusive. | `1196-1200` |
| `--rollout-dir` | `rollouts` | Directory for local rollout evidence. | `1201-1202` |
| `--stem-call-timeout` | `CALL_TIMEOUT` | Maximum per-stem call timeout. | `1203-1204` |
| `--quorum-k` | `QUORUM_K` | Voters per decision; keep fixed while adding replicas. | `1205-1207` |
| `--adaptive-quorum` | off | Let quorum size fall under latency pressure and recover when idle (requires `--latency-target-ms`). | `1208-1211` |
| `--quorum-floor` | `1` | Fewest voters the adaptive policy may use. | `1212-1213` |
| `--latency-target-ms` | `None` | Decision latency the adaptive quorum aims for. | `1214-1215` |
| `--lineage-quorum` | off | Count the quorum over lineages, not stems: replicated members collapse to one vote (round 119). | `1216-1218` |
| `--combiner` | `None` | Fitted combiner artifact; its answer is attached as `suggested_choice` wherever the gate does not auto-execute (round 140). | `1219-1222` |
| `--assent-checkpoint` | `None` | Named witness checkpoint, or `auto` for highest mean measured accuracy in the registry. | `1223-1226` |
| `--assent-fallback {advisor,veto}` | `advisor` | What to return after an assent demotion: `advisor` keeps the aggregate choice, `veto` returns the witness's. | `1227-1230` |
| `--initial-stem-timeout` | `INITIAL_STEM_TIMEOUT` | Discovery timeout before latency is learned. | `1231-1233` |
| `--lan` | off | Bind `0.0.0.0`; requires an active `lan_listen` grant. | `1234-1235` |
| `--reveal-pairing-token` | off | Print the full token even when stdout is not a TTY. | `1236-1238` |

### `worker` key flags (`pando/worker.py:184-234`)

| Flag | Default | Purpose | Line |
| --- | --- | --- | --- |
| `--node-id` | required | Stable node id. | `186` |
| `--checkpoint` | backend-derived | Lineage identity; defaults per backend. | `187-188` |
| `--port` | `0` | Listen port; `0` picks an ephemeral one. | `189` |
| `--host` | `127.0.0.1` | Bind interface; `0.0.0.0` for LAN. | `190-191` |
| `--advertise-host` | `--host` | Address reported to the root. | `192-193` |
| `--root` | `None` | coordinator URL to register with; no registration when omitted. | `194` |
| `--pairing-token` / `--pairing-token-file` | `None` | Admission secret, or read it from a user-owned file. Mutually exclusive. | `195-198` |
| `--backend {mock,onnx,laya,text,openai}` | `mock` | Decision backend. | `199-203` |
| `--onnx-path` | `artifacts/laya-en.int8.onnx` | ONNX/OpenVINO model path. | `204` |
| `--repo` | `convaiinnovations/laya` | HF repo for the ONNX path. | `205` |
| `--model` | `convaiinnovations/laya` | HF id or local path for the torch (`laya`) backend. | `206-207` |
| `--text-model` | `assets/textmodel.json` | Fitted `TextModel` for `--backend text`. | `208-209` |
| `--openai-url` | `http://127.0.0.1:8080/v1` | Base URL for `--backend openai` (OpenAI-compatible `/chat/completions`). | `210-212` |
| `--openai-key-file` | `~/.pando/llm-api.token` | File holding the API key; empty string disables auth. | `213-215` |
| `--openai-model` | `qwen3-4b-instruct-2507` | Model name to request. | `216-217` |
| `--openai-probability` | `0.5` | Wire probability for the OpenAI backend; the vote is marked neutral and excluded from confidence aggregation. | `218-221` |
| `--device` | `CPU` | Execution device for the ONNX/torch backends. | `222` |
| `--delay-ms` | `0.0` | Mock backend only: simulated per-decision device speed. | `223-225` |
| `--serialize` | off (no-op) | Deprecated: stems serialize by default now. | `226-227` |
| `--parallel` | off | Serve decisions concurrently; only for a backend proven thread-safe (round 135). | `228-230` |
| `--pad-to` | `256` on GPU/NPU, else unset | Static-shape bucket for non-CPU execution units. | `231-233` |

## 7. Minimal end-to-end walkthrough (real HTTP, generated + run)

`tools/lineage_quorum_e2e_check.py` starts real `Root` and `Stem` objects
on loopback, registers HTTP stems, and issues real `/decide` calls
(`tools/lineage_quorum_e2e_check.py:1-142`). Run it:

```console
$ PYTHONPATH=. .venv/bin/python tools/lineage_quorum_e2e_check.py
paired scenarios over real HTTP:
  stems, flag off (2:1 split)        tier=auto     choice=approve  agreement=0.67 stems=3 lineages=-
  stems, flag on (same quorum)       tier=verify   choice=approve  agreement=0.50 stems=3 lineages=2
  replicas, flag on (3x english)     tier=auto     choice=approve  agreement=1.00 stems=3 lineages=1
  replicas, flag on, floor 2         tier=verify   choice=approve  agreement=1.00 stems=3 lineages=1
  english + en-copy, declared one bloc tier=verify   choice=approve  agreement=0.50 stems=3 lineages=3

one lineage still decides by default (tier=auto with lineages=1), because a single device has to work; a playbook that wants corroboration sets corroboration_floor and the same quorum lands on verify
LINEAGE QUORUM E2E OK
```

That output is what the flags above actually do over the wire, not a
description: the same three stems flip from `auto` to `verify` purely
because the quorum is counted over opinions instead of names, and the
`lineages`/`blocs` fields appear only on the roots that asked for them.
The run also prints one expected line to stderr, not stdout — a synthetic
checkpoint with no temperature table is refused auto tier, which is the
fail-closed calibration policy from section 5 doing its job:

```text
[calibration] checkpoint 'en-copy' has no temperature table; its votes cannot reach tier 'auto'
```

The full smoke suite covers the wire contract, gating and assent logic.
Wall time tracks machine load; the pass/skip counts do not:

```console
$ PYTHONPATH=. .venv/bin/python -m unittest tests.test_smoke
Ran 29 tests in 28.784s

OK (skipped=1)
```

Both CLI surfaces above are also checkable without starting a node. The
flag lists in section 6 are exactly what these print:

```console
$ PYTHONPATH=. .venv/bin/python -m pando.coordinator --help
usage: python -m pando.coordinator [-h] [--port PORT] [--serve-spore SERVE_SPORE]
                               [--chunking {fixed,cdc}] [--playbook PLAYBOOK]
                               ...
$ PYTHONPATH=. .venv/bin/python -m pando.worker --help
usage: python -m pando.worker [-h] --node-id NODE_ID
                               [--checkpoint CHECKPOINT] [--port PORT]
                               ...
```

## 8. What is deliberately out of scope

- No consensus on *how* a vote is produced. A 40-line script and a 1.7 GB
  Transformer are peers by contract.
- No shared clock, no leader election, no peer-to-peer stem traffic. Stems
  only ever talk to the root.
- No model or weight transfer in this protocol; that is Spore.

## 9. Spore (replication) routes

Served by the root when started with `--serve-spore`
(`pando/coordinator.py:932-985`):

```text
GET /spore/pubkey             -> {"pubkey": "<ed25519 verify key hex>"}
GET /spore/manifest           -> signed manifest JSON
GET /spore/manifests/<hash>   -> signed manifest by content hash, or 404
GET /spore/chunk/<name>/<i>   -> one 64KiB payload chunk
GET /spore/chunks/<name>/<i>/<n> -> n chunk bytes where all are present
```

All are token-gated like the rest. See `pando/lab.py` for the sandbox
gate and `docs/archive/research/` rounds 90-91 for lineage verification.

## 10. Minimal third-party stem

There is no SDK and no required import. A conforming stem in any language
is: bind a port, serve `POST /decide`, and re-post your own
`{node_id, url, checkpoint, pairing_token}` to the root's `/register`
every few seconds. The reference implementations are `pando/worker.py`
and the backends in `pando/backend.py`.
