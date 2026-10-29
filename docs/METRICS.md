# PandoSI Metrics — dashboard design (v0.1)

The dashboard is the project's public face: every claim in the README should be
reproducible from a dashboard screenshot. Metrics are emitted by coordinator/worker as
JSON lines to `metrics/` and rendered by a single-file static viewer (no server
dependency beyond reading the directory).

## Core metrics

### 1. Fleet (Stem inventory)
- device count over time (join/leave events on one timeline)
- heterogeneity coverage: distinct (arch, ram_bucket, runtime) tuples seen
- role distribution: worker / observer
- per-device: checkpoint, decision latency p50/p95, ECE-at-graft, thermal flag

### 2. Decisions (Root throughput)
- decisions/s (1-minute windows), fan-out width k, responded/quorum ratio
- tier distribution: auto / verify / escalate
- consensus: agreement histogram, mean probability of winning choice
- disagreement log: request_id, vote matrix (the "playbook blind-spot" feed)

### 3. Replication (Spore, sandbox only)
- replication events on a timeline: SCAN -> GRAFT -> SPROUT -> VERIFY, with
  per-stage duration; success rate per device family
- payload integrity: chunks transferred, chunks re-fetched (Merkle rejections)
- doubling-time estimate (the Anthropic-parlor-metric, reported honestly with
  its sandbox caveats)

### 4. Lineage (evolution, Phase 5)
- lineage graph: parent checkpoint -> fine-tuned child, per-generation
  consensus accuracy on the fixed graft suite
- A/B ledger: lineage A vs B agreement on held-out states

## Non-goals
- No remote telemetry. Metrics stay on the operator's disk. The dashboard is
  a local file renderer, not a service.
