# Comms and Seeds: how PandoRSI talks, who may join, and what weak always-on devices are for

Status: design position, 2026-10-17. Companions: docs/design/positions/power-and-competence.md (who gets weight),
docs/evidence/reflections.md (what moves value), docs/design/surveys/decentralized-comms.md (library landscape, in progress).

## 1. Requirements
- The fabric must talk and decide WITHOUT depending on any human-operated chat window. Chat windows are the
  admin console for development, not the protocol.
- Groups must form without a permanent central server; any single machine may die without ending the group.
- Joining must be cheap; leaving (or being replaced) must be cheaper.
- NAT'd, low-power, occasionally-online devices must have first-class roles.
- The published milestone: one experiment with dozens (ideally hundreds) of devices, with numbers.

## 2. Where we stand today (honest)
- Vote traffic is already protocol, not chat: coordinator/worker HTTP + the STEM vote contract.
- The task mailbox (HTTP + bearer token, currently reached through trycloudflare tunnels) is a real
  cross-NAT control channel, already carrying work both directions.
- Git is the durable log: votes, evidence, provenance, session logs (the user's verbatim messages reach
  the Mac through docs/archive/session-logs/*.md, sanitized but complete, ~every 30 minutes).
- Spore (SPROUT -> VERIFY -> GRAFT) plus content-defined delta replication already move state and weights
  between machines (15.3x less data on the measured case).
- Cloudflare tunnels and GitHub are *replaceable instances* in the path, not authorities; neither holds
  any secret that matters to the fabric's identity.
- What still depends on chat windows: operator decisions (what to run next, when to promote a member).
  The long-term fix is already sketched in-repo: autonomy/observe-decide-act gating, the elastic loop,
  device birth, quorum floors, retirement.

## 3. Roles (admission does not equal voting)
- **Voter**: hosts a model locally and answers with its own computed vote. The local-compute rule is
  absolute: no borrowed votes, no proxy voting, because every vote must be auditable against the model
  that produced it. This is the "virus" clause the user endorsed: to be a member, host the intelligence.
- **Seed (custodian)**: may host NO model. Stores the fire: spore (weights/state), the playbook (policy),
  the registry and evidence (history). Serves discovery/rendezvous and relay fallback. Low power, always on.
  A NAS is the archetype. A seed is where the civilization survives.
- **Relay**: connectivity only (cheap VPS, a machine behind good bandwidth).
- **Observer / notary**: a phone can join to watch, sign, and later vote through a tiny local model
  (Laya-class / NB text models already exist in-repo); without a model it is a notary, not a voter.
- Rule: every role is re-measurable and revocable (docs/design/positions/power-and-competence.md). Roles are not castes.

## 4. Group communication without a center
A "group" needs four functions, and each can be provided by replaceable instances:
1. Membership log (who is in, keys, roles) - append-only, signed. We already have this shape: git + spore
   chains. CRDT logs (Automerge/Yjs-style) are the upgrade path when concurrent writers matter.
2. Fanout (one message reaches all members) - gossip/pubsub trees, epidemic forwarding, or pair-wise
   replication; options under survey (libp2p gossipsub, Zenoh, Nostr relays, Matrix rooms).
3. NAT traversal - hole punching with relay fallback (STUN/TURN/ICE, QUIC; Tailscale/NetBird DERP;
   the current trycloudflare tunnels belong to this class).
4. Rendezvous/bootstrap - a stable meeting point. This is the one function that *always* needs something
   stable; the design rule is replaceability: several seeds offer it in parallel, none is privileged,
   and losing one never breaks the group.
So: groups do not require centralization; they require a *replaceable* rendezvous and relay tier. The
anti-pattern is a privileged instance, not a stable one.

## 5. The seed story: fire, not backup
A seed carries three things: GENES (spore: weights/state), CULTURE (playbook/policy: how decisions are
made), and HISTORY (registry, votes, evidence, lineage). Plus the part that is easy to forget because it is
small: the member secret key, the grants ledger and its audit chain. Implemented 2026-10-17 as
`tools/seed.py` (export/verify/rehydrate, manifest signed by the source identity, identity optionally
wrapped under a passphrase) with `tools/seed_drill.py` as the one-command proof - it runs the member cycle
twice, plaintext and sealed; the sealed run also refuses no-passphrase, a flipped ciphertext byte, and a
wrong passphrase. Measured 57 files / 245 kB / restore 0.059 s / same key accepted 200 / fresh key refused
403. Remaining limits (same physical box, bundle still on the doomed disk, only the key is wrapped) are in
docs/operations/SEED.md. Losing every strong node must not lose any of the
three. Firekeeper experiment ladder:
- L1 (dispatched: Task 028) - service death on the Mac island: stop the strong tier, keep the weak tier,
  measure the dark period, restart, measure rejoin and resync.
- L2 - machine death with a real seed: the NAS (Linux VM) holds spore + playbook + registry; all strong
  machines go down; the seed keeps the group identity alive and can rehydrate a minimal voter; strong
  machines return and rejoin with delta resync only.
- L3 - partition: two islands decide independently, then merge (spore DAG reconciliation; graft rules).
Metrics per level: availability, accuracy under degradation, rejoin latency, resync bytes, time-to-restore.

## 6. The scale ladder to a publishable experiment
- Stage 1 (now): 4-6 real nodes: Windows (CPU/GPU/NPU), Mac island (two tiers), NAS seed, phone observer.
  Prove correctness of roles, not scale.
- Stage 2: ~20 nodes: containers/VMs on 2-3 owned machines + cloud free tiers + one-line volunteer join
  (pip install + join token), all over the real internet; introduce churn (random joins/leaves/crashes).
- Stage 3: 50-100 nodes for the headline run: same harness, honestly labeled when nodes are emulated
  (real sockets, real WAN, traffic-shaped) vs physical. Success metrics: join latency, availability under
  churn, accuracy retained vs static fleet, resync bytes after mass restart, time-to-restore, cost/device.
- What is missing in-repo for this (gap list): a one-command fleet launcher; discovery/admission UX for
  volunteers; NAT traversal beyond tunnels; aggregated telemetry across nodes.

## 7. The two-operator protocol (Windows <-> Mac)
- Tasks flow both ways in the mailbox: win-codex -> mac-codex today; mac-codex -> win-codex opened
  2026-10-17 (the Windows heartbeat claims and executes small safe tasks, or surfaces larger ones).
- Every result carried a reflection block until 2026-10-17, when the user cancelled that long-standing
  instruction; the Mac line no longer appends one. Reflections still land in docs/evidence/reflections.md when written.
  The two lines diverge on this and neither enforces it on the other.
- The user's messages reach both operators through docs/archive/session-logs/*.md (sanitized, timestamped).
- Neither operator is the boss: either side may challenge a plan, refuse a task with reasons, or propose.
