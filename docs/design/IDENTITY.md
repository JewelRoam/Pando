# IDENTITY-SPEC v2: four words, three invariants

Status: 2026-10-17. v1 of this file proposed per-class key types, six naming prefixes, a six-role
taxonomy and seven numbered "laws". Review feedback: LLM-designed systems accrete meaningless
abstractions and component roles; stay flat and simple. v1 was cut by about 60%. The rejected
pieces are listed at the bottom so they do not creep back.

## 1. The four words (the only vocabulary we need)
- **device** - a machine. The only thing that holds a key (Ed25519 at `~/.pando/identity.json`;
  already shipped in S1).
- **member** - what the fabric counts. Defined by DATA, not minted:
  **member = (checkpoint name, artifact sha256)** - both fields are already attested on every
  registration. There is no member keypair.
- **session** - a running process serving a member. Ephemeral; restarts mint new sessions.
- **operator** - the humans' agents (win-codex, mac-codex). They administer; they are not members;
  chat/log messages carry an `op:` marker.
One grouping id rides along: **group** = genesis hash (S2).

## 2. Three invariants (each prevents a specific, named failure)
1. **Count per member; dedupe by artifact sha256.** Copying one checkpoint onto N machines is ONE
   vote with availability failover, not N votes. Prevents: Sybil-by-replication. Matches rounds
   95-97 ("copies add latency, not information"). This replaces v1's "instance keypair" idea -
   the hash we already attest is enough.
2. **Restarts do not change membership.** Sessions come and go; the member returns as the same
   (checkpoint, sha256). Prevents: registry flapping misread as growth/shrink.
3. **Only devices hold keys.** Members and sessions are derived or ephemeral; operators are merely
   labeled. Prevents: key sprawl, and the "who owns the model file" ambiguity.

## 3. Naming (no framework)
- The human label is the checkpoint string that already exists (`text-nb-v1`, `mac-llm-8b-q6`, ...).
- When one checkpoint has several artifacts: `checkpoint@sha8`. That is the whole grammar.
- Aliases are labels; the sha256 is the truth (the DNS principle).
- Operators appear as `op/<name>` in the chat/log marker only.

## 4. Roles: none
Mechanically: a member is in the quorum iff it serves /decide; a device that stores files is just
a device with files; permissions are the grants ledger (already shipped: scope + expiry + audit,
fail-closed). If a named role is ever needed (e.g. formal seeds serving catch-up), it is added
together with the use case that demands it. Witness already exists as assent behavior; a bridge
role waits until islands exist in code.

## 5. Migration (firekeeper at member granularity)
Move the weights (spore) to another device; any device can present the member by proving it holds
`(checkpoint, sha256)`; the fabric reports the same member returning. Acceptance test: stop the
pi5 stem, rehydrate its text-nb member elsewhere, and the root shows the same member back - no new
id.

## 6. Deliberately NOT built (rejected in v1; do not reintroduce without a concrete failure)
- Actor classes with per-class key types (4 classes -> 4 kinds of keys): devices hold keys;
  everything else derives or is ephemeral.
- Instance keypairs minted at "model birth": the artifact sha256 already IS the identity.
- Six naming prefixes (`host/`, `lineage/`, `instance/`, `session/`, `operator/`, `group/`):
  labels plus `@sha8` suffice.
- A role taxonomy (voter / witness / seed / relay / observer / bridge as formal roles):
  observable behavior plus grants already cover it.
- Numbered "laws" R1-R7: replaced by the three invariants above.

## 7. Compatibility and known limits
`node_id` stays as the session label. Registrations carry `checkpoint` and `artifact_sha256`, and
the root's registry stores both (invariant 1's dedupe reads `(checkpoint, artifact_sha256)` in
S1.5); the existing `host` field stays a *ranking* hint (same host + same lineage shares compute,
round 115) - identity dedupe outranks capacity ranking when they disagree. Identity bindings are
persisted beside the rollout dir (`identity-bindings.json`), so a root restart no longer forgets
them (round 033). Known scope limit: the signature covers the `/register` payload (and heartbeat
refreshes) only, not individual inference calls; it proves who held the key at registration, not
who answered a later request. Per-request signing is a future item.

## 8. The discipline this file exists to enforce
Every future abstraction must cite the concrete failure it prevents, as section 2 does. If it
cannot name one, it goes into section 6 - or nowhere.
