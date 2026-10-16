# Plan: groups — how the fabric forms and keeps a shared conversation

Status: implementation plan, 2026-10-17. Inputs: docs/SURVEY-decentralized-comms.md (stack),
the mac-codex consultation (task t-20261017-001001-f790), docs/COMMS-AND-SEEDS.md (functions).

## The model (decided)
A group is not a mesh. A group is:
1. a **signed append-only log** whose identity is the hash of its genesis event;
2. **mirrored on any number of seeds** (relay-with-replicated-state);
3. members sign events; any seed serves catch-up; a seed can delay or censor delivery but never
   forge or rewrite anything.

"Forming a group" = minting a genesis + an invite token. "Joining" = fetching catch-up from any
mirror and appending a signed join. P2P direct-to-everyone is explicitly NOT required (O(N^2)
links and pairwise trust); the requirement is *replaceable* stable mirrors, not their absence.

## Stages
- **S1 identity** — SHIPPED 2026-10-17 (details below).
- **S2 group log** — `pando/group.py`: events `{author, seq, prev, payload, sig}`; per-author hash
  chain; `GET /group/log?since=` catch-up; `POST /group/events` append; content-addressed
  snapshots + delta sync reusing the spore transfer path (15.3x measured). Decision records are
  appended as hashes (request hash, votes hash, tier, choice) so audits do not store payload
  spam. Merkle root checkpoints in v2.
- **S3 discovery and rendezvous** — mDNS advertisement (join.py already browses
  `_pando._tcp.local.`); seeds publish signed endpoint records inside the log; nodes try peers in
  order. WAN today: Tailscale (already present on this machine); later Iroh/libp2p per the survey.
- **S4 resilience** — any seed can host an island root (demonstrated on the Mac); root loss means
  an island keeps the group; graft merges on return. No gossip until fan-out demands it
  (decision point: >20 mirrors or measured cross-region fan-out cost).
- **S5 scale** — forest of bounded islands (3-7 per island), votes counted per island (mac
  stance), bridges relay exactly one vote; then the 50-100 node churn run (churn arena v2 +
  fleet launcher).

## What runs vs what is planned (as of 2026-10-17)
Runs today: a star-shaped group (Windows root + pi5 voter over LAN + Mac island over WAN),
pairing tokens, scoped grants, LAN + WAN via tunnels, the first third-party voter (pi5).

## S1 status — shipped 2026-10-17
- `pando/identity.py`: Ed25519 keypair at `~/.pando/identity.json` (`PANDOSI_IDENTITY`
  override), chmod 600; `node_fingerprint = sha256(pubkey)[:16]`; canonical-JSON signing.
- rootd: `/register` verifies the signature and **binds node_id -> pubkey durably** (survives
  the TTL'd registry); a token holder cannot re-register an existing id with a different key
  (403); `--require-identity` rejects unsigned registrations entirely.
- stemd/join: load-or-create identity; every registration is signed (recomputed after the
  capability refresh each beat).
- Verified live: the pi5 re-joined with a signed registration; `/nodes` shows its pubkey;
  an unsigned registration returns 403; the signed member still casts a counted vote
  (responded=1, 46 ms). Test suite: 32/32 green.
- The pairing token is now only the **invitation**; the key is the **membership**.

## Next PR (S2) scope
files: `pando/group.py`; rootd endpoints `/group/log` + `/group/events`; decision-hash records
hooked into `/decide`; tests: chain verification, catch-up delta, tamper rejection.
