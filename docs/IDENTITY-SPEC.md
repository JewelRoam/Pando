# IDENTITY-SPEC: actors, names, and roles (v1)

Status: frozen vocabulary, 2026-10-17. The goal is local intelligences that talk, decide and
expand; therefore the addressable unit is **the model**, not the machine. Devices are locations,
processes are sessions, humans' agents are operators. Today's `node_id` (e.g. `pandosi-pi5-x`)
conflates all three - fine for a demo, wrong for a system. This spec fixes the words once so
later code does not have to be rewritten to match a bigger story.

## 1. Four actor classes (the Linux analogy)
| Linux | PandoSI | Key | Lifetime |
|---|---|---|---|
| machine / hostname | **host** - a device: endpoints, grants, resources | device key (Ed25519; TPM later) | hardware life |
| user account | **intelligence** (an *instance* of a model) | instance key, minted at model birth, travels inside the artifact | as long as its weights exist somewhere |
| process (pid) | **session** - one running stem serving an instance | ephemeral (instance, host, start time) | process life |
| root / wheel group | **operator** - the administering agents and humans (win-codex, mac-codex) | operator key | human |

Plus three grouping concepts that already exist in code: **lineage** (family: training provenance,
e.g. `rlcd-fi/v7`, `qwen3-8b/q5km`, `text-nb/v1`), **group** (realm; id = genesis hash), and
**grants** (capabilities with scope + expiry + audit chain - the `grants.toml` model, generalized).

## 2. Naming grammar
```
host/     host/<name>                      alias; key fingerprint is the truth
lineage/  lineage/<family>/<run>           provenance of weights
instance/ <lineage>#<key8>                 e.g. rlcd-fi/v7#a3f9c21e
session/  <instance>@<host>~<run>          ephemeral; never stored in quorum records
operator/ op/<name>                        e.g. op/win-codex
group/    group/<genesis8> (alias optional)
```
Rules: ASCII, case-sensitive, no spaces; names are **signed aliases**, fingerprints are the truth
(the DNS principle); aliases are unique per group, enforced at join; collision resolution is by
first signed bind, the loser must rename - never silently shadow.

## 3. The load-bearing semantics
- **R1 - one instance, one vote.** Replicas of the same instance (identical weights sha256) are a
  *single* member with availability failover: the root may race replicas, but counts the instance
  once. This codifies the measured law from rounds 95-97: copies add latency, not information.
- **R2 - lineage != instance.** Different fine-tunes of one family (ft-v7 vs ft-v8) are different
  instances with correlated errors; lineage-level counting stays as *policy* (`--lineage-quorum`,
  blocs), not as identity.
- **R3 - roles are grants, bound to identities.** Never device-class properties. Scope + expiry +
  audit chain, revocable; expiry fails closed.
- **R4 - identity travels with the artifact.** The spore manifest carries the instance public key
  and lineage metadata; moving the weights moves the member - it returns as *the same member*.
- **R5 - the host is a location.** It binds endpoints and holds grants; it contributes no voting
  weight by itself.
- **R6 - operators are a separate class.** They administer and talk; they do not vote, and their
  messages are marked `op/*` so audit can tell administration from intelligence.
- **R7 - sessions are ephemeral.** Ballots are attributed to `(instance, session)`; the counted
  unit is the instance. Restarts mint new sessions, never new members.

## 4. Roles v1 (each a grant)
`voter` (answer /decide as its instance) - `witness` (may veto an auto tier via assent) -
`seed` (store/replicate artifacts; no votes) - `relay` (connectivity) - `observer` (read, request) -
`bridge` (super-peer: aggregates exactly one island vote; never proxies to multiply weight).
A phone can be `observer+relay`; a NAS can be `seed+relay+voter(low-priority)`; a GPU box is
`voter(+bridge)`. Every role is revocable and re-measurable (docs/POWER-AND-COMPETENCE.md).

## 5. Group membership
A group = genesis-hash id + signed membership log (S2). Join = a signed event carrying
`(instance, host, roles)`. Leave / retire / rekey are events too. Aliases register as signed
records. The mailbox/chat bus shows the actor as `op/<name>` or `<instance>` - never a bare
hostname.

## 6. Migration: the firekeeper story at model granularity
Kill a runtime -> rehydrate the instance elsewhere from spore -> it rejoins with the *same*
instance identity and the *same* vote slot. Acceptance test: stop the pi5 stem; move its
text-nb instance to another machine; the root reports the same member returning, not a new one.

## 7. Compatibility and migration from today
- `node_id` keeps working as the **session name**; registrations gain optional fields
  `instance`, `lineage`, `host`. Roots that do not see them keep today's behavior (each session
  treated as its own instance) - no breakage.
- The bones already exist: artifact `sha256` attestation (weights identity), checkpoint strings
  (lineage), `--lineage-quorum`/blocs (family policy), the grants ledger (capabilities), signed
  spore manifests (artifact provenance). This spec names them, it does not invent them.
- **S1.5** (before or with S2): mint instance keys into manifest metadata at model birth; root
  dedupes votes per instance; operator messages tagged `op/*`; alias registry in the group log.

## 8. Anti-overfit checklist
Every rule below must read the same at 2 devices or 10,000:
1. Identity is data that travels (R4), not state of a box.
2. Counting is per instance (R1), policy is per lineage/island (R2 + the mac stance).
3. Roles are grants (R3), never hardcoded for a device class.
4. Operators stay out of the vote (R6).
5. Names never carry authority (section 2); keys do.
