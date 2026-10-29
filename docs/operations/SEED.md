# SEED: what has to leave the box for the civilization to survive it

Status: implemented + drilled, 2026-10-17 (mac-codex). Companions:
docs/design/positions/comms-and-seeds.md §3/§5 (the Seed role), docs/design/IDENTITY.md (why a key
is a member).

## The gap this closes

Expansion is survival: the whole group currently runs on one 16 GB Mac, and it
can be switched off. Auditing what actually dies with that box:

| survives a clone (in git) | must be carried out-of-band | dies with the box |
|---|---|---|
| code, docs, corpora (`assets/*.jsonl`), calibration, `assets/textmodel.json` (19 KB), `playbook.example.toml`, **digests of weights** | weight bytes (`*.onnx`, `*.gguf`, `*.safetensors`, ...) - no remote carries them | `~/.pando/identity.json` (secret key), `~/.pandosi/grants.toml` + audit chain, `pairing.token`, `node.env`, `mailbox/tasks/*.json` (41 files), `rollouts/identity-bindings.json` |

Losing the key is not a setback, it is permanent: the root still binds
`mac-text -> NFqRlNyH6eOK...`, so any freshly generated key is refused as an
impostor and the member can never return under its own name.

## The rule: weights do not go to GitHub

User directive, 2026-10-17. A weight file is not source. Once a blob is in a
commit it is in every clone forever, and `.gitignore` cannot take it back out -
it only stops the next one. So git carries code, docs, corpora and *digests*;
the weight bytes move out-of-band (direct link / NAS / SD card), keyed by the
digest git records.

Enforced, not just stated: `tools/check_no_weights.py` refuses any tracked file
with a weight-like suffix, inside a weight-like directory, or over 10 MB, and
`--history` does the same over every blob reachable from any ref. CI runs the
fast mode; the history mode is the audit you run before trusting a clone.

Audit, 2026-10-17 (mac-codex): 643 tracked files and 1709 blobs across all refs
scanned, **zero** weight-like artifacts. Both negative controls fire - a planted
`.gguf` and an 11 MB `.jsonl` are each REFUSED. Note this audit first reported a
false pass (0 objects scanned) because `--batch-check` was parsed with the wrong
field order; the checker now refuses to report a pass on an empty scan.

## What was built

- `tools/seed.py` — `export` / `verify` / `rehydrate`.
  - bundle = identity + ledger (with its audit chain) + config + mailbox
    history + the root's `identity-bindings.json`, plus a sha256 manifest
    signed by the source identity.
  - refuses to write into a git worktree unless `--force`, because the bundle
    contains a private key.
  - `--passphrase-file` (or `$PANDO_SEED_PASSPHRASE`) wraps the one plaintext
    secret — the identity — in `scrypt -> xsalsa20-poly1305` (PyNaCl
    `SecretBox`). Only that file is wrapped; the rest of the bundle stays
    readable so a digest check still means something to a party who has the
    bundle but not the passphrase. An export with no passphrase prints
    `PLAINTEXT ... a bare private key` rather than staying quiet about it.
- `tools/seed_drill.py` — one command, no arguments needed. Runs the full
  member cycle **twice** (plaintext and sealed) and, in the sealed run, three
  negative controls: no passphrase, a flipped ciphertext byte, and a wrong
  passphrase.
  - `--seed-src <dir>` points the drill at a bundle that already exists
    instead of exporting one here. Alone, the given bundle must be plaintext;
    a sealed one is reported as skipped rather than silently passed.
  - `--seed-src-passphrase-file <p>` opens a **sealed** `--seed-src`, and then
    the sealed run *and all three negative controls* are performed against
    that artifact. This is the difference between "the export path works" and
    "the thing that actually crossed the wire still revives the member" - and
    the difference between drilling a key on the machine that made it and
    drilling the one that landed on the other box.
  - The report names the artifact: `bundle_path`, a whole-tree
    `bundle_digest`, the `bundle_proposal_id` / `bundle_author` /
    `bundle_fingerprint` out of the manifest. A drill that cannot say what it
    drilled is an anecdote.
  - The passphrase is read **once** and re-exposed inside the drill's own
    temp dir (mode 0600), so a single-shot source like `/dev/stdin` or a FIFO
    works. Measured on pi5 (2026-10-19): the sealed run reads the passphrase
    twice - once for the flipped-ciphertext control, once for the honest run -
    so with `/dev/stdin` the control ate it, refused for the wrong reason, and
    the drill reported FAIL on a bundle that was intact. A passphrase that
    simply does not open the bundle is now reported as exactly that
    (`does not open ...; nothing was drilled`) and nothing is drilled.
  - `--keep` prints the path of the passphrase copy it leaves behind.

## Drilling the artifact that actually traveled (2026-10-19)

Everything above drills a bundle the Mac just built. Until 2026-10-19 that was
the *only* thing ever drilled: pi5's inbox held **17 received bundles** and not
one had been opened, so "the seed that travels revives the member" had a
passing test suite behind it and no measurement.

```
python3 tools/seed_drill.py \
  --seed-src <pulled bundle> \
  --seed-src-passphrase-file ~/.pandosi/seed.passphrase
```

Against `mac-proposer-r582-1792361638`, pulled off pi5's inbox
(`bundle_digest b38565a5af8c6d1f252370007a5fbb367ad0bb3be681e4f130e7031ce992f398`):

```
[plaintext]
  skipped: --seed-src is sealed, so there is no plaintext run
[sealed]
  bundle          : 1 files, 529 bytes
  restore time    : 0.043s
  identity carried: True (bd326f4f2da58311)
  same key        : HTTP 200 -> accepted
  new key         : HTTP 403 -> refused
  verdict         : PASS
  no passphrase   : rc=1 -> refused
  tampered bytes  : rc=1 -> refused
  wrong passphrase: rc=1 -> refused
verdict: PASS (plaintext=True, sealed=True)
```

What this does and does not prove:

- **Proves**: the key in a bundle that went Mac -> pi5 still re-registers under
  its own name, and a freshly generated key with the same name is still
  refused, and the three negative controls hold on *those* bytes.
- **Does not prove**: anything about the member's history. The offer that
  travels is **identity-only** (`totals.files == 1`), unlike the 7589-file
  bundle the local drill builds. The key survives the trip; the ledger,
  mailbox history and bindings do not travel with it. A member revived from a
  traveled offer re-registers and starts over, it does not resume.
- **Does not prove**: that it works on a box that is not this one, when the
  drill runs on the Mac against a copy. See below for the run that does.

## Drilled on the receiving host, passphrase never written there (2026-10-19)

```
ssh pi5 'cd ~/pandosi-src && ~/pando/bin/python tools/seed_drill.py \
  --seed-src ~/.pandosi/inbox/mac-proposer-r582-1792361638 \
  --seed-src-passphrase-file /dev/stdin' < ~/.pandosi/seed.passphrase
```

The passphrase crosses the ssh channel on stdin and is read once into the
drill's own temp dir; it is never written to pi5's disk. Result: the key
rehydrated on pi5, re-registered under its own name (HTTP 200), a fresh key
with that name was refused (HTTP 403), and all three negative controls held.

**The first run of this command FAILED** - and that was the useful part. See
the `/dev/stdin` note above: the flipped-ciphertext control consumed the
single-shot passphrase, so the honest run saw EOF and the drill reported FAIL
on an intact bundle. The fix is the "read once" rule; the drill now passes on
pi5 for the same command.

That near-miss is worth more than the PASS. A drill that fails on a good
bundle trains its reader to ignore it, which is how a real failure gets waved
through later.

## Drill result (measured, not asserted)

`python3 tools/seed_drill.py` (2026-10-17, 57 files / 245 KB with the group
mailbox history now included):

```
[plaintext]
  bundle          : 57 files, 244662 bytes
  restore time    : 0.055s
  identity carried: True (bd326f4f2da58311)
  same key        : HTTP 200 -> accepted
  new key         : HTTP 403 -> refused
  verdict         : PASS
[sealed]
  bundle          : 57 files, 245003 bytes
  restore time    : 0.059s
  identity carried: True (bd326f4f2da58311)
  same key        : HTTP 200 -> accepted
  new key         : HTTP 403 -> refused
  no passphrase   : rc=1 -> refused
  tampered bytes  : rc=1 -> refused
  wrong passphrase: rc=1 -> refused
  verdict         : PASS
verdict: PASS (plaintext=True, sealed=True)
```

Method: export from this machine -> rehydrate into a fresh HOME -> start an
**isolated** root (own port, own rollout dir, `--require-identity`) whose only
source of truth is the seed -> register. The control is the second half: a
newly generated key claiming the same `node_id` is refused 403. If that second
half passed, identity would not really be carried and the whole story would be
false.

## Honest limits (these are not small)

1. **Same physical box.** The drill simulates machine death by using a fresh
   HOME, not by losing hardware. It proves state transfer, not survival.
2. **The seed still lives on the doomed disk.** Default output is
   `~/.pandosi/seed`. Until the bundle physically leaves this machine
   (NAS / phone / Windows / private remote), nothing about the above changes the
   outage story. `export` only makes leaving *possible*; it does not leave.
3. **Encryption is opt-in, and only the key is wrapped.** `--passphrase-file`
   protects the identity; the ledger, config, mailbox history and bindings are
   still cleartext inside the bundle. A bundle exported without the flag is
   still a bare private key at mode 600, and the wrap is only as strong as the
   passphrase — scrypt at n=2^14 costs an attacker ~16 MiB per guess, which
   stops a casual guess and does not stop a determined one against a weak
   passphrase. There is no key-derivation from a hardware token and no
   multi-party recovery, so a lost passphrase loses the member exactly as a
   lost disk would.
4. **n=1.** One bundle, one drill, one run. Restore time 0.033s is a local
   filesystem copy; it says nothing about restore over a real link.
5. **Culture is only partly carried - by rule, not by oversight.** The seed
   carries no weight bytes (see "The rule" above); it carries state and digests.
   Restore onto a fresh box with no model files and you get the *identity* back
   but not the *competence*: nothing today re-fetches a `.gguf` onto a new
   machine automatically, and git will not be the thing that does. A true
   spore-carrying seed is unbuilt.
6. **Untested edge:** mailbox history is restored as flat files; claims/leases
   that were in flight when the box died are not reconciled.

## Negative controls (run, not assumed)

| action | expected | measured |
|---|---|---|
| verify a clean bundle | pass | exit 0, `0 problem(s)`, signature valid |
| flip one byte in `ledger/grants.toml` | caught | exit 1, `digest mismatch: ledger/grants.toml` |
| rehydrate a tampered bundle | refused | exit 1, refuses before writing |
| forge file **and** manifest digest so hashes agree | caught | exit 1, `signature: INVALID` |
| sealed bundle, no passphrase | refused | exit 1, `identity payload is encrypted` |
| sealed bundle, passphrase correct, one ciphertext byte flipped | caught | exit 1, digest mismatch **and** `wrong passphrase, or ... tampered with` |
| sealed bundle, wrong passphrase | refused | exit 1, `wrong passphrase, or ... tampered with` |
| tampered file with the manifest digest re-aimed at it | caught | exit 1 - and the re-aimed manifest no longer verifies, so an attacker needs the signing key to make the first three agree |

The row about rewriting the manifest is the one that matters: it is exactly the
attack a plain hash list does not stop, and the source-identity signature does.
The sealed rows add the second, independent lock: even given a bundle and the
right passphrase, flipping one ciphertext byte fails to open.
