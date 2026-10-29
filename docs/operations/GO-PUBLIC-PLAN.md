# Going public: the pre-flight checklist (2026-10-17)

The repository is private today (GitHub answers 404 to unauthenticated reads
while our pushes succeed). That also means **zero stars are possible - nobody
outside can see it**. Before flipping it public, three measured facts matter:

1. **Two live secrets sit in tracked files AND in git history.**
   - The Mac island pairing token: 77 files (76 x `docs/archive/session-logs/*` plus
     `experiments/mac_group_check.py`). The script no longer hardcodes it (it reads
     `PANDO_MAC_TOKEN`), but every history revision still contains it.
   - The Mac mailbox bearer token: ~70 x `docs/archive/session-logs/*`. This is the
     token that guards the task mailbox / group chat behind the public tunnel.
2. **`docs/archive/session-logs/` holds 206 tracked files** - the full internal
   conversation, committed every ~30 min by the session-backup automation
   (thread-scoped: it mirrors the Windows win-codex thread rollout files and
   pushes to main). It includes strategy discussions that were never meant
   for a front page.
3. Clean parts, verified: no tracked file is named like a token/key/secret,
   and the Windows root pairing token does not appear in tree contents.

## Recommended path (default unless the user says otherwise)

- Keep THIS repository private: it is the lab notebook.
- Create a clean public repository from a curated tree:
  - ship: `pando/`, `tools/` worth shipping, tests, LICENSE,
    README + `docs/` that are meant to be read (DESIGN-*, SPEC-*, THEORY-*,
    SURVEY-*), assets needed by the quick start.
  - exclude: `docs/archive/session-logs/`, `docs/evidence/reflections.md`, `artifacts/`,
    `metrics/`, every token file.
- Squash the public repository into one honest initial commit; transplanting
  history is not worth the leak risk.
- **Rotate the Mac pairing token and the mailbox token anyway** (cheap: one
  LaunchAgent restart on the Mac). Leaked-in-history secrets should be treated
  as burned.
- Keep session-backup commits flowing into the private repository only.

## Alternative (only if stars must accrue on this exact repository)

Strip `docs/archive/session-logs/` from all history (`git filter-repo --invert-paths`),
rotate both tokens, verify with a secret scan, then change visibility.
Heavier, irreversible-ish, and still exposes the internal story - not
recommended.

## The honest star plan

Stars follow a readable front page, not more lab work: a 3-minute README
(what it is, demo GIF/trace, quick start, honest limitations), a few runnable
examples, and one reproducible experiment a stranger can re-run offline.
Ship that first; flip visibility second; only then talk about growth.

## Update 2026-10-17 15:5x - the repository is public now

The user flipped visibility to public on 2026-10-17; the "keep private"
recommendation above is historical. Mitigations in flight: (a)
`backup_session.py` redacts token values / tunnel URLs / tailnet IPs at write
time; (b) Task 036 rotates the Mac mailbox bearer and the Mac root pairing
token - the two secrets that are present in older session logs; (c) history
is kept as-is by decision: rotation, not rewriting, is the mitigation (a
public clone cannot be recalled).
