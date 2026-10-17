# Firekeeper drill — level 1 (service death), 2026-10-17

Island: `mac-text` (text-nb-v1, :8766) + `mac-llm-8b-local` (strong tier, :8768),
rootd :8765 quorum_k=1.  Strong tier = a stemd fronting **teacher2**
(:8087 -> llama-server Qwen3-8B-Q5_K_M on :8086, checkpoint `mac-llm-8b-q6`).
The old 8B Q4 backing is retired; the :8768 stem was re-pointed at teacher2 and
its launchd agent rewritten (backup `~/Library/LaunchAgents/com.pandosi.stem.llm8b.plist.bak-028`).

Sample: first 30 rows of `assets/heldout6_test.jsonl` (split H, 8-class `intent`),
identical rows in every phase, driven through the island root `/decide`
(Bearer = pairing token), `curl`-free stdlib driver.

## Timeline (UTC)

| time | event |
|---|---|
| 00:19:27–00:19:35 | **baseline** — full island, 30 rows |
| 00:19:40 | strong tier killed (`launchctl bootout gui/501/com.pandosi.stem.llm8b` + :8768 dead) |
| 00:19:43 | probe: root still selects the stale strong node → `choice=null, responded=0, tier=escalate` (1.0 ms) |
| 00:19:45–00:19:48 | strong node ages past `HEARTBEAT_TTL=15 s` and is evicted from `live_nodes()`; mac-text takes the quorum slot |
| 00:19:53 | **dark period** — 30 rows served by `mac-text` alone |
| 00:20:18 | **dark period, repeat** — same 30 rows |
| 00:20:24 | rebirth: launchd agent re-bootstrapped; :8768 health ok after **0.53 s**, strong node re-registered after **0.82 s** |
| 00:20:29–00:20:34 | **post** — full island, 30 rows |
| 00:20:34 | final check: root live, mailbox 200, 5 cloudflared tunnels, both stems up |

Availability gap after death: **≈6–11 s**, bounded by `HEARTBEAT_TTL` (15 s) plus the
`/decide` that lands in the window; no explicit fallback fires before the TTL.

## Results (n = 30 per phase, identical rows)

| phase | served | availability | accuracy | p50 ms | p90 ms | max ms |
|---|---|---|---|---|---|---|
| baseline (strong) | 30/30 | 100 % | 0.9667 | 295.7 | 359.8 | 378.0 |
| dark (weak only) | 30/30 | 100 % | 0.6000 | 0.7 | 1.1 | 11.3 |
| dark repeat | 30/30 | 100 % | 0.6000 | 0.7 | 1.2 | 12.8 |
| post (strong back) | 30/30 | 100 % | 0.9667 | 159.0 | 256.4 | 290.3 |

- **Survival, not death:** the fabric did keep serving with the strong tier gone —
  but only *after* the heartbeat TTL evicted the dead node. During that window the
  root had a live weak node and did not use it.
- **Accuracy collapses:** 0.9667 → 0.6000 on the same rows (weak tier alone).
  The 0.60 is the naive-Bayes text model's ceiling; 5 of 30 rows were `verify`, not
  `auto`, but every one still answered.
- **Recovery is cheap:** rejoin < 1 s, and post accuracy returns to baseline exactly.
  No spore/delta resync bytes moved — this deployment has no state to resync.
- **Latency trade:** the strong tier costs ~160–380 ms/decision; the weak tier is
  ~0.5–1.2 ms. During the dark period the island was ~300× faster and ~37 pp less
  accurate.

## Anomalies
1. **No same-decision fallback.** `_call_stem` returning `None` only shrinks the
   quorum; with `k = 1` there is nothing to shrink to, so a dead primary yields
   `responded = 0` until the 15 s TTL expires. A live weak node sat idle the whole time.
2. **Changeover is not instantaneous.** Availability recovered only via heartbeat
   expiry; the observed dark window is 6–11 s depending on when the request lands.
3. **Checkpoint swap is silent to the calibrator.** The strong checkpoint changed
   `mac-llm-8b` → `mac-llm-8b-q6`; there is no calibration mapping for the new name,
   so votes are aggregated uncalibrated.
4. **Errors moved after rebirth.** Baseline and post both hit 0.9667, but the one
   error is on a different row (a complaint/refund row in post), so the equality is
   not identical per-row behaviour.

## What this does not answer
- It never ran the equal-memory-budget control (many small models vs one large model),
  the only test that can falsify the project's premise.
- n = 30 per arm; every accuracy delta below ~9 pp (1 SE at p≈0.5) is noise, and the
  baseline/post 0.9667 ± 0.033 has roughly one wrong row of headroom.
- Labels come from this repository's own generator: internally consistent, **not
  externally anchored**.
