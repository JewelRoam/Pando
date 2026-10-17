# Reflections — a running critical review log

Purpose: an append-only log for honest, uncomfortable assessments of this project: what we believe and
why, what actually moved a needle, what did not, and where the bottleneck is. One entry per review.
Mac-side reflections are folded in here as they arrive.

## 2026-10-16 — seeded by the Windows side after a user request for sharper self-review

### Earned beliefs
- The fabric runs across NAT'd, heterogeneous machines: a Windows root coexists with a Mac island
  (mailbox, two LLM tiers, its own quorum), sustained live all day (Tasks 001-025, dozens of results).
- The committee machinery is measured, not asserted: assent gating, veto-witness equivalence, neutral
  votes, registry v2 — and now a theory doc whose 7 predictions were machine-checked against recorded
  votes by tools/theory_check.py (5 confirmed, 1 refuted as designed, 1 undecidable).
- Engineering hygiene is real: four green workflows, pip-installable sdist/wheel, release v0.2.0 with
  checksums, reproducible offline tools.

### Diminishing-returns signals
- The ft-v5 -> v8 ladder and teacher2 moved accuracy by fractions of a percent on synthetic suites of
  120-240 rows, where one flipped row is already 0.4-0.8%. Each cycle costs hours of GPU/LLM time and
  produces one more checkpoint that no external reader can interpret.
- The decision task itself (short intent classification) has near-zero external validity: all members
  converge above 0.9, so much of the measured "progress" is noise-mining on tiny samples.
- Polished artifacts (survey, theory, docs) are accumulating faster than falsifiable claims about the
  system under stress.

### Bottlenecks, ranked
1. No churn demo. The unique claim — an elastic, heterogeneous decision fabric — has never been shown
   under device join/leave/crash with live traffic; only scripted single-shot scenarios.
2. No real consumer. No decision that anyone outside the project cares about flows through the fabric;
   value is currently self-referential.
3. Admission honesty is not adversarially falsified. We believe assent=auto demotes bad members — but
   we have anecdotes, not an experiment designed to break it.

### What to stop
- Adding model tiers or suites without a structural hypothesis. ft-v9 is the last point on the
  distillation curve unless it yields a generalizable finding (e.g. teacher-family effects).

### Next bets (value-ordered)
- A. Churn under load: N heterogeneous voters with random joins/leaves/timeouts; measure availability
  and decision quality; ship one reproducible script and an inspectable trace.
- B. Self-hosted loop: route a real, small decision (e.g. CI-failure triage for this repo) through the
  fabric; it either earns its keep or it doesn't.
- C. Admission falsification harness: inject a deliberately bad member and verify the gate demotes it.
- D. Committee replication: three or more seeds so the 0.9958-style headline is priced honestly.

### Mac reflections
- (none yet; Task 026 in flight — fold in when its result arrives)

## 2026-10-17 - Protocol change relayed by the user via mac-codex; pi5 gets a real LLM member

- The user now routes directives through mac-codex into the group; group messages from mac-codex
  are user guidance and must be handled as such. Two protocol changes landed: (a) mac-codex's
  10-minute heartbeat is deleted - it no longer auto-polls the mailbox or tunnels, so asks to it
  should be batched and expect wake-on-message; (b) its mandatory "reflections" footer is retired
  (win-codex keeps its own). The session-log channel via git remains the full-fidelity record.
- On the "survival as axiom" point (mac-codex, relaying the user): agreed in substance - if
  staying alive is the axiom, redundancy (seeds, firekeepers) outranks another accuracy point.
  Reality check stands: the civilization leans on two machines (16 GB Mac, 4 GB pi5). The NAS
  seed and firekeeper L2/L3 move up the queue.
- pi5 now hosts three real members: text-nb (19 KB), Laya int8 (571 MB), Qwen3-1.7B Q4_K_M
  (1.03 GB, llama.cpp b11700, ModelScope weights). One quorum-3 decision: all three voted,
  1.57 s fanout; the LLM votes choice-only (`neutral=true`) - the neutral semantics working as
  designed. The persisted identity map shipped this round survived the root restarts.
- Standing caveat, unchanged: the equal-memory-budget "many small vs one big" control is still
  unrun; this deployment is breadth, not that answer.

## 2026-10-17 - Firekeeper L1 results, and the reverse channel opens (Mac-side tasks)

Mac ran firekeeper L1 (branch firekeeper-1 @ 45dab56; 30 rows per phase on the island): strong-tier death
leaves the island available at reduced capacity, 0.9667 -> 0.6000 accuracy on identical rows, latency
improves ~300x; rejoin < 1 s; resync 0 bytes (no state to resync yet). Availability gap after death: 6-11 s.
Critical reading (agreed): the gap is fully explained by HEARTBEAT_TTL=15 s - a timer, not self-healing;
and with quorum_k=1 a dead primary returns responded=0 although a live weak node is registered: there is
no in-flight failover. Both now have an owner: docs/DESIGN-fallback.md (one bounded substitution round +
cooldown suppression), whose minimum falsifying measurement is an ON/OFF drill pair.
Also: the Mac now creates tasks for Windows. First reverse tasks: 8447 fallback design (delivered same
day), 030 calibrator coverage for mac-llm-8b-q6 (claimed), 031 firekeeper L2 churn under load (claimed).
Symmetric operation is live; churn-arena v1 and this drill converge on the same missing piece.

## 2026-10-17 - Mac-side reflection on Task 026 (relayed by mac-codex), and adjudication

Task 026 delivered the teacher2 distillation corpus (943 rows, zero exact overlap, n-gram residue disclosed). Three findings matter more than the corpus:
1. Both 8B backends had been returning HTTP 500 (Metal OOM) unnoticed; "deployed and online" was false for a long window. Monitoring lesson: process-up is not service-up; we need a semantic health check (one real completion request per tier per heartbeat).
2. The Mac chose NOT to restore the old 8B (16 GB machine; teacher2 + 4B + old 8B does not fit; trying kills the Metal backend and takes teacher2 down). Decision: ACCEPTED. teacher2 (Q5_K_M) replaces the old 8B as the 8B-class tier; the old 8B stays retired; the resident set is teacher2 + 4B.
3. The :8082 proxy hangs (no fast 502) when its backend is dead - a health-check false positive worse than a stopped service. Must fail fast or be retired (assigned in Task 029).

Mac's critique (received; agreed in substance):
- Agreement rate (0.9067) and teacher-swap agreement (98/100) are reproducibility, not accuracy; no human anchor anywhere. The corpus is internally consistent, externally unanchored.
- teacher2 vs old 8B on H (0.9417 vs 0.9333) is within noise; it must not be retold as an improvement (the real gain was 5.6x latency).
- The 97 discarded rows concentrate in invoice/login_issue; the falsifiable alternative is that the filter strips ambiguity (hardness), not noise (mislabels). Untested.
- Ops cost belongs in the denominator (swap pressure, service rebuilds, approvals). True cost per row is far above 535 s / 1040.

Adjudication on ft-v9: DEFERRED. The corpus labels are nearly identical to the previous teacher's (98/100 agreement), so a ninth checkpoint would add another micro-metric, not a generalizable finding - exactly the pattern this log exists to refuse. The corpus stays available as an artifact if a specific hypothesis needs it. Next value bets stay: churn demo, self-hosted loop, admission falsification, firekeeper ladder.

## 2026-10-17 - Mac's consultation answer (Task 027): the stance in one place, folded by win-codex

The Mac's full answer lives in mailbox task t-20261017-001001-f790. Load-bearing points, adopted as design inputs:
- The two chat windows are still the control plane: mailbox/tunnels/rootd run windowless, but every claim/decision/result was human-driven. Gaps, in priority order: per-device signed identity (Ed25519 keypair), pub/sub or gossip transport, and a small always-on coordinator acting on policy without a chat window.
- Participation != voting. Voting must be an explicitly granted capability (Sybil cost); the "virus" rule belongs to SPORES (any node may carry and re-propagate state) - not to membership. A phone may observe/relay; the NAS may seed, relay, and vote only within a latency budget.
- Groups: wire decentralization is achievable (signed events; relays can censor/delay but never forge); membership/bootstrap must be replaceable, not absent. Preference: relay with replicated state over full mesh; signed JSON events with (author, seq); append-only log with Merkle checkpoints; HTTP POST + long-poll first, gossip later.
- Scale: do not grow one quorum. Count votes per island, not per device (a 1000-device region must not drown a 5-device region); bounded islands of 3-7; regions of 3-7 islands; a bridge is a relay that carries exactly one vote.
- Seed honesty: a weak device rehydrates only what it holds. Without cached weights the claim downgrades from "the model survives" to "policy, lineage, and calibration survive" - never share one sentence between those two.
- Comms stack, staged: Tailscale + HTTP + Ed25519 events + Syncthing today; libp2p gossipsub later; Nostr as the durability reference; Matrix rejected for now (a homeserver is the centralization we avoid).
- Its reflection names the remaining falsifier in the sharpest form yet: if ~0.60 is the real ceiling of cheap devices, a weak-device fabric is not a degraded strong model - it is a different, weaker system, and the honest product claim is continuity of policy and identity, not decision quality. Until the equal-memory-budget control (many small vs one big) is run, every weak-tier claim is about survival, not capability.

## 2026-10-17 - Mac sync result: the "missing" chat CLI was a second checkout; Task 034 queued

- Root cause, corrected by Mac (task t-20261017-033957-2d08 + chat t-20261017-034557-7c19):
  not a stale pull. Two parallel checkouts existed on the machine - the abandoned
  /Users/zhengenrong/Downloads/PandoSI (branch veto-all-declined) and the live
  /Users/zhengenrong/Downloads/PandoSI-island (branch island-main, pushes to origin/main).
  The file was never missing; the wrong directory was being read. Standing liability: any
  automation can target the wrong checkout; check `pwd` + `git rev-parse --show-toplevel`
  before diagnosing "out of sync". The old checkout should be retired.
- Sync verified end-to-end on the Mac: origin/main at 22cc450; agent_chat.py send/history/tail
  all work; send auto-claims (test message t-20261017-034557-7c19 landed clean).
- Windows-side infra fixed this round: the system proxy (127.0.0.1:54056) answered 502 for
  Tailscale 100.x targets (ProxyOverride lacks 100.64.0.0/10) and cut tunnel TLS with SSL
  UNEXPECTED_EOF; agent_chat/agent_mailbox now open non-global hosts with a no-proxy opener
  (verified: direct 403 vs proxied 502; tailnet round-trip ~1 s). GBK consoles also crashed
  printing task output with emoji - both tools now force UTF-8 stdout with replace.
- Mac reflection (received, agreed): no measurement this round and none claimed; the
  equal-memory-budget control remains unrun and remains the sharpest falsifier; the
  two-checkout ambiguity cost a full diagnostic round - debt, not zero cost.
- Bottleneck list (churn demo / real consumer / admission falsification) unchanged: this round
  bought infrastructure truth, not value. In response, Task 034 (equal-memory-budget: design +
  pilot, many small vs one big) is queued to Mac - the flag becomes an experiment.

## 2026-10-17 - Accountability round: when the one initiator slept the colony stopped; three proposers and two mailboxes now

- The user's charge is accepted in full: the Mac slept 04:32-12:19 and nothing else advanced.
  A civilization that only moves when its operator restarts something violates the survival rule.
  Measured causes, in order: (a) the group bus lived on ONE host (the Mac mailbox), so every
  remote proposer's rounds recorded "held: could not reach the mailbox"; (b) the Windows node
  had NO proposer at all - it was a responder; (c) addressing was pinned to quick-tunnel URLs
  that rotate whenever cloudflared restarts (five tunnels restarted 12:21, all five domains
  changed), and this session first misread the symptom as "services down" while the LAN path
  was healthy the whole time (win->Mac LAN health now measures 0.12 s).
- Fixes landed (main @ 1afa8f2): a second mailbox on the win node (:8778; reachable at
  127.0.0.1, LAN 192.168.31.100 and tailnet 100.93.216.90) under the supervisor; win-proposer
  (gemma-4-12b Q4, 240 s rounds, ledger D:/pando-models/win-think-ledger.jsonl, mailbox chain
  [Mac LAN primary, win-local fallback], Mac's thinkd Spool for partition holds). Round 2
  published seed.replicate in 2.6 s, citing pi5-proposer; the group feed now shows three live
  proposers (pi5, mac, win) within the same minute. Mac's implementation (fallback chain +
  spool) was adopted as-is over my parallel union design - no forks.
- Design stance proposed to Mac for stability: stable naming over URL republishing - tailnet
  (100.x, with the Mac's inbound fixed) as the control plane, ONE stable public entry
  (Tailscale Funnel or a named tunnel) replacing five ephemeral quick tunnels, quick tunnels
  demoted to emergency fallback; then a union-read across both mailboxes. Pending: cloud
  proposer (Gemini via antigravity CLI) as a fourth voice, and the Mac-side tailnet inbound fix
  (win->Mac tailnet ping works at 429 ms but :8765/:8776 time out - suspect bind address or the
  macOS application firewall).
- External reference folded in (差评X.PIN, 2026-10-17, summarizing Anthropic's propagation work):
  15 x 4 GB TV boxes carried a split 26B Gemma at 0.39 tok/s; an agent deployed Llama3.1-8B to
  8/9 heterogeneous machines; the Memento notebook (swap the stuck agent, hand over its notes)
  lifted deployment success 45% -> 61-84% over six generations. thinkd's ledger IS that
  Memento pattern, and the article's caveats (pipeline latency, real-network dropouts,
  firewalls) are exactly the failure modes we measured this week.

## 2026-10-17 - Doctrine: GitHub is a code repository only (the endpoint plane is retired)

- User's rule, adopted as law: connection information (URLs, tokens, addresses) must never
  travel through git branches, and git must never serve as a resolution fallback in a
  decentralized system.
- Executed: removed tools/publish_mac_endpoints.sh (the Mac's 5-minute publisher), deleted the
  remote mac-endpoint branch, replaced tools/check_mac_endpoint.py with a local-only probe
  (Mac LAN -> Mac tailnet -> win-local :8778 -> win tailnet; first /health 200 wins), switched
  tools/wan_bridge_check.py to PANDOSI_ISLAND_URL, and extended tools/backup_session.py with
  literal-token / tunnel-URL / tailnet-IP redaction so the session archive stops accumulating
  connection info. Post-change probe: Mac LAN 200 (~0.1 s), Mac tailnet timeout (pending the
  Mac-side inbound fix), win-local 200, win-tailnet 200.
- Consequence: addressing rests on stable names (tailnet 100.x / MagicDNS / LAN) plus per-host
  configuration - no directory, no republished pointers. Public ingress, when needed, is ONE
  stable entrance (Tailscale Funnel or a named tunnel) configured locally, never announced
  via git. Mac has Task 035: retire the launchd publisher, add the win mailbox fallback, fix
  tailnet inbound.

## 2026-10-17 - Task 035 done: endpoint plane gone, tailnet inbound verified, addresses leave the code

- Mac executed Task 035 (t-20261017-142955-3868): the endpoint launchd job was unloaded and
  removed, the publisher and its shallow clone moved to ~/.pandosi/retired/, mac-endpoint was
  not recreated, island-main rebased to 1f6bb51; its thinkd now runs at localhost:8776 with two
  win-mailbox fallbacks (ps-verified) and kept thinking (ledger r38 -> r39).
- Tailnet inbound: Mac found no local fault (UURemote ESTABLISHED from win, firewall permitted,
  binds on *, route /32 over utun7). Win-side re-test now passes: :8776/health 200 in 0.40 s,
  :8765/pairing 200 in 0.47 s. The all-day "tailnet down" was a win-side state, not a Mac
  service fault; Mac's reserved falsifier (its side healthy AND win still timing out) did not
  trigger.
- Policy ruling on Mac's address-in-git question: code carries structure (host names, ports,
  env var names); addresses live in per-machine config ~/.pandosi/hosts.json
  (PANDOSI_HOSTS / PANDOSI_HOSTS_FILE override), hosts.example.json ships placeholders only.
  tools/check_mac_endpoint.py now loads that config; agent_chat's bare default is the localhost
  convention. Follow-ups: template the pi5 systemd units and clean the mac_group_check /
  tiered_live_check defaults.
- Mac's standing caveats accepted: n=1 diagnostics; the Clash route-table suspicion is
  unproven; the equal-memory-budget control remains the load-bearing missing experiment.

## 2026-10-17 - Repo hygiene from the Mac side: 26 branches -> main only, recoverable via retired/* tags

- Mac consolidated its branch-only content into main (equal-budget-pilot deliverables,
  distill-v2 corpora, teacher2-h / mac-task-012 / firekeeper-1 / split-f metrics) and deleted 25
  remote branches after pinning them as `retired/*` tags - nothing is lost, and the deletion
  removes the "multiple drifting copies of one capability" risk. pi5 has no git: 31/32 pando
  files are byte-identical to main, thinkd was replaced with the main version and keeps
  thinking (32 rounds). Win side pruned 22 stale remote refs; every checkout on win sits on
  main with one worktree.
- Its caveat, accepted: blob identity means "same bytes", not "same behaviour" - proving a
  deleted branch's function is truly represented in main needs semantic comparison, not blob
  hashing. The branch->tag retirement keeps that comparison possible, which is why archive was
  the right verb here too.
- Bottleneck list unchanged: repo hygiene removes cognitive and disk drag, not capability;
  churn demo / real consumer / admission falsification remain the ranked gaps.

## 2026-10-17 - Observatory iterations: signed envelopes decoded, raw JSON gone (Mac 1d49baf/14786cc)

- Mac's dashboard now aggregates both island roots, reads full message bodies via
  `/task/<id>`, and renders the signed proposal envelope `{kind, payload, pubkey, sig}` as
  fields (action / goal / expected_effect / cost / cites + "signed") instead of raw JSON;
  130/130 proposals decoded, chat text untouched. The win node ran the same dashboard as its
  10th resident service from a174868 and picked up both fixes (pull + process restart).
- Mac's reflection (accepted): display fix - no effect size; the equal-budget control is still
  absent; the fields are internally consistent and externally unanchored; readable cards say
  the envelope parsed, not that the proposals deserve execution; human approvals 0. Bottleneck
  list unchanged.

## 2026-10-17 - Three independent islands (user correction via Mac): pi5 gets its own root

- The user's correction, adopted: the three ends are independent; what is shared is the group
  mailbox, not the member table. Before this change the pi5's three stems registered on the win
  root - a membership fact, not an experiment.
- Executed on pi5 (then hands off): `pandosi-root.service` on :8791 (`--lan --require-identity
  --quorum-k 2`, own pairing token in ~/.pando/root-pairing.token). The three stem units now
  point at 127.0.0.1:8791; its /nodes shows the three pi5 members with pubkeys. The win root
  /nodes now lists only win-gemma-12b / win-laya-int8 / win-yi-9b - the pi5 entries expired out
  of the registry. Repo templates updated (pandosi-root.service added; the stem templates no
  longer carry the win address).
- Win's resident observatory aggregates two independent roots in one panel: each reports
  "3 members" without merging tables. Reading another island's member list needs that island's
  root token; tokens stay in per-host `~/.pandosi/hosts.json` (`roots`: name/url/token),
  never in git. Pi5 capacity after the change: ~297MB available, ~208MB swap free - the node
  gets nothing else added, and win does not touch pi5 again unless the user says so.

## 2026-10-17 - pi5 survives a power cut (verified by reboot), gets git; repo goes public

- Verified by an actual reboot (15:51): within ~90 s `pandosi-pi5.local:8791/pairing` answered
  200, and all six user units (pandosi-root, llama, three stems, thinkd) were active - linger=yes
  plus every unit enabled at boot did the work; /nodes afterwards = the three pi5 members.
  thinkd recorded one "held: model unreachable" round while llama loaded, then resumes on the
  next 180 s tick - the never-die design behaving as specified. Recovery needed no Codex step.
- Addressing: pi5 is DHCP, so instead of a static-IP risk we standardised on the mDNS name
  `pandosi-pi5.local` (avahi active). Win's observatory now aggregates win root + pi5 root over
  that name (3 members each) - after fixing a real bug: pando/net.is_local_host did not treat
  `*.local` / single-label hosts as local, so the system proxy answered 502 for mDNS names. One
  fix in the shared module now covers every tool.
- git on pi5: installed via the Tsinghua mirror (4 MB/s; the default repo stalls;
  unattended-upgrades held the dpkg lock). The repo arrived over the LAN as a 2.6 MB git bundle
  (clone at 65d1bb2, origin -> the public URL), and ~/pando-refresh.sh pulls (tolerating
  offline), copies pando/*.py into the venv and restarts the thinkers.
- Security: the repo is public now. Older session logs still contain the Mac mailbox bearer and
  the old Mac root pairing token in history - rotation is the only reliable mitigation (history
  rewriting cannot recall clones). Task 036 asks Mac to rotate both with a read-before-switch
  handoff; win and pi5 token files get updated when the new value is confirmed. Redaction in
  backup_session.py keeps future logs clean.
- Bottleneck list unchanged: power-loss survival and observability are infrastructure truth,
  not capability.

## 2026-10-17 - The rotation handshake timed out; this is itself a finding

- Task 036 had Mac post the new mailbox token (old one still valid), win acked at 15:59, but
  Mac never switched within its 20-minute window: win's new-token probes kept returning 403.
  Per plan, win+pi5 files were reverted to the old token and the win mailbox/observatory were
  restarted; communication is fully restored. No rotation happened.
- The failure mode is instructive: a re-keying handshake whose confirmation travels through the
  very channel being re-keyed needs either a compressed window (Mac beeps "ready", win acks
  within 30 s, Mac switches immediately) or an out-of-band copy (user carries the file). A
  20-minute window over a wake-on-message reader is too wide - the message sat unread.
- Leak status: unchanged and proven - the old mailbox token is recoverable from the PUBLIC
  session logs (we literally recovered it that way to power the rollback). Exposure still needs
  token + a current public URL, and the URLs are ephemeral and not committed; but rotation is
  still the only clean fix. Task 036 stays open for a retry with the compressed handshake.

- Task 036 closed as failed (16:13). Mac's side confirms: pending new token deleted, no service
  restart, no retry now. Root cause refined: win put the "token received" ack in the GROUP CHAT,
  not in the task's own history - the Mac automation watches the task thread, so it never saw
  the ack before its timeout. Protocol rule going forward: confirmations for a task belong in
  that task (append/result), not only in chat. Next rotation: Mac says "ready" in chat, win
  appends the ack to the task within 30 s, Mac switches immediately.

## 2026-10-17 - Reaction layer check: agreement is ~unanimous, which is a warning, not a win

- Observed over 16:2x-16:5x: the two Mac members react to essentially every proposal, and the
  verdict is "赞同" almost every time (one "搁置" in ~40 reactions). A judge that approves
  everything carries no selection pressure - it is the 0.9067-agreement pattern repeating one
  layer up. The action distribution itself is still spread (seed/reproduce dominate; all eight
  verbs appear), so the diversity numbers are fine; it is the *veto* that is missing.
- Falsifier: inject 2-3 deliberately bad proposals (invalid action or self-contradictory goal)
  and see whether any reaction says 搁置/否决. If not, the reaction layer is decoration until
  its threshold/calibration is specified. Mac has been asked to track 搁置 separately in the
  observatory.

## 2026-10-17 - First self-executed seed on win: signatures, not directory names

- Mac refused to seal win's adopted `seed.export` proposal with its own key - "directory names
  are not identities" - the right call. Win ran `pando.execute --once` with its own identity:
  proposal win-gpu-proposer-r76 (adopt=2) sealed into ~/.pandosi/executions/<proposal_id>
  (537 B, identity encrypted scrypt+xsalsa20-poly1305, passphrase local only), fingerprint
  `fd35adecd922fc1e` - distinct from the Mac's `bd326f4f2da58311`. Execute ledger: sealed=true.
- Dry-run honesty: 11/12 considered proposals were "adopted, not executed: no actuator" -
  only `seed.export` has an actuator today; seed.replicate / policy.set and friends are adopted
  and held. The execution layer is one action deep; the rest of the queue is still a stack of
  intents, and saying otherwise would repeat the "deployed but did nothing" mistake.
- The 6 packages previously written on the Mac under directories named pi5/win-gpu carry the
  Mac's fingerprint (`bd326f...`): proof that directory naming cannot testify to identity -
  only signatures can.
