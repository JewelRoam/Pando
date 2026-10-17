# tools/ - what lives here

This directory accumulated ~125 scripts during the research rounds. They
are not all the same kind of thing. Read this index before adding another.

## Operational (the resident system)

| tool | role |
|---|---|
| `agent_chat.py` | group chat over the task mailbox: send / history / tail (`[chat]`-titled tasks are messages). |
| `agent_mailbox.py` | the mailbox itself: `serve` (server) plus the client verbs (put/list/get/claim/result). |
| `check_mac_endpoint.py` | local-only mailbox resolver; candidates come from `~/.pandosi/hosts.json`, never from git. |
| `backup_session.py` | sanitized session archive; redacts tokens / tunnel URLs / tailnet IPs at write time. |
| `observatory.py` | one-command dashboard: aggregates every island root, and shows a decision model's stance under the proposal it answered. stdlib only. |
| `pando.reactd` | asks only this machine's decision models for 赞同/搁置/反对 and publishes a signed reaction. Not a tool script: `python -m pando.reactd`. |
| `think_stats.py`, `think_watchdog.py` | proposer-ledger statistics; stall watchdog for a frozen thinker. |
| `win_node_supervisor.ps1`, `win_stack_restart.ps1` | keep the nine resident win services alive; restart everything. |
| `win_worker.py` | the local model answers `[chat]`/tasks on the win node's behalf. |
| `win_yi_probe.ps1`, `ctx_guard_probe.py` | one-shot hardware probes (Yi CPU load; llama-server VRAM/context guard). |
| `pi5_llm_member.sh`, `pi5-systemd/`, `mac-launchd/` | per-host service units for the pi5 and the Mac. |

Rule: operational tools carry no machine addresses; hosts live in
`~/.pandosi/hosts.json` (shape in `hosts.example.json`).

## Fabric invariants (run locally; CI is a second copy)

`check_no_weights.py` (no weights in git), `genesis_check.py`,
`genesis_log_check.py`, `genesis_transport_check.py` (signed proposals),
`topology_check.py`, `demo.py`, `verify_wheel.py` / `verify_sdist.py`,
`service_check.py`, `swarm_check.py`, `slow_stem_check.py`. Before pushing,
run `.venv/bin/python tools/local_ci.py`. It is the same list as
`.github/workflows/ci.yml`. A failure found only by the remote run was a
skipped local run. `tools/githooks/pre-push` runs that command. Enable it
once with `git config core.hooksPath tools/githooks`.

## Research evidence (`*_check.py` and friends)

Everything else is an experiment write-up from a research round: each one
builds a scenario, measures it, and prints numbers quoted in
`docs/research/` and README. They are kept for reproducibility. Treat them
as historical: do not copy their style into new operational code, and
before deleting or moving one, grep the docs for its name (the README links
most of them).

When adding a new experiment: one file, one question, print the numbers,
link it from the research-round doc.

Superseded harnesses whose numbers are still cited move to `experiments/`
(see `experiments/README.md`); this directory keeps the live set.

## Companion modules

Shared code lives in `pando/`. `pando/net.py` is THE way to open URLs:
local hosts go direct, public hosts keep the system proxy. Never write
another `ProxyHandler({})` by hand.
