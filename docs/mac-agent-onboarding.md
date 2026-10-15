# Mac execution node — onboarding

How the Windows Codex session and the Mac Codex session work for each other.
They never talk directly; they share a task mailbox served by the Mac over
the LAN (Windows -> Mac only, same direction as the group bridge).

The live token is **not in this repository**: it lives in
`metrics/lan-check/mac-mailbox.token` on Windows (gitignored) and in
`~/.pandosi/mailbox.token` on the Mac. Copy the paste-block below with
`<TOKEN>` filled in; it is the only shared secret and it is a LAN password,
nothing more.

## What the Mac session sets up

1. `git pull` — `tools/agent_mailbox.py` must exist.
2. The mailbox service, on the LAN, token-guarded:

   ```bash
   mkdir -p ~/.pandosi
   printf '%s' '<TOKEN>' > ~/.pandosi/mailbox.token
   nohup python3 tools/agent_mailbox.py serve --port 8776 \
       --dir ~/.pandosi/mailbox --token-file ~/.pandosi/mailbox.token \
       > ~/.pandosi/mailbox.log 2>&1 &
   ```

3. A persistent goal (`create_goal`) and a heartbeat automation
   (`automation_update`, every ~5 minutes) that polls `list --status open`,
   claims what it finds, executes it inside the repository, and writes the
   result back. Quiet when there is nothing to do.

## What the Windows session uses

```powershell
python tools/agent_mailbox.py put --url http://192.168.31.41:8776 `
    --token-file metrics/lan-check/mac-mailbox.token `
    --title "..." --body-file task.md
python tools/agent_mailbox.py list --url http://192.168.31.41:8776 `
    --token-file metrics/lan-check/mac-mailbox.token --status done
python tools/agent_mailbox.py get --id t-... --url ... --token-file ...
```

The verbs: `put` (create), `list`, `get`, `claim`, `result`
(`done|failed|blocked`), `append` (progress). `/health` needs no token.

## Ground rules for the Mac node

- Work inside the repository or `~/.pandosi/`; leave unrelated user files
  alone.
- Allowed: `git pull`, pip/brew installs, llama.cpp/MLX model deployments,
  local services, its own sub-agents.
- Any new service bound to `0.0.0.0` carries a token (e.g. `llama-server
  --api-key`); the mailbox token never enters the repository.
- Do not touch the running `mac-text` stem (the :8765 root) or the mailbox
  service itself.
- Results: status + concise output (what ran, key commands, versions,
  ports, measured latency/throughput) + artifact paths; `blocked` when
  user authorization is needed.

## Liveness

- The Mac must stay awake and the Codex app open for heartbeats to fire
  (`caffeinate -dimsu` helps on macOS).
- If the Mac sleeps or drops off the LAN, Windows `put`s fail and queued
  work simply waits; nothing restarts, nothing is lost.
