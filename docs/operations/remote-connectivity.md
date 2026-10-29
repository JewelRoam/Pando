# Remote connectivity: how members find each other (doctrine 2026-10-17)

The user's rule, now project law: **GitHub is a code repository only.**
Connection information (URLs, tokens, addresses) must never be published
to branches, fetched from them, or kept as a resolution fallback. The
`mac-endpoint` branch, `tools/publish_mac_endpoints.sh`, and the
git-reading resolver were retired on 2026-10-17; anything that recreates
them is a regression.

## Resolution order (tools/check_mac_endpoint.py)

Candidates come from per-machine config, never from the repository:

- `~/.pando/hosts.json` (or `PANDO_HOSTS` / `PANDO_HOSTS_FILE`),
  shape: `{"mailboxes": ["<url>", ...]}` - see `hosts.example.json`;
- fallback without config: `http://127.0.0.1:8776` and
  `http://127.0.0.1:8778` (localhost conventions only - no machine
  addresses in code).

Each candidate answers the token-free `/health`; the first 200 wins.
LAN and tailnet probes bypass any system proxy. Recommended order on the
win node: Mac LAN -> Mac tailnet -> win-local -> win tailnet.

## Addresses are names, not pointers

- Tailscale gives each node a stable 100.x address and a MagicDNS name:
  that is the naming layer, not a directory on GitHub.
- The Mac binds :8765 (island root) and :8776 (mailbox); the win node
  runs llama :8086, its stems, and mailbox :8778.
- Quick tunnels rotate their random hostnames whenever cloudflared
  restarts - they are NOT an addressing mechanism. If a public entrance
  is ever needed for outsiders, use ONE stable ingress (Tailscale Funnel
  or a named Cloudflare Tunnel with fixed DNS), configured locally on the
  peers that need it. Never republish URLs through git.

## The bus

- The Mac's mailbox (:8776) is the group's home while it is up.
- The win mailbox (:8778) is the resilience host: always up locally, so
  a sleeping Mac cannot silence the colony. proposer publishes along its
  mailbox chain (primary + fallbacks) and spools only when every mailbox
  is unreachable.
- `docs/archive/session-logs/` is a sanitized historical archive, not a
  transport: token values (matched literally from the local secret
  files), tunnel URLs, tailnet/private IPs, emails and user paths are
  redacted at write time by `tools/backup_session.py`.

## Pending

- Every node keeps its own `~/.pandosi/hosts.json` (the win and Mac
  nodes have theirs); deployed config stays out of git.
- Union reads across both mailboxes; a cloud proposer (Gemini via the
  antigravity CLI) as an additional voice.
