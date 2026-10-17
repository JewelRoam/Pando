# Remote connectivity: how members find each other (doctrine 2026-10-17)

The user's rule, now project law: **GitHub is a code repository only.**
Connection information (URLs, tokens, addresses) must never be published
to branches, fetched from them, or kept as a resolution fallback. The
`mac-endpoint` branch, `tools/publish_mac_endpoints.sh`, and the
git-reading resolver were retired on 2026-10-17; anything that recreates
them is a regression.

## Resolution order (tools/check_mac_endpoint.py)

1. Mac mailbox on the LAN - `http://192.168.31.41:8776` (fastest at home)
2. Mac mailbox over Tailscale - `http://100.92.184.4:8776` (stable 100.x)
3. The local mailbox - `http://127.0.0.1:8778` (win node; survives Mac sleep)
4. Win mailbox over Tailscale - `http://100.93.216.90:8778` (other hosts)

Each candidate answers the token-free `/health`; the first 200 wins.
LAN and tailnet probes bypass any system proxy.

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
  a sleeping Mac cannot silence the colony. thinkd publishes along its
  mailbox chain (primary + fallbacks) and spools only when every mailbox
  is unreachable.
- `docs/session-logs/` is a sanitized historical archive, not a
  transport: token values (matched literally from the local secret
  files), tunnel URLs, tailnet/private IPs, emails and user paths are
  redacted at write time by `tools/backup_session.py`.

## Pending

- Mac: fix tailnet inbound (bind address / application firewall) so win
  can reach :8765/:8776 over the tailnet; add the win mailbox
  (`http://100.93.216.90:8778`) to its thinkd `--mailbox-fallback`.
- Union reads across both mailboxes; a cloud proposer (Gemini via the
  antigravity CLI) as an additional voice.
