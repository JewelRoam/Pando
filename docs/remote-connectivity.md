# Remote connectivity: reaching the Mac from the Windows fleet

The Mac left the Windows LAN. Everything in this project addresses members
by URL (the task mailbox on :8776, the LLM APIs on :8080/:8082, the Mac
root on :8765), so the problem is not the protocol - it is **how a URL
stays reachable across two NATs**, and **how the Windows side learns the
current URL without a human copying it**.

The architecture that answers both: split the problem.

- **Control plane = the git repository.** The Mac publishes its currently
  reachable endpoints as `endpoint/mac.json` on a volatile branch
  `mac-endpoint` (force-pushes allowed; it is a pointer, not history).
  Windows reads it with `tools/check_mac_endpoint.py`. No tokens ever go
  into this file.
- **Data plane = whatever reaches the Mac**, in order of preference:

## Option A - Tailscale (or ZeroTier): an overlay network

Both machines join the same tailnet and keep their existing services bound
to their local ports; addresses become stable (`http://100.x.y.z:8776`).
WireGuard NAT traversal does the rest; traffic is E2E encrypted; ACLs can
scope who reaches what.

Mac (one-time, needs the user to click an auth link once):

```bash
brew install --cask tailscale      # GUI app; log in
# or: brew install tailscale && sudo tailscaled &
tailscale ip -4                    # e.g. 100.101.102.103
```

Windows (one-time, same tailnet, user logs in):

```powershell
winget install --id Tailscale.Tailscale -e
# then log in from the tray icon; verify:
tailscale ip -4
```

Publish (Mac): write `endpoint/mac.json` with the tailnet base URL and
push the `mac-endpoint` branch. Everything else (tokens, tokens files,
mailbox protocol, stem contract) is unchanged.

Trade-offs: most robust and lowest latency of the options; requires one
account and one interactive login per machine.

## Option B - Cloudflare Quick Tunnel: zero-account bootstrap

The Mac runs `cloudflared` in front of each local service; each gets a
public `https://<random>.trycloudflare.com` URL that survives both NATs.
No account, no sudo; the URL lives exactly as long as the process.

```bash
brew install cloudflared            # or download the static binary
nohup cloudflared tunnel --url http://127.0.0.1:8776 > ~/.pando/tunnel-mailbox.log 2>&1 &
nohup cloudflared tunnel --url http://127.0.0.1:8080 > ~/.pando/tunnel-4b.log 2>&1 &
nohup cloudflared tunnel --url http://127.0.0.1:8082 > ~/.pando/tunnel-8b.log 2>&1 &
# URLs appear in each log; verify each through the public URL locally.
```

Security: the tunnel URL is public; **the existing bearer tokens are what
keep the endpoints closed** (mailbox token, per-API keys). Never publish a
token in `endpoint/mac.json` or in a tunnel URL. Quick tunnels are for the
experiment window; treat URL churn on restart as normal and re-publish.

Trade-offs: fastest to bring up with no accounts; URL churn and a shared
public TLS terminator make it a bootstrap/interim channel rather than the
long-term link.

## Option C - git as the data channel (fallback, not a substitute)

The mailbox protocol is already JSON-over-files by nature; a store-and-
forward variant can ride commits on the `mac-endpoint`-style branches
(Mac pulls tasks, pushes results). Latency is minutes and GitHub rate
limits apply, but it needs no new software and already works today - keep
it as the fallback when both tunnels and overlays are down.

## Option D - true P2P hole punching (research tier)

WebRTC data channels (aiortc on both ends) do UDP hole punching via STUN,
with the git branch as the signaling channel (offer/answer/ICE candidates
as commits). This gives a direct socket with no third-party overlay when
both NATs are cone-type; symmetric NATs need a TURN relay, which needs
hosting. Worth building only if A and B prove insufficient for the
experiment; the git-signaling design can be prototyped later without
changing the mailbox/stem contracts.

## What changes on the Windows side

Nothing in the protocol. The mailbox client takes `--url`; the LLM stems
take `--openai-url`. Read the current endpoints with:

```powershell
python tools/check_mac_endpoint.py
```

(Fetches branch `mac-endpoint` and prints `endpoint/mac.json`; falls back
to the LAN default when the branch is absent.) The Windows heartbeat uses
the same tool when the LAN address is unreachable.
