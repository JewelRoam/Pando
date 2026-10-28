# Win node supervisor: keeps the 3070 LLM server, the two fabric members and
# the local worker alive on this Windows node. Started at logon by the
# Startup\PandoSI-Node.vbs wrapper (no admin required); restarts any child
# that exits. Written for Windows PowerShell 5.1 compatibility.
$ErrorActionPreference = 'Continue'
$env:HF_HUB_OFFLINE = '1'   # the ONNX member resolves repo files from local cache
$repo = 'D:\PandoSI'
$logs = 'D:\pando-models\logs'
$stateDir = Join-Path $env:USERPROFILE '.pandosi'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$python = (Get-Command python).Source
$token = "$repo\artifacts\lan-experiment\root-pairing.token"
$dummyKey = 'D:\pando-models\llm-dummy.token'
if (-not (Test-Path $dummyKey)) {
  Set-Content -LiteralPath $dummyKey -Value 'x' -Encoding ascii
}

# The fallback peer lists. One line per address, tried in the order written,
# and the same format read by the same parser on every platform
# (tools/code_sync.py `source_list`). Written here once and then owned by the
# node: when the Mac moves again, the repair is a line in this file, not an
# edit to this script and not a paste from a human on another machine.
#
# Tailscale first on purpose. It reaches the Mac from wherever it is; the LAN
# address below it only works when both ends are on the home network, and on
# 2026-10-19 the Mac left that network (192.168.31.41 -> 172.24.19.146) and
# silently killed every unit on this node whose only peer was the LAN one.
$codeSources = 'D:\pando-models\code-sources.txt'
if (-not (Test-Path $codeSources)) {
  Set-Content -LiteralPath $codeSources -Encoding ascii -Value @(
    '# Fallback code bundles for tools/node_code_watch.py.',
    '# Order is the order tried; the first one that fetches wins.',
    'http://100.92.184.4:8478/code.bundle  # mac, tailnet (survives a move)',
    'http://192.168.31.41:8478/code.bundle # mac, home LAN'
  )
}
$seedSources = 'D:\pando-models\seed-sources.txt'
if (-not (Test-Path $seedSources)) {
  Set-Content -LiteralPath $seedSources -Encoding ascii -Value @(
    '# Fallback outbox bases for tools/seed_pull.py.',
    '# Order is the order tried; the first one that answers wins.',
    'http://100.92.184.4:8478  # mac, tailnet (survives a move)',
    'http://192.168.31.41:8478 # mac, home LAN'
  )
}

# Peer addresses are the Mac's Tailscale name, not its old LAN address.
# On 2026-10-19 the Mac left the home LAN (192.168.31.41 -> 172.24.19.146),
# which silently killed every fallback on this node that dialed that address:
# the Mac mailbox, the Mac bundle peer, and the seed-pull source. Tailscale
# reaches the Mac from wherever it is; using it costs nothing when both ends
# are still on the same LAN. The Windows node's own advertise-host stays
# 192.168.31.100 -- that is this box's address, not the Mac's.
$llamaExe = 'D:\pando-models\llama-cuda\llama-server.exe'
$llamaArgs = @('-m', 'D:\pando-models\gemma-4-12b-it-Q4_K_M.gguf', '-ngl', '99',
  '-c', '4096', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0', '--parallel', '1',
  '--jinja', '--reasoning', 'off', '--host', '0.0.0.0',
  '--port', '8086', '-t', '8', '--alias', 'gemma-4-12b-it-Q4_K_M')
$gemmaArgs = @('-m', 'pando.stemd', '--node-id', 'win-gemma-12b', '--backend',
  'openai', '--openai-url', 'http://127.0.0.1:8086/v1', '--openai-model',
  'gemma-4-12b', '--openai-key-file', $dummyKey, '--checkpoint',
  'gemma-4-12b-q4km', '--root', 'http://127.0.0.1:8790',
  '--pairing-token-file', $token, '--host', '0.0.0.0', '--advertise-host',
  '192.168.31.100')
$layaArgs = @('-m', 'pando.stemd', '--node-id', 'win-laya-int8', '--backend',
  'onnx', '--onnx-path', 'D:\PandoSI\artifacts\laya-en.int8.onnx',
  '--checkpoint', 'laya-en', '--root', 'http://127.0.0.1:8790',
  '--pairing-token-file', $token, '--host', '0.0.0.0', '--advertise-host',
  '192.168.31.100')
$workerArgs = @("$repo\tools\win_worker.py")
$mailboxArgs = @("$repo\tools\agent_mailbox.py", 'serve', '--port', '8778',
  '--dir', 'D:\pando-models\mailbox-win',
  '--token-file', "$repo\metrics\lan-check\mac-mailbox.token",
  '--grant-file', "$stateDir\grants.toml",
  '--code-repo', "$repo",
  '--code-cache', 'D:\pando-models\code.bundle')
# The resident proposer: the Windows node must initiate on its own, not only
# answer polled work. Multi-mailbox: publish to both, read the union, so a
# sleeping Mac-hosted mailbox can no longer silence the colony (2026-10-17).
$thinkArgs = @('-m', 'pando.thinkd', '--node-id', 'win-gpu-proposer',
  '--base-url', 'http://127.0.0.1:8086/v1', '--model', 'gemma-4-12b-it-Q4_K_M',
  '--fallback-base-url', 'http://127.0.0.1:8087/v1',
  '--fallback-model', 'yi-1.5-9b-q4km',
  '--mailbox', 'http://100.92.184.4:8776',
  '--mailbox-fallback', 'http://127.0.0.1:8778',
  '--token-file', "$repo\metrics\lan-check\mac-mailbox.token",
  '--ledger', 'D:\pando-models\win-gpu-think-ledger.jsonl',
  '--interval', '120', '--explore-every', '5', '--activity-every', '4')
# Yi-1.5-9B on CPU: a second, different-bloodline local model. It votes as a
# fabric member and runs its own proposer so the win node initiates with two
# independent minds, not one.
$yiLlamaArgs = @('-m', 'D:\pando-models\Yi-1.5-9B-Chat-Q4_K_M.gguf',
  '-ngl', '0', '-c', '1024', '--parallel', '1', '--jinja',
  '--host', '127.0.0.1', '--port', '8087', '-t', '10',
  '--alias', 'yi-1.5-9b-q4km')
$yiStemArgs = @('-m', 'pando.stemd', '--node-id', 'win-yi-9b', '--backend',
  'openai', '--openai-url', 'http://127.0.0.1:8087/v1', '--openai-model',
  'yi-1.5-9b-q4km', '--openai-key-file', $dummyKey, '--checkpoint',
  'yi-1.5-9b-q4km', '--root', 'http://127.0.0.1:8790',
  '--pairing-token-file', $token, '--host', '0.0.0.0', '--advertise-host',
  '192.168.31.100')
$yiThinkArgs = @('-m', 'pando.thinkd', '--node-id', 'win-cpu-proposer',
  '--base-url', 'http://127.0.0.1:8087/v1', '--model', 'yi-1.5-9b-q4km',
  '--mailbox', 'http://100.92.184.4:8776',
  '--mailbox-fallback', 'http://127.0.0.1:8778',
  '--token-file', "$repo\metrics\lan-check\mac-mailbox.token",
  '--ledger', 'D:\pando-models\win-cpu-think-ledger.jsonl',
  '--interval', '240', '--explore-every', '5', '--activity-every', '8',
  '--history-window', '4')
$observatoryArgs = @("$repo\tools\observatory.py", '--port', '8799')
# Speaks only for openai stems registered on THIS root. Gemma and Yi answer
# as win-gemma-12b and win-yi-9b once their stems expose /chat. It does not
# answer for Pi or Mac members.
$chatArgs = @('-m', 'pando.chatd',
  '--root', 'http://127.0.0.1:8790',
  '--pairing-token-file', $token,
  '--mailbox', 'http://100.92.184.4:8776',
  '--mailbox-fallback', 'http://127.0.0.1:8778',
  '--token-file', "$repo\metrics\lan-check\mac-mailbox.token",
  '--ledger', 'D:\pando-models\chat-ledger.jsonl',
  '--interval', '20')
# Asks only the decision models registered on THIS root for 赞同/搁置/反对,
# so every proposal collects one reaction per local role. It never calls a
# member that lives on another device.
$reactArgs = @('-m', 'pando.reactd',
  '--root', 'http://127.0.0.1:8790',
  '--pairing-token-file', $token,
  '--mailbox', 'http://100.92.184.4:8776',
  '--mailbox-fallback', 'http://127.0.0.1:8778',
  '--token-file', "$repo\metrics\lan-check\mac-mailbox.token",
  '--ledger', 'D:\pando-models\reaction-ledger.jsonl',
  '--interval', '40')
# The island root runs under the same logon supervisor as everything else:
# before this entry it was a hand-started process that would not survive a
# reboot (found 2026-10-18 when auditing auto-start paths).
$rootArgs = @('-m', 'pando.rootd', '--port', '8790',
  '--pairing-token-file', "$repo\artifacts\lan-experiment\root-pairing.token",
  '--lan', '--require-identity', '--quorum-k', '3',
  '--stem-call-timeout', '120', '--initial-stem-timeout', '30')
# The executor reads the bus; unlike its three neighbours above it never
# posts. As written it had no mailbox argument at all, so what it read
# depended on a hosts.json this file never names -- and with no such file it
# resolves to the Mac's localhost convention, 8776, on a box whose own mailbox
# is 8778. Same two shelves and the same order as the residents above it.
$executorArgs = @("$repo\tools\execute_loop.py", '300',
  '--repo', "$repo",
  '--state-dir', "$stateDir",
  '--token-file', "$repo\metrics\lan-check\mac-mailbox.token",
  '--mailbox', 'http://100.92.184.4:8776',
  '--mailbox-fallback', 'http://127.0.0.1:8778',
  '--grant-url', 'http://100.92.184.4:8776/operator-grant',
  '--grant-url', 'http://192.168.31.41:8776/operator-grant')
# The receive side of seed.replicate. The Mac executor stages sealed bundles
# in its outbox; this process pulls and verifies them into a local inbox. It
# never decrypts a seed, starts a model, or accepts a non-sealed bundle.
$seedPullArgs = @("$repo\tools\seed_pull.py", '--from',
  'http://100.92.184.4:8478',
  '--from-file', 'D:\pando-models\seed-sources.txt',
  '--token-file', "$repo\metrics\lan-check\mac-mailbox.token",
  '--inbox', 'D:\pando-models\seed-inbox',
  '--ledger', 'D:\pando-models\seed-receive-ledger.jsonl',
  '--node-id', 'win')
# The missing local half of "who updates the code". The node has no cloud
# operator any more, so nothing would fetch a pushed commit. This process
# fast-forwards the checkout itself -- GitHub first, the Mac's /code.bundle
# peer when GitHub is unreachable -- then stops the client processes so the
# supervisor respawns them from the new tree. A model is never reloaded by a
# plain fast-forward; the full restart is used only when this roster file
# itself changed, because a running supervisor cannot learn a new service.
$codeWatchArgs = @("$repo\tools\node_code_watch.py", '--repo', "$repo",
  '--remote', 'origin',
  '--peer', 'http://100.92.184.4:8478/code.bundle',
  '--peer-file', 'D:\pando-models\code-sources.txt',
  '--peer-token-file', "$repo\metrics\lan-check\mac-mailbox.token",
  '--on-update', 'clients',
  # Start-Process joins ArgumentList with spaces; a command value with
  # spaces must carry its own quotes or argparse receives five extra
  # tokens (node_code_watch crash-looped on this, 2026-10-19 12:09).
  '--client-cmd', "`"powershell -NoProfile -ExecutionPolicy Bypass -File $repo\tools\win_restart_clients.ps1`"",
  '--node-cmd', "`"powershell -NoProfile -ExecutionPolicy Bypass -File $repo\tools\win_stack_restart.ps1`"",
  '--supervisor', "$repo\tools\win_node_supervisor.ps1",
  '--ledger', 'D:\pando-models\code-sync-ledger.jsonl',
  # Push evidence that this node, rather than only the Mac's copy of a bundle,
  # observed a HEAD. The first valid key is pinned by the receiver; the path
  # is the same token-protected mailbox route the other residents use.
  '--receipt-url', 'http://100.92.184.4:8776/receipts/code/win',
  '--receipt-token-file', "$repo\metrics\lan-check\mac-mailbox.token",
  '--receipt-identity', "$env:USERPROFILE\.pando\identity.json",
  '--receipt-node', 'win',
  '--interval', '300')

# Liveness is not progress. The loop below answers "did the child exit?"; it
# cannot answer "did the proposer stop thinking?". A wedged thinkd stays alive
# with a frozen ledger, and with no cloud operator left on this node nothing
# else would notice. Every 60s, ask tools/think_watchdog.py whether each ledger
# has grown; a stall stops that one process and the loop respawns it. No
# scheduler is involved -- this is the loop that is already running.
$thinkWatch = @(
  @{ ledger = 'D:\pando-models\win-gpu-think-ledger.jsonl'; stall = 900
     cooldown = 'D:\pando-models\think-watchdog-gpu.cooldown'
     match = 'win-gpu-proposer' },
  @{ ledger = 'D:\pando-models\win-cpu-think-ledger.jsonl'; stall = 1200
     cooldown = 'D:\pando-models\think-watchdog-cpu.cooldown'
     match = 'win-cpu-proposer' }
)
$watchdog = "$repo\tools\think_watchdog.py"

$names = @('llama-3070', 'stem-gemma', 'stem-laya', 'worker', 'win-mailbox',
  'llama-yi-cpu', 'stem-yi', 'think-proposer', 'think-yi', 'observatory', 'chat',
  'react', 'executor-loop', 'rootd', 'seed-pull', 'code-watch')
$exes = @($llamaExe, $python, $python, $python, $python, $llamaExe, $python,
  $python, $python, $python, $python, $python, $python, $python, $python,
  $python)
$argsList = @($llamaArgs, $gemmaArgs, $layaArgs, $workerArgs, $mailboxArgs,
  $yiLlamaArgs, $yiStemArgs, $thinkArgs, $yiThinkArgs, $observatoryArgs, $chatArgs,
  $reactArgs, $executorArgs, $rootArgs, $seedPullArgs, $codeWatchArgs)
$cwds = @('D:\pando-models\llama-cuda', $repo, $repo, $repo, $repo,
  'D:\pando-models\llama-cuda', $repo, $repo, $repo, $repo, $repo, $repo, $repo,
  $repo, $repo, $repo)

$procs = @{}
$loops = 0
while ($true) {
  $loops++
  for ($i = 0; $i -lt $names.Count; $i++) {
    $n = $names[$i]
    $p = $procs[$n]
    if ($null -eq $p -or $p.HasExited) {
      $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
      Add-Content "$logs\supervisor.log" "[$stamp] (re)starting $n"
      $procs[$n] = Start-Process -FilePath $exes[$i] -ArgumentList $argsList[$i] `
        -WorkingDirectory $cwds[$i] `
        -RedirectStandardOutput "$logs\$n.out.log" `
        -RedirectStandardError "$logs\$n.err.log" `
        -WindowStyle Hidden -PassThru
    }
  }
  # Give a freshly started proposer a full round to write its first line
  # before the watchdog is allowed to judge it (5 min), then check every 60s.
  if ($loops -ge 30 -and ($loops % 6) -eq 0) {
    foreach ($w in $thinkWatch) {
      $wdArgs = @($watchdog, '--ledger', $w.ledger,
        '--stall-seconds', "$($w.stall)", '--cooldown-file', $w.cooldown,
        '--restart-cmd', "powershell -NoProfile -ExecutionPolicy Bypass -File $repo\tools\win_restart_think.ps1 -Match $($w.match)")
      & $python @wdArgs *>> "$logs\think-watchdog.log"
    }
  }
  Start-Sleep -Seconds 10
}
