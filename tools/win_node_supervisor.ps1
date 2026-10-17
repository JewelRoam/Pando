# Win node supervisor: keeps the 3070 LLM server, the two fabric members and
# the local worker alive on this Windows node. Started at logon by the
# Startup\PandoSI-Node.vbs wrapper (no admin required); restarts any child
# that exits. Written for Windows PowerShell 5.1 compatibility.
$ErrorActionPreference = 'Continue'
$env:HF_HUB_OFFLINE = '1'   # the ONNX member resolves repo files from local cache
$repo = 'D:\PandoSI'
$logs = 'D:\pando-models\logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$python = (Get-Command python).Source
$token = "$repo\artifacts\lan-experiment\root-pairing.token"
$dummyKey = 'D:\pando-models\llm-dummy.token'
if (-not (Test-Path $dummyKey)) {
  Set-Content -LiteralPath $dummyKey -Value 'x' -Encoding ascii
}

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
  '--token-file', "$repo\metrics\lan-check\mac-mailbox.token")
# The resident proposer: the Windows node must initiate on its own, not only
# answer polled work. Multi-mailbox: publish to both, read the union, so a
# sleeping Mac-hosted mailbox can no longer silence the colony (2026-10-17).
$thinkArgs = @('-m', 'pando.thinkd', '--node-id', 'win-gpu-proposer',
  '--base-url', 'http://127.0.0.1:8086/v1', '--model', 'gemma-4-12b-it-Q4_K_M',
  '--fallback-base-url', 'http://127.0.0.1:8087/v1',
  '--fallback-model', 'yi-1.5-9b-q4km',
  '--mailbox', 'http://192.168.31.41:8776',
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
  '--mailbox', 'http://192.168.31.41:8776',
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
  '--mailbox', 'http://192.168.31.41:8776',
  '--token-file', "$repo\metrics\lan-check\mac-mailbox.token",
  '--ledger', 'D:\pando-models\chat-ledger.jsonl',
  '--interval', '20')

$names = @('llama-3070', 'stem-gemma', 'stem-laya', 'worker', 'win-mailbox',
  'llama-yi-cpu', 'stem-yi', 'think-proposer', 'think-yi', 'observatory', 'chat')
$exes = @($llamaExe, $python, $python, $python, $python, $llamaExe, $python,
  $python, $python, $python, $python)
$argsList = @($llamaArgs, $gemmaArgs, $layaArgs, $workerArgs, $mailboxArgs,
  $yiLlamaArgs, $yiStemArgs, $thinkArgs, $yiThinkArgs, $observatoryArgs, $chatArgs)
$cwds = @('D:\pando-models\llama-cuda', $repo, $repo, $repo, $repo,
  'D:\pando-models\llama-cuda', $repo, $repo, $repo, $repo, $repo)

$procs = @{}
while ($true) {
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
  Start-Sleep -Seconds 10
}
