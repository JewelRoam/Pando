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
  '-c', '2048', '--jinja', '--reasoning', 'off', '--host', '0.0.0.0',
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

$names = @('llama-3070', 'stem-gemma', 'stem-laya', 'worker')
$exes = @($llamaExe, $python, $python, $python)
$argsList = @($llamaArgs, $gemmaArgs, $layaArgs, $workerArgs)
$cwds = @('D:\pando-models\llama-cuda', $repo, $repo, $repo)

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
