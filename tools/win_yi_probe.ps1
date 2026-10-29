# One-shot probe: can Yi-1.5-9B Q4 load on CPU within current RAM?
# Leaves the server running on :8087 when healthy so a completion can be
# measured; callers kill it before the supervisor takes the port.
$repo = if ($env:PANDO_REPO) {
  [IO.Path]::GetFullPath($env:PANDO_REPO)
} else {
  Split-Path -Parent $PSScriptRoot
}
$models = if ($env:PANDO_MODELS_DIR) {
  [IO.Path]::GetFullPath($env:PANDO_MODELS_DIR)
} else {
  Join-Path $repo 'models'
}
$exe = Join-Path $models 'llama-cuda\llama-server.exe'
$arguments = @('-m', (Join-Path $models 'Yi-1.5-9B-Chat-Q4_K_M.gguf'), '-ngl', '0',
  '-c', '1024', '--parallel', '1', '--jinja', '--host', '127.0.0.1',
  '--port', '8087', '-t', '10', '--alias', 'yi-1.5-9b-q4km')
$p = Start-Process -FilePath $exe -ArgumentList $arguments `
  -WorkingDirectory (Join-Path $models 'llama-cuda') `
  -RedirectStandardOutput (Join-Path $models 'logs\yi-probe.out.log') `
  -RedirectStandardError (Join-Path $models 'logs\yi-probe.err.log') `
  -WindowStyle Hidden -PassThru
"pid $($p.Id)"
for ($i = 0; $i -lt 20; $i++) {
  Start-Sleep -Seconds 5
  if ($p.HasExited) {
    "exited code $($p.ExitCode)"
    Get-Content (Join-Path $models 'logs\yi-probe.err.log') -Tail 5
    exit 1
  }
  try {
    $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8087/health' -TimeoutSec 2
    "health: $($h.status)"
    exit 0
  } catch {
    "waiting $i..."
  }
}
exit 1
