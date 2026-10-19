# Stop the client processes of this node so the logon supervisor respawns them
# from the code that is now on disk. A model is never touched: reloading
# llama-server costs a minute and buys nothing after a fast-forward.
#
# This file is an *adapter*, not a second roster. Before 2026-10-19 it carried
# its own copy of the process pattern and a test compared the two strings for
# equality -- but the copy that actually runs is the one on the node, and no
# test can see that one, so the test could only ever confirm that the two
# copies were equal at the moment the repo was checked out. The list of
# clients, the refusal to touch a model, and the choice of service manager now
# live in exactly one place (tools/code_sync.py, CLIENT_UNITS), and this script
# is one line of delegation to it. Keep it that way.
$ErrorActionPreference = 'Continue'
$repo = 'D:\PandoSI'
$python = (Get-Command python).Source
$log = 'D:\pando-models\logs\restart-clients.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

$out = & $python "$repo\tools\code_sync.py" --restart-clients --platform windows 2>&1
$code = $LASTEXITCODE
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Content -LiteralPath $log -Value "[$stamp] exit=$code $out"
if ($code -ne 0) {
  Write-Output "code_sync --restart-clients failed ($code): $out"
  exit 1
}
Write-Output $out
