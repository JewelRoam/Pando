# Restart only the client processes of the win node. A model is not touched.
#
# The logon supervisor (win_node_supervisor.ps1) respawns any child that has
# exited within 10 seconds, so this is how a fast-forward replaces the code a
# resident client is running without reloading gemma or yi: stop the client,
# let the supervisor start it again from the new tree.
#
# llama-server.exe is deliberately not matched -- a model reload costs a minute
# and buys nothing here -- and neither is the supervisor, which has to survive
# to do the respawning. Everything python is stopped together (stems and rootd
# included), because a half-registered registry is worse than a rebuilt one.
$ErrorActionPreference = 'Continue'
$pattern = 'pando\.stemd|win_worker|agent_mailbox|pando\.thinkd|observatory\.py|pando\.chatd|pando\.reactd|execute_loop\.py|pando\.rootd|seed_pull\.py|node_code_watch\.py'
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match $pattern -and $_.ProcessId -ne $PID } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Output 'clients restarted'
