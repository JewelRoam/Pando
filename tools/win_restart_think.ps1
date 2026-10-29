# Restart one proposer on the win node. Used by tools/think_watchdog.py when a
# ledger stops growing but the process is still alive.
#
# Why this exists at all: the logon supervisor answers "did the process exit?".
# It does not answer "did it stop thinking?". A proposer wedged on a socket that
# accepts and never replies stays running forever with a frozen ledger, and once
# no cloud operator is watching the win node nothing else would ever notice.
#
# This script does not start anything. It stops the wedged process and lets the
# supervisor respawn it from the same tree within 10 seconds -- one starter, so
# there is never a second copy racing the supervisor.
param(
  [string]$Match = 'pando\.proposer'
)
$ErrorActionPreference = 'Continue'
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match $Match -and $_.ProcessId -ne $PID } |
  ForEach-Object {
    Write-Output ("stopping {0} pid={1}" -f $Match, $_.ProcessId)
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }
Write-Output ("think restart requested: {0}" -f $Match)
