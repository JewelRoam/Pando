# Restart the resident Windows stack: supervisor + all children.
# Safe to re-run: stops every matching process, then relaunches the
# supervisor hidden. Verification stays outside the script.
$repo = 'D:\PandoSI'
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
  Where-Object { $_.CommandLine -match 'win_node_supervisor' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'pando\.stemd|win_worker|agent_mailbox|pando\.thinkd|observatory\.py|pando\.chatd|pando\.reactd|execute_loop\.py' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 3
Start-Process -FilePath 'powershell.exe' `
  -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$repo\tools\win_node_supervisor.ps1" `
  -WorkingDirectory $repo -WindowStyle Hidden
Write-Output 'supervisor restarted'
