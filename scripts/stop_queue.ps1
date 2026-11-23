# Detiene la cola de experimentos y todos sus procesos hijos (Windows).
# Uso: powershell -NoProfile -File scripts/stop_queue.ps1
$patterns = @('scripts/run_all.sh', 'src/train.py', 'src/evaluate.py', 'multiprocessing.spawn')
Get-CimInstance Win32_Process |
  Where-Object {
    $cmd = $_.CommandLine
    $_.ProcessId -ne $PID -and $cmd -and ($patterns | Where-Object { $cmd -like "*$_*" })
  } |
  ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>$null | Out-Null }
