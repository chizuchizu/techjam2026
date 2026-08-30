param([int]$ProcessId = 34500, [int]$Samples = 5, [int]$IntervalSec = 30)
for ($i = 0; $i -lt $Samples; $i++) {
    $p = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($p) {
        $cpu = '{0:N1}' -f $p.CPU
        $mem = '{0:N0}' -f ($p.WorkingSet / 1MB)
        Write-Output "$(Get-Date -Format 'HH:mm:ss') | alive | CPU: ${cpu}s | Mem: ${mem} MB | Responding: $($p.Responding)"
    } else {
        Write-Output "$(Get-Date -Format 'HH:mm:ss') | EXITED - process $ProcessId no longer running"
        break
    }
    Start-Sleep -Seconds $IntervalSec
}
