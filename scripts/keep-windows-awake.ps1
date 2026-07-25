param(
    [Parameter(Mandatory = $true)]
    [string]$HeartbeatFile,

    [ValidateRange(30, 3600)]
    [int]$StaleAfterSeconds = 180,

    [ValidateRange(1, 60)]
    [int]$PollSeconds = 10
)

$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class TradingExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint executionState);
}
"@

$continuous = [Convert]::ToUInt32("80000000", 16)
$systemRequired = [uint32]0x00000001
$awayModeRequired = [uint32]0x00000040
$requestedState = $continuous -bor $systemRequired -bor $awayModeRequired
$startedAt = [DateTime]::UtcNow

try {
    $result = [TradingExecutionState]::SetThreadExecutionState($requestedState)
    if ($result -eq 0) {
        throw "SetThreadExecutionState failed with Win32 error $([Runtime.InteropServices.Marshal]::GetLastWin32Error())."
    }

    [pscustomobject]@{
        event = "windows-sleep-inhibitor-ready"
        pid = $PID
        heartbeatFile = $HeartbeatFile
        staleAfterSeconds = $StaleAfterSeconds
        startedAt = $startedAt.ToString("o")
    } | ConvertTo-Json -Compress

    while ($true) {
        if (Test-Path -LiteralPath $HeartbeatFile) {
            $lastHeartbeat = [IO.File]::GetLastWriteTimeUtc($HeartbeatFile)
            if (([DateTime]::UtcNow - $lastHeartbeat).TotalSeconds -gt $StaleAfterSeconds) {
                break
            }
        }
        elseif (([DateTime]::UtcNow - $startedAt).TotalSeconds -gt $StaleAfterSeconds) {
            break
        }
        Start-Sleep -Seconds $PollSeconds
    }
}
finally {
    [void][TradingExecutionState]::SetThreadExecutionState($continuous)
    [pscustomobject]@{
        event = "windows-sleep-inhibitor-released"
        pid = $PID
        releasedAt = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json -Compress
}
