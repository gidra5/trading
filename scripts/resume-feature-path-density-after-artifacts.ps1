param(
    [Parameter(Mandatory = $true)] [string] $RequiredReport,
    [Parameter(Mandatory = $true)] [string] $RequiredModel,
    [Parameter(Mandatory = $true)] [DateTimeOffset] $NotBefore,
    [Parameter(Mandatory = $true)] [string] $Plan,
    [Parameter(Mandatory = $true)] [string] $PauseFile,
    [Parameter(Mandatory = $true)] [string] $QueueStatusFile,
    [int] $BatchSize = 1024,
    [int] $EvaluationBatchSize = 1024,
    [int] $EvaluationIntervalEpochs = 4,
    [int] $PollSeconds = 15
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repo ".venv-ml\Scripts\python.exe"
$trainer = Join-Path $repo "ml\train_feature_compressed_path_density.py"

function Resolve-RepoPath([string] $Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return $Value }
    return Join-Path $repo $Value
}

$requiredFiles = @($RequiredReport, $RequiredModel)
$required = @($requiredFiles | ForEach-Object { Resolve-RepoPath $_ })
$planPath = Resolve-RepoPath $Plan
$pausePath = Resolve-RepoPath $PauseFile
$queueStatus = Resolve-RepoPath $QueueStatusFile
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $queueStatus) |
    Out-Null

function Write-QueueStatus([string] $Stage, [hashtable] $Details = @{}) {
    $payload = @{
        stage = $Stage
        updatedAt = [DateTimeOffset]::Now.ToString("o")
        plan = $Plan
        requiredFiles = $requiredFiles
        artifactsNotBefore = $NotBefore.ToString("o")
        batchSize = $BatchSize
        evaluationBatchSize = $EvaluationBatchSize
        evaluationIntervalEpochs = $EvaluationIntervalEpochs
        compiledEvaluation = $true
    }
    foreach ($key in $Details.Keys) { $payload[$key] = $Details[$key] }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $queueStatus
}

try {
    Write-QueueStatus "waiting-for-feature-search"
    while ($true) {
        $ready = $true
        foreach ($file in $required) {
            $item = Get-Item -LiteralPath $file -ErrorAction SilentlyContinue
            if ($null -eq $item -or $item.Length -le 0 -or
                    [DateTimeOffset]$item.LastWriteTime -lt $NotBefore) {
                $ready = $false
                break
            }
        }
        if ($ready) {
            # The first artifact is the search report and must be complete JSON.
            Get-Content -Raw -LiteralPath $required[0] | ConvertFrom-Json |
                Out-Null
            break
        }
        Start-Sleep -Seconds $PollSeconds
    }

    if (Test-Path -LiteralPath $pausePath) {
        Remove-Item -LiteralPath $pausePath -Force
    }
    Write-QueueStatus "training" @{ resumedAfterFeatureSearch = $true }
    $arguments = @(
        $trainer,
        "--plan", $planPath,
        "--pause-file", $pausePath,
        "--batch-size", "$BatchSize",
        "--evaluation-batch-size", "$EvaluationBatchSize",
        "--evaluation-interval-epochs", "$EvaluationIntervalEpochs",
        "--matmul-precision", "high",
        "--compile-mode", "default"
    )
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "resumed training exited with code $LASTEXITCODE"
    }
    Write-QueueStatus "complete" @{ resumedAfterFeatureSearch = $true }
}
catch {
    Write-QueueStatus "failed" @{ error = $_.Exception.Message }
    throw
}
