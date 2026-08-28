param(
    [Parameter(Mandatory = $true)] [int] $WaitForPid,
    [Parameter(Mandatory = $true)] [string] $SourceStatusFile,
    [Parameter(Mandatory = $true)] [string] $Plan,
    [Parameter(Mandatory = $true)] [string] $PauseFile,
    [Parameter(Mandatory = $true)] [string] $QueueStatusFile,
    [int[]] $BatchCandidates = @(1024, 896, 800, 768, 640, 512),
    [int] $EvaluationBatchSize = 512,
    [int] $EvaluationIntervalEpochs = 4
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repo ".venv-ml\Scripts\python.exe"
$trainer = Join-Path $repo "ml\train_feature_compressed_path_density.py"

function Resolve-RepoPath([string] $Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return $Value }
    return Join-Path $repo $Value
}

$sourceStatus = Resolve-RepoPath $SourceStatusFile
$planPath = Resolve-RepoPath $Plan
$pausePath = Resolve-RepoPath $PauseFile
$queueStatus = Resolve-RepoPath $QueueStatusFile
$queueDirectory = Split-Path -Parent $queueStatus
New-Item -ItemType Directory -Force -Path $queueDirectory | Out-Null

function Write-QueueStatus([string] $Stage, [hashtable] $Details = @{}) {
    $payload = @{
        stage = $Stage
        updatedAt = [DateTimeOffset]::Now.ToString("o")
        waitForPid = $WaitForPid
        sourceStatusFile = $SourceStatusFile
        plan = $Plan
    }
    foreach ($key in $Details.Keys) { $payload[$key] = $Details[$key] }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $queueStatus
}

try {
    Write-QueueStatus "waiting-for-source"
    $source = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($null -ne $source) { $source | Wait-Process }

    if (-not (Test-Path -LiteralPath $sourceStatus)) {
        throw "source status does not exist: $sourceStatus"
    }
    $sourceState = Get-Content -Raw -LiteralPath $sourceStatus | ConvertFrom-Json
    if ($sourceState.stage -ne "complete") {
        throw "source run ended with stage '$($sourceState.stage)'"
    }

    $selectedBatch = $null
    foreach ($candidate in $BatchCandidates) {
        Write-QueueStatus "smoke-testing" @{ batchSize = $candidate }
        $smokeArguments = @(
            $trainer,
            "--plan", $planPath,
            "--pause-file", $pausePath,
            "--smoke-batches", "1",
            "--batch-size", "$candidate",
            "--evaluation-batch-size", "$EvaluationBatchSize",
            "--evaluation-interval-epochs", "$EvaluationIntervalEpochs",
            "--matmul-precision", "high",
            "--compile-mode", "default"
        )
        & $python @smokeArguments
        if ($LASTEXITCODE -eq 0) {
            $selectedBatch = $candidate
            break
        }
    }
    if ($null -eq $selectedBatch) {
        throw "no batch-size candidate completed the smoke test"
    }

    Write-QueueStatus "training" @{ batchSize = $selectedBatch }
    $trainingArguments = @(
        $trainer,
        "--plan", $planPath,
        "--pause-file", $pausePath,
        "--replace-smoke",
        "--batch-size", "$selectedBatch",
        "--evaluation-batch-size", "$EvaluationBatchSize",
        "--evaluation-interval-epochs", "$EvaluationIntervalEpochs",
        "--matmul-precision", "high",
        "--compile-mode", "default"
    )
    & $python @trainingArguments
    if ($LASTEXITCODE -ne 0) {
        throw "full training exited with code $LASTEXITCODE"
    }
    Write-QueueStatus "complete" @{ batchSize = $selectedBatch }
}
catch {
    Write-QueueStatus "failed" @{ error = $_.Exception.Message }
    throw
}
