param(
    [int]$BatchSize = 4,
    [int]$NumWorkers = 0,
    [string]$TrainDatasetRoot = "",
    [string]$TestDatasetRoot = "",
    [string]$OrientationTrainDatasetRoot = "",
    [string]$OrientationTestDatasetRoot = "",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $Candidates = [System.Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $Candidates.Add((Join-Path $env:CONDA_PREFIX "python.exe"))
    }

    $CondaEnvironmentFile = Join-Path $env:USERPROFILE ".conda\environments.txt"
    if (Test-Path -LiteralPath $CondaEnvironmentFile -PathType Leaf) {
        foreach ($EnvironmentRoot in (Get-Content -LiteralPath $CondaEnvironmentFile)) {
            if ((Split-Path -Leaf $EnvironmentRoot) -eq "pytorch-2.7.1-gpu") {
                $Candidates.Add((Join-Path $EnvironmentRoot "python.exe"))
            }
        }
    }

    $CommandPython = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $CommandPython) {
        $Candidates.Add($CommandPython.Source)
    }

    foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            & $Candidate -c "import torch" *> $null
            if ($LASTEXITCODE -eq 0) {
                $PythonExe = $Candidate
                break
            }
        }
    }
}

if ([string]::IsNullOrWhiteSpace($PythonExe) -or
    -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "A Python environment containing PyTorch was not found. Pass -PythonExe explicitly."
}

Push-Location $ProjectRoot
try {
    $DirectionSplitArgs = @("data_operate\make_resnet18_multiboard_direction_unified_5fold.py")
    if (-not [string]::IsNullOrWhiteSpace($TrainDatasetRoot)) {
        $DirectionSplitArgs += @("--train-dataset-root", $TrainDatasetRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace($TestDatasetRoot)) {
        $DirectionSplitArgs += @("--test-dataset-root", $TestDatasetRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace($OrientationTrainDatasetRoot)) {
        $DirectionSplitArgs += @("--orientation-train-dataset-root", $OrientationTrainDatasetRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace($OrientationTestDatasetRoot)) {
        $DirectionSplitArgs += @("--orientation-test-dataset-root", $OrientationTestDatasetRoot)
    }
    & $PythonExe @DirectionSplitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Direction-unified split creation failed with exit code $LASTEXITCODE"
    }

    $TestArgs = @(
        "test\test_resnet18_multiboard_direction_unified_3boards.py",
        "--batch-size", $BatchSize,
        "--num-workers", $NumWorkers
    )
    if (-not [string]::IsNullOrWhiteSpace($TestDatasetRoot)) {
        $TestArgs += @("--test-dataset-root", $TestDatasetRoot)
    }
    & $PythonExe @TestArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Locked test failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
