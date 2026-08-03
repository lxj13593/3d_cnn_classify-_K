param(
    [string]$Folds = "all",
    [int]$Epochs = 50,
    [int]$BatchSize = 4,
    [int]$NumWorkers = 0,
    [string]$DatasetRoot = "",
    [string]$PythonExe = "",
    [switch]$AllowOverwrite
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
        $EnvironmentRoots = Get-Content -LiteralPath $CondaEnvironmentFile
        foreach ($EnvironmentRoot in $EnvironmentRoots) {
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
    $SplitArgs = @("data_operate\make_multiboard_clear_5fold.py")
    if (-not [string]::IsNullOrWhiteSpace($DatasetRoot)) {
        $SplitArgs += @("--dataset-root", $DatasetRoot)
    }
    & $PythonExe @SplitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "5-fold split creation failed with exit code $LASTEXITCODE"
    }

    $TrainArgs = @(
        "train_val\main_resnet18_multiboard_clear_5fold.py",
        "--folds", $Folds,
        "--epochs", $Epochs,
        "--batch-size", $BatchSize,
        "--num-workers", $NumWorkers
    )
    if ($AllowOverwrite) {
        $TrainArgs += "--allow-overwrite"
    }
    if (-not [string]::IsNullOrWhiteSpace($DatasetRoot)) {
        $TrainArgs += @("--dataset-root", $DatasetRoot)
    }

    & $PythonExe @TrainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
