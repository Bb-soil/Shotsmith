# Shotsmith PowerShell command entry point (legacy AVS alias).
# Works in Windows PowerShell 5.1 and PowerShell 7. No secrets in argv.
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CommandArgs
)

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot

if ($null -eq $CommandArgs) {
    $CommandArgs = @()
}
if ($CommandArgs.Count -eq 0) {
    $Module = 'scripts.console'
} elseif ($CommandArgs[0] -eq 'shell') {
    $Module = 'scripts.console'
    $CommandArgs = @()
} elseif ($CommandArgs[0] -eq 'worker') {
    $Module = 'scripts.i1'
    if ($CommandArgs.Count -gt 1) {
        $CommandArgs = $CommandArgs[1..($CommandArgs.Count - 1)]
    } else {
        $CommandArgs = @()
    }
} elseif ($CommandArgs[0] -eq 'video') {
    $Module = 'scripts.video'
    if ($CommandArgs.Count -gt 1) { $CommandArgs = $CommandArgs[1..($CommandArgs.Count - 1)] } else { $CommandArgs = @() }
} elseif ($CommandArgs.Count -gt 0 -and $CommandArgs[0] -eq 'keyframes') {
    $Module = 'scripts.kf1'
    if ($CommandArgs.Count -gt 1) {
        $CommandArgs = $CommandArgs[1..($CommandArgs.Count - 1)]
    } else {
        $CommandArgs = @()
    }
} elseif ($CommandArgs.Count -gt 0 -and $CommandArgs[0] -eq 'project') {
    $Module = 'scripts.kf2'
    if ($CommandArgs.Count -gt 1) {
        $CommandArgs = $CommandArgs[1..($CommandArgs.Count - 1)]
    } else {
        $CommandArgs = @()
    }
} else {
    $Module = 'scripts.i1'
}

$Python = $null
if ($env:AVS_PYTHON) {
    $Python = $env:AVS_PYTHON
} else {
    $VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $VenvPython) {
        $Python = $VenvPython
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $Python = 'python'
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $Python = 'py'
    }
}

if (-not $Python) {
    [Console]::Error.WriteLine('Python не найден. Установите Python 3.12 или задайте AVS_PYTHON.')
    exit 2
}

$oldEncoding = $env:PYTHONIOENCODING
$env:PYTHONIOENCODING = 'utf-8'
try {
    Push-Location -LiteralPath $RepoRoot
    try {
        $ShotsmithExe = Join-Path $RepoRoot 'data\app\shotsmith\Shotsmith.exe'
        $UpdatedShotsmithExe = Join-Path $RepoRoot 'data\app\shotsmith-focusfix\Shotsmith.exe'
        if (Test-Path -LiteralPath $UpdatedShotsmithExe) { $ShotsmithExe = $UpdatedShotsmithExe }
        $CompleteShotsmithExe = Join-Path $RepoRoot 'data\app\shotsmith-complete\Shotsmith.exe'
        if (Test-Path -LiteralPath $CompleteShotsmithExe) { $ShotsmithExe = $CompleteShotsmithExe }
        if ($Module -eq 'scripts.console' -and $env:SHOTSMITH_PLAIN -ne '1' -and (Test-Path -LiteralPath $ShotsmithExe)) {
            & $ShotsmithExe --repo $RepoRoot --python $Python
        } elseif ($Python -eq 'py') {
            & py -3.12 -m $Module @CommandArgs
        } else {
            & $Python -m $Module @CommandArgs
        }
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    if ($null -eq $oldEncoding) {
        Remove-Item Env:\PYTHONIOENCODING -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONIOENCODING = $oldEncoding
    }
}
exit $code
