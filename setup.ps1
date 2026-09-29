param(
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Test-CompatiblePython {
    param([string]$Candidate)

    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate)) {
        return $false
    }

    try {
        $Version = & $Candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        return ($LASTEXITCODE -eq 0 -and $Version -in @("3.11", "3.12"))
    } catch {
        return $false
    }
}

function Find-CompatiblePython {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        $Resolved = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RequestedPath)
        if (Test-CompatiblePython $Resolved) {
            return $Resolved
        }
        throw "-PythonPath must point to Python 3.11 or 3.12: $Resolved"
    }

    $Candidates = [System.Collections.Generic.List[string]]::new()

    $PyLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        foreach ($Minor in @("3.12", "3.11")) {
            try {
                $Resolved = & $PyLauncher.Source "-$Minor" -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $Resolved) {
                    $Candidates.Add(($Resolved | Select-Object -Last 1))
                }
            } catch {
                # Continue with well-known locations when the launcher is stale.
            }
        }
    }

    if ($env:LOCALAPPDATA) {
        $Candidates.Add((Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"))
        $Candidates.Add((Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"))
    }
    $Candidates.Add("C:\Program Files\Python312\python.exe")
    $Candidates.Add("C:\Program Files\Python311\python.exe")

    $PythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        $Candidates.Add($PythonCommand.Source)
    }

    foreach ($Candidate in $Candidates) {
        if (Test-CompatiblePython $Candidate) {
            return $Candidate
        }
    }

    throw @"
Python 3.11 or 3.12 was not found.
Install Python from https://www.python.org/downloads/windows/ and enable
'Add python.exe to PATH', or rerun:
  .\setup.ps1 -PythonPath "C:\path\to\python.exe"
"@
}

Push-Location $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        $SystemPython = Find-CompatiblePython $PythonPath
        Write-Host "Creating .venv with $SystemPython"
        & $SystemPython -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
    } elseif (-not (Test-CompatiblePython $VenvPython)) {
        throw "Existing .venv is not Python 3.11/3.12. Remove .venv and run setup again."
    }

    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

    # A non-editable wheel avoids Windows legacy-codepage issues in Korean paths.
    & $VenvPython -m pip install --upgrade ".[test]"
    if ($LASTEXITCODE -ne 0) { throw "DrugPilot dependency installation failed." }

    if (-not (Test-Path -LiteralPath ".env")) {
        Copy-Item -LiteralPath ".env.example" -Destination ".env"
        Write-Host "Created .env in mock mode (no API key required)."
    }

    Write-Host "Setup complete. Run .\start_drugpilot.ps1"
} finally {
    Pop-Location
}
