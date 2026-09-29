param(
    [int]$Port = 8765,
    [switch]$NoBrowser,
    [switch]$ShareOnLan
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvRoot = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$Python = $VenvPython

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Python environment not found. Run .\setup.ps1 first."
}

# A Windows venv launcher can fail when its path contains non-ASCII
# characters.  Probe it without letting ErrorActionPreference=Stop abort the
# script before the documented base-interpreter fallback can run.
$VenvLauncherOk = $false
try {
    & $VenvPython --version *> $null
    $VenvLauncherOk = ($LASTEXITCODE -eq 0)
} catch {
    $VenvLauncherOk = $false
}

if (-not $VenvLauncherOk) {
    $VenvConfig = Join-Path $VenvRoot "pyvenv.cfg"
    $ExecutableLine = Get-Content -LiteralPath $VenvConfig -Encoding UTF8 |
        Where-Object { $_ -match '^executable\s*=' } |
        Select-Object -First 1
    if (-not $ExecutableLine) {
        throw "The virtual environment launcher failed and pyvenv.cfg has no base executable. Run .\setup.ps1 again."
    }
    $Python = ($ExecutableLine -replace '^executable\s*=\s*', '').Trim()
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "The Python executable recorded in pyvenv.cfg was not found: $Python"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".env"))) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot ".env.example") -Destination (Join-Path $ProjectRoot ".env")
    Write-Host "Created .env in mock mode (no API key required)."
}

$env:PYTHONUTF8 = "1"
$SourcePath = Join-Path $ProjectRoot "src"
$SitePackages = Join-Path $VenvRoot "Lib\site-packages"
$env:PYTHONPATH = "$SourcePath;$SitePackages"
# MA2 launches RDKit BRICS, Meeko and ADMET-AI as child Python processes.
# Use the exact interpreter that passed the startup probe (including the
# pyvenv.cfg base-interpreter fallback) instead of leaving a possibly broken
# copied venv launcher in Settings.
$env:DRUGPILOT_ADMET_PYTHON_EXECUTABLE = $Python
# This limits new DrugPilot live runs, not upstream OpenAI API calls. Keep the
# shared developer port conservative while allowing the LAN beta port its
# separately approved admission capacity on every scripted restart.
if ($Port -eq 8765) {
    $env:DRUGPILOT_BETA_MAX_NEW_RUNS_PER_HOUR = "3"
} elseif ($Port -eq 8766) {
    $env:DRUGPILOT_BETA_MAX_NEW_RUNS_PER_HOUR = "15"
    # Explicitly keep shared-service history separate from the development DB.
    $env:DRUGPILOT_DATABASE_PATH = Join-Path $ProjectRoot "data\drugpilot.sqlite3"
    $env:DRUGPILOT_BUILD_VARIANT = "8766-ma3-admission-postrun-20260929"
    # MA1 submits source-bound provisional hypotheses; only MA3-approved
    # hypotheses may enter Planner/MA2. Does not change either port's DB.
    $env:DRUGPILOT_MA1_PROVISIONAL_ADMISSION_ENABLED = "true"
}
$BindAddress = if ($ShareOnLan) { "0.0.0.0" } else { "127.0.0.1" }

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$Port"
}

if ($ShareOnLan) {
    Write-Warning "LAN sharing is enabled without application login. Use only on a trusted private network."
    $LanAddresses = [System.Net.Dns]::GetHostAddresses([System.Net.Dns]::GetHostName()) |
        Where-Object { $_.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork -and -not $_.IPAddressToString.StartsWith("127.") }
    foreach ($Address in $LanAddresses) {
        Write-Host "LAN URL: http://$($Address.IPAddressToString):$Port"
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "Starting DrugPilot LLM Intake Validation Test on http://$BindAddress`:$Port"
    & $Python -m uvicorn drugpilot.api.app:app --host $BindAddress --port $Port
} finally {
    Pop-Location
}
