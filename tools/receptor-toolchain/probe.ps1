$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $Root ".venv-receptor-tools\Scripts\python.exe"
$MeekoReceptor = Join-Path $Root ".venv-receptor-tools\Scripts\mk_prepare_receptor.exe"
$MeekoLigand = Join-Path $Root ".venv-receptor-tools\Scripts\mk_prepare_ligand.exe"
$RdkitRoot = Join-Path $Root ".venv-receptor-tools\Lib\site-packages\rdkit"
$OpenMMPlugins = Join-Path $Root ".venv-receptor-tools\Lib\site-packages\OpenMM.libs\lib\plugins"
$JavaHome = Join-Path $Root "tools\java\jdk-21.0.12+8-jre"
$P2Rank = Join-Path $Root "tools\p2rank\p2rank_2.5.1\prank.bat"
$Vina = Join-Path $Root "tools\bin\vina_1.2.7_win.exe"

foreach ($Path in @($Python, $MeekoReceptor, $MeekoLigand, $RdkitRoot, $OpenMMPlugins, $JavaHome, $P2Rank, $Vina)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required clone-local receptor tool is missing: $Path"
    }
}

Push-Location $Root
try {
    $env:PYTHONUTF8 = "1"
    $env:PYTHONPATH = Join-Path $Root "src"
    $env:JAVA_HOME = $JavaHome
    $env:RDBASE = [IO.Path]::GetRelativePath($Root, $RdkitRoot)
    $env:OPENMM_PLUGIN_DIR = [IO.Path]::GetRelativePath($Root, $OpenMMPlugins)

    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "pip check failed" }
    & $Python -c "from posebusters import PoseBusters; import openmm, pdbfixer, meeko, rdkit; print('python_imports=PASS'); print('openmm_platforms=' + ','.join(openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms()))); print('openmm_plugin_failures=' + repr(openmm.Platform.getPluginLoadFailures()))"
    if ($LASTEXITCODE -ne 0) { throw "Python receptor imports failed" }
    & $MeekoReceptor --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Meeko receptor CLI failed" }
    & $MeekoLigand --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Meeko ligand CLI failed" }
    & $P2Rank -v
    if ($LASTEXITCODE -ne 0) { throw "P2Rank version probe failed" }
    & $Vina --version
    if ($LASTEXITCODE -ne 0) { throw "Vina version probe failed" }

    Write-Output "SCIENTIFIC_EXECUTABLE_READINESS=PASS"
    Write-Output "PRODUCTION_PROFILE_ACCEPTANCE=NOT_RUN"
} finally {
    Pop-Location
}
