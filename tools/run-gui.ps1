param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Project environment not found. Run tools/setup.ps1 first.'
}

$null = & $pythonExe -c "import PySide6.QtWidgets" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'PySide6 is missing. Install it with: .\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt'
}

Push-Location $repoRoot
try {
    & $pythonExe ui/imupen_studio.py
    if ($LASTEXITCODE -ne 0) { throw 'IMUpen Studio exited with an error.' }
} finally {
    Pop-Location
}
