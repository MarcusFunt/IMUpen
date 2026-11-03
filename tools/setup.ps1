param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $pythonExe = $pythonCommand.Source
    $pythonVersionLine = (& $pythonExe --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $pythonVersionLine -notmatch 'Python\s+(\d+\.\d+\.\d+)') {
        throw 'Could not read the Python version.'
    }
    $pythonVersionText = $Matches[1]
    $pythonVersion = [version]$pythonVersionText
    if ($pythonVersion -lt [version]'3.10') {
        throw "Python 3.10 or later is required; found $pythonVersionText."
    }

    $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        & $pythonExe -m venv (Join-Path $repoRoot '.venv')
        if ($LASTEXITCODE -ne 0) {
            throw 'Creating .venv failed.'
        }
    }
    & $venvPython -m pip install --requirement (Join-Path $repoRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) {
        throw 'Installing requirements.txt failed.'
    }
    $serialInfo = (& $venvPython -m pip show pyserial 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw 'pyserial is not installed in the isolated Python environment.'
    }
    if ($serialInfo -match '(?m)^Version:\s*(.+)$') {
        Write-Host "Project dependency: pyserial $($Matches[1])"
    }

    $compiler = Get-Command g++ -ErrorAction SilentlyContinue
    if ($null -eq $compiler) {
        Write-Warning 'g++ was not found. Install a C++17 compiler before running the native hardware-free checks.'
    } else {
        $compilerVersion = (& $compiler.Source --version | Select-Object -First 1)
        Write-Host "C++ compiler: $compilerVersion"
    }

    $arduinoCli = Get-Command arduino-cli -ErrorAction SilentlyContinue
    if ($null -eq $arduinoCli) {
        Write-Warning 'arduino-cli was not found. Install Arduino CLI and Espressif core 3.3.6 before compiling firmware.'
    } else {
        $cliVersion = (& $arduinoCli.Source version | Select-Object -First 1)
        Write-Host "Arduino CLI: $cliVersion"
        $coreList = (& $arduinoCli.Source core list 2>&1 | Out-String)
        if ($coreList -match '(?m)^esp32:esp32\s+3\.3\.6\b') {
            Write-Host 'Espressif Arduino core 3.3.6: installed'
        } else {
            Write-Warning 'Espressif Arduino core 3.3.6 is not installed. Install it with: arduino-cli core install esp32:esp32@3.3.6'
        }
    }

    Write-Host "Python environment ready: $venvPython"
    Write-Host 'Next: run tools/run-hardware-free.ps1 and tools/compile-firmware.ps1.'
} finally {
    Pop-Location
}
