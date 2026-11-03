param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

function Invoke-CompiledTest {
    param(
        [string]$Executable,
        [string]$RuntimeDirectory,
        [string]$FailureMessage
    )

    $previousPath = $env:PATH
    try {
        $env:PATH = "$RuntimeDirectory;$previousPath"
        & $Executable
        $exitCode = $LASTEXITCODE
    } finally {
        $env:PATH = $previousPath
    }
    if ($exitCode -ne 0) {
        throw $FailureMessage
    }
}

Push-Location $repoRoot
try {
    $pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        throw 'Project environment not found. Run tools/setup.ps1 first.'
    }
    $compiler = Get-Command g++ -ErrorAction Stop
    $buildDir = Join-Path $repoRoot 'build\hardware-free'
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

    Write-Host 'Running base Python regression tests'
    & $pythonExe -m unittest discover -s tests -p 'test_*.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Base Python tests failed.' }

    Write-Host 'Running virtual Python integration tests'
    & $pythonExe -m unittest discover -s tests/hardware_free -p 'test_*.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Virtual Python tests failed.' }

    $decoderExe = Join-Path $buildDir 'fifo_decoder_test.exe'
    Write-Host 'Building standalone FIFO decoder test'
    & $compiler.Source -std=c++17 -Wall -Wextra -O2 -Ifirmware/IMUpen tests/fifo_decoder_test.cpp '-o' $decoderExe
    if ($LASTEXITCODE -ne 0) { throw 'FIFO decoder test build failed.' }
    Invoke-CompiledTest -Executable $decoderExe -RuntimeDirectory (Split-Path -Parent $compiler.Source) -FailureMessage 'FIFO decoder test failed.'

    $harnessExe = Join-Path $buildDir 'virtual_imu_test.exe'
    Write-Host 'Building native firmware harness'
    & $compiler.Source -std=c++17 -Wall -Wextra -O2 -Itests/hardware_free/stubs tests/hardware_free/native_harness.cpp '-o' $harnessExe
    if ($LASTEXITCODE -ne 0) { throw 'Native harness build failed.' }
    Invoke-CompiledTest -Executable $harnessExe -RuntimeDirectory (Split-Path -Parent $compiler.Source) -FailureMessage 'Native firmware scenarios failed.'

    Write-Host 'Hardware-free checks completed. Review documented expected-defect cases before treating this as a software-readiness gate.'
} finally {
    Pop-Location
}
