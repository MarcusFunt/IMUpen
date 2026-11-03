param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $arduinoCli = Get-Command arduino-cli -ErrorAction Stop
    $coreList = (& $arduinoCli.Source core list 2>&1 | Out-String)
    if ($coreList -notmatch '(?m)^esp32:esp32\s+3\.3\.6\b') {
        throw 'Espressif Arduino core 3.3.6 is required. Install it with: arduino-cli core install esp32:esp32@3.3.6'
    }

    $buildDir = Join-Path $repoRoot 'build\firmware'
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    & $arduinoCli.Source compile `
        --fqbn 'esp32:esp32:XIAO_ESP32S3:USBMode=hwcdc,CDCOnBoot=default' `
        --warnings all `
        --build-path $buildDir `
        (Join-Path $repoRoot 'firmware\IMUpen')
    if ($LASTEXITCODE -ne 0) {
        throw 'XIAO ESP32-S3 firmware compilation failed.'
    }
    Write-Host 'Firmware compiled successfully. No board was uploaded.'
} finally {
    Pop-Location
}
