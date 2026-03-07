#!/usr/bin/env pwsh
# NeuroLang Installer for Windows
# Usage: irm https://raw.githubusercontent.com/ChahelPaatur/Hacakthon-2026---Hriday-and-Pratheek-and-Chahel/main/Nueral%20Network%20Langauge/scripts/install.ps1 | iex

$ErrorActionPreference = "Stop"

$Version = if ($env:NEUROLANG_VERSION) { $env:NEUROLANG_VERSION } else { "latest" }
$InstallDir = if ($env:NEUROLANG_INSTALL) { $env:NEUROLANG_INSTALL } else { "$env:LOCALAPPDATA\neurolang" }

Write-Host ""
Write-Host "  ╔═══════════════════════════════════════════╗" -ForegroundColor Yellow
Write-Host "  ║       NeuroLang Installer (Windows)       ║" -ForegroundColor Yellow
Write-Host "  ╚═══════════════════════════════════════════╝" -ForegroundColor Yellow
Write-Host ""

$Binary = "neurolang-win-x64.exe"

Write-Host "  Platform:  Windows x64" -ForegroundColor Green
Write-Host "  Install:   $InstallDir\neurolang.exe" -ForegroundColor Green
Write-Host ""

$BaseUrl = "https://github.com/ChahelPaatur/Hacakthon-2026---Hriday-and-Pratheek-and-Chahel/releases"
if ($Version -eq "latest") {
    $DownloadUrl = "$BaseUrl/latest/download/$Binary"
} else {
    $DownloadUrl = "$BaseUrl/download/$Version/$Binary"
}

# Create install directory
if (!(Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

Write-Host "  Downloading $Binary..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $DownloadUrl -OutFile "$InstallDir\neurolang.exe" -UseBasicParsing
} catch {
    Write-Host "  Error: Failed to download from $DownloadUrl" -ForegroundColor Red
    Write-Host "  Make sure a release exists at:" -ForegroundColor Red
    Write-Host "    $BaseUrl" -ForegroundColor Red
    exit 1
}

# Add to PATH if not already there
$CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($CurrentPath -notlike "*$InstallDir*") {
    Write-Host "  Adding to PATH..." -ForegroundColor Yellow
    [Environment]::SetEnvironmentVariable("Path", "$CurrentPath;$InstallDir", "User")
    $env:Path = "$env:Path;$InstallDir"
}

Write-Host ""
Write-Host "  ✓ NeuroLang installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  Restart your terminal, then try:" -ForegroundColor White
Write-Host "    neurolang --help" -ForegroundColor Yellow
Write-Host "    echo `"Predict species from iris`" | neurolang --run" -ForegroundColor Yellow
Write-Host ""
