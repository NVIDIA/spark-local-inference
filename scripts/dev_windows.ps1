# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param(
    [ValidateSet('Shell', 'Build', 'Test', 'Smoke')]
    [string]$Action = 'Shell'
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$rootFile = Join-Path $repository '.local-dev/dev-root.txt'
$localDev = if (Test-Path -LiteralPath $rootFile) { (Get-Content -LiteralPath $rootFile -Raw).Trim() } else { Join-Path $env:LOCALAPPDATA 'FlareDev' }
$tools = Join-Path $localDev 'tools'
$localConfig = Join-Path $repository '.local-dev/config.toml'
$machine = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($machine -notin @('ARM64', 'AMD64')) { throw 'Flare development requires ARM64 or x64 Windows.' }
$vcTarget = if ($machine -eq 'ARM64') { 'arm64' } else { 'x64' }
$cmakePlatform = if ($machine -eq 'ARM64') { 'arm64' } else { 'x86_64' }
$component = if ($machine -eq 'ARM64') { 'Microsoft.VisualStudio.Component.VC.Tools.ARM64' } else { 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Run scripts/setup_windows.ps1 to install the build tools.' }
$installation = & $vswhere -latest -products '*' -requires $component -property installationPath
if (-not $installation) { throw "No Visual Studio C++ installation for $vcTarget was found. Run scripts/setup_windows.ps1." }
$vcvars = Join-Path $installation 'VC/Auxiliary/Build/vcvarsall.bat'
$vcReady = $env:VSCMD_ARG_TGT_ARCH -eq $vcTarget -and
    ([string]$env:VSINSTALLDIR).TrimEnd('\') -eq $installation.TrimEnd('\') -and
    (Get-Command cl.exe -ErrorAction SilentlyContinue) -and
    (Get-Command rc.exe -ErrorAction SilentlyContinue)
if (-not $vcReady) {
    $environment = & $env:ComSpec /d /s /c "`"$vcvars`" $vcTarget >nul && set"
    if ($LASTEXITCODE -ne 0) { throw "Visual Studio environment setup failed for $vcTarget." }
    foreach ($line in $environment) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
}

$env:CARGO_HOME = Join-Path $tools 'cargo'
$env:RUSTUP_HOME = Join-Path $tools 'rustup'
$env:CARGO_TARGET_DIR = Join-Path $localDev 'target'
$env:CARGO_BUILD_JOBS = '4'
$env:CMAKE_GENERATOR = 'Ninja'
$env:PROTOC = Join-Path $tools 'protoc/bin/protoc.exe'
$bins = @(
    (Join-Path $env:CARGO_HOME 'bin'),
    (Join-Path $tools "cmake/cmake-4.4.4-windows-$cmakePlatform/bin"),
    (Join-Path $tools 'protoc/bin'),
    (Join-Path $installation 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja'),
    (Join-Path $env:ProgramFiles 'NASM')
)
foreach ($clang in @('ARM64/bin', 'x64/bin', 'bin')) {
    $candidate = Join-Path $installation "VC/Tools/Llvm/$clang"
    if (Test-Path -LiteralPath (Join-Path $candidate 'clang.exe')) { $bins += $candidate; break }
}
$python = Join-Path $repository '.venv/Scripts/python.exe'
if (Test-Path -LiteralPath $python) {
    $env:VIRTUAL_ENV = Join-Path $repository '.venv'
    $bins = @((Join-Path $env:VIRTUAL_ENV 'Scripts')) + $bins
}
$gitPathFile = Join-Path $repository '.local-dev/git-path.txt'
if (Test-Path -LiteralPath $gitPathFile) {
    $gitPath = (Get-Content -LiteralPath $gitPathFile -Raw).Trim()
    if (Test-Path -LiteralPath $gitPath) { $bins += $gitPath }
}
$pathEntries = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$env:PATH = (@($bins) + @($env:PATH -split ';') | Where-Object { $_ -and $pathEntries.Add($_.TrimEnd('\')) }) -join ';'
foreach ($command in @('cargo', 'rustc', 'cl', 'cmake', 'ninja', 'protoc')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "$command is missing. Run scripts/setup_windows.ps1." }
}
$rustHost = if ($machine -eq 'ARM64') { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
$compiler = & rustc -vV
if ($LASTEXITCODE -ne 0 -or "host: $rustHost" -notin $compiler) {
    throw "The Rust host must be $rustHost. Configure the native toolchain before building."
}
$templates = Join-Path $repository '.local-dev/test-templates'
if (Test-Path -LiteralPath $templates) { $env:FLARE_TEST_TEMPLATE_DIR = $templates }
Set-Location -LiteralPath $repository

$flareExecutable = Join-Path $env:CARGO_TARGET_DIR 'release/flare.exe'
$runFlare = {
    if (-not (Test-Path -LiteralPath $flareExecutable)) { throw 'Build Flare with scripts/dev_windows.ps1 -Action Build.' }
    & $flareExecutable --config $localConfig @args
    if ($LASTEXITCODE -ne 0) { throw "Flare exited with code $LASTEXITCODE." }
}.GetNewClosure()
Set-Item -Path Function:global:flare -Value $runFlare

function Invoke-Cargo {
    param([string[]]$CargoArguments)
    & cargo @CargoArguments
    if ($LASTEXITCODE -ne 0) { throw "Cargo exited with code $LASTEXITCODE." }
}

switch ($Action) {
    'Build' {
        Invoke-Cargo @('build', '--locked', '--release', '--manifest-path', 'flare-cli/Cargo.toml')
    }
    'Test' {
        if (-not (Test-Path -LiteralPath $templates)) { throw 'Run scripts/setup_windows.ps1 to prepare frontend test fixtures.' }
        $failed = @()
        foreach ($crate in @('eos', 'selene', 'helios', 'flare', 'flare-cli')) {
            $cargoArguments = @('test', '--locked', '--release', '--manifest-path', "$crate/Cargo.toml")
            if ($crate -eq 'helios') { $cargoArguments += @('--features', 'runtime-core,server') }
            & cargo @cargoArguments
            if ($LASTEXITCODE -ne 0) { $failed += $crate; Write-Warning "$crate tests failed (exit $LASTEXITCODE)." }
        }
        if ($failed.Count) { throw "Native test suites failed: $($failed -join ', ')." }
    }
    'Smoke' {
        foreach ($arguments in @(@('--version'), @('--help'), @('download', '--help'), @('models', 'list', '--json'))) {
            & $runFlare @arguments
        }
    }
    'Shell' {
        Write-Host "Flare development: $vcTarget MSVC, native Rust, short build directory $env:CARGO_TARGET_DIR"
        Write-Host 'Build/test: & ./scripts/dev_windows.ps1 -Action Build  (or Test)'
        Write-Host 'Prepared cores: flare download PUBLISHER/PREPARED_MODEL --gpus 0'
        Write-Host 'Serve: flare run PUBLISHER/PREPARED_MODEL --gpus 0'
    }
}
