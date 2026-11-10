# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param([switch]$Test, [string]$DevRoot)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$rootFile = Join-Path $repository '.local-dev/dev-root.txt'
if (-not $DevRoot) {
    $DevRoot = if (Test-Path -LiteralPath $rootFile) { (Get-Content -LiteralPath $rootFile -Raw).Trim() } else { Join-Path $env:LOCALAPPDATA 'FlareDev' }
}
$localDev = [IO.Path]::GetFullPath($DevRoot)
$tools = Join-Path $localDev 'tools'
$downloads = Join-Path $localDev 'downloads'
$machine = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($machine -notin @('ARM64', 'AMD64')) { throw 'Flare development requires ARM64 or x64 Windows.' }
$rustHost = if ($machine -eq 'ARM64') { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
$cmakePlatform = if ($machine -eq 'ARM64') { 'arm64' } else { 'x86_64' }
New-Item -ItemType Directory -Force -Path $tools, $downloads, (Join-Path $repository '.local-dev') | Out-Null
[IO.File]::WriteAllText($rootFile, $localDev, [Text.UTF8Encoding]::new($false))
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Install-WingetPackage {
    param([string]$Id, [string[]]$Options = @())
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) { throw 'Install Microsoft App Installer (winget), then rerun setup.' }
    & winget.exe install --id $Id --exact --source winget --silent --accept-package-agreements --accept-source-agreements --disable-interactivity @Options
    if ($LASTEXITCODE -ne 0) { throw "winget could not install $Id (exit $LASTEXITCODE)." }
}

function Install-ZipTool {
    param([string]$Url, [string]$Archive, [string]$Destination, [string]$Executable)
    if (Test-Path -LiteralPath $Executable) { return }
    Write-Host "Downloading $Url"
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Archive
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
    if (-not (Test-Path -LiteralPath $Executable)) { throw "Tool extraction did not produce $Executable." }
}

$git = Get-Command git.exe -ErrorAction SilentlyContinue
if ($git) { Split-Path -Parent $git.Source | Set-Content -LiteralPath (Join-Path $repository '.local-dev/git-path.txt') }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$component = if ($machine -eq 'ARM64') { 'Microsoft.VisualStudio.Component.VC.Tools.ARM64' } else { 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' }
$requirements = @($component, 'Microsoft.VisualStudio.Component.VC.CMake.Project')
if ($machine -eq 'ARM64') { $requirements += 'Microsoft.VisualStudio.Component.VC.Llvm.Clang' }
$installation = if (Test-Path -LiteralPath $vswhere) { & $vswhere -latest -products '*' -requires $requirements -property installationPath }
if (-not $installation) {
    $components = @('Microsoft.VisualStudio.Workload.VCTools')
    if ($machine -eq 'ARM64') {
        $components += @('Microsoft.VisualStudio.Component.VC.Tools.ARM64', 'Microsoft.VisualStudio.Component.VC.Llvm.Clang', 'Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset')
    }
    $existing = if (Test-Path -LiteralPath $vswhere) { & $vswhere -latest -products '*' -property installationPath }
    if ($existing) {
        $arguments = @('modify', '--installPath', "`"$existing`"", '--quiet', '--norestart', '--includeRecommended')
        foreach ($item in $components) { $arguments += @('--add', $item) }
        $installer = Join-Path (Split-Path -Parent $vswhere) 'setup.exe'
        Write-Host 'Adding missing Visual Studio C++ components. Windows may request administrator consent.'
        $process = Start-Process -FilePath $installer -ArgumentList $arguments -Verb RunAs -WindowStyle Hidden -Wait -PassThru
        if ($process.ExitCode -notin @(0, 3010)) { throw "Visual Studio modification failed (exit $($process.ExitCode))." }
    } else {
        $override = '--quiet --wait --norestart --includeRecommended'
        foreach ($item in $components) { $override += " --add $item" }
        Write-Host 'Installing Visual Studio C++ Build Tools. Windows may request administrator consent.'
        Install-WingetPackage 'Microsoft.VisualStudio.2022.BuildTools' @('--override', $override)
    }
    $installation = & $vswhere -latest -products '*' -requires $requirements -property installationPath
    if (-not $installation) { throw 'Visual Studio did not register all required C++ components. Complete its installation and rerun setup.' }
}
if ($machine -eq 'AMD64' -and -not (Get-Command nasm.exe -ErrorAction SilentlyContinue) -and -not (Test-Path -LiteralPath (Join-Path $env:ProgramFiles 'NASM/nasm.exe'))) {
    Install-WingetPackage 'NASM.NASM'
}

$env:CARGO_HOME = Join-Path $tools 'cargo'
$env:RUSTUP_HOME = Join-Path $tools 'rustup'
$rustup = Join-Path $env:CARGO_HOME 'bin/rustup.exe'
if (-not (Test-Path -LiteralPath $rustup)) {
    $installer = Join-Path $downloads 'rustup-init.exe'
    Invoke-WebRequest -UseBasicParsing "https://static.rust-lang.org/rustup/dist/$rustHost/rustup-init.exe" -OutFile $installer
    & $installer -y --profile minimal --default-toolchain stable --default-host $rustHost --no-modify-path
    if ($LASTEXITCODE -ne 0) { throw 'Rust installation failed.' }
}
& $rustup component add rustfmt clippy
if ($LASTEXITCODE -ne 0) { throw 'Rust development component installation failed.' }
$cmakeRoot = Join-Path $tools 'cmake'
Install-ZipTool "https://github.com/Kitware/CMake/releases/download/v4.4.4/cmake-4.4.4-windows-$cmakePlatform.zip" (Join-Path $downloads 'cmake.zip') $cmakeRoot (Join-Path $cmakeRoot "cmake-4.4.4-windows-$cmakePlatform/bin/cmake.exe")
$protocRoot = Join-Path $tools 'protoc'
Install-ZipTool 'https://github.com/protocolbuffers/protobuf/releases/download/v36.2/protoc-36.2-win64.zip' (Join-Path $downloads 'protoc.zip') $protocRoot (Join-Path $protocRoot 'bin/protoc.exe')

$python = Join-Path $repository '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $candidates = @((Join-Path $env:LOCALAPPDATA 'Programs/Python/Python312/python.exe'))
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notmatch 'WindowsApps') { $candidates += $pythonCommand.Source }
    $basePython = $null
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            & $candidate -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'
            if ($LASTEXITCODE -eq 0) { $basePython = $candidate; break }
        }
    }
    if (-not $basePython) {
        Install-WingetPackage 'Python.Python.3.12' @('--scope', 'user', '--architecture', 'x64')
        $basePython = Join-Path $env:LOCALAPPDATA 'Programs/Python/Python312/python.exe'
        if (-not (Test-Path -LiteralPath $basePython)) { throw 'Locate CPython 3.12 and create .venv, then rerun setup.' }
    }
    & $basePython -m venv (Join-Path $repository '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python virtual environment creation failed.' }
}
Set-Location -LiteralPath $repository
& $python -m pip install --disable-pip-version-check -e '.[test]' huggingface-hub==1.33.0
if ($LASTEXITCODE -ne 0) { throw 'Python test dependency installation failed.' }
Copy-Item -LiteralPath $python -Destination (Join-Path $repository '.venv/Scripts/python3.exe') -Force
& $python tests/frontend/prepare_templates.py --output .local-dev/test-templates --hub-cache (Join-Path $localDev 'hub') --recipe gemma4 --recipe glm53 --recipe mimo_v2 --recipe nemotron_h --recipe qwen38 --recipe step5
if ($LASTEXITCODE -ne 0) { throw 'Frontend test fixture preparation failed.' }

$config = Join-Path $repository '.local-dev/config.toml'
if (-not (Test-Path -LiteralPath $config)) {
    $modelCache = (Join-Path $localDev 'models').Replace('\', '/')
    $hubCache = (Join-Path $localDev 'hub').Replace('\', '/')
    $text = @"
[server]
host = "127.0.0.1"
port = 8000

[defaults]
max_concurrency = 1

[memory]
reserve = "8G"

[paths]
core_cache = "$modelCache"
hub_cache = "$hubCache"
"@
    [IO.File]::WriteAllText($config, $text, [Text.UTF8Encoding]::new($false))
}
& (Join-Path $PSScriptRoot 'dev_windows.ps1') -Action Build
& (Join-Path $PSScriptRoot 'dev_windows.ps1') -Action Smoke
if ($Test) { & (Join-Path $PSScriptRoot 'dev_windows.ps1') -Action Test }
Write-Host 'Flare development is configured. Enter a shell with scripts/dev_windows.ps1.'
