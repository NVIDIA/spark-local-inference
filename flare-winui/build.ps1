# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [ValidateSet('ARM64', 'x64')][string]$Platform = $(if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'ARM64' } else { 'x64' }),
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Release',
    [switch]$SkipRust,
    [switch]$SingleFile
)
$ErrorActionPreference = 'Stop'
if ($SingleFile -and $Configuration -ne 'Release') { throw 'Single-file distribution requires a Release build.' }
$repository = Split-Path -Parent $PSScriptRoot
$sharedCargoTarget = $env:CARGO_TARGET_DIR
Push-Location -LiteralPath $repository
try {
    . (Join-Path $repository 'scripts/dev_windows.ps1') -Action Shell
    $sharedCargoTarget = $env:CARGO_TARGET_DIR
    $env:CARGO_TARGET_DIR = Join-Path $localDev 'winui-target'
    $nativeTools = Join-Path $repository '.local-dev/native-tools'
    $packages = Join-Path $repository '.local-dev/native-packages'
    New-Item -ItemType Directory -Force -Path $nativeTools | Out-Null
    $nuget = Join-Path $nativeTools 'nuget.exe'
    if (-not (Test-Path -LiteralPath $nuget)) {
        Invoke-WebRequest 'https://dist.nuget.org/win-x86-commandline/v6.14.0/nuget.exe' -OutFile $nuget
    }
    & $nuget restore (Join-Path $PSScriptRoot 'packages.config') -PackagesDirectory $packages -Source 'https://api.nuget.org/v3/index.json' -NonInteractive
    if ($LASTEXITCODE -ne 0) { throw "NuGet restore failed with exit code $LASTEXITCODE." }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $component = if ($Platform -eq 'ARM64') { 'Microsoft.VisualStudio.Component.VC.Tools.ARM64' } else { 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' }
    $installation = & $vswhere -latest -products '*' -requires $component -property installationPath
    if (-not $installation) { throw "Visual Studio 2022 C++ build tools for $Platform are required." }
    $hostArm = $env:PROCESSOR_ARCHITECTURE -eq 'ARM64'
    $vcArchitecture = if ($hostArm) { if ($Platform -eq 'ARM64') { 'arm64' } else { 'arm64_x64' } } else { if ($Platform -eq 'ARM64') { 'x64_arm64' } else { 'x64' } }
    $vcvars = Join-Path $installation 'VC/Auxiliary/Build/vcvarsall.bat'
    if ($env:VSCMD_ARG_TGT_ARCH -ne $Platform -or ([string]$env:VSINSTALLDIR).TrimEnd('\') -ne $installation.TrimEnd('\')) {
        $environment = & $env:ComSpec /d /s /c "`"$vcvars`" $vcArchitecture >nul && set"
        if ($LASTEXITCODE -ne 0) { throw "Visual Studio environment setup failed for $Platform." }
        foreach ($line in $environment) {
            if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
        }
    }
    $target = if ($Platform -eq 'ARM64') { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
    $profile = if ($Configuration -eq 'Release') { 'release' } else { 'debug' }
    $cross = ($hostArm -and $Platform -eq 'x64') -or (-not $hostArm -and $Platform -eq 'ARM64')
    $bridge = Join-Path $env:CARGO_TARGET_DIR "$target/$profile/flare_winui_bridge.lib"
    if (-not $SkipRust) {
        if ($cross) {
            & rustup target add $target
            if ($LASTEXITCODE -ne 0) { throw "Rust target installation failed for $target." }
        }
        $cargoArguments = @('--config', "target.$target.rustflags=['-C','target-feature=+crt-static']",
            'build', '--locked', '--manifest-path', (Join-Path $PSScriptRoot 'bridge/Cargo.toml'), '--target', $target)
        if ($Configuration -eq 'Release') { $cargoArguments += '--release' }
        & cargo @cargoArguments
        if ($LASTEXITCODE -ne 0) { throw "Rust bridge build failed with exit code $LASTEXITCODE." }
    }
    if (-not (Test-Path -LiteralPath $bridge)) { throw "The Rust bridge library is missing: $bridge" }
    $msbuildHost = if ($hostArm) { 'arm64' } else { 'amd64' }
    $msbuild = Join-Path $installation "MSBuild/Current/Bin/$msbuildHost/MSBuild.exe"
    & $msbuild (Join-Path $PSScriptRoot 'flare-winui.vcxproj') /m /nologo /v:minimal "/p:Platform=$Platform" "/p:Configuration=$Configuration" "/p:FlareBridgeLibrary=$bridge"
    if ($LASTEXITCODE -ne 0) { throw "WinUI build failed with exit code $LASTEXITCODE." }
    $output = Join-Path $repository ".local-dev/winui/$Platform/$Configuration"
    # The app-local runtime directory is named for the toolset that ships it,
    # such as Microsoft.VC143.CRT. Use the newest one the installation provides.
    $runtime = Get-ChildItem -LiteralPath (Join-Path $installation 'VC/Redist/MSVC') -Directory |
        Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } | Sort-Object { [version]$_.Name } -Descending |
        ForEach-Object { Get-ChildItem -Path (Join-Path $_.FullName "$($Platform.ToLowerInvariant())/Microsoft.VC*.CRT") -Directory -ErrorAction SilentlyContinue } |
        Select-Object -First 1
    if (-not $runtime) { throw "The Visual C++ app-local runtime files for $Platform were not found." }
    $runtimeFiles = @(Get-ChildItem -LiteralPath $runtime.FullName -Filter '*.dll')
    if (-not $runtimeFiles.Count) { throw "The Visual C++ app-local runtime directory is empty: $($runtime.FullName)" }
    $runtimeFiles | Copy-Item -Destination $output -Force
    Write-Host "Flare desktop: $(Join-Path $output 'flare-winui.exe')"
    if ($SingleFile) {
        & (Join-Path $PSScriptRoot 'package-single-file.ps1') -Platform $Platform
    } else {
        Write-Host 'Distribute this complete directory, or run package-single-file.ps1 to create Flare.exe from it.'
    }
} finally {
    $env:CARGO_TARGET_DIR = $sharedCargoTarget
    Pop-Location
}
