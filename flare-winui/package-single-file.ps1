# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [ValidateSet('ARM64', 'x64')][string]$Platform = $(if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'ARM64' } else { 'x64' }),
    [string]$InputDirectory,
    [switch]$ArchiveOnly
)
$ErrorActionPreference = 'Stop'
$bundleRepository = Split-Path -Parent $PSScriptRoot
if (-not $InputDirectory) { $InputDirectory = Join-Path $bundleRepository ".local-dev/winui/$Platform/Release" }
$bundleSource = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InputDirectory).TrimEnd([IO.Path]::DirectorySeparatorChar)
if (-not (Test-Path -LiteralPath $bundleSource -PathType Container)) { throw "Build the Release desktop folder first: $bundleSource" }
if ((Get-Item -LiteralPath $bundleSource -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'The runtime folder must not be a link or junction.' }
foreach ($bundleRequired in @('flare-winui.exe', 'resources.pri', 'Microsoft.UI.Xaml.dll')) {
    $bundleRequiredPath = Join-Path $bundleSource $bundleRequired
    if (-not (Test-Path -LiteralPath $bundleRequiredPath -PathType Leaf) -or (Get-Item -LiteralPath $bundleRequiredPath).Length -eq 0) {
        throw "Required runtime file is missing or empty: $bundleRequired"
    }
}

# Enumerate one directory at a time so links cannot redirect recursive traversal.
$bundleFiles = @{}
$bundlePaths = [Collections.Generic.List[string]]::new()
$bundleSeen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$bundleDirectories = [Collections.Generic.Stack[string]]::new()
$bundleDirectories.Push($bundleSource)
while ($bundleDirectories.Count) {
    $bundleCurrentDirectory = $bundleDirectories.Pop()
    foreach ($bundleItem in Get-ChildItem -LiteralPath $bundleCurrentDirectory -Force) {
        if ($bundleItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Runtime links are not supported: $($bundleItem.FullName)" }
        if ($bundleItem.PSIsContainer) { $bundleDirectories.Push($bundleItem.FullName); continue }
        $bundleRelative = $bundleItem.FullName.Substring($bundleSource.Length + 1).Replace('\', '/')
        $bundleExtension = [IO.Path]::GetExtension($bundleRelative).ToLowerInvariant()
        if ($bundleRelative -notin @('flare-winui.exe', 'RestartAgent.exe') -and $bundleExtension -notin @('.dll', '.mui', '.pri', '.winmd', '.png')) { continue }
        if (-not $bundleSeen.Add($bundleRelative)) { throw "Runtime paths collide on Windows: $bundleRelative" }
        $bundlePaths.Add($bundleRelative)
        $bundleFiles[$bundleRelative] = $bundleItem.FullName
    }
}
foreach ($bundleRequired in @('flare-winui.exe', 'resources.pri', 'Microsoft.UI.Xaml.dll')) {
    if (-not $bundleSeen.Contains($bundleRequired)) { throw "Required runtime file is missing: $bundleRequired" }
}
$bundlePaths.Sort([StringComparer]::Ordinal)
$bundleDirectory = Join-Path $bundleRepository ".local-dev/winui-bundle/$Platform"
New-Item -ItemType Directory -Force -Path $bundleDirectory | Out-Null
if ((Get-Item -LiteralPath $bundleDirectory -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'The bundle output directory must not be a link or junction.' }
$bundleZip = Join-Path $bundleDirectory 'payload.zip'
$bundleTemporary = Join-Path $bundleDirectory ("payload-" + [Guid]::NewGuid().ToString('N') + '.zip')
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$bundleTimestamp = [DateTimeOffset]::new(1980, 1, 1, 0, 0, 0, [TimeSpan]::Zero)
$bundleManifestFiles = [Collections.Generic.List[object]]::new()
try {
    $bundleStream = [IO.File]::Open($bundleTemporary, [IO.FileMode]::CreateNew, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $bundleArchive = $null
    try {
        $bundleArchive = [IO.Compression.ZipArchive]::new($bundleStream, [IO.Compression.ZipArchiveMode]::Create, $false, [Text.Encoding]::UTF8)
        foreach ($bundleRelative in $bundlePaths) {
            $bundleFilePath = $bundleFiles[$bundleRelative]
            if ((Get-Item -LiteralPath $bundleFilePath -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Runtime links are not supported: $bundleFilePath" }
            # Hold a read-only share through hashing and copying so the manifest
            # describes exactly the bytes placed in this archive.
            $bundleInput = [IO.FileStream]::new($bundleFilePath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read, 1048576, [IO.FileOptions]::SequentialScan)
            try {
                $bundleHasher = [Security.Cryptography.SHA256]::Create()
                try { $bundleHash = [BitConverter]::ToString($bundleHasher.ComputeHash($bundleInput)).Replace('-', '').ToLowerInvariant() } finally { $bundleHasher.Dispose() }
                $bundleSize = [UInt64]$bundleInput.Length
                $bundleInput.Position = 0
                $bundleEntry = $bundleArchive.CreateEntry($bundleRelative, [IO.Compression.CompressionLevel]::Optimal)
                $bundleEntry.LastWriteTime = $bundleTimestamp
                $bundleEntry.ExternalAttributes = 0
                $bundleOutput = $bundleEntry.Open()
                try { $bundleInput.CopyTo($bundleOutput, 1048576) } finally { $bundleOutput.Dispose() }
                $bundleManifestFiles.Add([ordered]@{ path = $bundleRelative; size = $bundleSize; sha256 = $bundleHash })
            } finally { $bundleInput.Dispose() }
        }
        $bundleManifest = [ordered]@{ format = 1; entrypoint = 'flare-winui.exe'; files = $bundleManifestFiles.ToArray() } | ConvertTo-Json -Depth 5 -Compress
        $bundleEntry = $bundleArchive.CreateEntry('bundle.json', [IO.Compression.CompressionLevel]::Optimal)
        $bundleEntry.LastWriteTime = $bundleTimestamp
        $bundleEntry.ExternalAttributes = 0
        $bundleOutput = $bundleEntry.Open()
        try { $bundleBytes = [Text.UTF8Encoding]::new($false).GetBytes($bundleManifest); $bundleOutput.Write($bundleBytes, 0, $bundleBytes.Length) } finally { $bundleOutput.Dispose() }
    } finally {
        try { if ($bundleArchive) { $bundleArchive.Dispose() } } finally { $bundleStream.Dispose() }
    }
    if (Test-Path -LiteralPath $bundleZip) { [IO.File]::Replace($bundleTemporary, $bundleZip, [NullString]::Value) } else { [IO.File]::Move($bundleTemporary, $bundleZip) }
} finally {
    if (Test-Path -LiteralPath $bundleTemporary) { Remove-Item -LiteralPath $bundleTemporary -Force }
}
Write-Host "Runtime archive: $bundleZip ($($bundlePaths.Count) files)"
if ($ArchiveOnly) { return }

$bundleCargoTargetBefore = $env:CARGO_TARGET_DIR
$bundleRustFlagsBefore = $env:RUSTFLAGS
$bundleEncodedRustFlagsBefore = $env:CARGO_ENCODED_RUSTFLAGS
Push-Location -LiteralPath $bundleRepository
try {
    . (Join-Path $bundleRepository 'scripts/dev_windows.ps1') -Action Shell
    $env:CARGO_TARGET_DIR = Join-Path $localDev 'launcher-target'
    $env:RUSTFLAGS = '-C target-feature=+crt-static'
    $env:CARGO_ENCODED_RUSTFLAGS = $null
    $bundleVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $bundleComponent = if ($Platform -eq 'ARM64') { 'Microsoft.VisualStudio.Component.VC.Tools.ARM64' } else { 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' }
    $bundleInstallation = & $bundleVswhere -latest -products '*' -requires $bundleComponent -property installationPath
    if (-not $bundleInstallation) { throw "Visual Studio C++ build tools for $Platform are required." }
    $bundleMachine = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    $bundleHostArm = $bundleMachine -eq 'ARM64'
    $bundleVcArchitecture = if ($bundleHostArm) { if ($Platform -eq 'ARM64') { 'arm64' } else { 'arm64_x64' } } else { if ($Platform -eq 'ARM64') { 'x64_arm64' } else { 'x64' } }
    $bundleVcvars = Join-Path $bundleInstallation 'VC/Auxiliary/Build/vcvarsall.bat'
    if ($env:VSCMD_ARG_TGT_ARCH -ne $Platform -or ([string]$env:VSINSTALLDIR).TrimEnd('\') -ne $bundleInstallation.TrimEnd('\')) {
        $bundleEnvironment = & $env:ComSpec /d /s /c "`"$bundleVcvars`" $bundleVcArchitecture >nul && set"
        if ($LASTEXITCODE -ne 0) { throw "Visual Studio environment setup failed for $Platform." }
        foreach ($bundleLine in $bundleEnvironment) { if ($bundleLine -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
    }
    $bundleTarget = if ($Platform -eq 'ARM64') { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
    & rustup target add $bundleTarget
    if ($LASTEXITCODE -ne 0) { throw "Rust target installation failed for $bundleTarget." }
    & cargo build --locked --release --manifest-path (Join-Path $PSScriptRoot 'launcher/Cargo.toml') --target $bundleTarget
    if ($LASTEXITCODE -ne 0) { throw "Single-file launcher build failed with exit code $LASTEXITCODE." }
    $bundleExecutable = Join-Path $env:CARGO_TARGET_DIR "$bundleTarget/release/flare-launcher.exe"
    $bundleDestination = Join-Path $bundleRepository ".local-dev/winui/$Platform/SingleFile"
    New-Item -ItemType Directory -Force -Path $bundleDestination | Out-Null
    Copy-Item -LiteralPath $bundleExecutable -Destination (Join-Path $bundleDestination 'Flare.exe') -Force
    Write-Host "Single-file desktop: $(Join-Path $bundleDestination 'Flare.exe')"
} finally {
    $env:CARGO_TARGET_DIR = $bundleCargoTargetBefore
    $env:RUSTFLAGS = $bundleRustFlagsBefore
    $env:CARGO_ENCODED_RUSTFLAGS = $bundleEncodedRustFlagsBefore
    Pop-Location
}
