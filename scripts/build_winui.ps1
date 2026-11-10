# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Release',
    [ValidateSet('ARM64', 'x64')][string]$Platform = $(if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'ARM64' } else { 'x64' }),
    [switch]$SkipRust,
    [switch]$SingleFile
)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../flare-winui/build.ps1') -Configuration $Configuration -Platform $Platform -SkipRust:$SkipRust -SingleFile:$SingleFile
