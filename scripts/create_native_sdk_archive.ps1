param(
    [Parameter(Mandatory = $true)]
    [string]$Prefix,
    [string]$OutDir,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Prefix = (Resolve-Path $Prefix).Path
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = [IO.Path]::GetTempPath()
}
$OutDir = [IO.Path]::GetFullPath($OutDir)

$PyProject = Get-Content (Join-Path $Root "pyproject.toml") -Raw
$VersionMatch = [regex]::Match($PyProject, '(?m)^version\s*=\s*"([^"]+)"\s*$')
if (-not $VersionMatch.Success) {
    throw "project.version not found in pyproject.toml"
}
$Version = $VersionMatch.Groups[1].Value

$Architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
$ArchiveArch = switch ($Architecture) {
    "x64" { "x86_64" }
    "arm64" { "arm64" }
    default { $Architecture }
}

$Required = @(
    "bin\tensorcore.dll",
    "lib\tensorcore.lib",
    "lib\tensorcore_static.lib",
    "lib\cmake\tensorcore\tensorcoreConfig.cmake",
    "lib\cmake\tensorcore\tensorcoreConfigVersion.cmake",
    "lib\pkgconfig\tensorcore.pc"
)
Get-ChildItem (Join-Path $Root "include\tensorcore") -Filter "*.h" -File | ForEach-Object {
    $Required += "include\tensorcore\$($_.Name)"
}

$Missing = @($Required | Where-Object { -not (Test-Path (Join-Path $Prefix $_)) })
if ($Missing.Count -ne 0) {
    throw "native SDK prefix is missing required files:`n  $($Missing -join "`n  ")"
}

$Pc = Get-Content (Join-Path $Prefix "lib\pkgconfig\tensorcore.pc") -Raw
$PcVersion = [regex]::Match($Pc, '(?m)^Version:\s*(\S+)\s*$')
if (-not $PcVersion.Success -or $PcVersion.Groups[1].Value -ne $Version) {
    throw "pkg-config version mismatch: expected $Version"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$Archive = Join-Path $OutDir "tensorcore-native-sdk-$Version-windows-$ArchiveArch.zip"
if (Test-Path $Archive) {
    Remove-Item $Archive -Force
}
Compress-Archive -Path (Join-Path $Prefix "*") -DestinationPath $Archive -CompressionLevel Optimal
Write-Output $Archive
