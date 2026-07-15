param(
    [Parameter(Mandatory = $true)]
    [string]$Archive,
    [string]$Config = "Release",
    [string]$CMake = "cmake"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Archive = (Resolve-Path $Archive).Path
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
$ExpectedName = "tensorcore-native-sdk-$Version-windows-$ArchiveArch.zip"
if ([IO.Path]::GetFileName($Archive) -ne $ExpectedName) {
    throw "native SDK archive name mismatch: expected $ExpectedName"
}

$TempRoot = Join-Path ([IO.Path]::GetTempPath()) ("tensorcore-sdk-check-" + [Guid]::NewGuid().ToString("N"))
$Sdk = Join-Path $TempRoot "sdk"
$ConsumerBuild = Join-Path $TempRoot "consumer-build"
New-Item -ItemType Directory -Path $Sdk -Force | Out-Null

try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Zip = [IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        $SdkRoot = [IO.Path]::GetFullPath($Sdk + [IO.Path]::DirectorySeparatorChar)
        foreach ($Entry in $Zip.Entries) {
            $Relative = $Entry.FullName.Replace('/', [IO.Path]::DirectorySeparatorChar)
            $Destination = [IO.Path]::GetFullPath((Join-Path $Sdk $Relative))
            if (-not $Destination.StartsWith($SdkRoot, [StringComparison]::OrdinalIgnoreCase)) {
                throw "unsafe archive member: $($Entry.FullName)"
            }
        }
    }
    finally {
        $Zip.Dispose()
    }
    Expand-Archive -Path $Archive -DestinationPath $Sdk -Force

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
    $Missing = @($Required | Where-Object { -not (Test-Path (Join-Path $Sdk $_)) })
    if ($Missing.Count -ne 0) {
        throw "native SDK archive is missing required files:`n  $($Missing -join "`n  ")"
    }

    $Pc = Get-Content (Join-Path $Sdk "lib\pkgconfig\tensorcore.pc") -Raw
    $PcVersion = [regex]::Match($Pc, '(?m)^Version:\s*(\S+)\s*$')
    if (-not $PcVersion.Success -or $PcVersion.Groups[1].Value -ne $Version) {
        throw "pkg-config version mismatch: expected $Version"
    }

    & $CMake -S (Join-Path $Root "examples\native_sdk_consumer") -B $ConsumerBuild "-DCMAKE_PREFIX_PATH=$Sdk"
    if ($LASTEXITCODE -ne 0) { throw "native SDK consumer configure failed" }
    & $CMake --build $ConsumerBuild --config $Config --parallel
    if ($LASTEXITCODE -ne 0) { throw "native SDK consumer build failed" }

    function Find-Consumer {
        param([string]$Name)
        $Found = Get-ChildItem -Path $ConsumerBuild -Filter "$Name.exe" -Recurse -File |
            Select-Object -First 1
        if (-not $Found) { throw "native SDK consumer executable not found: $Name.exe" }
        return $Found.FullName
    }

    $OldPath = $env:Path
    $env:Path = "$(Join-Path $Sdk "bin");$OldPath"
    try {
        foreach ($Name in @("consumer_shared", "consumer_cxx", "consumer_static")) {
            $Executable = Find-Consumer $Name
            & $Executable
            if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
        }
    }
    finally {
        $env:Path = $OldPath
    }

    Write-Host "native SDK archive OK: $Archive"
}
finally {
    if (Test-Path $TempRoot) {
        Remove-Item $TempRoot -Recurse -Force
    }
}
