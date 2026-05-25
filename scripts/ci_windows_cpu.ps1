param(
    [string]$BuildDir,
    [string]$Prefix,
    [string]$Config,
    [string]$Generator,
    [string]$Platform,
    [string]$Python,
    [switch]$SkipPython
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-RepoPath {
    param(
        [string]$Value,
        [string]$DefaultValue
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        $Value = $DefaultValue
    }
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $Root $Value))
}

function Invoke-Step {
    param(
        [string]$Name,
        [string]$File,
        [string[]]$Arguments
    )
    Write-Host "[tensorcore/windows] $Name"
    Write-Host "  $File $($Arguments -join ' ')"
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Find-TensorcoreDll {
    param([string]$BaseDir)
    $Candidates = @(
        (Join-Path $BaseDir "bin\tensorcore.dll"),
        (Join-Path $BaseDir "bin\$Config\tensorcore.dll"),
        (Join-Path $BaseDir "lib\tensorcore.dll"),
        (Join-Path $BaseDir "$Config\tensorcore.dll"),
        (Join-Path $BaseDir "tensorcore.dll")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) {
            return (Resolve-Path $Candidate).Path
        }
    }
    $Found = Get-ChildItem -Path $BaseDir -Filter "tensorcore.dll" -Recurse -File -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($Found) {
        return $Found.FullName
    }
    return $null
}

if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = if ($env:TC_WINDOWS_CONFIG) { $env:TC_WINDOWS_CONFIG } else { "Release" }
}
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = if ($env:TC_WINDOWS_BUILD_DIR) { $env:TC_WINDOWS_BUILD_DIR } else { "build-windows-cpu" }
}
if ([string]::IsNullOrWhiteSpace($Prefix)) {
    $DefaultPrefix = Join-Path ([IO.Path]::GetTempPath()) "tensorcore-windows-install"
    $Prefix = if ($env:TC_WINDOWS_PREFIX) { $env:TC_WINDOWS_PREFIX } else { $DefaultPrefix }
}
if ([string]::IsNullOrWhiteSpace($Generator) -and $env:TC_WINDOWS_GENERATOR) {
    $Generator = $env:TC_WINDOWS_GENERATOR
}
if ([string]::IsNullOrWhiteSpace($Platform) -and $env:TC_WINDOWS_PLATFORM) {
    $Platform = $env:TC_WINDOWS_PLATFORM
}
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
}

$BuildDir = Resolve-RepoPath $BuildDir "build-windows-cpu"
$Prefix = Resolve-RepoPath $Prefix (Join-Path ([IO.Path]::GetTempPath()) "tensorcore-windows-install")

$ConfigureArgs = @(
    "-S", $Root,
    "-B", $BuildDir,
    "-DTC_ENABLE_METAL=OFF",
    "-DTC_ENABLE_CUDA=OFF",
    "-DTC_ENABLE_HIP=OFF",
    "-DTC_BUILD_BENCH=OFF",
    "-DTC_BUILD_EXAMPLES=OFF",
    "-DCMAKE_BUILD_TYPE=$Config"
)
if (-not [string]::IsNullOrWhiteSpace($Generator)) {
    $ConfigureArgs += @("-G", $Generator)
}
if (-not [string]::IsNullOrWhiteSpace($Platform)) {
    $ConfigureArgs += @("-A", $Platform)
}

Invoke-Step "configure portable CPU" "cmake" $ConfigureArgs
Invoke-Step "build portable CPU" "cmake" @("--build", $BuildDir, "--config", $Config, "--parallel")
Invoke-Step "run CTest" "ctest" @("--test-dir", $BuildDir, "-C", $Config, "--output-on-failure")
Invoke-Step "install native SDK" "cmake" @("--install", $BuildDir, "--config", $Config, "--prefix", $Prefix)

$NativeDll = Find-TensorcoreDll $Prefix
if (-not $NativeDll) {
    $NativeDll = Find-TensorcoreDll $BuildDir
}
if (-not $NativeDll) {
    throw "tensorcore.dll was not produced under $BuildDir or $Prefix"
}

$InstalledHeader = Join-Path $Prefix "include\tensorcore\tensorcore.h"
if (-not (Test-Path $InstalledHeader)) {
    throw "installed public header not found: $InstalledHeader"
}

if (-not $SkipPython) {
    $env:TENSORCORE_LIB = $NativeDll
    $OldPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = Join-Path $Root "python"
    if ($OldPythonPath) {
        $env:PYTHONPATH = "${env:PYTHONPATH};$OldPythonPath"
    }

    Invoke-Step "Python constants" $Python @((Join-Path $Root "scripts\check_python_constants.py"))
    Invoke-Step "Python FFI surface" $Python @((Join-Path $Root "scripts\check_python_ffi_surface.py"))

    $PyProject = Get-Content (Join-Path $Root "pyproject.toml") -Raw
    $VersionMatch = [regex]::Match($PyProject, '(?m)^version\s*=\s*"([^"]+)"\s*$')
    if (-not $VersionMatch.Success) {
        throw "project.version not found in pyproject.toml"
    }
    $ExpectedVersion = $VersionMatch.Groups[1].Value
    $SmokePath = Join-Path $BuildDir "windows_python_smoke.py"
    $Smoke = @'
import sys
import tensorcore as tc

expected = sys.argv[1]
actual = tc.version()
if not actual.startswith(f"tensorcore {expected}"):
    raise SystemExit(f"version mismatch: expected tensorcore {expected}, got {actual}")
if tc.backend_name(tc.TC_BACKEND_PORTABLE_CPU) != "portable_cpu":
    raise SystemExit("portable CPU backend name mismatch")
if tc.hip_device_count() != 0 or tc.cuda_device_count() != 0:
    raise SystemExit("inactive GPU diagnostic mismatch")

ctx = tc.init()
buf = None
try:
    info = tc.device_info(ctx)
    if info.name_str != "portable-cpu":
        raise SystemExit(f"unexpected device name: {info.name_str}")
    buf = tc.buffer_alloc(ctx, 64)
    if tc.buffer_size(buf) != 64:
        raise SystemExit("buffer size mismatch")
    tc.buffer_set_tier_hint(buf, "warm")
    if tc.buffer_get_tier(buf) != tc.TC_TIER_L0_DEVICE:
        raise SystemExit("memory tier mismatch")
    if tc.memory_tier_usage(ctx, "l0") != (0, 0):
        raise SystemExit("memory tier usage mismatch")
finally:
    if buf is not None:
        tc.buffer_free(ctx, buf)
    tc.shutdown(ctx)

print(actual)
'@
    Set-Content -Path $SmokePath -Value $Smoke -Encoding UTF8
    Invoke-Step "Python native smoke" $Python @($SmokePath, $ExpectedVersion)
}

Write-Host "[tensorcore/windows] OK: $NativeDll"
