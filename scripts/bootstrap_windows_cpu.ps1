param(
    [string]$RepoDir,
    [string]$PythonVersion,
    [switch]$Install,
    [switch]$SkipPython,
    [switch]$NoSmoke
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if ([string]::IsNullOrWhiteSpace($RepoDir)) {
    $RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
if ([string]::IsNullOrWhiteSpace($PythonVersion)) {
    $PythonVersion = "3.12.10"
}

function Test-Admin {
    $Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $Principal = [Security.Principal.WindowsPrincipal]$Identity
    return $Principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Resolve-CommandPath {
    param([string]$Name)
    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }
    return $null
}

function Find-VsWhere {
    $Candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"),
        (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
    )
    foreach ($Candidate in $Candidates) {
        if ($Candidate -and (Test-Path $Candidate)) {
            return (Resolve-Path $Candidate).Path
        }
    }
    return $null
}

function Find-VsInstall {
    $VsWhere = Find-VsWhere
    if (-not $VsWhere) {
        return $null
    }
    $InstallPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($InstallPath)) {
        return $null
    }
    return $InstallPath
}

function Find-VsCMake {
    $InstallPath = Find-VsInstall
    if (-not $InstallPath) {
        return $null
    }
    $Candidate = Join-Path $InstallPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if (Test-Path $Candidate) {
        return (Resolve-Path $Candidate).Path
    }
    return $null
}

function Find-Python {
    $Python = Resolve-CommandPath "python"
    if ($Python) {
        return $Python
    }
    $MajorMinor = ($PythonVersion -split "\.")[0..1] -join ""
    $Candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$MajorMinor\python.exe"
    if (Test-Path $Candidate) {
        return (Resolve-Path $Candidate).Path
    }
    return $null
}

function Download-File {
    param(
        [string]$Uri,
        [string]$OutFile
    )
    Write-Host "[tensorcore/windows-bootstrap] download $Uri"
    Invoke-WebRequest -Uri $Uri -OutFile $OutFile
}

function Install-BuildTools {
    if (-not (Test-Admin)) {
        throw "Visual Studio Build Tools install requires an elevated PowerShell. Re-run this script as Administrator with -Install."
    }
    $Installer = Join-Path $env:TEMP "vs_BuildTools.exe"
    Download-File "https://aka.ms/vs/17/release/vs_BuildTools.exe" $Installer
    $Args = @(
        "--quiet",
        "--wait",
        "--norestart",
        "--nocache",
        "--includeRecommended",
        "--add", "Microsoft.VisualStudio.Workload.VCTools",
        "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "--add", "Microsoft.VisualStudio.Component.VC.CMake.Project",
        "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041"
    )
    Write-Host "[tensorcore/windows-bootstrap] install Visual Studio Build Tools"
    $Process = Start-Process -FilePath $Installer -ArgumentList $Args -Wait -PassThru
    if ($Process.ExitCode -ne 0 -and $Process.ExitCode -ne 3010) {
        throw "Visual Studio Build Tools installer failed with exit code $($Process.ExitCode)"
    }
}

function Install-UserPython {
    $Installer = Join-Path $env:TEMP "python-$PythonVersion-amd64.exe"
    Download-File "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe" $Installer
    $Args = @(
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=1",
        "Include_launcher=0",
        "Include_test=0"
    )
    Write-Host "[tensorcore/windows-bootstrap] install Python $PythonVersion for current user"
    $Process = Start-Process -FilePath $Installer -ArgumentList $Args -Wait -PassThru
    if ($Process.ExitCode -ne 0) {
        throw "Python installer failed with exit code $($Process.ExitCode)"
    }
}

$RepoDir = (Resolve-Path $RepoDir).Path
$IsAdmin = Test-Admin
$VsInstall = Find-VsInstall
$CMakeExe = Resolve-CommandPath "cmake"
if (-not $CMakeExe) {
    $CMakeExe = Find-VsCMake
}
$CTestExe = Resolve-CommandPath "ctest"
if (-not $CTestExe -and $CMakeExe) {
    $CandidateCTest = Join-Path (Split-Path -Parent $CMakeExe) "ctest.exe"
    if (Test-Path $CandidateCTest) {
        $CTestExe = (Resolve-Path $CandidateCTest).Path
    }
}
$PythonExe = if ($SkipPython) { $null } else { Find-Python }

Write-Host "[tensorcore/windows-bootstrap] repo: $RepoDir"
Write-Host "[tensorcore/windows-bootstrap] admin: $IsAdmin"
Write-Host "[tensorcore/windows-bootstrap] visual studio: $(if ($VsInstall) { $VsInstall } else { 'missing' })"
Write-Host "[tensorcore/windows-bootstrap] cmake: $(if ($CMakeExe) { $CMakeExe } else { 'missing' })"
Write-Host "[tensorcore/windows-bootstrap] ctest: $(if ($CTestExe) { $CTestExe } else { 'missing' })"
if (-not $SkipPython) {
    Write-Host "[tensorcore/windows-bootstrap] python: $(if ($PythonExe) { $PythonExe } else { 'missing' })"
}

if ($Install) {
    if (-not $VsInstall -or -not $CMakeExe -or -not $CTestExe) {
        Install-BuildTools
        $VsInstall = Find-VsInstall
        $CMakeExe = Find-VsCMake
        $CTestExe = Join-Path (Split-Path -Parent $CMakeExe) "ctest.exe"
    }
    if (-not $SkipPython -and -not $PythonExe) {
        Install-UserPython
        $PythonExe = Find-Python
    }
}

if (-not $CMakeExe -or -not $CTestExe) {
    throw "Windows C++ toolchain is not ready. Install Visual Studio Build Tools 2022 with C++ and CMake components, then rerun this script."
}
if (-not $SkipPython -and -not $PythonExe) {
    throw "Python is not ready. Install Python 3.11+ or rerun with -Install."
}

if (-not $NoSmoke) {
    $CiScript = Join-Path $RepoDir "scripts\ci_windows_cpu.ps1"
    $Args = @("-File", $CiScript, "-CMake", $CMakeExe, "-CTest", $CTestExe)
    if ($SkipPython) {
        $Args += "-SkipPython"
    } else {
        $Args += @("-Python", $PythonExe)
    }
    Write-Host "[tensorcore/windows-bootstrap] run ci_windows_cpu.ps1"
    powershell -NoProfile -ExecutionPolicy Bypass @Args
    if ($LASTEXITCODE -ne 0) {
        throw "ci_windows_cpu.ps1 failed with exit code $LASTEXITCODE"
    }
}

Write-Host "[tensorcore/windows-bootstrap] OK"
