param(
    [Parameter(Mandatory=$true)]
    [string]$Exe,
    [int]$TimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

if (-not (Test-Path $Exe)) {
    throw "test_dist_remote executable not found: $Exe"
}

$Listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse("127.0.0.1"), 0)
$Listener.Start()
$Port = $Listener.LocalEndpoint.Port
$Listener.Stop()

$Url = "tcp://127.0.0.1:$Port"
$TempDir = Join-Path ([IO.Path]::GetTempPath()) ("tensorcore-gloo-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

function Start-Rank {
    param([int]$Rank)
    $Out = Join-Path $TempDir "rank$Rank.out"
    $Err = Join-Path $TempDir "rank$Rank.err"
    $Args = @(
        "--rank", "$Rank",
        "--world", "2",
        "--url", $Url,
        "--test", "allreduce",
        "--elements", "32",
        "--iters", "2"
    )
    $Info = [System.Diagnostics.ProcessStartInfo]::new()
    $Info.FileName = $Exe
    $Info.Arguments = ($Args -join " ")
    $Info.UseShellExecute = $false
    $Info.RedirectStandardOutput = $true
    $Info.RedirectStandardError = $true
    $Info.CreateNoWindow = $true

    $Proc = [System.Diagnostics.Process]::new()
    $Proc.StartInfo = $Info
    if (-not $Proc.Start()) {
        throw "failed to start rank $Rank"
    }

    return [pscustomobject]@{
        Rank = $Rank
        Process = $Proc
        Out = $Out
        Err = $Err
        ExitCode = $null
    }
}

try {
    Write-Host "[tensorcore/windows-gloo] $Exe"
    Write-Host "[tensorcore/windows-gloo] url=$Url"
    $Rank0 = Start-Rank 0
    Start-Sleep -Milliseconds 300
    $Rank1 = Start-Rank 1

    $Deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    foreach ($RankProc in @($Rank0, $Rank1)) {
        $Proc = $RankProc.Process
        while (-not $Proc.HasExited -and [DateTime]::UtcNow -lt $Deadline) {
            Start-Sleep -Milliseconds 100
            $Proc.Refresh()
        }
    }

    $Failed = $false
    foreach ($RankProc in @($Rank0, $Rank1)) {
        $Proc = $RankProc.Process
        $Proc.Refresh()
        if (-not $Proc.HasExited) {
            $Failed = $true
            $Proc.Kill()
            $Proc.WaitForExit(5000) | Out-Null
        } else {
            $Proc.WaitForExit()
        }

        $RankProc.ExitCode = if ($Proc.HasExited) { $Proc.ExitCode } else { $null }
        Set-Content -Path $RankProc.Out -Value $Proc.StandardOutput.ReadToEnd() -Encoding UTF8
        Set-Content -Path $RankProc.Err -Value $Proc.StandardError.ReadToEnd() -Encoding UTF8

        if ($null -eq $RankProc.ExitCode -or $RankProc.ExitCode -ne 0) {
            Write-Host "[tensorcore/windows-gloo] rank $($RankProc.Rank) exit=$($RankProc.ExitCode)"
            $Failed = $true
        }
        $Proc.Dispose()
    }

    foreach ($Rank in 0, 1) {
        $Out = Join-Path $TempDir "rank$Rank.out"
        $Err = Join-Path $TempDir "rank$Rank.err"
        if (Test-Path $Out) { Get-Content $Out | ForEach-Object { Write-Host "[rank $Rank stdout] $_" } }
        if (Test-Path $Err) { Get-Content $Err | ForEach-Object { Write-Host "[rank $Rank stderr] $_" } }
    }

    if ($Failed) {
        throw "Windows GLOO local split-rank smoke failed"
    }
    Write-Host "[tensorcore/windows-gloo] OK"
} finally {
    Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
}
