# AVHA One-Click Live Launcher
# Starts FastAPI backend, opens a Cloudflare Quick Tunnel, and updates Vercel with the new HTTPS/WSS URL.

$ErrorActionPreference = "Stop"

$RootDir = $PSScriptRoot
$BackendDir = Join-Path $RootDir "backend"
$FrontendDir = Join-Path $RootDir "frontend"
$PythonExe = Join-Path $BackendDir "venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Backend virtualenv python not found at: $PythonExe" -ForegroundColor Red
    exit 1
}

$CloudflaredExe = Join-Path $env:LOCALAPPDATA "cloudflared\cloudflared.exe"
if (-not (Test-Path $CloudflaredExe)) {
    $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmd) {
        $CloudflaredExe = $cmd.Source
    } else {
        Write-Host "[ERROR] cloudflared.exe not found in $env:LOCALAPPDATA\cloudflared or PATH." -ForegroundColor Red
        exit 1
    }
}

$BackendProc = $null
$TunnelProc = $null

function Test-BackendHealthy {
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2
        return ($resp.status -eq "healthy")
    } catch {
        return $false
    }
}

try {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "   AVHA - Starting Live Deployment      " -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    # 1. Ensure FastAPI backend is running on 127.0.0.1:8000
    if (Test-BackendHealthy) {
        Write-Host "[1/3] Backend already healthy on http://127.0.0.1:8000" -ForegroundColor Green
    } else {
        Write-Host "[1/3] Starting FastAPI backend..." -ForegroundColor Yellow
        $BackendOutLog = Join-Path $env:TEMP "avha-backend.out.log"
        $BackendErrLog = Join-Path $env:TEMP "avha-backend.err.log"
        $BackendProc = Start-Process -FilePath $PythonExe `
            -ArgumentList "main.py" `
            -WorkingDirectory $BackendDir `
            -RedirectStandardOutput $BackendOutLog `
            -RedirectStandardError $BackendErrLog `
            -NoNewWindow -PassThru

        $healthy = $false
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Milliseconds 500
            if ($BackendProc.HasExited) {
                Write-Host "[ERROR] Backend exited early. Check $BackendErrLog" -ForegroundColor Red
                exit 1
            }
            if (Test-BackendHealthy) {
                $healthy = $true
                break
            }
        }
        if (-not $healthy) {
            Write-Host "[ERROR] Backend did not become healthy within 30s." -ForegroundColor Red
            exit 1
        }
        Write-Host "[1/3] Backend healthy!" -ForegroundColor Green
    }

    # 2. Start Cloudflare Quick Tunnel and capture its *.trycloudflare.com URL
    Write-Host "[2/3] Starting Cloudflare Quick Tunnel..." -ForegroundColor Yellow
    $TunnelOutLog = Join-Path $env:TEMP "avha-cloudflared.out.log"
    $TunnelErrLog = Join-Path $env:TEMP "avha-cloudflared.err.log"
    Remove-Item $TunnelOutLog, $TunnelErrLog -ErrorAction SilentlyContinue

    $TunnelProc = Start-Process -FilePath $CloudflaredExe `
        -ArgumentList "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate" `
        -RedirectStandardOutput $TunnelOutLog `
        -RedirectStandardError $TunnelErrLog `
        -NoNewWindow -PassThru

    $TunnelUrl = $null
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        if ($TunnelProc.HasExited) {
            Write-Host "[ERROR] cloudflared exited early." -ForegroundColor Red
            exit 1
        }
        foreach ($logPath in @($TunnelErrLog, $TunnelOutLog)) {
            if (Test-Path $logPath) {
                $fs = [System.IO.File]::Open($logPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
                try {
                    $sr = New-Object System.IO.StreamReader($fs)
                    $content = $sr.ReadToEnd()
                    $sr.Dispose()
                } finally {
                    $fs.Dispose()
                }
                if ($content -match 'https://[a-z0-9-]+\.trycloudflare\.com') {
                    $TunnelUrl = $Matches[0]
                    break
                }
            }
        }
        if ($TunnelUrl) { break }
    }

    if (-not $TunnelUrl) {
        Write-Host "[ERROR] Could not obtain trycloudflare.com URL within 30s." -ForegroundColor Red
        exit 1
    }

    $WssUrl = $TunnelUrl -replace '^https://', 'wss://'
    Write-Host "[2/3] Tunnel ready: $TunnelUrl" -ForegroundColor Green

    # 3. Redeploy Vercel production frontend pointing to the new tunnel URL
    Write-Host "[3/3] Updating Vercel production frontend (avha-assistant)..." -ForegroundColor Yellow
    Push-Location $FrontendDir
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        npx --yes vercel deploy --prod --yes --project avha-assistant --build-env "VITE_API_URL=$TunnelUrl" --build-env "VITE_WS_URL=$WssUrl"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Vercel deploy failed (exit code $LASTEXITCODE)." -ForegroundColor Red
            exit $LASTEXITCODE
        }
    } finally {
        $ErrorActionPreference = $prevEap
        Pop-Location
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "   AVHA IS LIVE!                        " -ForegroundColor Green
    Write-Host "   App URL : https://avha-assistant.vercel.app" -ForegroundColor White
    Write-Host "   Backend : $TunnelUrl" -ForegroundColor Gray
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Keep this window open while using AVHA. Press Ctrl+C to stop." -ForegroundColor Cyan

    Start-Process "https://avha-assistant.vercel.app" | Out-Null

    while (-not $TunnelProc.HasExited) {
        if ($null -ne $BackendProc -and $BackendProc.HasExited) {
            Write-Host "[WARN] Backend process stopped unexpectedly." -ForegroundColor Red
            break
        }
        Start-Sleep -Seconds 2
    }
} finally {
    Write-Host ""
    Write-Host "Shutting down launcher processes..." -ForegroundColor Yellow
    if ($null -ne $TunnelProc -and -not $TunnelProc.HasExited) {
        Stop-Process -Id $TunnelProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($null -ne $BackendProc -and -not $BackendProc.HasExited) {
        Stop-Process -Id $BackendProc.Id -Force -ErrorAction SilentlyContinue
    }
}
