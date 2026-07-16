# ─────────────────────────────────────────────────────────────
#  SocialPulse AI — Local Launcher (PowerShell)
# ─────────────────────────────────────────────────────────────

# 1. Load credentials from .env
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.+)$') {
            $key   = $matches[1].Trim()
            $value = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
    Write-Host "[OK] Credentials loaded from .env" -ForegroundColor Green
} else {
    Write-Host "[WARN] No .env file found — using defaults in app.py" -ForegroundColor Yellow
}

# 2. Validate key credential fields
$apiKey    = [System.Environment]::GetEnvironmentVariable("WATSONX_API_KEY")
$projectId = [System.Environment]::GetEnvironmentVariable("WATSONX_PROJECT_ID")

if ($apiKey -eq "your-watsonx-api-key-here" -or -not $apiKey) {
    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "  ║  ACTION REQUIRED: Set your IBM watsonx.ai credentials ║" -ForegroundColor Yellow
    Write-Host "  ║  Edit the .env file in this folder:                   ║" -ForegroundColor Yellow
    Write-Host "  ║    WATSONX_API_KEY    = <your API key>                ║" -ForegroundColor Yellow
    Write-Host "  ║    WATSONX_PROJECT_ID = <your project ID>             ║" -ForegroundColor Yellow
    Write-Host "  ║  Get them at: cloud.ibm.com → watsonx.ai              ║" -ForegroundColor Yellow
    Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""
}

# 3. Launch the app
Write-Host ""
Write-Host "  Starting SocialPulse AI on http://localhost:5000 ..." -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

py -3.13 app.py
