$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot "training\.venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    throw "Missing backend runtime environment at training\.venv. Create it before launching."
}

Set-Location $repoRoot
& $pythonPath -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
