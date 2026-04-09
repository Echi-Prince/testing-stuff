$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $repoRoot "frontend")
python -m http.server 3000
