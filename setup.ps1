# Cria o ambiente Python do projeto com uv (fora do OneDrive, para não sincronizar ~1 GB de bibliotecas).
#   powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = "$env:USERPROFILE\.local\bin\uv.exe" }
$venv = "$env:USERPROFILE\.venvs\potato-cv"

& $uv venv $venv --python 3.12
& $uv pip install --python "$venv\Scripts\python.exe" torch torchvision --index-url https://download.pytorch.org/whl/cpu
& $uv pip install --python "$venv\Scripts\python.exe" -r "$PSScriptRoot\requirements.txt"

Write-Host ""
Write-Host "Ambiente pronto. Python do projeto:" -ForegroundColor Green
Write-Host "  $venv\Scripts\python.exe"
Write-Host "Ative com:  $venv\Scripts\Activate.ps1"
