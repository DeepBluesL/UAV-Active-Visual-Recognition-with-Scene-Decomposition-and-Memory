param(
    [switch]$Retrain,
    [string]$Python = 'D:\anaconda3\envs\uav\python.exe'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python was not found: $Python. Pass -Python with a Python environment containing torch, numpy and Pillow."
}
if ($Retrain -or -not (Test-Path -LiteralPath 'output\demo\classifier.pt')) {
    & $Python demo\classifier.py --epochs 8
    if ($LASTEXITCODE -ne 0) { throw 'Classifier training failed.' }
}
& $Python demo\run_demo.py
if ($LASTEXITCODE -ne 0) { throw 'Demo evaluation failed.' }
Write-Host "Done. Open $PSScriptRoot\output\demo\report.html"
