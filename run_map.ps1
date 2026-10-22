param(
    [switch]$Retrain,
    [int]$Episodes = 300000,
    [string]$Python = 'D:\anaconda3\envs\uav\python.exe'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python was not found: $Python. Specify -Python with an environment containing torch, numpy, Pillow and matplotlib."
}
if (-not (Test-Path -LiteralPath 'output\demo\classifier.pt')) {
    & $Python demo\classifier.py --epochs 8
    if ($LASTEXITCODE -ne 0) { throw 'Classifier training failed.' }
}
& $Python -m map_baseline.features --data dataset --classifier output\demo\classifier.pt --out output\map\cache
if ($LASTEXITCODE -ne 0) { throw 'Observation cache preparation failed.' }
& $Python -m unittest map_baseline.test_environment
if ($LASTEXITCODE -ne 0) { throw 'Environment protocol checks failed.' }
if ($Retrain -or -not (Test-Path -LiteralPath 'output\map\development\best.pt')) {
    & $Python -m map_baseline.experiment train --phase development --episodes 30000 --out output\map\development
    if ($LASTEXITCODE -ne 0) { throw 'Development training failed.' }
}
if ($Retrain -or -not (Test-Path -LiteralPath 'output\map\full\last.pt')) {
    & $Python -m map_baseline.experiment train --phase full --episodes $Episodes --log-every 25000 --out output\map\full
    if ($LASTEXITCODE -ne 0) { throw 'Full training failed.' }
}
& $Python -m map_baseline.experiment evaluate --checkpoint output\map\full\last.pt --out output\map
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed. If the observation cache changed, rerun with -Retrain.' }
Write-Host "Done. Open $PSScriptRoot\output\map\report.html"
