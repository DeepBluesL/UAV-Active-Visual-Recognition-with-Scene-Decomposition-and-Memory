param(
    [switch]$Retrain,
    [int]$Episodes = 300000,
    [int]$SamBatchSize = 16,
    [string]$Python = 'D:\anaconda3\envs\uav\python.exe'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath 'output\demo\classifier.pt')) {
    & $Python demo\classifier.py --epochs 8
    if ($LASTEXITCODE -ne 0) { throw 'Classifier training failed.' }
}
& $Python -m map_baseline.features
if ($LASTEXITCODE -ne 0) { throw 'Raw observation cache preparation failed.' }
& $Python ibe_map\bootstrap_sam.py
if ($LASTEXITCODE -ne 0) { throw 'Official SAM source/weight preparation failed.' }
& $Python -m ibe_map.features --batch-size $SamBatchSize
if ($LASTEXITCODE -ne 0) { throw 'SAM decomposition/cache failed; completed batches can be resumed.' }
& $Python -m unittest map_baseline.test_environment ibe_map.test_recurrence ibe_map.test_sam ibe_map.test_sam_attention
if ($LASTEXITCODE -ne 0) { throw 'Environment / recurrent / SAM checks failed.' }
if (-not (Test-Path -LiteralPath 'output\map\full\last.pt')) {
    & $Python -m map_baseline.experiment train --phase full --episodes $Episodes
    if ($LASTEXITCODE -ne 0) { throw 'Reference MAP training failed.' }
}
foreach ($variant in @('memo_map','sam_map','ibe_map')) {
    $developmentCheckpoint = Join-Path "output\ibe_map\$variant\development" 'best.pt'
    $fullCheckpoint = Join-Path "output\ibe_map\$variant\full" 'last.pt'
    if ($Retrain -or -not (Test-Path -LiteralPath $developmentCheckpoint)) {
        & $Python -m ibe_map.experiment train --variant $variant --phase development --episodes 30000
        if ($LASTEXITCODE -ne 0) { throw "$variant development training failed." }
    }
    if ($Retrain -or -not (Test-Path -LiteralPath $fullCheckpoint)) {
        & $Python -m ibe_map.experiment train --variant $variant --phase full --episodes $Episodes --log-every 25000
        if ($LASTEXITCODE -ne 0) { throw "$variant full training failed." }
    }
}
& $Python -m ibe_map.experiment evaluate
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed. If cache identity changed, use -Retrain.' }
& $Python -m ibe_map.verify_results
if ($LASTEXITCODE -ne 0) { throw 'Full trajectory verification failed.' }
Write-Host "Done. Open $PSScriptRoot\output\ibe_map\report.html"
