param(
    [ValidateSet('all', 'plan', 'train', 'curves', 'final', 'summarize', 'report')]
    [string]$Stage = 'all',
    [int[]]$Seeds = @(42, 43, 44, 45, 46),
    [int]$Episodes = 300000,
    [int]$SaveEvery = 30000,
    [string]$Out = 'output\ibe_multiseed',
    [string]$Python = 'D:\anaconda3\envs\uav\python.exe'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if ($Stage -in @('all', 'train')) {
    & $Python -m unittest map_baseline.test_environment ibe_map.test_recurrence ibe_map.test_sam ibe_map.test_sam_attention ibe_map.test_snapshots ibe_map.test_multiseed
    if ($LASTEXITCODE -ne 0) { throw 'Experiment regression checks failed.' }
}
if ($Stage -ne 'report') {
    & $Python -m ibe_map.multiseed --stage $Stage --seeds $Seeds --episodes $Episodes --save-every $SaveEvery --out $Out
    if ($LASTEXITCODE -ne 0) { throw 'Repeated-seed experiment failed; inspect the per-job logs.' }
}
if ($Stage -in @('all', 'summarize', 'report')) {
    $summaryPath = Join-Path $Out 'summary.json'
    if ($Stage -ne 'report' -and $Episodes -eq 300000 -and $SaveEvery -eq 30000 -and $Seeds.Count -ge 2) {
        & $Python -m ibe_map.verify_multiseed --out $Out
        if ($LASTEXITCODE -ne 0) { throw 'Independent repeated-seed summary verification failed.' }
    }
    & $Python -c 'import sys; from ibe_map.multiseed_report import render_report; print(render_report(sys.argv[1],sys.argv[2]))' $summaryPath $Out
    if ($LASTEXITCODE -ne 0) { throw 'Repeated-seed report rendering failed.' }
    Write-Host "Report: $(Join-Path $Out 'report.html')"
}
