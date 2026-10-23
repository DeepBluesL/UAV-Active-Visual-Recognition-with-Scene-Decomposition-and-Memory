param(
    [ValidateSet('all', 'development', 'final', 'verify', 'report')]
    [string]$Stage = 'all',
    [string]$Out = 'output\active_evidence_ablations',
    [string]$Source = 'output\active_evidence',
    [string]$Python = 'D:\anaconda3\envs\uav\python.exe'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable not found; pass -Python.' }
if ($Stage -in @('all', 'development', 'final')) {
    & $Python -m unittest active_evidence.test_model active_evidence.test_ablation_model
    if ($LASTEXITCODE -ne 0) { throw 'Ablation model tests failed.' }
    & $Python -m active_evidence.ablations --stage $Stage --out $Out --source $Source
    if ($LASTEXITCODE -ne 0) { throw 'Ablation experiment failed.' }
}
if ($Stage -in @('all', 'final', 'verify')) {
    & $Python -m active_evidence.verify_ablations --out $Out --source $Source
    if ($LASTEXITCODE -ne 0) { throw 'Independent ablation verification failed.' }
    & $Python -m active_evidence.ablations --stage audit --out $Out --source $Source
    if ($LASTEXITCODE -ne 0) { throw 'Ablation model replay failed.' }
}
if ($Stage -in @('all', 'final', 'verify', 'report')) {
    & $Python -m active_evidence.ablation_report --out $Out
    if ($LASTEXITCODE -ne 0) { throw 'Ablation report generation failed.' }
}
