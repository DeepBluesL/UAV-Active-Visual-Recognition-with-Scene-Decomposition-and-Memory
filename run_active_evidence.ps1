param(
    [ValidateSet('all', 'development', 'final', 'verify', 'report')]
    [string]$Stage = 'all',
    [string]$Out = 'output\active_evidence',
    [string]$Python = 'D:\anaconda3\envs\uav\python.exe'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable not found; pass -Python.' }
if ($Stage -in @('all', 'development', 'final')) {
    & $Python -m unittest active_evidence.test_model
    if ($LASTEXITCODE -ne 0) { throw 'Model tests failed.' }
    & $Python -m active_evidence.experiment --stage $Stage --out $Out
    if ($LASTEXITCODE -ne 0) { throw 'Active evidence experiment failed.' }
}
if ($Stage -in @('all', 'final', 'verify')) {
    & $Python -m active_evidence.verify --out $Out
    if ($LASTEXITCODE -ne 0) { throw 'Independent trajectory verification failed.' }
}
if ($Stage -in @('all', 'final', 'verify', 'report')) {
    & $Python -m active_evidence.report --out $Out
    if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
}
