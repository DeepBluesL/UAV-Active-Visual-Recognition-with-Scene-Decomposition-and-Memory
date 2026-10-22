param(
    [string]$Python = 'D:\anaconda3\envs\uav\python.exe',
    [ValidateSet('all','plan','train','evaluate','report','publish')][string]$Stage = 'all',
    [string]$Out = 'output/e2e_multiseed',
    [int]$Microbatch = 64,
    [switch]$Publish
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($Stage -in @('all', 'train')) {
    & $Python -m unittest map_baseline.test_environment ibe_map.test_recurrence ibe_map.test_sam ibe_map.test_sam_attention ibe_map.test_snapshots ibe_map.test_multiseed e2e_map.test_model e2e_map.test_experiment e2e_map.test_report e2e_map.test_publish_results e2e_map.test_pipeline e2e_map.test_control
    if ($LASTEXITCODE -ne 0) { throw 'Regression checks failed' }
}
$arguments = @('-m', 'e2e_map.pipeline', '--stage', $Stage, '--out', $Out, '--microbatch', "$Microbatch")
if ($Publish) { $arguments += '--publish' }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'E2E pipeline failed; inspect output status.json and per-run logs' }
