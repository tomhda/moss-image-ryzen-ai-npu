param(
    [ValidateSet(512, 1024, 'all')][string]$Size = '512',
    [string]$NpuEnv = 'mossnpu-npu',
    [string]$TextEnv = 'mossnpu-text',
    [string]$RyzenPath = 'C:\Program Files\RyzenAI\1.8.0',
    [string]$MossPath = '',
    [string]$KleinPath = '',
    [string]$HfRevision = '',
    [switch]$SkipDownloads
)
# 環境を2つ作り、部品と上流の必要ファイルを固定した版で落とし、configs/local.json を作る (PowerShell 7)。
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if ($MossPath -eq '') { $MossPath = Join-Path $root 'models\MOSS-Image-SFT' }
if ($KleinPath -eq '') { $KleinPath = Join-Path $root 'models\klein' }
$example = Get-Content (Join-Path $root 'configs\example.json') -Raw | ConvertFrom-Json
if ($HfRevision -eq '') { $HfRevision = $example.hf_revision }
if ($HfRevision -eq 'REPLACE_WITH_HF_COMMIT') { Write-Error 'HF revision is not pinned in configs/example.json'; exit 2 }
$conda = (Get-Command conda.exe -ErrorAction SilentlyContinue).Source
if (-not $conda) { foreach ($c in @("$env:USERPROFILE\miniforge3\Scripts\conda.exe", 'C:\Program Files\Miniforge3\Scripts\conda.exe')) { if (Test-Path $c) { $conda = $c; break } } }
if (-not $conda) { Write-Error 'conda.exe not found'; exit 2 }
$condaRoot = Split-Path (Split-Path $conda)
function Env-Python($name) { return Join-Path $condaRoot "envs\$name\python.exe" }
function Ensure-CondaEnv($name) { (Test-Path (Env-Python $name)) }
# --- NPU env: Ryzen AI env.yaml + AMD wheels + pinned additions ---
if (Ensure-CondaEnv $NpuEnv) {
    Write-Host "NPU env exists: $NpuEnv"
} else {
    $tmpYaml = Join-Path $env:TEMP "mossnpu-env-$NpuEnv.yaml"
    (Get-Content (Join-Path $RyzenPath 'env.yaml') -Raw) -replace 'name:\s*\S+', "name: $NpuEnv" | Set-Content -Encoding utf8 $tmpYaml
    & $conda env create -f $tmpYaml
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$npuPy0 = Env-Python $NpuEnv
$wheels = Get-ChildItem (Join-Path $RyzenPath '*.whl') | ForEach-Object { $_.FullName }
& $npuPy0 -m pip install @wheels
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $npuPy0 -m pip install -r (Join-Path $root 'environments\npu-extra.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# --- Text env ---
if (Ensure-CondaEnv $TextEnv) {
    Write-Host "Text env exists: $TextEnv"
} else {
    & $conda create -n $TextEnv -y python=3.12.16 pip
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$textPy0 = Env-Python $TextEnv
& $textPy0 -m pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $textPy0 -m pip install -r (Join-Path $root 'environments\text.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$npuPy = Env-Python $NpuEnv; $textPy = Env-Python $TextEnv
$partsDir = Join-Path $root 'parts'
New-Item -ItemType Directory -Force -Path $partsDir | Out-Null
if (-not $SkipDownloads) {
    & $textPy -u (Join-Path $root 'tools\download_models.py') --size $Size --parts-dir $partsDir --moss-dir $MossPath --klein-dir $KleinPath `
        --hf-repo $example.hf_repo --hf-revision $HfRevision --moss-revision $example.moss_revision --klein-revision $example.klein_revision
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$genai = Join-Path $RyzenPath 'GenAI-SD'
$local = [ordered]@{
    npu_python = $npuPy; text_python = $textPy
    ryzen_ai_path = $RyzenPath; genai_sd_path = $genai
    dd_root = (Join-Path $genai 'lib'); dd_plugins_root = (Join-Path $genai 'lib\transaction\stx')
    moss_path = $MossPath; klein_vae_dd_path = (Join-Path $KleinPath 'vae_decoder\dynamic\dd')
    parts_path = $partsDir
    moss_revision = $example.moss_revision; lora_revision = $example.lora_revision; klein_revision = $example.klein_revision
    hf_repo = $example.hf_repo; hf_revision = $HfRevision
}
$local | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $root 'configs\local.json')
Write-Host "wrote configs/local.json (NPU=$NpuEnv Text=$TextEnv Size=$Size)"
