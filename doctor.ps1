param(
    [ValidateSet(512, 1024)][int]$Size = 512
)
# 環境と部品の点検 (PowerShell 7)。結果はそのまま表示する。
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
Write-Host '== System =='
try { Get-CimInstance Win32_Processor | ForEach-Object { Write-Host "CPU: $($_.Name) cores=$($_.NumberOfCores) logical=$($_.NumberOfLogicalProcessors)" } } catch { Write-Host "CPU: $_" }
try { $os = Get-CimInstance Win32_OperatingSystem; Write-Host "Windows: $($os.Caption) $($os.Version) build $($os.BuildNumber) RAM_GB=$([math]::Round($os.TotalVisibleMemorySize/1MB,1)) FreeMB=$([math]::Round((Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue))" } catch { Write-Host "OS: $_" }
try {
    Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.DeviceName -match 'NPU|Neural|XDNA' } |
        ForEach-Object { Write-Host "NPU driver: $($_.DeviceName) $($_.DriverVersion) $($_.DriverDate)" }
    Get-PnpDevice -Class ComputeAccelerator -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "Accelerator: $($_.FriendlyName) $($_.Status)" }
} catch { Write-Host "NPU: $_" }
$configPath = Join-Path $root 'configs\local.json'
if (-not (Test-Path $configPath)) { Write-Host 'configs/local.json: MISSING (run setup.ps1)'; exit 2 }
$cfg = Get-Content $configPath -Raw | ConvertFrom-Json
$configDir = Split-Path $configPath
function Resolve-Cfg([string]$p) { if ([System.IO.Path]::IsPathRooted($p)) { $p } else { [System.IO.Path]::GetFullPath((Join-Path $configDir $p)) } }
Write-Host '== Paths =='
foreach ($k in @('npu_python', 'text_python', 'ryzen_ai_path', 'genai_sd_path', 'dd_root', 'dd_plugins_root', 'moss_path', 'klein_vae_dd_path', 'parts_path')) {
    $p = Resolve-Cfg $cfg.$k
    Write-Host "$k : $p exists=$(Test-Path $p)"
}
Write-Host '== Python packages =='
foreach ($pair in @(@('npu', (Resolve-Cfg $cfg.npu_python)), @('text', (Resolve-Cfg $cfg.text_python)))) {
    Write-Host "-- $($pair[0]) $($pair[1])"
    & $pair[1] -c "import sys, importlib.metadata as m; print(sys.version); [print(f'{d}=={m.version(d)}') for d in ('torch','numpy','onnxruntime-vitisai','transformers','diffusers','safetensors') if True]" 2>&1 |
        ForEach-Object { Write-Host "  $_" }
}
Write-Host '== Parts manifest check =='
$parts = Resolve-Cfg $cfg.parts_path
$textPy = Resolve-Cfg $cfg.text_python
& $textPy -c "import json,hashlib,sys; from pathlib import Path; root=Path(sys.argv[1]); m=json.loads((root/'manifest.json').read_text()); bad=[e['path'] for e in m['files'] if not (root/e['path']).exists() or (root/e['path']).stat().st_size!=e['size']]; print(f\"files={len(m['files'])} missing_or_size_mismatch={bad[:10]}\"); print('github_commit', m.get('github_commit'))" $parts 2>&1 | ForEach-Object { Write-Host $_ }
Write-Host "== NPU probe size $Size =="
$npuPy = Resolve-Cfg $cfg.npu_python
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $npuPy; $psi.UseShellExecute = $false; $psi.WorkingDirectory = $root
[void]$psi.ArgumentList.Add('-u'); [void]$psi.ArgumentList.Add((Join-Path $root 'src\doctor_probe.py')); [void]$psi.ArgumentList.Add('--size'); [void]$psi.ArgumentList.Add("$Size")
$prefix = Split-Path $npuPy
$psi.Environment['PATH'] = "$prefix;$prefix\Library\bin;$prefix\Scripts;$env:PATH"
$psi.Environment['PYTHONUTF8'] = '1'
$psi.Environment['MOSSNP_MOSS'] = Resolve-Cfg $cfg.moss_path
$psi.Environment['MOSSNP_PARTS'] = $parts
$psi.Environment['MOSSNP_RYZEN'] = Resolve-Cfg $cfg.ryzen_ai_path
$psi.Environment['MOSSNP_GENAI_SD'] = Resolve-Cfg $cfg.genai_sd_path
$psi.Environment['MOSSNP_KLEIN_VAE_DD'] = Resolve-Cfg $cfg.klein_vae_dd_path
$proc = [System.Diagnostics.Process]::Start($psi); $proc.WaitForExit()
Write-Host "probe exit $($proc.ExitCode)"
Write-Host "FreeMB now $([math]::Round((Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue)) (1024 needs largest group 14 pairs; see gen memory check)"
exit $proc.ExitCode
