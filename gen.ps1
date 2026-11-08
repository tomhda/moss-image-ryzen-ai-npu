param(
    [Parameter(Mandatory = $true)][string]$Items,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$DitExtra = '',
    [int]$VaeThreads = 1,
    [int]$TextThreads = 1,
    [ValidateSet(512, 1024)][int]$Size = 512,
    [ValidateSet('', 'cpu', 'npu')][string]$Vae = ''
)
# 入口: 文字 -> DiT -> VAE を別プロセスで順に回す (PowerShell 7)。
# 終了コード: 0 = 全部できた, 3 = SKIP があった (ほかはできた), それ以外 = 失敗。
if ($Vae -eq '') { $Vae = if ($Size -eq 1024) { 'npu' } else { 'cpu' } }
if ($Vae -eq 'npu' -and $Size -ne 1024) { [Console]::Error.WriteLine('NPU VAE requires -Size 1024'); exit 2 }
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$configPath = Join-Path $root 'configs\local.json'
if (-not (Test-Path $configPath)) { [Console]::Error.WriteLine('configs/local.json not found. Run setup.ps1 first.'); exit 2 }
$configDir = Split-Path $configPath
$cfg = Get-Content $configPath -Raw | ConvertFrom-Json
function Resolve-Cfg([string]$p) {
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return [System.IO.Path]::GetFullPath((Join-Path $configDir $p))
}
function Resolve-Caller([string]$p) {
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return [System.IO.Path]::GetFullPath((Join-Path $PWD.Path $p))
}
$Items = Resolve-Caller $Items
$Out = Resolve-Caller $Out
if (-not (Test-Path $Items)) { [Console]::Error.WriteLine("Items not found: $Items"); exit 2 }
# 重複 id+seed は始める前に断る
$itemsData = Get-Content $Items -Raw | ConvertFrom-Json
$seen = @{}
foreach ($it in $itemsData) {
    $k = "$($it.id)|$($it.seed)"
    if ($seen.ContainsKey($k)) { [Console]::Error.WriteLine("duplicate id+seed: $k"); exit 2 }
    $seen[$k] = $true
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$runId = [guid]::NewGuid().ToString('N')
$lockPath = Join-Path $Out '.gen.lock'
try {
    # ロックは開いている間だけ有効にする。途中で止まっても、プロセスが消えれば回し直せる。
    $lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    $lock.SetLength(0)
    $sw = New-Object System.IO.StreamWriter($lock); $sw.Write($runId); $sw.Flush()
} catch { [Console]::Error.WriteLine("output is locked: $Out"); exit 2 }
function FreeMB { try { (Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue } catch { 0 } }
$basePath = $env:PATH
function RunStage($label, $python, $script, $extra, $isNpu) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $python
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $false
    $psi.RedirectStandardError = $false
    $psi.WorkingDirectory = $PWD.Path
    [void]$psi.ArgumentList.Add('-u')
    [void]$psi.ArgumentList.Add((Join-Path $root "src\$script"))
    foreach ($x in $extra) { [void]$psi.ArgumentList.Add($x) }
    $prefix = Split-Path $python
    $psi.Environment['PATH'] = "$prefix;$prefix\Library\bin;$prefix\Scripts;$basePath"
    $psi.Environment['PYTHONUTF8'] = '1'
    $psi.Environment['PYTHONDONTWRITEBYTECODE'] = '1'
    $psi.Environment['MOSSNP_MOSS'] = Resolve-Cfg $cfg.moss_path
    $psi.Environment['MOSSNP_PARTS'] = Resolve-Cfg $cfg.parts_path
    $psi.Environment['MOSSNP_MODEL_REV'] = $cfg.moss_revision
    if ($isNpu) {
        $ryzen = Resolve-Cfg $cfg.ryzen_ai_path
        $genai = Resolve-Cfg $cfg.genai_sd_path
        $psi.Environment['RYZEN_AI_INSTALLATION_PATH'] = $ryzen
        $psi.Environment['MOSSNP_RYZEN'] = $ryzen
        $psi.Environment['MOSSNP_GENAI_SD'] = $genai
        $psi.Environment['MOSSNP_KLEIN_VAE_DD'] = Resolve-Cfg $cfg.klein_vae_dd_path
        $psi.Environment['DD_ROOT'] = Resolve-Cfg $cfg.dd_root
        $psi.Environment['DD_PLUGINS_ROOT'] = Resolve-Cfg $cfg.dd_plugins_root
    }
    if ($env:MOSSNP_FAIL_WORKER) { $psi.Environment['MOSSNP_FAIL_WORKER'] = $env:MOSSNP_FAIL_WORKER }
    $freeBefore = [math]::Round((FreeMB))
    Write-Host "=== $label free ${freeBefore}MB run $runId ==="
    $t0 = Get-Date
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.WaitForExit()
    $code = $proc.ExitCode
    $wall = [math]::Round(((Get-Date) - $t0).TotalSeconds, 2)
    $freeAfter = [math]::Round((FreeMB))
    $cpuFile = Join-Path $Out ("{0}_cpu_{1}.json" -f $label, $runId)
    $cpu = $null; $workerCpu = $null
    if (Test-Path $cpuFile) {
        try { $cj = Get-Content $cpuFile -Raw | ConvertFrom-Json; $cpu = $cj.cpu_s; $workerCpu = $cj.worker_cpu_s } catch {}
    }
    Write-Host "=== $label done wall ${wall}s cpu ${cpu}s exit $code free ${freeAfter}MB ==="
    return [pscustomobject]@{ stage = $label; wall_s = $wall; cpu_s = $cpu; worker_cpu_s = $workerCpu; exit_code = $code; free_before_mb = $freeBefore; free_after_mb = $freeAfter; run_id = $runId }
}
$stages = @()
$finalCode = 0
try {
    $textPy = Resolve-Cfg $cfg.text_python
    $npuPy = Resolve-Cfg $cfg.npu_python
    $s = RunStage 'text' $textPy 'stage_text.py' @('--items', $Items, '--out', $Out, '--threads', "$TextThreads", '--size', "$Size", '--run-id', $runId) $false
    $stages += $s
    if ($s.exit_code -ne 0) { $finalCode = $s.exit_code } else {
        $ditArgs = @('--items', $Items, '--out', $Out, '--size', "$Size", '--run-id', $runId)
        if ($DitExtra -ne '') { $ditArgs += ($DitExtra -split '\s+' | Where-Object { $_ -ne '' }) }
        $s = RunStage 'dit' $npuPy 'stage_dit.py' $ditArgs $true
        $stages += $s
        if ($s.exit_code -ne 0) { $finalCode = $s.exit_code } else {
            if ($Vae -eq 'npu') {
                $s = RunStage 'vae' $npuPy 'stage_vae_npu.py' @('--out', $Out, '--intra', "$VaeThreads", '--size', "$Size", '--run-id', $runId) $true
            } else {
                $s = RunStage 'vae' $textPy 'stage_vae.py' @('--out', $Out, '--threads', "$VaeThreads", '--size', "$Size", '--run-id', $runId) $false
            }
            $stages += $s
            if ($s.exit_code -ne 0) { $finalCode = $s.exit_code }
            elseif ($Vae -eq 'npu') {
                $markerPath = Join-Path $Out "vae_done_$runId.json"
                $okMarker = $false
                if (Test-Path $markerPath) {
                    try {
                        $mk = Get-Content $markerPath -Raw | ConvertFrom-Json
                        if ($mk.run_id -eq $runId) {
                            $okMarker = $true
                            foreach ($mi in $mk.items) {
                                $png = Join-Path $Out $mi.png
                                if (-not (Test-Path $png)) { $okMarker = $false; break }
                                $h = (Get-FileHash $png -Algorithm SHA256).Hash.ToLower()
                                if ($h -ne $mi.sha256.ToLower()) { $okMarker = $false; break }
                            }
                        }
                    } catch { $okMarker = $false }
                }
                if (-not $okMarker) { Write-Error 'VAE(NPU) completion marker check failed'; $finalCode = 4 }
            }
        }
    }
    if ($finalCode -eq 0) {
        $manifestPath = Join-Path $Out 'text_manifest.json'
        if (Test-Path $manifestPath) {
            $mf = Get-Content $manifestPath -Raw | ConvertFrom-Json
            if ($mf.run_id -ne $runId) { $finalCode = 4 }
            elseif ($mf.failed -and $mf.failed.Count -gt 0) { $finalCode = 3 }
        } else { $finalCode = 4 }
    }
} finally {
    $stages | ForEach-Object { $_ | Add-Member -NotePropertyName size -NotePropertyValue $Size -Force }
    try { $stages | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $Out 'gen_summary.json') } catch {}
    try { $sw.Close(); $lock.Close(); Remove-Item $lockPath -Force -ErrorAction SilentlyContinue } catch {}
}
$stages | Format-Table | Out-String | Write-Host
exit $finalCode
