# 在服务器上重跑 PACER 训练探针，并把全过程日志拉回本地 logs/。
#
# 用法（在本地仓库根目录执行）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File script\pacer_train_probe.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File script\pacer_train_probe.ps1 -Envs 128 -Iters 5
#
# 说明：
#   - 服务器端脚本不修改任何源码；只临时生成 print_stats: True 的训练配置（退出时删除）
#   - 输出写进独立的 output/exp/pacer_<tag>，不碰 ljx 的 pacer_stage1_stairs
#   - 日志最终落在 <repo>\logs\pacer_<tag>.log

param(
    [string]$Tag = "probe",
    [int]$Envs = 1024,
    [int]$Iters = 500,
    [int]$Gpu = 3,
    [int]$TimeoutSec = 3600,
    [int]$PollSec = 20,
    [string]$HostAlias = "hoi"
)

$ErrorActionPreference = "Stop"
$ssh = "C:\Windows\System32\OpenSSH\ssh.exe"
$scp = "C:\Windows\System32\OpenSSH\scp.exe"
$repo = Split-Path -Parent $PSScriptRoot
$localLogDir = Join-Path $repo "logs"
$localScript = Join-Path $PSScriptRoot "pacer_train_probe.sh"
$remoteScript = "/data/ljx/tmp/pacer_train_probe.sh"
$remoteLog = "/data/ljx/tmp/pacer_$Tag.log"
$localLog = Join-Path $localLogDir "pacer_$Tag.log"

if (-not (Test-Path $localLogDir)) { New-Item -ItemType Directory -Force -Path $localLogDir | Out-Null }

function Invoke-Remote([string]$scriptText) {
    $lf = $scriptText -replace "`r`n", "`n"
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($lf))
    return (& $ssh -o BatchMode=yes -o LogLevel=ERROR $HostAlias "echo $b64 | base64 -d | bash" 2>&1)
}

# 1) 上传脚本 + 强制 LF + 语法检查
Write-Host "[1/4] upload + syntax check ..."
& $scp -o BatchMode=yes -o LogLevel=ERROR $localScript ("hoi:" + $remoteScript) | Out-Null
$chk = Invoke-Remote "sed -i 's/\r$//' $remoteScript; chmod +x $remoteScript; bash -n $remoteScript && echo SYNTAX_OK"
Write-Host ("      " + (($chk | Out-String).Trim()))

# 2) 后台启动
Write-Host "[2/4] launching ..."
$launchTpl = @'
rm -f {LOG}
cd /data/ljx/tmp
nohup bash {SCRIPT} {TAG} {ENVS} {ITERS} {GPU} {TMO} > /data/ljx/tmp/{TAG}_launch.log 2>&1 &
sleep 2
echo LAUNCHED
pgrep -af "pacer/run.py" | head -n 2
'@
$launch = $launchTpl.Replace("{LOG}", $remoteLog).Replace("{SCRIPT}", $remoteScript).
    Replace("{TAG}", $Tag).Replace("{ENVS}", "$Envs").Replace("{ITERS}", "$Iters").
    Replace("{GPU}", "$Gpu").Replace("{TMO}", "$TimeoutSec")
$out = Invoke-Remote $launch
Write-Host ("      " + (($out | Out-String).Trim()))
Write-Host "      envs=$Envs iters=$Iters gpu=$Gpu timeout=${TimeoutSec}s"
Write-Host "      remote log: $remoteLog"

# 3) 轮询并实时打印新增日志
Write-Host "[3/4] running (poll every ${PollSec}s) ..."
$seen = 0
$deadline = (Get-Date).AddSeconds($TimeoutSec + 240)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds $PollSec
    $state = Invoke-Remote "pgrep -f 'exp/pacer_$Tag' >/dev/null && echo RUNNING || echo DONE"
    $wcline = Invoke-Remote "if [ -f $remoteLog ]; then wc -l < $remoteLog; else echo 0; fi"
    $lines = 0
    [void][int]::TryParse((($wcline | Out-String).Trim()), [ref]$lines)

    if ($lines -gt $seen) {
        $chunk = Invoke-Remote "tail -n +$($seen + 1) $remoteLog | head -n $($lines - $seen)"
        foreach ($l in $chunk) { Write-Host ("  | " + $l) }
        $seen = $lines
    }
    if ((($state | Out-String).Trim()) -eq "DONE") { Write-Host "      remote process finished"; break }
}

# 4) 拉回本地（主日志 + 启动壳日志，后者兜底）
Write-Host "[4/4] fetching log ..."
$remoteLaunch = "/data/ljx/tmp/pacer_${Tag}_launch.log"
$localLaunch = Join-Path $localLogDir "pacer_${Tag}_launch.log"
& $scp -o BatchMode=yes -o LogLevel=ERROR ("hoi:" + $remoteLog) $localLog 2>&1 | Out-Null
try {
    & $scp -o BatchMode=yes -o LogLevel=ERROR ("hoi:" + $remoteLaunch) $localLaunch 2>&1 | Out-Null
} catch {
    Write-Host "      (launch shell log not present, skipped)"
}
if (Test-Path $localLog) {
    Write-Host ""
    Write-Host ("LOG SAVED -> " + $localLog + " (" + (Get-Item $localLog).Length + " bytes)")
    Write-Host "---------- tail 30 ----------"
    Get-Content $localLog -Tail 30
} else {
    Write-Warning ("log not found on remote: " + $remoteLog)
}
