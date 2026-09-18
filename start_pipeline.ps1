# 识别流水线一键启动脚本（PaddleDetection 检测/属性/车牌 + FastReID 特征，需 NVIDIA GPU）
#
# 用法：
#   双击 启动识别流水线.bat
#   或 powershell -NoProfile -ExecutionPolicy Bypass -File start_pipeline.ps1 [参数]
#
# 做五件事：
#   1. 检查/创建 .venv-reid 环境（必须是 Python 3.12，torch cu126 轮子为 cp312）；
#   2. 校验流水线依赖（缺了按 requirements-reid.txt 自动补装）；
#   3. 校验 GPU 与流水线必备目录（models/、third_party/，打包漏带会明确报错）；
#   4. 新照片同步入库：build_db.py 扫描 data/<园区>/ + OCR 时间戳 + 订单内车辆聚合（-SkipSync 可跳过）；
#   5. 调用 run_reid_pipeline.ps1 跑 GPU 检测、裁剪、属性、车牌与 FastReID 特征。
#
# 与 启动查询服务.bat 的关系：查询服务不跑任何识别，本脚本才跑模型；
# 新机器没有 NVIDIA 显卡时不要用本脚本（查询服务不受影响）。
param(
    [string]$Db = "",            # 目标 SQLite 库（默认 app\granary.db；多园区每园区一个库）
    [string]$OutputDir = "",     # 车辆裁剪图目录（默认 app\reid_crops；多园区建议每库独立目录）
    [string]$Park = "",          # 只同步 data/ 下指定园区的照片（对应 build_db --park）
    [switch]$SkipSync,           # 跳过照片同步/OCR，直接跑 GPU 流水线
    [int]$Limit = 0,             # 本次最多处理多少张（试跑用，0=不限）
    [switch]$Force,              # 已处理过的也强制重跑
    [switch]$NoAttr,             # 跳过车辆属性模型
    [switch]$AllowCpu            # 没有 NVIDIA GPU 时强制用 CPU 跑（极慢，仅应急）
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$appDir = Join-Path $projectRoot "app"

# PS5.1 在 $ErrorActionPreference="Stop" 时，原生命令的 stderr 重定向（2>&1/2>$null）
# 会把 findstr 包装器对空输入的提示误报为终止错误；采集原生命令输出统一走这里。
function Invoke-NativeQuiet {
    param([scriptblock]$Command)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        return (& $Command 2>&1 | Out-String)
    } finally {
        $ErrorActionPreference = $prev
    }
}

# ---- 1. Python 3.12 环境 ----
$venvDir = Join-Path $projectRoot ".venv-reid"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    $ver = (& $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($ver -ne "3.12") {
        throw ".venv-reid 是 Python $ver，流水线依赖要求 3.12（torch cu126 轮子为 cp312）。" +
              "请删除 .venv-reid 目录后重跑，让脚本用 Python 3.12 重建。"
    }
} else {
    Write-Host "未找到 .venv-reid，正在查找 Python 3.12 并创建环境..."
    $pyCmd = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $v = Invoke-NativeQuiet { & py -3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" }
        if ("$v".Trim() -eq "3.12") { $pyCmd = @("py", "-3.12") }
    }
    if (-not $pyCmd -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $v = Invoke-NativeQuiet { & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" }
        if ("$v".Trim() -eq "3.12") { $pyCmd = @("python") }
    }
    if (-not $pyCmd) {
        throw "未找到 Python 3.12。请先安装 Python 3.12 并勾选加入 PATH，再重跑本脚本。"
    }
    if ($pyCmd.Count -eq 2) { & $pyCmd[0] $pyCmd[1] -m venv $venvDir } else { & $pyCmd[0] -m venv $venvDir }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
        throw "创建虚拟环境失败（退出码 $LASTEXITCODE）"
    }
}

# ---- 2. 流水线依赖校验（缺了自动按 requirements-reid.txt 补装）----
# 注意：torch 与 paddlepaddle-gpu 的 cuDNN DLL 不能同进程共存（见 run_reid_pipeline.ps1 注释），
# 必须分开两个进程检查，放同一条 import 里会误报。
Invoke-NativeQuiet { & $venvPython -c "import torch, cv2, numpy, PIL" } | Out-Null
$torchOk = ($LASTEXITCODE -eq 0)
Invoke-NativeQuiet { & $venvPython -c "import paddle, cv2, numpy, PIL" } | Out-Null
$paddleOk = ($LASTEXITCODE -eq 0)
if (-not $torchOk -or -not $paddleOk) {
    Write-Host "流水线依赖缺失，开始安装（torch cu126 约 2.5 GB，请耐心等待）..."
    & $venvPython -m pip install --disable-pip-version-check `
        -r (Join-Path $projectRoot "requirements-reid.txt") `
        --find-links (Join-Path $projectRoot "_wheels") `
        --extra-index-url https://download.pytorch.org/whl/cu126 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "依赖安装失败。目标机无公网时，先在开发机执行：" +
              "python -m pip download -r requirements-reid.txt --extra-index-url https://download.pytorch.org/whl/cu126 -d _wheels，" +
              "把整个项目拷过去后改用：pip install --no-index --find-links=_wheels -r requirements-reid.txt"
    }
}

# ---- 3. GPU 与必备目录校验 ----
$cuda = (Invoke-NativeQuiet { & $venvPython -c "import torch; print(torch.cuda.is_available())" })
$gpuOk = ("$cuda".Trim() -eq "True")
if (-not $gpuOk -and -not $AllowCpu) {
    throw "未检测到可用的 NVIDIA GPU（torch.cuda.is_available()=$cuda）。" +
          "识别流水线按 GPU 配置，请确认显卡驱动；确要用 CPU 应急请加 -AllowCpu（极慢）。"
}
$required = @(
    "models\paddledetection\mot_ppyoloe_l_36e_ppvehicle",
    "models\paddledetection\vehicle_attribute",
    "models\paddledetection\ch_PP-OCRv3_det_infer",
    "models\paddledetection\ch_PP-OCRv3_rec_infer",
    "models\fastreid\veri_sbs_R50-ibn.pth",
    "third_party\PaddleDetection",
    "third_party\fast-reid"
)
$missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $projectRoot $_)) })
if ($missing) {
    throw "流水线必备目录缺失（打包时漏带）：$($missing -join '、')"
}

# ---- 4. 目标库 ----
$dbPath = if ($Db) { $Db } else { Join-Path $appDir "granary.db" }
if ($SkipSync -and -not (Test-Path -LiteralPath $dbPath)) {
    throw "数据库不存在：$dbPath（-SkipSync 需要库已建好；不带 -SkipSync 会自动建）"
}

$engine = if ($gpuOk) { "GPU" } else { "CPU（-AllowCpu 应急模式，极慢）" }
Write-Host ""
Write-Host "================================================"
Write-Host " 园区车辆识别流水线"
Write-Host "   数据库：      $dbPath"
Write-Host "   裁剪图目录：  $(if ($OutputDir) { $OutputDir } else { 'app\reid_crops（默认）' })"
Write-Host "   园区过滤：    $(if ($Park) { $Park } else { 'data/ 下全部园区' })"
Write-Host "   照片同步/OCR：$(if ($SkipSync) { '跳过（-SkipSync）' } else { '执行' })"
Write-Host "   计算设备：    $engine"
Write-Host "================================================"
Write-Host ""

# ---- 5. 照片同步入库（扫描 + OCR 时间戳/OSD + 订单内车辆聚合）----
if (-not $SkipSync) {
    $syncArgs = @((Join-Path $appDir "build_db.py"), "--db", $dbPath)
    if ($Park) { $syncArgs += @("--park", $Park) }
    if ($Limit -gt 0) { $syncArgs += @("--limit", $Limit) }
    & $venvPython @syncArgs
    if ($LASTEXITCODE -ne 0) {
        throw "照片同步/OCR 失败（退出码 $LASTEXITCODE）"
    }
}

# ---- 6. GPU 流水线（检测/裁剪/属性/车牌 + FastReID 特征）----
# 注意：对 ps1 脚本必须用字典 splatting（按参数名绑定）；数组 splatting 会按位置
# 误绑到 PaddleDevice 上。
$pipelineArgs = @{ Db = $dbPath }
if ($OutputDir) { $pipelineArgs.OutputDir = $OutputDir }
if ($Limit -gt 0) { $pipelineArgs.Limit = $Limit }
if ($Force) { $pipelineArgs.Force = $true }
if ($NoAttr) { $pipelineArgs.NoAttr = $true }
if (-not $gpuOk) { $pipelineArgs.PaddleDevice = "CPU"; $pipelineArgs.FastReIDDevice = "cpu" }
& (Join-Path $projectRoot "run_reid_pipeline.ps1") @pipelineArgs
if ($LASTEXITCODE -ne 0) {
    throw "识别流水线执行失败（退出码 $LASTEXITCODE）"
}
Write-Host ""
Write-Host "流水线执行完成。查询服务（启动查询服务.bat）会直接看到新数据。"
