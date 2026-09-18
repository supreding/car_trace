param(
    [ValidateSet("CPU", "GPU")]
    [string]$PaddleDevice = "GPU",
    [string]$FastReIDDevice = "cuda:0",
    [int]$Limit = 0,
    [switch]$Force,
    [switch]$NoAttr,
    [string]$Db = "",   # 指定 SQLite 库文件（默认 app\granary.db）
    [string]$OutputDir = ""   # 车辆裁剪图输出目录（默认 app\reid_crops；多园区建议每库独立目录，如 app\reid_crops_xxx）
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot ".venv-reid\Scripts\python.exe"
$paddleRoot = Join-Path $projectRoot "third_party\PaddleDetection"
$fastReIDRoot = Join-Path $projectRoot "third_party\fast-reid"
$runtimeHome = Join-Path $projectRoot ".reid-home"
$matplotlibConfig = Join-Path $projectRoot ".mpl"

if (-not (Test-Path -LiteralPath $python)) {
    throw "ReID Python 环境不存在：$python"
}
New-Item -ItemType Directory -Force -Path $runtimeHome, $matplotlibConfig | Out-Null

# 注意：GPU 版 torch 与 paddlepaddle-gpu 的 cuDNN DLL 不能同进程共存，
# reid_pipeline.py --stage all 会自动拆成 detect / embed 两个子进程，
# 因此这里不再把 torch\lib 预置到 PATH。
$env:PADDLE_DETECTION_ROOT = $paddleRoot
$env:PYTHONPATH = "$fastReIDRoot;$paddleRoot"
$env:HOME = $runtimeHome
$env:USERPROFILE = $runtimeHome
$env:MPLCONFIGDIR = $matplotlibConfig

$dbPath = if ($Db) { (Resolve-Path -LiteralPath $Db).Path } else { Join-Path $projectRoot "app\granary.db" }
if (-not (Test-Path -LiteralPath $dbPath)) {
    throw "数据库不存在：$dbPath（先用 build_db.py 建库，或用 -Db 指定库文件）"
}
$outputDir = if ($OutputDir) { $OutputDir } else { Join-Path $projectRoot "app\reid_crops" }

$arguments = @(
    (Join-Path $projectRoot "app\reid_pipeline.py"),
    "--db", $dbPath,
    "--detector-model", (Join-Path $projectRoot "models\paddledetection\mot_ppyoloe_l_36e_ppvehicle"),
    "--attr-model", (Join-Path $projectRoot "models\paddledetection\vehicle_attribute"),
    "--plate-det-model", (Join-Path $projectRoot "models\paddledetection\ch_PP-OCRv3_det_infer"),
    "--plate-rec-model", (Join-Path $projectRoot "models\paddledetection\ch_PP-OCRv3_rec_infer"),
    "--fastreid-config", (Join-Path $projectRoot "third_party\fast-reid\configs\VeRi\sbs_R50-ibn.yml"),
    "--fastreid-weights", (Join-Path $projectRoot "models\fastreid\veri_sbs_R50-ibn.pth"),
    "--output-dir", $outputDir,
    "--paddle-device", $PaddleDevice,
    "--device", $FastReIDDevice
)
if ($Limit -gt 0) { $arguments += @("--limit", $Limit) }
if ($Force) { $arguments += "--force" }
if ($NoAttr) { $arguments += "--no-attr" }

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "PaddleDetection + FastReID 流水线执行失败（退出码 $LASTEXITCODE）"
}
