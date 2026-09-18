# 园区车辆轨迹查询服务一键启动脚本
#
# 用法：
#   双击 启动查询服务.bat
#   或 powershell -NoProfile -ExecutionPolicy Bypass -File start_server.ps1 [参数]
#
# 常用参数：
#   -Port 5000            监听端口
#   -Bind 0.0.0.0         监听地址：0.0.0.0=局域网可访问；127.0.0.1=仅本机
#   -Db <路径>            指定 SQLite 库文件（默认 app\granary.db，也可用环境变量 GRANARY_DB）
#   -OpenApiToken <串>    开放接口 token（写入本进程环境变量；留空则沿用系统环境变量）
#   -NoBrowser            启动后不自动打开浏览器
#   -Dev                  用 Flask 开发服务器启动（默认用 waitress 生产引擎）
param(
    [int]$Port = 5000,
    [string]$Bind = "0.0.0.0",
    [string]$Db = "",
    [string]$OpenApiToken = "",
    [switch]$NoBrowser,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$appDir = Join-Path $projectRoot "app"

# ---- 1. 选择或创建 Python 环境（优先轻量 .venv-server，其次复用 .venv-reid）----
$serverVenvPython = Join-Path $projectRoot ".venv-server\Scripts\python.exe"
$reidVenvPython = Join-Path $projectRoot ".venv-reid\Scripts\python.exe"
if (Test-Path -LiteralPath $serverVenvPython) {
    $python = $serverVenvPython
    $venvName = ".venv-server"
} elseif (Test-Path -LiteralPath $reidVenvPython) {
    $python = $reidVenvPython
    $venvName = ".venv-reid"
} else {
    $venvName = ".venv-server"
    Write-Host "未找到现有虚拟环境，正在创建 $venvName ..."
    $creator = Get-Command "py" -ErrorAction SilentlyContinue
    if ($creator) {
        & py -3 -m venv (Join-Path $projectRoot $venvName)
    } else {
        & python -m venv (Join-Path $projectRoot $venvName)
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $serverVenvPython)) {
        throw "创建虚拟环境失败，请先安装 Python 3.10+ 并确保 py/python 在 PATH 中"
    }
    $python = $serverVenvPython
}
Write-Host "Python 环境：$venvName"

# ---- 2. 依赖检查，缺什么装什么（依赖清单见 requirements-server.txt）----
# PS5.1 在 $ErrorActionPreference="Stop" 时，原生命令的 stderr 重定向会误报终止错误，
# 采集原生命令输出统一走 Invoke-NativeQuiet。
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
$required = if ($Dev) { "flask numpy PIL" } else { "flask numpy PIL waitress" }
$missing = (Invoke-NativeQuiet { & $python -c "import importlib.util as u; print(' '.join(m for m in '$required'.split() if u.find_spec(m) is None))" }).Trim()
if ($LASTEXITCODE -ne 0) { $missing = $required }
if ($missing) {
    Write-Host "安装缺失依赖：$missing"
    Invoke-NativeQuiet { & $python -m pip install --disable-pip-version-check -r (Join-Path $projectRoot "requirements-server.txt") } | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "依赖安装失败。服务器无公网时，请先在有网机器下载轮子并离线安装 requirements-server.txt 中的包。"
    }
}

# ---- 3. 数据库检查 ----
if ($Db) {
    if (-not (Test-Path -LiteralPath $Db)) { throw "指定的数据库不存在：$Db" }
    $env:GRANARY_DB = (Resolve-Path -LiteralPath $Db).Path
}
$dbPath = if ($env:GRANARY_DB) { $env:GRANARY_DB } else { Join-Path $appDir "granary.db" }
if (-not (Test-Path -LiteralPath $dbPath)) {
    Write-Warning "未找到数据库：$dbPath —— 服务能启动但查询为空，可用 -Db 参数指定库文件位置"
}

# ---- 4. 开放接口 token ----
if ($OpenApiToken) { $env:OPEN_API_TOKEN = $OpenApiToken }
$tokenStatus = if ($env:OPEN_API_TOKEN) {
    $t = $env:OPEN_API_TOKEN
    if ($t.Length -gt 6) { $t = $t.Substring(0, 4) + "****" }
    "已启用（token=$t）"
} else {
    "未启用（/api/open/* 返回 503；如需对接第三方，设置 -OpenApiToken 或环境变量 OPEN_API_TOKEN）"
}

# ---- 5. 启动 ----
$env:HOST = $Bind
$env:PORT = "$Port"
$localIp = try {
    (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } |
        Select-Object -First 1).IPAddress
} catch { "<本机IP>" }

Write-Host ""
Write-Host "================================================"
Write-Host " 园区车辆轨迹查询服务"
Write-Host "   本机访问：  http://127.0.0.1:$Port/"
Write-Host "   局域网访问：http://$localIp`:$Port/  （其他机器）"
Write-Host "   标注页：    /annotate    健康自检：/api/health"
Write-Host "   数据库：    $dbPath"
Write-Host "   开放接口：  $tokenStatus"
Write-Host "   引擎：      $(if ($Dev) { 'Flask 开发服务器' } else { 'waitress（生产）' })"
Write-Host "   停止服务：  Ctrl+C"
Write-Host "================================================"
Write-Host ""

if (-not $NoBrowser) {
    Start-Job -ScriptBlock { param($u) Start-Sleep -Seconds 2; Start-Process $u } `
        -ArgumentList "http://127.0.0.1:$Port/" | Out-Null
}

Push-Location $appDir
try {
    if ($Dev) {
        & $python server.py
    } else {
        & $python -m waitress --host=$Bind --port=$Port --threads=8 server:app
    }
} finally {
    Pop-Location
}
