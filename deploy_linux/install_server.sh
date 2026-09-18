#!/usr/bin/env bash
# 服务器一键安装（Alibaba Cloud Linux / 其他 RHEL 系均可参考）
# 前提：本脚本位于 /opt/car-trace/deploy_linux/，部署包已解压到 /opt/car-trace，
#       机器有 NVIDIA GPU 且驱动已装好（nvidia-smi 可用）。
# 用法：sudo bash install_server.sh
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
ALIYUN_PYPI="https://mirrors.aliyun.com/pypi/simple/"
TUNA_PYPI="https://pypi.tuna.tsinghua.edu.cn/simple/"   # 大文件（torch/nvidia 轮子）实测比阿里云 CDN 快 80 倍
PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"

echo "==== 0. 前置检查 ===="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "错误：nvidia-smi 不可用，请先安装 NVIDIA 驱动（阿里云 GPU 实例可在控制台/帮助文档安装）。"
    exit 1
fi
nvidia-smi | head -4 || true
if [ ! -f app/server.py ]; then
    echo "错误：未找到 app/server.py，请先把部署包解压到 $project_root"
    exit 1
fi

echo "==== 1. 安装系统 Python（优先 3.12，其次 3.11）===="
if command -v python3.12 >/dev/null 2>&1; then
    PY=python3.12
elif command -v python3.11 >/dev/null 2>&1; then
    PY=python3.11
else
    dnf install -y python3.12 python3.12-pip 2>/dev/null || dnf install -y python3.11 python3.11-pip
    if command -v python3.12 >/dev/null 2>&1; then PY=python3.12; else PY=python3.11; fi
fi
echo "使用 Python: $PY ($($PY --version 2>&1))"
dnf install -y git unzip 2>/dev/null || true

echo "==== 2. 查询服务轻量环境 (.venv-server) ===="
if [ ! -x .venv-server/bin/python ]; then "$PY" -m venv .venv-server; fi
.venv-server/bin/pip install --disable-pip-version-check -r requirements-server.txt -i "$ALIYUN_PYPI"

echo "==== 3. 流水线 GPU 环境 (.venv-reid)，下载量约 4-5 GB ===="
if [ ! -x .venv-reid/bin/python ]; then "$PY" -m venv .venv-reid; fi
# 3.1 torch cu126（nvidia-* 大轮子走清华 TUNA 镜像）
if ! .venv-reid/bin/pip install --disable-pip-version-check \
        "torch==2.12.1+cu126" "torchvision==0.27.1+cu126" \
        --index-url "$TUNA_PYPI"; then
    echo "TUNA 未命中，回落 PyTorch 官方源（可能较慢）..."
    .venv-reid/bin/pip install --disable-pip-version-check \
        "torch==2.12.1+cu126" "torchvision==0.27.1+cu126" --index-url "$PYTORCH_INDEX"
fi
# 3.2 paddlepaddle-gpu cu126（走 Paddle 官方 index）。
#     --no-deps：paddle 钉的 nvidia-cudnn==9.5 与 torch 钉的 ==9.10 互斥，
#     paddle 的 Linux 轮子自带 CUDA 库（paddle/libs），其余依赖由 3.3 清单补齐，
#     不能让它把 torch 需要的 cudnn 降级。
.venv-reid/bin/pip install --disable-pip-version-check \
    --no-deps "paddlepaddle-gpu==3.3.1" -i "$PADDLE_INDEX"
# 3.3 其余依赖（清单即完整冻结，用 --no-deps 避免 paddle 的旧 nvidia 钉版反向降级）
.venv-reid/bin/pip install --disable-pip-version-check \
    --no-deps -r requirements-reid.txt -i "$TUNA_PYPI"

echo "==== 4. 分别验证 GPU（torch 与 paddle 必须分开 import，不能同进程）===="
.venv-reid/bin/python -c "import torch; print('torch cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
.venv-reid/bin/python -c "import paddle; print('paddle GPU:', paddle.device.is_compiled_with_cuda())"

echo "==== 5. 查询服务 systemd 单元（开机自启 + 崩溃自动拉起）===="
TOKEN_FILE="$project_root/deploy_linux/open_api_token.txt"
if [ ! -f "$TOKEN_FILE" ]; then
    (command -v openssl >/dev/null && openssl rand -hex 24 || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n') > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
fi
TOKEN="$(cat "$TOKEN_FILE")"
sed "s/__REPLACE_ME__/$TOKEN/" deploy_linux/car-trace.service > /etc/systemd/system/car-trace.service
systemctl daemon-reload
systemctl enable --now car-trace
sleep 2
systemctl --no-pager status car-trace | head -10
echo "健康检查: $(curl -s http://127.0.0.1:8000/api/health || echo '启动中，稍后重试')"

echo
echo "==== 安装完成 ===="
echo "查询服务:  http://<服务器IP>:8000/   （开放接口 token 见 deploy_linux/open_api_token.txt）"
echo "流水线:    bash run_reid_pipeline.sh --db app/granary.db   （多园区加 --db/--output-dir）"
echo "注意:      安全组需放行 TCP 8000；照片放 data/<园区>/<月份照片根>/<日期>/<订单>/ 后跑流水线。"
