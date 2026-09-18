# -*- coding: utf-8 -*-
"""SSH 引导：用密码登录一次，把本机公钥写入服务器 authorized_keys，
之后即可用 ~/.ssh/car_trace_ecs 免密操作。顺带摸底服务器环境。"""
import os
import sys

import paramiko

HOST = "47.101.32.59"
USER = "root"
PUBKEY = open(os.path.expanduser("~/.ssh/car_trace_ecs.pub")).read().strip()

RECON = r"""
echo "== os =="; head -2 /etc/os-release
echo "== arch/kernel =="; uname -m; uname -r
echo "== gpu =="; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | head -3
echo "== disk =="; df -h / /data 2>/dev/null | grep -v tmpfs
echo "== mem =="; free -g | head -2
echo "== cpu =="; nproc
echo "== python =="; ls /usr/bin/python3* 2>/dev/null; python3 --version 2>&1
echo "== dnf python candidates =="; dnf list available 'python3.1*' 2>/dev/null | grep -E "python3\.1[0-9]" | head -5
echo "== data dir =="; ls -la /data 2>/dev/null | head -8
echo "== ports =="; ss -tlnp 2>/dev/null | grep -E ":(8000|5000)\b" || echo "8000/5000 空闲"
"""


def main() -> None:
    password = os.environ["ZHIPU_SSH_PW"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=password, timeout=20)
    print("密码登录成功，正在写入公钥...")

    commands = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && "
        f'grep -qF "{PUBKEY}" ~/.ssh/authorized_keys || echo "{PUBKEY}" >> ~/.ssh/authorized_keys; '
        "chmod 600 ~/.ssh/authorized_keys && echo 公钥已写入"
    )
    _, stdout, stderr = client.exec_command(commands)
    print(stdout.read().decode().strip() or stderr.read().decode().strip())

    _, stdout, stderr = client.exec_command(RECON)
    print(stdout.read().decode())
    err = stderr.read().decode().strip()
    if err:
        print("[stderr]", err[:500])
    client.close()


if __name__ == "__main__":
    sys.exit(main())
