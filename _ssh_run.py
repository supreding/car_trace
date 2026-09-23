# -*- coding: utf-8 -*-
"""临时部署助手：SSH 执行远端命令（密码走环境变量 SSH_PW，不落盘）。"""
import sys, os, paramiko
host, user = "47.101.32.59", "root"
pw = os.environ["SSH_PW"]
cmd = sys.argv[1]
timeout = float(os.environ.get("SSH_TIMEOUT", "300"))
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(host, 22, user, pw, timeout=30)
stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
out = stdout.read().decode("utf-8", "replace")
err = stderr.read().decode("utf-8", "replace")
code = stdout.channel.recv_exit_status()
if out: print(out, end="")
if err: print("[stderr]", err, end="", file=sys.stderr)
sys.exit(code)
