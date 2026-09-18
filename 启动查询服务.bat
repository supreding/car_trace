@echo off
rem One-click launcher for the vehicle trace query server (start_server.ps1).
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_server.ps1" %*
pause
