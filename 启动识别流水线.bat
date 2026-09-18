@echo off
rem One-click launcher for the vehicle recognition pipeline (start_pipeline.ps1).
rem Requires NVIDIA GPU + Python 3.12; see requirements-reid.txt for deps.
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_pipeline.ps1" %*
pause
