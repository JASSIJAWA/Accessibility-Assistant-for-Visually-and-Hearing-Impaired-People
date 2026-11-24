@echo off
title AVHA Live Launcher
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-avha-live.ps1"
echo.
pause
