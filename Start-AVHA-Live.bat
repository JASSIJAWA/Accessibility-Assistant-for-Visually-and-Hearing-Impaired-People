@echo off
title AVHA Live Launcher
set "PROJECT_DIR=%~dp0"
if not exist "%PROJECT_DIR%start-avha-live.ps1" (
    set "PROJECT_DIR=c:\Users\Asus\Desktop\Major project 7th sem\"
)
cd /d "%PROJECT_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%start-avha-live.ps1" %*
echo.
pause
