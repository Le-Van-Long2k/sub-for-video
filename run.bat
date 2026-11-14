
@echo off
chcp 65001 > nul
title Video to SRT

if not exist .venv (
    py -3.12 -m venv .venv
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo ERROR: FFmpeg chua duoc cai dat!
    pause
    exit /b 1
)

pip install -r requirements.txt

python app.py

pause
