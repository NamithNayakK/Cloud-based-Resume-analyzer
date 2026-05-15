@echo off
REM Start ML Service without venv complications
cd /d D:\Resume\cloud-resume-analyzer\backend\mlService
python -m pip install --quiet -q Flask sentence-transformers numpy 2>nul
python app.py
pause
