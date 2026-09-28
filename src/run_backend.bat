@echo off
setlocal
cd /d "%~dp0backend"
if errorlevel 1 (
  echo ERROR: Could not change to backend folder: "%~dp0backend"
  pause
  exit /b 1
)
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo ERROR: Virtual environment Python not found:
  echo "%PY%"
  pause
  exit /b 1
)
echo Starting DeepTrace FastAPI backend...
echo Working directory: %CD%
"%PY%" -m uvicorn main:app --host 127.0.0.1 --port 8000
set "RC=%ERRORLEVEL%"
echo.
echo Backend stopped with exit code %RC%.
pause
exit /b %RC%
