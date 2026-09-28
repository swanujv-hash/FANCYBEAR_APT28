@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 (
  echo ERROR: Could not access the DeepTrace folder.
  pause
  exit /b 1
)
title DeepTrace - Cyber Forensic Assistant
set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "VENV_PY=%VENV%\Scripts\python.exe"
set "BACKEND=%ROOT%backend"
set "LOG=%ROOT%deeptrace_backend.log"

echo.
echo ============================================
echo        DEEPTRACE STARTUP
echo ============================================
echo.

where py >nul 2>&1
if not errorlevel 1 (
  set "PYTHON=py"
) else (
  where python >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON=python"
  ) else (
    echo Python was not found.
    echo Install Python 3.11+ and run this file again.
    pause
    exit /b 1
  )
)

if not exist "%BACKEND%\main.py" (
  echo ERROR: Backend not found at:
  echo "%BACKEND%"
  echo.
  echo Make sure the ZIP was extracted completely.
  pause
  exit /b 1
)

if not exist "%VENV_PY%" (
  echo [1/4] Creating virtual environment...
  %PYTHON% -m venv "%VENV%"
  if errorlevel 1 goto :fail
) else (
  echo [1/4] Virtual environment found.
)

echo [2/4] Installing/checking backend dependencies...
"%VENV_PY%" -m pip install --disable-pip-version-check -r "%BACKEND%\requirements.txt"
if errorlevel 1 goto :fail

echo [3/4] Checking backend files...
if not exist "%BACKEND%\models\case.py" (
  echo ERROR: backend\models\case.py is missing.
  goto :fail
)
"%VENV_PY%" -c "import fastapi, uvicorn, cv2, numpy; print('Dependencies OK')"
if errorlevel 1 goto :fail

if exist "%LOG%" del /q "%LOG%" >nul 2>&1

echo [4/4] Starting FastAPI backend...
echo Backend log: "%LOG%"
start "DeepTrace Backend" "%ComSpec%" /k call "%ROOT%run_backend.bat" >nul

 echo.
echo Waiting for backend (up to 120 seconds)...
set /a tries=0
:wait
set /a tries+=1
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/health' -TimeoutSec 2; if($r.StatusCode -eq 200){exit 0}else{exit 1} } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto :ready
if !tries! GEQ 120 goto :fail_wait
timeout /t 1 /nobreak >nul
goto :wait

:ready
echo.
echo ============================================
echo   DeepTrace is READY
echo   http://127.0.0.1:8000/
echo ============================================
echo.
start "" "http://127.0.0.1:8000/"
echo Keep the DeepTrace Backend window open while using the website.
pause
exit /b 0

:fail_wait
echo.
echo ============================================
echo   BACKEND DID NOT START
 echo ============================================
echo.
echo Check the DeepTrace Backend window for the Python error.
if exist "%LOG%" (
  echo.
  echo Log file: "%LOG%"
  type "%LOG%"
)
pause
exit /b 1

:fail
echo.
echo ============================================
echo   DEEPTRACE COULD NOT START
 echo ============================================
echo.
echo Read the error above.
pause
exit /b 1
