@echo off
REM ============================================================================
REM  Double-click to watch live vehicle detection on the demo traffic video.
REM  A window opens showing boxes + counts + congestion. Press q to quit.
REM
REM  Optional: pass a model variant -  watch_detection.bat pruned
REM  Variants: simple (default) | pruned | quantized | pruned_quantized | nas
REM ============================================================================
setlocal
cd /d "%~dp0"

set "VARIANT=%~1"
if "%VARIANT%"=="" set "VARIANT=simple"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] .venv not found. Run install.bat first.
  pause
  exit /b 1
)

echo Launching detection (variant=%VARIANT%) on the demo video...
echo A window titled "vehicle detection (q to quit)" will open. Press q to stop.
echo.
"%PY%" detect.py --variant %VARIANT%
if errorlevel 1 pause
endlocal
