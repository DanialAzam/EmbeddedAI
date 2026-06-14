@echo off
REM ============================================================================
REM  Launch the control-panel GUI (gui.py). Double-click me.
REM  Same real-Python hunt as install.bat (py launcher -> PATH -> install dirs).
REM ============================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "PYEXE="
py -3 -c "import sys" >nul 2>nul && set "PYEXE=py -3"
if defined PYEXE goto :run

python -c "import sys" >nul 2>nul && set "PYEXE=python"
if defined PYEXE goto :run

for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
  if not defined PYEXE if exist "%%D\python.exe" (
    "%%D\python.exe" -c "import sys" >nul 2>nul && set "PYEXE=%%D\python.exe"
  )
)
for /d %%D in ("%ProgramFiles%\Python3*") do (
  if not defined PYEXE if exist "%%D\python.exe" (
    "%%D\python.exe" -c "import sys" >nul 2>nul && set "PYEXE=%%D\python.exe"
  )
)
if defined PYEXE goto :run

echo [ERROR] No working Python found. Run install.bat first - it explains the fix.
pause
exit /b 1

:run
!PYEXE! gui.py %*
if errorlevel 1 pause
exit /b %errorlevel%
