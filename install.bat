@echo off
REM ============================================================================
REM  Semester Project installer (Windows) - thin launcher.
REM  Finds a REAL Python even when:
REM    - 'python' is the fake 0-byte Microsoft Store stub
REM    - Python was installed without "Add to PATH"
REM  then delegates the actual work to install.py.
REM
REM  Double-click me, or:   install.bat            full install
REM                         install.bat --edge     inference-only install
REM ============================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo Looking for a working Python installation...
set "PYEXE="

REM --- 1) the py launcher (best case) ---
py -3 -c "import sys" >nul 2>nul && set "PYEXE=py -3"
if defined PYEXE goto :run

REM --- 2) 'python' on PATH (executing code filters out the Store stub,
REM        which can't run anything and returns an error) ---
python -c "import sys" >nul 2>nul && set "PYEXE=python"
if defined PYEXE goto :run

REM --- 3) standard install folders, even if not on PATH ---
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
for /d %%D in ("C:\Python3*") do (
  if not defined PYEXE if exist "%%D\python.exe" (
    "%%D\python.exe" -c "import sys" >nul 2>nul && set "PYEXE=%%D\python.exe"
  )
)
if defined PYEXE goto :run

REM --- nothing usable found ---
echo.
echo ============================================================================
echo  [ERROR] No working Python found on this machine.
echo.
echo  Install it EITHER way:
echo    Option A (recommended) - in this terminal run:
echo        winget install -e --id Python.Python.3.12
echo    Option B - download from  https://www.python.org/downloads/
echo        IMPORTANT: tick "Add python.exe to PATH" on the first screen.
echo.
echo  Then CLOSE this window, open a NEW one, and run install.bat again.
echo ============================================================================
pause
exit /b 1

:run
echo Using: !PYEXE!
!PYEXE! install.py %*
exit /b %errorlevel%
