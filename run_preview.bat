@echo off
rem Serve the tabletop solar panel AR site (preview folder) on http://localhost:%PORT% and open it.
rem Needs only Python 3 on PATH (no packages). Close this window to stop.
rem Optional: set PORT=9000 before running; set NO_BROWSER=1 to skip opening it.
setlocal
if "%PORT%"=="" set PORT=8765
cd /d "%~dp0preview" || (echo preview folder not found next to this file & pause & exit /b 1)

set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY (
  echo Python 3 was not found. Install it from https://www.python.org/downloads/ and tick "Add to PATH".
  pause
  exit /b 1
)

echo Serving %CD% at http://localhost:%PORT%   (Ctrl+C or close this window to stop)
if not "%NO_BROWSER%"=="1" start "" "http://localhost:%PORT%/"
%PY% -m http.server %PORT% --bind 127.0.0.1
