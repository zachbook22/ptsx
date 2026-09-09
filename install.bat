@echo off
REM Install ptsx: Python 3.10+, ffmpeg, venv, Whisper base.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "PTSX_PYTHON="

call :find_python
if not defined PTSX_PYTHON (
  echo Python 3.10+ not found. Installing Python 3.12 with winget...
  call :ensure_winget
  if errorlevel 1 exit /b 1
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo winget could not install Python. Install it from https://www.python.org/downloads/ then re-run.
    exit /b 1
  )
  call :refresh_path
  call :find_python
)
if not defined PTSX_PYTHON (
  echo Python 3.10+ is still not on PATH. Close this window, open a new Command Prompt, and run install.bat again.
  exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg not found. Installing with winget...
  call :ensure_winget
  if errorlevel 1 exit /b 1
  winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo winget could not install ffmpeg. Install it from https://ffmpeg.org/download.html then re-run.
    exit /b 1
  )
  call :refresh_path
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg is still not on PATH. Close this window, open a new Command Prompt, and run install.bat again.
  exit /b 1
)

echo Creating virtualenv in .venv...
"%PTSX_PYTHON%" -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -U pip
echo Installing ptsx + Whisper (first time can take several minutes)...
python -m pip install -e .
echo Downloading Whisper base model (one-time, ~150 MB)...
python -c "import whisper; whisper.load_model('base', device='cpu')"

echo.
echo Installed. In a new Command Prompt:
echo   cd /d "%cd%"
echo   .venv\Scripts\activate
echo.
echo Then:
echo   ptsx gate --user C:\path\USER.wav --assistant C:\path\ASSISTANT.wav --outdir .\out --transcribe
echo   ptsx clean-cut --user C:\path\USER.wav --assistant C:\path\ASSISTANT.wav --outdir .\out --transcribe
echo.
echo In Cursor, trigger the ptsx-standard-cut or ptsx-clean-cut project skill.
echo Later updates: git pull, then re-run install.bat only if Python dependencies changed.
exit /b 0

:ensure_winget
where winget >nul 2>nul
if errorlevel 1 (
  echo winget is required to install Python and ffmpeg automatically.
  echo Install App Installer from the Microsoft Store, then re-run install.bat.
  exit /b 1
)
exit /b 0

:refresh_path
set "PATH=%LocalAppData%\Programs\Python\Python312;%LocalAppData%\Programs\Python\Python312\Scripts;%LocalAppData%\Programs\Python\Python313;%LocalAppData%\Programs\Python\Python313\Scripts;%ProgramFiles%\Python312;%ProgramFiles%\Python312\Scripts;%LocalAppData%\Microsoft\WinGet\Links;%PATH%"
exit /b 0

:find_python
set "PTSX_PYTHON="
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" 2>nul
if not errorlevel 1 (
  for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PTSX_PYTHON=%%I"
  if defined PTSX_PYTHON exit /b 0
)
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" 2>nul
if not errorlevel 1 (
  for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PTSX_PYTHON=%%I"
  if defined PTSX_PYTHON exit /b 0
)
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
  set "PTSX_PYTHON=%LocalAppData%\Programs\Python\Python312\python.exe"
  exit /b 0
)
if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
  set "PTSX_PYTHON=%LocalAppData%\Programs\Python\Python313\python.exe"
  exit /b 0
)
if exist "%ProgramFiles%\Python312\python.exe" (
  set "PTSX_PYTHON=%ProgramFiles%\Python312\python.exe"
  exit /b 0
)
exit /b 1
