@echo off
setlocal enabledelayedexpansion

:: Skyscope Sentinel Intelligence AI Platform
:: Startup Script
:: Created: July 16, 2025

title Skyscope Sentinel Intelligence AI Platform - Startup

:: Check for administrative privileges
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if %errorlevel% neq 0 (
    echo Administrative privileges required. Attempting to elevate...
    goto UACPrompt
) else (
    goto GotAdmin
)

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    exit /B

:GotAdmin
    if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
    pushd "%CD%"
    CD /D "%~dp0"

:: Set environment variables
set "INSTALL_DIR=%USERPROFILE%\Skyscope Sentinel Intelligence"
set "VENV_DIR=%INSTALL_DIR%\venv"
set "CONFIG_DIR=%INSTALL_DIR%\config"
set "LOG_DIR=%INSTALL_DIR%\logs"
set "STARTUP_MODE=normal"
set "ERROR_LOG=%LOG_DIR%\startup_errors.log"

:: Create log directory if it doesn't exist
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: Process command line arguments
if "%1"=="debug" set "STARTUP_MODE=debug"
if "%1"=="minimal" set "STARTUP_MODE=minimal"
if "%1"=="help" goto ShowHelp

:: Display banner
echo.
echo ===================================================
echo    SKYSCOPE SENTINEL INTELLIGENCE AI PLATFORM
echo ===================================================
echo  Starting system in %STARTUP_MODE% mode...
echo ===================================================
echo.

:: Check if installation directory exists
if not exist "%INSTALL_DIR%" (
    echo ERROR: Installation directory not found at:
    echo %INSTALL_DIR%
    echo.
    echo Please run the installer first or specify the correct path.
    echo.
    goto Error
)

:: Check for Python installation
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH.
    echo Please install Python 3.8 or higher and add it to your PATH.
    echo.
    goto Error
)

:: Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
echo Found Python version: %PYTHON_VERSION%

:: Check if virtual environment exists
if not exist "%VENV_DIR%" (
    echo WARNING: Virtual environment not found.
    echo Creating new virtual environment...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        goto Error
    )
    echo Virtual environment created successfully.
)

:: Activate virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    goto Error
)

:: Check for required packages
echo Checking for required packages...
python -c "import sys; sys.exit(0 if all(m in sys.modules or __import__(m) for m in ['numpy', 'pandas', 'torch', 'openai']) else 1)" 2>nul
if %errorlevel% neq 0 (
    echo Some required packages are missing. Installing dependencies...
    pip install -r "%INSTALL_DIR%\requirements.txt"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install required packages.
        goto Error
    )
)

:: Start the application based on startup mode
if "%STARTUP_MODE%"=="debug" (
    echo Starting in DEBUG mode with verbose logging...
    start "Skyscope AI - Debug Mode" python "%INSTALL_DIR%\skyscope_windows_app.py" --debug --log-level=DEBUG
) else if "%STARTUP_MODE%"=="minimal" (
    echo Starting in MINIMAL mode with reduced resource usage...
    start "Skyscope AI - Minimal Mode" python "%INSTALL_DIR%\skyscope_windows_app.py" --minimal --agents=100
) else (
    echo Starting in NORMAL mode...
    start "Skyscope AI" python "%INSTALL_DIR%\skyscope_windows_app.py"
)

if %errorlevel% neq 0 (
    echo ERROR: Failed to start the application.
    goto Error
)

echo.
echo Skyscope Sentinel Intelligence AI Platform started successfully!
echo.
echo You can access the system through:
echo - System tray icon
echo - Web interface at http://localhost:8000
echo - Desktop application
echo.
echo Press any key to exit this startup window...
pause > nul
exit /B 0

:Error
echo.
echo An error occurred during startup.
echo Error details have been saved to: %ERROR_LOG%
echo.
echo Please contact support or check the documentation.
echo.
echo Press any key to exit...
pause > nul
exit /B 1

:ShowHelp
echo.
echo SKYSCOPE SENTINEL INTELLIGENCE AI PLATFORM - HELP
echo ===================================================
echo Usage: start_skyscope.bat [mode]
echo.
echo Available modes:
echo   normal  - Start with default settings (default)
echo   debug   - Start with verbose logging and debugging enabled
echo   minimal - Start with minimal resource usage
echo   help    - Show this help message
echo.
echo Examples:
echo   start_skyscope.bat
echo   start_skyscope.bat debug
echo   start_skyscope.bat minimal
echo.
echo Press any key to exit...
pause > nul
exit /B 0
