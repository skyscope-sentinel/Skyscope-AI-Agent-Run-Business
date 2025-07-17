@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION

:: ==============================================================================
:: Skyscope Sentinel Intelligence - Windows Application Packager
:: ==============================================================================
:: This script automates the entire process of setting up the environment,
:: cloning the repository, and packaging the application into a distributable
:: Windows installer (.exe).
::
:: It performs the following steps:
:: 1. Checks for Administrator privileges and re-launches if necessary.
:: 2. Checks and installs Chocolatey package manager if not found.
:: 3. Checks and installs necessary system dependencies like Python, Git, and NSIS.
:: 4. Clones the application repository from GitHub or pulls the latest changes.
:: 5. Sets up a dedicated Python virtual environment.
:: 6. Installs all required Python packages from requirements.txt.
:: 7. Bundles the Python application into a standalone .exe using PyInstaller.
:: 8. Packages the bundled app into a professional installer using NSIS.
:: ==============================================================================

TITLE Skyscope Sentinel Intelligence - Windows Builder

:: --- Configuration Variables ---
SET "REPO_URL=https://github.com/skyscope-sentinel/Skyscope-Quantum-AI-Agentic-Swarm-Autonomous-System-WebUI.git"
SET "REPO_DIR=Skyscope-Quantum-AI-Agentic-Swarm-Autonomous-System-WebUI"
SET "VENV_DIR=venv"
SET "APP_NAME=Skyscope Sentinel"
SET "APP_VERSION=1.0.0"
SET "PYTHON_VERSION=3.11.5"
SET "MAIN_SCRIPT=app.py"
SET "ICON_FILE=logo.ico"
SET "COMPANY_NAME=Skyscope Sentinel Intelligence"

:: --- Helper Function for Logging ---
:log_step
ECHO.
ECHO ==================================================================
ECHO ^>^>^> %~1
ECHO ==================================================================
GOTO :EOF

:log_success
ECHO [SUCCESS] %~1
GOTO :EOF

:log_warn
ECHO [WARNING] %~1
GOTO :EOF

:log_error
ECHO [ERROR] %~1
ECHO [ERROR] Build process aborted.
PAUSE
EXIT /B 1
GOTO :EOF

:: 1. Check for Administrator Privileges
CALL :log_step "Checking for Administrator privileges..."
NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO This script requires Administrator privileges to install dependencies.
    ECHO Attempting to re-launch with elevated rights...
    PowerShell -Command "Start-Process '%~f0' -Verb RunAs"
    EXIT
)
CALL :log_success "Running with Administrator privileges."

:: 2. Check and Install Chocolatey
CALL :log_step "Checking for Chocolatey package manager..."
where choco >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    CALL :log_warn "Chocolatey not found. Installing..."
    PowerShell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    IF %ERRORLEVEL% NEQ 0 (
        CALL :log_error "Failed to install Chocolatey. Please install it manually from https://chocolatey.org and re-run this script."
    )
    CALL :log_success "Chocolatey installed successfully."
) ELSE (
    CALL :log_success "Chocolatey is already installed."
)

:: 3. Check and Install Dependencies (Python, Git, NSIS)
CALL :log_step "Checking for dependencies (Python, Git, NSIS)..."
SET "PACKAGES_TO_INSTALL="

where python >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    SET "PACKAGES_TO_INSTALL=!PACKAGES_TO_INSTALL! python --version=%PYTHON_VERSION%"
) ELSE (
    CALL :log_success "Python is already installed."
)

where git >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    SET "PACKAGES_TO_INSTALL=!PACKAGES_TO_INSTALL! git"
) ELSE (
    CALL :log_success "Git is already installed."
)

where makensis >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    SET "PACKAGES_TO_INSTALL=!PACKAGES_TO_INSTALL! nsis"
) ELSE (
    CALL :log_success "NSIS is already installed."
)

IF DEFINED PACKAGES_TO_INSTALL (
    ECHO Installing missing dependencies: !PACKAGES_TO_INSTALL!
    choco install !PACKAGES_TO_INSTALL! -y
    IF %ERRORLEVEL% NEQ 0 (
        CALL :log_error "Failed to install one or more dependencies using Chocolatey."
    )
    CALL :log_success "All dependencies are now installed."
)

:: 4. Clone or Update Repository
CALL :log_step "Acquiring project source code..."
IF EXIST "%REPO_DIR%" (
    CALL :log_warn "Repository directory exists. Pulling latest changes..."
    cd "%REPO_DIR%"
    git pull
    cd ..
) ELSE (
    git clone "%REPO_URL%"
    IF %ERRORLEVEL% NEQ 0 (
        CALL :log_error "Failed to clone the repository."
    )
)
cd "%REPO_DIR%"
CALL :log_success "Repository is up to date."

:: 5. Set up Python Virtual Environment
CALL :log_step "Setting up Python virtual environment..."
IF NOT EXIST "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
    IF %ERRORLEVEL% NEQ 0 (
        CALL :log_error "Failed to create Python virtual environment."
    )
    CALL :log_success "Virtual environment created."
) ELSE (
    CALL :log_success "Virtual environment already exists."
)

:: 6. Activate and Install Python Dependencies
CALL :log_step "Installing Python dependencies..."
CALL "%VENV_DIR%\Scripts\activate.bat"
pip install --upgrade pip
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    CALL :log_error "Failed to install dependencies from requirements.txt."
)
pip install pyinstaller pywin32-ctypes
IF %ERRORLEVEL% NEQ 0 (
    CALL :log_error "Failed to install PyInstaller."
)
CALL :log_success "All Python dependencies installed."

:: 7. Build the Standalone Application with PyInstaller
CALL :log_step "Building standalone executable with PyInstaller..."
IF NOT EXIST "%ICON_FILE%" (
    CALL :log_warn "logo.ico not found. A default icon will be used."
    SET "ICON_ARG="
) ELSE (
    SET "ICON_ARG=--icon=%ICON_FILE%"
)

pyinstaller --name "%APP_NAME%" --noconfirm --clean --windowed %ICON_ARG% ^
    --add-data "knowledge_base.md;." ^
    --add-data "config.py;." ^
    --add-data "models.py;." ^
    --add-data "utils.py;." ^
    --hidden-import="streamlit.web.cli" ^
    --hidden-import="sklearn.utils._cython_blas" ^
    --hidden-import="sklearn.neighbors._typedefs" ^
    --hidden-import="sklearn.neighbors._quad_tree" ^
    --hidden-import="sklearn.tree._utils" ^
    --hidden-import="pkg_resources.py2_warn" ^
    "%MAIN_SCRIPT%"

IF %ERRORLEVEL% NEQ 0 (
    CALL :log_error "PyInstaller failed to build the executable."
)
CALL :log_success "Executable bundle created in 'dist' directory."

:: 8. Create the NSIS Installer
CALL :log_step "Creating Windows installer with NSIS..."
SET "NSI_SCRIPT=installer.nsi"

(
    ECHO;
    ECHO !define APP_NAME "%APP_NAME%"
    ECHO !define APP_VERSION "%APP_VERSION%"
    ECHO !define COMPANY_NAME "%COMPANY_NAME%"
    ECHO !define EXE_NAME "%APP_NAME%.exe"
    ECHO !define INSTALLER_NAME "%APP_NAME%_Installer_v%APP_VERSION%.exe"
    ECHO;
    ECHO OutFile "${INSTALLER_NAME}"
    ECHO InstallDir "$PROGRAMFILES64\${APP_NAME}"
    ECHO RequestExecutionLevel admin
    ECHO;
    ECHO VIProductVersion "%APP_VERSION%.0"
    ECHO VIAddVersionKey "ProductName" "${APP_NAME}"
    ECHO VIAddVersionKey "CompanyName" "${COMPANY_NAME}"
    ECHO VIAddVersionKey "FileDescription" "${APP_NAME} - AI Agentic Swarm"
    ECHO VIAddVersionKey "FileVersion" "%APP_VERSION%"
    ECHO VIAddVersionKey "LegalCopyright" "Copyright 2025 ${COMPANY_NAME}"
    ECHO;
    ECHO Page directory
    ECHO Page instfiles
    ECHO;
    ECHO Section "Install"
    ECHO   SetOutPath "$INSTDIR"
    ECHO   File /r "dist\%APP_NAME%\*.*"
    ECHO;
    ECHO   ; Create Shortcuts
    ECHO   CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${EXE_NAME}"
    ECHO   CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    ECHO   CreateShortCut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${EXE_NAME}"
    ECHO   CreateShortCut "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
    ECHO;
    ECHO   ; Write Uninstaller
    ECHO   WriteUninstaller "$INSTDIR\Uninstall.exe"
    ECHO SectionEnd
    ECHO;
    ECHO Section "Uninstall"
    ECHO   Delete "$INSTDIR\*.*"
    ECHO   RMDir /r "$INSTDIR"
    ECHO;
    ECHO   ; Remove Shortcuts
    ECHO   Delete "$DESKTOP\${APP_NAME}.lnk"
    ECHO   RMDir /r "$SMPROGRAMS\${APP_NAME}"
    ECHO SectionEnd
) > "%NSI_SCRIPT%"

makensis "%NSI_SCRIPT%"
IF %ERRORLEVEL% NEQ 0 (
    CALL :log_error "NSIS failed to create the installer."
)
CALL :log_success "Installer '%APP_NAME%_Installer_v%APP_VERSION%.exe' created successfully."

:: 9. Cleanup
CALL :log_step "Cleaning up temporary build files..."
DEL "%NSI_SCRIPT%"
RMDIR /S /Q "build"
DEL "%APP_NAME%.spec"

ECHO.
CALL :log_success "Build process complete!"
ECHO The installer can be found in the current directory.
ECHO.

ENDLOCAL
EXIT /B 0
