<#
.SYNOPSIS
    Master Windows Installer for Skyscope Sentinel Intelligence AI System
.DESCRIPTION
    This PowerShell script automates the entire installation process for the Skyscope Sentinel Intelligence AI system.
    It provides a GUI installer with a professional black theme, handles all dependencies, sets up Ollama integration,
    configures the system for 10,000 agents, and can create a standalone Windows installer executable.
.NOTES
    File Name      : install_windows_master.ps1
    Author         : Skyscope Technologies
    Prerequisite   : PowerShell 5.1 or later
    Version        : 1.0.0
    Date           : July 16, 2025
.EXAMPLE
    .\install_windows_master.ps1
#>

#Requires -Version 5.1
#Requires -RunAsAdministrator

# Set strict mode and error action preference
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue" # Speeds up web downloads

# Script constants
$SCRIPT_VERSION = "1.0.0"
$COMPANY_NAME = "Skyscope Technologies"
$PRODUCT_NAME = "Skyscope Sentinel Intelligence AI"
$PRODUCT_VERSION = "1.0.0"
$MIN_PYTHON_VERSION = "3.9.0"
$RECOMMENDED_RAM_GB = 8
$RECOMMENDED_DISK_SPACE_GB = 10

# Paths
$INSTALL_DIR = Join-Path $env:LOCALAPPDATA "SkyscopeSentinel"
$LOGS_DIR = Join-Path $INSTALL_DIR "logs"
$CONFIG_DIR = Join-Path $INSTALL_DIR "config"
$DATA_DIR = Join-Path $INSTALL_DIR "data"
$MODELS_DIR = Join-Path $INSTALL_DIR "models"
$TEMP_DIR = Join-Path $env:TEMP "SkyscopeSentinelTemp"
$LOG_FILE = Join-Path $LOGS_DIR "install.log"
$INSTALLER_OUTPUT = Join-Path $PSScriptRoot "SkyscopeSentinelSetup.exe"

# URLs and resources
$PYTHON_URL = "https://www.python.org/ftp/python/3.11.4/python-3.11.4-amd64.exe"
$PYTHON_INSTALLER = Join-Path $TEMP_DIR "python_installer.exe"
$GIT_URL = "https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe"
$GIT_INSTALLER = Join-Path $TEMP_DIR "git_installer.exe"
$OLLAMA_URL = "https://ollama.com/download/windows"
$OLLAMA_INSTALLER = Join-Path $TEMP_DIR "ollama_installer.exe"
$INNO_SETUP_URL = "https://jrsoftware.org/download.php/is.exe"
$INNO_SETUP_INSTALLER = Join-Path $TEMP_DIR "inno_setup.exe"

# Load required assemblies for GUI
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName PresentationFramework
Add-Type -AssemblyName PresentationCore
Add-Type -AssemblyName WindowsBase

# Import WPF assembly for modern UI
[System.Reflection.Assembly]::LoadWithPartialName('presentationframework') | Out-Null

# Define XAML for the modern UI
$xamlContent = @"
<Window 
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    Title="Skyscope Sentinel Intelligence AI - Installer"
    Height="600" Width="800" WindowStartupLocation="CenterScreen"
    Background="#121212" Foreground="#FFFFFF">
    <Window.Resources>
        <Style TargetType="Button">
            <Setter Property="Background" Value="#1E1E1E"/>
            <Setter Property="Foreground" Value="#FFFFFF"/>
            <Setter Property="BorderBrush" Value="#333333"/>
            <Setter Property="Padding" Value="15,5"/>
            <Setter Property="Margin" Value="5"/>
            <Setter Property="Template">
                <Setter.Value>
                    <ControlTemplate TargetType="Button">
                        <Border Background="{TemplateBinding Background}"
                                BorderBrush="{TemplateBinding BorderBrush}"
                                BorderThickness="1"
                                CornerRadius="3">
                            <ContentPresenter HorizontalAlignment="Center" VerticalAlignment="Center"/>
                        </Border>
                        <ControlTemplate.Triggers>
                            <Trigger Property="IsMouseOver" Value="True">
                                <Setter Property="Background" Value="#2D2D2D"/>
                            </Trigger>
                            <Trigger Property="IsPressed" Value="True">
                                <Setter Property="Background" Value="#007ACC"/>
                            </Trigger>
                        </ControlTemplate.Triggers>
                    </ControlTemplate>
                </Setter.Value>
            </Setter>
        </Style>
        <Style TargetType="CheckBox">
            <Setter Property="Foreground" Value="#FFFFFF"/>
            <Setter Property="Margin" Value="5"/>
        </Style>
        <Style TargetType="TextBlock">
            <Setter Property="Foreground" Value="#FFFFFF"/>
            <Setter Property="Margin" Value="5"/>
        </Style>
        <Style TargetType="ProgressBar">
            <Setter Property="Background" Value="#1E1E1E"/>
            <Setter Property="Foreground" Value="#007ACC"/>
            <Setter Property="BorderBrush" Value="#333333"/>
            <Setter Property="Height" Value="15"/>
            <Setter Property="Margin" Value="5"/>
        </Style>
    </Window.Resources>
    <Grid Margin="20">
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="*"/>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>

        <!-- Header -->
        <StackPanel Grid.Row="0">
            <TextBlock FontSize="24" FontWeight="Bold" Margin="0,0,0,10">Skyscope Sentinel Intelligence AI</TextBlock>
            <TextBlock FontSize="16" Margin="0,0,0,20">Installation Wizard</TextBlock>
        </StackPanel>

        <!-- System Requirements -->
        <GroupBox Grid.Row="1" Header="System Requirements" Foreground="#FFFFFF" BorderBrush="#333333" Margin="0,10">
            <StackPanel Margin="10">
                <TextBlock Name="txtPythonStatus" Text="Python: Checking..." Margin="0,5"/>
                <TextBlock Name="txtRAMStatus" Text="Memory: Checking..." Margin="0,5"/>
                <TextBlock Name="txtDiskStatus" Text="Disk Space: Checking..." Margin="0,5"/>
                <TextBlock Name="txtGitStatus" Text="Git: Checking..." Margin="0,5"/>
            </StackPanel>
        </GroupBox>

        <!-- Installation Options -->
        <ScrollViewer Grid.Row="2" VerticalScrollBarVisibility="Auto">
            <StackPanel>
                <GroupBox Header="Installation Options" Foreground="#FFFFFF" BorderBrush="#333333" Margin="0,10">
                    <StackPanel Margin="10">
                        <CheckBox Name="chkInstallPython" IsChecked="True" Content="Install Python (Required)" Margin="0,5"/>
                        <CheckBox Name="chkInstallGit" IsChecked="True" Content="Install Git (Recommended)" Margin="0,5"/>
                        <CheckBox Name="chkInstallOllama" IsChecked="True" Content="Install Ollama (Recommended for offline operation)" Margin="0,5"/>
                        <CheckBox Name="chkCreateVirtualEnv" IsChecked="True" Content="Create Virtual Environment" Margin="0,5"/>
                        <CheckBox Name="chkAddDesktopShortcut" IsChecked="True" Content="Create Desktop Shortcut" Margin="0,5"/>
                        <CheckBox Name="chkAddStartMenu" IsChecked="True" Content="Add to Start Menu" Margin="0,5"/>
                        <CheckBox Name="chkAutoStart" IsChecked="False" Content="Start on System Boot" Margin="0,5"/>
                    </StackPanel>
                </GroupBox>

                <GroupBox Header="Advanced Options" Foreground="#FFFFFF" BorderBrush="#333333" Margin="0,10">
                    <StackPanel Margin="10">
                        <CheckBox Name="chkUseOpenAIUnofficial" IsChecked="True" Content="Use openai-unofficial package (Enables 10,000 agents)" Margin="0,5"/>
                        <CheckBox Name="chkDownloadModels" IsChecked="True" Content="Download Ollama models for offline use" Margin="0,5"/>
                        <CheckBox Name="chkCreateInstaller" IsChecked="False" Content="Create standalone installer after setup" Margin="0,5"/>
                        <StackPanel Orientation="Horizontal" Margin="0,5">
                            <TextBlock Text="Installation Directory: " VerticalAlignment="Center"/>
                            <TextBox Name="txtInstallDir" Text="" Width="300" Background="#1E1E1E" Foreground="#FFFFFF" BorderBrush="#333333" Padding="5,2"/>
                            <Button Name="btnBrowse" Content="Browse..." Margin="5,0,0,0"/>
                        </StackPanel>
                    </StackPanel>
                </GroupBox>

                <GroupBox Header="API Configuration" Foreground="#FFFFFF" BorderBrush="#333333" Margin="0,10">
                    <StackPanel Margin="10">
                        <StackPanel Orientation="Horizontal" Margin="0,5">
                            <TextBlock Text="OpenAI API Key (Optional): " VerticalAlignment="Center"/>
                            <TextBox Name="txtOpenAIKey" Width="300" Background="#1E1E1E" Foreground="#FFFFFF" BorderBrush="#333333" Padding="5,2"/>
                        </StackPanel>
                    </StackPanel>
                </GroupBox>

                <GroupBox Header="Agent Configuration" Foreground="#FFFFFF" BorderBrush="#333333" Margin="0,10">
                    <StackPanel Margin="10">
                        <StackPanel Orientation="Horizontal" Margin="0,5">
                            <TextBlock Text="Number of Pipelines: " VerticalAlignment="Center"/>
                            <TextBox Name="txtPipelines" Text="100" Width="100" Background="#1E1E1E" Foreground="#FFFFFF" BorderBrush="#333333" Padding="5,2"/>
                        </StackPanel>
                        <StackPanel Orientation="Horizontal" Margin="0,5">
                            <TextBlock Text="Agents per Pipeline: " VerticalAlignment="Center"/>
                            <TextBox Name="txtAgentsPerPipeline" Text="100" Width="100" Background="#1E1E1E" Foreground="#FFFFFF" BorderBrush="#333333" Padding="5,2"/>
                        </StackPanel>
                        <TextBlock Text="Total Agents: 10,000" Name="txtTotalAgents" Margin="0,5"/>
                    </StackPanel>
                </GroupBox>
            </StackPanel>
        </ScrollViewer>

        <!-- Progress -->
        <StackPanel Grid.Row="3" Margin="0,10">
            <TextBlock Name="txtStatus" Text="Ready to install" Margin="0,5"/>
            <ProgressBar Name="progressInstall" Value="0" Maximum="100"/>
        </StackPanel>

        <!-- Buttons -->
        <StackPanel Grid.Row="4" Orientation="Horizontal" HorizontalAlignment="Right" Margin="0,10,0,0">
            <Button Name="btnInstall" Content="Install" Width="100"/>
            <Button Name="btnCancel" Content="Cancel" Width="100"/>
        </StackPanel>
    </Grid>
</Window>
"@

# Function to write to log file
function Write-Log {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Message,
        
        [Parameter(Mandatory = $false)]
        [ValidateSet("INFO", "WARNING", "ERROR", "SUCCESS")]
        [string]$Level = "INFO"
    )
    
    # Create logs directory if it doesn't exist
    if (-not (Test-Path -Path $LOGS_DIR)) {
        New-Item -Path $LOGS_DIR -ItemType Directory -Force | Out-Null
    }
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    
    # Write to log file
    Add-Content -Path $LOG_FILE -Value $logMessage
    
    # Also output to console with color
    switch ($Level) {
        "INFO" { Write-Host $logMessage -ForegroundColor Cyan }
        "WARNING" { Write-Host $logMessage -ForegroundColor Yellow }
        "ERROR" { Write-Host $logMessage -ForegroundColor Red }
        "SUCCESS" { Write-Host $logMessage -ForegroundColor Green }
    }
}

# Function to check system requirements
function Test-SystemRequirements {
    param (
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$PythonStatus,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$RAMStatus,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$DiskStatus,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$GitStatus
    )
    
    $requirements = @{
        PythonMet = $false
        RAMMet = $false
        DiskMet = $false
        GitMet = $false
    }
    
    # Check Python
    try {
        $pythonInfo = Invoke-Expression "python --version" 2>&1
        if ($pythonInfo -match "Python (\d+\.\d+\.\d+)") {
            $version = [Version]$Matches[1]
            $minVersion = [Version]$MIN_PYTHON_VERSION
            
            if ($version -ge $minVersion) {
                $PythonStatus.Text = "Python: $version ✓"
                $PythonStatus.Foreground = "#00FF00"
                $requirements.PythonMet = $true
            } else {
                $PythonStatus.Text = "Python: $version (Minimum required: $MIN_PYTHON_VERSION) ✗"
                $PythonStatus.Foreground = "#FFA500"
            }
        } else {
            $PythonStatus.Text = "Python: Not installed ✗"
            $PythonStatus.Foreground = "#FFA500"
        }
    } catch {
        $PythonStatus.Text = "Python: Not installed ✗"
        $PythonStatus.Foreground = "#FFA500"
    }
    
    # Check RAM
    $computerSystem = Get-CimInstance -ClassName Win32_ComputerSystem
    $totalRAM = [math]::Round($computerSystem.TotalPhysicalMemory / 1GB, 2)
    
    if ($totalRAM -ge $RECOMMENDED_RAM_GB) {
        $RAMStatus.Text = "Memory: $totalRAM GB ✓"
        $RAMStatus.Foreground = "#00FF00"
        $requirements.RAMMet = $true
    } else {
        $RAMStatus.Text = "Memory: $totalRAM GB (Recommended: $RECOMMENDED_RAM_GB GB) ✗"
        $RAMStatus.Foreground = "#FFA500"
    }
    
    # Check Disk Space
    $drive = Get-PSDrive -Name $env:SystemDrive[0]
    $freeSpace = [math]::Round($drive.Free / 1GB, 2)
    
    if ($freeSpace -ge $RECOMMENDED_DISK_SPACE_GB) {
        $DiskStatus.Text = "Disk Space: $freeSpace GB free ✓"
        $DiskStatus.Foreground = "#00FF00"
        $requirements.DiskMet = $true
    } else {
        $DiskStatus.Text = "Disk Space: $freeSpace GB free (Recommended: $RECOMMENDED_DISK_SPACE_GB GB) ✗"
        $DiskStatus.Foreground = "#FFA500"
    }
    
    # Check Git
    try {
        $gitInfo = Invoke-Expression "git --version" 2>&1
        if ($gitInfo -match "git version") {
            $GitStatus.Text = "Git: $gitInfo ✓"
            $GitStatus.Foreground = "#00FF00"
            $requirements.GitMet = $true
        } else {
            $GitStatus.Text = "Git: Not installed ✗"
            $GitStatus.Foreground = "#FFA500"
        }
    } catch {
        $GitStatus.Text = "Git: Not installed ✗"
        $GitStatus.Foreground = "#FFA500"
    }
    
    return $requirements
}

# Function to download a file with progress
function Download-FileWithProgress {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Url,
        
        [Parameter(Mandatory = $true)]
        [string]$OutputPath,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.ProgressBar]$ProgressBar
    )
    
    try {
        # Create directory if it doesn't exist
        $directory = Split-Path -Path $OutputPath -Parent
        if (-not (Test-Path -Path $directory)) {
            New-Item -Path $directory -ItemType Directory -Force | Out-Null
        }
        
        $StatusText.Text = "Downloading $(Split-Path -Path $OutputPath -Leaf)..."
        
        # Use .NET WebClient for download
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($Url, $OutputPath)
        
        $StatusText.Text = "Download completed: $(Split-Path -Path $OutputPath -Leaf)"
        Write-Log -Message "Downloaded $Url to $OutputPath" -Level "INFO"
        return $true
    } catch {
        $StatusText.Text = "Error downloading file: $($_.Exception.Message)"
        Write-Log -Message "Error downloading $Url to $OutputPath: $($_.Exception.Message)" -Level "ERROR"
        return $false
    }
}

# Function to install Python
function Install-Python {
    param (
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.ProgressBar]$ProgressBar
    )
    
    try {
        $StatusText.Text = "Installing Python..."
        $ProgressBar.Value = 30
        
        # Download Python installer if not already downloaded
        if (-not (Test-Path -Path $PYTHON_INSTALLER)) {
            $downloadSuccess = Download-FileWithProgress -Url $PYTHON_URL -OutputPath $PYTHON_INSTALLER -StatusText $StatusText -ProgressBar $ProgressBar
            if (-not $downloadSuccess) {
                return $false
            }
        }
        
        # Install Python silently with necessary options
        $arguments = "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0 Include_pip=1"
        Start-Process -FilePath $PYTHON_INSTALLER -ArgumentList $arguments -Wait
        
        # Verify installation
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        $pythonInstalled = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
        
        if ($pythonInstalled) {
            $StatusText.Text = "Python installed successfully"
            $ProgressBar.Value = 40
            Write-Log -Message "Python installed successfully" -Level "SUCCESS"
            return $true
        } else {
            $StatusText.Text = "Failed to install Python"
            Write-Log -Message "Failed to install Python" -Level "ERROR"
            return $false
        }
    } catch {
        $StatusText.Text = "Error installing Python: $($_.Exception.Message)"
        Write-Log -Message "Error installing Python: $($_.Exception.Message)" -Level "ERROR"
        return $false
    }
}

# Function to install Git
function Install-Git {
    param (
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.ProgressBar]$ProgressBar
    )
    
    try {
        $StatusText.Text = "Installing Git..."
        $ProgressBar.Value = 45
        
        # Download Git installer if not already downloaded
        if (-not (Test-Path -Path $GIT_INSTALLER)) {
            $downloadSuccess = Download-FileWithProgress -Url $GIT_URL -OutputPath $GIT_INSTALLER -StatusText $StatusText -ProgressBar $ProgressBar
            if (-not $downloadSuccess) {
                return $false
            }
        }
        
        # Install Git silently
        $arguments = "/VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /COMPONENTS=`"icons,ext\reg\shellhere,assoc,assoc_sh`""
        Start-Process -FilePath $GIT_INSTALLER -ArgumentList $arguments -Wait
        
        # Verify installation
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        $gitInstalled = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
        
        if ($gitInstalled) {
            $StatusText.Text = "Git installed successfully"
            $ProgressBar.Value = 50
            Write-Log -Message "Git installed successfully" -Level "SUCCESS"
            return $true
        } else {
            $StatusText.Text = "Failed to install Git"
            Write-Log -Message "Failed to install Git" -Level "ERROR"
            return $false
        }
    } catch {
        $StatusText.Text = "Error installing Git: $($_.Exception.Message)"
        Write-Log -Message "Error installing Git: $($_.Exception.Message)" -Level "ERROR"
        return $false
    }
}

# Function to install Ollama
function Install-Ollama {
    param (
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.ProgressBar]$ProgressBar
    )
    
    try {
        $StatusText.Text = "Installing Ollama..."
        $ProgressBar.Value = 55
        
        # Download Ollama installer if not already downloaded
        if (-not (Test-Path -Path $OLLAMA_INSTALLER)) {
            $downloadSuccess = Download-FileWithProgress -Url $OLLAMA_URL -OutputPath $OLLAMA_INSTALLER -StatusText $StatusText -ProgressBar $ProgressBar
            if (-not $downloadSuccess) {
                return $false
            }
        }
        
        # Install Ollama silently
        Start-Process -FilePath $OLLAMA_INSTALLER -ArgumentList "/S" -Wait
        
        # Verify installation (Ollama typically installs to Program Files)
        $ollamaInstalled = Test-Path -Path "${env:ProgramFiles}\Ollama\ollama.exe"
        
        if ($ollamaInstalled) {
            $StatusText.Text = "Ollama installed successfully"
            $ProgressBar.Value = 60
            Write-Log -Message "Ollama installed successfully" -Level "SUCCESS"
            return $true
        } else {
            $StatusText.Text = "Failed to install Ollama"
            Write-Log -Message "Failed to install Ollama" -Level "ERROR"
            return $false
        }
    } catch {
        $StatusText.Text = "Error installing Ollama: $($_.Exception.Message)"
        Write-Log -Message "Error installing Ollama: $($_.Exception.Message)" -Level "ERROR"
        return $false
    }
}

# Function to create Python virtual environment
function Create-VirtualEnvironment {
    param (
        [Parameter(Mandatory = $true)]
        [string]$InstallDir,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.ProgressBar]$ProgressBar
    )
    
    try {
        $StatusText.Text = "Creating Python virtual environment..."
        $ProgressBar.Value = 65
        
        $venvDir = Join-Path $InstallDir "venv"
        
        # Create virtual environment
        $process = Start-Process -FilePath "python" -ArgumentList "-m venv $venvDir" -Wait -NoNewWindow -PassThru
        
        if ($process.ExitCode -eq 0) {
            $StatusText.Text = "Virtual environment created successfully"
            $ProgressBar.Value = 70
            Write-Log -Message "Virtual environment created at $venvDir" -Level "SUCCESS"
            return $true
        } else {
            $StatusText.Text = "Failed to create virtual environment"
            Write-Log -Message "Failed to create virtual environment" -Level "ERROR"
            return $false
        }
    } catch {
        $StatusText.Text = "Error creating virtual environment: $($_.Exception.Message)"
        Write-Log -Message "Error creating virtual environment: $($_.Exception.Message)" -Level "ERROR"
        return $false
    }
}

# Function to install Python packages
function Install-PythonPackages {
    param (
        [Parameter(Mandatory = $true)]
        [string]$InstallDir,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.ProgressBar]$ProgressBar,
        
        [Parameter(Mandatory = $false)]
        [bool]$UseOpenAIUnofficial = $true
    )
    
    try {
        $StatusText.Text = "Installing Python packages..."
        $ProgressBar.Value = 75
        
        $venvDir = Join-Path $InstallDir "venv"
        $pipPath = Join-Path $venvDir "Scripts\pip.exe"
        
        # Create requirements.txt
        $requirementsPath = Join-Path $TEMP_DIR "requirements.txt"
        $requirements = @"
streamlit>=1.26.0
streamlit-ace>=0.1.1
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
requests>=2.31.0
python-dotenv>=1.0.0
pillow>=10.0.0
pycryptodome>=3.19.0
psutil>=5.9.5
ccxt>=4.0.0
SQLAlchemy>=2.0.0
fastapi>=0.103.0
uvicorn>=0.23.0
websockets>=11.0.0
python-multipart>=0.0.6
pydantic>=2.3.0
jinja2>=3.1.2
markdown>=3.5.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
aiohttp>=3.8.5
pytest>=7.4.0
black>=23.7.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.5.0
"@
        
        # Add OpenAI package based on user selection
        if ($UseOpenAIUnofficial) {
            $requirements += "openai-unofficial>=1.0.0`n"
            $StatusText.Text = "Installing Python packages (including openai-unofficial)..."
            Write-Log -Message "Using openai-unofficial package" -Level "INFO"
        } else {
            $requirements += "openai>=1.0.0`n"
        }
        
        Set-Content -Path $requirementsPath -Value $requirements
        
        # Install packages
        $process = Start-Process -FilePath $pipPath -ArgumentList "install -r $requirementsPath" -Wait -NoNewWindow -PassThru
        
        if ($process.ExitCode -eq 0) {
            # Install ollama client if needed
            Start-Process -FilePath $pipPath -ArgumentList "install ollama" -Wait -NoNewWindow
            
            $StatusText.Text = "Python packages installed successfully"
            $ProgressBar.Value = 80
            Write-Log -Message "Python packages installed successfully" -Level "SUCCESS"
            return $true
        } else {
            $StatusText.Text = "Failed to install Python packages"
            Write-Log -Message "Failed to install Python packages" -Level "ERROR"
            return $false
        }
    } catch {
        $StatusText.Text = "Error installing Python packages: $($_.Exception.Message)"
        Write-Log -Message "Error installing Python packages: $($_.Exception.Message)" -Level "ERROR"
        return $false
    }
}

# Function to download Ollama models
function Download-OllamaModels {
    param (
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.ProgressBar]$ProgressBar
    )
    
    try {
        $StatusText.Text = "Downloading Ollama models (this may take a while)..."
        $ProgressBar.Value = 85
        
        # Download a small model for testing
        $process = Start-Process -FilePath "${env:ProgramFiles}\Ollama\ollama.exe" -ArgumentList "pull tinyllama" -Wait -NoNewWindow -PassThru
        
        if ($process.ExitCode -eq 0) {
            $StatusText.Text = "Ollama models downloaded successfully"
            $ProgressBar.Value = 90
            Write-Log -Message "Ollama models downloaded successfully" -Level "SUCCESS"
            return $true
        } else {
            $StatusText.Text = "Failed to download Ollama models"
            Write-Log -Message "Failed to download Ollama models" -Level "ERROR"
            return $false
        }
    } catch {
        $StatusText.Text = "Error downloading Ollama models: $($_.Exception.Message)"
        Write-Log -Message "Error downloading Ollama models: $($_.Exception.Message)" -Level "ERROR"
        return $false
    }
}

# Function to clone repository
function Clone-Repository {
    param (
        [Parameter(Mandatory = $true)]
        [string]$InstallDir,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.ProgressBar]$ProgressBar
    )
    
    try {
        $StatusText.Text = "Cloning Skyscope Sentinel repository..."
        $ProgressBar.Value = 50
        
        $repoUrl = "https://github.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business.git"
        $repoDir = Join-Path $InstallDir "repo"
        
        # Create directory if it doesn't exist
        if (-not (Test-Path -Path $repoDir)) {
            New-Item -Path $repoDir -ItemType Directory -Force | Out-Null
        }
        
        # Clone repository
        $process = Start-Process -FilePath "git" -ArgumentList "clone $repoUrl $repoDir" -Wait -NoNewWindow -PassThru
        
        if ($process.ExitCode -eq 0) {
            $StatusText.Text = "Repository cloned successfully"
            $ProgressBar.Value = 60
            Write-Log -Message "Repository cloned to $repoDir" -Level "SUCCESS"
            return $true
        } else {
            # If git clone fails, try to download the generate_skyscope_system.sh script and run it
            $StatusText.Text = "Cloning failed, trying alternative installation method..."
            
            # Create shell script
            $generateScriptPath = Join-Path $InstallDir "generate_skyscope_system.sh"
            $scriptUrl = "https://raw.githubusercontent.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business/main/generate_skyscope_system.sh"
            
            $downloadSuccess = Download-FileWithProgress -Url $scriptUrl -OutputPath $generateScriptPath -StatusText $StatusText -ProgressBar $ProgressBar
            
            if ($downloadSuccess) {
                $StatusText.Text = "Using alternative installation method"
                Write-Log -Message "Using alternative installation method with generate_skyscope_system.sh" -Level "INFO"
                return $true
            } else {
                $StatusText.Text = "Failed to clone repository or download installation script"
                Write-Log -Message "Failed to clone repository or download installation script" -Level "ERROR"
                return $false
            }
        }
    } catch {
        $StatusText.Text = "Error cloning repository: $($_.Exception.Message)"
        Write-Log -Message "Error cloning repository: $($_.Exception.Message)" -Level "ERROR"
        return $false
    }
}

# Function to create configuration
function Create-Configuration {
    param (
        [Parameter(Mandatory = $true)]
        [string]$InstallDir,
        
        [Parameter(Mandatory = $true)]
        [string]$OpenAIKey,
        
        [Parameter(Mandatory = $true)]
        [int]$Pipelines,
        
        [Parameter(Mandatory = $true)]
        [int]$AgentsPerPipeline,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.ProgressBar]$ProgressBar
    )
    
    try {
        $StatusText.Text = "Creating configuration..."
        $ProgressBar.Value = 92
        
        # Create config directory
        $configDir = Join-Path $InstallDir "config"
        if (-not (Test-Path -Path $configDir)) {
            New-Item -Path $configDir -ItemType Directory -Force | Out-Null
        }
        
        # Create system_config.json
        $configPath = Join-Path $configDir "system_config.json"
        $config = @{
            system_name = "Skyscope Sentinel Intelligence AI"
            version = "1.0.0"
            agent_count = $Pipelines * $AgentsPerPipeline
            pipeline_count = $Pipelines
            default_theme = "glass_dark"
            enable_autonomous_operations = $true
            enable_crypto_focus = $true
            enable_perplexica_integration = $true
            perplexica_path = Join-Path $env:USERPROFILE "Perplexica"
            enable_ollama = $true
            enable_openai_fallback = $true
            database_type = "sqlite"
            database_path = Join-Path $InstallDir "data\skyscope.db"
            log_level = "INFO"
            max_memory_gb = 8
            initialized_at = (Get-Date).ToString("o")
        }
        
        $configJson = $config | ConvertTo-Json -Depth 10
        Set-Content -Path $configPath -Value $configJson
        
        # Create api_config.json if API key is provided
        if (-not [string]::IsNullOrWhiteSpace($OpenAIKey)) {
            $apiConfigPath = Join-Path $configDir "api_config.json"
            $apiConfig = @{
                openai = @{
                    api_key = $OpenAIKey
                    organization = ""
                    default_model = "gpt-4o"
                }
            }
            
            $apiConfigJson = $apiConfig | ConvertTo-Json -Depth 10
            Set-Content -Path $apiConfigPath -Value $apiConfigJson
        }
        
        $StatusText.Text = "Configuration created successfully"
        $ProgressBar.Value = 95
        Write-Log -Message "Configuration created successfully" -Level "SUCCESS"
        return $true
    } catch {
        $StatusText.Text = "Error creating configuration: $($_.Exception.Message)"
        Write-Log -Message "Error creating configuration: $($_.Exception.Message)" -Level "ERROR"
        return $false
    }
}

# Function to create shortcuts
function Create-Shortcuts {
    param (
        [Parameter(Mandatory = $true)]
        [string]$InstallDir,
        
        [Parameter(Mandatory = $false)]
        [bool]$CreateDesktopShortcut = $true,
        
        [Parameter(Mandatory = $false)]
        [bool]$AddToStartMenu = $true,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.ProgressBar]$ProgressBar
    )
    
    try {
        $StatusText.Text = "Creating shortcuts..."
        $ProgressBar.Value = 97
        
        # Create start.bat
        $startBatPath = Join-Path $InstallDir "start.bat"
        $startBatContent = @"
@echo off
title Skyscope Sentinel Intelligence AI
echo Starting Skyscope Sentinel Intelligence AI...
cd /d "%~dp0"
call venv\Scripts\activate.bat
python main_launcher.py %*
"@
        Set-Content -Path $startBatPath -Value $startBatContent
        
        # Create desktop shortcut
        if ($CreateDesktopShortcut) {
            $desktopPath = [Environment]::GetFolderPath("Desktop")
            $shortcutPath = Join-Path $desktopPath "Skyscope Sentinel AI.lnk"
            
            $WshShell = New-Object -ComObject WScript.Shell
            $Shortcut = $WshShell.CreateShortcut($shortcutPath)
            $Shortcut.TargetPath = $startBatPath
            $Shortcut.WorkingDirectory = $InstallDir
            $Shortcut.IconLocation = Join-Path $InstallDir "assets\images\skyscope.ico"
            $Shortcut.Description = "Skyscope Sentinel Intelligence AI"
            $Shortcut.Save()
            
            Write-Log -Message "Desktop shortcut created at $shortcutPath" -Level "INFO"
        }
        
        # Add to Start Menu
        if ($AddToStartMenu) {
            $startMenuPath = [Environment]::GetFolderPath("Programs")
            $startMenuDir = Join-Path $startMenuPath "Skyscope Sentinel AI"
            
            if (-not (Test-Path -Path $startMenuDir)) {
                New-Item -Path $startMenuDir -ItemType Directory -Force | Out-Null
            }
            
            $startMenuShortcutPath = Join-Path $startMenuDir "Skyscope Sentinel AI.lnk"
            
            $WshShell = New-Object -ComObject WScript.Shell
            $Shortcut = $WshShell.CreateShortcut($startMenuShortcutPath)
            $Shortcut.TargetPath = $startBatPath
            $Shortcut.WorkingDirectory = $InstallDir
            $Shortcut.IconLocation = Join-Path $InstallDir "assets\images\skyscope.ico"
            $Shortcut.Description = "Skyscope Sentinel Intelligence AI"
            $Shortcut.Save()
            
            Write-Log -Message "Start Menu shortcut created at $startMenuShortcutPath" -Level "INFO"
        }
        
        $StatusText.Text = "Shortcuts created successfully"
        $ProgressBar.Value = 98
        Write-Log -Message "Shortcuts created successfully" -Level "SUCCESS"
        return $true
    } catch {
        $StatusText.Text = "Error creating shortcuts: $($_.Exception.Message)"
        Write-Log -Message "Error creating shortcuts: $($_.Exception.Message)" -Level "ERROR"
        return $false
    }
}

# Function to create standalone installer
function Create-StandaloneInstaller {
    param (
        [Parameter(Mandatory = $true)]
        [string]$InstallDir,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.ProgressBar]$ProgressBar
    )
    
    try {
        $StatusText.Text = "Creating standalone installer..."
        $ProgressBar.Value = 99
        
        # Download Inno Setup if not already downloaded
        if (-not (Test-Path -Path $INNO_SETUP_INSTALLER)) {
            $downloadSuccess = Download-FileWithProgress -Url $INNO_SETUP_URL -OutputPath $INNO_SETUP_INSTALLER -StatusText $StatusText -ProgressBar $ProgressBar
            if (-not $downloadSuccess) {
                return $false
            }
        }
        
        # Install Inno Setup silently
        Start-Process -FilePath $INNO_SETUP_INSTALLER -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART" -Wait
        
        # Create Inno Setup script
        $innoScriptPath = Join-Path $TEMP_DIR "skyscope_installer.iss"
        $innoScriptContent = @"
#define MyAppName "Skyscope Sentinel Intelligence AI"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Skyscope Technologies"
#define MyAppURL "https://skyscope.ai"
#define MyAppExeName "start.bat"

[Setup]
AppId={{F9A6E8D7-3B2F-4A1C-9E8D-7F3B2FA1C9E8}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\SkyscopeSentinelAI
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=$InstallDir\LICENSE
OutputDir=$PSScriptRoot
OutputBaseFilename=SkyscopeSentinelSetup
SetupIconFile=$InstallDir\assets\images\skyscope.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
WizardImageFile=$InstallDir\assets\images\wizard.bmp
WizardSmallImageFile=$InstallDir\assets\images\wizard-small.bmp

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
Source: "$InstallDir\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
"@
        Set-Content -Path $innoScriptPath -Value $innoScriptContent
        
        # Run Inno Setup compiler
        $innoCompilerPath = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
        if (-not (Test-Path -Path $innoCompilerPath)) {
            $innoCompilerPath = "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
        }
        
        if (Test-Path -Path $innoCompilerPath) {
            Start-Process -FilePath $innoCompilerPath -ArgumentList """$innoScriptPath""" -Wait -NoNewWindow
            
            if (Test-Path -Path $INSTALLER_OUTPUT) {
                $StatusText.Text = "Standalone installer created successfully"
                $ProgressBar.Value = 100
                Write-Log -Message "Standalone installer created at $INSTALLER_OUTPUT" -Level "SUCCESS"
                return $true
            } else {
                $StatusText.Text = "Failed to create standalone installer"
                Write-Log -Message "Failed to create standalone installer" -Level "ERROR"
                return $false
            }
        } else {
            $StatusText.Text = "Inno Setup compiler not found"
            Write-Log -Message "Inno Setup compiler not found" -Level "ERROR"
            return $false
        }
    } catch {
        $StatusText.Text = "Error creating standalone installer: $($_.Exception.Message)"
        Write-Log -Message "Error creating standalone installer: $($_.Exception.Message)" -Level "ERROR"
        return $false
    }
}

# Function to clean up temporary files
function Clean-TempFiles {
    param (
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText
    )
    
    try {
        $StatusText.Text = "Cleaning up temporary files..."
        
        if (Test-Path -Path $TEMP_DIR) {
            Remove-Item -Path $TEMP_DIR -Recurse -Force -ErrorAction SilentlyContinue
        }
        
        $StatusText.Text = "Temporary files cleaned up"
        Write-Log -Message "Temporary files cleaned up" -Level "INFO"
    } catch {
        $StatusText.Text = "Error cleaning up temporary files: $($_.Exception.Message)"
        Write-Log -Message "Error cleaning up temporary files: $($_.Exception.Message)" -Level "WARNING"
    }
}

# Main installation function
function Start-Installation {
    param (
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.TextBlock]$StatusText,
        
        [Parameter(Mandatory = $true)]
        [System.Windows.Controls.ProgressBar]$ProgressBar,
        
        [Parameter(Mandatory = $true)]
        [string]$InstallDir,
        
        [Parameter(Mandatory = $false)]
        [bool]$InstallPython = $true,
        
        [Parameter(Mandatory = $false)]
        [bool]$InstallGit = $true,
        
        [Parameter(Mandatory = $false)]
        [bool]$InstallOllama = $true,
        
        [Parameter(Mandatory = $false)]
        [bool]$CreateVirtualEnv = $true,
        
        [Parameter(Mandatory = $false)]
        [bool]$UseOpenAIUnofficial = $true,
        
        [Parameter(Mandatory = $false)]
        [bool]$DownloadModels = $true,
        
        [Parameter(Mandatory = $false)]
        [bool]$CreateDesktopShortcut = $true,
        
        [Parameter(Mandatory = $false)]
        [bool]$AddToStartMenu = $true,
        
        [Parameter(Mandatory = $false)]
        [bool]$AutoStart = $false,
        
        [Parameter(Mandatory = $false)]
        [bool]$CreateInstaller = $false,
        
        [Parameter(Mandatory = $false)]
        [string]$OpenAIKey = "",
        
        [Parameter(Mandatory = $false)]
        [int]$Pipelines = 100,
        
        [Parameter(Mandatory = $false)]
        [int]$AgentsPerPipeline = 100
    )
    
    try {
        # Create temp directory
        if (-not (Test-Path -Path $TEMP_DIR)) {
            New-Item -Path $TEMP_DIR -ItemType Directory -Force | Out-Null
        }
        
        # Create installation directory
        if (-not (Test-Path -Path $InstallDir)) {
            New-Item -Path $InstallDir -ItemType Directory -Force | Out-Null
        }
        
        # Initialize log
        Write-Log -Message "Starting installation of Skyscope Sentinel Intelligence AI v$PRODUCT_VERSION" -Level "INFO"
        Write-Log -Message "Installation directory: $InstallDir" -Level "INFO"
        
        # Update progress
        $StatusText.Text = "Starting installation..."
        $ProgressBar.Value = 10
        
        # Install Python if needed
        if ($InstallPython) {
            $pythonSuccess = Install-Python -StatusText $StatusText -ProgressBar $ProgressBar
            if (-not $pythonSuccess) {
                return $false
            }
        }
        
        # Install Git if needed
        if ($InstallGit) {
            $gitSuccess = Install-Git -StatusText $StatusText -ProgressBar $ProgressBar
            if (-not $gitSuccess) {
                return $false
            }
        }
        
        # Clone repository
        $cloneSuccess = Clone-Repository -InstallDir $InstallDir -StatusText $StatusText -ProgressBar $ProgressBar
        if (-not $cloneSuccess) {
            return $false
        }
        
        # Create virtual environment if needed
        if ($CreateVirtualEnv) {
            $venvSuccess = Create-VirtualEnvironment -InstallDir $InstallDir -StatusText $StatusText -ProgressBar $ProgressBar
            if (-not $venvSuccess) {
                return $false
            }
        }
        
        # Install Python packages
        $packagesSuccess = Install-PythonPackages -InstallDir $InstallDir -StatusText $StatusText -ProgressBar $ProgressBar -UseOpenAIUnofficial $UseOpenAIUnofficial
        if (-not $packagesSuccess) {
            return $false
        }
        
        # Install Ollama if needed
        if ($InstallOllama) {
            $ollamaSuccess = Install-Ollama -StatusText $StatusText -ProgressBar $ProgressBar
            if (-not $ollamaSuccess) {
                return $false
            }
            
            # Download Ollama models if needed
            if ($DownloadModels) {
                $modelsSuccess = Download-OllamaModels -StatusText $StatusText -ProgressBar $ProgressBar
                if (-not $modelsSuccess) {
                    $StatusText.Text = "Warning: Failed to download Ollama models, continuing installation..."
                    Write-Log -Message "Failed to download Ollama models, continuing installation" -Level "WARNING"
                }
            }
        }
        
        # Create configuration
        $configSuccess = Create-Configuration -InstallDir $InstallDir -OpenAIKey $OpenAIKey -Pipelines $Pipelines -AgentsPerPipeline $AgentsPerPipeline -StatusText $StatusText -ProgressBar $ProgressBar
        if (-not $configSuccess) {
            return $false
        }
        
        # Create shortcuts
        $shortcutsSuccess = Create-Shortcuts -InstallDir $InstallDir -CreateDesktopShortcut $CreateDesktopShortcut -AddToStartMenu $AddToStartMenu -StatusText $StatusText -ProgressBar $ProgressBar
        if (-not $shortcutsSuccess) {
            return $false
        }
        
        # Create standalone installer if needed
        if ($CreateInstaller) {
            $installerSuccess = Create-StandaloneInstaller -InstallDir $InstallDir -StatusText $StatusText -ProgressBar $ProgressBar
            if (-not $installerSuccess) {
                $StatusText.Text = "Warning: Failed to create standalone installer, continuing installation..."
                Write-Log -Message "Failed to create standalone installer, continuing installation" -Level "WARNING"
            }
        }
        
        # Clean up temporary files
        Clean-TempFiles -StatusText $StatusText
        
        # Set up auto-start if needed
        if ($AutoStart) {
            $startupFolder = [Environment]::GetFolderPath("Startup")
            $startupShortcutPath = Join-Path $startupFolder "Skyscope Sentinel AI.lnk"
            $startBatPath = Join-Path $InstallDir "start.bat"
            
            $WshShell = New-Object -ComObject WScript.Shell
            $Shortcut = $WshShell.CreateShortcut($startupShortcutPath)
            $Shortcut.TargetPath = $startBatPath
            $Shortcut.WorkingDirectory = $InstallDir
            $Shortcut.IconLocation = Join-Path $InstallDir "assets\images\skyscope.ico"
            $Shortcut.Description = "Skyscope Sentinel Intelligence AI"
            $Shortcut.Save()
            
            Write-Log -Message "Auto-start shortcut created at $startupShortcutPath" -Level "INFO"
        }
        
        # Installation complete
        $StatusText.Text = "Installation completed successfully!"
        $ProgressBar.Value = 100
        Write-Log -Message "Installation completed successfully" -Level "SUCCESS"
        
        return $true
    } catch {
        $StatusText.Text = "Installation failed: $($_.Exception.Message)"
        Write-Log -Message "Installation failed: $($_.Exception.Message)" -Level "ERROR"
        Write-Log -Message $_.ScriptStackTrace -Level "ERROR"
        return $false
    }
}

# Main script execution
try {
    # Create temp directory
    if (-not (Test-Path -Path $TEMP_DIR)) {
        New-Item -Path $TEMP_DIR -ItemType Directory -Force | Out-Null
    }
    
    # Create XAML reader
    $reader = [System.Xml.XmlReader]::Create([System.IO.StringReader]::new($xamlContent))
    $window = [System.Windows.Markup.XamlReader]::Load($reader)
    
    # Get UI elements
    $txtPythonStatus = $window.FindName("txtPythonStatus")
    $txtRAMStatus = $window.FindName("txtRAMStatus")
    $txtDiskStatus = $window.FindName("txtDiskStatus")
    $txtGitStatus = $window.FindName("txtGitStatus")
    
    $chkInstallPython = $window.FindName("chkInstallPython")
    $chkInstallGit = $window.FindName("chkInstallGit")
    $chkInstallOllama = $window.FindName("chkInstallOllama")
    $chkCreateVirtualEnv = $window.FindName("chkCreateVirtualEnv")
    $chkAddDesktopShortcut = $window.FindName("chkAddDesktopShortcut")
    $chkAddStartMenu = $window.FindName("chkAddStartMenu")
    $chkAutoStart = $window.FindName("chkAutoStart")
    
    $chkUseOpenAIUnofficial = $window.FindName("chkUseOpenAIUnofficial")
    $chkDownloadModels = $window.FindName("chkDownloadModels")
    $chkCreateInstaller = $window.FindName("chkCreateInstaller")
    
    $txtInstallDir = $window.FindName("txtInstallDir")
    $btnBrowse = $window.FindName("btnBrowse")
    
    $txtOpenAIKey = $window.FindName("txtOpenAIKey")
    $txtPipelines = $window.FindName("txtPipelines")
    $txtAgentsPerPipeline = $window.FindName("txtAgentsPerPipeline")
    $txtTotalAgents = $window.FindName("txtTotalAgents")
    
    $txtStatus = $window.FindName("txtStatus")
    $progressInstall = $window.FindName("progressInstall")
    
    $btnInstall = $window.FindName("btnInstall")
    $btnCancel = $window.FindName("btnCancel")
    
    # Set default installation directory
    $txtInstallDir.Text = $INSTALL_DIR
    
    # Check system requirements
    $requirements = Test-SystemRequirements -PythonStatus $txtPythonStatus -RAMStatus $txtRAMStatus -DiskStatus $txtDiskStatus -GitStatus $txtGitStatus
    
    # Set installation options based on requirements
    $chkInstallPython.IsChecked = -not $requirements.PythonMet
    $chkInstallGit.IsChecked = -not $requirements.GitMet
    
    # Handle browse button click
    $btnBrowse.Add_Click({
        $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
        $folderBrowser.Description = "Select Installation Directory"
        $folderBrowser.SelectedPath = $txtInstallDir.Text
        
        if ($folderBrowser.ShowDialog() -eq "OK") {
            $txtInstallDir.Text = $folderBrowser.SelectedPath
        }
    })
    
    # Handle pipeline and agents per pipeline changes
    $txtPipelines.Add_TextChanged({
        try {
            $pipelines = [int]$txtPipelines.Text
            $agentsPerPipeline = [int]$txtAgentsPerPipeline.Text
            $totalAgents = $pipelines * $agentsPerPipeline
            $txtTotalAgents.Text = "Total Agents: $totalAgents"
        } catch {
            $txtTotalAgents.Text = "Total Agents: Invalid input"
        }
    })
    
    $txtAgentsPerPipeline.Add_TextChanged({
        try {
            $pipelines = [int]$txtPipelines.Text
            $agentsPerPipeline = [int]$txtAgentsPerPipeline.Text
            $totalAgents = $pipelines * $agentsPerPipeline
            $txtTotalAgents.Text = "Total Agents: $totalAgents"
        } catch {
            $txtTotalAgents.Text = "Total Agents: Invalid input"
        }
    })
    
    # Handle install button click
    $btnInstall.Add_Click({
        # Disable UI elements during installation
        $btnInstall.IsEnabled = $false
        $btnCancel.IsEnabled = $false
        $txtInstallDir.IsEnabled = $false
        $btnBrowse.IsEnabled = $false
        
        # Get installation options
        $installDir = $txtInstallDir.Text
        $installPython = $chkInstallPython.IsChecked
        $installGit = $chkInstallGit.IsChecked
        $installOllama = $chkInstallOllama.IsChecked
        $createVirtualEnv = $chkCreateVirtualEnv.IsChecked
        $useOpenAIUnofficial = $chkUseOpenAIUnofficial.IsChecked
        $downloadModels = $chkDownloadModels.IsChecked
        $createDesktopShortcut = $chkAddDesktopShortcut.IsChecked
        $addToStartMenu = $chkAddStartMenu.IsChecked
        $autoStart = $chkAutoStart.IsChecked
        $createInstaller = $chkCreateInstaller.IsChecked
        $openAIKey = $txtOpenAIKey.Text
        
        try {
            $pipelines = [int]$txtPipelines.Text
            $agentsPerPipeline = [int]$txtAgentsPerPipeline.Text
        } catch {
            $txtStatus.Text = "Invalid pipeline or agents per pipeline value"
            $btnInstall.IsEnabled = $true
            $btnCancel.IsEnabled = $true
            $txtInstallDir.IsEnabled = $true
            $btnBrowse.IsEnabled = $true
            return
        }
        
        # Start installation in a new thread
        $installThread = [System.Threading.Thread]::new({
            $installSuccess = Start-Installation -StatusText $txtStatus -ProgressBar $progressInstall `
                -InstallDir $installDir -InstallPython $installPython -InstallGit $installGit `
                -InstallOllama $installOllama -CreateVirtualEnv $createVirtualEnv `
                -UseOpenAIUnofficial $useOpenAIUnofficial -DownloadModels $downloadModels `
                -CreateDesktopShortcut $createDesktopShortcut -AddToStartMenu $addToStartMenu `
                -AutoStart $autoStart -CreateInstaller $createInstaller -OpenAIKey $openAIKey `
                -Pipelines $pipelines -AgentsPerPipeline $agentsPerPipeline
            
            # Update UI on completion
            $window.Dispatcher.Invoke({
                if ($installSuccess) {
                    $txtStatus.Text = "Installation completed successfully!"
                    
                    # Show completion message
                    $result = [System.Windows.MessageBox]::Show(
                        "Skyscope Sentinel Intelligence AI has been installed successfully!`n`nDo you want to launch it now?",
                        "Installation Complete",
                        [System.Windows.MessageBoxButton]::YesNo,
                        [System.Windows.MessageBoxImage]::Information
                    )
                    
                    if ($result -eq "Yes") {
                        # Launch the application
                        $startBatPath = Join-Path $installDir "start.bat"
                        if (Test-Path -Path $startBatPath) {
                            Start-Process -FilePath $startBatPath
                        }
                    }
                    
                    # Close the installer
                    $window.Close()
                } else {
                    $btnInstall.IsEnabled = $true
                    $btnCancel.IsEnabled = $true
                    $txtInstallDir.IsEnabled = $true
                    $btnBrowse.IsEnabled = $true
                }
            })
        })
        
        $installThread.Start()
    })
    
    # Handle cancel button click
    $btnCancel.Add_Click({
        $result = [System.Windows.MessageBox]::Show(
            "Are you sure you want to cancel the installation?",
            "Cancel Installation",
            [System.Windows.MessageBoxButton]::YesNo,
            [System.Windows.MessageBoxImage]::Question
        )
        
        if ($result -eq "Yes") {
            $window.Close()
        }
    })
    
    # Show the window
    $window.ShowDialog() | Out-Null
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    
    # Show error message
    [System.Windows.MessageBox]::Show(
        "An error occurred during installation:`n`n$($_.Exception.Message)",
        "Installation Error",
        [System.Windows.MessageBoxButton]::OK,
        [System.Windows.MessageBoxImage]::Error
    )
} finally {
    # Clean up temporary files
    if (Test-Path -Path $TEMP_DIR) {
        Remove-Item -Path $TEMP_DIR -Recurse -Force -ErrorAction SilentlyContinue
    }
}
