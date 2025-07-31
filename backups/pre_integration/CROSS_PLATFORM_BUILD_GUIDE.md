# Skyscope Sentinel Intelligence - Cross-Platform Build Guide

## Overview

This guide provides comprehensive instructions for building cross-platform GUI applications for macOS, Windows, and Linux using modern Python packaging tools.

## Enhanced Build System Features

### 🚀 **Modern Tools Integration**
- **uv**: Ultra-fast Python package manager for dependency installation
- **conda**: Environment management for complex dependencies
- **PyInstaller**: Application packaging with optimized configurations
- **Platform-specific packaging**: DMG for macOS, AppImage for Linux, MSI for Windows

### 🔧 **Issues Fixed**
1. **Missing cryptography dependency** - Now properly installed
2. **Logo/icon file issues** - Uses provided `skyscope-logo.png` and `Skyscope.icns`
3. **PyInstaller path errors** - Fixed with absolute paths and proper data inclusion
4. **Hidden imports** - Comprehensive list of required modules
5. **Cross-platform compatibility** - Handles OS-specific requirements

## Build Scripts Available

### 1. `final_build.sh` - Complete Cross-Platform Builder
- Detects OS automatically
- Installs dependencies with uv/conda
- Creates optimized PyInstaller spec
- Builds native applications
- Creates distribution packages (DMG, AppImage, MSI)

### 2. `build_cross_platform.sh` - Advanced Multi-Platform Builder
- Supports building for all platforms from any OS
- Uses modern packaging tools (uv, conda)
- Creates professional distribution packages
- Includes code signing preparation

### 3. `quick_build.sh` - Fast Development Builder
- Quick setup and build for testing
- Minimal dependencies
- Good for development iterations

## Usage Instructions

### Prerequisites
```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or install conda/miniconda
# Download from: https://docs.conda.io/en/latest/miniconda.html
```

### Quick Start
```bash
# 1. Navigate to project directory
cd /path/to/skyscope-project

# 2. Run the final build script
chmod +x final_build.sh
./final_build.sh

# 3. Find your application in the 'distributions' folder
```

### Advanced Usage
```bash
# Build for all platforms
./build_cross_platform.sh --all

# Build for specific platform
./build_cross_platform.sh --platform macos
./build_cross_platform.sh --platform linux
./build_cross_platform.sh --platform windows

# Clean build directories
./build_cross_platform.sh --clean
```

## Manual Build Process

If you prefer to build manually:

### 1. Environment Setup
```bash
# Using uv (recommended)
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-build.txt

# Or using conda
conda env create -f environment.yml
conda activate skyscope
```

### 2. Verify Dependencies
```bash
python3 -c "
import streamlit, PIL, numpy, yaml, requests, cryptography, pyinstaller
print('✅ All dependencies available')
"
```

### 3. Create PyInstaller Spec
Use the provided `skyscope_optimized.spec` or create your own:

```python
# Key components for the spec file:
hidden_imports = [
    'streamlit',
    'streamlit.web.cli',
    'streamlit.runtime.scriptrunner.script_runner',
    'PIL', 'PIL.Image', 'PIL.ImageDraw',
    'numpy', 'yaml', 'requests', 'cryptography',
    # ... (see full list in spec files)
]

datas = [
    ('skyscope-logo.png', '.'),
    ('knowledge_base.md', '.'),
    # Streamlit static files
]

# macOS app bundle configuration
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='Skyscope Sentinel.app',
        icon='Skyscope.icns',
        bundle_identifier='com.skyscope.sentinel',
        # ... (see full configuration)
    )
```

### 4. Build Application
```bash
pyinstaller --clean --noconfirm skyscope_optimized.spec
```

### 5. Create Distribution Packages

#### macOS (DMG)
```bash
# Using create-dmg
create-dmg \
    --volname "Skyscope Sentinel" \
    --window-size 800 400 \
    --icon-size 100 \
    --app-drop-link 600 185 \
    "Skyscope_Sentinel.dmg" \
    "dist/Skyscope Sentinel.app"

# Or using hdiutil
hdiutil create -volname "Skyscope Sentinel" \
    -srcfolder "dist/Skyscope Sentinel.app" \
    -ov -format UDZO "Skyscope_Sentinel.dmg"
```

#### Linux (AppImage)
```bash
# Create AppDir structure
mkdir -p AppDir/usr/bin
cp -r dist/Skyscope_Sentinel/* AppDir/usr/bin/

# Download appimagetool
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage

# Create AppImage
./appimagetool-x86_64.AppImage AppDir Skyscope_Sentinel.AppImage
```

#### Windows (MSI/EXE)
```bash
# Using NSIS (if available)
makensis installer.nsi

# Or create ZIP archive
cd dist && zip -r ../Skyscope_Sentinel_Windows.zip Skyscope_Sentinel/
```

## Asset Requirements

### Logo Files
- **skyscope-logo.png**: Main application logo (provided)
- **Skyscope.icns**: macOS icon file (provided)
- **Skyscope.ico**: Windows icon file (auto-generated from PNG)

### Configuration Files
- **requirements-build.txt**: Core dependencies for building
- **environment.yml**: Conda environment specification
- **knowledge_base.md**: Application knowledge base

## Troubleshooting

### Common Issues and Solutions

#### 1. Missing Dependencies
```bash
# Install missing packages
uv pip install streamlit pillow numpy pyyaml requests cryptography pyinstaller

# Or with conda
conda install -c conda-forge streamlit pillow numpy pyyaml requests cryptography
pip install pyinstaller
```

#### 2. Icon/Logo Issues
```bash
# Verify files exist
ls -la skyscope-logo.png Skyscope.icns

# Create Windows icon from PNG
python3 -c "
from PIL import Image
img = Image.open('skyscope-logo.png')
img.save('Skyscope.ico', format='ICO', sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])
"
```

#### 3. PyInstaller Build Failures
```bash
# Clean build cache
rm -rf build dist __pycache__

# Rebuild with verbose output
pyinstaller --clean --noconfirm --log-level DEBUG skyscope_optimized.spec
```

#### 4. Missing Streamlit Static Files
```bash
# Find Streamlit installation
python3 -c "import streamlit; print(streamlit.__file__)"

# Verify static files exist
ls -la /path/to/streamlit/static/
```

## Performance Optimizations

### Build Size Reduction
- Exclude unnecessary modules in PyInstaller spec
- Use UPX compression (included in specs)
- Remove test files and documentation

### Runtime Performance
- Use `--onedir` instead of `--onefile` for faster startup
- Include only essential hidden imports
- Optimize Streamlit configuration

## Distribution

### Code Signing (macOS)
```bash
# Sign the application
codesign --force --deep --sign "Developer ID Application: Your Name" "dist/Skyscope Sentinel.app"

# Verify signature
codesign --verify --verbose "dist/Skyscope Sentinel.app"

# Notarize (requires Apple Developer account)
xcrun notarytool submit "Skyscope_Sentinel.dmg" --keychain-profile "notarytool-profile"
```

### Windows Code Signing
```bash
# Using signtool (requires certificate)
signtool sign /f certificate.p12 /p password /t http://timestamp.digicert.com "dist/Skyscope_Sentinel.exe"
```

## Testing

### Automated Testing
```bash
# Run system validation
python3 validate_system.py

# Test built application
./dist/Skyscope_Sentinel/Skyscope_Sentinel --help
```

### Manual Testing Checklist
- [ ] Application launches without errors
- [ ] UI loads correctly
- [ ] Core features work (agent management, configuration)
- [ ] File operations work (save/load configurations)
- [ ] Network requests work (API calls)
- [ ] Encryption/decryption works (secure storage)

## Deployment

### Distribution Channels
- **macOS**: Mac App Store, direct download, Homebrew
- **Windows**: Microsoft Store, direct download, Chocolatey
- **Linux**: Snap Store, AppImage, Flatpak, package repositories

### Update Mechanism
Consider implementing auto-update functionality using:
- **Sparkle** (macOS)
- **WinSparkle** (Windows)
- **AppImageUpdate** (Linux)

## Support

### Build System Maintenance
- Regular dependency updates
- Security patch monitoring
- Platform compatibility testing
- Performance optimization

### User Support
- Installation guides for each platform
- Troubleshooting documentation
- FAQ for common issues
- Community support channels

---

## Summary

The enhanced cross-platform build system provides:

✅ **Modern tooling** with uv and conda integration  
✅ **Comprehensive dependency management**  
✅ **Optimized PyInstaller configurations**  
✅ **Professional distribution packages**  
✅ **Cross-platform compatibility**  
✅ **Automated build processes**  
✅ **Error handling and recovery**  
✅ **Performance optimizations**  

The system is now ready for production deployment across all major platforms with professional-grade packaging and distribution capabilities.