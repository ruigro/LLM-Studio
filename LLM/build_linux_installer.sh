#!/bin/bash
# Build script for Linux - Creates standalone installer with auto-detection

set -e

echo "================================================"
echo "  Building LLM Fine-tuning Studio for Linux"
echo "================================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found!"
    echo "Please install Python 3.8+"
    exit 1
fi

echo "Step 1: Installing build dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements_build.txt

echo
echo "Step 2: Installing application dependencies..."
python3 -m pip install -r requirements.txt

echo
echo "Step 3: Running system detection test..."
python3 system_detector.py || echo "WARNING: System detection test failed"

echo
echo "Step 4: Building executable with PyInstaller..."
pyinstaller --clean llm_studio.spec

echo
echo "Step 5: Creating AppImage (portable format)..."
APPIMAGE_DIR="dist/LLM_Studio.AppDir"

# Create AppDir structure
mkdir -p "$APPIMAGE_DIR/usr/bin"
mkdir -p "$APPIMAGE_DIR/usr/lib"
mkdir -p "$APPIMAGE_DIR/usr/share/applications"
mkdir -p "$APPIMAGE_DIR/usr/share/icons"

# Copy executable
cp dist/LLM_Studio "$APPIMAGE_DIR/usr/bin/"

# Copy application files
cp gui.py "$APPIMAGE_DIR/usr/bin/"
cp finetune.py "$APPIMAGE_DIR/usr/bin/"
cp run_adapter.py "$APPIMAGE_DIR/usr/bin/"
cp validate_prompts.py "$APPIMAGE_DIR/usr/bin/"
cp system_detector.py "$APPIMAGE_DIR/usr/bin/"
cp smart_installer.py "$APPIMAGE_DIR/usr/bin/"
cp verify_installation.py "$APPIMAGE_DIR/usr/bin/"
cp requirements.txt "$APPIMAGE_DIR/usr/bin/"

# Create desktop entry
cat > "$APPIMAGE_DIR/usr/share/applications/llm-studio.desktop" << EOF
[Desktop Entry]
Name=LLM Fine-tuning Studio
Comment=Fine-tune Large Language Models
Exec=LLM_Studio
Icon=llm-studio
Type=Application
Categories=Development;Science;
EOF

# Create AppRun script
cat > "$APPIMAGE_DIR/AppRun" << 'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
exec "${HERE}/usr/bin/LLM_Studio" "$@"
EOF
chmod +x "$APPIMAGE_DIR/AppRun"

echo
echo "Note: To create final AppImage, install appimagetool:"
echo "  wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
echo "  chmod +x appimagetool-x86_64.AppImage"
echo "  ./appimagetool-x86_64.AppImage $APPIMAGE_DIR"
echo

echo
echo "================================================"
echo "  Build Complete!"
echo "================================================"
echo
echo "Executable: dist/LLM_Studio"
echo "AppDir: $APPIMAGE_DIR"
echo

