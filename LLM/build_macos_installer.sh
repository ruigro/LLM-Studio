#!/bin/bash
# Build script for macOS - Creates standalone installer with auto-detection

set -e

echo "================================================"
echo "  Building LLM Fine-tuning Studio for macOS"
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
echo "Step 5: Creating macOS app bundle..."
APP_NAME="LLM_Studio.app"
APP_DIR="dist/$APP_NAME"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Create app bundle structure
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Copy executable
cp dist/LLM_Studio "$MACOS_DIR/LLM_Studio"

# Copy application files
cp gui.py "$RESOURCES_DIR/"
cp finetune.py "$RESOURCES_DIR/"
cp run_adapter.py "$RESOURCES_DIR/"
cp validate_prompts.py "$RESOURCES_DIR/"
cp system_detector.py "$RESOURCES_DIR/"
cp smart_installer.py "$RESOURCES_DIR/"
cp verify_installation.py "$RESOURCES_DIR/"
cp requirements.txt "$RESOURCES_DIR/"

# Create Info.plist
cat > "$CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>LLM_Studio</string>
    <key>CFBundleIdentifier</key>
    <string>com.llmstudio.app</string>
    <key>CFBundleName</key>
    <string>LLM Fine-tuning Studio</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
EOF

# Create .dmg
echo
echo "Step 6: Creating DMG installer..."
DMG_NAME="LLM_Studio.dmg"
DMG_TEMP="dist/dmg_temp"

rm -rf "$DMG_TEMP"
mkdir -p "$DMG_TEMP"
cp -R "$APP_DIR" "$DMG_TEMP/"

# Create Applications symlink
ln -s /Applications "$DMG_TEMP/Applications"

# Create DMG
hdiutil create -volname "LLM Fine-tuning Studio" -srcfolder "$DMG_TEMP" -ov -format UDZO "dist/$DMG_NAME"

echo
echo "================================================"
echo "  Build Complete!"
echo "================================================"
echo
echo "App Bundle: $APP_DIR"
echo "DMG Installer: dist/$DMG_NAME"
echo

