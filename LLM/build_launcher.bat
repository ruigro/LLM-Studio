@echo off
echo =============================================
echo   Building LLM Studio Launcher
echo =============================================
echo.

REM Check if MinGW is installed
where gcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: GCC not found!
    echo.
    echo Please install MinGW-w64:
    echo 1. Download from: https://winlibs.com/
    echo 2. Extract to C:\mingw64
    echo 3. Add C:\mingw64\bin to PATH
    echo 4. Restart terminal and run this script again
    echo.
    pause
    exit /b 1
)

echo [1/3] Compiling resource file...
windres launcher.rc -o launcher_res.o
if errorlevel 1 (
    echo ERROR: Failed to compile resource file
    echo Make sure rocket.ico exists in this directory
    pause
    exit /b 1
)
echo SUCCESS: Resource compiled

echo.
echo [2/3] Compiling C++ source with static linking...
g++ -O2 -s -mwindows launcher.cpp launcher_res.o -o launcher.exe -static -static-libgcc -static-libstdc++ -lshlwapi -lurlmon
if errorlevel 1 (
    echo ERROR: Failed to compile launcher
    pause
    del launcher_res.o
    exit /b 1
)
echo SUCCESS: Launcher compiled (fully static, no DLL dependencies)

echo.
echo [3/3] Cleaning up temporary files...
del launcher_res.o
echo SUCCESS: Cleanup complete

echo.
echo =============================================
echo   launcher.exe created successfully!
echo =============================================
echo.

REM Show file size
for %%A in (launcher.exe) do echo File size: %%~zA bytes (%%~zAKB)

echo.
echo You can now:
echo 1. Test it: Double-click launcher.exe
echo 2. Commit it: git add launcher.exe
echo.
pause

