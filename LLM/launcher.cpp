#include <windows.h>
#include <string>
#include <vector>
#include <fstream>
#include <shlwapi.h>
#include <winreg.h>
#include <urlmon.h>

#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "urlmon.lib")

// Helper to convert std::string to std::wstring
std::wstring StringToWString(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

// Helper to check if file exists
bool FileExists(const std::wstring& path) {
    DWORD attrib = GetFileAttributesW(path.c_str());
    return (attrib != INVALID_FILE_ATTRIBUTES && !(attrib & FILE_ATTRIBUTE_DIRECTORY));
}

// Helper to create logs directory
void EnsureLogsDirectory(const std::wstring& exeDir) {
    std::wstring logsDir = exeDir + L"\\logs";
    CreateDirectoryW(logsDir.c_str(), NULL);
}

// Helper to open log file in Notepad
void OpenLogInNotepad(const std::wstring& logPath) {
    ShellExecuteW(NULL, L"open", L"notepad.exe", logPath.c_str(), NULL, SW_SHOW);
}

// Helper to find Python in PATH or registry
std::wstring FindSystemPython() {
    // First, try to find python.exe in PATH
    wchar_t pythonPath[MAX_PATH];
    DWORD result = SearchPathW(NULL, L"python.exe", NULL, MAX_PATH, pythonPath, NULL);
    if (result > 0 && result < MAX_PATH) {
        return std::wstring(pythonPath);
    }
    
    // Try pythonw.exe in PATH
    result = SearchPathW(NULL, L"pythonw.exe", NULL, MAX_PATH, pythonPath, NULL);
    if (result > 0 && result < MAX_PATH) {
        return std::wstring(pythonPath);
    }
    
    // Check registry for Python installation
    HKEY hKey;
    // Check Python 3.x in registry (HKEY_LOCAL_MACHINE\SOFTWARE\Python\PythonCore\<version>\InstallPath)
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Python\\PythonCore", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        DWORD index = 0;
        wchar_t versionKey[256];
        DWORD versionKeySize = sizeof(versionKey) / sizeof(wchar_t);
        
        while (RegEnumKeyExW(hKey, index, versionKey, &versionKeySize, NULL, NULL, NULL, NULL) == ERROR_SUCCESS) {
            std::wstring installPathKey = std::wstring(L"SOFTWARE\\Python\\PythonCore\\") + versionKey + L"\\InstallPath";
            HKEY hInstallKey;
            if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, installPathKey.c_str(), 0, KEY_READ, &hInstallKey) == ERROR_SUCCESS) {
                wchar_t installPath[MAX_PATH];
                DWORD pathSize = MAX_PATH * sizeof(wchar_t);
                DWORD type = REG_SZ;
                if (RegQueryValueExW(hInstallKey, L"ExecutablePath", NULL, &type, (LPBYTE)installPath, &pathSize) == ERROR_SUCCESS) {
                    if (FileExists(installPath)) {
                        RegCloseKey(hInstallKey);
                        RegCloseKey(hKey);
                        return std::wstring(installPath);
                    }
                }
                RegCloseKey(hInstallKey);
            }
            index++;
            versionKeySize = sizeof(versionKey) / sizeof(wchar_t);
        }
        RegCloseKey(hKey);
    }
    
    return L"";  // Not found
}

// Helper to download Python installer
bool DownloadPythonInstaller(const std::wstring& downloadPath) {
    // Download Python 3.10.x installer (64-bit) from python.org
    // Using embeddable package URL for smaller download, or full installer
    const wchar_t* pythonUrl = L"https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe";
    
    // Use URLDownloadToFileW to download
    HRESULT hr = URLDownloadToFileW(NULL, pythonUrl, downloadPath.c_str(), 0, NULL);
    return (hr == S_OK);
}

// Helper to install Python silently
bool InstallPython(const std::wstring& installerPath) {
    // Run Python installer with /quiet and /PrependPath flags
    // /quiet = silent installation
    // /PrependPath = add Python to PATH
    std::wstring cmdLine = L"\"" + installerPath + L"\" /quiet PrependPath=1 InstallAllUsers=1";
    
    SHELLEXECUTEINFOW sei = {0};
    sei.cbSize = sizeof(sei);
    sei.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_FLAG_NO_UI;
    sei.lpVerb = L"runas";  // Run as administrator (may be needed)
    sei.lpFile = installerPath.c_str();
    sei.lpParameters = L"/quiet PrependPath=1 InstallAllUsers=1";
    sei.nShow = SW_HIDE;
    
    if (!ShellExecuteExW(&sei)) {
        // Try without runas if that fails
        sei.lpVerb = L"open";
        if (!ShellExecuteExW(&sei)) {
            return false;
        }
    }
    
    // Wait for installation to complete (max 5 minutes)
    if (sei.hProcess) {
        WaitForSingleObject(sei.hProcess, 300000);  // 5 minutes timeout
        DWORD exitCode = 0;
        GetExitCodeProcess(sei.hProcess, &exitCode);
        CloseHandle(sei.hProcess);
        return (exitCode == 0);
    }
    
    return false;
}

// Main launcher function
int LaunchPythonApp(const std::wstring& exeDir, const std::wstring& pythonExe, 
                    const std::wstring& scriptArgs, const std::wstring& logFile) {
    // Ensure logs directory exists before creating log file
    EnsureLogsDirectory(exeDir);
    
    // Build command line: "pythonw.exe" <args>
    std::wstring cmdLine = L"\"" + pythonExe + L"\" " + scriptArgs;
    
    // Setup startup info with redirected output
    STARTUPINFOW si = {0};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    
    // Create log file for output
    SECURITY_ATTRIBUTES sa = {0};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    
    HANDLE hLogFile = CreateFileW(
        logFile.c_str(),
        GENERIC_WRITE,
        FILE_SHARE_READ,
        &sa,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    
    if (hLogFile != INVALID_HANDLE_VALUE) {
        si.hStdOutput = hLogFile;
        si.hStdError = hLogFile;
    } else {
        // Log file creation failed - write error to a fallback location
        std::wstring fallbackLog = exeDir + L"\\launcher_error.log";
        HANDLE hFallback = CreateFileW(
            fallbackLog.c_str(),
            GENERIC_WRITE,
            FILE_SHARE_READ,
            &sa,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );
        if (hFallback != INVALID_HANDLE_VALUE) {
            std::wstring errorMsg = L"Failed to create log file: " + logFile + L"\n";
            DWORD written = 0;
            WriteFile(hFallback, errorMsg.c_str(), errorMsg.length() * sizeof(wchar_t), &written, NULL);
            CloseHandle(hFallback);
        }
    }
    
    // Create the process
    PROCESS_INFORMATION pi = {0};
    BOOL success = CreateProcessW(
        pythonExe.c_str(),          // Application name
        &cmdLine[0],                // Command line (must be writable)
        NULL,                       // Process security attributes
        NULL,                       // Thread security attributes
        TRUE,                       // Inherit handles (for log redirection)
        CREATE_NO_WINDOW,           // Creation flags - no console window
        NULL,                       // Environment
        exeDir.c_str(),             // Working directory
        &si,                        // Startup info
        &pi                         // Process info
    );
    
    if (hLogFile != INVALID_HANDLE_VALUE) {
        CloseHandle(hLogFile);
    }
    
    if (!success) {
        return -1;
    }
    
    // Wait for process to complete
    WaitForSingleObject(pi.hProcess, INFINITE);
    
    // Get exit code
    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    
    return exitCode;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, 
                   LPSTR lpCmdLine, int nCmdShow) {
    // Get the directory where this .exe is located
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    
    // Extract directory path
    std::wstring exeDir(exePath);
    size_t lastSlash = exeDir.find_last_of(L"\\/");
    if (lastSlash != std::wstring::npos) {
        exeDir = exeDir.substr(0, lastSlash);
    }
    
    // Ensure logs directory exists
    EnsureLogsDirectory(exeDir);
    
    // Step 1: Check if system Python exists
    std::wstring systemPython = FindSystemPython();
    
    if (systemPython.empty()) {
        // Python not found - need to install it
        int result = MessageBoxW(NULL,
            L"Python is not installed on this system.\n\n"
            L"This application requires Python 3.10+ to run.\n\n"
            L"Would you like to download and install Python now?",
            L"Python Required",
            MB_YESNO | MB_ICONQUESTION);
        
        if (result != IDYES) {
            MessageBoxW(NULL,
                L"Python installation is required.\n\n"
                L"Please install Python 3.10+ from https://www.python.org/downloads/\n"
                L"Make sure to check 'Add Python to PATH' during installation.",
                L"Python Required",
                MB_OK | MB_ICONINFORMATION);
            return 1;
        }
        
        // Download Python installer
        std::wstring tempDir = exeDir + L"\\temp";
        CreateDirectoryW(tempDir.c_str(), NULL);
        std::wstring installerPath = tempDir + L"\\python-installer.exe";
        
        MessageBoxW(NULL, L"Downloading Python installer...\n\nThis may take a few minutes.", L"Downloading", MB_OK | MB_ICONINFORMATION);
        
        if (!DownloadPythonInstaller(installerPath)) {
            MessageBoxW(NULL,
                L"Failed to download Python installer.\n\n"
                L"Please download and install Python 3.10+ manually from:\n"
                L"https://www.python.org/downloads/",
                L"Download Failed",
                MB_OK | MB_ICONERROR);
            return 1;
        }
        
        // Install Python
        MessageBoxW(NULL, L"Installing Python...\n\nThis may take a few minutes.\n\nPlease wait...", L"Installing", MB_OK | MB_ICONINFORMATION);
        
        if (!InstallPython(installerPath)) {
            MessageBoxW(NULL,
                L"Python installation failed or was cancelled.\n\n"
                L"Please install Python 3.10+ manually from:\n"
                L"https://www.python.org/downloads/",
                L"Installation Failed",
                MB_OK | MB_ICONERROR);
            return 1;
        }
        
        // Clean up installer
        DeleteFileW(installerPath.c_str());
        
        // Refresh PATH and try to find Python again
        // Note: PATH refresh may require restart, but we'll try anyway
        systemPython = FindSystemPython();
        if (systemPython.empty()) {
            MessageBoxW(NULL,
                L"Python was installed, but the launcher cannot find it.\n\n"
                L"Please restart your computer or manually add Python to PATH,\n"
                L"then run this launcher again.",
                L"Python Installed",
                MB_OK | MB_ICONINFORMATION);
            return 1;
        }
    }
    
    // Step 2: Check if venv exists, if not, create it or launch installer GUI
    std::wstring pythonwExe = exeDir + L"\\.venv\\Scripts\\pythonw.exe";
    std::wstring pythonExe = exeDir + L"\\.venv\\Scripts\\python.exe";
    std::wstring venvPython = systemPython;  // Default to system Python
    
    if (FileExists(pythonwExe)) {
        venvPython = pythonwExe;
    } else if (FileExists(pythonExe)) {
        venvPython = pythonExe;
    } else {
        // Venv doesn't exist - launch installer GUI via bootstrap
        std::wstring runInstallerBat = exeDir + L"\\run_installer.bat";
        if (FileExists(runInstallerBat)) {
            // Use run_installer.bat which ensures bootstrap is used
            ShellExecuteW(NULL, L"open", runInstallerBat.c_str(), NULL, exeDir.c_str(), SW_SHOW);
            return 0;
        } else {
            // Fallback: try installer_gui.py directly (it has bootstrap guard)
            std::wstring installerGui = exeDir + L"\\installer_gui.py";
            if (FileExists(installerGui)) {
                ShellExecuteW(NULL, L"open", systemPython.c_str(), installerGui.c_str(), exeDir.c_str(), SW_SHOW);
                return 0;
            } else {
                MessageBoxW(NULL,
                    L"Virtual environment not found and installer GUI not available.\n\n"
                    L"Please run the setup manually.",
                    L"Setup Required",
                    MB_OK | MB_ICONERROR);
                return 1;
            }
        }
    }
    
    // Ensure logs directory exists
    EnsureLogsDirectory(exeDir);
    
    // Step 3: Health check - Test if PySide6 can import
    // NOTE: All PySide6 packages MUST be at version 6.8.1 (PySide6, Essentials, Addons, shiboken6)
    // Version mismatches cause "procedure could not be found" DLL errors
    std::wstring healthCheckCmd = L"-c \"import PySide6.QtCore; print('OK')\"";
    std::wstring healthCheckLog = exeDir + L"\\logs\\health_check.log";
    
    int healthCheckResult = LaunchPythonApp(exeDir, venvPython, healthCheckCmd, healthCheckLog);
    
    if (healthCheckResult != 0) {
        // PySide6 is broken - launch installer GUI via bootstrap
        std::wstring runInstallerBat = exeDir + L"\\run_installer.bat";
        if (FileExists(runInstallerBat)) {
            ShellExecuteW(NULL, L"open", runInstallerBat.c_str(), NULL, exeDir.c_str(), SW_SHOW);
            return 0;
        } else {
            std::wstring installerGui = exeDir + L"\\installer_gui.py";
            if (FileExists(installerGui)) {
                ShellExecuteW(NULL, L"open", systemPython.c_str(), installerGui.c_str(), exeDir.c_str(), SW_SHOW);
                return 0;
            } else {
                MessageBoxW(NULL,
                    L"Critical dependencies are broken and installer GUI is not available.\n\n"
                    L"Please run the setup manually or check the installation.",
                    L"Setup Required",
                    MB_OK | MB_ICONERROR);
                return 1;
            }
        }
    }
    
    // Step 4: Dependency health check - Verify critical packages are installed
    std::wstring dependencyCheckScript = exeDir + L"\\check_dependencies.py";
    std::wstring dependencyCheckLog = exeDir + L"\\logs\\dependency_check.log";
    
    if (FileExists(dependencyCheckScript)) {
        // Run check_dependencies.py as a script
        std::wstring dependencyCheckCmd = L"\"" + dependencyCheckScript + L"\"";
        int dependencyCheckResult = LaunchPythonApp(exeDir, venvPython, dependencyCheckCmd, dependencyCheckLog);
        
        if (dependencyCheckResult != 0) {
            // Dependencies are missing or wrong - launch installer GUI via bootstrap
            std::wstring runInstallerBat = exeDir + L"\\run_installer.bat";
            if (FileExists(runInstallerBat)) {
                // Use run_installer.bat which ensures bootstrap is used
                ShellExecuteW(NULL, L"open", runInstallerBat.c_str(), NULL, exeDir.c_str(), SW_SHOW);
                return 0;  // Exit - installer GUI will handle repair
            } else {
                // Fallback: try installer_gui.py directly (it has bootstrap guard)
                std::wstring installerGui = exeDir + L"\\installer_gui.py";
                if (FileExists(installerGui)) {
                    ShellExecuteW(NULL, L"open", systemPython.c_str(), installerGui.c_str(), exeDir.c_str(), SW_SHOW);
                    return 0;
                } else {
                    MessageBoxW(NULL,
                        L"Critical dependencies are missing or have wrong versions.\n\n"
                        L"Installer GUI is not available.\n"
                        L"Please run the setup manually or check the installation.",
                        L"Setup Required",
                        MB_OK | MB_ICONERROR);
                    return 1;
                }
            }
        }
    } else {
        // Dependency check script missing - log warning but continue
        // (This shouldn't happen in normal operation, but don't block launch)
        std::wstring warnLog = exeDir + L"\\logs\\launcher_warning.log";
        std::ofstream warnFile(warnLog.c_str(), std::ios::app);
        if (warnFile.is_open()) {
            warnFile << "WARNING: check_dependencies.py not found, skipping dependency check\n";
            warnFile.close();
        }
    }
    
    // Step 5: Launch main application (PySide6 and dependencies are working)
    std::wstring scriptArgs = L"-m desktop_app.main";
    std::wstring logFile = exeDir + L"\\logs\\app.log";
    
    // Ensure .setup_complete marker exists so setup wizard is skipped
    std::wstring setupMarker = exeDir + L"\\.setup_complete";
    if (!FileExists(setupMarker)) {
        std::ofstream marker(setupMarker.c_str());
        if (marker.is_open()) {
            marker << "ready";
            marker.close();
        }
    }

    int exitCode = LaunchPythonApp(exeDir, venvPython, scriptArgs, logFile);
    if (exitCode != 0) {
        // App failed - open log in Notepad
        MessageBoxW(NULL, 
                   L"Application failed to start!\n\n"
                   L"The error log will open in Notepad.\n"
                   L"Please review the errors.",
                   L"Application Error", 
                   MB_OK | MB_ICONERROR);
        OpenLogInNotepad(logFile);
        return exitCode;
    }

    return exitCode;
}
