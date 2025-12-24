#include <windows.h>
#include <string>
#include <vector>
#include <shlwapi.h>

#pragma comment(lib, "shlwapi.lib")

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

// Main launcher function
int LaunchPythonApp(const std::wstring& exeDir, const std::wstring& pythonExe, 
                    const std::wstring& scriptArgs, const std::wstring& logFile) {
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
    }
    
    // Create the process
    PROCESS_INFORMATION pi = {0};
    BOOL success = CreateProcessW(
        pythonExe.c_str(),          // Application name
        &cmdLine[0],                // Command line (must be writable)
        NULL,                       // Process security attributes
        NULL,                       // Thread security attributes
        TRUE,                       // Inherit handles (for log redirection)
        0,                          // Creation flags (0 = normal, no console)
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
    
    // Check if first-time setup is needed (check BEFORE checking venv)
    std::wstring setupCompleteMarker = exeDir + L"\\.setup_complete";
    bool needsSetup = !FileExists(setupCompleteMarker);
    
    if (needsSetup) {
        // First run: launch bootstrap_setup.py (uses tkinter, no deps)
        std::wstring bootstrapScript = exeDir + L"\\bootstrap_setup.py";
        
        // Find system Python
        std::wstring systemPython = L"python.exe";
        
        // Try to find Python in common locations
        std::vector<std::wstring> pythonPaths = {
            L"python.exe",  // From PATH
            exeDir + L"\\python.exe"
        };
        
        // Check AppData
        wchar_t appdata[MAX_PATH];
        if (GetEnvironmentVariableW(L"LOCALAPPDATA", appdata, MAX_PATH) > 0) {
            std::wstring appdataPath(appdata);
            pythonPaths.push_back(appdataPath + L"\\Programs\\Python\\Python312\\python.exe");
            pythonPaths.push_back(appdataPath + L"\\Programs\\Python\\Python311\\python.exe");
            pythonPaths.push_back(appdataPath + L"\\Programs\\Python\\Python310\\python.exe");
        }
        
        // Check standard locations
        pythonPaths.push_back(L"C:\\Python312\\python.exe");
        pythonPaths.push_back(L"C:\\Python311\\python.exe");
        pythonPaths.push_back(L"C:\\Python310\\python.exe");
        
        std::wstring foundPython;
        for (const auto& path : pythonPaths) {
            if (path.find(L":") != std::wstring::npos) {
                // Absolute path
                if (FileExists(path)) {
                    foundPython = path;
                    break;
                }
            } else {
                // Relative or PATH
                foundPython = path;
                break;
            }
        }
        
        if (foundPython.empty()) {
            MessageBoxW(NULL,
                       L"Python not found!\n\n"
                       L"Please install Python 3.8+ from:\n"
                       L"https://www.python.org/downloads/\n\n"
                       L"Make sure to check 'Add Python to PATH' during installation.",
                       L"Python Required",
                       MB_OK | MB_ICONERROR);
            return 1;
        }
        
        // Launch bootstrap
        std::wstring args = L"\"" + foundPython + L"\" \"" + bootstrapScript + L"\"";
        
        STARTUPINFOW si = {0};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi = {0};
        
        BOOL success = CreateProcessW(
            NULL,
            &args[0],
            NULL, NULL, FALSE, 0, NULL,
            exeDir.c_str(),
            &si, &pi
        );
        
        if (!success) {
            MessageBoxW(NULL,
                       L"Failed to launch bootstrap setup!\n\n"
                       L"Please run LAUNCHER.bat manually.",
                       L"Setup Error",
                       MB_OK | MB_ICONERROR);
            return 1;
        }
        
        // Wait for bootstrap to complete
        WaitForSingleObject(pi.hProcess, INFINITE);
        
        DWORD exitCode = 0;
        GetExitCodeProcess(pi.hProcess, &exitCode);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        
        if (exitCode != 0) {
            return exitCode;
        }
        
        // Bootstrap launches main setup, so just exit here
        return 0;
    }
    
    // Check which Python interpreter to use (venv should exist now)
    std::wstring pythonwExe = exeDir + L"\\.venv\\Scripts\\pythonw.exe";
    std::wstring pythonExe = exeDir + L"\\.venv\\Scripts\\python.exe";
    
    std::wstring selectedPython;
    if (FileExists(pythonwExe)) {
        selectedPython = pythonwExe;  // Prefer pythonw.exe (no console)
    } else if (FileExists(pythonExe)) {
        selectedPython = pythonExe;   // Fallback to python.exe
    } else {
        MessageBoxW(NULL, 
                    L"Python virtual environment not found!\n\n"
                    L"Setup may have failed. Please run LAUNCHER.bat manually.",
                    L"LLM Studio Launcher Error", 
                    MB_OK | MB_ICONERROR);
        return 1;
    }
    
    // Launch main application
    std::wstring scriptArgs = L"-m desktop_app.main";
    std::wstring logFile = exeDir + L"\\logs\\app.log";
    
    int exitCode = LaunchPythonApp(exeDir, selectedPython, scriptArgs, logFile);
    
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
    
    return 0;
}
