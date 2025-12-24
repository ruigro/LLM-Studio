#include <windows.h>
#include <string>
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
    
    std::wstring scriptArgs;
    std::wstring logFile;
    int exitCode;
    
    if (needsSetup) {
        // First run: use system Python to run setup wizard
        std::wstring systemPython = L"python.exe";  // From PATH
        
        scriptArgs = L"first_run_setup.py";
        logFile = exeDir + L"\\logs\\setup.log";
        
        exitCode = LaunchPythonApp(exeDir, systemPython, scriptArgs, logFile);
        
        if (exitCode != 0) {
            // Setup failed - open log in Notepad
            MessageBoxW(NULL, 
                       L"First-time setup failed!\n\n"
                       L"The setup log will open in Notepad.\n"
                       L"Please review the errors and try again.\n\n"
                       L"Make sure Python 3.8+ is installed and in PATH.",
                       L"Setup Error", 
                       MB_OK | MB_ICONERROR);
            OpenLogInNotepad(logFile);
            return exitCode;
        }
        
        // Setup succeeded, now continue to launch app
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
                    L"Setup may have failed. Please check logs\\setup.log",
                    L"LLM Studio Launcher Error", 
                    MB_OK | MB_ICONERROR);
        return 1;
    }
    
    // Launch main application
    scriptArgs = L"-m desktop_app.main";
    logFile = exeDir + L"\\logs\\app.log";
    
    exitCode = LaunchPythonApp(exeDir, selectedPython, scriptArgs, logFile);
    
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
