#include <windows.h>
#include <string>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, 
                   LPSTR lpCmdLine, int nCmdShow) {
    // Get the directory where this .exe is located
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    
    // Extract directory path
    std::string exeDir(exePath);
    size_t lastSlash = exeDir.find_last_of("\\/");
    if (lastSlash != std::string::npos) {
        exeDir = exeDir.substr(0, lastSlash);
    }
    
    // Build path to LAUNCHER.bat
    std::string batPath = exeDir + "\\LAUNCHER.bat";
    
    // Launch the batch file
    SHELLEXECUTEINFOA sei = {0};
    sei.cbSize = sizeof(sei);
    sei.fMask = SEE_MASK_NOCLOSEPROCESS;
    sei.lpVerb = "open";
    sei.lpFile = batPath.c_str();
    sei.lpDirectory = exeDir.c_str();
    sei.nShow = SW_SHOW;
    
    if (!ShellExecuteExA(&sei)) {
        MessageBoxA(NULL, 
                    "Failed to launch LAUNCHER.bat\n"
                    "Make sure it exists in the same directory.",
                    "LLM Studio Launcher Error", 
                    MB_OK | MB_ICONERROR);
        return 1;
    }
    
    return 0;
}

