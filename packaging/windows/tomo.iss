; Tomo — the Windows installer (Inno Setup 6).
;
; Built by .github/workflows/build.yml, with uv.exe downloaded next to this
; file (uv\uv.exe):
;
;   ISCC.exe /DAppVersion=0.2.0 packaging\windows\tomo.iss
;
; It installs for the current user only (no admin needed) into
; %LOCALAPPDATA%\Programs\Tomo: Tomo's Python code, the helper scripts and the
; characters. setup.ps1 then makes Tomo's own Python environment (.venv, with
; uv) and, if chosen, downloads the speech and camera models; get-model.ps1
; gets a model for Ollama. Everything runs on the computer.

#ifndef AppVersion
  #define AppVersion "0.2.0"
#endif

[Setup]
AppId={{B17E68AD-81CF-48DC-B460-84DCF1752092}
AppName=Tomo
AppVersion={#AppVersion}
AppVerName=Tomo {#AppVersion}
AppPublisher=NodeVortex
AppPublisherURL=https://github.com/NodeVortexGit/Tomo
AppSupportURL=https://github.com/NodeVortexGit/Tomo/issues
DefaultDirName={localappdata}\Programs\Tomo
DefaultGroupName=Tomo
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\..\LICENSE
OutputDir=Output
OutputBaseFilename=Tomo-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Tasks]
Name: "models"; Description: "Set up Tomo's voice, ""Hey Tomo"" and the health programme's camera (downloads the models, about 1 GB, once)"
Name: "model"; Description: "Download the model Tomo thinks with, for Ollama (qwen3.5:9b, about 6.6 GB; needs Ollama)"
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked
Name: "autostart"; Description: "Start Tomo when I sign in"; Flags: unchecked

[Files]
Source: "..\..\tomo\*.py"; DestDir: "{app}\tomo"; Flags: ignoreversion
Source: "..\..\tomo\body\*.py"; DestDir: "{app}\tomo\body"; Flags: ignoreversion
Source: "..\..\tomo\body\*.glsl"; DestDir: "{app}\tomo\body"; Flags: ignoreversion
Source: "..\..\scripts\*.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\..\assets\characters\*.vrm"; DestDir: "{app}\assets\characters"; Flags: ignoreversion
Source: "..\..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
; The settings: only on a first install, and kept when uninstalling.
Source: "..\..\.env.example"; DestDir: "{app}"; DestName: ".env"; Flags: onlyifdoesntexist uninsneveruninstall
Source: "uv\uv.exe"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "setup.ps1"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "get-model.ps1"; DestDir: "{app}\tools"; Flags: ignoreversion

[Icons]
Name: "{group}\Tomo"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m tomo"; WorkingDir: "{app}"
Name: "{group}\Set up Tomo's voice and camera"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\setup.ps1"" -Models"; WorkingDir: "{app}"
Name: "{group}\Download a model for Tomo"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\get-model.ps1"""; WorkingDir: "{app}"
Name: "{group}\Tomo settings"; Filename: "notepad.exe"; Parameters: """{app}\.env"""
Name: "{group}\Uninstall Tomo"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Tomo"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m tomo"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{userstartup}\Tomo"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m tomo"; WorkingDir: "{app}"; Tasks: autostart

[Run]
; Tomo's Python environment is always needed; the models only if chosen.
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\setup.ps1"""; WorkingDir: "{app}"; StatusMsg: "Setting up Tomo's Python (about 700 MB, once)..."; Flags: waituntilterminated; Tasks: not models
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\setup.ps1"" -Models"; WorkingDir: "{app}"; StatusMsg: "Setting up Tomo's Python, voice and camera (about 1.7 GB, once)..."; Flags: waituntilterminated; Tasks: models
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\get-model.ps1"""; WorkingDir: "{app}"; StatusMsg: "Getting a model for Ollama..."; Flags: waituntilterminated; Tasks: model
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m tomo"; Description: "Start Tomo"; WorkingDir: "{app}"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\tomo\__pycache__"
Type: filesandordirs; Name: "{app}\tomo\body\__pycache__"
Type: filesandordirs; Name: "{app}\scripts\__pycache__"
