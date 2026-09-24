; Tomo — the Windows installer (Inno Setup 6).
;
; Built by .github/workflows/build.yml after `cargo build --release`, with
; uv.exe downloaded next to this file (uv\uv.exe):
;
;   ISCC.exe /DAppVersion=0.1.0 packaging\windows\tomo.iss
;
; It installs for the current user only (no admin needed) into
; %LOCALAPPDATA%\Programs\Tomo, then optionally sets up the voice — a private
; Python with Piper, Vosk and Whisper, and their models (setup-speech.ps1) —
; and a model for Ollama (get-model.ps1). Everything runs on the computer.

#ifndef AppVersion
  #define AppVersion "0.1.0"
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
UninstallDisplayIcon={app}\tomo.exe
CloseApplications=yes

[Tasks]
Name: "speech"; Description: "Set up Tomo's voice and ""Hey Tomo"" (downloads Python and the speech models, about 1 GB, once)"
Name: "model"; Description: "Download a model for Ollama to think with (qwen2.5:7b, about 4.7 GB; needs Ollama)"
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked
Name: "autostart"; Description: "Start Tomo when I sign in"; Flags: unchecked

[Files]
Source: "..\..\target\release\tomo.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\scripts\*.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\..\scripts\requirements.txt"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\..\assets\characters\*.vrm"; DestDir: "{app}\assets\characters"; Flags: ignoreversion
; The settings: only on a first install, and kept when uninstalling.
Source: "..\..\.env.example"; DestDir: "{app}"; DestName: ".env"; Flags: onlyifdoesntexist uninsneveruninstall
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "uv\uv.exe"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "setup-speech.ps1"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "get-model.ps1"; DestDir: "{app}\tools"; Flags: ignoreversion

[Icons]
Name: "{group}\Tomo"; Filename: "{app}\tomo.exe"; WorkingDir: "{app}"
Name: "{group}\Tomo (compatibility)"; Filename: "{app}\tomo.exe"; Parameters: "--renderer gl"; WorkingDir: "{app}"; Comment: "If Tomo's background shows black, try this"
Name: "{group}\Set up Tomo's voice"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\setup-speech.ps1"""; WorkingDir: "{app}"
Name: "{group}\Download a model for Tomo"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\get-model.ps1"""; WorkingDir: "{app}"
Name: "{group}\Tomo settings"; Filename: "notepad.exe"; Parameters: """{app}\.env"""
Name: "{group}\Uninstall Tomo"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Tomo"; Filename: "{app}\tomo.exe"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{userstartup}\Tomo"; Filename: "{app}\tomo.exe"; WorkingDir: "{app}"; Tasks: autostart

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\setup-speech.ps1"""; WorkingDir: "{app}"; StatusMsg: "Setting up Tomo's voice (this downloads about 1 GB)..."; Flags: waituntilterminated; Tasks: speech
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\get-model.ps1"""; WorkingDir: "{app}"; StatusMsg: "Getting a model for Ollama..."; Flags: waituntilterminated; Tasks: model
Filename: "{app}\tomo.exe"; Description: "Start Tomo"; WorkingDir: "{app}"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\scripts\.venv"
Type: filesandordirs; Name: "{app}\scripts\__pycache__"
