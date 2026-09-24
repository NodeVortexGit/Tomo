//! What differs between the systems Tomo runs on — Linux and Windows (and
//! mostly macOS): which shell runs commands, finding a program, where a
//! Python virtual environment keeps its interpreter, and keeping helper
//! processes from flashing a console window on Windows.

use std::path::{Path, PathBuf};

use tokio::process::Command;

/// The system's name, for the model and the logs.
pub fn os_name() -> &'static str {
    if cfg!(windows) {
        "Windows"
    } else if cfg!(target_os = "macos") {
        "macOS"
    } else {
        "Linux"
    }
}

/// The shell `execute_command` lines run in.
pub fn shell_name() -> &'static str {
    if cfg!(windows) {
        "PowerShell"
    } else {
        "bash"
    }
}

/// A command that runs `line` in the system's shell: PowerShell on Windows,
/// else a bash login shell (so the user's own PATH applies).
pub fn shell(line: &str) -> Command {
    let mut command = if cfg!(windows) {
        let mut command = Command::new("powershell.exe");
        command.args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", line]);
        command
    } else {
        let mut command = Command::new("bash");
        command.arg("-lc").arg(line);
        command
    };
    quiet(&mut command);
    command
}

/// Keep a helper process from opening a console window (Windows); nothing
/// to do elsewhere.
pub fn quiet(command: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

/// Where `program` is, if it's on the PATH (on Windows, trying the
/// executable extensions too).
pub fn which(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let mut extensions = vec![String::new()];
    if cfg!(windows) {
        let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into());
        extensions.extend(pathext.split(';').filter(|e| !e.is_empty()).map(str::to_lowercase));
    }
    std::env::split_paths(&path).find_map(|dir| {
        extensions
            .iter()
            .map(|ext| dir.join(format!("{program}{ext}")))
            .find(|candidate| candidate.is_file())
    })
}

/// The Python interpreter of the virtual environment at `venv`.
pub fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python3")
    }
}

/// The system's Python, when there's no virtual environment.
pub fn system_python() -> &'static str {
    if cfg!(windows) {
        "python"
    } else {
        "python3"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_finds_programs_on_the_path() {
        let found = which(if cfg!(windows) { "cmd" } else { "sh" });
        assert!(found.is_some_and(|p| p.is_file()));
        assert!(which("surely-no-such-program-exists").is_none());
    }

    #[tokio::test]
    async fn commands_run_in_the_systems_shell() {
        let output = shell("echo tomo").output().await.unwrap();
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "tomo");
    }
}
