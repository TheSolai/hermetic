//! A real shell in a pseudo-terminal.
//!
//! The terminal is a zsh running in a PTY; its output goes through a vt100
//! grid so colours, cursor movement and line editing survive. The user types
//! into it like any other terminal.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{Context, Result};

/// A running shell.
pub struct Shell {
    master: Arc<Mutex<Box<dyn portable_pty::MasterPty + Send>>>,
    /// Bytes waiting to be written to the shell.
    ///
    /// The write itself happens on a thread of its own. Writing to a PTY
    /// whose reader is not reading blocks, and the terminal's event loop
    /// would stop: the panel would freeze and the user could not even
    /// press Ctrl-C.
    input: mpsc::Sender<Vec<u8>>,
    child: Arc<Mutex<Box<dyn portable_pty::Child + Send + Sync>>>,
    exited: Arc<AtomicBool>,
    /// Bytes the shell wrote.
    pub output: Receiver<Vec<u8>>,
}

impl Shell {
    /// Start a login shell.
    pub fn spawn(
        shell: &str,
        cwd: &Path,
        rows: u16,
        cols: u16,
        env: &[(String, String)],
        zdotdir: Option<&Path>,
    ) -> Result<Shell> {
        let pty_system = portable_pty::native_pty_system();
        let pair = pty_system
            .openpty(portable_pty::PtySize {
                rows: rows.max(8),
                cols: cols.max(20),
                pixel_width: 0,
                pixel_height: 0,
            })
            .context("open PTY")?;

        let mut cmd = portable_pty::CommandBuilder::new(shell);
        if shell.ends_with("zsh") || shell.ends_with("bash") {
            cmd.arg("-l");
        }
        cmd.cwd(cwd);
        cmd.env("TERM", "xterm-256color");
        cmd.env("COLORTERM", "truecolor");
        cmd.env("LANG", "en_US.UTF-8");
        for (key, value) in env {
            cmd.env(key, value);
        }
        if let Some(dir) = zdotdir {
            // A login shell reads $ZDOTDIR/.zshrc; ours sources the user's
            // own files first and adds the interception on top.
            cmd.env("ZDOTDIR", dir);
        }

        let child = pair.slave.spawn_command(cmd).context("spawn shell")?;
        drop(pair.slave);

        let mut reader = pair.master.try_clone_reader().context("clone reader")?;
        let writer = pair.master.take_writer().context("take writer")?;
        let master = pair.master;

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut buf = [0u8; 8192];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if tx.send(buf[..n].to_vec()).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        // The writing half, on a thread of its own.
        let (input_tx, input_rx) = mpsc::channel::<Vec<u8>>();
        thread::spawn(move || {
            let mut writer = writer;
            while let Ok(bytes) = input_rx.recv() {
                if writer.write_all(&bytes).is_err() {
                    break;
                }
                let _ = writer.flush();
            }
        });

        Ok(Shell {
            master: Arc::new(Mutex::new(master)),
            input: input_tx,
            child: Arc::new(Mutex::new(child)),
            exited: Arc::new(AtomicBool::new(false)),
            output: rx,
        })
    }

    /// Send bytes to the shell (the user's typing).
    ///
    /// This never blocks: the bytes are queued and written by the shell's
    /// own thread.
    pub fn write(&self, bytes: &[u8]) {
        let _ = self.input.send(bytes.to_vec());
    }

    /// Send a line (with newline) to the shell.
    pub fn write_line(&self, line: &str) {
        let mut bytes = line.as_bytes().to_vec();
        bytes.push(b'\r');
        self.write(&bytes);
    }

    /// Tell the shell the window size changed.
    pub fn resize(&self, rows: u16, cols: u16) {
        if let Ok(master) = self.master.lock() {
            let _ = master.resize(portable_pty::PtySize {
                rows: rows.max(8),
                cols: cols.max(20),
                pixel_width: 0,
                pixel_height: 0,
            });
        }
    }

    /// Has the shell exited?
    pub fn exited(&self) -> bool {
        if self.exited.load(Ordering::Relaxed) {
            return true;
        }
        if let Ok(mut c) = self.child.lock() {
            if c.try_wait().ok().flatten().is_some() {
                self.exited.store(true, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    /// Stop the shell.
    ///
    /// It is used when the terminal has decided it does not need the shell
    /// after all - a check, for instance - and the shell would otherwise stay
    /// behind with a socket that is about to be removed.
    pub fn stop(&self) {
        if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
        }
        self.exited.store(true, Ordering::Relaxed);
    }
}

/// The rc fragment that teaches the shell to route model commands to the
/// helper. It is written to a file and sourced, so the user's own rc files
/// are untouched.
pub fn injection(helper: &Path, socket: &Path, ollama_bin: &Path, llama_cli_bin: &Path) -> String {
    format!(
        r#"
# --- HERMETIC --------------------------------------------------
# Route model-loading commands through the helper, which applies the
# Heretic adapter before the model is loaded.
export HERMETIC_SOCKET="{socket}"
export HERMETIC_SHIM="{helper}"

# The terminal's own commands: `hermetic run <model>` loads a model,
# `hermetic models` lists what is there.
hermetic() {{
  "$HERMETIC_SHIM" "$@"
}}

if command -v ollama >/dev/null 2>&1 && ! alias ollama >/dev/null 2>&1; then
  hermetic_ollama() {{
    if [ "$1" = "run" ] && [ -n "$2" ]; then
      "$HERMETIC_SHIM" ollama-run "$2" "${{@:3}}"
    else
      "{ollama}" "$@"
    fi
  }}
  alias ollama=hermetic_ollama
fi

if command -v llama-cli >/dev/null 2>&1 && ! alias llama-cli >/dev/null 2>&1; then
  hermetic_llama_cli() {{
    model=""
    prev=""
    for a in "$@"; do
      if [ "$prev" = "-m" ] || [ "$prev" = "--model" ]; then model="$a"; fi
      prev="$a"
    done
    if [ -n "$model" ]; then
      "$HERMETIC_SHIM" llama-cli "$model"
    else
      "{llama_cli}" "$@"
    fi
  }}
  alias llama-cli=hermetic_llama_cli
fi
# ------------------------------------------------------------------
"#,
        socket = socket.display(),
        helper = helper.display(),
        ollama = ollama_bin.display(),
        llama_cli = llama_cli_bin.display(),
    )
}

/// A directory with a `.zshrc` that sources the user's own files and then
/// the interception.
///
/// Everything in here belongs to *this* terminal and to nobody else. The
/// interception carries the socket the engines report to, and that socket
/// dies with the terminal: two terminals sharing one file would leave the
/// second one's shells pointing at the first one's socket, and an engine
/// that cannot reach it says so with a stack trace instead of building
/// anything.
pub fn zdotdir(base: &Path, injection: &str) -> Result<PathBuf> {
    let dir = base.join(format!("zdot-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;

    // The interception lives inside the directory, so it goes when the
    // directory goes.
    let injection_path = dir.join("injection.zsh");
    let rc = format!(
        r#"# Generated by HERMETIC. The user's own files are sourced first.
for f in "$HOME/.zshenv" "$HOME/.zprofile" "$HOME/.zshrc"; do
  [ -f "$f" ] && source "$f"
done
source "{rc}"
"#,
        rc = injection_path.display(),
    );

    std::fs::write(&injection_path, injection)?;
    std::fs::write(dir.join(".zshrc"), rc)?;
    std::fs::write(dir.join(".zshenv"), "")?;
    Ok(dir)
}

/// Remove this terminal's directory, and the ones of terminals that are gone.
///
/// Every terminal makes a directory of its own, named after its process, so
/// that two terminals never share one. A terminal that is killed leaves its
/// directory behind, and they pile up: sixty of them is three megabytes of
/// nothing. The directory of a process that is no longer running is dead
/// weight, so it goes.
///
/// A directory that is less than ten minutes old is left alone even if the
/// process is gone: the number may simply have been given to another process
/// in the meantime, and that process's terminal must keep its directory.
pub fn cleanup(base: &Path) {
    let _ = std::fs::remove_dir_all(base.join(format!("zdot-{}", std::process::id())));
    prune(base);
}

/// Remove the directories of terminals that are no longer running.
pub fn prune(base: &Path) {
    let Ok(entries) = std::fs::read_dir(base) else {
        return;
    };
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(pid) = name
            .strip_prefix("zdot-")
            .and_then(|rest| rest.parse::<i32>().ok())
        else {
            continue;
        };
        if pid == std::process::id() as i32 || is_running(pid) {
            continue;
        }
        // Too young to be sure: the number may have been given to another
        // process in the meantime.
        let Ok(modified) = entry.metadata().and_then(|meta| meta.modified()) else {
            continue;
        };
        let age = now.duration_since(modified).unwrap_or_default();
        if age < std::time::Duration::from_secs(600) {
            continue;
        }
        let _ = std::fs::remove_dir_all(entry.path());
    }
}

/// Is there still a process with this number?
fn is_running(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    // Signal 0 does not signal anything: it only asks whether the process
    // could be signalled. A process that exists but belongs to somebody else
    // answers with a permission error, which still means it exists.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}
