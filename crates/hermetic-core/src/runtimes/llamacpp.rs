//! llama.cpp: the runtime that can bind an adapter to the weights.
//!
//! This is the runtime that makes the whole thing work: llama.cpp takes a
//! base GGUF and a LoRA adapter and applies one to the other. The adapter
//! only binds if the shapes match exactly, which is why adapters are keyed
//! by shape and not by name.
//!
//! It serves the OpenAI chat API, so the terminal's one chat client talks to
//! it without knowing anything about llama.cpp.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use crate::runtime::{Modification, ModelRuntime, ServeHandle, ServeMode, ServeRequest};
use crate::source::ModelSource;

/// The line llama-server prints when the model is in memory and ready.
const READY_MARKER: &str = "model loaded";

/// How long the server may say nothing before it is treated as stuck.
///
/// A llama-server that is working prints its banner in under a second. One
/// that has said nothing for this long has not started at all - the GPU is
/// taken, or the process is wedged - and waiting for it only wastes the
/// user's time.
const SILENCE_LIMIT: Duration = Duration::from_secs(45);

/// A running llama-server.
pub struct Server {
    child: Child,
    /// The port it is on.
    pub port: u16,
    /// The weights it is serving.
    pub base: PathBuf,
    /// The adapter bound to them, if there is one.
    pub adapter: Option<PathBuf>,
    lines: Option<Receiver<String>>,
}

impl Server {
    /// Start llama-server with the base model and, if we have one, the
    /// adapter that makes it uncensored.
    pub fn start(
        base: &std::path::Path,
        adapter: Option<&std::path::Path>,
        port: u16,
        ctx: u32,
    ) -> Result<Server> {
        let mut cmd = Command::new("llama-server");
        cmd.arg("-m")
            .arg(base)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("-c")
            .arg(ctx.to_string())
            .arg("--no-warmup")
            .arg("-ngl")
            .arg("99")
            .arg("--flash-attn")
            .arg("on")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(adapter) = adapter {
            // `--lora-scaled` takes the file and the strength: the adapter is
            // applied at full strength, which is what Heretic optimised.
            cmd.arg("--lora-scaled")
                .arg(format!("{}:1.0", adapter.display()));
        }

        let mut child = cmd.spawn().context("start llama-server")?;

        let (tx, rx) = mpsc::channel();
        let out = child.stdout.take();
        let err = child.stderr.take();
        if let Some(out) = out {
            let tx = tx.clone();
            std::thread::spawn(move || {
                let reader = BufReader::new(out);
                for line in reader.lines().map_while(Result::ok) {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            });
        }
        if let Some(err) = err {
            std::thread::spawn(move || {
                let reader = BufReader::new(err);
                for line in reader.lines().map_while(Result::ok) {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            });
        }

        Ok(Server {
            child,
            port,
            base: base.to_path_buf(),
            adapter: adapter.map(|a| a.to_path_buf()),
            lines: Some(rx),
        })
    }

    /// Wait until the model is loaded.
    ///
    /// The server answers `/health` before the weights are in memory, so the
    /// only honest signal is the log line it prints when it is done.
    ///
    /// A server that says nothing at all is a different thing: it has not
    /// even started. That happens when the GPU is busy with another model,
    /// and waiting ten minutes for a line that will never come is what
    /// makes a screen look stuck.
    pub fn wait_ready(&mut self, timeout: Duration) -> Result<()> {
        let Some(lines) = self.lines.as_ref() else {
            bail!("the server's log is gone");
        };
        let deadline = Instant::now() + timeout;
        // A healthy server prints its banner in well under a second.
        let mut silence = Duration::ZERO;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                bail!("llama-server did not finish loading the model in time");
            }
            match lines.recv_timeout(remaining.min(SILENCE_LIMIT)) {
                Ok(line) => {
                    silence = Duration::ZERO;
                    if line.contains(READY_MARKER) {
                        return Ok(());
                    }
                    if adapter_failed(&line) {
                        bail!("llama-server refused the adapter: {line}");
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    silence += SILENCE_LIMIT;
                    if silence >= SILENCE_LIMIT {
                        bail!(
                            "llama-server has said nothing for {} seconds, so it never \
                             started.\n\
                             This is usually the GPU: another model is still loaded on it. \
                             Close the other models and try again.",
                            SILENCE_LIMIT.as_secs()
                        );
                    }
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    match self.child.try_wait() {
                        Ok(Some(status)) => bail!("llama-server exited with {status}"),
                        _ => bail!("llama-server stopped talking"),
                    }
                }
            }
        }
    }

    /// The process id of the server.
    pub fn child_id(&self) -> i32 {
        self.child.id() as i32
    }

    /// Take the remaining log lines, so they can be shown while the model is
    /// being talked to.
    pub fn take_lines(&mut self) -> Option<Receiver<String>> {
        self.lines.take()
    }

    /// Stop the server.
    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Did llama.cpp refuse the adapter?
///
/// A LoRA that does not fit the weights is not an error the server stops
/// for: it is a line in the log, and the model is served *without* the
/// adapter. The terminal would then say the model is hereticed when it is
/// not, so the line has to be recognised.
///
/// llama.cpp says "failed to load LoRA adapter" and "cannot apply LoRA", in
/// either case. The word is capitalised, and the word "error" is not
/// always there, so the test is on the parts that always are.
pub fn adapter_failed(line: &str) -> bool {
    let lower = line.to_lowercase();
    if !lower.contains("lora") && !lower.contains("adapter") {
        return false;
    }
    lower.contains("fail")
        || lower.contains("error")
        || lower.contains("cannot")
        || lower.contains("incompatible")
        || lower.contains("not supported")
}

/// Is this log line worth showing to the user?
pub fn is_interesting(line: &str) -> bool {
    let mut parts = line.split_whitespace();
    let _timestamp = parts.next();
    match parts.next() {
        Some("D") => return false,
        Some(_) => {}
        None => return false,
    }
    if line.contains("slot ")
        || line.contains("print_timing")
        || line.contains("release:")
        || line.contains("launch_slot")
        || line.contains("get_availabl")
        || line.contains("update_slots")
    {
        return false;
    }
    true
}

/// The llama.cpp runtime.
#[derive(Debug, Clone, Default)]
pub struct LlamaCppRuntime {
    /// The `llama-server` program, when it is not on the PATH.
    pub program: Option<PathBuf>,
}

impl LlamaCppRuntime {
    /// The runtime, with `llama-server` found on the PATH.
    pub fn new() -> LlamaCppRuntime {
        LlamaCppRuntime::default()
    }

    /// The runtime, with a particular `llama-server`.
    pub fn with_program(program: PathBuf) -> LlamaCppRuntime {
        LlamaCppRuntime {
            program: Some(program),
        }
    }

    /// The program to run.
    pub fn program(&self) -> PathBuf {
        self.program
            .clone()
            .unwrap_or_else(|| PathBuf::from("llama-server"))
    }
}

impl ModelRuntime for LlamaCppRuntime {
    fn id(&self) -> &str {
        "llama.cpp"
    }

    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn modification(&self) -> Modification {
        Modification::Native
    }

    fn accepts(&self, source: &ModelSource) -> bool {
        matches!(
            source,
            ModelSource::Gguf { .. } | ModelSource::Ollama { .. } | ModelSource::Mlx { .. }
        )
    }

    fn serve(&self, request: &ServeRequest) -> Result<ServeHandle> {
        let weights = request
            .weights
            .clone()
            .ok_or_else(|| anyhow::anyhow!("there are no weights to serve"))?;
        let port = free_port()?;
        let mut server = Server::start(
            &weights,
            request.adapter.as_deref(),
            port,
            request.context,
        )?;
        server.wait_ready(Duration::from_secs(600))?;

        let mode = if request.is_modified() {
            ServeMode::WeightModified
        } else {
            ServeMode::Native
        };
        let name = crate::gguf::gguf_name(&weights)
            .unwrap_or_else(|| weights.display().to_string());
        let endpoint = crate::chat::base_url("127.0.0.1", port);
        let pid = server.child_id();
        let mut handle = ServeHandle::new(&endpoint, mode, &name).served_by(self.id());
        // Say what was actually done. An adapter bound in and an adapter
        // folded into the weights are different things, and the difference
        // is the whole point of the product.
        handle.notes.push(match (request.adapter.is_some(), request.merged) {
            (true, _) => format!("served by llama.cpp on port {port}, with the adapter bound in"),
            (false, true) => format!(
                "served by llama.cpp on port {port}, with the adapter folded into the weights"
            ),
            (false, false) => format!("served by llama.cpp on port {port}, with the weights as they are"),
        });
        handle.stop = Box::new(move || {
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        });
        // The server is moved into a thread so it lives as long as the
        // handle does, and dies with it.
        //
        // It has to be *held*, not dropped: `Server::drop` stops the
        // process, and `let _ = server` drops it at once. That is what
        // killed every model this terminal ever served - the screen said
        // "serving" and the server was already gone.
        std::thread::spawn(move || {
            let mut server = server;
            // Wait for the process to end. The thread then ends, and the
            // server is dropped, which stops it cleanly.
            let _ = server.child.wait();
        });
        Ok(handle)
    }

    fn is_available(&self) -> bool {
        which(&self.program().to_string_lossy()).is_some()
    }
}

/// Find a program on the PATH.
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path) {
        let candidate = directory.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// A port nobody is using.
pub fn free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_lines_are_not_interesting() {
        assert!(!is_interesting("0.02.842.564 D ggml_metal_library: loaded kernel"));
        assert!(is_interesting("0.02.842.564 I srv llama_server: model loaded"));
        assert!(!is_interesting(""));
    }

    #[test]
    fn llama_cpp_serves_weights_but_not_servers() {
        let runtime = LlamaCppRuntime::new();
        assert_eq!(runtime.id(), "llama.cpp");
        assert_eq!(runtime.modification(), Modification::Native);
        assert!(runtime.accepts(&ModelSource::Gguf {
            path: "/tmp/x.gguf".into()
        }));
        assert!(!runtime.accepts(&ModelSource::Remote {
            base_url: "http://x/v1".into(),
            model: "m".into()
        }));
    }
}
