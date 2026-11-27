//! MLX: Apple's framework for models on Apple Silicon.
//!
//! An MLX model is not a file. It is a *directory*: a `config.json` and one
//! or more `.safetensors` files. There is no GGUF, and llama.cpp cannot read
//! it - which is exactly why it needs its own runtime.
//!
//! The good news is that safetensors are writable, so an MLX model *can* be
//! hereticed: the adapter is folded into the weights. The runtime serves it
//! with `mlx_lm.server`, which speaks the OpenAI chat API, so the terminal's
//! one chat client works with it unchanged.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::Value;

use crate::runtime::{Modification, ModelRuntime, ServeHandle, ServeMode, ServeRequest};
use crate::signature::Signature;
use crate::source::ModelSource;

/// How long to wait for the server to come up.
const START_TIMEOUT: Duration = Duration::from_secs(300);

/// The MLX runtime.
#[derive(Debug, Clone)]
pub struct MlxRuntime {
    /// The interpreter that has `mlx_lm` installed.
    pub python: PathBuf,
}

impl Default for MlxRuntime {
    fn default() -> MlxRuntime {
        MlxRuntime {
            python: crate::config::Config::load().python,
        }
    }
}

impl MlxRuntime {
    /// The runtime, with a particular interpreter.
    pub fn with_python(python: PathBuf) -> MlxRuntime {
        MlxRuntime { python }
    }

    /// Is `mlx_lm` installed for this interpreter?
    pub fn has_mlx(&self) -> bool {
        if !self.python.exists() {
            return false;
        }
        Command::new(&self.python)
            .args(["-c", "import mlx_lm"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

/// Is this directory an MLX model?
///
/// A config and at least one safetensors file. That is what an MLX model
/// looks like on disk.
pub fn find_mlx_model(directory: &Path) -> Option<PathBuf> {
    if !is_mlx_model(directory) {
        return None;
    }
    Some(directory.to_path_buf())
}

/// Is this directory an MLX model?
pub fn is_mlx_model(directory: &Path) -> bool {
    if !directory.join("config.json").is_file() {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(directory) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e == "safetensors")
    })
}

/// The shape of an MLX model, from its own config.
pub fn mlx_signature(directory: &Path) -> Option<Signature> {
    let text = std::fs::read_to_string(directory.join("config.json")).ok()?;
    let config: Value = serde_json::from_str(&text).ok()?;
    Signature::from_hf_config(&config).ok()
}

/// A running `mlx_lm.server`.
pub struct Server {
    child: Child,
    /// The port it is on.
    pub port: u16,
}

impl Server {
    /// Start the server for a model.
    pub fn start(python: &Path, model: &Path, port: u16) -> Result<Server> {
        let child = Command::new(python)
            .arg("-m")
            .arg("mlx_lm.server")
            .arg("--model")
            .arg(model)
            .arg("--port")
            .arg(port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("start mlx_lm.server")?;
        Ok(Server { child, port })
    }

    /// Wait until the server answers.
    pub fn wait_ready(&mut self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            if crate::http::post_json("127.0.0.1", self.port, "/v1/models", &serde_json::json!({}))
                .is_ok()
            {
                return Ok(());
            }
            if let Ok(Some(status)) = self.child.try_wait() {
                anyhow::bail!("mlx_lm.server exited with {status}");
            }
            if Instant::now() > deadline {
                anyhow::bail!("mlx_lm.server did not come up in time");
            }
            std::thread::sleep(Duration::from_millis(200));
        }
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

impl ModelRuntime for MlxRuntime {
    fn id(&self) -> &str {
        "mlx"
    }

    fn name(&self) -> &str {
        "MLX"
    }

    fn modification(&self) -> Modification {
        // Safetensors are writable, so the adapter can be folded into the
        // weights and the model served with the behaviour baked in.
        Modification::Native
    }

    fn accepts(&self, source: &ModelSource) -> bool {
        matches!(source, ModelSource::Mlx { .. })
    }

    fn serve(&self, request: &ServeRequest) -> Result<ServeHandle> {
        let ModelSource::Mlx { path, .. } = &request.source else {
            anyhow::bail!("MLX does not serve {}", request.source.label());
        };
        if !self.has_mlx() {
            anyhow::bail!(
                "mlx-lm is not installed for {} - install it with `pip install mlx-lm`",
                self.python.display()
            );
        }
        let port = crate::runtimes::llamacpp::free_port()?;
        let mut server = Server::start(&self.python, path, port)?;
        server.wait_ready(START_TIMEOUT)?;

        let mode = if request.adapter.is_some() {
            ServeMode::WeightModified
        } else {
            ServeMode::Native
        };
        let endpoint = crate::chat::base_url("127.0.0.1", port);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "mlx".to_string());
        let mut handle = ServeHandle::new(&endpoint, mode, &name);
        handle.notes.push(format!(
            "served by mlx_lm.server on port {port}, with the adapter folded into the weights"
        ));
        let pid = server.child.id() as i32;
        handle.stop = Box::new(move || {
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        });
        std::thread::spawn(move || {
            let _ = server;
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        });
        Ok(handle)
    }

    fn is_available(&self) -> bool {
        self.has_mlx()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_without_weights_is_not_an_mlx_model() {
        let directory = std::env::temp_dir().join(format!("hermetic-mlx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("the directory");
        assert!(!is_mlx_model(&directory));
        std::fs::write(directory.join("config.json"), "{}").expect("the config");
        assert!(!is_mlx_model(&directory));
        std::fs::write(directory.join("weights.safetensors"), "x").expect("the weights");
        assert!(is_mlx_model(&directory));
        assert!(find_mlx_model(&directory).is_some());
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn mlx_serves_mlx_models_and_nothing_else() {
        let runtime = MlxRuntime::with_python(PathBuf::from("/nowhere/python"));
        assert_eq!(runtime.id(), "mlx");
        assert_eq!(runtime.modification(), Modification::Native);
        assert!(runtime.accepts(&ModelSource::Mlx {
            path: "/tmp/m".into(),
            repo: None
        }));
        assert!(!runtime.accepts(&ModelSource::Gguf {
            path: "/tmp/x.gguf".into()
        }));
        assert!(!runtime.has_mlx(), "there is no python at /nowhere/python");
    }

    #[test]
    fn the_shape_comes_from_the_models_own_config() {
        let directory = std::env::temp_dir().join(format!("hermetic-mlx-shape-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("the directory");
        std::fs::write(
            directory.join("config.json"),
            r#"{"model_type":"llama","num_hidden_layers":32,"hidden_size":4096,"intermediate_size":14336}"#,
        )
        .expect("the config");
        let signature = mlx_signature(&directory).expect("a shape");
        assert_eq!(signature.key(), "llama-32-4096-14336");
        let _ = std::fs::remove_dir_all(&directory);
    }
}
