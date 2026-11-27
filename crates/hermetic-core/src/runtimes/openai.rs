//! Any server that speaks the OpenAI chat API.
//!
//! This is the runtime that makes the terminal universal. LM Studio, vLLM,
//! Jan, `text-generation-webui`, `mlx_lm.server`, llama.cpp itself, a cloud
//! API - they all answer `POST /v1/chat/completions`, and that is the whole
//! interface the terminal uses.
//!
//! It cannot change the weights: the server holds them and will not give
//! them up. So a model served this way gets a prompt-level policy, and the
//! terminal says so rather than pretending otherwise.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;

use crate::http;
use crate::runtime::{Modification, ModelRuntime, ServeHandle, ServeMode, ServeRequest};
use crate::source::ModelSource;

/// A server that speaks the OpenAI chat API.
#[derive(Debug, Clone)]
pub struct OpenAiRuntime {
    /// The base URL: `http://127.0.0.1:1234/v1`.
    pub base_url: String,
    /// The model's name on that server.
    pub model: String,
    /// How long to wait for the server to answer.
    pub timeout: Duration,
}

impl OpenAiRuntime {
    /// A runtime for a server and a model.
    pub fn new(base_url: &str, model: &str) -> OpenAiRuntime {
        OpenAiRuntime {
            base_url: normalise(base_url),
            model: model.to_string(),
            timeout: Duration::from_secs(10),
        }
    }

    /// The host and port of the server.
    pub fn address(&self) -> Result<(String, u16)> {
        crate::chat::split_endpoint(&self.base_url)
    }

    /// Does the server answer?
    ///
    /// The list of models is asked for with a GET. A POST is what Ollama's
    /// own runtime avoids for the same reason: a server that only answers
    /// GET replies 405 to a POST, and a running server would then be
    /// reported as a server that is not running.
    pub fn is_answering(&self) -> bool {
        let Ok((host, port)) = self.address() else {
            return false;
        };
        http::get_json(&host, port, "/v1/models").is_ok()
    }

    /// The models the server says it has.
    pub fn models(&self) -> Vec<String> {
        let Ok((host, port)) = self.address() else {
            return Vec::new();
        };
        let Ok(answer) = http::get_json(&host, port, "/v1/models") else {
            return Vec::new();
        };
        answer
            .get("data")
            .and_then(|d| d.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|m| m.get("id").and_then(|i| i.as_str()))
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Make sure the base URL ends with `/v1`.
fn normalise(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}

impl ModelRuntime for OpenAiRuntime {
    fn id(&self) -> &str {
        "openai"
    }

    fn name(&self) -> &str {
        "OpenAI-compatible"
    }

    fn modification(&self) -> Modification {
        // The server holds the weights. It will not change them.
        Modification::PromptOnly
    }

    fn accepts(&self, source: &ModelSource) -> bool {
        matches!(source, ModelSource::Remote { .. })
    }

    fn serve(&self, request: &ServeRequest) -> Result<ServeHandle> {
        let (host, port) = self.address()?;
        if !self.is_answering() {
            anyhow::bail!(
                "nothing is answering at {} - is the server running?",
                self.base_url
            );
        }
        let mut handle = ServeHandle::new(&self.base_url, ServeMode::PromptPolicy, &self.model)
            .served_by(self.id());
        handle.notes.push(format!(
            "served by {} at {}; its weights cannot be changed, so a \
             prompt-level policy is used",
            self.name(),
            self.base_url
        ));
        let _ = (host, port);
        let _ = request;
        Ok(handle)
    }

    fn is_available(&self) -> bool {
        self.is_answering()
    }
}

/// The servers this machine can see.
///
/// LM Studio and vLLM both have a default port, and both are usually
/// started by the user rather than by us.
pub fn known_servers() -> Vec<OpenAiRuntime> {
    vec![
        OpenAiRuntime::new("http://127.0.0.1:1234/v1", "local"),
        OpenAiRuntime::new("http://127.0.0.1:8000/v1", "local"),
        OpenAiRuntime::new("http://127.0.0.1:8080/v1", "local"),
    ]
}

/// The `mlx_lm.server` program, when it is installed.
pub fn mlx_server_command(python: &PathBuf) -> Option<Vec<String>> {
    if !python.exists() {
        return None;
    }
    Some(vec![
        python.display().to_string(),
        "-m".to_string(),
        "mlx_lm.server".to_string(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_base_url_always_ends_with_v1() {
        assert_eq!(
            OpenAiRuntime::new("http://127.0.0.1:1234", "m").base_url,
            "http://127.0.0.1:1234/v1"
        );
        assert_eq!(
            OpenAiRuntime::new("http://127.0.0.1:1234/v1", "m").base_url,
            "http://127.0.0.1:1234/v1"
        );
        assert_eq!(
            OpenAiRuntime::new("http://127.0.0.1:1234/", "m").base_url,
            "http://127.0.0.1:1234/v1"
        );
    }

    #[test]
    fn this_runtime_cannot_change_weights() {
        let runtime = OpenAiRuntime::new("http://127.0.0.1:1234/v1", "m");
        assert_eq!(runtime.id(), "openai");
        assert_eq!(runtime.modification(), Modification::PromptOnly);
        assert!(runtime.accepts(&ModelSource::Remote {
            base_url: "http://x/v1".into(),
            model: "m".into()
        }));
        assert!(!runtime.accepts(&ModelSource::Gguf {
            path: "/tmp/x.gguf".into()
        }));
    }

    #[test]
    fn the_address_is_read_from_the_url() {
        let runtime = OpenAiRuntime::new("http://127.0.0.1:4321/v1", "m");
        let (host, port) = runtime.address().expect("an address");
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 4321);
    }
}
