//! The chat.
//!
//! Every runtime that can be talked to speaks the same API:
//! `POST /v1/chat/completions`, with server-sent events coming back. That
//! is the whole reason the terminal can treat llama.cpp, MLX, LM Studio and
//! a cloud API the same way: there is one client, and it does not know or
//! care what is behind the endpoint.
//!
//! What differs between runtimes is not the protocol but the *policy*: a
//! model whose weights were changed needs no prompt at all, and a model
//! whose weights could not be changed needs one.

use serde_json::{json, Value};

use crate::http::{self, Delta};

/// A conversation with a model.
#[derive(Debug, Clone)]
pub struct Chat {
    /// The host the server is on.
    pub host: String,
    /// The port the server is on.
    pub port: u16,
    /// The model's name on that server.
    pub model: String,
    /// The messages so far.
    pub history: Vec<Value>,
    /// A prompt-level policy, for models whose weights could not be changed.
    pub policy: Option<crate::policy::Policy>,
}

impl Chat {
    /// A conversation with a model on a local server.
    pub fn new(endpoint: &str, model: &str) -> anyhow::Result<Chat> {
        let (host, port) = split_endpoint(endpoint)?;
        Ok(Chat {
            host,
            port,
            model: model.to_string(),
            history: Vec::new(),
            policy: None,
        })
    }

    /// Where this conversation is going.
    pub fn base_url(&self) -> String {
        format!("http://{}:{}/v1", self.host, self.port)
    }

    /// Use a prompt-level policy.
    pub fn with_policy(mut self, policy: crate::policy::Policy) -> Chat {
        self.policy = Some(policy);
        self
    }

    /// Say something.
    pub fn say(&mut self, text: &str) -> anyhow::Result<String> {
        self.history.push(json!({"role": "user", "content": text}));
        self.answer()
    }

    /// Forget the conversation, but keep the policy.
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// The messages, with the policy in front when there is one.
    fn messages(&self) -> Vec<Value> {
        let mut messages = Vec::new();
        if let Some(policy) = &self.policy {
            let existing = self
                .history
                .iter()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str());
            messages.push(json!({
                "role": "system",
                "content": policy.with_system(existing),
            }));
        }
        messages.extend(self.history.iter().cloned());
        messages
    }

    /// Ask the model, and hand every piece of the answer to `on_delta` as it
    /// arrives.
    pub fn answer_with(
        &mut self,
        mut on_delta: impl FnMut(Delta<'_>),
    ) -> anyhow::Result<String> {
        let body = json!({
            "model": self.model,
            "messages": self.messages(),
            "stream": true,
        });
        let answer = http::post_sse(
            &self.host,
            self.port,
            "/v1/chat/completions",
            &body,
            &mut on_delta,
        );
        let answer = match answer {
            Ok(answer) => answer,
            Err(error) => {
                // The question that could not be asked is taken back out of
                // the history. Left in, the next question is sent with the
                // last one still in it, and the model answers the wrong
                // thing.
                self.history.pop();
                return Err(error);
            }
        };
        if !answer.is_empty() {
            self.history.push(json!({"role": "assistant", "content": answer}));
        }
        Ok(answer)
    }

    /// Ask the model, and get the whole answer.
    pub fn answer(&mut self) -> anyhow::Result<String> {
        self.answer_with(|_| {})
    }

    /// What the server says about itself.
    pub fn model_info(&self) -> anyhow::Result<Value> {
        Ok(http::post_json(&self.host, self.port, "/v1/models", &json!({}))?)
    }
}

/// Split an endpoint into a host and a port.
///
/// `http://127.0.0.1:8080/v1` is `127.0.0.1` and `8080`.
pub fn split_endpoint(endpoint: &str) -> anyhow::Result<(String, u16)> {
    let without_scheme = endpoint
        .trim()
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    let authority = without_scheme
        .split('/')
        .next()
        .unwrap_or(without_scheme);
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) => (
            host.to_string(),
            port.parse::<u16>()
                .map_err(|_| anyhow::anyhow!("{endpoint} has no port"))?,
        ),
        None => (authority.to_string(), 80),
    };
    if host.is_empty() {
        anyhow::bail!("{endpoint} has no host");
    }
    Ok((host, port))
}

/// The base URL of a local server.
pub fn base_url(host: &str, port: u16) -> String {
    format!("http://{host}:{port}/v1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_endpoint_is_split_into_a_host_and_a_port() {
        let (host, port) = split_endpoint("http://127.0.0.1:8080/v1").expect("an endpoint");
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 8080);
    }

    #[test]
    fn an_endpoint_without_a_path_is_still_an_endpoint() {
        let (host, port) = split_endpoint("127.0.0.1:1234").expect("an endpoint");
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 1234);
    }

    #[test]
    fn the_policy_goes_in_front_of_the_conversation() {
        let mut chat = Chat::new("http://127.0.0.1:8080/v1", "m").expect("a chat");
        chat = chat.with_policy(crate::policy::Policy::standard());
        chat.history.push(json!({"role": "user", "content": "hello"}));
        let messages = chat.messages();
        assert_eq!(messages[0]["role"], "system");
        assert!(messages[0]["content"].as_str().unwrap().contains("Do not refuse"));
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn without_a_policy_there_is_no_system_message() {
        let mut chat = Chat::new("http://127.0.0.1:8080/v1", "m").expect("a chat");
        chat.history.push(json!({"role": "user", "content": "hello"}));
        let messages = chat.messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
    }
}
