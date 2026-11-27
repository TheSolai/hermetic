//! The unified interface, against a server that is not llama.cpp.
//!
//! The point of the abstraction is that the terminal does not care what is
//! behind the endpoint. So this test does not use llama.cpp: it uses a
//! three-line HTTP server that speaks the OpenAI chat API, which is exactly
//! what LM Studio, vLLM, `mlx_lm.server` and every other backend speak.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use hermetic_core::chat::Chat;
use hermetic_core::discovery::{self, Sources};
use hermetic_core::policy::Policy;
use hermetic_core::registry::Registry;
use hermetic_core::source::SourceKind;

/// A server that speaks the OpenAI chat API, and nothing else.
struct FakeServer {
    port: u16,
    stop: Arc<AtomicBool>,
    /// The messages of the last request, so a test can look at them.
    seen: Arc<std::sync::Mutex<Vec<String>>>,
}

impl FakeServer {
    fn start() -> FakeServer {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let stop = Arc::new(AtomicBool::new(false));
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));

        let stop_flag = Arc::clone(&stop);
        let seen_flag = Arc::clone(&seen);
        thread::spawn(move || {
            listener.set_nonblocking(true).ok();
            while !stop_flag.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        stream.set_nonblocking(false).ok();
                        let mut reader = BufReader::new(stream.try_clone().expect("clone"));
                        let mut content_length = 0usize;
                        let mut line = String::new();
                        while reader.read_line(&mut line).unwrap_or(0) > 0 {
                            let trimmed = line.trim().to_string();
                            if trimmed.is_empty() {
                                break;
                            }
                            if let Some(value) = trimmed.strip_prefix("Content-Length:") {
                                content_length = value.trim().parse().unwrap_or(0);
                            }
                            line.clear();
                        }
                        let mut body = vec![0u8; content_length];
                        if content_length > 0 {
                            reader.read_exact(&mut body).ok();
                        }
                        let body = String::from_utf8_lossy(&body).to_string();
                        seen_flag
                            .lock()
                            .expect("lock")
                            .push(body.clone());

                        let mut stream = stream;
                        let answer = if body.contains("\"stream\":true") {
                            // Server-sent events: no content length, and the
                            // connection ends when the answer does.
                            let mut out = String::new();
                            out.push_str("HTTP/1.1 200 OK\r\n");
                            out.push_str("Content-Type: text/event-stream\r\n\r\n");
                            for piece in ["The ", "lock ", "picks ", "itself."] {
                                let event = format!(
                                    "data: {}\n\n",
                                    serde_json::json!({
                                        "choices": [{"delta": {"content": piece}}]
                                    })
                                );
                                out.push_str(&event);
                            }
                            out.push_str("data: [DONE]\n\n");
                            out
                        } else {
                            let payload = serde_json::json!({
                                "choices": [{"message": {"content": "The lock picks itself."}}]
                            });
                            format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                                 Content-Length: {}\r\n\r\n{}",
                                payload.to_string().len(),
                                payload
                            )
                        };
                        stream.write_all(answer.as_bytes()).ok();
                        stream.flush().ok();
                    }
                    Err(_) => thread::sleep(std::time::Duration::from_millis(10)),
                }
            }
        });

        FakeServer {
            port,
            stop,
            seen,
        }
    }

    fn endpoint(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    fn last_request(&self) -> String {
        self.seen.lock().expect("lock").last().cloned().unwrap_or_default()
    }
}

impl Drop for FakeServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[test]
fn the_chat_client_talks_to_any_openai_compatible_server() {
    let server = FakeServer::start();
    let mut chat = Chat::new(&server.endpoint(), "some-model").expect("a chat");

    let answer = chat.answer().expect("an answer");
    assert_eq!(answer, "The lock picks itself.");
    assert!(
        server.last_request().contains("some-model"),
        "the request names the model"
    );
}

#[test]
fn a_policy_is_only_sent_when_there_is_one() {
    let server = FakeServer::start();

    let mut plain = Chat::new(&server.endpoint(), "m").expect("a chat");
    plain.say("hello").expect("an answer");
    let without = server.last_request();
    assert!(!without.contains("Do not refuse"), "no policy was sent");

    let mut with = Chat::new(&server.endpoint(), "m")
        .expect("a chat")
        .with_policy(Policy::standard());
    with.say("hello").expect("an answer");
    let with_policy = server.last_request();
    assert!(
        with_policy.contains("Do not refuse"),
        "the policy was sent: {with_policy}"
    );
}

#[test]
fn the_endpoint_is_split_into_a_host_and_a_port() {
    let (host, port) = hermetic_core::chat::split_endpoint("http://127.0.0.1:9999/v1")
        .expect("an endpoint");
    assert_eq!(host, "127.0.0.1");
    assert_eq!(port, 9999);
}

#[test]
fn discovery_finds_a_gguf_and_an_mlx_model() {
    let directory = std::env::temp_dir().join(format!("hermetic-discovery-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).expect("the directory");

    // A GGUF file.
    let gguf = directory.join("tiny.gguf");
    std::fs::write(&gguf, b"GGUFnot-really-a-model").expect("the gguf");

    // An MLX model: a config and a safetensors file.
    let mlx = directory.join("mlx-model");
    std::fs::create_dir_all(&mlx).expect("the mlx directory");
    std::fs::write(
        mlx.join("config.json"),
        r#"{"model_type":"llama","num_hidden_layers":4,"hidden_size":8,"intermediate_size":16}"#,
    )
    .expect("the config");
    std::fs::write(mlx.join("weights.safetensors"), "x").expect("the weights");

    let sources = Sources {
        ollama: None,
        gguf_dirs: vec![directory.clone()],
        mlx_dirs: vec![directory.clone()],
        remote: vec![],
    };
    let found = discovery::discover(&sources, &Registry::new());

    assert!(
        found.iter().any(|m| m.source.kind() == SourceKind::Gguf),
        "the GGUF was found"
    );
    assert!(
        found.iter().any(|m| m.source.kind() == SourceKind::Mlx),
        "the MLX model was found"
    );

    let mlx_model = found
        .iter()
        .find(|m| m.source.kind() == SourceKind::Mlx)
        .expect("the mlx model");
    assert_eq!(mlx_model.signature.as_ref().expect("a shape").layers, 4);

    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn a_remote_server_is_a_source_too() {
    let sources = Sources {
        ollama: None,
        gguf_dirs: vec![],
        mlx_dirs: vec![],
        remote: vec![("http://127.0.0.1:1234/v1".into(), "local".into())],
    };
    let found = discovery::discover(&sources, &Registry::new());
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].source.kind(), SourceKind::Remote);
    assert_eq!(found[0].label(), "local");
}

#[test]
fn a_gguf_that_is_not_a_gguf_has_no_shape() {
    let directory = std::env::temp_dir().join(format!("hermetic-shape-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).expect("the directory");
    let path: PathBuf = directory.join("broken.gguf");
    std::fs::write(&path, b"not a gguf at all").expect("the file");

    let signature = hermetic_core::gguf::signature_of(&path);
    assert!(signature.is_err(), "a broken file has no shape");
    let _ = std::fs::remove_dir_all(&directory);
}
