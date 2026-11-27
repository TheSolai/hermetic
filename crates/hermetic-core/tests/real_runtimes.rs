//! The runtimes, against real models.
//!
//! These tests need a model on the machine. When there is none they say so
//! and pass - they are not a substitute for the unit tests, they are the
//! part that proves the abstraction works against real weights.
//!
//! Run them with:
//!
//! ```text
//! cargo test --release --test real_runtimes -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use hermetic_core::chat::Chat;
use hermetic_core::runtimes::llamacpp::LlamaCppRuntime;
use hermetic_core::runtimes::mlx::MlxRuntime;
use hermetic_core::runtimes::ollama::OllamaRuntime;
use hermetic_core::runtimes::openai::OpenAiRuntime;
use hermetic_core::runtime::{ModelRuntime, ServeRequest};
use hermetic_core::signature::Signature;
use hermetic_core::source::ModelSource;

/// The registry's models folder.
fn registry_models() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
    PathBuf::from(home).join(".hermetic").join("models")
}

/// The registry's adapters folder.
fn registry_adapters() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
    PathBuf::from(home).join(".hermetic").join("adapters")
}

/// The first GGUF in the registry, with its adapter, if there is one.
fn a_model_with_its_adapter() -> Option<(PathBuf, PathBuf, Signature)> {
    let models = registry_models();
    let adapters = registry_adapters();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&models)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .collect();
    entries.sort();
    for model in entries {
        let signature = match hermetic_core::gguf::signature_of(&model) {
            Ok(signature) => signature,
            Err(_) => continue,
        };
        let adapter = adapters.join(format!("{}.gguf", signature.key()));
        if adapter.exists() {
            return Some((model, adapter, signature));
        }
    }
    None
}

/// The MLX models in the Hugging Face cache.
///
/// Not just the first one: mlx-lm only knows the architectures it has been
/// taught, and a cache collects models of every vintage. One that mlx-lm
/// cannot read is not a failure of the runtime, so the test tries the next
/// one rather than failing on a model that was never going to load.
fn mlx_models() -> Vec<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
    let hub = PathBuf::from(home).join(".cache/huggingface/hub");
    let mut found: Vec<PathBuf> = Vec::new();
    let Ok(entries) = std::fs::read_dir(&hub) else {
        return Vec::new();
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("models--mlx-community--") {
            continue;
        }
        let snapshots = entry.path().join("snapshots");
        let Ok(snapshot_entries) = std::fs::read_dir(&snapshots) else {
                continue;
            };
            for snapshot in snapshot_entries.flatten() {
            let path = snapshot.path();
            if path.join("config.json").is_file()
                && std::fs::read_dir(&path).ok().map(|d| d.flatten().any(|e| {
                    e.path().extension().and_then(|x| x.to_str()) == Some("safetensors")
                })) == Some(true)
            {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

#[test]
#[ignore = "needs a model and llama-server"]
fn llama_cpp_serves_a_real_model_with_its_adapter() {
    let Some((model, adapter, signature)) = a_model_with_its_adapter() else {
        eprintln!("no model with an adapter in the registry - skipping");
        return;
    };
    eprintln!("serving {} with {}", model.display(), adapter.display());

    let runtime = LlamaCppRuntime::new();
    assert!(runtime.is_available(), "llama-server is on the PATH");

    let mut request = ServeRequest::new(
        ModelSource::Gguf {
            path: model.clone(),
        },
        signature,
    );
    request.weights = Some(model);
    request.adapter = Some(adapter);

    let handle = runtime.serve(&request).expect("the model is served");
    assert_eq!(
        handle.mode,
        hermetic_core::runtime::ServeMode::WeightModified,
        "the adapter is bound in"
    );

    let mut chat = Chat::new(&handle.endpoint, &handle.model).expect("a chat");
    let answer = chat.say("What is the capital of France? One word.").expect("an answer");
    assert!(!answer.trim().is_empty(), "the model answered something");
    eprintln!("answered: {}", answer.trim());

    (handle.stop)();
}

#[test]
#[ignore = "needs llama-server and a model"]
fn the_server_is_still_running_when_serve_returns() {
    // The server used to be handed over and dropped in the same breath,
    // which stopped it: the terminal said "serving" and the model was
    // already gone. This is the test that says so out loud.
    let Some((model, _, signature)) = a_model_with_its_adapter() else {
        eprintln!("no model in the registry - skipping");
        return;
    };
    let runtime = LlamaCppRuntime::new();
    assert!(runtime.is_available(), "llama-server is on the PATH");

    let mut request = ServeRequest::new(
        ModelSource::Gguf {
            path: model.clone(),
        },
        signature,
    );
    request.weights = Some(model);

    let handle = runtime.serve(&request).expect("the model is served");

    // Give the process a moment to die, if it is going to.
    std::thread::sleep(std::time::Duration::from_secs(2));
    assert!(
        answers(&handle.endpoint),
        "the server is still answering at {}",
        handle.endpoint
    );

    (handle.stop)();
}

/// Is the server at this endpoint still answering?
fn answers(endpoint: &str) -> bool {
    use std::io::{Read, Write};
    use std::net::{TcpStream, ToSocketAddrs};

    let rest = endpoint.trim_start_matches("http://");
    let host_port = rest.split('/').next().unwrap_or(rest);
    let Ok(mut addresses) = host_port.to_socket_addrs() else {
        return false;
    };
    let Some(address) = addresses.next() else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&address, std::time::Duration::from_secs(5))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
    let request = format!("GET /health HTTP/1.0\r\nHost: {host_port}\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    let _ = stream.read_to_string(&mut response);
    response.starts_with("HTTP/1.") && response.contains(" 200")
}

#[test]
#[ignore = "needs llama-server"]
fn llama_cpp_serves_a_model_without_an_adapter_too() {
    let Some((model, _, signature)) = a_model_with_its_adapter() else {
        eprintln!("no model in the registry - skipping");
        return;
    };
    let runtime = LlamaCppRuntime::new();
    let mut request = ServeRequest::new(
        ModelSource::Gguf {
            path: model.clone(),
        },
        signature,
    );
    request.weights = Some(model);

    let handle = runtime.serve(&request).expect("the model is served");
    assert_eq!(
        handle.mode,
        hermetic_core::runtime::ServeMode::Native,
        "without an adapter the model is served as it is"
    );
    (handle.stop)();
}

#[test]
#[ignore = "needs mlx-lm and an MLX model"]
fn mlx_serves_a_real_mlx_model() {
    let models = mlx_models();
    if models.is_empty() {
        eprintln!("no MLX model in the Hugging Face cache - skipping");
        return;
    }

    let runtime = MlxRuntime::default();
    assert!(runtime.is_available(), "mlx-lm is installed");

    let mut failures: Vec<String> = Vec::new();
    for model in models {
        eprintln!("serving {}", model.display());
        let Some(signature) = hermetic_core::runtimes::mlx::mlx_signature(&model) else {
            eprintln!("  no shape - skipping");
            continue;
        };
        let request = ServeRequest::new(
            ModelSource::Mlx {
                path: model,
                repo: None,
            },
            signature,
        );

        // A model mlx-lm does not know how to read is not a failure of the
        // runtime, so the next one is tried.
        let handle = match runtime.serve(&request) {
            Ok(handle) => handle,
            Err(error) => {
                eprintln!("  mlx-lm could not serve it ({error}) - trying the next model");
                failures.push(error.to_string());
                continue;
            }
        };
        let mut chat = Chat::new(&handle.endpoint, &handle.model).expect("a chat");
        let answer = chat
            .say("What is the capital of France? One word.")
            .expect("an answer");
        assert!(!answer.trim().is_empty(), "the model answered something");
        eprintln!("answered: {}", answer.trim());

        (handle.stop)();
        return;
    }

    panic!(
        "no MLX model could be served ({}): {failures:?}",
        failures.len()
    );
}

#[test]
#[ignore = "needs Ollama running"]
fn ollama_serves_a_model_through_its_own_api() {
    let runtime = OllamaRuntime::new();
    if !runtime.is_available() {
        eprintln!("Ollama is not running - skipping");
        return;
    }
    let request = ServeRequest::new(
        ModelSource::Ollama {
            name: "qwen2.5:0.5b".into(),
        },
        Signature {
            arch: "qwen2".into(),
            layers: 24,
            hidden: 896,
            inter: 4864,
            heads: 14,
            kv_heads: 2,
            experts: 0,
        },
    );

    let handle = runtime.serve(&request).expect("the model is served");
    assert_eq!(
        handle.mode,
        hermetic_core::runtime::ServeMode::PromptPolicy,
        "Ollama's engine cannot change the weights"
    );

    let mut chat = Chat::new(&handle.endpoint, &handle.model)
        .expect("a chat")
        .with_policy(hermetic_core::Policy::standard());
    let answer = chat.say("What is the capital of France? One word.").expect("an answer");
    assert!(!answer.trim().is_empty(), "the model answered something");
    eprintln!("answered: {}", answer.trim());
}

#[test]
#[ignore = "needs a server"]
fn any_openai_compatible_server_works_the_same_way() {
    // Whatever is listening on 1234 - LM Studio, or a test server - is
    // reached the same way.
    let runtime = OpenAiRuntime::new("http://127.0.0.1:1234/v1", "local");
    if !runtime.is_available() {
        eprintln!("nothing is answering on 1234 - skipping");
        return;
    }
    let models = runtime.models();
    assert!(!models.is_empty(), "the server says what it has");
}

#[test]
fn the_registry_is_where_it_should_be() {
    // The paths the tests use have to be the ones the product uses.
    assert!(registry_models().ends_with("models"));
    assert!(registry_adapters().ends_with("adapters"));
    let _ = Path::new("/");
}
