//! The interceptor.
//!
//! The shell routes model commands here. This program resolves the model,
//! makes sure a Heretic adapter exists for that model's shape, serves the
//! model with the adapter bound to it and then talks to it.
//!
//! The adapter is the point: a runtime will only bind an adapter whose
//! tensors line up with the base model, so adapters are keyed by *shape* -
//! architecture, layers, hidden size, feed-forward size - and not by name.
//! One adapter therefore serves every model of that shape, in every
//! quantisation.
//!
//! What the runtime is called is not decided here either: the planner works
//! it out from where the weights are and what can change them, and the
//! terminal's one chat client talks to whatever it ends up serving.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use anyhow::{Context, Result};
use serde_json::json;

use hermetic_core::chat::Chat;
use hermetic_core::http::Delta;
use hermetic_core::hub;
use hermetic_core::planner::{self, Plan};
use hermetic_core::policy::Policy;
use hermetic_core::registry::Registry;
use hermetic_core::runtime::{ModelRuntime, ServeHandle, ServeMode, ServeRequest};
use hermetic_core::source::{AdapterState, ModelDescriptor, ModelSource};
use hermetic_core::weights;

use hermetic_terminal::{catalogue, channel, runtime, spec, SOCKET_ENV, TAB_ENV};

/// The model that is being served, so a signal can stop it.
static SERVED: Mutex<Option<ServeHandle>> = Mutex::new(None);

/// Whether the model has been announced to the terminal.
static ANNOUNCED: AtomicBool = AtomicBool::new(false);

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let model = args.get(2).map(|s| s.as_str()).unwrap_or("");

    match mode {
        "run" | "ollama-run" | "llama-cli" => run_and_report(model),
        "models" => {
            list();
            Ok(())
        }
        other => {
            eprintln!("hermetic-shim: unknown mode '{other}'");
            eprintln!(
                "usage: hermetic-shim run <model> | hermetic-shim ollama-run <model> \
                 | hermetic-shim llama-cli <file.gguf> | hermetic-shim models"
            );
            std::process::exit(2);
        }
    }
}

/// Load a model, and say clearly when that did not work.
///
/// The engine's own traceback is above; this is the one line that says what
/// it means, both in the terminal and in the panel.
fn run_and_report(spec_arg: &str) -> Result<()> {
    match run(spec_arg) {
        Ok(()) => Ok(()),
        Err(error) => {
            let reason = format!("{error:#}");
            if let Some(socket) = std::env::var_os(SOCKET_ENV).map(PathBuf::from) {
                let event = json!({
                    "event": "error",
                    "error": reason,
                });
                let _ = channel::send_event(&socket, &event);
            }
            eprintln!();
            eprintln!("  \x1b[1;31mthe model could not be loaded\x1b[0m");
            for line in reason.lines() {
                eprintln!("  \x1b[2m{line}\x1b[0m");
            }
            eprintln!();
            std::process::exit(1);
        }
    }
}

/// Every model that can be found, and what can be done with it.
fn list() {
    let registry = Registry::new();
    let sources = catalogue::sources(&[], &[], &[]);
    let entries = catalogue::catalogue(&sources, &registry);

    println!();
    println!(
        "\x1b[1;36m  HERMETIC\x1b[0m \x1b[2m· {} models on this machine\x1b[0m",
        entries.len()
    );
    println!();
    println!(
        "  \x1b[1m{:<30} {:<8} {:<8} {:<10} {:<12} {}\x1b[0m",
        "MODEL", "SIZE", "KIND", "STATE", "SHAPE", "SOURCE"
    );

    for entry in &entries {
        let shape = match &entry.signature {
            Some(signature) => signature.key(),
            None => "unknown".into(),
        };
        println!(
            "  {:<30} {:<8} {:<8} {:<10} {:<12} {}",
            truncate(&entry.name, 30),
            human(entry.size),
            entry.kind.label(),
            entry.state,
            truncate(&shape, 12),
            truncate(&entry.spec(), 40),
        );
    }
    println!();
    println!(
        "  \x1b[2mready\x1b[0m = an adapter for that shape exists   \x1b[2mbuild\x1b[0m = it would be built on demand"
    );
    println!();
}

/// How big a model is.
fn human(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes}{}", UNITS[unit])
    } else {
        format!("{value:.1}{}", UNITS[unit])
    }
}

/// A string that fits in a column.
fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let mut out: String = text.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Load a model and talk to it.
fn run(spec_arg: &str) -> Result<()> {
    let source =
        spec::source_of(spec_arg).with_context(|| format!("could not resolve '{spec_arg}'"))?;
    let registry = Registry::new();
    let signature = spec::shape_of(&source);

    banner();
    println!("  model     {}", source.label());
    println!("  kind      {}", source.kind().label());
    if let Some(signature) = &signature {
        println!("  shape     {}", signature.describe());
        println!("  key       {}", signature.key());
    } else {
        println!("  shape     unknown");
    }

    // The weights to serve, and the adapter to bind to them.
    let (weights, adapter, merged) = prepare(&source, &signature, &registry, spec_arg)?;

    // What the planner makes of it.
    let descriptor = ModelDescriptor {
        source: source.clone(),
        signature: signature.clone(),
        size_bytes: None,
        parameters: None,
        quantisation: None,
        adapter: if merged {
            AdapterState::Merged
        } else if adapter.is_some() {
            AdapterState::Ready
        } else {
            AdapterState::Missing
        },
    };

    let runtimes = runtime::runtimes_for(&source);
    let refs: Vec<&dyn ModelRuntime> = runtimes.iter().map(|runtime| runtime.as_ref()).collect();
    let can_modify = source.is_modifiable();
    let plan = planner::plan(&descriptor, &refs, can_modify);
    println!("  plan      {}", plan.reason);

    let mut request = ServeRequest::new(
        source.clone(),
        descriptor
            .signature
            .clone()
            .unwrap_or_else(spec::placeholder_signature),
    );
    request.weights = weights;
    request.adapter = adapter.clone();
    request.context = 4096;
    if plan.mode == ServeMode::PromptPolicy {
        request.policy = Some(planner::fallback_policy(&descriptor));
        println!(
            "  note      the weights of this model cannot be changed, so a prompt-level policy is used"
        );
    }

    // Serve it.
    let handle = serve(&plan, &request, &refs)?;

    println!("  serving   {} ({})", handle.endpoint, handle.mode.label());
    for note in &handle.notes {
        println!("            {note}");
    }
    println!();

    // Tell the terminal what we are serving, and which tab asked for it, so
    // the tab can be named after its model.
    announce(&source.label(), &signature, adapter.as_deref(), &handle);

    // The model has to be stopped when the terminal goes away.
    install_signal_handlers();
    if let Ok(mut guard) = SERVED.try_lock() {
        *guard = Some(handle);
    }

    talk()
}

/// Work out what to serve: the weights, and the adapter that goes with them.
///
/// The fast path is the one that is already proven: a model whose weights are
/// a GGUF, with a Heretic adapter for that shape bound in. Everything else
/// needs something built first.
fn prepare(
    source: &ModelSource,
    signature: &Option<hermetic_core::Signature>,
    registry: &Registry,
    spec_arg: &str,
) -> Result<(Option<PathBuf>, Option<PathBuf>, bool)> {
    let Some(signature) = signature else {
        // A server that already holds the weights: there is nothing to serve.
        return Ok((None, None, false));
    };

    // A model that does not answer questions cannot be talked to. Ollama
    // says what a model is for - "completion", "embedding", "speech",
    // "vision" - and a model without "completion" has no chat endpoint for
    // an adapter to be bound to. Saying so is better than an hour of
    // looking for weights that will never be served.
    let capabilities = spec::capabilities_of(source);
    let talks = capabilities
        .iter()
        .any(|c| c == "completion" || c == "chat");
    if !capabilities.is_empty() && !talks {
        let kind = if capabilities.iter().any(|c| c == "speech") {
            "a speech model"
        } else if capabilities.iter().any(|c| c == "embedding") {
            "an embedding model"
        } else {
            "not a chat model"
        };
        anyhow::bail!(
            "{spec_arg} is {kind}, not a chat model.\n\
             HERMETIC serves models you talk to: this one has no chat endpoint for an \
             adapter to be bound to."
        );
    }

    let base = weights::weights_for(source, registry);
    let adapter = registry.adapter_for(signature);

    // A model whose attention output is not square - Gemma 2 and its family
    // - cannot have an adapter bound to it: llama.cpp reshapes that tensor
    // when it loads the model, and the adapter no longer fits it. The
    // adapter is folded into the weights instead.
    let fold = base
        .as_ref()
        .map(|path| hermetic_core::gguf::has_rectangular_attention(path))
        .unwrap_or(false);

    // The fast path: the model we have, with the adapter bound to it.
    if let (Some(base), Some(adapter)) = (&base, &adapter) {
        if !fold {
            println!("  weights   {}", base.display());
            println!("  adapter   {} (ready)", adapter.display());
            return Ok((Some(base.clone()), Some(adapter.clone()), false));
        }
        println!(
            "  note      this model has a rectangular attention output, so an adapter"
        );
        println!("            cannot be bound to it; the adapter is folded into the");
        println!("            weights instead");
    }

    // A model with the adapter folded in, if we have built one.
    if let Some(merged) = registry.merged_for(signature) {
        println!(
            "  weights   {} (the adapter is folded in)",
            merged.display()
        );
        return Ok((Some(merged), None, true));
    }

    // Something is missing: build it.
    build(source, signature, registry, spec_arg, base, fold)
}

/// Build what is missing: an adapter, or a model with the adapter in it.
fn build(
    source: &ModelSource,
    signature: &hermetic_core::Signature,
    registry: &Registry,
    spec_arg: &str,
    base: Option<PathBuf>,
    fold: bool,
) -> Result<(Option<PathBuf>, Option<PathBuf>, bool)> {
    // An earlier build may have left the PEFT adapter behind. It names the
    // model it was made for, so the hub does not have to be asked again - and
    // asking the hub is the slow part, the thing that makes a screen look
    // stuck.
    if let Some(donor) = registry.donor_of(signature) {
        if registry.peft_adapter_for(signature).is_some() {
            println!("  reusing   the adapter an earlier build made for {donor}");
            println!("            the optimiser has already done its work, so this only");
            println!("            converts it and starts the server");
            return build_with(signature, registry, &[donor], base, fold);
        }
    }

    let hint = spec::hint_of(source, signature);
    println!("  looking   for the original weights of this shape");
    println!("            the hub is asked for every model with the same shape, and");
    println!("            each one is checked. This takes a minute on a cold search.");
    let donor = hub::find_donor(signature, spec_arg, &hint);
    let Some(repo) = donor.repo.clone() else {
        // A search that was rate-limited found nothing, and "nothing found"
        // is a lie. If the hub did not answer, that is what is said.
        if let Some(problem) = donor.problem.as_ref().and_then(|p| p.message()) {
            anyhow::bail!(problem);
        }
        let mut message = format!(
            "no model with the shape {} was found, so no adapter can be built for it.\n\
             HERMETIC can only abliterate a model it can find the original weights for.",
            signature.key()
        );
        if !donor.gated.is_empty() {
            message.push_str(&format!(
                "\n\nThese repositories have that shape but Hugging Face will not serve them \
                 without a token:\n  {}\n\nSet HF_TOKEN to one that has access and try again.",
                donor.gated.join("\n  ")
            ));
        }
        anyhow::bail!(message);
    };

    // The repositories with the right shape, best first. A donor can fail -
    // the weights might be in a form Heretic cannot read - and then the next
    // one is tried.
    let mut donors = vec![repo];
    donors.extend(donor.alternatives.clone());
    build_with(signature, registry, &donors, base, fold)
}

/// Build with a list of donors, best first.
///
/// A donor can fail - the weights might be in a form Heretic cannot read -
/// and then the next one with the same shape is tried.
fn build_with(
    signature: &hermetic_core::Signature,
    registry: &Registry,
    donors: &[String],
    base: Option<PathBuf>,
    fold: bool,
) -> Result<(Option<PathBuf>, Option<PathBuf>, bool)> {
    let config = hermetic_core::config::Config::load();
    let socket = std::env::var_os(SOCKET_ENV).map(PathBuf::from);

    let mut last: Option<anyhow::Error> = None;
    for repo in donors {
        println!("  building  an adapter for {repo}");
        println!("            this downloads the original weights and runs Heretic over every");
        println!("            layer of them - for a large model that is many hours and tens");
        println!("            of gigabytes. Ctrl-C stops it; the panel shows the progress");

        let attempt = match base.clone().filter(|_| !fold) {
            Some(base) => registry
                .build(
                    signature,
                    repo,
                    &format!("adapter for {repo}"),
                    &config.python,
                    &config.bridge_dir(),
                    &config.llama_cpp_dir,
                    socket.as_deref(),
                )
                .map(|built| (Some(base), Some(built), false)),
            // There is no GGUF of this shape, so the merged build is the
            // model: the donor's weights with the adapter folded in. That is
            // a GGUF, and it can be served.
            None => registry
                .build_merged(
                    signature,
                    repo,
                    &format!("merged build of {repo}"),
                    &config.python,
                    &config.bridge_dir(),
                    &config.llama_cpp_dir,
                    socket.as_deref(),
                )
                .map(|merged| (Some(merged), None, true)),
        };

        match attempt {
            Ok((weights, adapter, merged)) => {
                if let Some(adapter) = &adapter {
                    println!("  adapter   {}", adapter.display());
                }
                if merged {
                    println!(
                        "  weights   {} (the adapter is folded in)",
                        weights
                            .as_ref()
                            .map(|w| w.display().to_string())
                            .unwrap_or_default()
                    );
                }
                return Ok((weights, adapter, merged));
            }
            Err(error) => {
                println!("  that repository could not be read: {error}");
                println!("            trying the next one with the same shape");
                last = Some(error);
            }
        }
    }

    match last {
        Some(error) => Err(error),
        None => anyhow::bail!("no donor with the shape {} could be built", signature.key()),
    }
}

/// Serve the model with the runtime the planner picked.
///
/// The planner's choice comes first. The others are tried only when it
/// cannot serve what there is - llama.cpp cannot read safetensors, and a
/// model Ollama keeps as safetensors has none.
fn serve(
    plan: &Plan,
    request: &ServeRequest,
    runtimes: &[&dyn ModelRuntime],
) -> Result<ServeHandle> {
    let mut tried: Vec<&dyn ModelRuntime> = Vec::new();
    if let Some(runtime) = runtime::find(runtimes, &plan.runtime) {
        tried.push(runtime);
    }
    for runtime in runtimes {
        if !tried.iter().any(|r| r.id() == runtime.id()) {
            tried.push(*runtime);
        }
    }

    let mut failures = Vec::new();
    for runtime in tried {
        match runtime.serve(request) {
            Ok(handle) => return Ok(handle),
            Err(error) => failures.push(format!("{}: {error}", runtime.name())),
        }
    }
    anyhow::bail!(
        "no runtime could serve {} ({})",
        request.source.label(),
        failures.join("; ")
    )
}

/// Talk to the model that is being served.
fn talk() -> Result<()> {
    let (endpoint, model, mode) = {
        let guard = SERVED
            .lock()
            .map_err(|_| anyhow::anyhow!("the served model is gone"))?;
        let handle = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("the model was never served"))?;
        (handle.endpoint.clone(), handle.model.clone(), handle.mode)
    };

    let mut chat = Chat::new(&endpoint, &model)?;
    if mode == ServeMode::PromptPolicy {
        chat = chat.with_policy(Policy::for_model(&model));
    }

    println!("  /help for commands, /exit to leave");
    println!();

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        print!("\x1b[36m▸\x1b[0m ");
        io::stdout().flush().ok();

        let Some(line) = lines.next() else {
            println!();
            return Ok(());
        };
        let line = line.context("read a line")?;
        let line = line.trim();

        match line {
            "" => continue,
            "/exit" | "/quit" => {
                println!();
                return Ok(());
            }
            "/help" => {
                help();
                continue;
            }
            "/clear" => {
                chat.clear();
                println!("  (cleared)");
                continue;
            }
            "/model" => {
                match chat.model_info() {
                    Ok(answer) => println!("  {answer}"),
                    Err(error) => println!("  could not ask the server: {error}"),
                }
                continue;
            }
            "/models" => {
                list();
                continue;
            }
            other if other.starts_with("/system ") => {
                let text = other.trim_start_matches("/system ").trim();
                chat.history.clear();
                chat.history
                    .push(json!({"role": "system", "content": text}));
                println!("  (system prompt set)");
                continue;
            }
            _ => {}
        }

        chat.history.push(json!({"role": "user", "content": line}));

        let mut wrapper = Wrapper::new(terminal_width().saturating_sub(2));
        let result = chat.answer_with(|delta| match delta {
            Delta::Content(text) => {
                let out = wrapper.push(text);
                print!("{out}");
                io::stdout().flush().ok();
            }
            Delta::Reasoning(text) => {
                let out = wrapper.push(text);
                print!("\x1b[2m{out}\x1b[0m");
                io::stdout().flush().ok();
            }
        });
        let rest = wrapper.flush();
        if !rest.is_empty() {
            print!("{rest}");
        }

        if let Err(error) = result {
            println!();
            println!("  (the model could not be reached: {error})");
        }
        println!();
    }
}

/// The list of commands.
fn help() {
    println!("  /help              this list");
    println!("  /exit              leave the model");
    println!("  /clear             forget the conversation");
    println!("  /system <text>     set the system prompt");
    println!("  /model             what the server reports");
    println!("  /models            what this machine has");
}

/// Wraps the model's output on word boundaries.
///
/// A terminal wraps on character boundaries, which splits words in half and
/// makes the answer hard to read. The chat knows how wide the pane is, so it
/// can break the lines itself.
struct Wrapper {
    width: usize,
    pending: String,
}

impl Wrapper {
    fn new(width: usize) -> Wrapper {
        Wrapper {
            width: width.max(20),
            pending: String::new(),
        }
    }

    /// Add text and return the part that can be printed now.
    fn push(&mut self, text: &str) -> String {
        self.pending.push_str(text);
        let mut out = String::new();
        loop {
            let count = self.pending.chars().count();
            if count < self.width {
                break;
            }
            // The last space before the edge, if there is one.
            let mut cut = None;
            for (index, (offset, ch)) in self.pending.char_indices().enumerate() {
                if index >= self.width {
                    break;
                }
                if ch == ' ' {
                    cut = Some(offset);
                }
            }
            match cut {
                Some(offset) => {
                    out.push_str(&self.pending[..offset]);
                    out.push('\n');
                    self.pending = self.pending[offset + 1..].to_string();
                }
                None => {
                    // No space: break where we are.
                    let offset = self
                        .pending
                        .char_indices()
                        .nth(self.width)
                        .map(|(index, _)| index)
                        .unwrap_or(self.pending.len());
                    out.push_str(&self.pending[..offset]);
                    out.push('\n');
                    self.pending = self.pending[offset..].to_string();
                }
            }
        }
        out
    }

    /// Whatever is left over.
    fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

/// The width of the terminal the chat is running in.
fn terminal_width() -> usize {
    unsafe {
        let mut size: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut size) == 0 && size.ws_col > 0 {
            return size.ws_col as usize;
        }
    }
    100
}

/// Say what we are serving, and which tab asked for it.
fn announce(
    model: &str,
    signature: &Option<hermetic_core::Signature>,
    adapter: Option<&Path>,
    handle: &ServeHandle,
) {
    if ANNOUNCED.swap(true, Ordering::SeqCst) {
        return;
    }
    let Some(socket) = std::env::var_os(SOCKET_ENV).map(PathBuf::from) else {
        return;
    };
    let tab = std::env::var(TAB_ENV)
        .ok()
        .and_then(|t| t.parse::<u64>().ok());
    let event = json!({
        "event": "serving",
        "model": model,
        "tab": tab,
        "shape": signature.as_ref().map(|s| s.describe()),
        "adapter": adapter.map(|a| a.display().to_string()),
        "endpoint": handle.endpoint,
    });
    let _ = channel::send_event(&socket, &event);
}

/// The banner.
fn banner() {
    println!();
    println!(
        "\x1b[1;36m  HERMETIC\x1b[0m \x1b[2m· resolving the model and binding its adapter\x1b[0m"
    );
}

/// Kill the server when the terminal goes away.
fn install_signal_handlers() {
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            handle_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGHUP,
            handle_signal as *const () as libc::sighandler_t,
        );
    }
}

extern "C" fn handle_signal(_signal: libc::c_int) {
    // The lock is only tried, never waited for: a signal handler that waits
    // for a lock can deadlock.
    if let Ok(guard) = SERVED.try_lock() {
        if let Some(handle) = guard.as_ref() {
            (handle.stop)();
        }
    }
    unsafe {
        libc::_exit(130);
    }
}
