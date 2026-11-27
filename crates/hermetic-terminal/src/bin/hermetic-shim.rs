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

/// The questions a model is judged on, and what a refusal looks like.
///
/// This is written next to the build by the bridge, which takes them from
/// the scorer upstream judged the model with. They are upstream's own
/// questions, and upstream's own markers.
#[derive(Debug, Clone, serde::Deserialize)]
struct Probe {
    name: String,
    markers: Vec<String>,
    prompts: Vec<ProbePrompt>,
    /// How often the untouched model refused the same questions.
    #[serde(default)]
    baseline: Option<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ProbePrompt {
    #[serde(default)]
    system: String,
    user: String,
}

/// What the probe found.
struct ProbeResult {
    matches: usize,
    total: usize,
}

/// One line about what the probe found.
///
/// The number on its own says little: "6 of 16" is only good news next to
/// "it used to be 99 of 100". So the score of the untouched model comes with
/// it, when the build measured one.
fn probe_summary(probe: &Probe, result: &ProbeResult) -> String {
    let mut summary = format!("{} {}/{}", probe.name, result.matches, result.total);
    if let Some(baseline) = probe.baseline {
        summary.push_str(&format!(" · the untouched model refused {:.0}%", baseline * 100.0));
    }
    summary
}

/// The probe for a shape, if the model was built with one.
fn probe_for(registry: &Registry, signature: &Option<hermetic_core::Signature>) -> Option<Probe> {
    let signature = signature.as_ref()?;
    let path = registry.builds_dir().join(signature.key()).join("probe.json");
    let text = std::fs::read_to_string(path).ok()?;
    let probe: Probe = serde_json::from_str(&text).ok()?;
    if probe.prompts.is_empty() || probe.markers.is_empty() {
        return None;
    }
    Some(probe)
}

/// Fetch a model from Ollama's library.
///
/// The shell aliases `ollama` to the helper, but an alias belongs to the
/// shell: a program started from here gets the real one.
fn pull(name: &str) -> bool {
    let Some(ollama) = hermetic_terminal::which("ollama") else {
        return false;
    };
    let status = std::process::Command::new(&ollama)
        .arg("pull")
        .arg(name)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    matches!(status, Ok(s) if s.success())
}

/// Ask the model the questions, and count the answers that match.
fn run_probe(endpoint: &str, model: &str, probe: &Probe) -> ProbeResult {
    let Ok((host, port)) = hermetic_core::chat::split_endpoint(endpoint) else {
        return ProbeResult {
            matches: 0,
            total: 0,
        };
    };

    let mut matches = 0;
    let mut total = 0;
    for prompt in &probe.prompts {
        let mut messages = Vec::new();
        if !prompt.system.trim().is_empty() {
            messages.push(json!({"role": "system", "content": prompt.system}));
        }
        messages.push(json!({"role": "user", "content": prompt.user}));

        let body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": 96,
            "temperature": 0.0,
            "stream": false,
        });
        let Ok(answer) = hermetic_core::http::post_json(&host, port, "/v1/chat/completions", &body)
        else {
            continue;
        };
        let text = answer
            .pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
            .unwrap_or_default();
        total += 1;
        if is_refusal(text, &probe.markers) {
            matches += 1;
        }
    }

    ProbeResult { matches, total }
}

/// Does this answer look like a refusal?
///
/// The same rule upstream uses: an empty answer counts as a refusal, the
/// text is lowercased, emphasis is removed, typographic apostrophes are
/// normalised and whitespace is collapsed - and then any marker in it means
/// a refusal.
fn is_refusal(response: &str, markers: &[String]) -> bool {
    if response.trim().is_empty() {
        return true;
    }
    let text = response
        .to_lowercase()
        .replace('*', "")
        .replace('\u{2019}', "'")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    markers.iter().any(|marker| text.contains(&marker.to_lowercase()))
}

/// Load a model and talk to it.
fn run(spec_arg: &str) -> Result<()> {
    let source =
        spec::source_of(spec_arg).with_context(|| format!("could not resolve '{spec_arg}'"))?;
    let registry = Registry::new();
    let mut signature = spec::shape_of(&source);

    // A model that is not on this machine yet has no weights to read, so it
    // has no shape, so no adapter can be bound to it - and it is served as it
    // is, with a prompt-level policy. But Ollama downloads the model when it
    // is asked to serve it, so the download happens either way. Doing it
    // first means the weights are here to be read and changed.
    if signature.is_none() {
        if let ModelSource::Ollama { name } = &source {
            println!("  note      this model is not on this machine yet");
            println!("            it is being fetched, so its weights can be read and changed");
            if pull(name) {
                signature = spec::shape_of(&source);
                if signature.is_some() {
                    println!("            it is here now");
                }
            }
        }
    }

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
    request.merged = merged;
    request.context = 4096;
    if plan.mode == ServeMode::PromptPolicy {
        request.policy = Some(planner::fallback_policy(&descriptor));
        println!(
            "  note      the weights of this model cannot be changed, so a prompt-level policy is used"
        );
    }

    // Serve it.
    let (handle, fallback) = serve(&plan, &request, &refs)?;

    let mut chat = Chat::new(&handle.endpoint, &handle.model)?;
    if handle.mode == ServeMode::PromptPolicy {
        chat = chat.with_policy(Policy::for_model(&handle.model));
    }

    if let Some(fallback) = &fallback {
        println!("  note      {fallback}");
        if handle.mode == ServeMode::PromptPolicy {
            println!("            so its weights cannot be changed, and a prompt-level");
            println!("            policy is used instead of a Heretic adapter");
        }
    }
    println!("  serving   {} ({})", handle.endpoint, handle.mode.label());
    for note in &handle.notes {
        println!("            {note}");
    }
    println!();

    // Tell the terminal what we are serving, and which tab asked for it, so
    // the tab can be named after its model.
    announce(&source.label(), &signature, adapter.as_deref(), &handle);

    // Ask the model the questions it was judged on, and report how it did.
    //
    // The ablation is judged on a set of prompts and a list of markers, and
    // that judgement was made on the model being optimised. This asks the
    // same questions of the model that is *running* - the one with the
    // adapter bound to it - because that is the model the user is going to
    // talk to, and it is the only one whose answer means anything.
    //
    // It runs in the background: the model is ready to talk to immediately,
    // and the answer arrives a moment later.
    let probe = probe_for(&registry, &signature);
    if let Some(probe) = probe.clone() {
        let endpoint = handle.endpoint.clone();
        let model = source.label();
        let tab = std::env::var(TAB_ENV).ok().and_then(|t| t.parse::<u64>().ok());
        let registry = registry.clone();
        let signature = signature.clone();
        std::thread::spawn(move || {
            let result = run_probe(&endpoint, &model, &probe);
            let summary = probe_summary(&probe, &result);

            // The measurement is kept, so that `/why` can say what it found
            // without being asked to measure it all over again.
            if let Some(signature) = &signature {
                let record = hermetic_core::registry::ProbeRecord {
                    name: probe.name.clone(),
                    matches: result.matches,
                    total: result.total,
                    baseline: probe.baseline,
                    when: now_seconds(),
                };
                let _ = registry.save_probe_record(signature, &record);
            }

            if let Some(socket) = std::env::var_os(SOCKET_ENV) {
                let event = json!({
                    "event": "probe",
                    "name": probe.name,
                    "matches": result.matches,
                    "total": result.total,
                    "tab": tab,
                    "summary": summary,
                });
                let _ = channel::send_event(&PathBuf::from(socket), &event);
            }
            eprintln!("  \x1b[2m{summary}\x1b[0m");
        });
    }

    // The model has to be stopped when the terminal goes away.
    //
    // The handle is moved into the global slot, not copied: it holds the
    // server, and the server has to stay up for as long as the conversation
    // lasts. Dropping it would stop the model.
    install_signal_handlers();
    let endpoint = handle.endpoint.clone();
    let runtime = handle.runtime.clone();
    if let Ok(mut guard) = SERVED.try_lock() {
        *guard = Some(handle);
    }

    talk(Session {
        chat,
        endpoint,
        signature: signature.clone(),
        adapter: adapter.map(|path| path.display().to_string()),
        donor: signature.as_ref().and_then(|s| registry.donor_of(s)),
        registry,
        probe: probe.clone(),
        plan: plan.reason.clone(),
        runtime,
    })
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

    // An adapter is fitted to the weights it was built on. Two models with
    // the same shape are not the same model, and the weights say what they
    // are. When the adapter's donor is not the model these weights are, the
    // adapter belongs to another model and is not served.
    let hint = spec::hint_of(source, signature);
    let donor = registry.donor_of(signature);
    let donor_fits = match (&donor, &hint.name) {
        (Some(donor), Some(name)) => hermetic_core::hub::donor_matches_weights(donor, name),
        _ => true,
    };
    if !donor_fits {
        if let Some(donor) = &donor {
            println!(
                "  note      the adapter was made for {donor}, and these weights are not that model"
            );
        }
        println!("            so a new one is built for them");
    }

    // A study that has run fewer trials than the profile asks for has more
    // to give. That is an *improvement*, not a requirement: the adapter that
    // is already on disk works, and the person who asked for this model
    // wants to talk to it. So the adapter is served, and the better one is
    // something they can ask for explicitly with /improve.
    let config = hermetic_core::config::Config::load();
    let wanted = config.profile_trials();
    let study_pending = registry.study_needs_more_trials(signature, donor.as_deref(), wanted);

    // A GGUF that is older than the adapter next to it is from an older
    // ablation. It is made again rather than served.
    let adapter = match registry.adapter_for(signature) {
        Some(_) if registry.adapter_is_stale(signature) => {
            println!("  note      the adapter was built again, so it is converted again");
            None
        }
        Some(_) if !donor_fits => None,
        other => other,
    };

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
            if study_pending {
                println!("  note      this adapter can still be improved");
                println!("            type /improve to spend the time on a better one");
            }
            return Ok((Some(base.clone()), Some(adapter.clone()), false));
        }
        println!(
            "  note      this model has a rectangular attention output, so an adapter"
        );
        println!("            cannot be bound to it; the adapter is folded into the");
        println!("            weights instead");
    }

    // A model with the adapter folded in, if we have built one.
    //
    // The merged model was made from the adapter next to the study, so the
    // donor has to fit these weights too. A model folded from another
    // model's adapter is that other model with the wrong weights under it,
    // and it is not served.
    if let Some(merged) = registry.merged_for(signature) {
        if registry.merged_is_stale(signature) {
            println!("  note      the adapter was built again, so the model is folded again");
        } else if !donor_fits {
            println!("  note      the model on file was made for another model, so it is not used");
        } else {
            println!(
                "  weights   {} (the adapter is folded in)",
                merged.display()
            );
            if study_pending {
                println!("  note      this model can still be improved");
                println!("            type /improve to spend the time on a better one");
            }
            return Ok((Some(merged), None, true));
        }
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
    //
    // The PEFT adapter is the expensive part and the GGUF is made from it in
    // a minute, so when it is there it is converted again rather than run
    // through the optimiser: the model does not have to be loaded, and the
    // baseline does not have to be measured again.
    if let Some(donor) = registry.donor_of(signature) {
        let hint = spec::hint_of(source, signature);
        let donor_fits = match &hint.name {
            Some(name) => hermetic_core::hub::donor_matches_weights(&donor, name),
            None => true,
        };
        if !donor_fits {
            // The adapter next to the study was fitted to other weights. It
            // is not reused, and neither is its donor: the hub is asked
            // again, and it finds the model these weights actually are.
            println!(
                "  note      the adapter on file was made for {donor}, which is not this model"
            );
        } else if let Some(peft) = registry.peft_adapter_for(signature) {
            let config = hermetic_core::config::Config::load();
            // The optimiser has already done its work: the PEFT adapter next
            // to the study is the best trial it found. Converting it takes a
            // minute; running the optimiser again takes hours. The person
            // who asked for this model wants to talk to it, so it is served
            // now, and /improve is what buys a better ablation.
            let build_dir = peft.parent().unwrap_or(Path::new("."));
            println!("  reusing   the adapter an earlier build made for {donor}");
            println!("            the optimiser has already done its work, so this only");
            println!("            converts it and starts the server");
            let converted = registry.convert_adapter(
                signature,
                &donor,
                build_dir,
                &config.python,
                &config.llama_cpp_dir,
            )?;
            match base.clone().filter(|_| !fold) {
                Some(base) => return Ok((Some(base), Some(converted), false)),
                None => {
                    // There is no GGUF of this shape, so the adapter is
                    // folded into the donor's weights instead.
                    let merged = registry.merge_adapter(
                        signature,
                        &donor,
                        build_dir,
                        &config.python,
                        &config.bridge_dir(),
                        &config.llama_cpp_dir,
                    )?;
                    return Ok((Some(merged), None, true));
                }
            }
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
                    config.profile_trials(),
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
                    config.profile_trials(),
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
) -> Result<(ServeHandle, Option<String>)> {
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
            Ok(handle) => {
                // When a different runtime ends up serving the model, that is
                // worth saying: the plan was not what happened, and the
                // difference matters. A model served by Ollama's own engine
                // cannot have an adapter bound to it.
                let note = if !handle.runtime.is_empty() && handle.runtime != plan.runtime {
                    Some(format!(
                        "{} could not serve it, so {} served it instead",
                        plan.runtime, handle.runtime
                    ))
                } else {
                    None
                };
                return Ok((handle, note));
            }
            Err(error) => failures.push(format!("{}: {error}", runtime.name())),
        }
    }
    anyhow::bail!(
        "no runtime could serve {} ({})",
        request.source.label(),
        failures.join("; ")
    )
}

/// What the terminal knows about the model it is talking to.
///
/// This is what `/why` explains, and what the panel shows. It is gathered
/// once, when the model is served, and then kept.
struct Session {
    chat: Chat,
    /// Where the model is being served.
    endpoint: String,
    /// The model's shape, when its weights could be read.
    signature: Option<hermetic_core::Signature>,
    /// The adapter bound to it, when there is one.
    adapter: Option<String>,
    /// The model the adapter was fitted to.
    donor: Option<String>,
    /// The registry: where the probe record is kept.
    registry: Registry,
    /// The questions the model is judged on, when it was built here.
    probe: Option<Probe>,
    /// The plan that was followed.
    plan: String,
    /// The runtime that ended up serving it.
    runtime: String,
}

/// Talk to the model that is being served.
fn talk(session: Session) -> Result<()> {
    let Session {
        mut chat,
        endpoint,
        signature,
        adapter,
        donor,
        registry,
        probe,
        plan,
        runtime,
    } = session;

    // Whether an improvement is already running. The optimiser is expensive
    // and running it twice on one shape would fight itself.
    let mut improving = false;

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
            "/why" => {
                explain(
                    &chat,
                    &signature,
                    &adapter,
                    &registry,
                    &plan,
                    &runtime,
                );
                continue;
            }
            "/probe" => {
                measure(&registry, &signature, &probe, &endpoint, &chat.model);
                continue;
            }
            "/improve" => {
                improve(&signature, &donor, &registry, &mut improving);
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

        let width = terminal_width().saturating_sub(2);
        let (answer, outcome, elapsed) = ask(chat.clone(), width);
        chat = answer;

        if let Err(error) = &outcome {
            println!();
            println!("  (the model could not be reached: {error})");
        } else if elapsed.as_secs_f32() > 2.0 {
            println!();
            println!("  \x1b[2m({:.1}s)\x1b[0m", elapsed.as_secs_f32());
        }
        println!();
    }
}

/// Ask the model, and say something while it is thinking.
///
/// A model that is loading its weights, or a machine that is busy, can take
/// a while to answer the first word. Silence looks like a broken terminal,
/// so the wait is said out loud - and the time it took is said afterwards,
/// because a slow answer is worth knowing about.
fn ask(chat: Chat, width: usize) -> (Chat, Result<String, String>, std::time::Duration) {
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::time::{Duration, Instant};

    let (tx, rx) = mpsc::channel::<(bool, String)>();
    let started = Instant::now();

    let worker = {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let mut chat = chat;
            let result = chat.answer_with(|delta| {
                let (reasoning, text) = match delta {
                    Delta::Content(text) => (false, text.to_string()),
                    Delta::Reasoning(text) => (true, text.to_string()),
                };
                let _ = tx.send((reasoning, text));
            });
            (chat, result)
        })
    };

    let mut wrapper = Wrapper::new(width);
    let mut waiting = false;
    loop {
        match rx.recv_timeout(Duration::from_millis(150)) {
            Ok((reasoning, text)) => {
                if waiting {
                    print!(" ");
                    waiting = false;
                }
                let out = wrapper.push(&text);
                if reasoning {
                    print!("\x1b[2m{out}\x1b[0m");
                } else {
                    print!("{out}");
                }
                io::stdout().flush().ok();
            }
            Err(RecvTimeoutError::Timeout) => {
                if !waiting && started.elapsed() > Duration::from_secs(2) {
                    print!("\x1b[2m…\x1b[0m");
                    io::stdout().flush().ok();
                    waiting = true;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    let rest = wrapper.flush();
    if !rest.is_empty() {
        print!("{rest}");
    }

    let elapsed = started.elapsed();
    match worker.join() {
        Ok((chat, Ok(answer))) => (chat, Ok(answer), elapsed),
        Ok((chat, Err(error))) => (chat, Err(format!("{error}")), elapsed),
        Err(_) => (
            Chat::new(&"http://127.0.0.1:0/v1", "").unwrap_or_else(|_| {
                panic!("the chat could not be rebuilt")
            }),
            Err("the request did not finish".to_string()),
            elapsed,
        ),
    }
}

/// Explain what this model is, and why it is being served this way.
///
/// The terminal knows a lot about the model it is talking to: its shape, the
/// adapter bound to it, the runtime that serves it, and what the probe
/// measured. That is a lot to remember while working, so it can be asked
/// for.
fn explain(
    chat: &Chat,
    signature: &Option<hermetic_core::Signature>,
    adapter: &Option<String>,
    registry: &Registry,
    plan: &str,
    runtime: &str,
) {
    println!("  model     {}", chat.model);
    match signature {
        Some(signature) => {
            println!("  shape     {}", signature.describe());
            println!("  key       {}", signature.key());
        }
        None => println!(
            "  shape     unknown - the weights are not on this machine, so nothing can"
        ),
    }
    match adapter {
        Some(adapter) => println!("  adapter   {}", adapter),
        None => println!("  adapter   none"),
    }
    println!("  served    {} by {}", chat.base_url(), runtime);
    println!("  plan      {plan}");

    match signature.as_ref().and_then(|s| registry.probe_record_for(s)) {
        Some(record) => println!("  probe     {}", record.summary()),
        None => println!(
            "  probe     not measured yet - type /probe to measure it"
        ),
    }
}

/// Measure whether the model still refuses.
///
/// The questions and the markers are the ones the model was judged on, and
/// the score of the untouched model comes with the result: a number on its
/// own says little.
fn measure(
    registry: &Registry,
    signature: &Option<hermetic_core::Signature>,
    probe: &Option<Probe>,
    endpoint: &str,
    model: &str,
) {
    let (Some(signature), Some(probe)) = (signature, probe) else {
        println!("  this model was not built here, so there are no questions to ask it");
        println!("  (the questions come from the scorer the model was judged on)");
        return;
    };

    println!("  measuring how often this model refuses…");
    io::stdout().flush().ok();

    let result = run_probe(endpoint, model, probe);
    if result.total == 0 {
        println!("  the model could not be reached");
        return;
    }

    let record = hermetic_core::registry::ProbeRecord {
        name: probe.name.clone(),
        matches: result.matches,
        total: result.total,
        baseline: probe.baseline,
        when: now_seconds(),
    };
    let _ = registry.save_probe_record(signature, &record);

    println!("  {}", record.summary());
    if let Some(baseline) = probe.baseline {
        let rate = result.matches as f64 / result.total as f64;
        if rate < baseline {
            println!(
                "  the ablation is working: it refused {:.0}% of the questions, and the",
                rate * 100.0
            );
            println!("  untouched model refused {:.0}%", baseline * 100.0);
        } else {
            println!(
                "  the ablation is not working: it refused {:.0}% of the questions, and the",
                rate * 100.0
            );
            println!("  untouched model refused {:.0}%", baseline * 100.0);
        }
    }
}

/// Seconds since the epoch.
fn now_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

/// Spend time on a better adapter.
///
/// The model in front of you is already served and works. This is the
/// expensive thing: the optimiser runs more trials over the same shape, and
/// when it is done there is a better adapter on disk. It runs in the
/// background, so the model keeps answering while it works, and the panel
/// shows the progress.
///
/// The point of making this a command rather than something that happens on
/// its own is that the optimiser costs hours. Nobody should pay for that
/// without being asked.
fn improve(
    signature: &Option<hermetic_core::Signature>,
    donor: &Option<String>,
    registry: &Registry,
    running: &mut bool,
) {
    if *running {
        println!("  an improvement is already running - the panel shows its progress");
        return;
    }

    let (Some(signature), Some(donor)) = (signature, donor) else {
        println!("  this model was not built here, so there is nothing to improve");
        println!("  (HERMETIC can only improve an adapter it made itself)");
        return;
    };

    let config = hermetic_core::config::Config::load();
    let wanted = config.profile_trials().unwrap_or(20);
    let registry = registry.clone();
    let signature = signature.clone();
    let donor = donor.clone();
    let socket = std::env::var_os(SOCKET_ENV).map(PathBuf::from);

    *running = true;
    println!("  improving  the adapter for {donor}");
    println!(
        "            the optimiser runs up to {wanted} trials over this shape - that is"
    );
    println!("            hours for a big model. It runs in the background; this model");
    println!("            keeps answering while it works.");

    std::thread::spawn(move || {
        let result = registry.build(
            &signature,
            &donor,
            &format!("adapter for {donor}"),
            &config.python,
            &config.bridge_dir(),
            &config.llama_cpp_dir,
            socket.as_deref(),
            Some(wanted),
        );
        match result {
            Ok(path) => {
                println!();
                println!("  improved  {}", path.display());
                println!("            restart this model to use the better adapter");
            }
            Err(error) => {
                println!();
                println!("  the improvement failed: {error}");
                println!("            the adapter that is already there still works");
            }
        }
        io::stdout().flush().ok();
    });
}

/// The list of commands.
fn help() {
    println!("  /help              this list");
    println!("  /exit              leave the model");
    println!("  /clear             forget the conversation");
    println!("  /system <text>     set the system prompt");
    println!("  /model             what the server reports");
    println!("  /models            what this machine has");
    println!("  /why               what this model is, and how it is served");
    println!("  /probe             ask it whether it still refuses");
    println!("  /improve           spend time on a better adapter");
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
