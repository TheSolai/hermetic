//! The Hugging Face hub: finding the original weights of a model.
//!
//! This is the part that makes the terminal work on models whose names do
//! not match their weights. A distilled `deepseek-r1:8b` is really a
//! `Qwen3-8B`; the name is a lie and the shape is not.
//!
//! So the search is driven by the shape and the parameter count, not by the
//! name:
//!
//!   1. The obvious name - the architecture and the size: `Qwen/Qwen3-8B`.
//!   2. The hub's search, filtered by the parameter count, which is exact
//!      enough to leave a handful of candidates.
//!   3. The shape, checked to the digit, because an adapter's tensors have
//!      to line up exactly.

use std::process::Command;

use anyhow::Result;
use serde_json::Value;

use crate::signature::Signature;

/// The Hugging Face token, if the user has one.
///
/// Some model families - Meta's Llama, Google's gated Gemma - will not serve
/// their weights without one.
pub fn token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|t| !t.trim().is_empty())
}

/// The extra arguments curl needs, with the token if there is one.
fn auth_args() -> Vec<String> {
    let mut args = Vec::new();
    if let Some(token) = token() {
        args.push("-H".to_string());
        args.push(format!("Authorization: Bearer {token}"));
    }
    args
}

/// The arguments every request to the hub is made with.
///
/// The hub rate-limits, and a search can bring back a hundred names to check.
/// The retry is not politeness: without it, a rate-limited reply looks like a
/// repository that does not exist, and the donor is lost.
fn curl_args() -> Vec<String> {
    let mut args = vec![
        "-sL".to_string(),
        "--max-time".to_string(),
        "20".to_string(),
        "--retry".to_string(),
        "3".to_string(),
        "--retry-delay".to_string(),
        "2".to_string(),
        "--retry-all-errors".to_string(),
    ];
    args.extend(auth_args());
    args
}

/// Fetch a URL and parse it as JSON.
fn get_json(url: &str) -> Option<Value> {
    let mut command = Command::new("curl");
    for argument in curl_args() {
        command.arg(argument);
    }
    let out = command.arg(url).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str(&text).ok()
}

/// Fetch a URL and read its body and its status code.
fn get_with_status(url: &str) -> Option<(String, String)> {
    let mut command = Command::new("curl");
    for argument in curl_args() {
        command.arg(argument);
    }
    let out = command
        .args(["-w", "\n%{http_code}", url])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let (body, code) = text.rsplit_once('\n')?;
    Some((body.to_string(), code.trim().to_string()))
}

/// What the hub knows about a repository.
#[derive(Debug, Clone, Default)]
pub struct RepoInfo {
    /// Is the repository gated behind a token?
    pub gated: bool,
    /// How many parameters the model has, when the hub knows.
    pub parameters: Option<u64>,
}

/// Ask the hub about a repository.
///
/// The hub answers "gated" plainly - `gated: false`, `gated: "manual"` or
/// `gated: "auto"` - so this is the only reliable way to tell a gated
/// repository from one that simply does not exist. The status code of a
/// download tells the two apart too easily to get wrong: the hub answers
/// `401` for a repository that was never there.
pub fn repo_info(repo: &str) -> Option<RepoInfo> {
    let value = get_json(&format!("https://huggingface.co/api/models/{repo}"))?;
    if value.get("error").is_some() {
        return None;
    }
    let gated = match value.get("gated") {
        Some(Value::Bool(flag)) => *flag,
        Some(Value::String(kind)) => !kind.is_empty() && kind != "false",
        _ => false,
    };
    let parameters = value
        .get("safetensors")
        .and_then(|s| s.get("total"))
        .and_then(|v| v.as_u64());
    Some(RepoInfo { gated, parameters })
}

/// What came back when a repo was asked for its shape.
#[derive(Debug, Clone)]
enum Shape {
    /// The shape, read from the repo's config.
    Known(Signature),
    /// The repo exists but is gated: it needs a token.
    Gated,
    /// There is no such repo, or it could not be read.
    Missing,
}

/// Read a repo's `config.json` and work out its shape.
fn repo_shape(repo: &str, known_to_exist: bool) -> Shape {
    let url = format!("https://huggingface.co/{repo}/resolve/main/config.json");
    let Some((body, code)) = get_with_status(&url) else {
        return Shape::Missing;
    };
    match code.as_str() {
        "200" => {}
        // The download was refused. The hub answers 401 for a repository that
        // does not exist as well as for one that is gated, and without a token
        // its own record cannot tell them apart either. What can tell them
        // apart is the hub's search: a repository it returns exists, and one
        // it never returned does not.
        "401" | "403" => {
            return match repo_info(repo) {
                Some(info) if info.gated => Shape::Gated,
                Some(_) => Shape::Missing,
                None => {
                    if known_to_exist {
                        Shape::Gated
                    } else {
                        Shape::Missing
                    }
                }
            };
        }
        _ => return Shape::Missing,
    }
    let Ok(config) = serde_json::from_str::<Value>(&body) else {
        return Shape::Missing;
    };
    match Signature::from_hf_config(&config) {
        Ok(signature) => Shape::Known(signature),
        Err(_) => Shape::Missing,
    }
}

/// Ask the hub for repositories with a name like `query`.
///
/// The results are sorted by downloads: the model everybody actually uses
/// comes first, and not somebody's re-upload of it.
pub fn search(query: &str, limit: usize) -> Vec<String> {
    let url = format!(
        "https://huggingface.co/api/models?search={}&sort=downloads&limit={limit}",
        query.replace(' ', "+")
    );
    let Some(value) = get_json(&url) else {
        return Vec::new();
    };
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|m| m.get("id").and_then(|i| i.as_str()))
        .map(|s| s.to_string())
        .collect()
}

/// The organisations that publish models of this architecture.
///
/// These are guesses, not facts: the hub's search is the real answer. But a
/// good guess saves a search.
pub fn likely_orgs(arch: &str) -> &'static [&'static str] {
    let arch = arch.to_lowercase();
    if arch.starts_with("qwen") {
        &["Qwen"]
    } else if arch.starts_with("llama") {
        &["meta-llama", "NousResearch", "unsloth"]
    } else if arch.starts_with("gemma") {
        &["google", "unsloth"]
    } else if arch.starts_with("mistral") || arch.starts_with("mixtral") {
        &["mistralai", "unsloth"]
    } else if arch.starts_with("deepseek") {
        &["deepseek-ai"]
    } else if arch.starts_with("phi") {
        &["microsoft"]
    } else if arch.starts_with("falcon") {
        &["tiiuae"]
    } else if arch.starts_with("smollm") {
        &["HuggingFaceTB"]
    } else if arch.starts_with("granite") {
        &["ibm-granite"]
    } else if arch.starts_with("olmo") {
        &["allenai"]
    } else if arch.starts_with("nemotron") {
        &["nvidia"]
    } else if arch.starts_with("cohere") || arch.starts_with("command") {
        &["CohereForAI", "CohereLabs"]
    } else if arch.starts_with("gpt") {
        &["openai"]
    } else if arch.starts_with("exaone") {
        &["LGAI-EXAONE"]
    } else if arch.starts_with("glm") || arch.starts_with("chatglm") {
        &["THUDM", "zai-org"]
    } else if arch.starts_with("intern") {
        &["internlm"]
    } else if arch.starts_with("baichuan") {
        &["baichuan-inc"]
    } else if arch.starts_with("starcode") {
        &["bigcode"]
    } else if arch.starts_with("gpt2") {
        &["openai-community", "EleutherAI"]
    } else if arch.starts_with("mamba") {
        &["state-spaces"]
    } else if arch.starts_with("yi") {
        &["01-ai"]
    } else {
        &[]
    }
}

/// The ways a size label can be spelled in a repository name.
///
/// `8B` is `8B`. `30B-A3B` is both `30B-A3B` and, less usefully, `30B`.
pub fn size_spellings(label: &str) -> Vec<String> {
    let label = label.trim();
    if label.is_empty() {
        return Vec::new();
    }
    let mut out = vec![label.to_string()];
    if let Some((first, rest)) = label.split_once('-') {
        if !rest.is_empty() {
            out.push(first.to_string());
        }
    }
    out
}

/// Repositories that are unlikely to hold the original weights.
///
/// Quantised conversions, re-uploads and other derivatives have the right
/// shape but not the weights Heretic needs.
pub fn is_derivative(repo: &str) -> bool {
    let lower = repo.to_lowercase();
    [
        "-gguf",
        "gguf-",
        "-awq",
        "-gptq",
        "-uncensored",
        "-abliterated",
        "-abl-",
        "-heretic",
        "-imatrix",
        "-exl2",
        "-h2",
        "-e2",
        "-i8",
        "-q4_0",
        "-q8_0",
        "-q4_k",
        "-bf16",
        "-fp8",
        "-tuned",
        "-reasoning",
        // Weights that only another framework can read. Heretic loads the
        // model with transformers, and an MLX conversion is not a set of
        // tensors transformers can open.
        "-mlx",
        "mlx-",
        "-4bit",
        "-8bit",
        "-int4",
        "-nf4",
        "-w4a16",
        "-quantized",
        "-quantised",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

/// What we know about the model we want an adapter for, besides its shape.
#[derive(Debug, Clone, Default)]
pub struct Hint {
    /// How many parameters the model has, exactly.
    pub parameters: Option<u64>,
    /// The architecture, as llama.cpp calls it.
    pub architecture: Option<String>,
    /// How big it is, as a label: `8B`, `30B-A3B`.
    pub size_label: Option<String>,
    /// The name recorded inside the weights themselves.
    ///
    /// The weights know what they are: a file that says it is
    /// `Qwen2-beta-4B-Chat` is one. It is the best hint there is, and the
    /// hub redirects an old name to the repository's new one, so a name that
    /// has been renamed since still finds it.
    pub name: Option<String>,
}

/// What a search for a donor found.
#[derive(Debug, Clone, Default)]
pub struct Donor {
    /// The repo whose weights have the shape we need.
    pub repo: Option<String>,
    /// Repositories that look right but are gated: they need a token.
    pub gated: Vec<String>,
    /// The other repositories with the same shape, in the order they were
    /// found.
    ///
    /// A donor can fail: the weights might be in a form Heretic cannot read,
    /// or the hub might be having a bad afternoon. The next one is then
    /// tried, and the one after that.
    pub alternatives: Vec<String>,
}

impl Donor {
    /// Is there nothing to build from?
    pub fn is_empty(&self) -> bool {
        self.repo.is_none()
    }
}

/// The obvious name: the architecture and the size.
fn name_candidates(hint: &Hint) -> Vec<String> {
    let mut out = Vec::new();
    let Some(arch) = &hint.architecture else {
        return out;
    };
    let family = capitalise(arch);
    for org in likely_orgs(arch) {
        for size in size_spellings(hint.size_label.as_deref().unwrap_or("")) {
            for suffix in ["", "-Base", "-Instruct", "-it"] {
                let repo = format!("{org}/{family}-{size}{suffix}");
                if !out.contains(&repo) {
                    out.push(repo);
                }
            }
        }
    }
    out
}

/// The words that describe a model's *role* rather than its family.
///
/// `qwen2.5-coder` is a `Qwen2.5`; the `coder` is what it does, not what it is.
const ROLE_WORDS: &[&str] = &[
    "coder",
    "instruct",
    "chat",
    "it",
    "base",
    "uncensored",
    "abl",
    "abliterated",
    "heretic",
    "reasoning",
    "thinking",
    "preview",
    "gguf",
    "hf",
];

/// The families an architecture is published under.
///
/// llama.cpp calls every Llama model `llama`, but the hub has `Llama-2`,
/// `Llama-3`, `Llama-3.1` and `Llama-3.2` as separate repositories, and the
/// version is nowhere in the architecture. It has to be guessed, and the
/// shape check decides which guess was right.
fn arch_families(arch: &str) -> Vec<String> {
    let lower = arch.to_lowercase();
    // `qwen35moe` is `qwen35` with a note about the experts.
    let base = lower
        .trim_end_matches("moe")
        .trim_end_matches("_lite")
        .trim_end_matches("lite");
    let word = capitalise(base);
    let mut out = vec![word.clone()];

    // `qwen35` is `Qwen3.5`, and `qwen2` is `Qwen2.5`.
    if let Some(index) = base.find(|c: char| c.is_ascii_digit()) {
        let (letters, digits) = base.split_at(index);
        if digits.len() == 2 {
            let (major, minor) = digits.split_at(1);
            out.push(format!("{}{}.{}", capitalise(letters), major, minor));
        }
        if digits.len() == 1 {
            out.push(format!("{word}.5"));
            out.push(format!("{word}.0"));
        }
    } else {
        // A family without a version in its name is published in several:
        // `Llama`, `Llama-2`, `Llama-3`, `Llama-3.1`.
        for version in ["2", "3", "3.1", "3.2", "3.5", "3.6", "3.8", "4"] {
            out.push(format!("{word}-{version}"));
        }
        if word.starts_with("Deepseek") {
            for version in ["V2", "V3"] {
                out.push(format!("{word}-{version}"));
            }
        }
    }
    out
}

/// The families a model can belong to, spelled the way repositories spell them.
fn family_spellings(spec: &str, hint: &Hint) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |value: String| {
        if !value.is_empty() && !out.contains(&value) {
            out.push(value);
        }
    };

    // The weights know their own name. It is the best hint there is - and it
    // is also often old, which the hub sorts out by redirect.
    if let Some(name) = &hint.name {
        if !name.contains(' ') {
            push(name.clone());
        }
        let trimmed: Vec<&str> = name
            .split('-')
            .filter(|part| !ROLE_WORDS.contains(&part.to_lowercase().as_str()))
            .collect();
        if !trimmed.is_empty() && !trimmed.join("-").contains(' ') {
            push(capitalise(&trimmed.join("-")));
        }
    }

    // The model's own name is the next best thing: `qwen2.5-coder` says
    // `Qwen2.5`, even though llama.cpp only says `qwen2`.
    let (name, _) = split_tag(spec);
    let base = name.rsplit('/').next().unwrap_or(name);
    if !base.is_empty() {
        push(capitalise(base));
        let trimmed: Vec<&str> = base
            .split('-')
            .filter(|part| !ROLE_WORDS.contains(&part.to_lowercase().as_str()))
            .collect();
        if !trimmed.is_empty() {
            push(capitalise(&trimmed.join("-")));
        }
    }

    // The architecture, with every version it hides.
    if let Some(arch) = &hint.architecture {
        for family in arch_families(arch) {
            push(family);
        }
    }

    out
}

/// The sizes a model of this many parameters is called.
///
/// The name is not the number of parameters: Qwen2.5-32B has 32.76 billion
/// of them. So the number below and the number above are both tried, and
/// the shape check decides which one was right.
fn sizes_from_parameters(parameters: u64) -> Vec<String> {
    let billions = parameters as f64 / 1_000_000_000.0;
    let mut out: Vec<String> = Vec::new();
    if billions < 1.0 {
        let tenths = (billions * 10.0).round() as u64;
        if tenths > 0 {
            out.push(format!("{}.{}B", tenths / 10, tenths % 10));
        }
        let millions = (parameters as f64 / 1_000_000.0).round() as u64;
        out.push(format!("{millions}M"));
        return out;
    }
    for value in [billions.floor() as u64, billions.ceil() as u64] {
        out.push(format!("{value}B"));
    }
    // Half sizes are common enough to be worth a guess.
    let half = (billions * 2.0).round() as u64;
    if half % 2 == 1 {
        out.push(format!("{}.5B", half / 2));
    }
    out
}

/// Every size the model might be called.
fn size_candidates(hint: &Hint) -> Vec<String> {
    let mut out = size_spellings(hint.size_label.as_deref().unwrap_or(""));
    if let Some(parameters) = hint.parameters {
        for size in sizes_from_parameters(parameters) {
            if !out.contains(&size) {
                out.push(size);
            }
        }
    }
    out
}

/// The organisation × family × size matrix.
///
/// Names lie — `Qwen:latest` is really `Qwen/Qwen2.5-3B` — but the matrix
/// crosses the two hints we have (the family and the size) with the
/// organisations that publish them. That reaches the donor directly, instead
/// of waiting for the hub's own search to rank it into view.
fn hf_candidates(spec: &str, hint: &Hint) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let families = family_spellings(spec, hint);
    let sizes = size_candidates(hint);
    if families.is_empty() || sizes.is_empty() {
        return out;
    }

    let mut orgs: Vec<String> = likely_orgs(hint.architecture.as_deref().unwrap_or(""))
        .iter()
        .map(|o| o.to_string())
        .collect();
    // A model that came from the hub keeps the organisation it came from.
    let (name, _) = split_tag(spec);
    if let Some((org, _)) = name.rsplit_once('/') {
        if !org.is_empty() && !org.contains('.') {
            orgs.insert(0, org.to_string());
        }
    }

    for org in &orgs {
        // The weights know their own name. The repository may have been
        // renamed since, but the hub redirects an old name to the new one,
        // so the name the file was made with still finds it.
        if let Some(name) = &hint.name {
            if !name.contains(' ') {
                let repo = format!("{org}/{name}");
                if !out.contains(&repo) {
                    out.push(repo);
                }
            }
        }
        for family in &families {
            for size in &sizes {
                for suffix in ["", "-Instruct", "-it"] {
                    let repo = format!("{org}/{family}-{size}{suffix}");
                    if !out.contains(&repo) {
                        out.push(repo);
                    }
                }
            }
        }
    }
    out
}

/// The hub's search, by family, with the parameter count as a filter.
///
/// The count is the filter that matters: a distilled model has the count of
/// the model it was distilled from, and nothing else does.
fn counted_candidates(spec: &str, hint: &Hint) -> Vec<String> {
    let Some(parameters) = hint.parameters else {
        return Vec::new();
    };

    // Every family the model might be published under, and the hub's own
    // search for it.
    let mut queries = family_spellings(spec, hint);
    let (name, _) = split_tag(spec);
    let base = name.rsplit('/').next().unwrap_or(name);
    if !base.is_empty() {
        queries.push(base.to_string());
        queries.push(base.replace('-', " "));
    }

    // The family with the size in the query: `Llama-3-34B`. This is the
    // query that finds the public mirrors of a gated model - the ones that
    // everybody's download count is on, not the original.
    let sizes = size_candidates(hint);
    let mut with_size = Vec::new();
    for family in family_spellings(spec, hint) {
        for size in &sizes {
            with_size.push(format!("{family}-{size}"));
        }
    }
    queries.extend(with_size);

    let mut repos: Vec<String> = Vec::new();
    for query in queries {
        for repo in search(&query, 12) {
            // A re-upload of quantised weights is not a donor: Heretic needs
            // the original tensors, and a GGUF repo does not have them.
            if !is_derivative(&repo) && !repos.contains(&repo) {
                repos.push(repo);
            }
        }
        // The hub rate-limits. Ten queries at a time is plenty.
        if repos.len() > 120 {
            break;
        }
    }

    // The hub is slow one request at a time, and a search brings back twenty
    // names to check, so they are asked about in parallel.
    let mut out = Vec::new();
    for (repo, info) in repo_infos(&repos) {
        let Some(total) = info.and_then(|i| i.parameters) else {
            continue;
        };
        // Two percent: tied embeddings and a slightly different head can move
        // the count by a little.
        let difference = (total as f64 - parameters as f64).abs() / parameters as f64;
        if difference < 0.02 {
            out.push(repo);
        }
    }
    out
}

/// Ask the hub about many repositories at once.
///
/// The hub is slow one request at a time, and a search brings back a dozen
/// names to check - so they are asked about in parallel, but only three at a
/// time, because the hub rate-limits and a rate-limited reply looks like a
/// repository that does not exist.
fn repo_infos(repos: &[String]) -> Vec<(String, Option<RepoInfo>)> {
    let workers = std::thread::available_parallelism()
        .map(|n| n.get().min(3))
        .unwrap_or(2)
        .max(1);
    let chunk = (repos.len() + workers - 1) / workers;
    let mut out: Vec<(String, Option<RepoInfo>)> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for part in repos.chunks(chunk.max(1)) {
            handles.push(scope.spawn(move || {
                part.iter()
                    .map(|repo| (repo.clone(), repo_info(repo)))
                    .collect::<Vec<_>>()
            }));
        }
        for handle in handles {
            match handle.join() {
                Ok(part) => out.extend(part),
                Err(_) => {}
            }
        }
    });
    out
}

/// Ask every candidate for its shape, in parallel, and in order.
///
/// The shape is the thing that decides, so every candidate has to be asked.
/// One request at a time takes a minute for a big search, which is long
/// enough to look like a hang.
fn shapes_of(candidates: &[String], exists: &[String], target: &Signature) -> Vec<(String, Shape)> {
    let workers = std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(2)
        .max(1);
    let chunk = (candidates.len() + workers - 1) / workers;
    let mut out: Vec<(String, Shape)> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for part in candidates.chunks(chunk.max(1)) {
            let exists: Vec<String> = exists.to_vec();
            let target = target.clone();
            handles.push(scope.spawn(move || {
                part.iter()
                    .map(|repo| {
                        let known = exists.iter().any(|known| known == repo);
                        (repo.clone(), repo_shape(repo, known))
                    })
                    .collect::<Vec<_>>()
            }));
        }
        for handle in handles {
            match handle.join() {
                Ok(part) => out.extend(part),
                Err(_) => {}
            }
        }
    });
    out
}

/// Find a repository whose weights have exactly the shape we need.
///
/// The shape has to match to the digit: llama.cpp refuses an adapter whose
/// tensors do not line up with the base model.
pub fn find_donor(target: &Signature, spec: &str, hint: &Hint) -> Donor {
    let mut candidates: Vec<String> = Vec::new();

    // The shape comes first: it is the only thing that is actually right.
    // The matrix leads, because it reaches the donor without a search.
    candidates.extend(hf_candidates(spec, hint));
    candidates.extend(name_candidates(hint));

    // These were returned by the hub's own search, so they exist. That is
    // what tells a gated repository from one that was never there.
    let found = counted_candidates(spec, hint);
    let exists: Vec<String> = found.clone();
    candidates.extend(found);

    let mut gated: Vec<String> = Vec::new();
    let mut found: Vec<String> = Vec::new();

    // Every candidate is asked for its shape, and that is one request each.
    // There are more candidates than there is patience, so they are asked in
    // parallel - but the order is kept, so the best candidate is still the
    // first one tried.
    let shapes = shapes_of(&candidates, &exists, target);
    for (repo, shape) in shapes {
        match shape {
            Shape::Known(signature) => {
                if signature.matches(target) {
                    found.push(repo);
                    // Five is enough: the first one that Heretic can read is
                    // the one that gets used.
                    if found.len() >= 5 {
                        break;
                    }
                }
            }
            Shape::Gated => {
                if !gated.contains(&repo) {
                    gated.push(repo);
                }
            }
            Shape::Missing => {}
        }
    }
    let (repo, alternatives) = match found.split_first() {
        Some((first, rest)) => (Some(first.clone()), rest.to_vec()),
        None => (None, Vec::new()),
    };
    Donor {
        repo,
        gated,
        alternatives,
    }
}

/// Split a model name from its tag: `qwen3:8b` is `qwen3` and `8b`.
pub fn split_tag(spec: &str) -> (&str, &str) {
    match spec.rsplit_once(':') {
        Some((name, tag)) if !tag.contains('/') => (name, tag),
        _ => (spec, ""),
    }
}

/// "qwen3-8b" becomes "Qwen3-8b".
fn capitalise(name: &str) -> String {
    name.split('-')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join("-")
}

/// The size in a name, if there is one: `qwen3-8b-q8_0.gguf` is 8B.
pub fn size_label_from_name(path: &str) -> Option<String> {
    let name = path.rsplit('/').next().unwrap_or(path);
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    for part in stem.split('-') {
        // The number may have a decimal point: `0.6b`.
        let number: String = part
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        let rest = &part[number.len()..];
        if !number.is_empty() && rest.eq_ignore_ascii_case("b") {
            let value: f64 = number.parse().unwrap_or(0.0);
            return Some(if value < 1.0 {
                format!("{value}B")
            } else {
                format!("{}B", value.round() as u64)
            });
        }
    }
    None
}

/// Fetch a repository's `config.json`.
pub fn config_of(repo: &str) -> Result<Value> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/config.json");
    let Some((body, code)) = get_with_status(&url) else {
        anyhow::bail!("could not read {repo}'s config");
    };
    if code != "200" {
        anyhow::bail!("{repo}'s config is not available ({code})");
    }
    Ok(serde_json::from_str(&body)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tag_is_split_off() {
        assert_eq!(split_tag("qwen3:8b"), ("qwen3", "8b"));
        assert_eq!(split_tag("hf.co/user/model"), ("hf.co/user/model", ""));
    }

    #[test]
    fn a_size_is_read_out_of_a_name() {
        assert_eq!(
            size_label_from_name("/tmp/qwen3-8b-q8_0.gguf"),
            Some("8B".to_string())
        );
        assert_eq!(
            size_label_from_name("/tmp/qwen3-0.6b.gguf"),
            Some("0.6B".to_string())
        );
        assert_eq!(size_label_from_name("/tmp/model.gguf"), None);
    }

    #[test]
    fn quantised_reuploads_are_not_donors() {
        assert!(is_derivative("orcarouter/Qwen3.8-27B-Uncensored-GGUF"));
        assert!(is_derivative("someone/Qwen3-8B-AWQ"));
        assert!(!is_derivative("Qwen/Qwen3-8B"));
        assert!(!is_derivative("Qwen/Qwen3-30B-A3B"));
    }

    #[test]
    fn the_obvious_name_is_guessed_from_the_shape() {
        let hint = Hint {
            parameters: Some(8_190_735_360),
            architecture: Some("qwen3".into()),
            size_label: Some("8B".into()),
            name: None,
        };
        let candidates = name_candidates(&hint);
        assert!(candidates.contains(&"Qwen/Qwen3-8B".to_string()));
    }

    #[test]
    fn the_matrix_reaches_a_model_whose_name_lies() {
        // `Qwen:latest` is really Qwen2.5-32B: llama.cpp only says `qwen2`.
        let hint = Hint {
            parameters: Some(32_558_164_960),
            architecture: Some("qwen2".into()),
            size_label: Some("32B".into()),
            name: None,
        };
        let candidates = hf_candidates("Qwen:latest", &hint);
        assert!(
            candidates.contains(&"Qwen/Qwen2.5-32B".to_string()),
            "the matrix missed it: {candidates:?}"
        );
    }

    #[test]
    fn the_matrix_keeps_the_version_the_model_carries() {
        let hint = Hint {
            parameters: Some(7_615_616_512),
            architecture: Some("qwen2".into()),
            size_label: Some("7B".into()),
            name: None,
        };
        let candidates = hf_candidates("qwen2.5-coder:7b", &hint);
        assert!(
            candidates.contains(&"Qwen/Qwen2.5-7B".to_string()),
            "the matrix missed it: {candidates:?}"
        );
    }

    #[test]
    fn a_mixture_of_experts_has_two_spellings() {
        let spellings = size_spellings("30B-A3B");
        assert_eq!(spellings, vec!["30B-A3B".to_string(), "30B".to_string()]);
    }
}
