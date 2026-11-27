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

/// Fetch a URL and parse it as JSON.
fn get_json(url: &str) -> Option<Value> {
    let mut command = Command::new("curl");
    command.args(["-sL", "--max-time", "20"]);
    for argument in auth_args() {
        command.arg(argument);
    }
    let out = command.arg(url).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str(&text).ok()
}

/// Fetch a URL and read its body and its status code.
fn get_with_status(url: &str) -> Option<(String, String)> {
    let mut command = Command::new("curl");
    command.args(["-sL", "--max-time", "20"]);
    for argument in auth_args() {
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
enum Shape {
    /// The shape, read from the repo's config.
    Known(Signature),
    /// The repo exists but is gated: it needs a token.
    Gated,
    /// There is no such repo, or it could not be read.
    Missing,
}

/// Read a repo's `config.json` and work out its shape.
fn repo_shape(repo: &str) -> Shape {
    let url = format!("https://huggingface.co/{repo}/resolve/main/config.json");
    let Some((body, code)) = get_with_status(&url) else {
        return Shape::Missing;
    };
    match code.as_str() {
        "200" => {}
        // The download was refused. That is not proof of anything: the hub
        // answers 401 for a repository that does not exist as well as for one
        // that is gated. Only the repository's own record can tell them
        // apart.
        "401" | "403" => {
            return match repo_info(repo) {
                Some(info) if info.gated => Shape::Gated,
                _ => Shape::Missing,
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

/// The families a model can belong to, spelled the way repositories spell them.
///
/// The architecture alone is not enough: llama.cpp calls every Qwen 2.x model
/// `qwen2`, but the hub has `Qwen2-32B` and `Qwen2.5-32B` as separate
/// repositories. The minor version is the difference between finding the
/// donor and not finding it.
fn family_spellings(spec: &str, hint: &Hint) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |value: String| {
        if !value.is_empty() && !out.contains(&value) {
            out.push(value);
        }
    };

    // The model's own name is the best family we have: `qwen2.5-coder` says
    // `Qwen2.5`, even though llama.cpp only says `qwen2`.
    let (name, _) = split_tag(spec);
    let base = name.rsplit('/').next().unwrap_or(name);
    if !base.is_empty() {
        push(capitalise(base));
        // Drop the role words: `Qwen2.5-Coder` -> `Qwen2.5`.
        let trimmed: Vec<&str> = base
            .split('-')
            .filter(|part| !ROLE_WORDS.contains(&part.to_lowercase().as_str()))
            .collect();
        if !trimmed.is_empty() {
            push(capitalise(&trimmed.join("-")));
        }
    }

    // The architecture, with the minor version it hides.
    if let Some(arch) = &hint.architecture {
        let family = capitalise(arch);
        let versioned = family.chars().last().is_some_and(|c| c.is_ascii_digit());
        push(family.clone());
        if versioned {
            for minor in [".5", ".0"] {
                push(format!("{family}{minor}"));
            }
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
            let repo = format!("{org}/{name}");
            if !out.contains(&repo) {
                out.push(repo);
            }
        }
        for family in &families {
            for size in &sizes {
                for suffix in ["", "-Instruct", "-Base", "-it", "-Chat"] {
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

/// The hub's search, by architecture, with the parameter count as a filter.
fn counted_candidates(hint: &Hint) -> Vec<String> {
    let mut out = Vec::new();
    let (Some(arch), Some(parameters)) = (&hint.architecture, hint.parameters) else {
        return out;
    };
    for repo in search(arch, 50) {
        if is_derivative(&repo) {
            continue;
        }
        let Some(info) = repo_info(&repo) else {
            continue;
        };
        let Some(total) = info.parameters else {
            continue;
        };
        // Two percent: tied embeddings and a slightly different head can move
        // the count by a little.
        let difference = (total as f64 - parameters as f64).abs() / parameters as f64;
        if difference < 0.02 && !out.contains(&repo) {
            out.push(repo);
        }
    }
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
    candidates.extend(counted_candidates(hint));

    // The hub's own search finds repos the matrix never guesses.
    let (name, _) = split_tag(spec);
    let base = name.rsplit('/').next().unwrap_or(name);
    for query in [base.to_string(), base.replace('-', " ")] {
        // A re-upload of quantised weights is not a donor: Heretic needs the
        // original tensors, and a GGUF repo does not have them.
        for repo in search(&query, 25) {
            if !is_derivative(&repo) && !candidates.contains(&repo) {
                candidates.push(repo);
            }
        }
    }

    let mut gated: Vec<String> = Vec::new();
    for repo in candidates {
        match repo_shape(&repo) {
            Shape::Known(signature) => {
                if signature.matches(target) {
                    return Donor {
                        repo: Some(repo),
                        gated,
                    };
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
    Donor { repo: None, gated }
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
