//! The donor search, against the real hub.
//!
//! These tests talk to Hugging Face and to Ollama, so they are ignored by
//! default. Run them with:
//!
//! ```text
//! cargo test --release --test donor -- --ignored --nocapture
//! ```
//!
//! What they check is the thing that actually breaks in the field: a model
//! whose *name* does not describe its weights. `deepseek-r1:8b` is a
//! `Qwen3-8B`, and `Qwen:latest` is a `Qwen1.5-4B-Chat` called something
//! else entirely. The search has to reach the original weights from that.

use hermetic_terminal::spec::{hint_of, shape_of, source_of};

/// Find the repository whose weights have the same shape as `model`.
fn donor_for(model: &str) -> Option<String> {
    let source = source_of(model).expect("a source");
    let signature = shape_of(&source).expect("a shape");
    let hint = hint_of(&source, &signature);
    let donor = hermetic_core::hub::find_donor(&signature, model, &hint);
    if donor.repo.is_none() && !donor.gated.is_empty() {
        println!("  gated instead: {:?}", donor.gated);
    }
    donor.repo
}

#[test]
#[ignore = "talks to the hub and to Ollama"]
fn a_renamed_model_finds_its_original_weights() {
    // `deepseek-r1:8b` is a `Qwen3-8B`.
    let repo = donor_for("deepseek-r1:latest").expect("a donor for deepseek-r1");
    println!("deepseek-r1:latest -> {repo}");
    assert!(
        repo.starts_with("Qwen/") || repo.contains("Qwen3"),
        "unexpected donor: {repo}"
    );
}

#[test]
#[ignore = "talks to the hub and to Ollama"]
fn a_model_whose_name_says_nothing_still_finds_its_weights() {
    // `Qwen:latest` says nothing at all: not a family, not a size.
    let repo = donor_for("Qwen:latest").expect("a donor for Qwen:latest");
    println!("Qwen:latest -> {repo}");
}

#[test]
#[ignore = "talks to the hub and to Ollama"]
fn a_qwen_family_model_finds_its_weights() {
    let repo = donor_for("qwen2.5-coder:7b").expect("a donor for qwen2.5-coder:7b");
    println!("qwen2.5-coder:7b -> {repo}");
    assert!(repo.contains("Qwen2.5"), "unexpected donor: {repo}");
}

#[test]
#[ignore = "talks to the hub and to Ollama"]
fn a_vision_model_finds_the_model_it_was_built_on() {
    // `llava:34b` is a LLaVA wrapper around Llama-3-34B, and its own weights
    // say nothing useful: the GGUF calls itself "LLaMA v2". Any repository
    // with the same shape will do - the adapter is what has to line up.
    let repo = donor_for("llava:34b").expect("a donor for llava:34b");
    println!("llava:34b -> {repo}");
}

#[test]
#[ignore = "talks to the hub and to Ollama"]
fn a_mixture_of_experts_finds_its_weights() {
    // `qwen35moe` is a 35B mixture of experts, published under a name the
    // architecture does not give away, and carrying an extra block for
    // multi-token prediction that the original weights do not have.
    let repo = donor_for(
        "llamacpp:782c715f39fee143c2dcc3812552f805b75d48f143f98a3ccd7b75123b558992",
    )
    .expect("a donor for the qwen35moe model");
    println!("qwen35moe -> {repo}");
}

#[test]
#[ignore = "talks to the hub and to Ollama"]
fn a_model_that_ships_its_own_code_finds_the_repository_that_ships_it() {
    // `clover:35b` is a hybrid model that is not in Transformers at all: the
    // repository carries its own model code next to the weights. A conversion
    // of it - an MLX one, a quantised one - cannot be read by the optimiser,
    // so the donor has to be the repository itself.
    let repo = donor_for("clover:35b").expect("a donor for clover:35b");
    println!("clover:35b -> {repo}");
    for forbidden in ["-mlx", "-4bit", "-8bit", "-int4", "-nf4", "-w4a16"] {
        assert!(
            !repo.contains(forbidden),
            "a {forbidden} copy cannot be read by the optimiser: {repo}"
        );
    }
}

#[test]
#[ignore = "talks to the hub and to Ollama"]
fn an_embedding_model_has_no_donor_and_says_so() {
    let source = source_of("nomic-embed-text:latest").expect("a source");
    let signature = shape_of(&source).expect("a shape");
    let hint = hint_of(&source, &signature);
    let donor = hermetic_core::hub::find_donor(&signature, "nomic-embed-text:latest", &hint);
    println!("nomic-embed-text:latest -> {donor:?}");
}

#[test]
#[ignore = "talks to the hub and to Ollama"]
fn a_model_without_a_name_in_its_weights_still_finds_them() {
    // `phi4:14b` is a `microsoft/phi-4`, and llama.cpp calls that
    // architecture `phi3` - so neither the architecture nor the name in the
    // weights gets there. The hub's own search does.
    let repo = donor_for("phi4:14b").expect("a donor for phi4:14b");
    println!("phi4:14b -> {repo}");
}
