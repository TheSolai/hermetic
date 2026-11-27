//! The adapter cache: one Heretic adapter per model shape.
//!
//! One adapter serves every model with that shape, in every quantisation and
//! from every runtime - that is the whole point of keying by shape.
//!
//! Building an adapter is expensive, so it is done once per *signature* and
//! then reused by every model with that shape, in every quantisation.
//!
//! The cache holds two things per shape:
//!
//!   * `models/<sig>.gguf`  - a GGUF of that shape, to serve the model from.
//!   * `adapters/<sig>.gguf` - the Heretic adapter, to serve it with.

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result};

use crate::signature::Signature;

/// Where everything lives.
#[derive(Debug, Clone)]
pub struct Registry {
    pub root: PathBuf,
}

impl Registry {
    pub fn new() -> Registry {
        let root = std::env::var_os("HERMETIC_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let home = std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("/"));
                home.join(".hermetic")
            });
        Registry { root }
    }

    pub fn adapters_dir(&self) -> PathBuf {
        self.root.join("adapters")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    pub fn builds_dir(&self) -> PathBuf {
        self.root.join("builds")
    }

    pub fn study_dir(&self) -> PathBuf {
        self.root.join("study")
    }

    /// Models with the adapter folded into their weights.
    pub fn merged_dir(&self) -> PathBuf {
        self.root.join("merged")
    }

    /// A merged model for a shape, if we have built one.
    pub fn merged_for(&self, sig: &Signature) -> Option<PathBuf> {
        let path = self.merged_dir().join(format!("{}.gguf", sig.key()));
        path.exists().then_some(path)
    }

    /// The adapter for a shape, if we have built one.
    pub fn adapter_for(&self, sig: &Signature) -> Option<PathBuf> {
        let path = self.adapters_dir().join(format!("{}.gguf", sig.key()));
        path.exists().then_some(path)
    }

    /// A GGUF of a shape, if we have one.
    pub fn model_for(&self, sig: &Signature) -> Option<PathBuf> {
        let path = self.models_dir().join(format!("{}.gguf", sig.key()));
        path.exists().then_some(path)
    }

    /// The PEFT adapter a previous build left behind, if there is one.
    ///
    /// Heretic writes its result as a PEFT adapter before it is converted to
    /// GGUF. That file is the expensive part - hours of optimisation - so it
    /// is worth more than the GGUF next to it: the GGUF can be made again in
    /// a minute, the adapter cannot.
    pub fn peft_adapter_for(&self, sig: &Signature) -> Option<PathBuf> {
        let path = self
            .builds_dir()
            .join(sig.key())
            .join("adapter_model.safetensors");
        path.exists().then_some(path)
    }

    /// The donor a previous build used, if there is one.
    ///
    /// The adapter's own config says which model it was made for. That is
    /// better than asking the hub again: the hub is slow, it rate-limits, and
    /// the answer it would give is the one already written down.
    pub fn donor_of(&self, sig: &Signature) -> Option<String> {
        let path = self.builds_dir().join(sig.key()).join("adapter_config.json");
        let text = std::fs::read_to_string(path).ok()?;
        let value: serde_json::Value = serde_json::from_str(&text).ok()?;
        let base = value.get("base_model_name_or_path")?.as_str()?.trim();
        (!base.is_empty()).then(|| base.to_string())
    }

    /// Put a GGUF into the cache under a shape.
    ///
    /// A hard link is tried first: the weights are usually large, and Ollama's
    /// blobs must not be copied around.
    pub fn adopt_model(&self, sig: &Signature, source: &Path) -> Result<PathBuf> {
        let _ = std::fs::create_dir_all(self.models_dir());
        let target = self.models_dir().join(format!("{}.gguf", sig.key()));
        if target.exists() {
            return Ok(target);
        }
        match std::fs::hard_link(source, &target) {
            Ok(()) => return Ok(target),
            Err(_) => {
                std::fs::copy(source, &target).with_context(|| {
                    format!("copy {} into the cache", source.display())
                })?;
                Ok(target)
            }
        }
    }

    /// Build the adapter for a shape from a donor repository.
    ///
    /// The donor must have exactly the target shape; that is what makes the
    /// adapter bindable.
    /// Run upstream Heretic on a donor and leave the PEFT adapter behind.
    ///
    /// This is the slow part, and it is the same for every way of using the
    /// result: the adapter is what Heretic optimises.
    fn run_heretic(
        &self,
        sig: &Signature,
        donor: &str,
        label: &str,
        python: &Path,
        bridge_dir: &Path,
        socket: Option<&Path>,
    ) -> Result<PathBuf> {
        let _ = std::fs::create_dir_all(self.builds_dir());
        let _ = std::fs::create_dir_all(self.study_dir());

        let key = sig.key();
        let build_dir = self.builds_dir().join(&key);

        // If the optimiser has already run for this shape, its PEFT adapter
        // is still there, and running Heretic again would throw it away and
        // spend hours making the same answer. So it is kept, and the build
        // goes straight to the conversion.
        if build_dir.join("adapter_model.safetensors").exists() {
            println!("  reusing the adapter the optimiser already produced");
            return Ok(build_dir);
        }

        let _ = std::fs::remove_dir_all(&build_dir);
        std::fs::create_dir_all(&build_dir)?;

        let mut cmd = Command::new(python);
        cmd.current_dir(bridge_dir)
            .env("PYTHONPATH", bridge_dir)
            .env("PYTHONUNBUFFERED", "1")
            .arg("-m")
            .arg("heretical_bridge")
            .arg("--model")
            .arg(donor)
            .arg("--model-action")
            .arg("save")
            .arg("--export-strategy")
            .arg("ADAPTER")
            .arg("--save-directory")
            .arg(&build_dir)
            .arg("--study-checkpoint-dir")
            .arg(self.study_dir().join(&key))
            // The trial to keep is chosen by the engine, which knows the
            // scores: upstream sorts its Pareto front by the scorers one
            // after another, so the first trial is the one that refuses not
            // at all - even when it has moved the model so far that it no
            // longer speaks. Asking is what lets the better trial be
            // picked, and the bridge answers.
            .stdin(Stdio::null());

        // The profile is upstream Heretic's own configuration file. It is
        // handed over the way upstream expects - through the environment
        // variable it reads - so the number of trials, the batch size and
        // the dtypes are the ones that were written down, and not a number
        // hardcoded here that would quietly override all of them.
        let profile = crate::config::Config::load().profile();
        if profile.is_file() {
            cmd.env("HERETIC_CONFIG_FILE", &profile);
        }

        if let Some(sock) = socket {
            cmd.env("HERETICAL_CHANNEL_SOCKET", sock);
            cmd.arg("--heretical-socket").arg(sock);
        }
        cmd.arg("--heretical-label").arg(label);

        // The engine's own console goes to the terminal the user is looking
        // at, so the build is not a silent wait. Its errors are kept as well:
        // "it failed" on its own is not an answer.
        cmd.stdout(Stdio::inherit());
        cmd.stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("run heretic on {donor}"))?;
        let tail = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        if let Some(stderr) = child.stderr.take() {
            let tail = std::sync::Arc::clone(&tail);
            std::thread::spawn(move || {
                let reader = std::io::BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    eprintln!("{line}");
                    let mut tail = tail.lock().unwrap_or_else(|e| e.into_inner());
                    tail.push(line);
                    if tail.len() > 20 {
                        tail.remove(0);
                    }
                }
            });
        }

        let status = child
            .wait()
            .with_context(|| format!("run heretic on {donor}"))?;
        if !status.success() {
            // The last thing the engine said is the reason it stopped.
            let tail = tail.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let reason = tail
                .iter()
                .rev()
                .find(|line| !line.trim().is_empty())
                .cloned()
                .unwrap_or_default();
            if reason.is_empty() {
                anyhow::bail!("heretic failed on {donor} ({status})");
            }
            anyhow::bail!("heretic failed on {donor} ({status}): {reason}");
        }

        Ok(build_dir)
    }

    /// Build the adapter for a shape from a donor repository.
    ///
    /// The donor must have exactly the target shape; that is what makes the
    /// adapter bindable.
    pub fn build(
        &self,
        sig: &Signature,
        donor: &str,
        label: &str,
        python: &Path,
        bridge_dir: &Path,
        llama_cpp: &Path,
        socket: Option<&Path>,
    ) -> Result<PathBuf> {
        let _ = std::fs::create_dir_all(self.adapters_dir());
        let key = sig.key();
        let build_dir = self.run_heretic(
            sig,
            donor,
            label,
            python,
            bridge_dir,
            socket,
        )?;

        // Convert the PEFT adapter to GGUF.
        let converter = llama_cpp.join("convert_lora_to_gguf.py");
        let out_file = self.adapters_dir().join(format!("{key}.gguf"));
        let _ = std::fs::remove_file(&out_file);

        let conv = Command::new(python)
            .current_dir(llama_cpp)
            .env("PYTHONPATH", llama_cpp.join("gguf-py"))
            .arg(&converter)
            .arg(&build_dir)
            .arg("--base-model-id")
            .arg(donor)
            .arg("--outfile")
            .arg(&out_file)
            .arg("--outtype")
            .arg("f16")
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .with_context(|| format!("run {}", converter.display()))?;

        if !conv.success() || !out_file.exists() {
            anyhow::bail!(
                "could not convert the adapter to GGUF ({} failed)",
                converter.display()
            );
        }

        Ok(out_file)
    }

    /// Build a model with the adapter folded into its weights.
    ///
    /// This is the way out when a LoRA cannot be used: an architecture
    /// llama.cpp will not bind a LoRA to, or a model that has no GGUF of its
    /// own. The adapter is merged into the donor's weights, and the result is
    /// an ordinary GGUF.
    pub fn build_merged(
        &self,
        sig: &Signature,
        donor: &str,
        label: &str,
        python: &Path,
        bridge_dir: &Path,
        llama_cpp: &Path,
        socket: Option<&Path>,
    ) -> Result<PathBuf> {
        let _ = std::fs::create_dir_all(self.merged_dir());
        let key = sig.key();
        let build_dir = self.builds_dir().join(&key);

        // If the optimiser has already run for this shape, its PEFT adapter
        // is still there. Merging it is minutes cheaper than optimising it
        // again, and the answer is the same.
        let existing = build_dir.join("adapter_model.safetensors");
        let build_dir = if existing.exists() {
            println!("  reusing the adapter the optimiser already produced");
            build_dir
        } else {
            self.run_heretic(
                sig,
                donor,
                label,
                python,
                bridge_dir,
                socket,
            )?
        };

        // 2. Fold the adapter into the weights.
        let merged_dir = build_dir.join("merged");
        let merger = bridge_dir.join("scripts").join("merge_adapter.py");
        let merge = Command::new(python)
            .env("PYTHONUNBUFFERED", "1")
            .arg(&merger)
            .arg("--base")
            .arg(donor)
            .arg("--adapter")
            .arg(&build_dir)
            .arg("--out")
            .arg(&merged_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .with_context(|| format!("run {}", merger.display()))?;
        if !merge.success() {
            anyhow::bail!("could not merge the adapter into {donor}");
        }

        // 3. Convert the merged weights to GGUF.
        let converter = llama_cpp.join("convert_hf_to_gguf.py");
        let out_file = self.merged_dir().join(format!("{key}.gguf"));
        let _ = std::fs::remove_file(&out_file);

        let conv = Command::new(python)
            .current_dir(llama_cpp)
            .env("PYTHONPATH", llama_cpp.join("gguf-py"))
            .arg(&converter)
            .arg(&merged_dir)
            .arg("--outfile")
            .arg(&out_file)
            .arg("--outtype")
            .arg("f16")
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .with_context(|| format!("run {}", converter.display()))?;

        if !conv.success() || !out_file.exists() {
            anyhow::bail!(
                "could not convert the merged model to GGUF ({} failed)",
                converter.display()
            );
        }

        Ok(out_file)
    }
}

/// A free TCP port on the loopback interface.
pub fn free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A registry in a temporary directory, so a test never touches the
    /// real one. The name is unique per test, because tests run at the same
    /// time and would otherwise share a directory.
    fn a_registry(name: &str) -> (Registry, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "hermetic-registry-test-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        (Registry { root: root.clone() }, root)
    }

    fn a_signature() -> Signature {
        Signature {
            arch: "qwen2".into(),
            layers: 24,
            hidden: 896,
            inter: 4864,
            heads: 14,
            kv_heads: 2,
            experts: 0,
        }
    }

    #[test]
    fn free_ports_are_not_zero() {
        assert_ne!(free_port().unwrap(), 0);
    }

    #[test]
    fn an_existing_adapter_names_its_own_donor() {
        let (registry, root) = a_registry("donor");
        let signature = a_signature();
        assert!(registry.donor_of(&signature).is_none());

        let build = registry.builds_dir().join(signature.key());
        std::fs::create_dir_all(&build).unwrap();
        std::fs::write(
            build.join("adapter_config.json"),
            r#"{"base_model_name_or_path": "Qwen/qwen2.5-0.5b"}"#,
        )
        .unwrap();

        assert_eq!(
            registry.donor_of(&signature).as_deref(),
            Some("Qwen/qwen2.5-0.5b"),
            "the donor is remembered, so the hub does not have to be asked again"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_peft_adapter_is_found_where_it_was_left() {
        let (registry, root) = a_registry("peft");
        let signature = a_signature();
        assert!(registry.peft_adapter_for(&signature).is_none());

        let build = registry.builds_dir().join(signature.key());
        std::fs::create_dir_all(&build).unwrap();
        std::fs::write(build.join("adapter_model.safetensors"), b"weights").unwrap();

        assert!(registry.peft_adapter_for(&signature).is_some());

        let _ = std::fs::remove_dir_all(&root);
    }
}
