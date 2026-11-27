//! Where the engine lives.
//!
//! The terminal is two things: a Rust program and a Python environment with
//! Heretic in it. The Python environment is far too big to ship inside an app
//! bundle, so the bundle carries a small configuration file that says where
//! the engine lives, and the binaries fall back to the repository they were
//! built in.

use std::path::{Path, PathBuf};

/// The engine's location.
#[derive(Debug, Clone)]
pub struct Config {
    /// The directory holding the engine: the virtualenv, the bridge, the
    /// vendored Heretic.
    pub engine_dir: PathBuf,
    /// The directory holding the llama.cpp conversion scripts.
    pub llama_cpp_dir: PathBuf,
    /// The interpreter that has Heretic installed.
    pub python: PathBuf,
}

impl Config {
    /// The bridge directory.
    pub fn bridge_dir(&self) -> PathBuf {
        self.engine_dir.join("bridge")
    }

    /// The Heretic profile to use for builds.
    pub fn profile(&self) -> PathBuf {
        self.engine_dir
            .parent()
            .map(|p| p.join("profiles/fast-test.toml"))
            .unwrap_or_else(|| PathBuf::from("profiles/fast-test.toml"))
    }

    /// Read the configuration.
    pub fn load() -> Config {
        if let Some(dir) = std::env::var_os("HERMETIC_ENGINE_DIR") {
            return Config::from_engine(PathBuf::from(dir));
        }

        // A configuration file next to the binary, or in the bundle's
        // resources.
        for candidate in config_file_candidates() {
            if let Some(engine_dir) = read_engine_dir(&candidate) {
                return Config::from_engine(engine_dir);
            }
        }

        Config::from_engine(repo_root().join("engine"))
    }

    /// The engine, at a known place.
    pub fn from_engine(engine_dir: PathBuf) -> Config {
        let llama_cpp_dir = engine_dir.join("llamacpp");
        let python = engine_dir.join(".venv/bin/python");
        Config {
            engine_dir,
            llama_cpp_dir,
            python,
        }
    }
}

/// The repository this was built in.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn config_file_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join("hermetic.conf"));
            // Inside an app bundle: Contents/MacOS/../Resources/hermetic.conf
            if let Some(resources) = dir.parent().map(|p| p.join("Resources")) {
                out.push(resources.join("hermetic.conf"));
            }
        }
    }
    out
}

fn read_engine_dir(path: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(path).ok()?;
    for line in text.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("engine_dir=") {
            let value = value.trim().trim_matches('"');
            let candidate = PathBuf::from(value);
            if candidate.join(".venv").exists() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_engine_is_where_it_says() {
        let config = Config::load();
        assert!(config.engine_dir.to_string_lossy().contains("engine"));
        assert!(config.python.to_string_lossy().contains(".venv"));
    }
}
