//! HERMETIC - shared pieces of the terminal and of the interceptor.
//!
//! Two binaries are built from this crate:
//!
//!   * `hermetic` - the terminal. It hosts a shell in a pseudo-terminal, so the
//!     user types at a real prompt, and it keeps a side panel with whatever
//!     the Heretic engine has to say.
//!   * `hermetic-shim` - the interceptor. The shell routes model commands to
//!     it; it resolves the model, makes sure a Heretic adapter exists for that
//!     model shape, serves the model with the adapter bound to it and talks to
//!     it.
//!
//! Neither of them knows which runtime is behind the model. That is
//! [`hermetic_core`]'s job: a model is a [`hermetic_core::ModelSource`] and a
//! runtime is a [`hermetic_core::ModelRuntime`], and the terminal only ever
//! talks to whichever of them the planner picked.

pub mod ansi;
pub mod app;
pub mod catalogue;
pub mod channel;
pub mod runtime;
pub mod shell;
pub mod spec;
pub mod ui;

/// The socket the engines report to.
pub const SOCKET_ENV: &str = "HERMETIC_SOCKET";

/// Where the interceptor lives.
pub const SHIM_ENV: &str = "HERMETIC_SHIM";

/// Where the adapter cache lives.
pub const HOME_ENV: &str = "HERMETIC_HOME";

/// Where the engine lives.
pub const ENGINE_ENV: &str = "HERMETIC_ENGINE_DIR";

/// Which tab a shell belongs to.
pub const TAB_ENV: &str = "HERMETIC_TAB";

/// Where the clickable regions are written out, when that is wanted.
pub const REGIONS_ENV: &str = "HERMETIC_DEBUG_REGIONS";

/// The command that opens a model in a window of its own.
pub const OPEN_CMD_ENV: &str = "HERMETIC_OPEN_CMD";

pub use hermetic_core::config::repo_root;

/// The sibling binary of the running one.
///
/// The app bundle keeps both binaries next to each other, so the interceptor
/// is always where the terminal expects it.
pub fn sibling_binary(name: &str) -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let candidate = dir.join(name);
    candidate.exists().then_some(candidate)
}

/// Find a program on the PATH.
pub fn which(name: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
