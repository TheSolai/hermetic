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

/// Take the control characters out of a string that is about to be drawn.
///
/// A model's name can come from a file name on disk, and macOS only forbids
/// `/` and NUL in those - so an escape sequence or a newline can be in one.
/// Drawn as they are, the terminal would *run* them: the screen would be
/// changed by a name, and a newline in a name would move every row below it,
/// so a click would land on the wrong model.
///
/// Control characters are replaced by a space, and the result is trimmed.
pub fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_control() {
            out.push(' ');
        } else {
            out.push(ch);
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_escape_sequence_in_a_name_is_defused() {
        let name = "evil\u{1b}[2Jmodel";
        assert_eq!(clean(name), "evil [2Jmodel");
    }

    #[test]
    fn a_newline_in_a_name_does_not_move_the_rows() {
        assert_eq!(clean("one\ntwo"), "one two");
    }

    #[test]
    fn an_ordinary_name_is_left_alone() {
        assert_eq!(clean("qwen3:0.6b"), "qwen3:0.6b");
    }
}
