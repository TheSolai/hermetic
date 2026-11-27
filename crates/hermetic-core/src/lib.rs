//! HERMETIC - one interface to every model runtime.
//!
//! The terminal is not an Ollama client, a llama.cpp client or an MLX
//! client. It is a terminal that can talk to whatever is on the machine, and
//! this crate is the part that knows how.
//!
//! There are two ideas here, and everything else follows from them.
//!
//! **Where the weights are** - [`source::ModelSource`]. A model is
//! identified by where it lives: in Ollama's store, in a GGUF on disk, in an
//! MLX directory, on the Hugging Face hub, or in a server that is already
//! running. The same weights can be reached in more than one way, and the
//! same runtime can serve weights from more than one place.
//!
//! **How it is served** - [`runtime::ModelRuntime`]. A runtime is a way of
//! putting a model in front of the user: llama.cpp, Ollama's own engine, MLX,
//! or any server that speaks the OpenAI chat API. Every runtime that can be
//! talked to exposes the same chat API, so the terminal has exactly one chat
//! client and does not care what is behind it.
//!
//! What the runtimes do *not* share is whether they can change weights.
//! That is [`runtime::Modification`], and it is the honest answer to "can
//! this model be hereticed?": sometimes yes, sometimes only by converting
//! them first, and sometimes not at all - in which case the model is served
//! with a prompt-level policy and the terminal says so.
//!
//! The adapter cache - the Heretic adapters, keyed by shape - lives in
//! [`registry`], and the decision of what to do first lives in [`planner`].

pub mod config;
pub mod chat;
pub mod discovery;
pub mod gguf;
pub mod http;
pub mod hub;
pub mod planner;
pub mod policy;
pub mod registry;
pub mod runtimes;
pub mod runtime;
pub mod signature;
pub mod weights;
pub mod source;

pub use planner::Plan;
pub use policy::Policy;
pub use registry::Registry;
pub use runtime::{Modification, ModelRuntime, ServeHandle, ServeMode, ServeRequest};
pub use signature::Signature;
pub use source::{AdapterState, ModelDescriptor, ModelSource, SourceKind};
