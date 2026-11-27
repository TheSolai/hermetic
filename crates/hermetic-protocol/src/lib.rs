//! Wire protocol between the HERMETIC client and the Python bridge.
//!
//! The bridge (`engine/bridge/hermetic_bridge.py`) runs upstream Heretic
//! unchanged and exposes it over a newline-delimited JSON channel. Events
//! flow from the engine to the client; answers flow back the other way on the
//! same socket.
//!
//! Nothing here is Heretic-specific beyond the event names: the same types
//! describe any command the terminal spawns.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An event emitted by the engine.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// A line of console output, with Rich markup still in it.
    Log { text: String },
    /// An engine has started: this is the model it is working on.
    BuildStart { label: String },
    /// A model is being served: this is the tab it belongs to.
    Serving {
        model: String,
        tab: Option<u64>,
        #[serde(default)]
        shape: Option<String>,
        #[serde(default)]
        adapter: Option<String>,
        #[serde(default)]
        endpoint: Option<String>,
    },
    /// Scorer results for a phase.
    Scores { phase: String, scores: Vec<Score> },
    /// A new optimisation trial has been proposed.
    TrialStart {
        number: Option<u64>,
        modifier: String,
        parameters: Value,
    },
    /// The proposed parameters are being applied to the model.
    TrialModify {
        modifier: String,
        parameters: Value,
    },
    /// The engine is asking the user something.
    Question {
        kind: QuestionKind,
        message: String,
        payload: QuestionPayload,
        default: Value,
    },
    /// The model was asked the questions it was judged on, and this is how it
    /// answered.
    ///
    /// This is the product's promise made visible: a model that has been
    /// abliterated refuses less, and this says by how much.
    Probe {
        /// What was measured - "Refusals", for the refusal probe.
        name: String,
        /// How many answers matched.
        matches: usize,
        /// How many answers there were.
        total: usize,
        /// One line about the result, with the score of the untouched model
        /// in it when there is one.
        #[serde(default)]
        summary: Option<String>,
    },
    /// The engine finished.
    Done { exit_code: i32 },
    /// The engine crashed.
    Error { error: String },
    /// An event this client does not know about. Forward compatible.
    #[serde(other)]
    Unknown,
}

/// One scorer's result.
#[derive(Debug, Clone, Deserialize)]
pub struct Score {
    pub name: String,
    pub value: f64,
    pub display: String,
}

/// The kind of question the engine asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionKind {
    Select,
    Checkbox,
    Text,
    Path,
    Password,
    Confirm,
}

/// Extra data attached to a question.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct QuestionPayload {
    #[serde(default)]
    pub choices: Vec<Choice>,
}

/// One selectable option.
#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub title: String,
    pub value: Value,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub trial: Option<Trial>,
}

/// The optimisation trial behind a choice, when the choice stands for one.
#[derive(Debug, Clone, Deserialize)]
pub struct Trial {
    pub number: Option<u64>,
    #[serde(default)]
    pub parameters: Option<Value>,
    #[serde(default)]
    pub scores: Option<Vec<Value>>,
}

/// An answer sent back to the engine.
///
/// `select` questions are answered with the *index* of the chosen option;
/// `checkbox` questions with the list of chosen indices. The bridge resolves
/// those against its own choice list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    pub answer: Value,
}

impl Answer {
    pub fn index(i: usize) -> Self {
        Answer {
            answer: Value::from(i),
        }
    }

    pub fn indices(is: &[usize]) -> Self {
        Answer {
            answer: Value::from(is.to_vec()),
        }
    }

    pub fn text(s: impl Into<String>) -> Self {
        Answer {
            answer: Value::from(s.into()),
        }
    }

    pub fn confirm(b: bool) -> Self {
        Answer {
            answer: Value::from(b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_event_parses() {
        let raw = r#"{"event":"log","ts":1.0,"text":"hello"}"#;
        match serde_json::from_str::<Event>(raw).unwrap() {
            Event::Log { text } => assert_eq!(text, "hello"),
            other => panic!("expected Log, got {other:?}"),
        }
    }

    #[test]
    fn an_unknown_event_is_not_an_error() {
        let raw = r#"{"event":"something_new","payload":42}"#;
        assert!(matches!(
            serde_json::from_str::<Event>(raw).unwrap(),
            Event::Unknown
        ));
    }

    #[test]
    fn a_question_carries_its_choices() {
        let raw = r#"{"event":"question","kind":"select","message":"pick",
            "payload":{"choices":[{"title":"a","value":{"x":1}}]},"default":null}"#;
        match serde_json::from_str::<Event>(raw).unwrap() {
            Event::Question { kind, payload, .. } => {
                assert_eq!(kind, QuestionKind::Select);
                assert_eq!(payload.choices.len(), 1);
                assert_eq!(payload.choices[0].title, "a");
            }
            other => panic!("expected Question, got {other:?}"),
        }
    }

    #[test]
    fn an_answer_serialises_as_the_bridge_expects() {
        let line = serde_json::to_string(&Answer::index(2)).unwrap();
        assert_eq!(line, r#"{"answer":2}"#);
    }
}
