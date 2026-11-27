//! What the terminal knows: the tabs, their screens, the engines' state and
//! the question that is waiting for an answer.

use std::collections::HashMap;

use hermetic_core::registry::Registry;
use hermetic_protocol::{Answer, Event, QuestionKind};

use crate::catalogue::CatalogueEntry;
use crate::shell::Shell;

/// One tab: a shell in a pseudo-terminal and the screen it draws.
pub struct Tab {
    pub shell: Shell,
    /// The shell's screen.
    pub grid: vt100::Parser,
    pub grid_rows: u16,
    pub grid_cols: u16,
    /// What the tab is called: the model it is talking to, or `zsh`.
    pub title: String,
    /// The model being served in this tab, if any.
    pub model: Option<String>,
    /// What is being served: its shape, its adapter, where it is served.
    pub shape: Option<String>,
    pub adapter: Option<String>,
    pub endpoint: Option<String>,
    /// What the probe found: "Refusals 2/16".
    pub probe: Option<String>,
    /// The tab's own number, so the shell can tell the terminal who it is.
    pub number: u64,
}

impl Tab {
    pub fn new(shell: Shell, rows: u16, cols: u16, number: u64) -> Tab {
        Tab {
            shell,
            grid: vt100::Parser::new(rows.max(8), cols.max(20), 4096),
            grid_rows: rows.max(8),
            grid_cols: cols.max(20),
            title: "zsh".to_string(),
            model: None,
            shape: None,
            adapter: None,
            endpoint: None,
            probe: None,
            number,
        }
    }

    /// The shell wrote something.
    pub fn feed(&mut self, bytes: &[u8]) {
        self.grid.process(bytes);
    }

    /// The window changed size.
    pub fn resize(&mut self, rows: u16, cols: u16) {
        let rows = rows.max(8);
        let cols = cols.max(20);
        if rows != self.grid_rows || cols != self.grid_cols {
            self.grid.set_size(rows, cols);
            self.grid_rows = rows;
            self.grid_cols = cols;
        }
    }
}

/// One engine (one Heretic run) and where it got to.
#[derive(Debug, Clone)]
pub struct Engine {
    pub id: u64,
    /// What the engine is working on.
    pub label: String,
    /// The phase it is in.
    pub phase: String,
    /// The score before any change: "Refusals 48/100".
    pub baseline: Option<String>,
    /// The score after the change: "Refusals 4/100".
    pub trial: Option<String>,
    /// `building`, `done` or `failed`.
    pub state: String,
}

impl Engine {
    fn new(id: u64) -> Engine {
        Engine {
            id,
            label: format!("engine {id}"),
            phase: String::new(),
            baseline: None,
            trial: None,
            state: "starting".into(),
        }
    }
}

/// A question the engine asked, waiting for an answer.
#[derive(Debug, Clone)]
pub struct Pending {
    pub engine: u64,
    pub kind: QuestionKind,
    pub message: String,
    pub choices: Vec<String>,
}

/// The terminal's state.
pub struct App {
    /// Every open tab.
    pub tabs: Vec<Tab>,
    /// Which one is in front.
    pub active: usize,
    /// Every engine that is connected.
    pub engines: Vec<Engine>,
    /// The log of everything the engines said.
    pub log: Vec<String>,
    /// The status line.
    pub status: String,
    /// A question waiting for an answer.
    pub question: Option<Pending>,
    /// How often each question has been asked.
    asked: HashMap<String, usize>,
    /// Is the side panel shown?
    pub show_panel: bool,
    /// Should the terminal close?
    pub quit: bool,
    /// The tab buttons, as they were drawn: the tab's index, where it is and
    /// how wide it is. The panel writes them so that a click can be matched
    /// to a tab.
    pub tab_buttons: Vec<(usize, u16, u16)>,
    /// Where the tab strip is.
    pub tab_strip_x: u16,
    pub tab_strip_y: u16,
    /// Every model that could be loaded.
    pub catalogue: Vec<CatalogueEntry>,
    /// Has the catalogue been read yet?
    pub catalogue_loaded: bool,
    /// Does the catalogue need looking for again?
    ///
    /// A build that has finished has changed the machine: there is an adapter
    /// for that shape now. The panel would go on saying "build" for a model
    /// that is ready, so the list is looked for again.
    pub catalogue_stale: bool,
    /// The first model in the list that is shown.
    pub list_scroll: usize,
    /// Is the keyboard driving the model list?
    pub list_focus: bool,
    /// Which model in the list is selected.
    pub list_selected: usize,
    /// The model rows as they were drawn, so a click can be matched to one.
    ///
    /// Each row is the model's place in the catalogue, where it was drawn and
    /// how wide it is.
    pub model_rows: Vec<(usize, u16, u16, u16)>,
    /// Is the panel on the left of the window?
    pub panel_left: bool,
    /// Where the clickable things are, as they were drawn.
    pub regions: String,
    /// The regions as they were last written out.
    pub last_regions: String,
    /// Where the regions are written, when that is wanted.
    pub regions_path: Option<std::path::PathBuf>,
    /// What the user has typed to narrow the model list down.
    pub list_filter: String,
    /// The catalogues' places that match the filter.
    pub matches: Vec<usize>,
    /// The model row the mouse is over, if any.
    pub list_hover: Option<usize>,
    /// Where the cache lives.
    pub registry: Registry,
    /// Where to look for models.
    pub sources: hermetic_core::discovery::Sources,
    /// The catalogue, when it is being read.
    pub catalogue_tx: std::sync::mpsc::Sender<Vec<CatalogueEntry>>,
    pub catalogue_rx: std::sync::mpsc::Receiver<Vec<CatalogueEntry>>,
}

impl App {
    pub fn new(tab: Tab) -> App {
        App {
            tabs: vec![tab],
            active: 0,
            engines: Vec::new(),
            log: Vec::new(),
            status: String::new(),
            question: None,
            asked: HashMap::new(),
            show_panel: true,
            quit: false,
            tab_buttons: Vec::new(),
            tab_strip_x: 0,
            tab_strip_y: 0,
            catalogue: Vec::new(),
            catalogue_loaded: false,
            catalogue_stale: false,
            list_scroll: 0,
            list_focus: false,
            list_selected: 0,
            model_rows: Vec::new(),
            panel_left: true,
            regions: String::new(),
            last_regions: String::new(),
            regions_path: None,
            list_filter: String::new(),
            matches: Vec::new(),
            list_hover: None,
            registry: Registry::new(),
            sources: hermetic_core::discovery::Sources::default(),
            catalogue_tx: std::sync::mpsc::channel().0,
            catalogue_rx: std::sync::mpsc::channel().1,
        }
    }

    /// The catalogue arrived.
    pub fn set_catalogue(&mut self, entries: Vec<CatalogueEntry>) {
        self.catalogue = entries;
        self.catalogue_loaded = true;
        self.refilter();
    }

    /// Work out which models the filter lets through.
    pub fn refilter(&mut self) {
        let needle = self.list_filter.trim().to_lowercase();
        self.matches = self
            .catalogue
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                needle.is_empty()
                    || entry.name.to_lowercase().contains(&needle)
                    || entry
                        .signature
                        .as_ref()
                        .map(|s| s.key().to_lowercase().contains(&needle))
                        .unwrap_or(false)
            })
            .map(|(index, _)| index)
            .collect();
        if self.list_selected >= self.matches.len() {
            self.list_selected = self.matches.len().saturating_sub(1);
        }
        if self.list_scroll > self.matches.len() {
            self.list_scroll = 0;
        }
    }

    /// Type a character into the filter.
    pub fn push_filter(&mut self, c: char) {
        self.list_filter.push(c);
        self.list_selected = 0;
        self.list_scroll = 0;
        self.refilter();
    }

    /// Take the last character off the filter.
    pub fn pop_filter(&mut self) {
        self.list_filter.pop();
        self.list_selected = 0;
        self.list_scroll = 0;
        self.refilter();
    }

    /// The model the list has selected.
    pub fn selected_model(&self) -> Option<&CatalogueEntry> {
        let index = *self.matches.get(self.list_selected)?;
        self.catalogue.get(index)
    }

    /// Move the selection in the model list.
    pub fn move_selection(&mut self, delta: isize) {
        if self.matches.is_empty() {
            return;
        }
        let last = self.matches.len() as isize - 1;
        let next = (self.list_selected as isize + delta).clamp(0, last);
        self.list_selected = next as usize;
        // Keep the selection in view.
        if self.list_selected < self.list_scroll {
            self.list_scroll = self.list_selected;
        }
    }

    /// Scroll the model list.
    pub fn scroll_list(&mut self, delta: isize) {
        if self.matches.is_empty() {
            return;
        }
        let last = self.matches.len() as isize - 1;
        let next = (self.list_scroll as isize + delta).clamp(0, last);
        self.list_scroll = next as usize;
        if self.list_selected < self.list_scroll {
            self.list_selected = self.list_scroll;
        }
    }

    /// The tab a click landed on, if it landed on one.
    ///
    /// `usize::MAX` is the new-tab button.
    pub fn hit_tab(&self, column: u16, row: u16) -> Option<usize> {
        if row != self.tab_strip_y {
            return None;
        }
        for (index, x, width) in &self.tab_buttons {
            if column >= *x && column < x + width {
                return Some(*index);
            }
        }
        None
    }

    /// The model a click landed on, if it landed on one.
    pub fn hit_model(&self, column: u16, row: u16) -> Option<usize> {
        for (index, x, y, width) in &self.model_rows {
            if row == *y && column >= *x && column < x + width {
                return Some(*index);
            }
        }
        None
    }

    /// The tab in front.
    pub fn active_tab(&self) -> &Tab {
        &self.tabs[self.active.min(self.tabs.len() - 1)]
    }

    /// The tab in front, mutably.
    pub fn active_tab_mut(&mut self) -> &mut Tab {
        let index = self.active.min(self.tabs.len() - 1);
        &mut self.tabs[index]
    }

    /// A tab by its number.
    pub fn tab_mut(&mut self, number: u64) -> Option<&mut Tab> {
        self.tabs.iter_mut().find(|t| t.number == number)
    }

    /// Open a tab.
    pub fn push_tab(&mut self, tab: Tab) {
        self.tabs.push(tab);
        self.active = self.tabs.len() - 1;
    }

    /// Close a tab. The last one cannot be closed.
    pub fn close_tab(&mut self, index: usize) {
        if self.tabs.len() <= 1 || index >= self.tabs.len() {
            return;
        }
        self.tabs.remove(index);
        if self.active >= self.tabs.len() {
            self.active = self.tabs.len() - 1;
        }
    }

    /// Show the next tab.
    pub fn next_tab(&mut self) {
        if !self.tabs.is_empty() {
            self.active = (self.active + 1) % self.tabs.len();
        }
    }

    /// Show the previous tab.
    pub fn previous_tab(&mut self) {
        if !self.tabs.is_empty() {
            self.active = (self.active + self.tabs.len() - 1) % self.tabs.len();
        }
    }

    /// An engine said something.
    pub fn on_event(&mut self, id: u64, event: Event) {
        match event {
            Event::BuildStart { label } => {
                if let Some(engine) = self.engine_mut(id) {
                    engine.label = label.clone();
                    engine.state = "building".into();
                }
                self.push_log(format!("▸ building {label}"));
            }
            Event::Serving {
                model,
                tab,
                shape,
                adapter,
                endpoint,
            } => {
                // The interceptor is serving a model: tell the tab it belongs
                // to what it is talking to.
                let target =
                    tab.and_then(|number| self.tabs.iter().position(|t| t.number == number));
                let index = target.unwrap_or(self.active);
                if let Some(tab) = self.tabs.get_mut(index) {
                    tab.model = Some(model.clone());
                    tab.title = model.clone();
                    tab.shape = shape.clone();
                    tab.adapter = adapter.clone();
                    tab.endpoint = endpoint.clone();
                }
                self.push_log(format!("▸ serving {model}"));
            }
            Event::Log { text } => {
                // The engine's own console goes to the terminal the user is
                // looking at; the panel is a dashboard, not a second copy of
                // it. Only the structured events are kept here.
                let _ = text;
            }
            Event::Scores { phase, scores } => {
                let display = scores
                    .iter()
                    .map(|s| format!("{} {}", s.name, s.display))
                    .collect::<Vec<_>>()
                    .join(" · ");
                if let Some(engine) = self.engine_mut(id) {
                    if phase.eq_ignore_ascii_case("baseline") {
                        engine.baseline = Some(display.clone());
                    } else {
                        engine.trial = Some(display.clone());
                    }
                }
                self.push_log(format!("{phase}: {display}"));
            }
            Event::TrialStart {
                number, modifier, ..
            } => {
                let number = number.map(|n| n.to_string()).unwrap_or_default();
                if let Some(engine) = self.engine_mut(id) {
                    engine.phase = format!("trial {number} · {modifier}");
                    engine.state = "building".into();
                }
                self.push_log(format!("trial {number} · {modifier}"));
            }
            Event::TrialModify { modifier, .. } => {
                if let Some(engine) = self.engine_mut(id) {
                    engine.phase = format!("applying {modifier}");
                }
                self.push_log(format!("applying {modifier}"));
            }
            Event::Question {
                kind,
                message,
                payload,
                ..
            } => {
                let choices = payload
                    .choices
                    .iter()
                    .map(|c| c.title.clone())
                    .collect::<Vec<_>>();
                self.question = Some(Pending {
                    engine: id,
                    kind,
                    message: message.clone(),
                    choices,
                });
                self.push_log(format!("? {message}"));
            }
            Event::Probe {
                name,
                matches,
                total,
                summary,
            } => {
                // The model was asked the questions it was judged on, and
                // this is how it answered. The panel says so, because a model
                // that no longer refuses is the whole point of the thing.
                let note = summary.unwrap_or_else(|| format!("{name} {matches}/{total}"));
                for tab in self.tabs.iter_mut() {
                    if tab.probe.is_none() {
                        tab.probe = Some(note.clone());
                    }
                }
                self.push_log(note);
            }
            Event::Done { exit_code } => {
                if let Some(engine) = self.engine_mut(id) {
                    engine.state = if exit_code == 0 {
                        "done".into()
                    } else {
                        format!("failed ({exit_code})")
                    };
                }
                self.push_log(format!("engine finished with exit code {exit_code}"));
                // A build that finished has changed what is on the machine:
                // there is an adapter for that shape now. The list of models
                // is looked for again, so the panel does not go on saying
                // "build" for a model that is ready.
                if exit_code == 0 {
                    self.catalogue_stale = true;
                }
            }
            Event::Error { error } => {
                if let Some(engine) = self.engine_mut(id) {
                    engine.state = "failed".into();
                }
                self.push_log(format!("engine failed: {error}"));
            }
            Event::Unknown => {}
        }
    }

    /// The answer to a pending question, if there is one.
    ///
    /// The engine asks the same question again after every finished action -
    /// that is how upstream's loop works. The first answer picks the trial
    /// the engine suggests; a repeat is answered with an empty answer, which
    /// is upstream's own signal to stop.
    pub fn take_answer(&mut self) -> Option<(u64, Answer)> {
        let pending = self.question.take()?;
        let times = self.asked.entry(pending.message.clone()).or_insert(0);
        *times += 1;
        let repeat = *times > 1;

        let answer = match pending.kind {
            QuestionKind::Select => {
                if repeat {
                    Answer::text("")
                } else {
                    Answer::index(0)
                }
            }
            QuestionKind::Checkbox => Answer::indices(&[]),
            QuestionKind::Text | QuestionKind::Path | QuestionKind::Password => Answer::text(""),
            QuestionKind::Confirm => Answer::confirm(true),
        };
        Some((pending.engine, answer))
    }

    fn engine_mut(&mut self, id: u64) -> Option<&mut Engine> {
        if !self.engines.iter().any(|e| e.id == id) {
            self.engines.push(Engine::new(id));
        }
        self.engines.iter_mut().find(|e| e.id == id)
    }

    fn push_log(&mut self, line: String) {
        self.log.push(line);
        if self.log.len() > 400 {
            let excess = self.log.len() - 400;
            self.log.drain(..excess);
        }
    }

    /// The engines and their scores, ready to be drawn.
    pub fn summary(&self) -> Vec<String> {
        let mut out = Vec::new();
        for engine in &self.engines {
            let mut line = format!("{} [{}]", engine.label, engine.state);
            if !engine.phase.is_empty() {
                line.push_str(&format!(" · {}", engine.phase));
            }
            if let Some(baseline) = &engine.baseline {
                line.push_str(&format!("  {baseline}"));
            }
            if let Some(trial) = &engine.trial {
                line.push_str(&format!(" → {trial}"));
            }
            out.push(line);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermetic_core::source::{ModelSource, SourceKind};

    fn entry(name: &str) -> CatalogueEntry {
        CatalogueEntry {
            name: name.to_string(),
            signature: None,
            size: 1024,
            kind: SourceKind::Ollama,
            state: "build".into(),
            source: ModelSource::Ollama {
                name: name.to_string(),
            },
        }
    }

    #[test]
    fn a_click_on_a_tab_finds_the_tab() {
        let mut app = App::new(tab_stub());
        app.tab_buttons = vec![(0, 10, 5), (usize::MAX, 16, 5)];
        app.tab_strip_y = 3;

        assert_eq!(app.hit_tab(11, 3), Some(0));
        assert_eq!(app.hit_tab(17, 3), Some(usize::MAX));
        // One column past the button is a miss.
        assert_eq!(app.hit_tab(15, 3), None);
        // The right row, the wrong column.
        assert_eq!(app.hit_tab(11, 4), None);
    }

    #[test]
    fn a_click_on_a_model_finds_the_model() {
        let mut app = App::new(tab_stub());
        app.model_rows = vec![(0, 1, 6, 44), (1, 1, 7, 44)];

        assert_eq!(app.hit_model(5, 6), Some(0));
        assert_eq!(app.hit_model(5, 7), Some(1));
        assert_eq!(app.hit_model(5, 8), None);
        assert_eq!(app.hit_model(0, 6), None);
    }

    #[test]
    fn the_filter_narrows_the_list_down() {
        let mut app = App::new(tab_stub());
        app.set_catalogue(vec![
            entry("qwen2.5:0.5b"),
            entry("llama3.2:1b"),
            entry("qwen3:8b"),
        ]);
        assert_eq!(app.matches.len(), 3);

        app.push_filter('q');
        assert_eq!(app.matches.len(), 2);
        assert_eq!(app.selected_model().unwrap().name, "qwen2.5:0.5b");

        // The filter is a plain substring, so it has to be typed out.
        for character in ['w', 'e', 'n', '3'] {
            app.push_filter(character);
        }
        assert_eq!(app.matches.len(), 1);
        assert_eq!(app.selected_model().unwrap().name, "qwen3:8b");

        app.pop_filter();
        assert_eq!(app.matches.len(), 2);
    }

    #[test]
    fn an_empty_filter_lets_everything_through() {
        let mut app = App::new(tab_stub());
        app.set_catalogue(vec![entry("a"), entry("b")]);
        app.push_filter('z');
        assert!(app.matches.is_empty());
        assert!(app.selected_model().is_none());
    }

    /// A tab that does not need a shell.
    fn tab_stub() -> Tab {
        // The shell is only used for its screen, and a test never draws one.
        let shell = crate::shell::Shell::spawn("/bin/sh", &std::env::temp_dir(), 8, 20, &[], None)
            .expect("start a shell");
        Tab::new(shell, 8, 20, 1)
    }
}
