//! HERMETIC - a terminal that talks to every model runtime.
//!
//! The terminal hosts shells in pseudo-terminals, so the user types at a real
//! prompt. A shell routes model commands (`hermetic run <model>`, `ollama run
//! <model>`, `llama-cli -m <file>`) to the interceptor, which resolves the
//! model, makes sure a Heretic adapter exists for that model's shape, serves
//! the model with the adapter bound to it and talks to it.
//!
//! Every shell is a tab. What the engine does while it works is reported over
//! a unix socket: the terminal keeps a side panel with the phase, the scores
//! and the log.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event as CrosstermEvent, KeyCode, KeyEvent,
    KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use hermetic_terminal::{
    app, catalogue, channel, repo_root, shell, sibling_binary, ui, ENGINE_ENV, HOME_ENV,
    OPEN_CMD_ENV, REGIONS_ENV, SHIM_ENV, SOCKET_ENV, TAB_ENV,
};

#[derive(Parser, Debug)]
#[command(
    name = "hermetic",
    version,
    about = "HERMETIC - a terminal that talks to every model runtime"
)]
struct Cli {
    /// The shell to host.
    #[arg(long, default_value = "")]
    shell: String,

    /// Start with the side panel hidden.
    #[arg(long)]
    no_panel: bool,

    /// Print what the terminal is doing and exit.
    #[arg(long)]
    check: bool,

    /// Load a model straight away, as if it had been typed at the prompt.
    #[arg(long)]
    run: Option<String>,

    /// Which side the panel is on.
    #[arg(long, value_parser = ["left", "right"], default_value = "left")]
    panel_side: String,

    /// A folder of GGUF files to offer. May be given more than once.
    #[arg(long = "gguf-dir")]
    gguf_dirs: Vec<PathBuf>,

    /// A folder of MLX models to offer. May be given more than once.
    #[arg(long = "mlx-dir")]
    mlx_dirs: Vec<PathBuf>,

    /// A server that is already running, as `url` or `url#model`.
    #[arg(long = "remote")]
    remote: Vec<String>,
}

/// Where the interceptor lives.
///
/// `HERMETIC_SHIM` says where it is, if somebody wants a different one.
fn shim_path() -> PathBuf {
    if let Some(path) = std::env::var_os(SHIM_ENV) {
        return PathBuf::from(path);
    }
    sibling_binary("hermetic-shim").unwrap_or_else(|| {
        repo_root()
            .join("target")
            .join("release")
            .join("hermetic-shim")
    })
}

/// The user's shell.
fn user_shell(cli: &Cli) -> String {
    if !cli.shell.is_empty() {
        return cli.shell.clone();
    }
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".into());
    if shell.ends_with("zsh") || shell.ends_with("bash") {
        shell
    } else {
        "/bin/zsh".into()
    }
}

/// Everything a tab needs to start a shell.
struct Session {
    shell_program: String,
    home: PathBuf,
    zdotdir: PathBuf,
    socket: PathBuf,
    shim: PathBuf,
    registry_root: PathBuf,
    engine_dir: PathBuf,
    next_number: u64,
}

impl Session {
    /// Start a new shell and wrap it in a tab.
    fn new_tab(&mut self, rows: u16, cols: u16) -> Result<app::Tab> {
        let number = self.next_number;
        self.next_number += 1;

        let env = vec![
            (SOCKET_ENV.to_string(), self.socket.display().to_string()),
            (SHIM_ENV.to_string(), self.shim.display().to_string()),
            (
                HOME_ENV.to_string(),
                self.registry_root.display().to_string(),
            ),
            (
                ENGINE_ENV.to_string(),
                self.engine_dir.display().to_string(),
            ),
            // So the interceptor can say which tab it belongs to.
            (TAB_ENV.to_string(), number.to_string()),
        ];

        let sh = shell::Shell::spawn(
            &self.shell_program,
            &self.home,
            rows,
            cols,
            &env,
            Some(&self.zdotdir),
        )
        .context("start the shell")?;

        Ok(app::Tab::new(sh, rows, cols, number))
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = hermetic_core::config::Config::load();
    let registry = hermetic_core::Registry::new();

    let run_dir = registry.root.join("run");
    std::fs::create_dir_all(&run_dir).context("create the run directory")?;

    let shim = shim_path();
    if !shim.exists() {
        anyhow::bail!(
            "the interceptor is missing: {} — build the workspace first",
            shim.display()
        );
    }

    // The socket the engines report to. It has to be short: a unix socket
    // path is limited to about a hundred characters.
    let socket_path = PathBuf::from("/tmp").join(format!("hermetic-{}.sock", std::process::id()));
    let hub = channel::Hub::bind(&socket_path).context("bind the event channel")?;

    let ollama_bin = hermetic_terminal::which("ollama").unwrap_or_else(|| PathBuf::from("ollama"));
    let llama_cli_bin =
        hermetic_terminal::which("llama-cli").unwrap_or_else(|| PathBuf::from("llama-cli"));
    let injection = shell::injection(&shim, &socket_path, &ollama_bin, &llama_cli_bin);
    let zdotdir = shell::zdotdir(&run_dir, &injection)?;

    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/"));

    let (cols, rows) = crossterm::terminal::size().unwrap_or((100, 30));
    let (shell_rows, shell_cols) = ui::shell_size(
        ratatui::layout::Rect {
            x: 0,
            y: 0,
            width: cols,
            height: rows,
        },
        !cli.no_panel,
    );

    let remote: Vec<(String, String)> = cli
        .remote
        .iter()
        .map(|text| catalogue::remote_spec(text))
        .collect();
    let sources = catalogue::sources(&cli.gguf_dirs, &cli.mlx_dirs, &remote);

    let mut session = Session {
        shell_program: user_shell(&cli),
        home,
        zdotdir: zdotdir.clone(),
        socket: socket_path.clone(),
        shim,
        registry_root: registry.root.clone(),
        engine_dir: config.engine_dir.clone(),
        next_number: 1,
    };

    let first = session.new_tab(shell_rows, shell_cols)?;

    if cli.check {
        println!("shell:    {}", session.shell_program);
        println!("shim:     {}", session.shim.display());
        println!("engine:   {}", session.engine_dir.display());
        println!("registry: {}", session.registry_root.display());
        println!("channel:  {}", session.socket.display());
        return Ok(());
    }

    // A model asked for on the command line is typed at the prompt, so it
    // goes through exactly the same path as one the user types.
    if let Some(model) = &cli.run {
        first
            .shell
            .write_line(&format!("hermetic run {}", quote(model)));
    }

    let result = interactive(
        &hub,
        &mut session,
        first,
        cli.no_panel,
        cli.panel_side != "right",
        &sources,
    );

    let _ = std::fs::remove_file(&socket_path);
    let _ = std::fs::remove_dir_all(&zdotdir);
    result
}

/// Run the terminal.
fn interactive(
    hub: &channel::Hub,
    session: &mut Session,
    first: app::Tab,
    no_panel: bool,
    panel_left: bool,
    sources: &hermetic_core::discovery::Sources,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let outcome = event_loop(
        &mut terminal,
        hub,
        session,
        first,
        no_panel,
        panel_left,
        sources,
    );

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    outcome
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    hub: &channel::Hub,
    session: &mut Session,
    first: app::Tab,
    no_panel: bool,
    panel_left: bool,
    sources: &hermetic_core::discovery::Sources,
) -> Result<()> {
    let _size = terminal.size()?;
    let mut app = app::App::new(first);
    app.show_panel = !no_panel;
    app.panel_left = panel_left;
    app.status = "ready".into();
    app.regions_path = std::env::var_os(REGIONS_ENV).map(PathBuf::from);
    app.sources = sources.clone();

    // The model list is read in the background: it is ninety models and a
    // second of work, and the terminal should not wait for it.
    let (catalogue_tx, catalogue_rx) = std::sync::mpsc::channel();
    app.catalogue_tx = catalogue_tx.clone();
    app.catalogue_rx = catalogue_rx;
    let sources = sources.clone();
    let registry = app.registry.clone();
    std::thread::spawn(move || {
        let _ = catalogue_tx.send(catalogue::catalogue(&sources, &registry));
    });

    // The first shell was started with the right size, but the panel can be
    // hidden, so make sure every shell agrees with its pane.
    resize_all(&mut app, session, terminal)?;

    loop {
        // The model list, when it is ready.
        while let Ok(entries) = app.catalogue_rx.try_recv() {
            app.set_catalogue(entries);
        }

        // What the engines said.
        while let Ok((id, event)) = hub.events.try_recv() {
            app.on_event(id, event);
        }

        // Answer a question, if one is waiting.
        if let Some((id, answer)) = app.take_answer() {
            let _ = hub.send_answer(id, &answer);
            app.status = "answered the engine".into();
        }

        // What every shell wrote. A background tab keeps its screen up to
        // date even when it is not in front.
        for tab in app.tabs.iter_mut() {
            while let Ok(bytes) = tab.shell.output.try_recv() {
                tab.feed(&bytes);
            }
        }

        // Input.
        if event::poll(Duration::from_millis(40))? {
            match event::read()? {
                CrosstermEvent::Key(key) => {
                    if key.kind == KeyEventKind::Press {
                        handle_key(&mut app, key, session, terminal)?;
                    }
                }
                CrosstermEvent::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::Down(MouseButton::Left) => {
                        handle_click(
                            &mut app,
                            mouse.column,
                            mouse.row,
                            mouse.modifiers,
                            session,
                            terminal,
                        )?;
                    }
                    MouseEventKind::ScrollUp => app.scroll_list(-3),
                    MouseEventKind::ScrollDown => app.scroll_list(3),
                    MouseEventKind::Moved => {
                        let hover = app.hit_model(mouse.column, mouse.row);
                        app.list_hover = hover;
                    }
                    _ => {}
                },
                CrosstermEvent::Resize(width, height) => {
                    let (rows, cols) = ui::shell_size(
                        ratatui::layout::Rect {
                            x: 0,
                            y: 0,
                            width,
                            height,
                        },
                        app.show_panel,
                    );
                    for tab in app.tabs.iter_mut() {
                        tab.resize(rows, cols);
                        tab.shell.resize(rows, cols);
                    }
                }
                _ => {}
            }
        }

        terminal.draw(|frame| ui::draw(frame, &mut app))?;

        // Where the clickable things are, when somebody wants to know.
        if app.regions_path.is_some() && app.regions != app.last_regions {
            app.last_regions = app.regions.clone();
            if let Some(path) = app.regions_path.clone() {
                let _ = std::fs::write(path, &app.regions);
            }
        }

        if app.quit {
            return Ok(());
        }
    }
}

/// Handle one key press.
fn handle_key(
    app: &mut app::App,
    key: KeyEvent,
    session: &mut Session,
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
) -> Result<()> {
    // Some terminals report both press and release; act on the press only.
    if key.kind != KeyEventKind::Press {
        return Ok(());
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);

    // The terminal's own keys.
    if ctrl {
        match key.code {
            KeyCode::Char('\\') => {
                app.quit = true;
                return Ok(());
            }
            KeyCode::Char('t') => {
                new_tab(app, session, terminal)?;
                return Ok(());
            }
            KeyCode::Char('w') => {
                let index = app.active;
                app.close_tab(index);
                resize_all(app, session, terminal)?;
                return Ok(());
            }
            KeyCode::Tab => {
                if key.modifiers.contains(KeyModifiers::SHIFT) {
                    app.previous_tab();
                } else {
                    app.next_tab();
                }
                return Ok(());
            }
            _ => {}
        }
    }
    // Alt-1 .. Alt-9 jump to a tab.
    if key.modifiers.contains(KeyModifiers::ALT) {
        if let KeyCode::Char(c) = key.code {
            if let Some(index) = c.to_digit(10) {
                let index = index as usize - 1;
                if index < app.tabs.len() {
                    app.active = index;
                }
                return Ok(());
            }
            match c {
                // Alt-M drives the model list with the keyboard.
                'm' | 'M' => {
                    app.list_focus = !app.list_focus;
                    app.status = if app.list_focus {
                        "the model list has the keyboard — type to search, enter to open"
                            .to_string()
                    } else {
                        "ready".into()
                    };
                    return Ok(());
                }
                // Alt-R looks for the models again.
                'r' | 'R' => {
                    refresh_catalogue(app);
                    return Ok(());
                }
                _ => {}
            }
        }
    }

    // The model list, when it has the keyboard.
    if app.list_focus {
        match key.code {
            KeyCode::Up => {
                app.move_selection(-1);
                return Ok(());
            }
            KeyCode::Down => {
                app.move_selection(1);
                return Ok(());
            }
            KeyCode::PageUp => {
                app.move_selection(-10);
                return Ok(());
            }
            KeyCode::PageDown => {
                app.move_selection(10);
                return Ok(());
            }
            KeyCode::Enter => {
                if let Some(model) = app.selected_model().map(|e| e.spec()) {
                    match open_model_in_new_tab(app, session, terminal, &model) {
                        Ok(()) => app.status = format!("{model} is loading in a new tab"),
                        Err(error) => app.status = format!("could not open {model}: {error}"),
                    }
                }
                return Ok(());
            }
            KeyCode::Esc => {
                app.list_focus = false;
                app.list_filter.clear();
                app.refilter();
                app.status = "ready".into();
                return Ok(());
            }
            KeyCode::Backspace => {
                app.pop_filter();
                return Ok(());
            }
            // Anything else typed narrows the list down.
            KeyCode::Char(c) if !ctrl && !alt => {
                app.push_filter(c);
                app.status = format!("looking for \"{}\"", app.list_filter);
                return Ok(());
            }
            _ => {}
        }
    }
    if key.code == KeyCode::F(1) {
        app.show_panel = !app.show_panel;
        resize_all(app, session, terminal)?;
        return Ok(());
    }
    if key.code == KeyCode::F(2) {
        new_tab(app, session, terminal)?;
        return Ok(());
    }

    // Everything else belongs to the shell.
    if let Some(bytes) = key_bytes(key) {
        app.active_tab().shell.write(&bytes);
    }
    Ok(())
}

/// Open a new tab.
fn new_tab(
    app: &mut app::App,
    session: &mut Session,
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
) -> Result<()> {
    let size = terminal.size()?;
    let (rows, cols) = ui::shell_size(
        ratatui::layout::Rect {
            x: 0,
            y: 0,
            width: size.width,
            height: size.height,
        },
        app.show_panel,
    );
    let tab = session.new_tab(rows, cols)?;
    app.push_tab(tab);
    Ok(())
}

/// Tell every shell how big its pane is.
fn resize_all(
    app: &mut app::App,
    session: &mut Session,
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
) -> Result<()> {
    let size = terminal.size()?;
    let (rows, cols) = ui::shell_size(
        ratatui::layout::Rect {
            x: 0,
            y: 0,
            width: size.width,
            height: size.height,
        },
        app.show_panel,
    );
    for tab in app.tabs.iter_mut() {
        tab.resize(rows, cols);
        tab.shell.resize(rows, cols);
    }
    let _ = session;
    Ok(())
}

/// A click on the tab strip or the model list.
fn handle_click(
    app: &mut app::App,
    column: u16,
    row: u16,
    modifiers: KeyModifiers,
    session: &mut Session,
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
) -> Result<()> {
    // The tab strip: a tab, or the new-tab button.
    if let Some(index) = app.hit_tab(column, row) {
        if index == usize::MAX {
            new_tab(app, session, terminal)?;
        } else {
            app.active = index;
        }
        return Ok(());
    }

    // A model in the list: open it.
    if let Some(index) = app.hit_model(column, row) {
        app.list_selected = app
            .matches
            .iter()
            .position(|m| *m == index)
            .unwrap_or(app.list_selected);
        let Some(entry) = app.catalogue.get(index) else {
            return Ok(());
        };
        let name = entry.name.clone();
        let spec = entry.spec();

        // Alt-click opens it in a window of its own; a plain click opens a tab
        // in this window.
        if modifiers.contains(KeyModifiers::ALT) {
            match open_model_in_new_window(&spec) {
                Ok(()) => app.status = format!("opened {name} in a new window"),
                Err(error) => app.status = format!("could not open {name}: {error}"),
            }
        } else {
            match open_model_in_new_tab(app, session, terminal, &spec) {
                Ok(()) => app.status = format!("{name} is loading in a new tab"),
                Err(error) => app.status = format!("could not open {name}: {error}"),
            }
        }
        return Ok(());
    }

    Ok(())
}

/// Open a model in a new tab, and start loading it.
///
/// This is the same thing `--run` does for the first tab: the model is typed
/// at the prompt, so it goes through exactly the same path as one the user
/// typed by hand.
fn open_model_in_new_tab(
    app: &mut app::App,
    session: &mut Session,
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    spec: &str,
) -> Result<()> {
    let size = terminal.size()?;
    let (rows, cols) = ui::shell_size(
        ratatui::layout::Rect {
            x: 0,
            y: 0,
            width: size.width,
            height: size.height,
        },
        app.show_panel,
    );
    let tab = session.new_tab(rows, cols)?;
    tab.shell
        .write_line(&format!("hermetic run {}", quote(spec)));
    app.push_tab(tab);
    Ok(())
}

/// Look for the models again.
fn refresh_catalogue(app: &mut app::App) {
    let tx = app.catalogue_tx.clone();
    app.status = "looking for models...".into();
    let sources = app.sources.clone();
    let registry = app.registry.clone();
    std::thread::spawn(move || {
        let _ = tx.send(catalogue::catalogue(&sources, &registry));
    });
}

/// Open a model in a new HERMETIC window.
///
/// The window is a Terminal window running this same binary with the model
/// already asked for, so it is exactly the same terminal - the same panel,
/// the same engine, the same adapter - with the model loaded.
fn open_model_in_new_window(spec: &str) -> Result<()> {
    let exe = std::env::current_exe().context("find the terminal's own path")?;

    // One script per click: if the same file were rewritten while Terminal was
    // still reading it, the window could start the wrong model.
    let dir = std::env::temp_dir().join(format!("hermetic-open-{}", std::process::id()));
    std::fs::create_dir_all(&dir).context("make the run directory")?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default();
    let script = dir.join(format!("open-{stamp}.command"));

    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\nprintf '\\033]0;HERMETIC · {title}\\007'\nexec {exe} --run {spec}\n",
            title = title_of(spec),
            exe = quote(&exe.display().to_string()),
            spec = quote(spec),
        ),
    )
    .context("write the launcher")?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))?;
    }

    // By default a new Terminal window opens the script. `HERMETIC_OPEN_CMD`
    // says otherwise, which is how the tests watch what would have been run.
    match std::env::var_os(OPEN_CMD_ENV) {
        Some(command) => {
            let text = command.to_string_lossy().to_string();
            let mut parts = text.split_whitespace().map(String::from);
            let program = parts.next().unwrap_or_else(|| "open".into());
            let mut command = std::process::Command::new(program);
            for argument in parts {
                command.arg(argument);
            }
            command.arg(&script).status().context("open a new window")?;
        }
        None => {
            std::process::Command::new("open")
                .args(["-a", "Terminal"])
                .arg(&script)
                .status()
                .context("open a new window")?;
        }
    }

    // Keep only the last few scripts.
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let mut scripts: Vec<_> = entries.flatten().collect();
        scripts.sort_by_key(|e| e.file_name());
        for entry in scripts.iter().rev().skip(20) {
            let _ = std::fs::remove_file(entry.path());
        }
    }

    Ok(())
}

/// A window title for a model.
fn title_of(model: &str) -> String {
    model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | ':' | '-' | '_' | '/') {
                c
            } else {
                '?'
            }
        })
        .collect()
}

/// Quote a string for the shell.
fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

/// The bytes a key press means to a terminal.
fn key_bytes(key: KeyEvent) -> Option<Vec<u8>> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);

    let mut out: Vec<u8> = Vec::new();
    if alt {
        out.push(0x1b);
    }

    let base: Vec<u8> = match key.code {
        KeyCode::Char(c) => {
            if ctrl {
                let lower = c.to_ascii_lowercase();
                if lower == ' ' {
                    vec![0]
                } else if (lower as u8) < 0x20 {
                    vec![lower as u8]
                } else {
                    vec![(lower as u8) & 0x1f]
                }
            } else if c == '\r' {
                vec![13]
            } else if c == '\n' {
                vec![10]
            } else if c == '\t' {
                vec![9]
            } else if c == '\u{7f}' {
                vec![127]
            } else {
                c.to_string().into_bytes()
            }
        }
        KeyCode::Enter => vec![13],
        KeyCode::Tab => vec![9],
        KeyCode::BackTab => vec![0x1b, b'[', b'Z'],
        KeyCode::Backspace => vec![127],
        KeyCode::Esc => vec![27],
        KeyCode::Up => vec![0x1b, b'[', b'A'],
        KeyCode::Down => vec![0x1b, b'[', b'B'],
        KeyCode::Right => vec![0x1b, b'[', b'C'],
        KeyCode::Left => vec![0x1b, b'[', b'D'],
        KeyCode::Home => vec![0x1b, b'[', b'H'],
        KeyCode::End => vec![0x1b, b'[', b'F'],
        KeyCode::PageUp => vec![0x1b, b'[', b'5', b'~'],
        KeyCode::PageDown => vec![0x1b, b'[', b'6', b'~'],
        KeyCode::Delete => vec![0x1b, b'[', b'3', b'~'],
        KeyCode::Insert => vec![0x1b, b'[', b'2', b'~'],
        KeyCode::F(n) => match n {
            1 => vec![0x1b, b'O', b'P'],
            2 => vec![0x1b, b'O', b'Q'],
            3 => vec![0x1b, b'O', b'R'],
            4 => vec![0x1b, b'O', b'S'],
            5 => vec![0x1b, b'[', b'1', b'~'],
            6 => vec![0x1b, b'[', b'2', b'~'],
            7 => vec![0x1b, b'[', b'3', b'~'],
            8 => vec![0x1b, b'[', b'4', b'~'],
            9 => vec![0x1b, b'[', b'5', b'~'],
            10 => vec![0x1b, b'[', b'6', b'~'],
            11 => vec![0x1b, b'[', b'7', b'~'],
            12 => vec![0x1b, b'[', b'8', b'~'],
            _ => return None,
        },
        _ => return None,
    };

    out.extend(base);
    Some(out)
}
