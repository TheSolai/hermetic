//! Does clicking the panel actually do something?
//!
//! The terminal is driven through a real pseudo-terminal: the test writes
//! SGR mouse sequences to it, exactly as a terminal would, and watches what
//! happens. Nothing is mocked, and the binary that runs is the one that
//! ships.
//!
//! The interceptor is a stub, so clicking a model does not really start
//! loading a ninety-gigabyte model in the middle of a test.

use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// The binary under test: it sits next to the test binary's parent.
fn binary(name: &str) -> PathBuf {
    let mut path = std::env::current_exe().expect("the test binary has a path");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.join(name)
}

/// The mouse, as a terminal sends it: a press and a release.
///
/// A terminal counts rows and columns from one; the terminal counts them
/// from zero. So everything is sent one higher than it is on screen.
fn click(writer: &mut Box<dyn Write + Send>, column: u16, row: u16) {
    press(writer, column, row, 0);
}

/// A click with the alt key held down.
fn alt_click(writer: &mut Box<dyn Write + Send>, column: u16, row: u16) {
    press(writer, column, row, 8);
}

fn press(writer: &mut Box<dyn Write + Send>, column: u16, row: u16, button: u16) {
    let column = column + 1;
    let row = row + 1;
    write!(writer, "\x1b[<{button};{column};{row}M").expect("write the press");
    write!(writer, "\x1b[<{button};{column};{row}m").expect("write the release");
    writer.flush().expect("flush the mouse");
}

/// One thing the panel can be clicked on.
#[derive(Debug, Clone)]
struct Region {
    kind: String,
    x: u16,
    y: u16,
    width: u16,
    name: String,
}

/// Read the regions the terminal drew.
fn regions(path: &PathBuf) -> Vec<Region> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in text.lines() {
        let mut parts = line.split(' ');
        match parts.next() {
            Some("tab") => {
                let kind = parts.next().unwrap_or("").to_string();
                let x = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                let y = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                let width = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                out.push(Region {
                    kind,
                    x,
                    y,
                    width,
                    name: String::new(),
                });
            }
            Some("model") => {
                let _index = parts.next();
                let x = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                let y = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                let width = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                let name = parts.collect::<Vec<_>>().join(" ");
                out.push(Region {
                    kind: "model".into(),
                    x,
                    y,
                    width,
                    name,
                });
            }
            _ => {}
        }
    }
    out
}

/// How many tabs are open.
fn tab_count(regions: &[Region]) -> usize {
    regions.iter().filter(|r| r.kind != "new").count()
}

/// Wait for something to happen, or give up.
fn wait_for<T>(mut what: impl FnMut() -> Option<T>, seconds: u64) -> Option<T> {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    loop {
        if let Some(value) = what() {
            return Some(value);
        }
        if Instant::now() > deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Make a script runnable.
fn make_executable(path: &PathBuf) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .expect("make the script runnable");
    }
}

#[test]
fn clicking_the_panel_opens_a_model_in_a_tab() {
    let tmp = std::env::temp_dir().join(format!("hermetic-click-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("make the run directory");
    let regions_file = tmp.join("regions.txt");
    let opened = tmp.join("opened.command");

    // A stand-in for `open -a Terminal`: it copies the script the terminal
    // wanted to run, so the test can look at it.
    let recorder = tmp.join("record.sh");
    std::fs::write(
        &recorder,
        format!("#!/bin/sh\ncp \"$1\" '{}'\n", opened.display()),
    )
    .expect("write the recorder");
    make_executable(&recorder);

    // An interceptor that does nothing, so no model is really loaded.
    let stub = tmp.join("shim.sh");
    std::fs::write(&stub, "#!/bin/sh\necho \"stub: $@\"\n").expect("write the stub");
    make_executable(&stub);

    // The terminal, in a terminal.
    let pty = native_pty_system();
    let pair = pty
        .openpty(PtySize {
            rows: 40,
            cols: 160,
            pixel_width: 0,
            pixel_height: 0,
        })
        .expect("open a pseudo-terminal");
    let mut command = CommandBuilder::new(binary("hermetic"));
    command.env("HERMETIC_DEBUG_REGIONS", &regions_file);
    command.env("HERMETIC_OPEN_CMD", &recorder);
    command.env("HERMETIC_SHIM", &stub);
    command.env("TERM", "xterm-256color");
    let mut child = pair
        .slave
        .spawn_command(command)
        .expect("start the terminal");
    drop(pair.slave);

    let mut reader = pair.master.try_clone_reader().expect("read the terminal");
    let mut writer = pair.master.take_writer().expect("write to the terminal");
    std::thread::spawn(move || {
        let mut buffer = [0u8; 4096];
        while let Ok(count) = reader.read(&mut buffer) {
            if count == 0 {
                break;
            }
        }
    });

    // The model list has to be there before anything can be clicked.
    let drawn = wait_for(
        || {
            let found = regions(&regions_file);
            if found.iter().any(|r| r.kind == "model") {
                Some(found)
            } else {
                None
            }
        },
        60,
    )
    .expect("the panel drew the model list");

    // ── clicking a model opens it in a new tab ─────────────────────────────
    let model = drawn
        .iter()
        .find(|r| r.kind == "model")
        .expect("a model row")
        .clone();
    assert!(!model.name.is_empty(), "the model row knows its name");
    assert!(model.width > 10, "the model row is wide enough to click");

    let tabs_before = tab_count(&drawn);
    click(&mut writer, model.x + 2, model.y);

    let tabs_after = wait_for(
        || {
            let found = regions(&regions_file);
            let count = tab_count(&found);
            if count > tabs_before {
                Some(count)
            } else {
                None
            }
        },
        20,
    )
    .expect("the click opened a new tab");
    assert!(
        tabs_after > tabs_before,
        "the model opened in a tab, not in a window of its own"
    );

    // ── alt-clicking a model opens it in a window of its own ───────────────
    let second = regions(&regions_file)
        .into_iter()
        .find(|r| r.kind == "model")
        .expect("a model row");
    alt_click(&mut writer, second.x + 2, second.y);

    let script = wait_for(|| std::fs::read_to_string(&opened).ok(), 30)
        .expect("the alt-click opened a new window");
    assert!(
        script.contains("--run") && script.contains(&second.name),
        "the new window runs the model that was clicked: {script}"
    );

    // ── clicking NEW adds a tab ───────────────────────────────────────────
    let before = tab_count(&regions(&regions_file));
    let new_button = regions(&regions_file)
        .into_iter()
        .find(|r| r.kind == "new")
        .expect("the new-tab button");
    click(&mut writer, new_button.x + 1, new_button.y);

    let after = wait_for(
        || {
            let found = regions(&regions_file);
            let count = tab_count(&found);
            if count > before {
                Some(count)
            } else {
                None
            }
        },
        20,
    )
    .expect("the click added a tab");

    assert!(after > before, "there is now more than one tab");

    let _ = child.kill();
    let _ = std::fs::remove_dir_all(&tmp);
}
