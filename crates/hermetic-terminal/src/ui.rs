//! Drawing the terminal.
//!
//! Three things on screen: the shell, which is a real terminal the user types
//! into; a side panel with the tabs, the engines and the model list; and a
//! status line. Everything is drawn in low-contrast greys so the shell's own
//! output is what stands out.

use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::ansi;
use crate::app::App;
use crate::catalogue::CatalogueEntry;

/// How wide the side panel is.
pub const PANEL_WIDTH: u16 = 46;

/// The size of the shell's pane: the window minus the panel and the status
/// line.
///
/// The shell's pseudo-terminal has to be exactly this size. When it is not,
/// the shell draws wider lines than the pane and they are cut off at the
/// panel - which is exactly what a clipped terminal looks like.
pub fn shell_size(area: Rect, show_panel: bool) -> (u16, u16) {
    let rows = area.height.saturating_sub(1);
    let roomy = area.width >= PANEL_WIDTH + 48;
    let cols = if show_panel && roomy {
        area.width - PANEL_WIDTH
    } else {
        area.width
    };
    (rows, cols)
}

/// The colours the panel uses.
const PANEL_BG: Color = Color::Indexed(234);
const PANEL_BORDER: Color = Color::Indexed(240);
const PRIMARY: Color = Color::Indexed(252);
const SECONDARY: Color = Color::Indexed(246);
const ACCENT: Color = Color::Indexed(80);
const BAR_BG: Color = Color::Indexed(236);
const BAR_FG: Color = Color::Indexed(248);

/// Draw the whole screen.
pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    if area.height < 3 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(area);

    let roomy = app.show_panel && area.width >= PANEL_WIDTH + 48;

    if roomy {
        // The panel is a fixed width; the shell takes the rest. Which one is
        // on the left is the user's choice.
        let columns = if app.panel_left {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(PANEL_WIDTH), Constraint::Min(20)])
                .split(rows[0])
        } else {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(20), Constraint::Length(PANEL_WIDTH)])
                .split(rows[0])
        };
        if app.panel_left {
            draw_panel(frame, columns[0], app);
            draw_shell(frame, columns[1], app);
        } else {
            draw_shell(frame, columns[0], app);
            draw_panel(frame, columns[1], app);
        }
    } else {
        draw_shell(frame, rows[0], app);
    }

    draw_status(frame, rows[1], app);

    // Where the clickable things are, so a click can be matched to one and a
    // test can find them.
    app.regions = describe_regions(app, area);
}

/// A description of every clickable thing on screen.
///
/// This is what the terminal writes to `HERMETIC_DEBUG_REGIONS` when it is
/// set. It is also how the tests find the coordinates to click.
fn describe_regions(app: &App, area: Rect) -> String {
    let mut out = format!("size {}x{}\n", area.width, area.height);
    for (index, x, width) in &app.tab_buttons {
        if *index == usize::MAX {
            out.push_str(&format!("tab new {x} {} {width}\n", app.tab_strip_y));
        } else {
            out.push_str(&format!("tab {index} {x} {} {width}\n", app.tab_strip_y));
        }
    }
    for (index, x, y, width) in &app.model_rows {
        let name = app
            .catalogue
            .get(*index)
            .map(|e| e.name.as_str())
            .unwrap_or("");
        out.push_str(&format!("model {index} {x} {y} {width} {name}\n"));
    }
    out
}

/// The shell's screen.
fn draw_shell(frame: &mut Frame, area: Rect, app: &mut App) {
    let tab = app.active_tab();
    let screen = tab.grid.screen();
    let width = area.width.min(tab.grid_cols);
    let rows: Vec<Vec<u8>> = screen.rows_formatted(0, width).collect();

    for (index, row) in rows.iter().enumerate() {
        if index as u16 >= area.height {
            break;
        }
        let text = String::from_utf8_lossy(row);
        let line = Line::from(ansi::spans(&text));
        let rect = Rect {
            x: area.x,
            y: area.y + index as u16,
            width: area.width,
            height: 1,
        };
        frame.render_widget(Paragraph::new(line), rect);
    }

    // Show where the shell's cursor is, so the user can see what they type.
    if app.question.is_none() {
        let (row, column) = screen.cursor_position();
        if row < area.height && column < area.width {
            frame.set_cursor_position(Position::new(area.x + column, area.y + row));
        }
    }
}

/// The side panel: the tabs, what is loaded, and every model that is there.
///
/// Nothing here is allowed to wrap. A wrapped line pushes every line below it
/// down one row, and the clickable rectangles - which are counted from the
/// top of the panel - would no longer line up with what is on screen. So
/// every line is cut to the width of the panel instead.
fn draw_panel(frame: &mut Frame, area: Rect, app: &mut App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" HERMETIC ")
        .title_style(Style::default().fg(PRIMARY).add_modifier(Modifier::BOLD))
        .border_style(Style::default().fg(PANEL_BORDER))
        .style(Style::default().bg(PANEL_BG).fg(PRIMARY));
    let inner = block.inner(area);
    frame.render_widget(Clear, area);
    frame.render_widget(block, area);

    let width = inner.width as usize;

    // The head: the tabs, what is loaded and what the engine is doing.
    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(tab_strip(app, inner.x, inner.y, inner.width));
    lines.push(Line::from(""));

    let tab = app.active_tab();
    if let Some(model) = &tab.model {
        lines.push(head_line("LOADED"));
        lines.push(detail_line(&format!(
            "  {}",
            shorten(model, width.saturating_sub(2))
        )));
        if let Some(shape) = &tab.shape {
            lines.push(detail_line(&shorten(shape, width)));
        }
        if let Some(adapter) = &tab.adapter {
            lines.push(detail_line(&format!(
                "  adapter  {}",
                shorten(file_name(adapter), width.saturating_sub(11))
            )));
        }
        if let Some(endpoint) = &tab.endpoint {
            lines.push(detail_line(&format!(
                "  serving  {}",
                shorten(endpoint, width.saturating_sub(10))
            )));
        }
        lines.push(Line::from(""));
    }

    // What the engine is doing.
    if !app.engines.is_empty() {
        lines.push(head_line("ENGINE"));
        for line in app.summary() {
            lines.push(detail_line(&shorten(&line, width)));
        }
        lines.push(Line::from(""));
    } else if tab.model.is_none() {
        lines.push(detail_line("Type a model command, or click one below."));
        lines.push(Line::from(""));
    }

    // The list of models.
    let total = app.catalogue.len();
    let shown = app.matches.len();
    let filter = app.list_filter.trim();
    let header = if total == 0 {
        "MODELS — looking for them...".to_string()
    } else if filter.is_empty() {
        format!("MODELS ({total})")
    } else {
        format!(
            "MODELS ({shown}/{total}) · \"{}\"",
            shorten(filter, width.saturating_sub(14))
        )
    };
    lines.push(head_line(&header));

    // How many of them can be loaded right now.
    if total > 0 {
        let mut ready = 0usize;
        let mut merged = 0usize;
        let mut build = 0usize;
        for entry in &app.catalogue {
            match entry.state.as_str() {
                "ready" => ready += 1,
                "merged" => merged += 1,
                _ => build += 1,
            }
        }
        let mut counts = Vec::new();
        if ready > 0 {
            counts.push(format!("{ready} ready"));
        }
        if merged > 0 {
            counts.push(format!("{merged} merged"));
        }
        if build > 0 {
            counts.push(format!("{build} to build"));
        }
        lines.push(detail_line(&shorten(&counts.join(" · "), width)));
    }

    // One line for the hint at the bottom.
    let hint = 1usize;
    let used = lines.len();
    let mut visible = (inner.height as usize).saturating_sub(used + hint);
    if visible == 0 {
        visible = 1;
    }

    // Keep the selection in view.
    if app.list_selected < app.list_scroll {
        app.list_scroll = app.list_selected;
    } else if app.list_selected >= app.list_scroll + visible {
        app.list_scroll = app.list_selected + 1 - visible;
    }
    if app.list_scroll + visible > shown {
        app.list_scroll = shown.saturating_sub(visible);
    }

    // The rows start right below the header.
    let first_row = inner.y + used as u16;
    app.model_rows.clear();
    for offset in 0..visible {
        let position = app.list_scroll + offset;
        let Some(catalogue_index) = app.matches.get(position).copied() else {
            break;
        };
        let Some(entry) = app.catalogue.get(catalogue_index) else {
            break;
        };
        let selected = position == app.list_selected;
        let hovered = app.list_hover == Some(catalogue_index);
        let style = if selected && app.list_focus {
            Style::default().fg(Color::Indexed(232)).bg(ACCENT)
        } else if selected {
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
        } else if hovered {
            Style::default()
                .fg(PRIMARY)
                .add_modifier(Modifier::UNDERLINED)
        } else {
            Style::default().fg(SECONDARY)
        };
        let row = model_row(entry);
        lines.push(Line::from(Span::styled(shorten(&row, width), style)));
        app.model_rows.push((
            catalogue_index,
            inner.x,
            first_row + offset as u16,
            inner.width,
        ));
    }

    // The hint, at the very bottom.
    while lines.len() + 1 < inner.height as usize {
        lines.push(Line::from(""));
    }
    lines.push(Line::from(Span::styled(
        "click a model to open it in a tab",
        Style::default().fg(Color::Indexed(244)),
    )));

    frame.render_widget(Paragraph::new(lines), inner);
}

/// One row of the model list: its name, how big it is, where it comes from
/// and what the terminal can do with it.
fn model_row(entry: &CatalogueEntry) -> String {
    format!(
        "{:<17} {:>8} {:<6} {:<9}",
        shorten(&entry.name, 17),
        size_label(entry.size),
        entry.kind.label(),
        entry.state,
    )
}

/// A heading in the panel.
fn head_line(text: &str) -> Line<'static> {
    Line::from(Span::styled(
        text.to_string(),
        Style::default().fg(PRIMARY).add_modifier(Modifier::BOLD),
    ))
}

/// A line of detail in the panel.
fn detail_line(text: &str) -> Line<'static> {
    Line::from(Span::styled(
        text.to_string(),
        Style::default().fg(SECONDARY),
    ))
}

/// How big a model is.
fn size_of(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes}{}", UNITS[unit])
    } else {
        format!("{value:.1}{}", UNITS[unit])
    }
}

/// A size that fits in eight characters.
fn size_label(bytes: u64) -> String {
    let text = size_of(bytes);
    if text.chars().count() > 8 {
        let trimmed = text.chars().take(7).collect::<String>();
        let number: f64 = trimmed
            .trim_end_matches(|c: char| c.is_ascii_alphabetic())
            .parse()
            .unwrap_or(0.0);
        let unit = text
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_alphabetic())
            .collect::<String>();
        format!("{number:.0}{unit}")
    } else {
        text
    }
}

/// The tab strip: one button per tab, and a new-tab button.
///
/// The buttons are recorded as they are drawn - where each one starts and how
/// wide it is - so that a click can be matched to a tab. The strip never
/// wraps: a tab that does not fit is left out, and the new-tab button is
/// always kept.
fn tab_strip(app: &mut App, origin_x: u16, origin_y: u16, width: u16) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    app.tab_buttons.clear();
    app.tab_strip_x = origin_x;
    app.tab_strip_y = origin_y;

    let limit = origin_x + width;
    let mut x = origin_x;

    // Leave room for the new-tab button and the gap before it.
    let tab_limit = limit.saturating_sub(6);

    for (index, tab) in app.tabs.iter().enumerate() {
        let active = index == app.active;
        let style = if active {
            Style::default()
                .fg(Color::Indexed(232))
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(SECONDARY)
        };
        let label = format!(" {} ", shorten(&tab.title, 14));
        let w = label.chars().count() as u16;
        if x + w > tab_limit {
            break;
        }
        spans.push(Span::styled(label, style));
        app.tab_buttons.push((index, x, w));
        x += w;
        if x < tab_limit {
            spans.push(Span::styled(" ", Style::default()));
            x += 1;
        }
    }

    // The new tab button.
    if x + 5 <= limit {
        spans.push(Span::styled(
            " NEW ",
            Style::default()
                .fg(Color::Indexed(232))
                .bg(Color::Indexed(66))
                .add_modifier(Modifier::BOLD),
        ));
        app.tab_buttons.push((usize::MAX, x, 5));
    }

    Line::from(spans)
}

/// A tab title that fits in the strip.
fn shorten(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let mut out: String = text.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// The last part of a path: the file's name.
fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// The status line.
fn draw_status(frame: &mut Frame, area: Rect, app: &mut App) {
    let style = Style::default().fg(BAR_FG).bg(BAR_BG);
    let mut spans = vec![
        Span::styled(
            " HERMETIC ",
            Style::default()
                .fg(Color::Indexed(232))
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" ", style),
    ];

    let title = app.active_tab().title.clone();
    spans.push(Span::styled(title, style.add_modifier(Modifier::BOLD)));
    if !app.status.is_empty() {
        spans.push(Span::styled(format!("  ·  {}", app.status), style));
    }

    spans.push(Span::styled(
        "   F1 panel · Ctrl-T new tab · Alt-M search · Ctrl-\\ quit ",
        Style::default().fg(Color::Indexed(244)).bg(BAR_BG),
    ));

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermetic_core::source::{ModelSource, SourceKind};

    fn entry(name: &str, state: &str) -> CatalogueEntry {
        CatalogueEntry {
            name: name.to_string(),
            signature: None,
            size: 2 * 1024 * 1024 * 1024,
            kind: SourceKind::Ollama,
            state: state.to_string(),
            source: ModelSource::Ollama {
                name: name.to_string(),
            },
        }
    }

    #[test]
    fn a_model_row_fits_in_the_panel() {
        let row = model_row(&entry("qwen2.5:0.5b", "ready"));
        assert!(row.chars().count() <= 43, "the row is too wide: {row}");
        assert!(row.contains("qwen2.5:0.5b"));
        assert!(row.contains("ollama"));
        assert!(row.contains("ready"));
    }

    #[test]
    fn a_long_name_is_cut_rather_than_wrapped() {
        let row = model_row(&entry("a-very-long-model-name-that-does-not-fit", "build"));
        assert!(row.chars().count() <= 43, "the row is too wide: {row}");
    }
}
