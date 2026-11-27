//! Just enough ANSI to draw a shell's output with its colours intact.
//!
//! The terminal renders the shell's screen through a vt100 grid. That grid
//! holds plain characters; the colours live in the escape sequences the shell
//! wrote. This turns a row of text with SGR sequences into styled spans.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

/// Parse one line of ANSI text into styled spans.
pub fn spans(line: &str) -> Vec<Span<'static>> {
    let mut out: Vec<Span<'static>> = Vec::new();
    let mut style = Style::default();
    let mut text = String::new();
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if c != '\x1b' {
            text.push(c);
            continue;
        }
        // An escape sequence: CSI (ESC [ ... final byte) or a short one.
        if chars.peek() == Some(&'[') {
            chars.next();
            let mut params = String::new();
            let mut final_byte = None;
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() || c == '@' {
                    final_byte = Some(c);
                    break;
                }
                params.push(c);
            }
            if final_byte != Some('m') {
                // Not a colour change (cursor moves and the like): the grid
                // already applied them, so there is nothing to do here.
                continue;
            }
            if !text.is_empty() {
                out.push(Span::styled(std::mem::take(&mut text), style));
            }
            style = apply_sgr(&style, &params);
            continue;
        }
        // A two-byte escape: consume it.
        chars.next();
    }

    if !text.is_empty() {
        out.push(Span::styled(text, style));
    }
    out
}

/// Apply a `SGR` parameter list to a style.
fn apply_sgr(base: &Style, params: &str) -> Style {
    let mut style = *base;
    let nums: Vec<u32> = params
        .split(';')
        .map(|p| p.trim().parse::<u32>().unwrap_or(0))
        .collect();
    let mut i = 0;
    while i < nums.len() {
        let n = nums[i];
        match n {
            0 => style = Style::default(),
            1 => style = style.add_modifier(Modifier::BOLD),
            2 => style = style.add_modifier(Modifier::DIM),
            3 => style = style.add_modifier(Modifier::ITALIC),
            4 => style = style.add_modifier(Modifier::UNDERLINED),
            7 => style = style.add_modifier(Modifier::REVERSED),
            22 => style = style.remove_modifier(Modifier::BOLD),
            23 => style = style.remove_modifier(Modifier::ITALIC),
            24 => style = style.remove_modifier(Modifier::UNDERLINED),
            27 => style = style.remove_modifier(Modifier::REVERSED),
            30..=37 => style = style.fg(basic_color(n - 30)),
            39 => style = style.fg(Color::Reset),
            40..=47 => style = style.bg(basic_color(n - 40)),
            49 => style = style.bg(Color::Reset),
            90..=97 => style = style.fg(bright_color(n - 90)),
            100..=107 => style = style.bg(bright_color(n - 100)),
            38 | 48 => {
                // Extended colour: 5;<n> (256) or 2;<r>;<g>;<b> (truecolour).
                let target = if n == 38 { true } else { false };
                let mode = nums.get(i + 1).copied().unwrap_or(0);
                let color = match mode {
                    5 => {
                        let idx = nums.get(i + 2).copied().unwrap_or(0);
                        i += 2;
                        Some(indexed(idx))
                    }
                    2 => {
                        let r = nums.get(i + 2).copied().unwrap_or(0) as u8;
                        let g = nums.get(i + 3).copied().unwrap_or(0) as u8;
                        let b = nums.get(i + 4).copied().unwrap_or(0) as u8;
                        i += 4;
                        Some(Color::Rgb(r, g, b))
                    }
                    _ => None,
                };
                if let Some(c) = color {
                    style = if target { style.fg(c) } else { style.bg(c) };
                }
            }
            _ => {}
        }
        i += 1;
    }
    style
}

fn basic_color(n: u32) -> Color {
    match n {
        0 => Color::Black,
        1 => Color::Red,
        2 => Color::Green,
        3 => Color::Yellow,
        4 => Color::Blue,
        5 => Color::Magenta,
        6 => Color::Cyan,
        _ => Color::Gray,
    }
}

fn bright_color(n: u32) -> Color {
    match n {
        0 => Color::DarkGray,
        1 => Color::LightRed,
        2 => Color::LightGreen,
        3 => Color::LightYellow,
        4 => Color::LightBlue,
        5 => Color::LightMagenta,
        6 => Color::LightCyan,
        _ => Color::White,
    }
}

/// The 256-colour palette, in the usual arrangement.
fn indexed(idx: u32) -> Color {
    match idx {
        0..=15 => basic_color(idx % 8),
        16..=231 => {
            let i = idx - 16;
            let steps = [0u8, 95, 135, 175, 215, 255];
            let r = steps[(i / 36) as usize];
            let g = steps[((i % 36) / 6) as usize];
            let b = steps[(i % 6) as usize];
            Color::Rgb(r, g, b)
        }
        232..=255 => {
            let v = 8 + (idx - 232) * 10;
            Color::Rgb(v as u8, v as u8, v as u8)
        }
        _ => Color::Reset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_one_span() {
        let s = spans("hello");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].content, "hello");
    }

    #[test]
    fn a_colour_change_splits_the_line() {
        let s = spans("\x1b[31mred\x1b[0m plain");
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].content, "red");
        assert_eq!(s[0].style.fg, Some(Color::Red));
        assert_eq!(s[1].content, " plain");
    }

    #[test]
    fn truecolour_is_understood() {
        let s = spans("\x1b[38;2;10;20;30mrgb\x1b[0m");
        assert_eq!(s[0].style.fg, Some(Color::Rgb(10, 20, 30)));
    }
}
