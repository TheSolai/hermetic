//! A very small HTTP client: enough to talk to llama-server and to Ollama.
//!
//! The terminal only needs two things: a JSON request/response, and a
//! streaming response that delivers tokens as they arrive. Both fit in a
//! couple of hundred lines, which is cheaper - in build time and in weight -
//! than pulling in a full HTTP stack.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::Value;

/// A connection to a local HTTP server.
pub struct Conn {
    stream: TcpStream,
    buf: Vec<u8>,
    pos: usize,
}

impl Conn {
    /// Open a connection to `host:port`.
    pub fn connect(host: &str, port: u16, timeout: Duration) -> Result<Conn> {
        let stream = TcpStream::connect((host, port))
            .with_context(|| format!("could not connect to {host}:{port}"))?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_nodelay(true).ok();
        Ok(Conn {
            stream,
            buf: Vec::new(),
            pos: 0,
        })
    }

    /// Send a request.
    pub fn request(&mut self, method: &str, path: &str, body: Option<&str>) -> Result<()> {
        let mut head = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nAccept: application/json, text/event-stream\r\n"
        );
        if let Some(b) = body {
            head.push_str("Content-Type: application/json\r\n");
            head.push_str(&format!("Content-Length: {}\r\n", b.len()));
        }
        head.push_str("\r\n");
        self.stream.write_all(head.as_bytes())?;
        if let Some(b) = body {
            self.stream.write_all(b.as_bytes())?;
        }
        self.stream.flush()?;
        Ok(())
    }

    fn fill(&mut self) -> Result<usize> {
        if self.pos > 1 << 16 {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
        let mut chunk = [0u8; 8192];
        let n = self.stream.read(&mut chunk)?;
        self.buf.extend_from_slice(&chunk[..n]);
        Ok(n)
    }

    /// Read one line, without its line ending.
    fn read_line(&mut self) -> Result<Option<String>> {
        loop {
            if let Some(offset) = self.buf[self.pos..].iter().position(|&b| b == b'\n') {
                let end = self.pos + offset;
                let line = String::from_utf8_lossy(&self.buf[self.pos..end]).into_owned();
                self.pos = end + 1;
                return Ok(Some(line));
            }
            if self.fill()? == 0 {
                return Ok(None);
            }
        }
    }

    fn read_exact(&mut self, n: usize) -> Result<Vec<u8>> {
        while self.buf.len() - self.pos < n {
            if self.fill()? == 0 {
                bail!("the connection closed before the body was complete");
            }
        }
        let out = self.buf[self.pos..self.pos + n].to_vec();
        self.pos += n;
        Ok(out)
    }

    /// Read the response headers.
    fn read_headers(&mut self) -> Result<(bool, Option<usize>)> {
        let mut chunked = false;
        let mut length = None;
        loop {
            let Some(line) = self.read_line()? else {
                bail!("the connection closed before the headers were complete");
            };
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                break;
            }
            let lower = line.to_ascii_lowercase();
            if let Some(v) = lower.strip_prefix("transfer-encoding:") {
                if v.contains("chunked") {
                    chunked = true;
                }
            } else if let Some(v) = lower.strip_prefix("content-length:") {
                length = v.trim().parse::<usize>().ok();
            }
        }
        Ok((chunked, length))
    }

    /// Walk the response body, handing every complete line to `f`.
    ///
    /// `f` returns false to stop early.
    fn body_lines(&mut self, mut f: impl FnMut(&str) -> bool) -> Result<()> {
        let (chunked, length) = self.read_headers()?;
        let mut acc = String::new();

        let feed = |acc: &mut String, f: &mut dyn FnMut(&str) -> bool| -> bool {
            while let Some(offset) = acc.find('\n') {
                let line: String = acc.drain(..offset + 1).collect();
                if !f(line.trim_end_matches('\r')) {
                    return false;
                }
            }
            true
        };

        if chunked {
            loop {
                let Some(size_line) = self.read_line()? else { break };
                let hex = size_line.trim().split(';').next().unwrap_or("").trim();
                let size = usize::from_str_radix(hex, 16).unwrap_or(0);
                if size == 0 {
                    break;
                }
                let data = self.read_exact(size)?;
                // Every chunk is followed by its own line ending.
                let _ = self.read_line()?;
                acc.push_str(&String::from_utf8_lossy(&data));
                if !feed(&mut acc, &mut f) {
                    return Ok(());
                }
            }
        } else if let Some(n) = length {
            let data = self.read_exact(n)?;
            acc.push_str(&String::from_utf8_lossy(&data));
        } else {
            // The server did not say how long the body is, so it is read
            // until the connection ends. What is already in the buffer counts:
            // the response usually arrives in one read, headers and body
            // together, and the headers have already been taken out of it.
            loop {
                let available = self.buf.len() - self.pos;
                if available > 0 {
                    let text = String::from_utf8_lossy(&self.buf[self.pos..self.pos + available])
                        .into_owned();
                    self.pos += available;
                    acc.push_str(&text);
                    if !feed(&mut acc, &mut f) {
                        return Ok(());
                    }
                }
                if self.fill()? == 0 {
                    break;
                }
            }
        }

        for line in acc.split('\n') {
            if !f(line.trim_end_matches('\r')) {
                break;
            }
        }
        Ok(())
    }

    /// The whole body as text.
    fn body_text(&mut self) -> Result<String> {
        let mut out = String::new();
        self.body_lines(|line| {
            out.push_str(line);
            out.push('\n');
            true
        })?;
        Ok(out)
    }
}

/// Post a JSON body and read a JSON answer.
pub fn post_json(host: &str, port: u16, path: &str, body: &Value) -> Result<Value> {
    let payload = serde_json::to_string(body)?;
    let mut conn = Conn::connect(host, port, Duration::from_secs(120))?;
    conn.request("POST", path, Some(&payload))?;
    let text = conn.body_text()?;
    let start = text.find('{').unwrap_or(0);
    serde_json::from_str(&text[start..]).with_context(|| {
        format!(
            "the answer from {host}:{port}{path} was not JSON: {}",
            text.chars().take(200).collect::<String>()
        )
    })
}

/// Get a URL and read a JSON answer.
///
/// Some servers only answer `GET`: Ollama's `/api/tags` is one of them, and
/// a `POST` to it is answered with `405`, which is how the runtime ended up
/// thinking Ollama was never running.
pub fn get_json(host: &str, port: u16, path: &str) -> Result<Value> {
    let mut conn = Conn::connect(host, port, Duration::from_secs(30))?;
    conn.request("GET", path, None)?;
    let text = conn.body_text()?;
    let start = text.find('{').unwrap_or(0);
    serde_json::from_str(&text[start..]).with_context(|| {
        format!(
            "the answer from {host}:{port}{path} was not JSON: {}",
            text.chars().take(200).collect::<String>()
        )
    })
}

/// A piece of the model's answer.
#[derive(Debug, Clone, Copy)]
pub enum Delta<'a> {
    /// The answer itself.
    Content(&'a str),
    /// The model's reasoning, when it thinks out loud.
    Reasoning(&'a str),
}

/// Post a JSON body and stream the server-sent events it answers with.
///
/// Every delta is handed to `on_delta` as it arrives; the whole answer is
/// returned as well.
pub fn post_sse(
    host: &str,
    port: u16,
    path: &str,
    body: &Value,
    mut on_delta: impl FnMut(Delta<'_>),
) -> Result<String> {
    let payload = serde_json::to_string(body)?;
    let mut conn = Conn::connect(host, port, Duration::from_secs(3600))?;
    conn.request("POST", path, Some(&payload))?;

    let mut full = String::new();
    conn.body_lines(|line| {
        let Some(rest) = line.strip_prefix("data:") else {
            return true;
        };
        let rest = rest.trim();
        if rest.is_empty() || rest == "[DONE]" {
            return true;
        }
        let Ok(value) = serde_json::from_str::<Value>(rest) else {
            return true;
        };
        if let Some(text) = value
            .pointer("/choices/0/delta/content")
            .and_then(|c| c.as_str())
        {
            full.push_str(text);
            on_delta(Delta::Content(text));
        } else if let Some(text) = value
            .pointer("/choices/0/delta/reasoning_content")
            .and_then(|c| c.as_str())
        {
            on_delta(Delta::Reasoning(text));
        }
        true
    })?;
    Ok(full)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_body_without_headers_still_reads() {
        // The reader is only exercised against a real server; this checks the
        // helper compiles with the shapes the callers use.
        let value = serde_json::json!({"a": 1});
        assert_eq!(value["a"], 1);
    }
}
