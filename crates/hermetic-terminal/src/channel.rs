//! The channel: NDJSON frames over a unix socket.
//!
//! The engine connects to a socket the client listens on and then uses it in
//! both directions - events come out of it, answers go back into it.
//!
//! More than one engine can be running at once (the user may start a second
//! build while the first is still going), so the channel keeps a table of
//! connections and tags every event with the connection it came from.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use hermetic_protocol::{Answer, Event};
use serde_json::from_str;

/// The reading half: events from an engine.
pub struct Reader {
    inner: BufReader<UnixStream>,
}

impl Reader {
    /// Read the next event, or `None` at end of stream.
    pub fn next_event(&mut self) -> io::Result<Option<Event>> {
        let mut line = String::new();
        let n = self.inner.read_line(&mut line)?;
        if n == 0 {
            return Ok(None);
        }
        if line.trim().is_empty() {
            return self.next_event();
        }
        match from_str::<Event>(line.trim()) {
            Ok(event) => Ok(Some(event)),
            // A frame we cannot parse must not kill the session: the engine is
            // still running and the next frame may be fine.
            Err(_) => Ok(Some(Event::Unknown)),
        }
    }
}

/// The writing half: answers to the engine.
pub struct Writer {
    inner: UnixStream,
}

impl Writer {
    /// Send an answer back to the engine.
    pub fn send(&mut self, answer: &Answer) -> io::Result<()> {
        let mut line = serde_json::to_string(answer).map_err(io::Error::other)?;
        line.push('\n');
        self.inner.write_all(line.as_bytes())?;
        self.inner.flush()
    }
}

/// The hub: every engine that connects to this terminal.
pub struct Hub {
    /// Events from every engine, tagged with the connection they came from.
    pub events: std::sync::mpsc::Receiver<(u64, Event)>,
    writers: Arc<Mutex<HashMap<u64, Writer>>>,
    next_id: Arc<AtomicU64>,
    path: std::path::PathBuf,
}

impl Hub {
    /// Bind the socket and start accepting engines.
    ///
    /// The socket is bound before this returns, so an engine that is started
    /// afterwards cannot race us.
    pub fn bind(path: &Path) -> Result<Hub> {
        let listener = bind(path)?;
        let (tx, rx) = std::sync::mpsc::channel();
        let writers: Arc<Mutex<HashMap<u64, Writer>>> = Arc::new(Mutex::new(HashMap::new()));
        let next_id = Arc::new(AtomicU64::new(1));

        let w = Arc::clone(&writers);
        let n = Arc::clone(&next_id);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let _ = stream.set_nonblocking(false);
                let id = n.fetch_add(1, Ordering::Relaxed);
                let Ok(writer) = stream.try_clone() else {
                    continue;
                };
                if let Ok(mut map) = w.lock() {
                    map.insert(id, Writer { inner: writer });
                }
                let mut reader = Reader {
                    inner: BufReader::new(stream),
                };
                let tx = tx.clone();
                let writers = Arc::clone(&w);
                thread::spawn(move || {
                    loop {
                        match reader.next_event() {
                            Ok(Some(event)) => {
                                if tx.send((id, event)).is_err() {
                                    break;
                                }
                            }
                            Ok(None) => break,
                            Err(_) => break,
                        }
                    }
                    // The engine is gone: stop routing answers to it.
                    if let Ok(mut map) = writers.lock() {
                        map.remove(&id);
                    }
                });
            }
        });

        Ok(Hub {
            events: rx,
            writers,
            next_id,
            path: path.to_path_buf(),
        })
    }

    /// The id the next engine will get.
    pub fn next_id(&self) -> u64 {
        self.next_id.load(Ordering::Relaxed)
    }

    /// Send an answer to one engine.
    pub fn send_answer(&self, id: u64, answer: &Answer) -> Result<()> {
        let mut map = self
            .writers
            .lock()
            .map_err(|_| anyhow::anyhow!("the channel is poisoned"))?;
        match map.get_mut(&id) {
            Some(writer) => {
                writer.send(answer)?;
                Ok(())
            }
            None => anyhow::bail!("engine {id} is gone"),
        }
    }

    /// Forget an engine that has finished.
    pub fn drop_engine(&self, id: u64) {
        if let Ok(mut map) = self.writers.lock() {
            map.remove(&id);
        }
    }
}

impl Drop for Hub {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Bind the socket an engine will connect to.
///
/// A stale socket file from a previous run would make `bind` fail, so it is
/// removed first.
pub fn bind(path: &Path) -> io::Result<UnixListener> {
    if path.exists() {
        let _ = std::fs::remove_file(path);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    UnixListener::bind(path)
}

/// Send one event to the terminal, if it is listening.
///
/// The interceptor uses this to say which model it is serving, and which tab
/// asked for it.
pub fn send_event(socket: &Path, event: &serde_json::Value) -> io::Result<()> {
    let mut stream = UnixStream::connect(socket)?;
    let mut line = serde_json::to_string(event).map_err(io::Error::other)?;
    line.push('\n');
    stream.write_all(line.as_bytes())?;
    stream.flush()
}

/// Wait for one engine to connect, up to `timeout`.
///
/// Kept for the single-engine case; the terminal itself uses [`Hub`].
pub fn accept(listener: &UnixListener, timeout: Duration) -> io::Result<(Reader, Writer)> {
    listener.set_nonblocking(true)?;
    let deadline = std::time::Instant::now() + timeout;
    let stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                if std::time::Instant::now() >= deadline {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "the engine never connected",
                    ));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(e),
        }
    };
    stream.set_nonblocking(false)?;
    let writer = stream.try_clone()?;
    Ok((
        Reader {
            inner: BufReader::new(stream),
        },
        Writer { inner: writer },
    ))
}
