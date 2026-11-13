// SPDX-License-Identifier: Apache-2.0
//! In-memory chat through the application's owned HTTP listener.
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::{Shutdown, SocketAddr, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
const MAX_PROMPT: usize = 16 * 1024;
const MAX_OUTPUT: usize = 256 * 1024;
const MAX_EVENT: usize = 64 * 1024;
const MAX_TURNS: usize = 32;
const MAX_HISTORY: usize = 2 * 1024 * 1024;
const MAX_ERROR: usize = 4096;

#[derive(Clone, Debug, Default)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub reasoning: String,
    pub error: String,
    pub cancelled: bool,
}

#[derive(Clone, Debug, Default)]
pub struct PreviewSnapshot {
    pub conversation_id: u64,
    pub messages: Vec<ChatMessage>,
    pub busy: bool,
    pub output: String,
    pub reasoning: String,
    pub error: String,
    pub session_id: u64,
    pub request_id: String,
    pub operation_id: u64,
    pub cancelled: bool,
}
struct State {
    view: PreviewSnapshot,
    cancel: Option<Arc<AtomicBool>>,
    /// User-message indices of exchanges completed without failure/cancellation.
    successful: Vec<usize>,
}
struct Shared {
    state: Mutex<State>,
    socket: Mutex<Option<(u64, TcpStream)>>,
    closing: AtomicBool,
    closed: AtomicBool,
    next: AtomicU64,
}
impl Shared {
    fn update(&self, update: impl FnOnce(&mut State)) {
        update(&mut self.state.lock().unwrap_or_else(|e| e.into_inner()));
    }
    fn cancel(&self) {
        let mut operation = None;
        self.update(|s| {
            if s.view.busy {
                operation = Some(s.view.operation_id);
                s.view.cancelled = true;
                if let Some(assistant) = s.view.messages.last_mut() {
                    assistant.cancelled = true;
                }
                if let Some(cancel) = &s.cancel {
                    cancel.store(true, Ordering::Release);
                }
            }
        });
        if let Some((id, socket)) = self
            .socket
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            if operation == Some(*id) {
                let _ = socket.shutdown(Shutdown::Both);
            }
        }
    }
}
struct Job {
    endpoint: SocketAddr,
    operation_id: u64,
    messages: Vec<Value>,
    cancel: Arc<AtomicBool>,
}
pub struct Preview {
    shared: Arc<Shared>,
    sender: mpsc::SyncSender<Job>,
    worker: Mutex<Option<JoinHandle<()>>>,
    api_key: Option<String>,
}
impl Preview {
    pub fn new(api_key: Option<String>) -> Result<Self, String> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                view: PreviewSnapshot::default(),
                cancel: None,
                successful: Vec::new(),
            }),
            socket: Mutex::new(None),
            closing: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            next: AtomicU64::new(1),
        });
        let (sender, receiver) = mpsc::sync_channel::<Job>(1);
        let owned = shared.clone();
        let key = api_key.clone();
        let worker = thread::Builder::new()
            .name("flare-chat".into())
            .spawn(move || {
                while !owned.closing.load(Ordering::Acquire) {
                    let job = match receiver.recv_timeout(Duration::from_millis(50)) {
                        Ok(job) => job,
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(_) => break,
                    };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        stream(&owned, &job, key.as_deref())
                    }))
                    .unwrap_or_else(|_| Err("The chat worker failed".into()));
                    owned
                        .socket
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .take();
                    owned.update(|s| {
                        let successful = result.is_ok() && !job.cancel.load(Ordering::Acquire);
                        s.view.busy = false;
                        s.view.cancelled = job.cancel.load(Ordering::Acquire);
                        s.view.error = if s.view.cancelled {
                            String::new()
                        } else {
                            bounded_error(result.err().unwrap_or_default())
                        };
                        if let Some(assistant) = s.view.messages.last_mut() {
                            assistant.error = s.view.error.clone();
                            assistant.cancelled = s.view.cancelled;
                        }
                        if successful {
                            s.successful.push(s.view.messages.len() - 2);
                        }
                        s.cancel = None;
                    });
                }
                owned.update(|s| {
                    s.view.busy = false;
                    s.cancel = None;
                });
                owned.closed.store(true, Ordering::Release);
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            shared,
            sender,
            worker: Mutex::new(Some(worker)),
            api_key,
        })
    }
    /// The caller validates the owned endpoint/session and cancels chat work
    /// before replacing or closing that serving session.
    pub fn start(
        &self,
        endpoint: SocketAddr,
        session_id: u64,
        prompt: String,
    ) -> Result<u64, String> {
        if !endpoint.ip().is_loopback() {
            return Err("Chat requires the owned loopback listener".into());
        }
        self.start_owned(endpoint, session_id, prompt)
    }
    /// Dispatch to an address derived from the active owned serving session.
    /// The caller holds its session/admission lock through dispatch and maps
    /// wildcard listener addresses to loopback in the same address family.
    pub(crate) fn start_owned(
        &self,
        endpoint: SocketAddr,
        session_id: u64,
        prompt: String,
    ) -> Result<u64, String> {
        if self.shared.closing.load(Ordering::Acquire) {
            return Err("Chat is closing".into());
        }
        validate_owned_endpoint(endpoint)?;
        if session_id == 0 || prompt.trim().is_empty() || prompt.len() > MAX_PROMPT {
            return Err("Enter a prompt of at most 16 KiB for a live model session".into());
        }
        if self
            .api_key
            .as_ref()
            .is_some_and(|key| key.bytes().any(|b| !(32..=126).contains(&b)))
        {
            return Err("The chat API key must contain printable ASCII".into());
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let id = self.shared.next.fetch_add(1, Ordering::Relaxed);
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if self.shared.closing.load(Ordering::Acquire) {
            return Err("Chat is closing".into());
        }
        if state.view.busy {
            return Err("A chat response is already running".into());
        }
        if state.view.session_id != session_id {
            reset(&mut state, session_id)?;
        }
        let history_bytes = state
            .view
            .messages
            .iter()
            .map(|m| m.content.len() + m.reasoning.len() + m.error.len())
            .sum::<usize>();
        // Reserving the maximum response/error space rejects an oversized chat
        // before dispatch, preserving every retained exchange without truncation.
        if state.view.messages.len() / 2 >= MAX_TURNS
            || history_bytes + prompt.len() + 2 * MAX_OUTPUT + MAX_ERROR > MAX_HISTORY
        {
            return Err("Chat history has reached its limit (32 turns or 2 MiB). Start a New chat to continue.".into());
        }
        let mut messages = Vec::with_capacity(state.successful.len() * 2 + 1);
        for &user in &state.successful {
            for message in &state.view.messages[user..user + 2] {
                messages.push(json!({"role":message.role,"content":message.content}));
            }
        }
        messages.push(json!({"role":"user","content":prompt}));
        state.view.messages.push(ChatMessage {
            role: "user".into(),
            content: prompt,
            ..Default::default()
        });
        state.view.messages.push(ChatMessage {
            role: "assistant".into(),
            ..Default::default()
        });
        state.view.busy = true;
        state.view.output.clear();
        state.view.reasoning.clear();
        state.view.error.clear();
        state.view.request_id.clear();
        state.view.cancelled = false;
        state.view.operation_id = id;
        state.cancel = Some(cancel.clone());
        if self
            .sender
            .try_send(Job {
                endpoint,
                operation_id: id,
                messages,
                cancel,
            })
            .is_err()
        {
            state.view.busy = false;
            state.cancel = None;
            state.view.error = "The chat worker is unavailable".into();
            state.view.messages.last_mut().unwrap().error = state.view.error.clone();
            return Err(state.view.error.clone());
        }
        Ok(id)
    }
    /// Clears only in-memory conversation data. An in-flight response must be
    /// cancelled and allowed to finish before its transcript can be cleared.
    pub fn clear(&self) -> Result<(), String> {
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if self.shared.closing.load(Ordering::Acquire) {
            return Err("Chat is closing".into());
        }
        if state.view.busy {
            return Err("Stop the chat response before starting a New chat".into());
        }
        let session_id = state.view.session_id;
        reset(&mut state, session_id)
    }
    pub fn cancel(&self) {
        self.shared.cancel();
    }
    pub fn snapshot(&self) -> PreviewSnapshot {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .view
            .clone()
    }
    pub fn close(&self) {
        self.shared.closing.store(true, Ordering::Release);
        self.cancel();
    }
    pub fn is_closed(&self) -> bool {
        self.shared.closed.load(Ordering::Acquire)
    }
}
fn validate_owned_endpoint(endpoint: SocketAddr) -> Result<(), String> {
    if endpoint.port() == 0
        || endpoint.ip().is_unspecified()
        || endpoint.ip().is_multicast()
        || matches!(endpoint, SocketAddr::V4(address) if address.ip().is_broadcast())
    {
        return Err("Chat requires a concrete address and port from the owned listener".into());
    }
    Ok(())
}
fn reset(state: &mut State, session_id: u64) -> Result<(), String> {
    let conversation_id = state
        .view
        .conversation_id
        .checked_add(1)
        .ok_or("Chat identifiers exhausted; restart the application")?;
    state.view = PreviewSnapshot {
        conversation_id,
        session_id,
        ..Default::default()
    };
    state.successful.clear();
    state.cancel = None;
    Ok(())
}
fn bounded_error(mut error: String) -> String {
    if error.len() > MAX_ERROR {
        let mut end = MAX_ERROR;
        while !error.is_char_boundary(end) {
            end -= 1;
        }
        error.truncate(end);
    }
    error
}
impl Drop for Preview {
    fn drop(&mut self) {
        self.close();
        if let Some(worker) = self
            .worker
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            let _ = worker.join();
        }
    }
}

struct Transport {
    socket: TcpStream,
    cancel: Arc<AtomicBool>,
    deadline: Instant,
}
impl Read for Transport {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.cancel.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "Chat cancelled",
                ));
            }
            if Instant::now() >= self.deadline {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "Chat timed out"));
            }
            match self.socket.read(bytes) {
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                    ) =>
                {
                    continue;
                }
                result => return result,
            }
        }
    }
}
fn line(reader: &mut impl BufRead, limit: usize) -> io::Result<Vec<u8>> {
    let mut result = Vec::new();
    loop {
        let bytes = reader.fill_buf()?;
        if bytes.is_empty() {
            return Ok(result);
        }
        let count = bytes
            .iter()
            .position(|b| *b == b'\n')
            .map_or(bytes.len(), |i| i + 1);
        if result.len() + count > limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Chat protocol line exceeds its limit",
            ));
        }
        let finished = bytes[count - 1] == b'\n';
        result.extend_from_slice(&bytes[..count]);
        reader.consume(count);
        if finished {
            return Ok(result);
        }
    }
}
struct Body<R> {
    reader: R,
    chunked: bool,
    remaining: usize,
    first: bool,
    finished: bool,
    total: usize,
    length: Option<usize>,
}
impl<R: BufRead> Read for Body<R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if bytes.is_empty() || self.finished {
            return Ok(0);
        }
        if !self.chunked && self.length == Some(0) {
            return Ok(0);
        }
        if self.chunked && self.remaining == 0 {
            if !self.first {
                let mut end = [0; 2];
                self.reader.read_exact(&mut end)?;
                if end != *b"\r\n" {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid chat HTTP chunk",
                    ));
                }
            }
            self.first = false;
            let header = line(&mut self.reader, 128)?;
            let header = std::str::from_utf8(&header).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "Invalid HTTP chunk size")
            })?;
            self.remaining =
                usize::from_str_radix(header.trim().split(';').next().unwrap_or(""), 16).map_err(
                    |_| io::Error::new(io::ErrorKind::InvalidData, "Invalid HTTP chunk size"),
                )?;
            if self.remaining == 0 {
                self.finished = true;
                return Ok(0);
            }
            if self.remaining > 2 * 1024 * 1024 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Chat response exceeds its limit",
                ));
            }
        }
        let count = if self.chunked {
            bytes.len().min(self.remaining)
        } else {
            bytes.len().min(self.length.unwrap_or(usize::MAX))
        };
        let count = self.reader.read(&mut bytes[..count])?;
        if self.chunked {
            self.remaining -= count;
            if count == 0 {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
        }
        if !self.chunked {
            if let Some(remaining) = &mut self.length {
                if count == 0 && *remaining > 0 {
                    return Err(io::ErrorKind::UnexpectedEof.into());
                }
                *remaining -= count;
            }
        }
        self.total += count;
        if self.total > 2 * 1024 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Chat response exceeds its limit",
            ));
        }
        Ok(count)
    }
}

fn stream(shared: &Shared, job: &Job, api_key: Option<&str>) -> Result<(), String> {
    if job.cancel.load(Ordering::Acquire) || shared.closing.load(Ordering::Acquire) {
        return Err("Chat cancelled".into());
    }
    let mut socket = TcpStream::connect_timeout(&job.endpoint, Duration::from_secs(1))
        .map_err(|e| e.to_string())?;
    socket
        .set_read_timeout(Some(Duration::from_millis(100)))
        .map_err(|e| e.to_string())?;
    socket
        .set_write_timeout(Some(Duration::from_millis(250)))
        .map_err(|e| e.to_string())?;
    *shared.socket.lock().unwrap_or_else(|e| e.into_inner()) = Some((
        job.operation_id,
        socket.try_clone().map_err(|e| e.to_string())?,
    ));
    if job.cancel.load(Ordering::Acquire) || shared.closing.load(Ordering::Acquire) {
        return Err("Chat cancelled".into());
    }
    let body =
        json!({"model":"flare-local","messages":job.messages,"stream":true,"max_tokens":2048})
            .to_string();
    let auth = api_key
        .map(|key| format!("Authorization: Bearer {key}\r\n"))
        .unwrap_or_default();
    write!(socket,"POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nAccept: text/event-stream\r\nContent-Type: application/json\r\nConnection: close\r\n{auth}Content-Length: {}\r\n\r\n{body}", job.endpoint, body.len()).map_err(|e| e.to_string())?;
    let transport = Transport {
        socket,
        cancel: job.cancel.clone(),
        deadline: Instant::now() + Duration::from_secs(120),
    };
    let mut reader = BufReader::new(transport);
    let status = line(&mut reader, 4096).map_err(|e| e.to_string())?;
    let status = std::str::from_utf8(&status).map_err(|_| "Invalid chat HTTP status")?;
    if !status.starts_with("HTTP/1.") {
        return Err("Invalid chat HTTP status".into());
    }
    let code: u16 = status
        .split_whitespace()
        .nth(1)
        .ok_or("Missing chat HTTP status")?
        .parse()
        .map_err(|_| "Invalid chat HTTP status")?;
    let mut total = 0;
    let mut chunked = false;
    let mut event_stream = false;
    let mut length = None;
    loop {
        let header = line(&mut reader, 8192).map_err(|e| e.to_string())?;
        total += header.len();
        if total > 32768 {
            return Err("Chat response headers exceed their limit".into());
        }
        if header == b"\r\n" || header == b"\n" {
            break;
        }
        if header.is_empty() {
            return Err("Chat ended before its HTTP headers".into());
        }
        let header = std::str::from_utf8(&header).map_err(|_| "Invalid chat HTTP header")?;
        let (name, value) = header
            .split_once(':')
            .ok_or("Invalid chat HTTP header")?;
        let value = value.trim();
        if name.eq_ignore_ascii_case("transfer-encoding") {
            chunked = value.eq_ignore_ascii_case("chunked");
        }
        if name.eq_ignore_ascii_case("content-length") {
            let count: usize = value
                .parse()
                .map_err(|_| "Invalid chat content length")?;
            if count > 2 * 1024 * 1024 {
                return Err("Chat response exceeds its limit".into());
            }
            length = Some(count);
        }
        if name.eq_ignore_ascii_case("content-type") {
            event_stream = value
                .split(';')
                .next()
                .is_some_and(|v| v.trim().eq_ignore_ascii_case("text/event-stream"));
        }
        if name.eq_ignore_ascii_case("x-request-id") {
            shared.update(|s| s.view.request_id = value.into());
        }
    }
    let body = Body {
        reader,
        chunked,
        remaining: 0,
        first: true,
        finished: false,
        total: 0,
        length,
    };
    let mut body = BufReader::new(body);
    if code != 200 {
        let mut bytes = Vec::new();
        body.by_ref()
            .take(MAX_EVENT as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        let message = serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(str::to_owned))
            .unwrap_or_else(|| "The local listener rejected the chat request".into());
        return Err(format!("HTTP {code}: {message}"));
    }
    if !event_stream {
        return Err("The local listener did not return an event stream".into());
    }
    let mut event = Vec::new();
    loop {
        if job.cancel.load(Ordering::Acquire) {
            return Err("Chat cancelled".into());
        }
        let bytes = line(&mut body, MAX_EVENT).map_err(|e| e.to_string())?;
        if bytes.is_empty() {
            return Err("Chat ended before a terminal event".into());
        }
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| "Invalid UTF-8 in chat stream")?
            .trim_end_matches(['\r', '\n']);
        if text.is_empty() {
            if event.is_empty() {
                continue;
            }
            let data = std::str::from_utf8(&event).map_err(|_| "Invalid chat event")?;
            if data.trim() == "[DONE]" {
                return Ok(());
            }
            let value: Value =
                serde_json::from_str(data).map_err(|_| "Invalid JSON in chat stream")?;
            if let Some(error) = value["error"]["message"].as_str() {
                return Err(error.to_owned());
            }
            let delta = &value["choices"][0]["delta"];
            let answer = delta["content"].as_str().unwrap_or("");
            let reasoning = delta["reasoning_content"]
                .as_str()
                .or_else(|| delta["reasoning"].as_str())
                .unwrap_or("");
            let mut exceeded = false;
            shared.update(|s| {
                if s.view.request_id.is_empty() {
                    if let Some(id) = value["id"].as_str() {
                        s.view.request_id = id.into();
                    }
                }
                exceeded = s.view.output.len() + answer.len() > MAX_OUTPUT
                    || s.view.reasoning.len() + reasoning.len() > MAX_OUTPUT;
                if !exceeded {
                    s.view.output.push_str(answer);
                    s.view.reasoning.push_str(reasoning);
                    if let Some(assistant) = s.view.messages.last_mut() {
                        assistant.content.push_str(answer);
                        assistant.reasoning.push_str(reasoning);
                    }
                }
            });
            if exceeded {
                return Err("Chat output exceeds its display limit".into());
            }
            event.clear();
        } else if let Some(data) = text.strip_prefix("data:") {
            let data = data.strip_prefix(' ').unwrap_or(data);
            if event.len() + data.len() + 1 > MAX_EVENT {
                return Err("Chat event exceeds its limit".into());
            }
            if !event.is_empty() {
                event.push(b'\n');
            }
            event.extend_from_slice(data.as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    #[test]
    fn public_chat_rejects_remote_addresses_without_dispatch() {
        let preview = Preview::new(None).unwrap();
        for endpoint in ["192.168.1.20:8000", "[fd00::20]:8000"] {
            let error = preview
                .start(endpoint.parse().unwrap(), 1, "Hello".into())
                .unwrap_err();
            assert!(error.contains("loopback"));
        }
        assert!(!preview.snapshot().busy);
        assert!(preview.snapshot().messages.is_empty());
    }

    #[test]
    fn owned_chat_validates_concrete_addresses_and_request_identity_without_dispatch() {
        for endpoint in [
            "127.0.0.1:8000", "[::1]:8000", "192.168.1.20:8000", "[fd00::20]:8000",
        ] {
            assert!(validate_owned_endpoint(endpoint.parse().unwrap()).is_ok());
        }
        let preview = Preview::new(None).unwrap();
        for endpoint in [
            "0.0.0.0:8000", "[::]:8000", "127.0.0.1:0", "224.0.0.1:8000",
            "[ff02::1]:8000", "255.255.255.255:8000",
        ] {
            assert!(preview
                .start_owned(endpoint.parse().unwrap(), 1, "Hello".into())
                .is_err());
        }
        let endpoint = "192.168.1.20:8000".parse().unwrap();
        assert!(preview.start_owned(endpoint, 0, "Hello".into()).is_err());
        assert!(preview.start_owned(endpoint, 1, " ".into()).is_err());
        assert!(!preview.snapshot().busy);
        assert!(preview.snapshot().messages.is_empty());
        preview.close();
        assert!(preview.start_owned(endpoint, 1, "Hello".into()).is_err());
    }

    fn wait(preview: &Preview, predicate: impl Fn(&PreviewSnapshot) -> bool) -> PreviewSnapshot {
        let started = Instant::now();
        loop {
            let view = preview.snapshot();
            if predicate(&view) {
                return view;
            }
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "Chat timed out: {view:?}"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
    fn request(socket: &TcpStream) -> (String, Value) {
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut reader = BufReader::new(socket.try_clone().unwrap());
        let mut headers = String::new();
        let mut length = 0;
        loop {
            let mut row = String::new();
            reader.read_line(&mut row).unwrap();
            if row == "\r\n" {
                break;
            }
            if let Some(value) = row.strip_prefix("Content-Length:") {
                length = value.trim().parse().unwrap();
            }
            headers.push_str(&row);
        }
        let mut bytes = vec![0; length];
        reader.read_exact(&mut bytes).unwrap();
        (headers, serde_json::from_slice(&bytes).unwrap())
    }
    fn chunks(socket: &mut TcpStream, bytes: &[u8]) {
        for part in bytes.chunks(3) {
            write!(socket, "{:x}\r\n", part.len()).unwrap();
            socket.write_all(part).unwrap();
            socket.write_all(b"\r\n").unwrap();
        }
    }
    fn reply(socket: &mut TcpStream, answer: &str, reasoning: &str, complete: bool) {
        let delta = json!({"choices":[{"delta":{"content":answer,"reasoning_content":reasoning}}]});
        write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {delta}\n\n").unwrap();
        if complete {
            socket.write_all(b"data: [DONE]\n\n").unwrap();
        }
    }

    #[test]
    fn chat_second_request_replays_successful_content_without_reasoning() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let (_, body) = request(&socket);
            assert_eq!(body["max_tokens"], 2048);
            assert_eq!(
                body["messages"],
                json!([{"role":"user","content":"Remember amber"}])
            );
            reply(
                &mut socket,
                "Amber remembered",
                "Private deliberation",
                true,
            );
            drop(socket);
            let (mut socket, _) = listener.accept().unwrap();
            let (_, body) = request(&socket);
            assert_eq!(
                body["messages"],
                json!([
                    {"role":"user","content":"Remember amber"},
                    {"role":"assistant","content":"Amber remembered"},
                    {"role":"user","content":"What color?"},
                ])
            );
            assert!(!body.to_string().contains("Private deliberation"));
            reply(&mut socket, "Amber", "", true);
        });
        let preview = Preview::new(None).unwrap();
        preview
            .start(endpoint, 81, "Remember amber".into())
            .unwrap();
        let first = wait(&preview, |s| !s.busy);
        assert_eq!(first.messages.len(), 2);
        assert_eq!(first.messages[1].reasoning, "Private deliberation");
        preview.start(endpoint, 81, "What color?".into()).unwrap();
        let second = wait(&preview, |s| !s.busy);
        assert!(second.error.is_empty(), "{second:?}");
        assert_eq!(second.conversation_id, first.conversation_id);
        assert_eq!(second.messages.len(), 4);
        assert_eq!(second.messages[3].content, "Amber");
        server.join().unwrap();
    }
    #[test]
    fn chat_failed_and_cancelled_exchanges_remain_visible_but_are_not_replayed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for (prompt, answer, complete) in [
                ("Failed question", "Incomplete answer", false),
                ("Cancelled question", "Cancelled answer", false),
                ("Successful question", "Complete answer", true),
            ] {
                let (mut socket, _) = listener.accept().unwrap();
                let (_, body) = request(&socket);
                assert_eq!(body["messages"], json!([{"role":"user","content":prompt}]));
                reply(&mut socket, answer, "", complete);
                if prompt == "Cancelled question" {
                    let result = socket.read(&mut [0]);
                    assert!(matches!(result, Ok(0)) || result.is_err());
                }
            }
        });
        let preview = Preview::new(None).unwrap();
        preview
            .start(endpoint, 82, "Failed question".into())
            .unwrap();
        let failed = wait(&preview, |s| !s.busy);
        assert_eq!(failed.messages[1].content, "Incomplete answer");
        assert!(!failed.messages[1].error.is_empty());
        preview
            .start(endpoint, 82, "Cancelled question".into())
            .unwrap();
        let partial = wait(&preview, |s| s.output == "Cancelled answer");
        assert!(partial.busy);
        assert_eq!(partial.messages[3].content, "Cancelled answer");
        assert!(preview.clear().is_err());
        preview.cancel();
        let cancelled = wait(&preview, |s| !s.busy);
        assert!(cancelled.messages[3].cancelled);
        preview
            .start(endpoint, 82, "Successful question".into())
            .unwrap();
        let completed = wait(&preview, |s| !s.busy);
        assert_eq!(completed.messages.len(), 6);
        assert_eq!(completed.messages[5].content, "Complete answer");
        assert!(completed.error.is_empty());
        server.join().unwrap();
    }

    #[test]
    fn chat_clear_and_session_change_isolate_transcripts_and_context() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for prompt in [
                "First chat",
                "Cleared chat",
                "Different session",
                "Returned session",
            ] {
                let (mut socket, _) = listener.accept().unwrap();
                let (_, body) = request(&socket);
                assert_eq!(body["messages"], json!([{"role":"user","content":prompt}]));
                reply(&mut socket, "Answer", "", true);
            }
        });
        let preview = Preview::new(None).unwrap();
        preview.start(endpoint, 83, "First chat".into()).unwrap();
        let first = wait(&preview, |s| !s.busy);
        preview.clear().unwrap();
        let cleared = preview.snapshot();
        assert_eq!(cleared.session_id, 83);
        assert!(cleared.conversation_id > first.conversation_id);
        assert!(cleared.messages.is_empty());
        assert!(cleared.output.is_empty());
        preview.start(endpoint, 83, "Cleared chat".into()).unwrap();
        assert_eq!(
            wait(&preview, |s| !s.busy).conversation_id,
            cleared.conversation_id
        );
        preview
            .start(endpoint, 84, "Different session".into())
            .unwrap();
        let different = wait(&preview, |s| !s.busy);
        assert!(different.conversation_id > cleared.conversation_id);
        assert_eq!(different.messages.len(), 2);
        preview
            .start(endpoint, 83, "Returned session".into())
            .unwrap();
        let returned = wait(&preview, |s| !s.busy);
        assert!(returned.conversation_id > different.conversation_id);
        assert_eq!(returned.messages.len(), 2);
        server.join().unwrap();
    }

    #[test]
    fn chat_turn_limit_rejects_without_truncating_and_clear_restores_admission() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for turn in 0..=MAX_TURNS {
                let (mut socket, _) = listener.accept().unwrap();
                let (_, body) = request(&socket);
                let expected = if turn == MAX_TURNS { 1 } else { turn * 2 + 1 };
                assert_eq!(body["messages"].as_array().unwrap().len(), expected);
                reply(&mut socket, "Answer", "", true);
            }
        });
        let preview = Preview::new(None).unwrap();
        assert!(
            preview
                .start(endpoint, 85, "a".repeat(MAX_PROMPT + 1))
                .is_err()
        );
        assert!(preview.snapshot().messages.is_empty());
        for _ in 0..MAX_TURNS {
            preview.start(endpoint, 85, "Question".into()).unwrap();
            assert!(wait(&preview, |s| !s.busy).error.is_empty());
        }
        let full = preview.snapshot();
        assert_eq!(full.messages.len(), MAX_TURNS * 2);
        let error = preview
            .start(endpoint, 85, "Over limit".into())
            .unwrap_err();
        assert!(error.contains("New chat"));
        assert_eq!(preview.snapshot().messages.len(), full.messages.len());
        assert_eq!(preview.snapshot().conversation_id, full.conversation_id);
        preview.clear().unwrap();
        preview
            .start(endpoint, 85, "After clearing".into())
            .unwrap();
        assert_eq!(wait(&preview, |s| !s.busy).messages.len(), 2);
        server.join().unwrap();
    }

    #[test]
    fn chat_byte_limit_reserves_response_space_without_truncating_history() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for _ in 0..3 {
                let (mut socket, _) = listener.accept().unwrap();
                request(&socket);
                socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
                for _ in 0..16 {
                    let data = json!({"choices":[{"delta":{"content":"a".repeat(MAX_OUTPUT / 16),"reasoning_content":"r".repeat(MAX_OUTPUT / 16)}}]});
                    write!(socket, "data: {data}\n\n").unwrap();
                }
                socket.write_all(b"data: [DONE]\n\n").unwrap();
            }
        });
        let preview = Preview::new(None).unwrap();
        for _ in 0..3 {
            preview.start(endpoint, 86, "Question".into()).unwrap();
            let complete = wait(&preview, |s| !s.busy);
            assert!(complete.error.is_empty(), "{}", complete.error);
        }
        let history = preview.snapshot();
        assert_eq!(history.messages.len(), 6);
        assert_eq!(history.messages[1].content.len(), MAX_OUTPUT);
        assert_eq!(history.messages[1].reasoning.len(), MAX_OUTPUT);
        let error = preview
            .start(endpoint, 86, "Over byte limit".into())
            .unwrap_err();
        assert!(error.contains("New chat"));
        let retained = preview.snapshot();
        assert_eq!(retained.messages.len(), history.messages.len());
        assert_eq!(retained.messages[1].content.len(), MAX_OUTPUT);
        server.join().unwrap();
    }

    #[test]
    fn real_loopback_stream_updates_reasoning_answer_and_request_identity() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let (finish, finishing) = mpsc::sync_channel(1);
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket.set_nodelay(true).unwrap();
            let (headers, body) = request(&socket);
            assert!(headers.contains("Authorization: Bearer test-key"));
            assert_eq!(body["model"], "flare-local");
            assert_eq!(body["messages"][0]["content"], "Hello ☀");
            assert_eq!(body["stream"], true);
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nX-Request-Id: chat-preview-1\r\n\r\n").unwrap();
            chunks(&mut socket,b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Thinking\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n");
            finishing.recv().unwrap();
            chunks(
                &mut socket,
                b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\ndata: [DONE]\n\n",
            );
            let _ = socket.write_all(b"0\r\n\r\n");
        });
        let preview = Preview::new(Some("test-key".into())).unwrap();
        let operation = preview.start(endpoint, 41, "Hello ☀".into()).unwrap();
        let partial = wait(&preview, |s| s.output == "Hello");
        assert!(partial.busy);
        assert_eq!(partial.reasoning, "Thinking");
        finish.send(()).unwrap();
        let view = wait(&preview, |s| !s.busy);
        server.join().unwrap();
        assert_eq!(view.output, "Hello world");
        assert!(view.error.is_empty());
        assert_eq!(view.request_id, "chat-preview-1");
        assert_eq!(view.session_id, 41);
        assert_eq!(view.operation_id, operation);
    }
    #[test]
    fn cancellation_closes_a_stalled_stream_and_preserves_single_request_admission() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let (started, ready) = mpsc::sync_channel(1);
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            request(&socket);
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n: waiting\n\n").unwrap();
            started.send(()).unwrap();
            let result = socket.read(&mut [0]);
            assert!(matches!(result, Ok(0)) || result.is_err());
        });
        let preview = Preview::new(None).unwrap();
        preview.start(endpoint, 51, "Prompt".into()).unwrap();
        ready.recv().unwrap();
        assert!(
            preview
                .start(endpoint, 51, "Another prompt".into())
                .is_err()
        );
        let before = Instant::now();
        preview.cancel();
        assert!(before.elapsed() < Duration::from_millis(100));
        let view = wait(&preview, |s| !s.busy);
        assert!(view.cancelled);
        assert!(view.error.is_empty());
        server.join().unwrap();
        preview.close();
        let before = Instant::now();
        while !preview.is_closed() {
            assert!(before.elapsed() < Duration::from_secs(2));
            thread::sleep(Duration::from_millis(5));
        }
        assert!(preview.start(endpoint, 51, "After close".into()).is_err());
        assert!(
            Preview::new(None)
                .unwrap()
                .start("192.0.2.1:80".parse().unwrap(), 51, "Remote".into())
                .is_err()
        );
    }
    #[test]
    fn premature_eof_reports_failure_instead_of_successful_partial_output() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            request(&socket);
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"Partial\"}}]}\n\n").unwrap();
        });
        let preview = Preview::new(None).unwrap();
        preview.start(endpoint, 61, "Prompt".into()).unwrap();
        let view = wait(&preview, |s| !s.busy);
        assert_eq!(view.output, "Partial");
        assert!(view.error.contains("terminal event"));
        server.join().unwrap();
    }

    #[test]
    fn http_error_uses_declared_length_without_waiting_for_connection_close() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            request(&socket);
            let body = r#"{"error":{"message":"Admission is paused"}}"#;
            write!(socket,"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();
            assert!(matches!(socket.read(&mut [0]), Ok(0)));
        });
        let preview = Preview::new(None).unwrap();
        let before = Instant::now();
        preview.start(endpoint, 71, "Prompt".into()).unwrap();
        let view = wait(&preview, |s| !s.busy);
        assert!(before.elapsed() < Duration::from_secs(2));
        assert!(view.error.contains("HTTP 503: Admission is paused"));
        server.join().unwrap();
    }
}
