//! A client of any System One server: TypeSafe's hosted Jev
//! (`https://api.typesafe.ai`), Intel-Phi-Jev's `xks serve`, or this crate's
//! `jevbase serve`. It follows TypeSafe's documentation (OpenAPI 0.2.0,
//! https://docs.typesafe.ai/api/system-one; models and rate limits,
//! https://docs.typesafe.ai/models; read 2026-11-21) and its SDKs'
//! configuration:
//!
//! - `TYPESAFE_BASE_URL` (default `http://127.0.0.1:8097`, `jevbase serve`),
//!   `TYPESAFE_API_KEY` (sent as `Authorization: Bearer KEY`),
//!   `TYPESAFE_DEFAULT_MODEL` (default `jev-latest`, the SDKs' default);
//! - every request carries `model` (the wire requires it): the client's
//!   default when the request names none;
//! - `429 Too Many Requests` is retried, waiting longer each time and
//!   following `retry-after` (capped), as the SDKs do;
//! - refusals carry the server's `detail` (a string, an object, or a list of
//!   validation errors).
//!
//! Plain `http://` goes over `std::net`; `https://` through the `curl`
//! program (no TLS library in this crate), its configuration, the key
//! included, passed on standard input rather than the command line. No model
//! and no llama.cpp: tools that only ask build with `default-features =
//! false`.

use std::fmt;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::Value;

use crate::protocol::{Evaluation, Request};

/// Why a call did not produce an evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientError {
    /// Nothing answered at the address (no server running there).
    Unreachable(String),
    /// The server refused the request (HTTP 4xx), with its `detail`.
    Rejected(u16, String),
    /// The server failed (HTTP 5xx), or its reply could not be read.
    Failed(String),
}

impl fmt::Display for ClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreachable(m) => write!(
                f,
                "no System One server: {m} (start one with `jevbase serve`, or set TYPESAFE_BASE_URL)"
            ),
            Self::Rejected(code @ (401 | 403), m) => {
                write!(f, "refused ({code}): {m} (set TYPESAFE_API_KEY)")
            }
            Self::Rejected(code, m) => write!(f, "refused ({code}): {m}"),
            Self::Failed(m) => write!(f, "server failed: {m}"),
        }
    }
}

impl std::error::Error for ClientError {}

/// A System One server and how to ask it.
#[derive(Debug, Clone)]
pub struct Client {
    /// `scheme://host[:port]`, no trailing slash.
    base: String,
    api_key: Option<String>,
    model: String,
    long_context: bool,
    connect_timeout: Duration,
    timeout: Duration,
    /// Retries of a 429 before giving up.
    retries: u32,
}

/// One HTTP response: status, `retry-after` seconds, body.
struct Response {
    status: u16,
    retry_after: Option<f64>,
    body: Vec<u8>,
}

impl Client {
    /// Where `jevbase serve` listens by default.
    pub const DEFAULT_BASE: &'static str = "http://127.0.0.1:8097";
    /// The SDKs' default model alias.
    pub const DEFAULT_MODEL: &'static str = "jev-latest";
    /// Longest wait for a 429's `retry-after`.
    const MAX_WAIT: f64 = 60.0;

    /// A client of `base` (`http://host:port`, `https://host`, or a bare
    /// `host:port`, taken as http), no key, the default model.
    pub fn new(base: impl Into<String>) -> Self {
        let base = base.into();
        let base = base.trim().trim_end_matches('/');
        let base = if base.contains("://") {
            base.to_string()
        } else {
            format!("http://{base}")
        };
        Self {
            base,
            api_key: None,
            model: Self::DEFAULT_MODEL.to_string(),
            long_context: false,
            connect_timeout: Duration::from_secs(2),
            timeout: Duration::from_secs(300),
            retries: 3,
        }
    }

    /// From the SDKs' variables: `TYPESAFE_BASE_URL`, `TYPESAFE_API_KEY`,
    /// `TYPESAFE_DEFAULT_MODEL` (empty counts as unset).
    pub fn from_env() -> Self {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let mut c =
            Self::new(var("TYPESAFE_BASE_URL").unwrap_or_else(|| Self::DEFAULT_BASE.into()));
        c.api_key = var("TYPESAFE_API_KEY").map(|k| k.trim().to_string());
        if let Some(m) = var("TYPESAFE_DEFAULT_MODEL") {
            c.model = m;
        }
        c
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Send `TypeSafe-Danger-Zone-Long-Context: true` (both context limits
    /// 262k tokens; slower and, on the hosted Jev, less accurate).
    pub fn with_long_context(mut self, on: bool) -> Self {
        self.long_context = on;
        self
    }

    /// Give up on an answer after `t` (a decision on a long state takes seconds).
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// `GET /health` (xks and jevbase; the hosted Jev has none).
    pub fn health(&self) -> Result<Value, ClientError> {
        let r = self.call("GET", "/health", None)?;
        parse_reply(r.status, &r.body)
    }

    /// `GET /v1/models`: the names the server accepts in `model`.
    pub fn models(&self) -> Result<Value, ClientError> {
        let r = self.call("GET", "/v1/models", None)?;
        parse_reply(r.status, &r.body)
    }

    /// `POST /v1/systemone`: the request's evaluation. A request without a
    /// model is sent with this client's.
    pub fn evaluate(&self, req: &Request) -> Result<Evaluation, ClientError> {
        let mut req = req.clone();
        if req.model.is_none() {
            req.model = Some(self.model.clone());
        }
        let body = serde_json::to_vec(&req).map_err(|e| ClientError::Failed(e.to_string()))?;
        let r = self.call("POST", "/v1/systemone", Some(&body))?;
        let v = parse_reply(r.status, &r.body)?;
        serde_json::from_value(v).map_err(|e| ClientError::Failed(format!("evaluation: {e}")))
    }

    /// One call, a 429 retried up to `retries` times.
    fn call(&self, method: &str, path: &str, body: Option<&[u8]>) -> Result<Response, ClientError> {
        let mut attempt = 0;
        loop {
            let r = if self.base.starts_with("https://") {
                self.call_curl(method, path, body)?
            } else {
                self.call_tcp(method, path, body)?
            };
            if r.status != 429 || attempt >= self.retries {
                return Ok(r);
            }
            let backoff = f64::from(1u32 << attempt);
            let wait = r.retry_after.unwrap_or(backoff).clamp(0.0, Self::MAX_WAIT);
            std::thread::sleep(Duration::from_secs_f64(wait));
            attempt += 1;
        }
    }

    /// The headers every call sends besides Host and Content-Length.
    fn headers(&self) -> Vec<String> {
        let mut h = vec!["Content-Type: application/json".to_string()];
        if let Some(k) = &self.api_key {
            h.push(format!("Authorization: Bearer {k}"));
        }
        if self.long_context {
            h.push("TypeSafe-Danger-Zone-Long-Context: true".into());
        }
        h
    }

    fn call_tcp(
        &self,
        method: &str,
        path: &str,
        body: Option<&[u8]>,
    ) -> Result<Response, ClientError> {
        let hostport = self.base.trim_start_matches("http://");
        let sock: SocketAddr = hostport
            .to_socket_addrs()
            .map_err(|e| ClientError::Unreachable(format!("{hostport}: {e}")))?
            .next()
            .ok_or_else(|| ClientError::Unreachable(format!("{hostport}: no address")))?;
        let mut stream = TcpStream::connect_timeout(&sock, self.connect_timeout)
            .map_err(|e| ClientError::Unreachable(format!("{hostport}: {e}")))?;
        stream
            .set_read_timeout(Some(self.timeout))
            .and_then(|_| stream.set_write_timeout(Some(self.timeout)))
            .map_err(|e| ClientError::Failed(e.to_string()))?;
        let body = body.unwrap_or(&[]);
        let mut head = format!("{method} {path} HTTP/1.1\r\nHost: {hostport}\r\n");
        for h in self.headers() {
            head.push_str(&h);
            head.push_str("\r\n");
        }
        head.push_str(&format!(
            "Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        ));
        stream
            .write_all(head.as_bytes())
            .and_then(|_| stream.write_all(body))
            .map_err(|e| ClientError::Failed(format!("sending: {e}")))?;
        let mut raw = Vec::new();
        match stream.read_to_end(&mut raw) {
            Ok(_) => parse_http(&raw, false),
            // A server that closes with part of the request unread makes the
            // kernel reset the connection; a reply received whole before the
            // reset still counts.
            Err(e) => parse_http(&raw, false)
                .map_err(|_| ClientError::Failed(format!("reading the reply: {e}"))),
        }
    }

    /// HTTPS through `curl`: its configuration (URL, headers, body) on
    /// standard input, the response with its headers on standard output.
    fn call_curl(
        &self,
        method: &str,
        path: &str,
        body: Option<&[u8]>,
    ) -> Result<Response, ClientError> {
        let quote = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
        let mut cfg = String::new();
        cfg.push_str(&format!(
            "url = {}\n",
            quote(&format!("{}{path}", self.base))
        ));
        cfg.push_str(&format!("request = {}\n", quote(method)));
        for h in self.headers() {
            cfg.push_str(&format!("header = {}\n", quote(&h)));
        }
        if let Some(b) = body {
            cfg.push_str(&format!(
                "data-binary = {}\n",
                quote(&String::from_utf8_lossy(b))
            ));
        }
        cfg.push_str(&format!(
            "connect-timeout = {}\nmax-time = {}\nsilent\nshow-error\ninclude\nhttp1.1\n",
            self.connect_timeout.as_secs().max(1),
            self.timeout.as_secs().max(1)
        ));
        let mut child = Command::new("curl")
            .args(["--config", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ClientError::Failed(format!("https needs the `curl` program: {e}")))?;
        child
            .stdin
            .take()
            .expect("piped stdin")
            .write_all(cfg.as_bytes())
            .map_err(|e| ClientError::Failed(format!("curl: {e}")))?;
        let out = child
            .wait_with_output()
            .map_err(|e| ClientError::Failed(format!("curl: {e}")))?;
        match out.status.code() {
            Some(0) => parse_http(&out.stdout, true),
            // 6: could not resolve, 7: could not connect, 28: timed out.
            Some(6 | 7 | 28) => Err(ClientError::Unreachable(format!(
                "{}: {}",
                self.base,
                String::from_utf8_lossy(&out.stderr).trim()
            ))),
            _ => Err(ClientError::Failed(format!(
                "curl: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ))),
        }
    }
}

impl Default for Client {
    /// `Client::from_env()`.
    fn default() -> Self {
        Self::from_env()
    }
}

/// Status, `retry-after` and body of a raw HTTP response. `decoded`: the
/// body is already free of transfer encoding (curl decodes it), so only the
/// headers are read.
fn parse_http(raw: &[u8], decoded: bool) -> Result<Response, ClientError> {
    let bad = |m: &str| ClientError::Failed(format!("malformed HTTP reply: {m}"));
    // curl may print an interim `100 Continue` block first: take the last.
    let mut start = 0;
    loop {
        let split = raw[start..]
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| bad("no header end"))?
            + start;
        let head = std::str::from_utf8(&raw[start..split]).map_err(|_| bad("header not UTF-8"))?;
        let rest = &raw[split + 4..];
        let mut lines = head.split("\r\n");
        let status: u16 = lines
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|c| c.parse().ok())
            .ok_or_else(|| bad("no status code"))?;
        if (100..200).contains(&status) {
            start = split + 4;
            continue;
        }
        let mut length = None;
        let mut chunked = false;
        let mut retry_after = None;
        for l in lines {
            if let Some((k, v)) = l.split_once(':') {
                let (k, v) = (k.trim(), v.trim());
                if k.eq_ignore_ascii_case("content-length") {
                    length = v.parse::<usize>().ok();
                } else if k.eq_ignore_ascii_case("transfer-encoding")
                    && v.eq_ignore_ascii_case("chunked")
                {
                    chunked = true;
                } else if k.eq_ignore_ascii_case("retry-after") {
                    retry_after = v.parse::<f64>().ok();
                }
            }
        }
        let body = if decoded {
            rest.to_vec()
        } else if chunked {
            dechunk(rest).ok_or_else(|| bad("broken chunked body"))?
        } else {
            match length {
                Some(n) if n <= rest.len() => rest[..n].to_vec(),
                Some(_) => return Err(bad("body shorter than Content-Length")),
                None => rest.to_vec(),
            }
        };
        return Ok(Response {
            status,
            retry_after,
            body,
        });
    }
}

fn dechunk(mut b: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let eol = b.windows(2).position(|w| w == b"\r\n")?;
        let size = std::str::from_utf8(&b[..eol])
            .ok()?
            .split(';')
            .next()?
            .trim();
        let n = usize::from_str_radix(size, 16).ok()?;
        b = &b[eol + 2..];
        if n == 0 {
            return Some(out);
        }
        out.extend_from_slice(b.get(..n)?);
        b = b.get(n + 2..)?;
    }
}

/// The message of a refusal's `detail`: a string, an object (the hosted Jev
/// sends `{"error_type": ..., "message": ...}`), or a list of validation
/// errors (`loc: msg` each).
fn detail_text(v: &Value) -> String {
    let d = v.get("detail").or_else(|| v.get("error")).unwrap_or(v);
    match d {
        Value::String(s) => s.clone(),
        Value::Object(o) if o.get("message").is_some_and(Value::is_string) => {
            let msg = o["message"].as_str().unwrap_or_default();
            match o.get("error_type").and_then(Value::as_str) {
                Some(t) => format!("{msg} ({t})"),
                None => msg.to_string(),
            }
        }
        Value::Array(items) => items
            .iter()
            .map(|i| {
                let loc: Vec<String> = i
                    .get("loc")
                    .and_then(Value::as_array)
                    .map(|l| {
                        l.iter()
                            .map(|p| p.as_str().map_or_else(|| p.to_string(), str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                let msg = i.get("msg").and_then(Value::as_str).unwrap_or("invalid");
                format!("{}: {msg}", loc.join("."))
            })
            .collect::<Vec<_>>()
            .join("; "),
        other => other.to_string(),
    }
}

/// The JSON of a reply, or the error its `detail` carries.
fn parse_reply(status: u16, body: &[u8]) -> Result<Value, ClientError> {
    let v: Result<Value, _> = serde_json::from_slice(body);
    let message = || match &v {
        Ok(j) => detail_text(j),
        Err(_) => String::from_utf8_lossy(body).into_owned(),
    };
    match status {
        200..=299 => v.map_err(|e| ClientError::Failed(format!("reply JSON: {e}"))),
        400..=499 => Err(ClientError::Rejected(status, message())),
        _ => Err(ClientError::Failed(format!("HTTP {status}: {}", message()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// Serve `replies` in turn on a loopback port, each after reading a whole
    /// request; returns the address and the requests' heads as received.
    fn serve(
        replies: Vec<(&'static str, &'static str, &'static str)>,
    ) -> (String, std::sync::mpsc::Receiver<String>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for (status, extra, body) in replies {
                let (mut s, _) = l.accept().unwrap();
                let mut req = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let n = s.read(&mut buf).unwrap();
                    req.extend_from_slice(&buf[..n]);
                    if let Some(end) = req.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&req[..end]).to_string();
                        let len = head
                            .to_ascii_lowercase()
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .map_or(0, |v| v.trim().parse::<usize>().unwrap());
                        if req.len() >= end + 4 + len {
                            tx.send(String::from_utf8_lossy(&req).to_string()).unwrap();
                            break;
                        }
                    }
                }
                let reply = format!(
                    "HTTP/1.1 {status}\r\n{extra}Content-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                s.write_all(reply.as_bytes()).unwrap();
            }
        });
        (addr, rx)
    }

    fn request() -> Request {
        serde_json::from_str(
            r#"{"state":"s","questions":{"q":{"type":"noul","instructions":"i"}}}"#,
        )
        .unwrap()
    }

    #[test]
    fn sends_model_and_key_and_reads_the_evaluation() {
        let (addr, rx) = serve(vec![(
            "200 OK",
            "",
            r#"{"model":"m","answers":{"q":{"type":"noul","noul":0.25}},"usage":{"input_tokens":7,"output_tokens":0}}"#,
        )]);
        let c = Client::new(format!("http://{addr}/"))
            .with_api_key("k1")
            .with_model("jev-1.14.0");
        let eval = c.evaluate(&request()).unwrap();
        assert_eq!(eval.usage.input_tokens, 7);
        assert_eq!(
            eval.answer("q"),
            Some(crate::protocol::Answer::Noul { noul: 0.25 })
        );
        let sent = rx.recv().unwrap();
        assert!(sent.contains("Authorization: Bearer k1"));
        assert!(sent.contains(r#""model":"jev-1.14.0""#));
    }

    #[test]
    fn refusals_carry_the_documented_detail() {
        let (addr, _rx) = serve(vec![
            (
                "422 Unprocessable Entity",
                "",
                r#"{"detail":[{"loc":["body","model"],"msg":"Field required","type":"missing"}]}"#,
            ),
            (
                "400 Bad Request",
                "",
                r#"{"detail":"box questions need a vision model"}"#,
            ),
            (
                "401 Unauthorized",
                "WWW-Authenticate: Bearer\r\n",
                r#"{"detail":"Not authenticated"}"#,
            ),
        ]);
        let c = Client::new(addr);
        assert_eq!(
            c.evaluate(&request()).unwrap_err(),
            ClientError::Rejected(422, "body.model: Field required".into())
        );
        assert_eq!(
            c.models().unwrap_err(),
            ClientError::Rejected(400, "box questions need a vision model".into())
        );
        let e = c.models().unwrap_err();
        assert_eq!(e, ClientError::Rejected(401, "Not authenticated".into()));
        assert!(e.to_string().contains("TYPESAFE_API_KEY"));
    }

    #[test]
    fn rate_limits_are_retried_after_the_header() {
        let (addr, _rx) = serve(vec![
            (
                "429 Too Many Requests",
                "retry-after: 0.05\r\n",
                r#"{"detail":"slow down"}"#,
            ),
            ("200 OK", "", r#"{"models":[]}"#),
        ]);
        let t0 = std::time::Instant::now();
        assert_eq!(
            Client::new(addr).models().unwrap(),
            serde_json::json!({"models": []})
        );
        assert!(t0.elapsed() >= Duration::from_millis(50));
    }

    #[test]
    fn transfer_encodings_interim_replies_and_missing_servers() {
        let r = parse_http(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\n{\"a\r\n4\r\n\":1}\r\n0\r\n\r\n", false).unwrap();
        assert_eq!((r.status, r.body.as_slice()), (200, &b"{\"a\":1}"[..]));
        let r = parse_http(b"HTTP/1.1 100 Continue\r\n\r\nHTTP/2 401\r\ntransfer-encoding: chunked\r\n\r\n{\"detail\":\"x\"}", true).unwrap();
        assert_eq!(
            (r.status, r.body.as_slice()),
            (401, &b"{\"detail\":\"x\"}"[..])
        );
        // Port 9 on loopback: nothing listens there.
        assert!(matches!(
            Client::new("127.0.0.1:9").health(),
            Err(ClientError::Unreachable(_))
        ));
    }

    #[test]
    fn the_environment_configures_like_the_sdks() {
        let c = Client::new("api.example:443");
        assert_eq!(c.base(), "http://api.example:443");
        assert_eq!(c.model(), "jev-latest");
        assert_eq!(
            Client::new("https://api.typesafe.ai/").base(),
            "https://api.typesafe.ai"
        );
        let hosted = r#"{"detail":{"error_type":"authentication_error","message":"Must supply an API key!"}}"#;
        assert_eq!(
            parse_reply(403, hosted.as_bytes()).unwrap_err(),
            ClientError::Rejected(403, "Must supply an API key! (authentication_error)".into())
        );
    }

    /// The https path (curl) against the hosted Jev, without a key: it must
    /// come back as the documented refusal, not as a transport failure.
    /// Network; run with `cargo test -- --ignored hosted`.
    #[test]
    #[ignore]
    fn hosted_jev_without_a_key_is_refused() {
        let c = Client::new("https://api.typesafe.ai");
        match c.models() {
            Err(ClientError::Rejected(code, m)) => {
                assert!(code == 401 || code == 403, "{code}: {m}");
                assert!(m.contains("authentication_error"), "{m}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }
}
