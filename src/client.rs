//! A client of a running System One server (`jevbase serve`, or
//! Intel-Phi-Jev's `xks serve`): one `POST /v1/systemone` per evaluation,
//! HTTP/1.1 over `std::net`. No model and no llama.cpp: tools that only ask
//! (a hook, an MCP server) build with `default-features = false` and stay
//! small and quick to start, while one server keeps the model and its
//! cached state.

use std::fmt;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use serde_json::Value;

use crate::protocol::{Evaluation, Request};

/// Why a call did not produce an evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientError {
    /// Nothing answered at the address (the server is not running).
    Unreachable(String),
    /// The server refused the request (HTTP 4xx), with its message.
    Rejected(u16, String),
    /// The server failed (HTTP 5xx), or its reply could not be read.
    Failed(String),
}

impl fmt::Display for ClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreachable(m) => {
                write!(f, "no System One server: {m} (start one: jevbase serve)")
            }
            Self::Rejected(code, m) => write!(f, "request refused ({code}): {m}"),
            Self::Failed(m) => write!(f, "server failed: {m}"),
        }
    }
}

impl std::error::Error for ClientError {}

/// A System One server at one address.
#[derive(Debug, Clone)]
pub struct Client {
    addr: String,
    connect_timeout: Duration,
    timeout: Duration,
}

impl Client {
    /// Where `jevbase serve` listens by default.
    pub const DEFAULT_ADDR: &'static str = "127.0.0.1:8097";

    /// A client of `addr` (`host:port`, an `http://` prefix is accepted).
    /// Connecting gives up after 2 s, an answer after 300 s.
    pub fn new(addr: impl Into<String>) -> Self {
        let addr = addr.into();
        let addr = addr
            .strip_prefix("http://")
            .unwrap_or(&addr)
            .trim_end_matches('/')
            .to_string();
        Self {
            addr,
            connect_timeout: Duration::from_secs(2),
            timeout: Duration::from_secs(300),
        }
    }

    /// Give up on an answer after `t` (a decision on a long state takes seconds).
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// `GET /health`: the server's status object.
    pub fn health(&self) -> Result<Value, ClientError> {
        let (status, body) = self.call("GET", "/health", None)?;
        parse_reply(status, &body)
    }

    /// `POST /v1/systemone`: the request's evaluation.
    pub fn evaluate(&self, req: &Request) -> Result<Evaluation, ClientError> {
        let body = serde_json::to_vec(req).map_err(|e| ClientError::Failed(e.to_string()))?;
        let (status, reply) = self.call("POST", "/v1/systemone", Some(&body))?;
        let v = parse_reply(status, &reply)?;
        serde_json::from_value(v).map_err(|e| ClientError::Failed(format!("evaluation: {e}")))
    }

    fn socket(&self) -> Result<SocketAddr, ClientError> {
        self.addr
            .to_socket_addrs()
            .map_err(|e| ClientError::Unreachable(format!("{}: {e}", self.addr)))?
            .next()
            .ok_or_else(|| ClientError::Unreachable(format!("{}: no address", self.addr)))
    }

    /// One request on a fresh connection; the server closes it after the
    /// response. Returns the status code and the body.
    fn call(
        &self,
        method: &str,
        path: &str,
        body: Option<&[u8]>,
    ) -> Result<(u16, Vec<u8>), ClientError> {
        let sock = self.socket()?;
        let mut stream = TcpStream::connect_timeout(&sock, self.connect_timeout)
            .map_err(|e| ClientError::Unreachable(format!("{}: {e}", self.addr)))?;
        stream
            .set_read_timeout(Some(self.timeout))
            .and_then(|_| stream.set_write_timeout(Some(self.timeout)))
            .map_err(|e| ClientError::Failed(e.to_string()))?;
        let body = body.unwrap_or(&[]);
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            self.addr,
            body.len()
        );
        stream
            .write_all(head.as_bytes())
            .and_then(|_| stream.write_all(body))
            .map_err(|e| ClientError::Failed(format!("sending: {e}")))?;
        let mut raw = Vec::new();
        match stream.read_to_end(&mut raw) {
            Ok(_) => parse_http(&raw),
            // A server that closes with part of the request unread makes the
            // kernel reset the connection; a reply received whole before the
            // reset still counts.
            Err(e) => {
                parse_http(&raw).map_err(|_| ClientError::Failed(format!("reading the reply: {e}")))
            }
        }
    }
}

impl Default for Client {
    fn default() -> Self {
        Self::new(Self::DEFAULT_ADDR)
    }
}

/// Status code and body of a raw HTTP/1.1 response (Content-Length, chunked,
/// or read to the close).
fn parse_http(raw: &[u8]) -> Result<(u16, Vec<u8>), ClientError> {
    let bad = |m: &str| ClientError::Failed(format!("malformed HTTP reply: {m}"));
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| bad("no header end"))?;
    let head = std::str::from_utf8(&raw[..split]).map_err(|_| bad("header not UTF-8"))?;
    let rest = &raw[split + 4..];
    let mut lines = head.split("\r\n");
    let status: u16 = lines
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse().ok())
        .ok_or_else(|| bad("no status code"))?;
    let mut length = None;
    let mut chunked = false;
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            let (k, v) = (k.trim(), v.trim());
            if k.eq_ignore_ascii_case("content-length") {
                length = v.parse::<usize>().ok();
            } else if k.eq_ignore_ascii_case("transfer-encoding")
                && v.eq_ignore_ascii_case("chunked")
            {
                chunked = true;
            }
        }
    }
    let body = if chunked {
        dechunk(rest).ok_or_else(|| bad("broken chunked body"))?
    } else {
        match length {
            Some(n) if n <= rest.len() => rest[..n].to_vec(),
            Some(_) => return Err(bad("body shorter than Content-Length")),
            None => rest.to_vec(),
        }
    };
    Ok((status, body))
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

/// The JSON of a reply, or the error it carries.
fn parse_reply(status: u16, body: &[u8]) -> Result<Value, ClientError> {
    let v: Result<Value, _> = serde_json::from_slice(body);
    let message = || match &v {
        Ok(j) => j
            .get("error")
            .map(|e| e.as_str().map_or_else(|| e.to_string(), str::to_string))
            .unwrap_or_else(|| j.to_string()),
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

    /// Serve one response (`status` line, JSON `body`) on a loopback port,
    /// after reading the whole request; returns the address.
    fn serve_once(status: &'static str, body: &'static str) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut req = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = s.read(&mut buf).unwrap();
                req.extend_from_slice(&buf[..n]);
                if let Some(end) = req.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&req[..end]).to_ascii_lowercase();
                    let len = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .map_or(0, |v| v.trim().parse::<usize>().unwrap());
                    if req.len() >= end + 4 + len {
                        break;
                    }
                }
            }
            let reply = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            s.write_all(reply.as_bytes()).unwrap();
        });
        addr
    }

    #[test]
    fn evaluates_against_a_server() {
        let addr = serve_once(
            "200 OK",
            r#"{"model":"m","answers":{"q":{"type":"noul","noul":0.25}},"usage":{"input_tokens":7,"output_tokens":0}}"#,
        );
        let req: Request = serde_json::from_str(
            r#"{"state":"s","questions":{"q":{"type":"noul","instructions":"i"}}}"#,
        )
        .unwrap();
        let eval = Client::new(format!("http://{addr}/"))
            .evaluate(&req)
            .unwrap();
        assert_eq!(eval.usage.input_tokens, 7);
        assert_eq!(
            eval.answer("q"),
            Some(crate::protocol::Answer::Noul { noul: 0.25 })
        );
    }

    #[test]
    fn refusals_carry_the_message() {
        let addr = serve_once("422 Unprocessable Entity", r#"{"error":"no good"}"#);
        let e = Client::new(addr).health().unwrap_err();
        assert_eq!(e, ClientError::Rejected(422, "no good".into()));
    }

    #[test]
    fn chunked_bodies_and_missing_servers() {
        let (s, b) = parse_http(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\n{\"a\r\n4\r\n\":1}\r\n0\r\n\r\n").unwrap();
        assert_eq!((s, b.as_slice()), (200, &b"{\"a\":1}"[..]));
        // Port 9 on loopback: nothing listens there.
        assert!(matches!(
            Client::new("127.0.0.1:9").health(),
            Err(ClientError::Unreachable(_))
        ));
    }
}
