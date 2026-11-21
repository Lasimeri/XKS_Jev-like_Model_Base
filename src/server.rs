//! `jevbase serve`: the System One wire over HTTP/1.1, so TypeSafe's SDKs,
//! Mechanical Jev (`mjev`) and any other client of Intel-Phi-Jev's `xks`
//! work against this engine unchanged.
//!
//!   POST /v1/systemone   a request -> its evaluation (400 bad JSON,
//!                        422 a request that cannot be read as asked,
//!                        500 the engine failed)
//!   GET  /health         {"status":"ok","subject":...,"site":"cpu"}
//!   GET  /v1/models      the loaded model as an OpenAI-style list
//!
//! One connection at a time, closed after each response: the engine answers
//! one request at a time anyway (it holds one cached state).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::engine::EngineError;
use crate::judge::Judge;
use crate::protocol::Request;

/// Largest request body accepted (TypeSafe's 64k-token request is far less).
const MAX_BODY: usize = 8 << 20;

/// Serve until the process is stopped. `on_ready` runs once the socket is
/// bound (to print the address).
pub fn serve(judge: &Judge, bind: &str, on_ready: impl FnOnce(&str)) -> Result<(), String> {
    let listener = TcpListener::bind(bind).map_err(|e| format!("cannot listen on {bind}: {e}"))?;
    let addr = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| bind.into());
    on_ready(&addr);
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                if let Err(e) = handle(judge, s) {
                    eprintln!("jevbase serve: {e}");
                }
            }
            Err(e) => eprintln!("jevbase serve: accept: {e}"),
        }
    }
    Ok(())
}

fn handle(judge: &Judge, stream: TcpStream) -> Result<(), String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("request line: {e}"))?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut length = 0usize;
    loop {
        let mut h = String::new();
        if reader
            .read_line(&mut h)
            .map_err(|e| format!("header: {e}"))?
            == 0
        {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let name = judge.engine.info.name.clone();
    let (status, body) = match (method.as_str(), path.as_str()) {
        ("GET", "/health") => (200, json!({"status": "ok", "subject": name, "site": "cpu"})),
        ("GET", "/v1/models") => (
            200,
            json!({"object": "list", "data": [{"id": name, "object": "model", "owned_by": "local"}]}),
        ),
        ("POST", "/v1/systemone") if length > MAX_BODY => (
            413,
            json!({"error": format!("request body over {MAX_BODY} bytes")}),
        ),
        ("POST", "/v1/systemone") => {
            let mut buf = vec![0u8; length];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("body: {e}"))?;
            answer(judge, &buf)
        }
        (_, "/health" | "/v1/models" | "/v1/systemone") => (
            405,
            json!({"error": format!("{method} is not allowed on {path}")}),
        ),
        _ => (
            404,
            json!({"error": format!("no route {path} (POST /v1/systemone)")}),
        ),
    };
    respond(stream, status, &body)
}

fn answer(judge: &Judge, body: &[u8]) -> (u16, Value) {
    let t0 = Instant::now();
    let req: Request = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return (400, json!({"error": format!("request JSON: {e}")})),
    };
    let reply = match judge.evaluate(&req) {
        Ok(eval) => (200, serde_json::to_value(eval).expect("evaluation to JSON")),
        Err(EngineError::Rejected(m)) => (422, json!({"error": m})),
        Err(EngineError::Failed(m)) => (500, json!({"error": m})),
    };
    eprintln!(
        "jevbase serve: {} question(s), {} in {:.0} ms",
        req.questions.len(),
        reply.0,
        t0.elapsed().as_secs_f64() * 1e3
    );
    reply
}

fn respond(mut stream: TcpStream, status: u16, body: &Value) -> Result<(), String> {
    let text = serde_json::to_string(body).expect("JSON text");
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    )
    .and_then(|_| stream.flush())
    .map_err(|e| format!("response: {e}"))
}
