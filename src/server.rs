//! `jevbase serve`: TypeSafe's System One API over HTTP/1.1 as its
//! documentation specifies it (OpenAPI 0.2.0, https://docs.typesafe.ai/api/system-one;
//! models, aliases and limits, https://docs.typesafe.ai/models; read
//! 2026-11-21), so TypeSafe's SDKs pointed at it, Mechanical Jev, Intel-Phi-Jev's
//! clients and this crate's `Client` use it as they use the hosted Jev.
//!
//! ```text
//! POST /v1/systemone  200 {model, answers, usage}
//!                     400 {"detail": "..."}  valid but not servable here: images,
//!                         box questions, an unknown model, past the context limits
//!                     403 / 401 {"detail": {"error_type": "authentication_error",
//!                         "message": ...}}  `api_keys` set and no key / a wrong key
//!                         (as the hosted Jev answers)
//!                     422 {"detail": [...]}  not JSON, or not the documented schema
//!                     500 {"detail": "..."}  the engine failed
//! GET  /v1/models     {"models": [{name, description, release_date}]} as documented,
//!                     with an OpenAI-style "data" list beside it (Mechanical Jev
//!                     reads data[].id)
//! GET  /health        {"status": "ok", "subject": ..., "site": "cpu"} (as xks)
//! ```
//!
//! Model names: the loaded model's id, the documented aliases `jev-latest` and
//! `jev-preview`, and any versioned `jev-` id are all answered by the loaded
//! model; a response names the model that answered ("may differ from the
//! alias supplied in the request"). `TypeSafe-Danger-Zone-Long-Context: true`
//! lifts the context limits to 262k. One connection at a time, closed after
//! each response: the engine answers one request at a time anyway.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::engine::{EngineError, Limits};
use crate::judge::Judge;
use crate::protocol::Request;
use crate::validate;

/// Largest request body accepted (TypeSafe's 64k-token request is far less).
const MAX_BODY: usize = 8 << 20;

/// The documented aliases, answered by the loaded model.
const ALIASES: [&str; 2] = ["jev-latest", "jev-preview"];

/// One response: status, extra headers, JSON body.
struct Reply {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: Value,
}

impl Reply {
    fn json(status: u16, body: Value) -> Reply {
        Reply {
            status,
            headers: Vec::new(),
            body,
        }
    }

    fn detail(status: u16, detail: impl Into<String>) -> Reply {
        Reply::json(status, json!({ "detail": detail.into() }))
    }
}

/// What the server knows besides the judge.
struct Ctx<'a> {
    judge: &'a Judge,
    api_keys: &'a [String],
    /// The model file's date (`release_date`) and its Unix time (`created`).
    released: (String, u64),
}

/// Serve until the process is stopped. `on_ready` runs once the socket is
/// bound (to print the address).
pub fn serve(
    judge: &Judge,
    bind: &str,
    api_keys: &[String],
    on_ready: impl FnOnce(&str),
) -> Result<(), String> {
    let listener = TcpListener::bind(bind).map_err(|e| format!("cannot listen on {bind}: {e}"))?;
    let addr = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| bind.into());
    let ctx = Ctx {
        judge,
        api_keys,
        released: released(&judge.engine.info.path),
    };
    on_ready(&addr);
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                if let Err(e) = handle(&ctx, s) {
                    eprintln!("jevbase serve: {e}");
                }
            }
            Err(e) => eprintln!("jevbase serve: accept: {e}"),
        }
    }
    Ok(())
}

/// The model file's modification date, `YYYY-MM-DD`, and Unix time.
fn released(path: &std::path::Path) -> (String, u64) {
    let secs = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    (format!("{y:04}-{m:02}-{d:02}"), secs)
}

/// Whether `name` is answered by the loaded model `local`.
fn accepts(name: &str, local: &str) -> bool {
    name == local || name.starts_with("jev-")
}

/// Equal without an early exit on the first differing byte.
fn same_key(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0u8, |acc, (x, y)| acc | (x ^ y))
            == 0
}

fn handle(ctx: &Ctx, stream: TcpStream) -> Result<(), String> {
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
    let path = parts
        .next()
        .unwrap_or("")
        .split('?')
        .next()
        .unwrap_or("")
        .to_string();
    let mut headers: Vec<(String, String)> = Vec::new();
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
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    let header = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    };
    let length: usize = header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let t0 = Instant::now();
    let reply = if let Some(r) = authorize(ctx, &path, header("authorization")) {
        r
    } else {
        match (method.as_str(), path.as_str()) {
            ("GET", "/health") => Reply::json(
                200,
                json!({"status": "ok", "subject": ctx.judge.engine.info.name, "site": "cpu"}),
            ),
            ("GET", "/v1/models") => models(ctx),
            ("POST", "/v1/systemone") if length > MAX_BODY => {
                Reply::detail(413, format!("request body over {MAX_BODY} bytes"))
            }
            ("POST", "/v1/systemone") => {
                let mut body = vec![0u8; length];
                reader
                    .read_exact(&mut body)
                    .map_err(|e| format!("body: {e}"))?;
                let long = header("typesafe-danger-zone-long-context")
                    .is_some_and(|v| v.eq_ignore_ascii_case("true"));
                let multipart = header("content-type")
                    .is_some_and(|v| v.to_ascii_lowercase().starts_with("multipart/"));
                let r = system_one(ctx, &body, long, multipart);
                eprintln!(
                    "jevbase serve: POST /v1/systemone {} in {:.0} ms",
                    r.status,
                    t0.elapsed().as_secs_f64() * 1e3
                );
                r
            }
            (_, "/health" | "/v1/models") => Reply {
                status: 405,
                headers: vec![("Allow", "GET".into())],
                body: json!({"detail": "Method Not Allowed"}),
            },
            (_, "/v1/systemone") => Reply {
                status: 405,
                headers: vec![("Allow", "POST".into())],
                body: json!({"detail": "Method Not Allowed"}),
            },
            _ => Reply::detail(404, "Not Found"),
        }
    };
    respond(stream, &reply)
}

/// None when the call may proceed; otherwise the refusal the hosted Jev
/// gives (measured 2026-11-21 against https://api.typesafe.ai): 403 when
/// no key is sent, 401 when the key is wrong, both with
/// `{"detail": {"error_type": "authentication_error", "message": ...}}`.
/// `/health` is open.
fn authorize(ctx: &Ctx, path: &str, auth: Option<&str>) -> Option<Reply> {
    if ctx.api_keys.is_empty() || path == "/health" {
        return None;
    }
    let refuse = |status: u16, message: &str| Reply {
        status,
        headers: vec![("WWW-Authenticate", "Bearer".into())],
        body: json!({"detail": {"error_type": "authentication_error", "message": message}}),
    };
    let Some(token) = auth.and_then(|a| {
        a.split_once(' ')
            .filter(|(s, _)| s.eq_ignore_ascii_case("bearer"))
    }) else {
        return Some(refuse(
            403,
            "Must supply an API key (Authorization: Bearer KEY).",
        ));
    };
    let token = token.1.trim();
    if ctx.api_keys.iter().any(|k| same_key(k, token)) {
        None
    } else {
        Some(refuse(
            401,
            "Cannot authenticate with the server: check the API key.",
        ))
    }
}

/// A request id in the hosted Jev's form (`req_` and 32 hex digits), sent
/// as `x-typesafe-request-id` with every response for clients that log it.
fn request_id() -> String {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!(
        "req_{:016x}{:016x}",
        t as u64,
        n ^ u64::from(std::process::id())
    )
}

fn models(ctx: &Ctx) -> Reply {
    let info = &ctx.judge.engine.info;
    let (date, created) = &ctx.released;
    let local = info.name.clone();
    let mut names = vec![(
        local.clone(),
        format!(
            "Jev-like System One answers read from {} ({}) on the CPU",
            info.desc, local
        ),
    )];
    for a in ALIASES {
        names.push((a.to_string(), format!("Alias answered by {local}")));
    }
    Reply::json(
        200,
        json!({
            "models": names.iter().map(|(n, d)| json!({"name": n, "description": d, "release_date": date})).collect::<Vec<_>>(),
            "object": "list",
            "data": names.iter().map(|(n, _)| json!({"id": n, "object": "model", "created": created, "owned_by": "jevbase"})).collect::<Vec<_>>(),
        }),
    )
}

fn system_one(ctx: &Ctx, body: &[u8], long: bool, multipart: bool) -> Reply {
    if multipart {
        return Reply::detail(
            400,
            "multipart requests carry images, which need a vision-language model; this \
             server's model reads text only (send application/json)",
        );
    }
    let v: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return Reply::json(
                422,
                json!({"detail": [{
                    "loc": ["body", e.column()],
                    "msg": "JSON decode error",
                    "type": "json_invalid",
                    "input": {},
                    "ctx": {"error": e.to_string()},
                }]}),
            );
        }
    };
    let issues = validate::check(&v);
    if !issues.is_empty() {
        return Reply::json(422, validate::detail(&issues));
    }
    if let Some(why) = validate::needs_vision(&v) {
        return Reply::detail(400, why);
    }
    let local = &ctx.judge.engine.info.name;
    let asked = v.get("model").and_then(Value::as_str).unwrap_or("");
    if !accepts(asked, local) {
        return Reply::detail(
            400,
            format!(
                "unknown model `{asked}`: this server answers `{local}`, `jev-latest`, \
                 `jev-preview` and versioned `jev-` ids (GET /v1/models)"
            ),
        );
    }
    let req: Request = match serde_json::from_value(v) {
        Ok(r) => r,
        Err(e) => return Reply::detail(400, e.to_string()),
    };
    let limits = if long {
        Limits::LONG_CONTEXT
    } else {
        Limits::DEFAULT
    };
    match ctx.judge.evaluate_with(&req, limits) {
        Ok(eval) => Reply::json(200, serde_json::to_value(eval).expect("evaluation to JSON")),
        Err(EngineError::Rejected(m)) => Reply::detail(400, m),
        Err(EngineError::Failed(m)) => Reply::detail(500, m),
    }
}

fn respond(mut stream: TcpStream, reply: &Reply) -> Result<(), String> {
    let text = serde_json::to_string(&reply.body).expect("JSON text");
    let reason = match reply.status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        _ => "Internal Server Error",
    };
    let mut head = format!(
        "HTTP/1.1 {} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
         x-typesafe-request-id: {}\r\nConnection: close\r\n",
        reply.status,
        text.len(),
        request_id()
    );
    for (k, v) in &reply.headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(text.as_bytes()))
        .and_then(|_| stream.flush())
        .map_err(|e| format!("response: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_keys_and_dates() {
        assert!(accepts("Qwen3.8-2B-Q8_0", "Qwen3.8-2B-Q8_0"));
        assert!(accepts("jev-latest", "Qwen3.8-2B-Q8_0"));
        assert!(accepts("jev-1.14.0", "Qwen3.8-2B-Q8_0"));
        assert!(!accepts("gpt-5", "Qwen3.8-2B-Q8_0"));
        assert!(same_key("abc", "abc") && !same_key("abc", "abd") && !same_key("abc", "ab"));
        let (date, _) = released(std::path::Path::new("/nonexistent"));
        assert_eq!(date.len(), 10);
        assert_eq!(&date[4..5], "-");
    }
}
