//! Request validation as TypeSafe documents `POST /v1/systemone` (OpenAPI
//! 0.2.0, https://docs.typesafe.ai/api/system-one, read 2026-11-21), so a
//! request this server accepts is one the hosted Jev accepts, and a refusal
//! has the body the documentation gives it:
//!
//! - **422 `HTTPValidationError`**: the body does not match the schema,
//!   `{"detail": [{"loc": [...], "msg": "...", "type": "...", "input": "...",
//!   "ctx": {...}}]}`. The error types and messages are pydantic v2's, which
//!   the documented server uses (its example: `missing`, "Field required").
//! - **400 `HTTPBadRequestError`**: valid by the schema but not servable by a
//!   text model, `{"detail": "..."}`: image `assets` and `box` questions.
//!
//! The schema: `model` (string), `state` (string, object or array) and
//! `questions` (an object of at least one) are required; `assets` is
//! optional. A question is tagged by `type`: `noul` (optional
//! `instructions`, optional `criteria` with optional `true` and `false`),
//! `choice` (`criteria`: 1 to 255 named options, each a string, object,
//! array or null), `score` (`criteria`: 1 to 10 levels, each a string,
//! object or array), `box` (optional `instructions`). `instructions` may be
//! a string, object, array or null.

use serde_json::{Map, Value, json};

/// The question types the documented schema knows.
const TAGS: &str = "'noul', 'choice', 'score', 'box'";

/// One schema violation.
#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    /// Path to the value: `body` then field names and array indices.
    pub loc: Vec<Value>,
    pub msg: String,
    /// pydantic's machine-readable code (`missing`, `string_type`, ...).
    pub kind: &'static str,
    /// The rejected value, compact JSON, shortened with `…`.
    pub input: Option<String>,
    pub ctx: Option<Value>,
}

impl Issue {
    pub fn to_json(&self) -> Value {
        let mut o = Map::new();
        o.insert("loc".into(), Value::Array(self.loc.clone()));
        o.insert("msg".into(), json!(self.msg));
        o.insert("type".into(), json!(self.kind));
        if let Some(i) = &self.input {
            o.insert("input".into(), json!(i));
        }
        if let Some(c) = &self.ctx {
            o.insert("ctx".into(), c.clone());
        }
        Value::Object(o)
    }
}

/// The 422 body for `issues`.
pub fn detail(issues: &[Issue]) -> Value {
    json!({ "detail": issues.iter().map(Issue::to_json).collect::<Vec<_>>() })
}

/// A preview of a rejected value: compact JSON, at most 100 characters.
fn preview(v: &Value) -> String {
    let s = v.to_string();
    if s.chars().count() <= 100 {
        s
    } else {
        s.chars().take(99).collect::<String>() + "…"
    }
}

struct Checker {
    issues: Vec<Issue>,
}

impl Checker {
    fn push(
        &mut self,
        loc: &[Value],
        kind: &'static str,
        msg: String,
        input: Option<&Value>,
        ctx: Option<Value>,
    ) {
        self.issues.push(Issue {
            loc: loc.to_vec(),
            msg,
            kind,
            input: input.map(preview),
            ctx,
        });
    }

    fn missing(&mut self, loc: &[Value], parent: &Value) {
        self.push(loc, "missing", "Field required".into(), Some(parent), None);
    }

    /// A value that may be a string, an object or an array (and null when
    /// `nullable`): pydantic reports one error per member of the union.
    fn text_like(&mut self, loc: &[Value], v: &Value, nullable: bool) {
        let ok = matches!(v, Value::String(_) | Value::Object(_) | Value::Array(_))
            || (nullable && v.is_null());
        if ok {
            return;
        }
        for (member, kind, msg) in [
            ("str", "string_type", "Input should be a valid string"),
            (
                "dict[str,any]",
                "dict_type",
                "Input should be a valid dictionary",
            ),
            ("list[any]", "list_type", "Input should be a valid list"),
        ] {
            let mut l = loc.to_vec();
            l.push(json!(member));
            self.push(&l, kind, msg.into(), Some(v), None);
        }
    }

    /// Length bounds of a dictionary or list, pydantic's `too_short` / `too_long`.
    fn bounds(
        &mut self,
        loc: &[Value],
        v: &Value,
        len: usize,
        field: &str,
        min: usize,
        max: Option<usize>,
    ) {
        if len < min {
            self.push(
                loc,
                "too_short",
                format!(
                    "{field} should have at least {min} item{} after validation, not {len}",
                    if min == 1 { "" } else { "s" }
                ),
                Some(v),
                Some(json!({"field_type": field, "min_length": min, "actual_length": len})),
            );
        }
        if let Some(max) = max.filter(|m| len > *m) {
            self.push(
                loc,
                "too_long",
                format!("{field} should have at most {max} items after validation, not {len}"),
                Some(v),
                Some(json!({"field_type": field, "max_length": max, "actual_length": len})),
            );
        }
    }

    fn question(&mut self, loc: &[Value], q: &Value) {
        let Some(o) = q.as_object() else {
            self.push(
                loc,
                "model_attributes_type",
                "Input should be a valid dictionary or object to extract fields from".into(),
                Some(q),
                None,
            );
            return;
        };
        let tag = match o.get("type") {
            None => {
                self.push(
                    loc,
                    "union_tag_not_found",
                    "Unable to extract tag using discriminator 'type'".into(),
                    Some(q),
                    Some(json!({"discriminator": "'type'"})),
                );
                return;
            }
            Some(Value::String(t)) if matches!(t.as_str(), "noul" | "choice" | "score" | "box") => {
                t.as_str()
            }
            Some(t) => {
                let shown = t.as_str().map_or_else(|| t.to_string(), str::to_string);
                self.push(
                    loc,
                    "union_tag_invalid",
                    format!("Input tag '{shown}' found using 'type' does not match any of the expected tags: {TAGS}"),
                    Some(q),
                    Some(json!({"discriminator": "'type'", "tag": shown, "expected_tags": TAGS})),
                );
                return;
            }
        };
        let mut at = loc.to_vec();
        at.push(json!(tag));
        let field = |name: &str| {
            let mut l = at.clone();
            l.push(json!(name));
            l
        };
        if let Some(i) = o.get("instructions") {
            self.text_like(&field("instructions"), i, true);
        }
        match tag {
            "noul" => match o.get("criteria") {
                None | Some(Value::Null) => {}
                Some(Value::Object(c)) => {
                    for k in ["true", "false"] {
                        if let Some(v) = c.get(k) {
                            let mut l = field("criteria");
                            l.push(json!(k));
                            self.text_like(&l, v, true);
                        }
                    }
                }
                Some(other) => self.push(
                    &field("criteria"),
                    "model_attributes_type",
                    "Input should be a valid dictionary or object to extract fields from".into(),
                    Some(other),
                    None,
                ),
            },
            "choice" => match o.get("criteria") {
                None => self.missing(&field("criteria"), q),
                Some(Value::Object(c)) => {
                    self.bounds(
                        &field("criteria"),
                        &Value::Object(c.clone()),
                        c.len(),
                        "Dictionary",
                        1,
                        Some(255),
                    );
                    for (k, v) in c {
                        let mut l = field("criteria");
                        l.push(json!(k));
                        self.text_like(&l, v, true);
                    }
                }
                Some(other) => self.push(
                    &field("criteria"),
                    "dict_type",
                    "Input should be a valid dictionary".into(),
                    Some(other),
                    None,
                ),
            },
            "score" => match o.get("criteria") {
                None => self.missing(&field("criteria"), q),
                Some(Value::Array(levels)) => {
                    self.bounds(
                        &field("criteria"),
                        &Value::Array(levels.clone()),
                        levels.len(),
                        "List",
                        1,
                        Some(10),
                    );
                    for (i, v) in levels.iter().enumerate() {
                        let mut l = field("criteria");
                        l.push(json!(i));
                        self.text_like(&l, v, false);
                    }
                }
                Some(other) => self.push(
                    &field("criteria"),
                    "list_type",
                    "Input should be a valid list".into(),
                    Some(other),
                    None,
                ),
            },
            _ => {}
        }
    }
}

/// Every schema violation of a request body (empty: it matches).
pub fn check(body: &Value) -> Vec<Issue> {
    let mut c = Checker { issues: Vec::new() };
    let root = vec![json!("body")];
    let Some(o) = body.as_object() else {
        c.push(
            &root,
            "model_attributes_type",
            "Input should be a valid dictionary or object to extract fields from".into(),
            Some(body),
            None,
        );
        return c.issues;
    };
    let at = |name: &str| vec![json!("body"), json!(name)];
    match o.get("model") {
        None => c.missing(&at("model"), body),
        Some(Value::String(_)) => {}
        Some(other) => c.push(
            &at("model"),
            "string_type",
            "Input should be a valid string".into(),
            Some(other),
            None,
        ),
    }
    match o.get("state") {
        None => c.missing(&at("state"), body),
        Some(s) => c.text_like(&at("state"), s, false),
    }
    match o.get("questions") {
        None => c.missing(&at("questions"), body),
        Some(Value::Object(qs)) => {
            c.bounds(
                &at("questions"),
                &Value::Object(qs.clone()),
                qs.len(),
                "Dictionary",
                1,
                None,
            );
            for (id, q) in qs {
                let mut loc = at("questions");
                loc.push(json!(id));
                c.question(&loc, q);
            }
        }
        Some(other) => c.push(
            &at("questions"),
            "dict_type",
            "Input should be a valid dictionary".into(),
            Some(other),
            None,
        ),
    }
    if let Some(a) = o.get("assets") {
        match a {
            Value::Object(m) => c.bounds(&at("assets"), a, m.len(), "Dictionary", 1, None),
            other => c.push(
                &at("assets"),
                "dict_type",
                "Input should be a valid dictionary".into(),
                Some(other),
                None,
            ),
        }
    }
    c.issues
}

/// Why a schema-valid request needs what a text model does not have (the
/// 400 detail), or None.
pub fn needs_vision(body: &Value) -> Option<String> {
    if body.get("assets").is_some() {
        return Some(
            "image assets need a vision-language model; this server's model reads text only".into(),
        );
    }
    let boxes: Vec<&String> = body
        .get("questions")?
        .as_object()?
        .iter()
        .filter(|(_, q)| q.get("type").and_then(Value::as_str) == Some("box"))
        .map(|(id, _)| id)
        .collect();
    (!boxes.is_empty()).then(|| {
        format!(
            "box questions locate things in images and need a vision-language model; this server's model reads text only (questions: {})",
            boxes.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(body: Value) -> Vec<(String, &'static str)> {
        check(&body)
            .into_iter()
            .map(|i| {
                let loc: Vec<String> = i
                    .loc
                    .iter()
                    .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string))
                    .collect();
                (loc.join("."), i.kind)
            })
            .collect()
    }

    #[test]
    fn documented_examples_pass() {
        for q in [
            json!({"type": "noul", "instructions": "Is this message about billing?"}),
            json!({"type": "noul"}),
            json!({"type": "choice", "instructions": "Tone?", "criteria": {"angry": "upset", "calm": null}}),
            json!({"type": "choice", "criteria": {"only": null}}),
            json!({"type": "score", "criteria": ["Can wait"]}),
            json!({"type": "noul", "criteria": {"true": "spam"}}),
        ] {
            let body = json!({"model": "jev-latest", "state": "s", "questions": {"q": q}});
            assert_eq!(check(&body), vec![], "{body}");
        }
        let structured = json!({"model": "m", "state": {"subject": "x"}, "questions": {"q": {"type": "noul", "instructions": {"task": "t"}}}});
        assert!(check(&structured).is_empty());
    }

    #[test]
    fn violations_are_located_and_typed() {
        assert_eq!(
            kinds(json!({"state": "s", "questions": {"q": {"type": "noul"}}})),
            vec![("body.model".to_string(), "missing")]
        );
        assert_eq!(
            kinds(json!({"model": "m", "state": "s", "questions": {}})),
            vec![("body.questions".to_string(), "too_short")]
        );
        assert_eq!(
            kinds(json!({"model": "m", "state": "s", "questions": {"u": {"type": "score"}}})),
            vec![("body.questions.u.score.criteria".to_string(), "missing")]
        );
        let eleven: Vec<String> = (0..11).map(|i| i.to_string()).collect();
        assert_eq!(
            kinds(
                json!({"model": "m", "state": "s", "questions": {"u": {"type": "score", "criteria": eleven}}})
            ),
            vec![("body.questions.u.score.criteria".to_string(), "too_long")]
        );
        assert_eq!(
            kinds(json!({"model": "m", "state": "s", "questions": {"q": {"type": "rank"}}})),
            vec![("body.questions.q".to_string(), "union_tag_invalid")]
        );
        assert_eq!(
            kinds(json!({"model": "m", "state": 5, "questions": {"q": {"type": "noul"}}})).len(),
            3,
            "one error per member of the union"
        );
        assert_eq!(
            kinds(json!([1, 2])),
            vec![("body".to_string(), "model_attributes_type")]
        );
        let many: Map<String, Value> = (0..256).map(|i| (format!("o{i}"), Value::Null)).collect();
        assert_eq!(
            kinds(
                json!({"model": "m", "state": "s", "questions": {"c": {"type": "choice", "criteria": many}}})
            ),
            vec![("body.questions.c.choice.criteria".to_string(), "too_long")]
        );
    }

    #[test]
    fn vision_requests_are_named() {
        let b = json!({"model": "m", "state": "photo", "questions": {"r": {"type": "box"}}});
        assert!(check(&b).is_empty());
        assert!(needs_vision(&b).unwrap().contains("questions: r"));
        let a = json!({"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}, "assets": {"p": {"type": "image"}}});
        assert!(needs_vision(&a).unwrap().contains("assets"));
        let issue = &check(
            &json!({"model": "m", "state": "s", "questions": {"q": {"type": "x".repeat(200)}}}),
        )[0];
        assert!(issue.input.as_ref().unwrap().ends_with('…'));
        assert_eq!(
            detail(std::slice::from_ref(issue))["detail"][0]["type"],
            "union_tag_invalid"
        );
    }
}
