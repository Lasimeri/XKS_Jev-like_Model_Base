//! The terminal interface: `jevbase` with no command, or `jevbase tui [FILE]`.
//!
//! Three panes: the document (the request's state), the questions, and the
//! answers, a bar per option. The model loads in the background while the
//! request is edited, and asking runs in the background too, so keys keep
//! working. Requests open from and save to the same JSON files `jevbase ask`
//! and `jevbase serve` take.

pub mod edit;
pub mod term;
mod view;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Instant;

use serde_json::{Map, Value, json};

use crate::config::Config;
use crate::judge::Judge;
use crate::protocol::{Evaluation, Request, text_of};
use edit::Field;
use term::{Key, Term};

/// A question's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Noul,
    Choice,
    Score,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
        }
    }

    fn cycle(self, forward: bool) -> Self {
        match (self, forward) {
            (Self::Noul, true) | (Self::Score, false) => Self::Choice,
            (Self::Choice, true) | (Self::Noul, false) => Self::Score,
            (Self::Score, true) | (Self::Choice, false) => Self::Noul,
        }
    }

    /// What the options field holds for this type.
    pub fn options_hint(self) -> &'static str {
        match self {
            Self::Noul => "optional: a `true: ...` line and a `false: ...` line",
            Self::Choice => "one option a line: `key` or `key: description`",
            Self::Score => "one level a line, lowest first (2 to 10)",
        }
    }
}

/// A question as edited: its options as text, one a line.
#[derive(Debug, Clone)]
pub struct Draft {
    pub id: String,
    pub kind: Kind,
    pub instructions: String,
    pub options: String,
}

impl Draft {
    fn lines(&self) -> Vec<&str> {
        self.options
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect()
    }

    /// The question in System One form.
    pub fn to_json(&self) -> Result<Value, String> {
        let lines = self.lines();
        let q = match self.kind {
            Kind::Noul => {
                let pick = |name: &str| {
                    lines
                        .iter()
                        .find_map(|l| l.strip_prefix(name).and_then(|r| r.strip_prefix(':')))
                        .map(|r| r.trim().to_string())
                };
                match (pick("true"), pick("false")) {
                    (Some(t), Some(f)) => json!({"type": "noul", "instructions": self.instructions,
                        "criteria": {"true": t, "false": f}}),
                    _ => json!({"type": "noul", "instructions": self.instructions}),
                }
            }
            Kind::Choice => {
                if lines.is_empty() {
                    return Err(format!("`{}`: a choice needs at least one option", self.id));
                }
                let mut criteria = Map::new();
                for l in lines {
                    let (k, d) = match l.split_once(':') {
                        Some((k, d)) => (k.trim(), Value::String(d.trim().to_string())),
                        None => (l, Value::Null),
                    };
                    criteria.insert(k.to_string(), d);
                }
                json!({"type": "choice", "instructions": self.instructions, "criteria": criteria})
            }
            Kind::Score => {
                if !(2..=10).contains(&lines.len()) {
                    return Err(format!("`{}`: a score needs 2 to 10 levels", self.id));
                }
                json!({"type": "score", "instructions": self.instructions, "criteria": lines})
            }
        };
        Ok(q)
    }

    /// A question from System One form (structured descriptions as JSON text).
    pub fn from_json(id: &str, v: &Value) -> Result<Self, String> {
        let kind = match v.get("type").and_then(Value::as_str) {
            Some("noul") => Kind::Noul,
            Some("choice") => Kind::Choice,
            Some("score") => Kind::Score,
            _ => return Err(format!("question `{id}`: unknown type")),
        };
        let crit = v.get("criteria");
        let options = match kind {
            Kind::Noul => match crit {
                Some(c) if c.is_object() => format!(
                    "true: {}\nfalse: {}",
                    text_of(c.get("true").unwrap_or(&Value::Null)),
                    text_of(c.get("false").unwrap_or(&Value::Null))
                ),
                _ => String::new(),
            },
            Kind::Choice => crit
                .and_then(Value::as_object)
                .map(|o| {
                    o.iter()
                        .map(|(k, d)| match text_of(d) {
                            t if t.is_empty() => k.clone(),
                            t => format!("{k}: {t}"),
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default(),
            Kind::Score => crit
                .and_then(Value::as_array)
                .map(|a| a.iter().map(text_of).collect::<Vec<_>>().join("\n"))
                .unwrap_or_default(),
        };
        Ok(Self {
            id: id.to_string(),
            kind,
            instructions: text_of(v.get("instructions").unwrap_or(&Value::Null)),
            options,
        })
    }

    pub fn summary(&self) -> String {
        let n = self.lines().len();
        match self.kind {
            Kind::Noul => String::new(),
            Kind::Choice => format!("{n} options"),
            Kind::Score => format!("{n} levels"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pane {
    Document,
    Questions,
    Answers,
}

/// The question editor: id, type, instructions, options.
pub struct Form {
    /// The draft being edited, or None for a new one.
    pub index: Option<usize>,
    pub id: Field,
    pub kind: Kind,
    pub instructions: Field,
    pub options: Field,
    /// Field in focus: 0 id, 1 type, 2 instructions, 3 options.
    pub at: usize,
    pub error: Option<String>,
}

pub enum PathAction {
    Open,
    Save,
}

pub enum Mode {
    Main,
    Form(Form),
    Path(PathAction, Field),
    Help,
}

pub enum Status {
    Loading(Instant),
    Ready,
    Asking(Instant),
    Failed(String),
}

pub struct Answered {
    pub eval: Evaluation,
    pub seconds: f64,
}

enum Event {
    Loaded(Result<Judge, String>),
    Answered(Result<Evaluation, String>, f64),
}

pub struct App {
    pub doc: Field,
    pub drafts: Vec<Draft>,
    pub selected: usize,
    pub pane: Pane,
    pub mode: Mode,
    pub answered: Option<Answered>,
    pub answers_scroll: usize,
    pub status: Status,
    /// The last thing worth saying (a save, an error).
    pub note: String,
    /// Model and CPU line for the header.
    pub model: String,
    pub file: Option<PathBuf>,
    /// The requests opened (one per .json file, one per line of a .jsonl
    /// file), each with where it came from; Ctrl-N / Ctrl-P step through.
    pub set: Vec<(String, Request)>,
    /// The request of `set` shown.
    pub at: usize,
    judge: Option<Arc<Judge>>,
    tx: Sender<Event>,
    rx: Receiver<Event>,
    quit: bool,
}

/// The CPU features llama.cpp's kernels use, from its system info.
fn cpu_features() -> String {
    let info = crate::engine::system_info();
    ["AVX2", "FMA", "F16C", "AVX512", "OPENMP"]
        .iter()
        .filter(|f| info.contains(&format!("{f} = 1")))
        .copied()
        .collect::<Vec<_>>()
        .join(" ")
}

impl App {
    fn new(cfg: &Config) -> Self {
        let (tx, rx) = mpsc::channel();
        let stem = cfg
            .model
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        Self {
            doc: Field::new("", true),
            drafts: Vec::new(),
            selected: 0,
            pane: Pane::Document,
            mode: Mode::Main,
            answered: None,
            answers_scroll: 0,
            status: Status::Loading(Instant::now()),
            note: String::new(),
            model: format!(
                "{stem} · {} threads · {} · readout {}",
                cfg.threads,
                cpu_features(),
                cfg.readout.name()
            ),
            file: None,
            set: Vec::new(),
            at: 0,
            judge: None,
            tx,
            rx,
            quit: false,
        }
    }

    /// The request the panes describe.
    pub fn request(&self) -> Result<Request, String> {
        if self.drafts.is_empty() {
            return Err("no questions: press `a` in the Questions pane to add one".into());
        }
        let mut questions = Map::new();
        for d in &self.drafts {
            questions.insert(d.id.clone(), d.to_json()?);
        }
        Ok(Request {
            model: None,
            state: Value::String(self.doc.text()),
            questions,
        })
    }

    /// Open request files (a .json file is one request, a .jsonl file one a
    /// line, `#` lines skipped) as the set Ctrl-N / Ctrl-P step through, and
    /// show the first.
    fn open(&mut self, paths: &[PathBuf]) -> Result<(), String> {
        let mut set = Vec::new();
        for path in paths {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let name = path.file_name().map_or_else(
                || path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            if let Ok(req) = serde_json::from_str::<Request>(&text) {
                set.push((name, req));
                continue;
            }
            for (n, line) in text.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let req = serde_json::from_str::<Request>(line)
                    .map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?;
                set.push((format!("{name}:{}", n + 1), req));
            }
        }
        if set.is_empty() {
            return Err("no requests in those files".into());
        }
        self.set = set;
        self.file = paths.first().cloned();
        self.show(0)?;
        self.note = if self.set.len() > 1 {
            format!("{} requests loaded", self.set.len())
        } else {
            format!("opened {}", self.set[0].0)
        };
        Ok(())
    }

    /// Show request `i` of the set; edits to the one shown before are
    /// dropped (Ctrl-S saves them first).
    fn show(&mut self, i: usize) -> Result<(), String> {
        let (_, req) = &self.set[i];
        let drafts = req
            .questions
            .iter()
            .map(|(id, q)| Draft::from_json(id, q))
            .collect::<Result<Vec<_>, _>>()?;
        self.doc.set(&crate::protocol::render_state(&req.state));
        self.doc.to_start();
        self.drafts = drafts;
        self.at = i;
        self.selected = 0;
        self.answered = None;
        self.answers_scroll = 0;
        Ok(())
    }

    /// Step `by` requests through the set, wrapping around.
    fn step(&mut self, by: isize) {
        if self.set.len() < 2 {
            self.note = "one request open: Ctrl-O opens a .jsonl file of many".into();
            return;
        }
        if matches!(self.status, Status::Asking(_)) {
            self.note = "wait for the answers before moving on".into();
            return;
        }
        let n = self.set.len() as isize;
        let i = ((self.at as isize + by) % n + n) % n;
        match self.show(i as usize) {
            Ok(()) => self.note.clear(),
            Err(e) => self.note = e,
        }
    }

    fn save(&mut self, path: &Path) -> Result<(), String> {
        let req = self.request()?;
        let text = serde_json::to_string_pretty(&req).expect("request to JSON");
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))?;
        self.file = Some(path.to_path_buf());
        self.note = format!("saved {}", path.display());
        Ok(())
    }

    /// Start answering the request in the background.
    fn ask(&mut self) {
        let Some(judge) = self.judge.clone() else {
            self.note = "the model is still loading".into();
            return;
        };
        if matches!(self.status, Status::Asking(_)) {
            return;
        }
        let req = match self.request() {
            Ok(r) => r,
            Err(e) => {
                self.note = e;
                return;
            }
        };
        self.status = Status::Asking(Instant::now());
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let t0 = Instant::now();
            let r = judge.evaluate(&req).map_err(|e| e.to_string());
            let _ = tx.send(Event::Answered(r, t0.elapsed().as_secs_f64()));
        });
    }

    fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok(ev) = self.rx.try_recv() {
            changed = true;
            match ev {
                Event::Loaded(Ok(judge)) => {
                    self.model = format!("{} · {}", judge.engine.info.desc, self.model);
                    self.judge = Some(Arc::new(judge));
                    self.status = Status::Ready;
                }
                Event::Loaded(Err(e)) => self.status = Status::Failed(e),
                Event::Answered(Ok(eval), seconds) => {
                    self.answered = Some(Answered { eval, seconds });
                    self.answers_scroll = 0;
                    self.status = Status::Ready;
                    self.note.clear();
                }
                Event::Answered(Err(e), _) => {
                    self.status = Status::Ready;
                    self.note = e;
                }
            }
        }
        changed
    }

    fn open_form(&mut self, index: Option<usize>) {
        let d = index
            .and_then(|i| self.drafts.get(i))
            .cloned()
            .unwrap_or(Draft {
                id: format!("q{}", self.drafts.len() + 1),
                kind: Kind::Noul,
                instructions: String::new(),
                options: String::new(),
            });
        self.mode = Mode::Form(Form {
            index,
            id: Field::new(&d.id, false),
            kind: d.kind,
            instructions: Field::new(&d.instructions, false),
            options: Field::new(&d.options, true),
            at: if index.is_some() { 2 } else { 0 },
            error: None,
        });
    }

    fn key(&mut self, k: Key) {
        if matches!(k, Key::Ctrl('q') | Key::Ctrl('c')) {
            self.quit = true;
            return;
        }
        match std::mem::replace(&mut self.mode, Mode::Main) {
            Mode::Help => {}
            Mode::Form(f) => self.form_key(f, k),
            Mode::Path(action, f) => self.path_key(action, f, k),
            Mode::Main => self.main_key(k),
        }
    }

    fn main_key(&mut self, k: Key) {
        match k {
            Key::F(5) | Key::Ctrl('r') => self.ask(),
            Key::Ctrl('n') => self.step(1),
            Key::Ctrl('p') => self.step(-1),
            Key::F(1) => self.mode = Mode::Help,
            Key::Tab | Key::BackTab => {
                let forward = k == Key::Tab;
                self.pane = match (self.pane, forward) {
                    (Pane::Document, true) | (Pane::Answers, false) => Pane::Questions,
                    (Pane::Questions, true) | (Pane::Document, false) => Pane::Answers,
                    (Pane::Answers, true) | (Pane::Questions, false) => Pane::Document,
                };
            }
            Key::Ctrl('o') => {
                let start = self
                    .file
                    .as_ref()
                    .map_or("examples/query.json".into(), |p| p.display().to_string());
                self.mode = Mode::Path(PathAction::Open, Field::new(&start, false));
            }
            Key::Ctrl('s') => {
                let start = self
                    .file
                    .as_ref()
                    .map_or("request.json".into(), |p| p.display().to_string());
                self.mode = Mode::Path(PathAction::Save, Field::new(&start, false));
            }
            _ => match self.pane {
                Pane::Document => {
                    self.doc.key(&k);
                }
                Pane::Questions => self.questions_key(k),
                Pane::Answers => match k {
                    Key::Up => self.answers_scroll = self.answers_scroll.saturating_sub(1),
                    Key::Down => self.answers_scroll += 1,
                    Key::PageUp => self.answers_scroll = self.answers_scroll.saturating_sub(10),
                    Key::PageDown => self.answers_scroll += 10,
                    Key::Home => self.answers_scroll = 0,
                    Key::Char('?') => self.mode = Mode::Help,
                    _ => {}
                },
            },
        }
    }

    fn questions_key(&mut self, k: Key) {
        match k {
            Key::Up => self.selected = self.selected.saturating_sub(1),
            Key::Down if self.selected + 1 < self.drafts.len() => self.selected += 1,
            Key::Char('a') | Key::Char('n') => self.open_form(None),
            Key::Enter | Key::Char('e') if !self.drafts.is_empty() => {
                self.open_form(Some(self.selected))
            }
            Key::Delete | Key::Char('d') if !self.drafts.is_empty() => {
                let gone = self.drafts.remove(self.selected);
                self.selected = self.selected.min(self.drafts.len().saturating_sub(1));
                self.note = format!("removed `{}`", gone.id);
            }
            Key::Char('?') => self.mode = Mode::Help,
            _ => {}
        }
    }

    fn form_key(&mut self, mut f: Form, k: Key) {
        match k {
            Key::Esc => return,
            Key::F(2) | Key::Ctrl('s') => {
                let d = Draft {
                    id: f.id.text().trim().to_string(),
                    kind: f.kind,
                    instructions: f.instructions.text(),
                    options: f.options.text(),
                };
                let clash = self
                    .drafts
                    .iter()
                    .enumerate()
                    .any(|(i, o)| o.id == d.id && Some(i) != f.index);
                let check = if d.id.is_empty() {
                    Err("the id is empty".to_string())
                } else if clash {
                    Err(format!("another question is already called `{}`", d.id))
                } else if d.instructions.trim().is_empty() {
                    Err("the instructions are empty".to_string())
                } else {
                    d.to_json().map(|_| ())
                };
                match check {
                    Ok(()) => {
                        match f.index {
                            Some(i) => self.drafts[i] = d,
                            None => {
                                self.drafts.push(d);
                                self.selected = self.drafts.len() - 1;
                            }
                        }
                        return;
                    }
                    Err(e) => f.error = Some(e),
                }
            }
            Key::Tab => f.at = (f.at + 1) % 4,
            Key::BackTab => f.at = (f.at + 3) % 4,
            Key::Enter if f.at < 3 => f.at += 1,
            _ => match f.at {
                0 => {
                    f.id.key(&k);
                }
                1 => match k {
                    Key::Left => f.kind = f.kind.cycle(false),
                    Key::Right | Key::Char(' ') => f.kind = f.kind.cycle(true),
                    _ => {}
                },
                2 => {
                    f.instructions.key(&k);
                }
                _ => {
                    f.options.key(&k);
                }
            },
        }
        self.mode = Mode::Form(f);
    }

    fn path_key(&mut self, action: PathAction, mut f: Field, k: Key) {
        match k {
            Key::Esc => {}
            Key::Enter => {
                let path = PathBuf::from(f.text().trim());
                let r = match action {
                    PathAction::Open => self.open(&[path]),
                    PathAction::Save => self.save(&path),
                };
                if let Err(e) = r {
                    self.note = e;
                }
            }
            _ => {
                f.key(&k);
                self.mode = Mode::Path(action, f);
            }
        }
    }
}

/// The examples opened when no file is named, in this order, when present.
const EXAMPLES: [&str; 4] = [
    "examples/query.json",
    "examples/wide_choice.jsonl",
    "examples/long_sessions.jsonl",
    "examples/dev_tasks.jsonl",
];

/// Start an app on `files`, or on the examples when none are named.
fn start(cfg: &Config, files: &[PathBuf]) -> Result<App, String> {
    let mut app = App::new(cfg);
    let files: Vec<PathBuf> = if files.is_empty() {
        EXAMPLES
            .iter()
            .map(PathBuf::from)
            .filter(|p| p.is_file())
            .collect()
    } else {
        files.to_vec()
    };
    if files.is_empty() {
        app.note = "type the document, then Tab to Questions and `a` to add one".into();
    } else {
        app.open(&files)?;
    }
    Ok(app)
}

/// Run the interface until Ctrl-Q.
pub fn run(cfg: &Config, files: &[PathBuf]) -> Result<(), String> {
    let mut app = start(cfg, files)?;
    let tx = app.tx.clone();
    let c = cfg.clone();
    std::thread::spawn(move || {
        let _ = tx.send(Event::Loaded(crate::open(&c)));
    });
    let mut term = Term::open()?;
    let mut size = (0, 0);
    let mut dirty = true;
    while !app.quit {
        dirty |= app.poll();
        let now = Term::size();
        if now != size {
            size = now;
            dirty = true;
        }
        // A running clock (loading, asking) redraws every tick.
        dirty |= matches!(app.status, Status::Loading(_) | Status::Asking(_));
        if dirty {
            let frame = view::draw(&mut app, size.0, size.1);
            use std::io::Write;
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(frame.ansi().as_bytes());
            let _ = out.flush();
            dirty = false;
        }
        if let Some(k) = term.key() {
            app.key(k);
            dirty = true;
        }
    }
    drop(term);
    Ok(())
}

/// One frame as plain text, for checking the layout without a terminal;
/// with `ask`, after loading the model and answering the request.
pub fn snapshot(
    cfg: &Config,
    files: &[PathBuf],
    cols: usize,
    rows: usize,
    ask: bool,
) -> Result<String, String> {
    let mut app = start(cfg, files)?;
    if ask {
        let judge = crate::open(cfg)?;
        let _ = app.tx.send(Event::Loaded(Ok(judge)));
        app.poll();
        let req = app.request()?;
        let t0 = Instant::now();
        let r = app
            .judge
            .as_ref()
            .expect("loaded")
            .evaluate(&req)
            .map_err(|e| e.to_string());
        let _ = app.tx.send(Event::Answered(r, t0.elapsed().as_secs_f64()));
        app.poll();
    }
    Ok(view::draw(&mut app, cols, rows).plain())
}
