//! Drawing: a canvas of styled cells, the panes and overlays laid out on it,
//! and one ANSI frame out (inside a synchronized update, so no flicker).

use serde_json::Value;

use super::{App, Mode, Pane, PathAction, Status};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Style {
    /// ANSI foreground (30..=37, 90..=97), 0 for the default.
    fg: u8,
    bold: bool,
    dim: bool,
    rev: bool,
}

const PLAIN: Style = Style {
    fg: 0,
    bold: false,
    dim: false,
    rev: false,
};
const DIM: Style = Style {
    fg: 0,
    bold: false,
    dim: true,
    rev: false,
};
const BOLD: Style = Style {
    fg: 0,
    bold: true,
    dim: false,
    rev: false,
};
const REV: Style = Style {
    fg: 0,
    bold: false,
    dim: false,
    rev: true,
};
const CYAN: Style = Style {
    fg: 36,
    bold: true,
    dim: false,
    rev: false,
};
const GREEN: Style = Style {
    fg: 32,
    bold: false,
    dim: false,
    rev: false,
};
const YELLOW: Style = Style {
    fg: 33,
    bold: true,
    dim: false,
    rev: false,
};
const RED: Style = Style {
    fg: 31,
    bold: true,
    dim: false,
    rev: false,
};

pub struct Canvas {
    w: usize,
    h: usize,
    cells: Vec<(char, Style)>,
    cursor: Option<(usize, usize)>,
}

impl Canvas {
    fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            cells: vec![(' ', PLAIN); w * h],
            cursor: None,
        }
    }

    /// Write `s` from (x, y), clipped to `max` cells and the canvas; returns
    /// the cells written.
    fn put(&mut self, x: usize, y: usize, s: &str, st: Style, max: usize) -> usize {
        if y >= self.h {
            return 0;
        }
        let mut n = 0;
        for c in s.chars() {
            if n >= max || x + n >= self.w {
                break;
            }
            let c = if c.is_control() { ' ' } else { c };
            self.cells[y * self.w + x + n] = (c, st);
            n += 1;
        }
        n
    }

    fn fill(&mut self, x: usize, y: usize, w: usize, c: char, st: Style) {
        for i in 0..w {
            self.put(x + i, y, &c.to_string(), st, 1);
        }
    }

    /// A box with a title; a focused box is drawn bold cyan.
    fn frame(&mut self, x: usize, y: usize, w: usize, h: usize, title: &str, focused: bool) {
        if w < 2 || h < 2 {
            return;
        }
        let st = if focused { CYAN } else { DIM };
        self.put(x, y, "┌", st, 1);
        self.fill(x + 1, y, w - 2, '─', st);
        self.put(x + w - 1, y, "┐", st, 1);
        for r in 1..h - 1 {
            self.put(x, y + r, "│", st, 1);
            self.fill(x + 1, y + r, w - 2, ' ', PLAIN);
            self.put(x + w - 1, y + r, "│", st, 1);
        }
        self.put(x, y + h - 1, "└", st, 1);
        self.fill(x + 1, y + h - 1, w - 2, '─', st);
        self.put(x + w - 1, y + h - 1, "┘", st, 1);
        if w > 6 {
            let t = format!(" {title} ");
            self.put(x + 2, y, &t, if focused { CYAN } else { BOLD }, w - 4);
        }
    }

    /// The frame as terminal output.
    pub fn ansi(&self) -> String {
        let mut out = String::with_capacity(self.w * self.h * 4);
        out.push_str("\x1b[?2026h\x1b[?25l\x1b[H");
        let mut cur: Option<Style> = None;
        for y in 0..self.h {
            out.push_str(&format!("\x1b[{};1H", y + 1));
            for x in 0..self.w {
                let (c, st) = self.cells[y * self.w + x];
                if cur != Some(st) {
                    out.push_str("\x1b[0");
                    if st.bold {
                        out.push_str(";1");
                    }
                    if st.dim {
                        out.push_str(";2");
                    }
                    if st.rev {
                        out.push_str(";7");
                    }
                    if st.fg != 0 {
                        out.push_str(&format!(";{}", st.fg));
                    }
                    out.push('m');
                    cur = Some(st);
                }
                out.push(c);
            }
        }
        out.push_str("\x1b[0m");
        if let Some((x, y)) = self.cursor {
            out.push_str(&format!("\x1b[{};{}H\x1b[?25h", y + 1, x + 1));
        }
        out.push_str("\x1b[?2026l");
        out
    }

    /// The frame as plain text (for `--snapshot`), trailing spaces kept so
    /// every line has the canvas width.
    pub fn plain(&self) -> String {
        let mut out = String::new();
        for y in 0..self.h {
            out.extend(self.cells[y * self.w..(y + 1) * self.w].iter().map(|c| c.0));
            out.push('\n');
        }
        out
    }
}

/// A bar of `width` cells filled to `p` in eighths of a cell.
fn bar(p: f64, width: usize) -> String {
    const PARTS: [char; 8] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'];
    let eighths = (p.clamp(0.0, 1.0) * (width * 8) as f64).round() as usize;
    let mut s = "█".repeat(eighths / 8);
    if eighths % 8 > 0 {
        s.push(PARTS[eighths % 8]);
    }
    s
}

/// One line of the answers pane: spans of (text, style).
type Line = Vec<(String, Style)>;

fn answer_lines(app: &App, width: usize) -> Vec<Line> {
    let Some(a) = &app.answered else {
        let hint = match app.status {
            Status::Loading(_) => "the model is loading; edit the request meanwhile",
            _ => "F5 or Ctrl-R asks the questions",
        };
        return vec![vec![(hint.into(), DIM)]];
    };
    let mut lines = Vec::new();
    let idw = a
        .eval
        .answers
        .keys()
        .map(|k| k.chars().count())
        .max()
        .unwrap_or(2)
        .clamp(2, 18);
    for (id, v) in &a.eval.answers {
        let kind = v.get("type").and_then(Value::as_str).unwrap_or("");
        let num = |k: &str| v.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        let probs: Vec<(String, f64)> = v
            .get("probabilities")
            .and_then(Value::as_object)
            .map(|o| {
                o.iter()
                    .map(|(k, p)| (k.clone(), p.as_f64().unwrap_or(0.0)))
                    .collect()
            })
            .unwrap_or_default();
        let head = format!("{id:<idw$}  {kind:<6}  ");
        match kind {
            "noul" => {
                let p = num("noul");
                let label = if p >= 0.5 { "yes" } else { "no" };
                let bw = width.saturating_sub(head.chars().count() + 24).max(4);
                lines.push(vec![
                    (head, BOLD),
                    (
                        format!("{label:<3} "),
                        if p >= 0.5 { GREEN } else { YELLOW },
                    ),
                    (format!("P(yes) {p:.4} "), PLAIN),
                    (format!("{:<bw$}", bar(p, bw)), GREEN),
                ]);
            }
            "choice" => {
                let pick = v.get("choice").and_then(Value::as_str).unwrap_or("");
                lines.push(vec![
                    (head, BOLD),
                    (format!("→ {pick}"), GREEN),
                    (format!("   confidence {:.2}", num("confidence")), DIM),
                ]);
                let mut sorted = probs.clone();
                sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
                let kw = sorted
                    .iter()
                    .map(|(k, _)| k.chars().count())
                    .max()
                    .unwrap_or(1)
                    .min(20);
                for (k, p) in sorted {
                    let bw = width.saturating_sub(idw + kw + 16).max(4);
                    let st = if k == pick { GREEN } else { DIM };
                    lines.push(vec![
                        (format!("{:idw$}    {k:<kw$} {p:.4} ", ""), st),
                        (bar(p, bw), st),
                    ]);
                }
            }
            "score" => {
                let legend = v.get("legend").and_then(Value::as_object);
                lines.push(vec![
                    (head, BOLD),
                    (
                        format!(
                            "level {:.2} of 0..{}",
                            num("score"),
                            probs.len().saturating_sub(1)
                        ),
                        GREEN,
                    ),
                    (format!("   confidence {:.2}", num("confidence")), DIM),
                ]);
                for (k, p) in probs {
                    let name = legend
                        .and_then(|l| l.get(&k))
                        .map(crate::protocol::text_of)
                        .unwrap_or_default();
                    let name: String = format!("{k} {name}").chars().take(22).collect();
                    let bw = width.saturating_sub(idw + 22 + 16).max(4);
                    lines.push(vec![
                        (format!("{:idw$}    {name:<22} {p:.4} ", ""), DIM),
                        (bar(p, bw), GREEN),
                    ]);
                }
            }
            _ => lines.push(vec![(head, BOLD), (v.to_string(), RED)]),
        }
    }
    lines
}

/// Draw the whole interface for a terminal of `w` x `h`.
pub fn draw(app: &mut App, w: usize, h: usize) -> Canvas {
    let mut c = Canvas::new(w, h);
    if w < 60 || h < 16 {
        c.put(
            0,
            0,
            &format!("jevbase needs at least 60x16 (this is {w}x{h})"),
            RED,
            w,
        );
        return c;
    }
    // Header.
    let status = match &app.status {
        Status::Loading(t) => format!("loading model {:.1} s", t.elapsed().as_secs_f64()),
        Status::Ready => "ready".to_string(),
        Status::Asking(t) => format!("asking {:.1} s", t.elapsed().as_secs_f64()),
        Status::Failed(e) => format!("model failed: {e}"),
    };
    c.fill(0, 0, w, ' ', REV);
    let left = format!(" jevbase │ {} ", app.model);
    let right = format!(" {status} ");
    let rw = right.chars().count().min(w / 2);
    c.put(0, 0, &left, REV, w.saturating_sub(rw + 1));
    let st = match app.status {
        Status::Failed(_) => RED,
        Status::Ready => Style {
            rev: true,
            bold: true,
            ..PLAIN
        },
        _ => Style {
            rev: true,
            ..YELLOW
        },
    };
    c.put(w - rw, 0, &right, st, rw);

    // Panes.
    let top = ((h - 3) * 9 / 20).max(5);
    let docw = w * 3 / 5;
    let source = match app.set.get(app.at) {
        Some((name, _)) if app.set.len() > 1 => {
            format!(" · {name} ({}/{})", app.at + 1, app.set.len())
        }
        Some((name, _)) => format!(" · {name}"),
        None => String::new(),
    };
    c.frame(
        0,
        1,
        docw,
        top,
        &format!("Document{source}"),
        app.pane == Pane::Document,
    );
    let (rows, cur) = app.doc.view(docw - 2, top - 2);
    for (i, r) in rows.iter().enumerate() {
        c.put(1, 2 + i, r, PLAIN, docw - 2);
    }
    if app.doc.is_empty() {
        c.put(1, 2, "the state the questions are about", DIM, docw - 2);
    }
    if app.pane == Pane::Document && matches!(app.mode, Mode::Main) {
        c.cursor = Some((1 + cur.0, 2 + cur.1));
    }

    let qw = w - docw;
    c.frame(
        docw,
        1,
        qw,
        top,
        &format!("Questions ({})", app.drafts.len()),
        app.pane == Pane::Questions,
    );
    let visible = top - 2;
    let first = app.selected.saturating_sub(visible.saturating_sub(1));
    for (i, d) in app.drafts.iter().enumerate().skip(first).take(visible) {
        let sel = i == app.selected;
        let line = format!(
            "{} {:<12} {:<6} {}",
            if sel { '▸' } else { ' ' },
            d.id,
            d.kind.name(),
            d.summary()
        );
        let st = if sel && app.pane == Pane::Questions {
            REV
        } else if sel {
            BOLD
        } else {
            PLAIN
        };
        c.put(docw + 1, 2 + i - first, &line, st, qw - 2);
    }
    if app.drafts.is_empty() {
        c.put(docw + 1, 2, "none yet: Tab here, then `a`", DIM, qw - 2);
    }

    let ay = 1 + top;
    let ah = h - ay - 1;
    let title = match &app.answered {
        Some(a) => {
            let n = a.eval.answers.len();
            format!(
                "Answers · {n} decisions in {:.2} s · {:.2} decisions/s · {} prompt tokens",
                a.seconds,
                n as f64 / a.seconds.max(1e-9),
                a.eval.usage.input_tokens
            )
        }
        None => "Answers".to_string(),
    };
    c.frame(0, ay, w, ah, &title, app.pane == Pane::Answers);
    let lines = answer_lines(app, w - 2);
    app.answers_scroll = app.answers_scroll.min(lines.len().saturating_sub(1));
    for (i, line) in lines
        .iter()
        .skip(app.answers_scroll)
        .take(ah - 2)
        .enumerate()
    {
        let mut x = 1;
        for (text, st) in line {
            x += c.put(x, ay + 1 + i, text, *st, w - 1 - x);
        }
    }

    // Footer: the keys that work here, or the last note.
    let keys = match (&app.mode, app.pane) {
        (Mode::Form(_), _) => "Tab next field · ←/→ type · F2/Ctrl-S keep · Esc cancel",
        (Mode::Path(..), _) => "Enter confirm · Esc cancel",
        (Mode::Help, _) => "any key closes the help",
        (_, Pane::Document) => {
            "F5 ask · Ctrl-N/P next/prev request · Tab panes · Ctrl-O open · Ctrl-S save · F1 help · Ctrl-Q quit"
        }
        (_, Pane::Questions) => {
            "F5 ask · Ctrl-N/P next/prev request · a add · Enter edit · d remove · ↑/↓ select · Tab panes · Ctrl-Q quit"
        }
        (_, Pane::Answers) => {
            "F5 ask · Ctrl-N/P next/prev request · ↑/↓ PgUp/PgDn scroll · Tab panes · Ctrl-Q quit"
        }
    };
    let note_style = if app.note.contains(':') || app.note.starts_with("no ") {
        YELLOW
    } else {
        DIM
    };
    if app.note.is_empty() {
        c.put(1, h - 1, keys, DIM, w - 2);
    } else {
        let n = c.put(1, h - 1, &app.note, note_style, w - 2);
        c.put(n + 3, h - 1, keys, DIM, w.saturating_sub(n + 4));
    }

    match &mut app.mode {
        Mode::Main => {}
        Mode::Help => help(&mut c),
        Mode::Path(action, f) => {
            let bw = (w - 8).min(80);
            let (x, y) = ((w - bw) / 2, h / 2 - 2);
            let title = match action {
                PathAction::Open => "Open a request file (.json)",
                PathAction::Save => "Save the request as",
            };
            c.frame(x, y, bw, 3, title, true);
            let (rows, cur) = f.view(bw - 2, 1);
            c.put(
                x + 1,
                y + 1,
                rows.first().map_or("", |s| s.as_str()),
                PLAIN,
                bw - 2,
            );
            c.cursor = Some((x + 1 + cur.0, y + 1));
        }
        Mode::Form(f) => {
            let bw = (w - 6).min(96);
            let bh = (h - 4).min(18);
            let (x, y) = ((w - bw) / 2, (h - bh) / 2);
            c.frame(
                x,
                y,
                bw,
                bh,
                if f.index.is_some() {
                    "Edit question"
                } else {
                    "New question"
                },
                true,
            );
            let lw = 14;
            let fw = bw - lw - 3;
            let label = |c: &mut Canvas, row: usize, text: &str, on: bool| {
                c.put(x + 2, y + row, text, if on { CYAN } else { BOLD }, lw);
            };
            label(&mut c, 1, "id", f.at == 0);
            let (r, cur) = f.id.view(fw, 1);
            c.put(
                x + 2 + lw,
                y + 1,
                r.first().map_or("", |s| s.as_str()),
                PLAIN,
                fw,
            );
            if f.at == 0 {
                c.cursor = Some((x + 2 + lw + cur.0, y + 1));
            }
            label(&mut c, 2, "type", f.at == 1);
            for (i, k) in [super::Kind::Noul, super::Kind::Choice, super::Kind::Score]
                .iter()
                .enumerate()
            {
                let on = *k == f.kind;
                let st = if on && f.at == 1 {
                    REV
                } else if on {
                    BOLD
                } else {
                    DIM
                };
                c.put(x + 2 + lw + i * 9, y + 2, &format!(" {} ", k.name()), st, 8);
            }
            label(&mut c, 3, "instructions", f.at == 2);
            let (r, cur) = f.instructions.view(fw, 1);
            c.put(
                x + 2 + lw,
                y + 3,
                r.first().map_or("", |s| s.as_str()),
                PLAIN,
                fw,
            );
            if f.at == 2 {
                c.cursor = Some((x + 2 + lw + cur.0, y + 3));
            }
            label(&mut c, 4, "options", f.at == 3);
            c.put(x + 2 + lw, y + 4, f.kind.options_hint(), DIM, fw);
            let oh = bh.saturating_sub(8).max(1);
            let (rows, cur) = f.options.view(bw - 4, oh);
            for (i, r) in rows.iter().enumerate() {
                c.put(x + 2, y + 5 + i, r, PLAIN, bw - 4);
            }
            if f.at == 3 {
                c.cursor = Some((x + 2 + cur.0, y + 5 + cur.1));
            }
            if let Some(e) = &f.error {
                c.put(x + 2, y + bh - 2, e, RED, bw - 4);
            }
        }
    }
    c
}

fn help(c: &mut Canvas) {
    const TEXT: &[&str] = &[
        "The document is the state every question is about; each question is",
        "answered by reading the model's next-token probabilities, never text.",
        "",
        "F5, Ctrl-R    ask every question (runs in the background)",
        "Tab, S-Tab    move between Document, Questions and Answers",
        "Ctrl-N/P      next / previous request of the files opened",
        "a / Enter / d add, edit, remove a question (Questions pane)",
        "Ctrl-O        open a request file (the JSON `jevbase ask` takes)",
        "Ctrl-S        save the request to a file",
        "Ctrl-Q        quit",
        "",
        "noul   yes or no: P(yes)",
        "choice one of N named options: a probability each (3 rotations averaged)",
        "score  a level on an ordered scale of 2 to 10: the expected level",
    ];
    let bw = TEXT.iter().map(|l| l.chars().count()).max().unwrap_or(40) + 4;
    let bh = TEXT.len() + 2;
    let x = c.w.saturating_sub(bw) / 2;
    let y = c.h.saturating_sub(bh) / 2;
    c.frame(x, y, bw.min(c.w), bh.min(c.h), "Help", true);
    for (i, l) in TEXT.iter().enumerate() {
        c.put(x + 2, y + 1 + i, l, PLAIN, bw - 4);
    }
}
