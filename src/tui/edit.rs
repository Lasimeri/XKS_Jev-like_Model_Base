//! A text field: lines of characters and a cursor, edited by keys. Shown
//! soft-wrapped to a width, scrolled so the cursor stays in view. A single
//! line field refuses newlines (Enter is left to the caller).

use super::term::Key;

#[derive(Debug, Clone)]
pub struct Field {
    lines: Vec<Vec<char>>,
    /// Cursor: line index and character index within the line.
    row: usize,
    col: usize,
    multiline: bool,
    /// First wrapped row shown.
    scroll: usize,
}

impl Field {
    pub fn new(text: &str, multiline: bool) -> Self {
        let mut f = Self {
            lines: vec![Vec::new()],
            row: 0,
            col: 0,
            multiline,
            scroll: 0,
        };
        f.set(text);
        f
    }

    /// Replace the text; the cursor goes to the end.
    pub fn set(&mut self, text: &str) {
        let text = if self.multiline {
            text.to_string()
        } else {
            text.replace('\n', " ")
        };
        self.lines = text.split('\n').map(|l| l.chars().collect()).collect();
        self.row = self.lines.len() - 1;
        self.col = self.lines[self.row].len();
        self.scroll = 0;
    }

    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|l| l.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn is_empty(&self) -> bool {
        self.lines.len() == 1 && self.lines[0].is_empty()
    }

    /// Apply an editing key; false when the key is not an edit.
    pub fn key(&mut self, k: &Key) -> bool {
        match k {
            Key::Char(c) => self.insert(*c),
            Key::Paste(s) => {
                for c in s.chars() {
                    if c == '\n' && !self.multiline {
                        self.insert(' ');
                    } else if c != '\r' {
                        self.insert(c);
                    }
                }
            }
            Key::Enter if self.multiline => self.insert('\n'),
            Key::Backspace => {
                if self.col > 0 {
                    self.col -= 1;
                    self.lines[self.row].remove(self.col);
                } else if self.row > 0 {
                    let line = self.lines.remove(self.row);
                    self.row -= 1;
                    self.col = self.lines[self.row].len();
                    self.lines[self.row].extend(line);
                }
            }
            Key::Delete => {
                if self.col < self.lines[self.row].len() {
                    self.lines[self.row].remove(self.col);
                } else if self.row + 1 < self.lines.len() {
                    let next = self.lines.remove(self.row + 1);
                    self.lines[self.row].extend(next);
                }
            }
            Key::Left => {
                if self.col > 0 {
                    self.col -= 1;
                } else if self.row > 0 {
                    self.row -= 1;
                    self.col = self.lines[self.row].len();
                }
            }
            Key::Right => {
                if self.col < self.lines[self.row].len() {
                    self.col += 1;
                } else if self.row + 1 < self.lines.len() {
                    self.row += 1;
                    self.col = 0;
                }
            }
            Key::Up if self.row > 0 => {
                self.row -= 1;
                self.col = self.col.min(self.lines[self.row].len());
            }
            Key::Down if self.row + 1 < self.lines.len() => {
                self.row += 1;
                self.col = self.col.min(self.lines[self.row].len());
            }
            Key::Home => self.col = 0,
            Key::End => self.col = self.lines[self.row].len(),
            Key::Ctrl('u') => {
                self.lines[self.row].drain(..self.col);
                self.col = 0;
            }
            _ => return false,
        }
        true
    }

    fn insert(&mut self, c: char) {
        if c == '\n' {
            let rest = self.lines[self.row].split_off(self.col);
            self.row += 1;
            self.lines.insert(self.row, rest);
            self.col = 0;
        } else if !c.is_control() {
            self.lines[self.row].insert(self.col, c);
            self.col += 1;
        }
    }

    /// The text wrapped to `width` columns, `height` rows of it from the
    /// scroll position, and the cursor's (x, y) within them. The scroll
    /// position follows the cursor.
    pub fn view(&mut self, width: usize, height: usize) -> (Vec<String>, (usize, usize)) {
        let width = width.max(1);
        let mut rows = Vec::new();
        let mut cursor = (0, 0);
        for (i, line) in self.lines.iter().enumerate() {
            let chunks = line.len().max(1).div_ceil(width);
            if i == self.row {
                // A cursor at the end of a line that fills its last row
                // exactly sits on the row after it (pushed below).
                let r = self.col / width;
                cursor = (self.col - r * width, rows.len() + r);
            }
            for k in 0..chunks {
                let end = ((k + 1) * width).min(line.len());
                rows.push(line[k * width..end].iter().collect::<String>());
            }
            if i == self.row
                && self.col == line.len()
                && !line.is_empty()
                && line.len() % width == 0
            {
                rows.push(String::new());
            }
        }
        let height = height.max(1);
        if cursor.1 < self.scroll {
            self.scroll = cursor.1;
        } else if cursor.1 >= self.scroll + height {
            self.scroll = cursor.1 + 1 - height;
        }
        let shown = rows
            .iter()
            .skip(self.scroll)
            .take(height)
            .cloned()
            .collect();
        (shown, (cursor.0, cursor.1 - self.scroll))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_and_wraps() {
        let mut f = Field::new("ab", true);
        f.key(&Key::Enter);
        f.key(&Key::Char('c'));
        assert_eq!(f.text(), "ab\nc");
        f.key(&Key::Home);
        f.key(&Key::Backspace);
        assert_eq!(f.text(), "abc");
        let mut g = Field::new("abcdef", true);
        let (rows, cursor) = g.view(4, 5);
        assert_eq!(rows, vec!["abcd", "ef"]);
        assert_eq!(cursor, (2, 1));
        let mut h = Field::new("abcd", true);
        let (rows, cursor) = h.view(4, 5);
        assert_eq!(rows, vec!["abcd", ""]);
        assert_eq!(cursor, (0, 1));
        let mut s = Field::new("one", false);
        s.key(&Key::Paste("two\nthree".into()));
        assert_eq!(s.text(), "onetwo three");
        assert!(!s.key(&Key::Enter));
    }
}
