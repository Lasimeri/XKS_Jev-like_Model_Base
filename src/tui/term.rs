//! The terminal, with no crates: raw mode through termios (declared here for
//! Linux x86-64 glibc, its layout checked at compile time), the window size
//! through TIOCGWINSZ, keys decoded from the byte stream (bracketed paste
//! included), and the alternate screen. Dropping `Term` restores everything,
//! and a panic restores the terminal before its message prints.

use std::io::{Read, Write};

#[repr(C)]
#[derive(Clone, Copy)]
struct Termios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [u8; 32],
    c_ispeed: u32,
    c_ospeed: u32,
}

// glibc's struct termios on x86-64 is 60 bytes, c_cc at offset 17.
const _: () = assert!(std::mem::size_of::<Termios>() == 60);
const _: () = assert!(std::mem::offset_of!(Termios, c_cc) == 17);

#[repr(C)]
struct Winsize {
    rows: u16,
    cols: u16,
    xpixel: u16,
    ypixel: u16,
}

unsafe extern "C" {
    fn tcgetattr(fd: i32, t: *mut Termios) -> i32;
    fn tcsetattr(fd: i32, action: i32, t: *const Termios) -> i32;
    fn ioctl(fd: i32, request: u64, ...) -> i32;
}

const TCSANOW: i32 = 0;
const TIOCGWINSZ: u64 = 0x5413;
// c_iflag
const BRKINT: u32 = 0o2;
const INPCK: u32 = 0o20;
const ISTRIP: u32 = 0o40;
const ICRNL: u32 = 0o400;
const IXON: u32 = 0o2000;
// c_oflag
const OPOST: u32 = 0o1;
// c_cflag
const CS8: u32 = 0o60;
// c_lflag
const ISIG: u32 = 0o1;
const ICANON: u32 = 0o2;
const ECHO: u32 = 0o10;
const IEXTEN: u32 = 0o100000;
// c_cc
const VTIME: usize = 5;
const VMIN: usize = 6;

/// Enter the alternate screen, hide the cursor, bracketed paste on.
const ENTER: &str = "\x1b[?1049h\x1b[?25l\x1b[?2004h";
/// Undo `ENTER`.
const LEAVE: &str = "\x1b[?2004l\x1b[?25h\x1b[?1049l";

/// A key, or text pasted in one piece.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Key {
    Char(char),
    /// Ctrl + a letter, as the lowercase letter.
    Ctrl(char),
    Enter,
    Tab,
    BackTab,
    Backspace,
    Delete,
    Esc,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    F(u8),
    Paste(String),
}

pub struct Term {
    saved: Termios,
    pending: Vec<u8>,
}

impl Term {
    /// Raw mode on standard input, the alternate screen on standard output.
    /// Reads return after at most 100 ms so the caller can poll other work.
    pub fn open() -> Result<Self, String> {
        let mut t = Termios {
            c_iflag: 0,
            c_oflag: 0,
            c_cflag: 0,
            c_lflag: 0,
            c_line: 0,
            c_cc: [0; 32],
            c_ispeed: 0,
            c_ospeed: 0,
        };
        // SAFETY: `t` is a correctly laid out termios (checked above).
        if unsafe { tcgetattr(0, &mut t) } != 0 {
            return Err("standard input is not a terminal (the TUI needs one)".into());
        }
        let saved = t;
        t.c_iflag &= !(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
        t.c_oflag &= !OPOST;
        t.c_cflag |= CS8;
        t.c_lflag &= !(ECHO | ICANON | IEXTEN | ISIG);
        t.c_cc[VMIN] = 0;
        t.c_cc[VTIME] = 1;
        // SAFETY: as above.
        if unsafe { tcsetattr(0, TCSANOW, &t) } != 0 {
            return Err("could not put the terminal in raw mode".into());
        }
        let restore = saved;
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // SAFETY: restores the terminal settings read at open.
            unsafe { tcsetattr(0, TCSANOW, &restore) };
            print!("{LEAVE}");
            let _ = std::io::stdout().flush();
            hook(info);
        }));
        print!("{ENTER}");
        let _ = std::io::stdout().flush();
        Ok(Self {
            saved,
            pending: Vec::new(),
        })
    }

    /// (columns, rows); 80x24 when the size cannot be read.
    pub fn size() -> (usize, usize) {
        let mut w = Winsize {
            rows: 0,
            cols: 0,
            xpixel: 0,
            ypixel: 0,
        };
        // SAFETY: TIOCGWINSZ writes one winsize.
        let ok = unsafe { ioctl(1, TIOCGWINSZ, &mut w as *mut Winsize) } == 0;
        if ok && w.cols > 0 && w.rows > 0 {
            (w.cols as usize, w.rows as usize)
        } else {
            (80, 24)
        }
    }

    /// The next key, or None when nothing arrived within 100 ms.
    pub fn key(&mut self) -> Option<Key> {
        if self.pending.is_empty() {
            let mut buf = [0u8; 4096];
            let n = std::io::stdin().read(&mut buf).unwrap_or(0);
            self.pending.extend_from_slice(&buf[..n]);
        }
        let (key, used) = decode(&self.pending)?;
        self.pending.drain(..used);
        Some(key)
    }
}

impl Drop for Term {
    fn drop(&mut self) {
        print!("{LEAVE}");
        let _ = std::io::stdout().flush();
        // SAFETY: restores the settings read at open.
        unsafe { tcsetattr(0, TCSANOW, &self.saved) };
        let _ = std::panic::take_hook();
    }
}

/// Decode one key from the front of `b`: the key and the bytes it used.
/// An incomplete paste or UTF-8 sequence waits for more bytes (None).
pub fn decode(b: &[u8]) -> Option<(Key, usize)> {
    let first = *b.first()?;
    if first == 0x1b {
        if b.len() == 1 {
            return Some((Key::Esc, 1));
        }
        if b.starts_with(b"\x1b[200~") {
            let end = b"\x1b[201~";
            let at = b.windows(end.len()).position(|w| w == end)?;
            let text = String::from_utf8_lossy(&b[6..at])
                .replace("\r\n", "\n")
                .replace('\r', "\n");
            return Some((Key::Paste(text), at + end.len()));
        }
        if b[1] == b'[' || b[1] == b'O' {
            // CSI / SS3: parameters, then one final byte in 0x40..=0x7e.
            let end = b[2..].iter().position(|&c| (0x40..=0x7e).contains(&c))? + 2;
            let params = &b[2..end];
            let key = match (b[end], params) {
                (b'A', _) => Key::Up,
                (b'B', _) => Key::Down,
                (b'C', _) => Key::Right,
                (b'D', _) => Key::Left,
                (b'H', _) => Key::Home,
                (b'F', _) => Key::End,
                (b'Z', _) => Key::BackTab,
                (b'P', _) => Key::F(1),
                (b'Q', _) => Key::F(2),
                (b'R', _) => Key::F(3),
                (b'S', _) => Key::F(4),
                (b'~', b"1" | b"7") => Key::Home,
                (b'~', b"4" | b"8") => Key::End,
                (b'~', b"3") => Key::Delete,
                (b'~', b"5") => Key::PageUp,
                (b'~', b"6") => Key::PageDown,
                (b'~', b"15") => Key::F(5),
                (b'~', b"17") => Key::F(6),
                (b'~', b"18") => Key::F(7),
                (b'~', b"19") => Key::F(8),
                (b'~', b"20") => Key::F(9),
                (b'~', b"21") => Key::F(10),
                _ => Key::Esc,
            };
            return Some((key, end + 1));
        }
        return Some((Key::Esc, 1));
    }
    let key = match first {
        b'\r' | b'\n' => Key::Enter,
        b'\t' => Key::Tab,
        0x7f | 0x08 => Key::Backspace,
        1..=26 => Key::Ctrl((b'a' + first - 1) as char),
        _ => {
            let len = match first {
                0xc0..=0xdf => 2,
                0xe0..=0xef => 3,
                0xf0..=0xf7 => 4,
                _ => 1,
            };
            if b.len() < len {
                return None;
            }
            let c = std::str::from_utf8(&b[..len])
                .ok()
                .and_then(|s| s.chars().next())
                .unwrap_or('\u{fffd}');
            return Some((Key::Char(c), len));
        }
    };
    Some((key, 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_decode() {
        assert_eq!(decode(b"a"), Some((Key::Char('a'), 1)));
        assert_eq!(decode("é".as_bytes()), Some((Key::Char('é'), 2)));
        assert_eq!(decode(b"\x1b[A"), Some((Key::Up, 3)));
        assert_eq!(decode(b"\x1b[3~x"), Some((Key::Delete, 4)));
        assert_eq!(decode(b"\x1b[15~"), Some((Key::F(5), 5)));
        assert_eq!(decode(b"\x12"), Some((Key::Ctrl('r'), 1)));
        assert_eq!(decode(b"\r"), Some((Key::Enter, 1)));
        assert_eq!(
            decode(b"\x1b[200~two\r\nlines\x1b[201~"),
            Some((Key::Paste("two\nlines".into()), 22))
        );
        assert_eq!(decode(b"\x1b[200~unfinished"), None);
        assert_eq!(decode(b"\x1b"), Some((Key::Esc, 1)));
    }
}
