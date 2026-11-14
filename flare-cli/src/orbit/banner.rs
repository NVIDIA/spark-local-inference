/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! The orbit banner: the wordmark above Flare's sun, with a body circling it. `flare orbit run`
//! animates one revolution while it reads the nodes and places the group, then leaves the
//! final frame above the placement.
//!
//! The body travels the back of its orbit above the wordmark from right to left, dim, and the
//! front of its orbit between the wordmark and the sun from left to right, bright, with a short
//! fading trail. Every frame has the same number of lines, so the animation redraws in place.

use std::fmt::Write as _;
use std::io::{self, IsTerminal, Write};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// One revolution of the body.
pub const REVOLUTION: Duration = Duration::from_secs(3);
const FRAMES: usize = 48;
const WIDTH: usize = 72;

const WORDMARK: [&str; 6] = [
    "      ██████╗ ██████╗ ██████╗ ██╗████████╗",
    "     ██╔═══██╗██╔══██╗██╔══██╗██║╚══██╔══╝",
    "     ██║   ██║██████╔╝██████╔╝██║   ██║",
    "     ██║   ██║██╔══██╗██╔══██╗██║   ██║",
    "     ╚██████╔╝██║  ██║██████╔╝██║   ██║",
    "      ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝   ╚═╝",
];
const SUN: [&str; 4] = [
    "                  ⣀⣀⣀⣀⣠⠤⢤⠤⢤⡤⠤⣶⡒⢒⣒⠒⢒⡒⠤⢤⡤⠤⢤⠤⡤⡤⠤⣤⠤⠤⠤⣄⣀⣀⣀⣀",
    "      ⢀⣀⣀⡤⢤⢴⢲⠒⡞⡽⡫⣉⢗⢦⢠⠒⡄⡴⡢⢀⢖⢆⡔⡢⢀⢖⢆⠰⡺⡍⡝⢫⢳⢖⢖⡔⡢⢀⢖⡲⡄⡔⣢⢠⢖⢄⠰⡺⡍⡝⢫⢳⢖⢖⠦⡤⢤⣀⣀⡀",
    "⡠⢤⠔⡒⢊⡭⣭⢚⡒⢿⠶⡀⠥⡞⢓⠺⡷⣿⠖⡋⢠⠭⡄⢉⡀⣿⢖⡀⠥⡞⢓⠺⡷⢌⠁⣨⢥⡜⣓⡀⠥⡾⢒⠚⣦⡀⠥⡞⠥⡞⢓⠺⡷⢌⠁⣨⢥⡜⣓⠺⡷⢎⠁⡨⢥⡉⣛⢵⡶⢢⠤⢄",
    "⣊⣸⠮⠞⠈⠪⠈⠂⠵⠃⠮⠼⠈⣊⣼⣮⡚⠈⠈⠂⠵⠃⠑⢥⠔ ⠣⢵⣇⣓⣁⠧⠵⠁⢓⣵⠈⠂⠵⠃⠵⠃⣓⠈⠂⠵⠃⠑⠥⣇⣑⠈⠂⠵⠃⠑⢥⣧⣑⡃⠳⠥⠇⢑⣡⣷⣕⠃⠱⠥⢇⣑",
];
/// The sun's row colours, warm at the limb and red at the edge, as the serving banner paints it.
const SUN_COLORS: [u32; 4] = [0xffbf69, 0xffe8b0, 0xff9c42, 0xe54b32];
const WORDMARK_COLOR: u32 = 0xfff0c7;
const ORBIT_COLOR: u32 = 0x4a5a6e;
const BODY_COLOR: u32 = 0xfff8e7;
const BACK_BODY_COLOR: u32 = 0x9fb0c4;
const TRAIL_COLORS: [u32; 3] = [0xffd27b, 0xc7954f, 0x7a6244];
/// The orbit's horizontal extent and the row of each half.
const ORBIT_LEFT: usize = 3;
const ORBIT_RIGHT: usize = 68;
const BACK_ROW: usize = 1;
const FRONT_ROW: usize = 8;
/// Lines in a frame: a blank row, the back orbit, the wordmark, the front orbit, the sun.
pub const LINES: usize = 1 + 1 + 6 + 1 + 4;

/// Where the body is on frame `frame`: (row, column, in front).
fn body(frame: usize) -> (usize, usize, bool) {
    let span = (ORBIT_RIGHT - ORBIT_LEFT) as f64;
    let angle = frame as f64 / FRAMES as f64 * std::f64::consts::TAU;
    // The front half runs left to right while the angle sweeps 0..π.
    let column = ORBIT_LEFT as f64 + span * (1.0 - angle.cos()) / 2.0;
    let front = angle.sin() >= 0.0;
    (
        if front { FRONT_ROW } else { BACK_ROW },
        column.round() as usize,
        front,
    )
}

fn rgb(output: &mut String, color: u32, bold: bool, text: &str) {
    let _ = write!(
        output,
        "\x1b[{};38;2;{};{};{}m{text}",
        if bold { 1 } else { 22 },
        color >> 16,
        (color >> 8) & 255,
        color & 255
    );
}

/// One frame, `LINES` lines of at most `width` columns, each cleared before it is drawn.
pub fn frame(frame: usize, width: usize) -> String {
    let frame = frame % FRAMES;
    let width = width.min(WIDTH);
    let (row, column, front) = body(frame);
    let trail: Vec<(usize, usize, u32)> = (1..=3)
        .map(|back| {
            let (r, c, _) = body((frame + FRAMES - back) % FRAMES);
            (r, c, TRAIL_COLORS[back - 1])
        })
        .filter(|&(r, _, _)| r == row)
        .collect();
    let mut output = String::new();
    for line in 0..LINES {
        output.push_str("\r\x1b[2K");
        match line {
            0 => {}
            BACK_ROW | FRONT_ROW => {
                for c in 0..width {
                    if c == column && (line == row) {
                        rgb(
                            &mut output,
                            if front { BODY_COLOR } else { BACK_BODY_COLOR },
                            front,
                            if front { "●" } else { "•" },
                        );
                    } else if let Some(&(_, _, color)) =
                        trail.iter().find(|&&(r, tc, _)| r == line && tc == c)
                    {
                        rgb(&mut output, color, false, "·");
                    } else if (ORBIT_LEFT..=ORBIT_RIGHT).contains(&c) {
                        let arc = if c == ORBIT_LEFT || c == ORBIT_RIGHT {
                            if line == FRONT_ROW { "╯" } else { "╮" }
                        } else {
                            "─"
                        };
                        let arc = if c == ORBIT_LEFT {
                            if line == FRONT_ROW { "╰" } else { "╭" }
                        } else {
                            arc
                        };
                        rgb(&mut output, ORBIT_COLOR, false, arc);
                    } else {
                        output.push(' ');
                    }
                }
            }
            2..=7 => {
                let text: String = WORDMARK[line - 2].chars().take(width).collect();
                rgb(&mut output, WORDMARK_COLOR, true, &text);
            }
            _ => {
                let text: String = SUN[line - 9].chars().take(width).collect();
                rgb(&mut output, SUN_COLORS[line - 9], false, &text);
            }
        }
        output.push_str("\x1b[0m\n");
    }
    output
}

/// The animation on stderr while the controller works; dropping it, or calling `finish`, leaves
/// the final frame on screen.
pub struct Banner {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

fn terminal_width() -> usize {
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    // SAFETY: TIOCGWINSZ fills a winsize for a terminal descriptor; failure leaves zeros.
    if unsafe { libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size) } == 0
        && size.ws_col > 0
    {
        size.ws_col as usize
    } else {
        WIDTH
    }
}

impl Banner {
    /// Start the animation when stderr is a terminal; otherwise print nothing.
    pub fn start() -> Option<Self> {
        if !io::stderr().is_terminal() || std::env::var("TERM").is_ok_and(|t| t == "dumb") {
            return None;
        }
        let width = terminal_width();
        let (stop, receiver) = mpsc::channel::<()>();
        let started = Instant::now();
        let thread = thread::spawn(move || {
            let mut out = io::stderr().lock();
            let _ = out.write_all(b"\x1b[?25l");
            let mut index = 0usize;
            let mut stopping = false;
            loop {
                let _ = out.write_all(frame(index, width).as_bytes());
                let _ = out.flush();
                let next = started + REVOLUTION / FRAMES as u32 * (index as u32 + 1);
                let wait = next.saturating_duration_since(Instant::now());
                // Once told to stop, frames keep their pace until the revolution completes.
                if stopping {
                    thread::sleep(wait);
                } else {
                    stopping = !matches!(
                        receiver.recv_timeout(wait),
                        Err(mpsc::RecvTimeoutError::Timeout)
                    );
                }
                if stopping && started.elapsed() >= REVOLUTION {
                    break;
                }
                index += 1;
                let _ = write!(out, "\x1b[{LINES}A");
            }
            let _ = out.write_all(b"\x1b[?25h");
            let _ = out.flush();
        });
        Some(Self {
            stop: Some(stop),
            thread: Some(thread),
        })
    }

    /// Stop after the revolution in progress completes and leave the last frame on screen.
    pub fn finish(mut self) {
        self.end();
    }

    fn end(&mut self) {
        drop(self.stop.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Banner {
    fn drop(&mut self) {
        self.end();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_frame_has_the_same_lines_and_the_body_circles_the_sun() {
        let mut front_columns = Vec::new();
        for index in 0..FRAMES {
            let text = frame(index, 72);
            assert_eq!(text.matches('\n').count(), LINES, "frame {index}");
            assert!(text.contains("██████╗"));
            let (row, column, front) = body(index);
            assert!((ORBIT_LEFT..=ORBIT_RIGHT).contains(&column));
            assert_eq!(row, if front { FRONT_ROW } else { BACK_ROW });
            if front {
                front_columns.push(column);
            }
        }
        // The front half crosses the whole orbit from left to right.
        assert_eq!(front_columns.first(), Some(&ORBIT_LEFT));
        assert!(front_columns.windows(2).all(|w| w[1] >= w[0]));
        assert_eq!(front_columns.last(), Some(&ORBIT_RIGHT));
        // Narrow terminals get truncated lines, never more.
        for line in frame(7, 40).lines() {
            let visible: String = line
                .split("\x1b[")
                .skip(1)
                .map(|s| s.split_once('m').map_or("", |(_, t)| t))
                .collect();
            assert!(visible.chars().count() <= 40, "{visible}");
        }
    }
}
