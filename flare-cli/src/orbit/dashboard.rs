/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! `flare orbit dashboard`: the serving process's live metrics dashboard, drawn for a whole
//! orbit.
//!
//! One thread per node polls METRICS twice a second and reconnects when a node goes away. The
//! view keeps a [`History`] per group from the metrics documents rank zero's process serves,
//! so prefill, generation, verification and acceptance rates and their plots are the ones the
//! in-process dashboard shows, and lists every GPU of every node with its utilization, memory,
//! temperature and the group holding it. Groups and nodes appear and disappear as the agents
//! report them; the arrow keys select the group whose history is plotted, and `q` leaves.

use std::collections::BTreeMap;
use std::io::{self, IsTerminal};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use flare_helios::server::view::{
    BACKGROUND, CYAN, GOLD, GREEN, History, MUTED, PURPLE, Snapshot, TEXT, crossterm, gibibytes,
    kv_cache_lines, plot, ratatui, rate, weight_bandwidth,
};
use serde_json::{Value, json};

use crossterm::{
    cursor::{Hide, Show},
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
};

use super::controller::Controller;
use super::store::Node;
use super::{Failure, frame};

const POLL: Duration = Duration::from_millis(500);
const RECONNECT: Duration = Duration::from_secs(2);
const REFRESH: Duration = Duration::from_millis(250);

/// What one poll of a node produced.
struct Update {
    node: usize,
    result: Result<Value, String>,
}

#[derive(Clone, Default)]
struct GpuView {
    index: u64,
    name: String,
    revision: String,
    memory_total: u64,
    memory_free: u64,
    utilization: Option<u64>,
    memory_utilization: Option<u64>,
    temperature: Option<u64>,
    group: Option<String>,
}

#[derive(Default)]
struct NodeView {
    error: Option<String>,
    gpus: Vec<GpuView>,
    /// Groups the node reported in its latest poll: (id, name, rank, running, gpu indices).
    groups: Vec<(String, String, usize, bool, Vec<u64>)>,
}

/// A group's name, its members as (node index, rank, GPU indices), and whether any rank runs.
type SeenGroup = (String, Vec<(usize, usize, Vec<u64>)>, bool);

struct GroupView {
    name: String,
    /// (node index, rank, GPU indices on that node), in rank order.
    members: Vec<(usize, usize, Vec<u64>)>,
    running: bool,
    history: Option<History>,
    metrics_at: Option<u64>,
}

struct OrbitView {
    cluster: String,
    nodes: Vec<Node>,
    views: Vec<NodeView>,
    groups: BTreeMap<String, GroupView>,
    selected: usize,
    start: Instant,
}

impl OrbitView {
    fn apply(&mut self, update: Update) {
        let view = &mut self.views[update.node];
        let metrics = match update.result {
            Ok(metrics) => {
                view.error = None;
                metrics
            }
            Err(error) => {
                view.error = Some(error);
                view.groups.clear();
                self.rebuild_groups();
                return;
            }
        };
        view.gpus = metrics["gpus"]
            .as_array()
            .map(|gpus| {
                gpus.iter()
                    .map(|g| GpuView {
                        index: g["index"].as_u64().unwrap_or(0),
                        name: g["name"].as_str().unwrap_or("GPU").to_owned(),
                        revision: g["revision"].as_str().unwrap_or("").to_owned(),
                        memory_total: g["memory_total"].as_u64().unwrap_or(0),
                        memory_free: g["memory_free"].as_u64().unwrap_or(0),
                        utilization: g["utilization"].as_u64(),
                        memory_utilization: g["memory_utilization"].as_u64(),
                        temperature: g["temperature"].as_u64(),
                        group: g["group"].as_str().map(str::to_owned),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let gpu_index = |uuid: &str| {
            metrics["gpus"]
                .as_array()
                .and_then(|gpus| gpus.iter().find(|g| g["uuid"] == uuid))
                .and_then(|g| g["index"].as_u64())
        };
        view.groups.clear();
        let now = Instant::now();
        for g in metrics["groups"].as_array().into_iter().flatten() {
            let id = g["id"].as_str().unwrap_or_default().to_owned();
            let name = g["name"].as_str().unwrap_or_default().to_owned();
            let rank = g["rank"].as_u64().unwrap_or(0) as usize;
            let running = g["running"].as_bool().unwrap_or(false);
            let gpus: Vec<u64> = g["gpus"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|u| u.as_str().and_then(gpu_index))
                .collect();
            view.groups
                .push((id.clone(), name.clone(), rank, running, gpus));
            if rank == 0
                && let Some(at) = g["metrics_at"].as_u64()
            {
                let entry = self.groups.entry(id).or_insert_with(|| GroupView {
                    name,
                    members: Vec::new(),
                    running,
                    history: None,
                    metrics_at: None,
                });
                if entry.metrics_at != Some(at)
                    && let Ok(snapshot) = serde_json::from_value::<Snapshot>(g["metrics"].clone())
                {
                    match entry.history.as_mut() {
                        Some(history) => history.sample(now, snapshot),
                        None => entry.history = Some(History::new(now, snapshot)),
                    }
                    entry.metrics_at = Some(at);
                }
            }
        }
        self.rebuild_groups();
    }

    /// Recompute every group's members and state from the nodes' latest reports, dropping
    /// groups no node reports any more.
    fn rebuild_groups(&mut self) {
        let mut seen: BTreeMap<String, SeenGroup> = BTreeMap::new();
        for (node, view) in self.views.iter().enumerate() {
            for (id, name, rank, running, gpus) in &view.groups {
                let entry = seen
                    .entry(id.clone())
                    .or_insert_with(|| (name.clone(), Vec::new(), false));
                entry.1.push((node, *rank, gpus.clone()));
                entry.2 |= *running;
            }
        }
        self.groups.retain(|id, _| seen.contains_key(id));
        for (id, (name, mut members, running)) in seen {
            members.sort_by_key(|(_, rank, _)| *rank);
            let entry = self.groups.entry(id).or_insert_with(|| GroupView {
                name: name.clone(),
                members: Vec::new(),
                running,
                history: None,
                metrics_at: None,
            });
            entry.name = name;
            entry.members = members;
            entry.running = running;
        }
        if self.selected >= self.groups.len() {
            self.selected = self.groups.len().saturating_sub(1);
        }
    }

    fn selected_group(&self) -> Option<(&String, &GroupView)> {
        self.groups.iter().nth(self.selected)
    }
}

/// Restores the terminal when dropped.
struct Screen;

impl Screen {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        execute!(io::stderr(), EnterAlternateScreen, Hide)?;
        Ok(Self)
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        let _ = execute!(io::stderr(), Show, LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}

/// Poll every node of `cluster` and draw the orbit dashboard until the operator leaves.
pub fn run(controller: &Controller, cluster: Option<&str>) -> Result<(), Failure> {
    if !io::stderr().is_terminal() || std::env::var("TERM").is_ok_and(|term| term == "dumb") {
        return Err("the orbit dashboard needs a terminal on stderr".into());
    }
    let (cluster, nodes) = controller.cluster_nodes(cluster)?;
    let (sender, receiver) = mpsc::channel::<Update>();
    let mut orbit = OrbitView {
        cluster,
        views: nodes.iter().map(|_| NodeView::default()).collect(),
        nodes,
        groups: BTreeMap::new(),
        selected: 0,
        start: Instant::now(),
    };
    let stop = Arc::new(AtomicBool::new(false));
    let nodes = orbit.nodes.clone();
    thread::scope(|scope| -> Result<(), Failure> {
        for (index, node) in nodes.iter().enumerate() {
            let sender = sender.clone();
            let stop = stop.clone();
            scope.spawn(move || poll_node(controller, node, index, sender, &stop));
        }
        drop(sender);
        let result = draw(&mut orbit, &receiver);
        stop.store(true, Ordering::Relaxed);
        result
    })
}

fn poll_node(
    controller: &Controller,
    node: &Node,
    index: usize,
    sender: mpsc::Sender<Update>,
    stop: &AtomicBool,
) {
    while !stop.load(Ordering::Relaxed) {
        let mut session = match controller.node_session(node) {
            Ok((session, _)) => session,
            Err(error) => {
                if sender
                    .send(Update {
                        node: index,
                        result: Err(error.to_string()),
                    })
                    .is_err()
                {
                    return;
                }
                thread::sleep(RECONNECT);
                continue;
            }
        };
        loop {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            let result = session
                .request(frame::METRICS, &json!({}))
                .map_err(|e| e.to_string())
                .and_then(|body| serde_json::from_slice::<Value>(&body).map_err(|e| e.to_string()));
            let failed = result.is_err();
            if sender
                .send(Update {
                    node: index,
                    result,
                })
                .is_err()
            {
                return;
            }
            if failed {
                thread::sleep(RECONNECT);
                break;
            }
            thread::sleep(POLL);
        }
    }
}

fn draw(orbit: &mut OrbitView, updates: &mpsc::Receiver<Update>) -> Result<(), Failure> {
    let _screen = Screen::enter()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stderr()))?;
    loop {
        while let Ok(update) = updates.try_recv() {
            orbit.apply(update);
        }
        terminal.draw(|frame| render(frame, orbit))?;
        if event::poll(REFRESH)?
            && let Event::Key(key) = event::read()?
        {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return finish(),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return finish();
                }
                KeyCode::Left | KeyCode::Up => {
                    orbit.selected = orbit.selected.saturating_sub(1);
                }
                KeyCode::Right | KeyCode::Down if orbit.selected + 1 < orbit.groups.len() => {
                    orbit.selected += 1;
                }
                _ => {}
            }
        }
    }
}

fn finish() -> Result<(), Failure> {
    Ok(())
}

fn render(frame: &mut Frame, orbit: &OrbitView) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(BACKGROUND).fg(TEXT)),
        area,
    );
    let group_rows = (orbit.groups.len() as u16).clamp(1, 8) + 2;
    let gpu_rows = orbit
        .views
        .iter()
        .map(|v| v.gpus.len().max(1) as u16 + 1)
        .sum::<u16>()
        .clamp(2, 14);
    let [header, groups, plots, nodes, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(group_rows),
        Constraint::Fill(1),
        Constraint::Length(gpu_rows),
        Constraint::Length(1),
    ])
    .areas(area);
    render_header(frame, orbit, header);
    render_groups(frame, orbit, groups);
    render_plots(frame, orbit, plots);
    render_nodes(frame, orbit, nodes);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ↑ ↓ ", Style::default().fg(MUTED)),
            Span::styled("select group", Style::default().fg(MUTED)),
            Span::styled("   q ", Style::default().fg(MUTED)),
            Span::styled("leave", Style::default().fg(MUTED)),
        ])),
        footer,
    );
}

fn render_header(frame: &mut Frame, orbit: &OrbitView, area: Rect) {
    let running = orbit.groups.values().filter(|g| g.running).count();
    let age = orbit.start.elapsed().as_secs();
    let unreachable = orbit.views.iter().filter(|v| v.error.is_some()).count();
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " FLARE ",
                Style::default()
                    .fg(BACKGROUND)
                    .bg(CYAN)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" / ORBIT  ", Style::default().fg(CYAN)),
            Span::styled(
                format!(
                    "● {running} RUNNING  {} NODES",
                    orbit.nodes.len() - unreachable
                ),
                Style::default().fg(if running > 0 { GREEN } else { MUTED }),
            ),
            Span::styled(
                if unreachable > 0 {
                    format!("  {unreachable} UNREACHABLE")
                } else {
                    String::new()
                },
                Style::default().fg(GOLD),
            ),
            Span::styled(
                format!("    {:02}:{:02}:{:02}", age / 3600, age / 60 % 60, age % 60),
                Style::default().fg(MUTED),
            ),
            Span::styled(format!("    {}", orbit.cluster), Style::default().fg(MUTED)),
        ])),
        area,
    );
}

fn render_groups(frame: &mut Frame, orbit: &OrbitView, area: Rect) {
    let mut lines = vec![
        Line::from(format!(
            " {:<14} {:<10} {:>10} {:>12} {:>7} {:>7} {:>7} {:>10}  KV TOKENS",
            "GROUP", "RANK 0", "PREFILL", "GENERATION", "ACTIVE", "QUEUED", "ACCEPT", "WEIGHTS"
        ))
        .style(Style::default().fg(MUTED)),
    ];
    if orbit.groups.is_empty() {
        lines.push(Line::from(" no groups on this cluster").style(Style::default().fg(MUTED)));
    }
    for (index, (_, group)) in orbit.groups.iter().enumerate() {
        let rank_zero = group
            .members
            .first()
            .map_or("-", |(node, _, _)| orbit.nodes[*node].name.as_str());
        let (prefill, generation, active, queued, accept, weights, kv) = match &group.history {
            Some(history) => {
                let r = history.rates();
                let s = history.snapshot();
                (
                    rate(r.prefill),
                    rate(r.generated),
                    s.active.to_string(),
                    s.pending.to_string(),
                    r.acceptance
                        .map_or_else(|| "—".into(), |v| format!("{v:.2}")),
                    weight_bandwidth(s, r),
                    format!(
                        "{} / {}",
                        s.cached_pages.saturating_mul(s.page_tokens),
                        s.total_pages.saturating_mul(s.page_tokens)
                    ),
                )
            }
            None => (
                "—".into(),
                "—".into(),
                "—".into(),
                "—".into(),
                "—".into(),
                "—".into(),
                if group.running {
                    "loading".into()
                } else {
                    "exited".into()
                },
            ),
        };
        let marker = if index == orbit.selected { "▸" } else { " " };
        let style = if index == orbit.selected {
            Style::default().fg(CYAN).add_modifier(Modifier::BOLD)
        } else if group.running {
            Style::default().fg(TEXT)
        } else {
            Style::default().fg(MUTED)
        };
        lines.push(
            Line::from(format!(
                "{marker}{:<14} {:<10} {:>10} {:>12} {:>7} {:>7} {:>7} {:>10}  {kv}",
                group.name, rank_zero, prefill, generation, active, queued, accept, weights
            ))
            .style(style),
        );
    }
    frame.render_widget(Paragraph::new(lines), area);
}

fn render_plots(frame: &mut Frame, orbit: &OrbitView, area: Rect) {
    let Some((_, group)) = orbit.selected_group() else {
        return;
    };
    let Some(history) = &group.history else {
        frame.render_widget(
            Paragraph::new(format!(
                " {}: no metrics yet{}",
                group.name,
                if group.running {
                    "; rank zero is loading"
                } else {
                    ""
                }
            ))
            .style(Style::default().fg(MUTED)),
            area,
        );
        return;
    };
    let r = history.rates();
    let s = history.snapshot();
    let acceptance = r
        .acceptance
        .map_or_else(|| "—".into(), |v| format!("{v:.2}"));
    let depth = r.depth.map_or_else(|| "—".into(), |v| format!("{v:.2}"));
    let [kv_tokens] = kv_cache_lines(s);
    if area.width < 64 || area.height < 12 {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(format!(" {}", group.name)).style(Style::default().fg(CYAN)),
                Line::from(format!(" Prefill      {} tok/s", rate(r.prefill))),
                Line::from(format!(" Generation   {} tok/s", rate(r.generated))),
                Line::from(format!(" Verify       {} steps/s", rate(r.verify))),
                Line::from(format!(" Acceptance   {acceptance} drafts/round")),
                kv_tokens,
                Line::from(" Enlarge terminal for history plots.")
                    .style(Style::default().fg(MUTED)),
            ]),
            area,
        );
        return;
    }
    let [top, bottom, lines] = Layout::vertical([
        Constraint::Min(5),
        Constraint::Min(5),
        Constraint::Length(2),
    ])
    .areas(area);
    let [top_left, top_right] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(top);
    let [bottom_left, bottom_right] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(bottom);
    plot(
        frame,
        top_left,
        &format!(" {}  PREFILL  {} tok/s ", group.name, rate(r.prefill)),
        history,
        &[("", CYAN, |r| r.prefill)],
    );
    plot(
        frame,
        top_right,
        &format!(" GENERATION  {} tok/s ", rate(r.generated)),
        history,
        &[("", GREEN, |r| r.generated)],
    );
    plot(
        frame,
        bottom_left,
        &format!(" VERIFY  {} steps/s ", rate(r.verify)),
        history,
        &[("", PURPLE, |r| r.verify)],
    );
    plot(
        frame,
        bottom_right,
        &format!(" ACCEPTANCE {acceptance} / DEPTH {depth} "),
        history,
        &[
            ("accepted", GOLD, |r| r.acceptance.unwrap_or(0.0)),
            ("depth", CYAN, |r| r.depth.unwrap_or(0.0)),
        ],
    );
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                " ACTIVE {}   QUEUED {}   COMPLETE {}   FAILED {}   PREFILL {}   GENERATED {}   REUSED {}",
                s.active,
                s.pending,
                s.completed,
                s.failed,
                s.prefill_tokens,
                s.generated_tokens,
                s.reused_tokens
            ))
            .style(Style::default().fg(MUTED)),
            Line::from(format!(
                "{}   HOST CACHE {} hits   {} in memory / {} on disk   {} out / {} in",
                kv_tokens
                    .spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>(),
                s.host_hits,
                s.host_memory_records,
                s.host_disk_records,
                gibibytes(s.host_bytes_out),
                gibibytes(s.host_bytes_in)
            ))
            .style(Style::default().fg(MUTED)),
        ]),
        lines,
    );
}

fn render_nodes(frame: &mut Frame, orbit: &OrbitView, area: Rect) {
    let mut lines = vec![
        Line::from(format!(
            " {:<10} {:<5} {:<30} {:>5} {:>8} {:>18} {:>6}  HELD BY",
            "NODE", "GPU", "", "UTIL", "MEM BUS", "MEMORY FREE", "TEMP"
        ))
        .style(Style::default().fg(MUTED)),
    ];
    for (index, node) in orbit.nodes.iter().enumerate() {
        let view = &orbit.views[index];
        if let Some(error) = &view.error {
            lines.push(
                Line::from(format!(" {:<10} unreachable: {error}", node.name))
                    .style(Style::default().fg(GOLD)),
            );
            continue;
        }
        if view.gpus.is_empty() {
            lines.push(
                Line::from(format!(" {:<10} no GPUs reported", node.name))
                    .style(Style::default().fg(MUTED)),
            );
        }
        for (position, gpu) in view.gpus.iter().enumerate() {
            let holder = gpu
                .group
                .as_ref()
                .and_then(|id| orbit.groups.get(id))
                .map_or_else(|| gpu.group.clone().unwrap_or_default(), |g| g.name.clone());
            let style = if gpu.group.is_some() {
                Style::default().fg(TEXT)
            } else {
                Style::default().fg(MUTED)
            };
            lines.push(
                Line::from(format!(
                    " {:<10} {:<5} {:<30} {:>5} {:>8} {:>18} {:>6}  {holder}",
                    if position == 0 {
                        node.name.as_str()
                    } else {
                        ""
                    },
                    format!("gpu{}", gpu.index),
                    format!("{} ({})", gpu.name, gpu.revision)
                        .chars()
                        .take(30)
                        .collect::<String>(),
                    gpu.utilization
                        .map_or_else(|| "—".into(), |u| format!("{u}%")),
                    gpu.memory_utilization
                        .map_or_else(|| "—".into(), |u| format!("{u}%")),
                    format!(
                        "{} / {}",
                        gibibytes(gpu.memory_free),
                        gibibytes(gpu.memory_total)
                    ),
                    gpu.temperature
                        .map_or_else(|| "—".into(), |t| format!("{t}°C")),
                ))
                .style(style),
            );
        }
    }
    frame.render_widget(Paragraph::new(lines), area);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orbit(names: &[&str]) -> OrbitView {
        let nodes: Vec<Node> = names
            .iter()
            .map(|name| Node {
                name: (*name).into(),
                fingerprint: String::new(),
                public_key: String::new(),
                address: format!("{name}:29900"),
            })
            .collect();
        OrbitView {
            cluster: "test".into(),
            views: nodes.iter().map(|_| NodeView::default()).collect(),
            nodes,
            groups: BTreeMap::new(),
            selected: 0,
            start: Instant::now(),
        }
    }

    fn report(id: &str, rank: usize, running: bool, metrics: Value, at: u64) -> Value {
        json!({
            "at": 1,
            "gpus_at": 1,
            "gpus": [{"index": 0, "uuid": "GPU-a", "name": "NVIDIA GB10", "revision": "gb10",
                      "memory_total": 1024, "memory_free": 512, "utilization": 50,
                      "temperature": 40, "group": if running { Value::from(id) } else { Value::Null }}],
            "groups": [{"id": id, "name": "ds41", "rank": rank, "gpus": ["GPU-a"], "running": running,
                        "exit": null, "uptime_seconds": 3, "metrics_at": at, "metrics": metrics}],
        })
    }

    #[test]
    fn groups_follow_the_nodes_reports_and_rank_zero_metrics_feed_a_history() {
        let mut orbit = orbit(&["zero", "one"]);
        let first = json!({"generated_tokens": 100, "prefill_tokens": 10, "active": 1,
                           "total_pages": 4, "page_tokens": 128});
        orbit.apply(Update {
            node: 0,
            result: Ok(report("g-1", 0, true, first, 1000)),
        });
        orbit.apply(Update {
            node: 1,
            result: Ok(report("g-1", 1, true, Value::Null, 0)),
        });
        let group = &orbit.groups["g-1"];
        assert_eq!(group.name, "ds41");
        assert_eq!(group.members.len(), 2);
        assert_eq!(group.members[0], (0, 0, vec![0]));
        assert!(group.running);
        let history = group.history.as_ref().unwrap();
        assert_eq!(history.snapshot().generated_tokens, 100);
        assert_eq!(orbit.views[0].gpus[0].utilization, Some(50));

        // The same document again does not add a sample; a newer one does and yields a rate.
        let second = json!({"generated_tokens": 300, "prefill_tokens": 10, "active": 1,
                            "total_pages": 4, "page_tokens": 128});
        orbit.apply(Update {
            node: 0,
            result: Ok(report("g-1", 0, true, second.clone(), 1000)),
        });
        assert_eq!(
            orbit.groups["g-1"].history.as_ref().unwrap().points().len(),
            0
        );
        std::thread::sleep(Duration::from_millis(20));
        orbit.apply(Update {
            node: 0,
            result: Ok(report("g-1", 0, true, second, 1500)),
        });
        let history = orbit.groups["g-1"].history.as_ref().unwrap();
        assert_eq!(history.snapshot().generated_tokens, 300);
        assert!(history.rates().generated > 0.0);

        // A node that stops reporting the group removes it; an unreachable node is shown as such.
        orbit.apply(Update {
            node: 0,
            result: Ok(report("g-2", 0, false, Value::Null, 0)),
        });
        orbit.apply(Update {
            node: 1,
            result: Err("cannot reach".into()),
        });
        assert_eq!(orbit.groups.keys().collect::<Vec<_>>(), ["g-2"]);
        assert!(!orbit.groups["g-2"].running);
        assert_eq!(orbit.views[1].error.as_deref(), Some("cannot reach"));
        assert_eq!(orbit.selected, 0);
    }

    #[test]
    fn the_view_renders_at_small_and_large_sizes() {
        use ratatui::backend::TestBackend;
        let mut orbit = orbit(&["zero"]);
        orbit.apply(Update {
            node: 0,
            result: Ok(report(
                "g-1",
                0,
                true,
                json!({"generated_tokens": 5, "total_pages": 1, "page_tokens": 128}),
                7,
            )),
        });
        for (width, height) in [(40, 12), (120, 40)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| render(frame, &orbit)).unwrap();
            let text = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol().to_owned())
                .collect::<String>();
            assert!(text.contains("FLARE"), "{width}x{height}");
            assert!(text.contains("ds41"), "{width}x{height}");
            assert!(text.contains("gpu0"), "{width}x{height}");
        }
    }
}
