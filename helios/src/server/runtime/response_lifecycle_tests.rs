/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use std::collections::{BTreeSet, VecDeque};

const CAPACITY: usize = 2;
const RESPONSES: usize = 3;
const OUTPUT_EVENTS: u64 = 2;
const MAX_DEPTH: usize = 10;
type Action = [u64; 3];

#[derive(Default)]
struct Response {
    producer: Option<Incoming>,
    reader: Option<Generation>,
    group: usize,
    sent: u64,
    received: u64,
    terminal: u64,
}

struct Responses {
    runtime: ModelRuntime,
    receiver: mpsc::Receiver<Vec<Incoming>>,
    rows: [Response; RESPONSES],
    sizes: [u64; RESPONSES],
}

impl Responses {
    fn new(fixture: &Fixture) -> Self {
        let (mut runtime, receiver) = group_runtime(fixture, CAPACITY);
        runtime.event_capacity = 2;
        Self {
            runtime,
            receiver,
            rows: std::array::from_fn(|_| Response::default()),
            sizes: [0; RESPONSES],
        }
    }

    fn prompt() -> Prompt {
        Fixture::prompt_for(
            vec![0],
            OUTPUT_EVENTS as u32,
            PromptData {
                positions: vec![[0; 3]],
                vision_token_indices: vec![-1],
                ..Default::default()
            },
        )
    }

    fn projection(&self) -> Vec<u64> {
        let mut state = vec![self.runtime.capacity.available_permits() as u64];
        state.extend(self.sizes);
        for row in &self.rows {
            let cancellation = row
                .producer
                .as_ref()
                .map(|p| &p.cancelled)
                .or_else(|| row.reader.as_ref().map(|r| &r.cancelled));
            state.extend([
                row.group as u64,
                u64::from(row.producer.is_some()),
                u64::from(row.reader.is_some()),
                row.sent,
                row.received,
                row.reader.as_ref().map_or(0, |r| r.events.len() as u64),
                row.terminal,
                u64::from(cancellation.is_some_and(|c| c.load(Ordering::Acquire))),
            ]);
        }
        state
    }

    fn assert_ownership(&self) {
        let held: u64 = self
            .sizes
            .iter()
            .enumerate()
            .filter_map(|(group, &size)| {
                self.rows
                    .iter()
                    .any(|row| {
                        row.group == group + 1 && (row.producer.is_some() || row.reader.is_some())
                    })
                    .then_some(size)
            })
            .sum();
        assert_eq!(
            self.runtime.capacity.available_permits() as u64 + held,
            CAPACITY as u64,
            "retained output lost its admission reservation"
        );
        for row in &self.rows {
            if let Some(producer) = &row.producer {
                assert_eq!(
                    producer.failure.lock().unwrap().is_some(),
                    row.terminal == 2,
                    "response queue must cover its complete output allowance"
                );
                assert_eq!(
                    producer.cancelled.load(Ordering::Acquire),
                    row.reader.is_none()
                );
            }
            if let Some(reader) = &row.reader {
                assert_eq!(row.sent, row.received + reader.events.len() as u64);
            }
        }
    }

    fn apply(&mut self, [kind, arg, value]: Action) {
        if kind == 1 || kind == 8 {
            let slots: Vec<_> = (0..RESPONSES)
                .filter(|slot| arg & (1 << slot) != 0)
                .collect();
            let result = self
                .runtime
                .generate_many(slots.iter().map(|_| Self::prompt()).collect());
            if kind == 8 {
                assert!(
                    matches!(result, Err(Error::Admission(AdmissionError::Capacity(_)))),
                    "unread responses must prevent excess admission"
                );
                return;
            }
            let readers = result.unwrap();
            let producers = self.receiver.try_recv().unwrap();
            self.sizes[value as usize - 1] = slots.len() as u64;
            for ((slot, producer), reader) in slots.into_iter().zip(producers).zip(readers) {
                self.rows[slot] = Response {
                    producer: Some(producer),
                    reader: Some(reader),
                    group: value as usize,
                    ..Default::default()
                };
            }
            return;
        }
        let row = &mut self.rows[arg as usize - 1];
        match kind {
            2 | 9 => {
                row.producer.as_ref().unwrap().emit_token(1);
                if kind == 2 {
                    row.sent += 1;
                }
            }
            3 => {
                row.producer
                    .as_ref()
                    .unwrap()
                    .emit(GenerationEvent::Finished {
                        prompt_tokens: 1,
                        generated_tokens: row.sent as u32,
                        reused_prompt_tokens: 0,
                        reason: "length",
                        stop_sequence: None,
                        termination: crate::output::Termination::OutputLimit,
                    });
                row.sent += 1;
                row.terminal = 1;
            }
            4 => {
                row.reader.as_mut().unwrap().events.try_recv().unwrap();
                row.received += 1;
            }
            5 => {
                drop(row.reader.take());
            }
            6 => {
                drop(row.producer.take());
            }
            7 | 10 => {
                let error =
                    Error::execution(ExecutionError::Worker, "response lifecycle test failure");
                row.producer.as_ref().unwrap().fail(error);
                if kind == 7 {
                    row.sent += 1;
                    row.terminal = 2;
                }
            }
            _ => unreachable!(),
        }
        if row.reader.is_none() && row.producer.is_none() {
            let group = row.group;
            *row = Response::default();
            if !self.rows.iter().any(|other| other.group == group) {
                self.sizes[group - 1] = 0;
            }
        }
    }

    fn actions(&self) -> Vec<Action> {
        let mut actions = Vec::new();
        let free = self.rows.iter().enumerate().fold(0, |mask, (slot, row)| {
            mask | if row.group == 0 { 1u64 << slot } else { 0 }
        });
        for mask in 1u64..1 << RESPONSES {
            if mask & free != mask || mask.count_ones() as usize > CAPACITY {
                continue;
            }
            if mask.count_ones() as usize <= self.runtime.capacity.available_permits() {
                let group = self.sizes.iter().position(|size| *size == 0).unwrap() + 1;
                actions.push([1, mask, group as u64]);
            } else {
                actions.push([8, mask, 0]);
            }
        }
        for (slot, row) in self.rows.iter().enumerate() {
            let id = slot as u64 + 1;
            if row.producer.is_some() {
                if row.reader.is_none() {
                    actions.extend([[9, id, 0], [6, id, 0]]);
                } else if row.terminal == 0 {
                    if row.sent < OUTPUT_EVENTS {
                        actions.push([2, id, 0]);
                    }
                    actions.extend([[3, id, 0], [7, id, 0]]);
                } else {
                    actions.push([6, id, 0]);
                    if row.terminal == 2 {
                        actions.push([10, id, 0]);
                    }
                }
            }
            if let Some(reader) = &row.reader {
                actions.push([5, id, 0]);
                if !reader.events.is_empty() {
                    actions.push([4, id, 0]);
                }
            }
        }
        actions
    }
}

#[test]
fn native_responses_match_lifecycle_model() {
    let fixture = Fixture::new();
    let mut paths = VecDeque::from([vec![]]);
    let mut visited = BTreeSet::from([Responses::new(&fixture).projection()]);
    let mut edges = BTreeSet::new();
    let mut kinds = BTreeSet::new();
    let mut shared_retention = false;
    let mut completed_unread = false;
    while let Some(path) = paths.pop_front() {
        let mut state = Responses::new(&fixture);
        for action in &path {
            state.apply(*action);
        }
        state.assert_ownership();
        if path.len() == MAX_DEPTH {
            continue;
        }
        let before = state.projection();
        for action in state.actions() {
            let mut next = Responses::new(&fixture);
            for step in &path {
                next.apply(*step);
            }
            next.apply(action);
            next.assert_ownership();
            let after = next.projection();
            kinds.insert(action[0]);
            shared_retention |= next.sizes.iter().enumerate().any(|(group, &size)| {
                size == 2
                    && next
                        .rows
                        .iter()
                        .filter(|row| row.group == group + 1)
                        .count()
                        == 1
            });
            completed_unread |= next.rows.iter().any(|row| {
                row.producer.is_none()
                    && row
                        .reader
                        .as_ref()
                        .is_some_and(|reader| !reader.events.is_empty())
            });
            edges.insert((before.clone(), action, after.clone()));
            if visited.insert(after) {
                let mut extended = path.clone();
                extended.push(action);
                paths.push_back(extended);
            }
        }
    }
    assert_eq!(kinds, (1..=10).collect());
    assert!(shared_retention && completed_unread);
    let mut recycled = Responses::new(&fixture);
    for _ in 0..3 {
        for action in [
            [1, 3, 1],
            [3, 1, 0],
            [6, 1, 0],
            [5, 1, 0],
            [5, 2, 0],
            [6, 2, 0],
        ] {
            let before = recycled.projection();
            recycled.apply(action);
            recycled.assert_ownership();
            edges.insert((before, action, recycled.projection()));
        }
        assert_eq!(recycled.runtime.capacity.available_permits(), CAPACITY);
    }
    for (before, action, after) in &edges {
        println!("RESPONSE_EDGE [{before:?},{action:?},{after:?}]");
    }
    println!(
        "RESPONSE_COVERAGE states={} edges={} depth={MAX_DEPTH}",
        visited.len(),
        edges.len()
    );
}

#[tokio::test]
async fn terminal_stream_frames_retain_http_admission_after_native_retirement() {
    use axum::{Extension, extract::State};
    use futures_util::StreamExt;
    use serde_json::json;
    for path in [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        "/v1/responses",
    ] {
        for drop_terminal in [false, true] {
            let (sender, receiver) = mpsc::sync_channel(4);
            let state = http_state(sender, None);
            let ingress = Arc::new(state.ingress_slots.clone().acquire_owned().await.unwrap());
            let mut body = json!({"model":"qwen-worker-phases", "stream":true,
            "messages":[{"role":"user", "content":"a"}], "max_tokens":2});
            if path == "/v1/completions" {
                body = json!({"model":"qwen-worker-phases", "stream":true, "prompt":"a", "max_tokens":2});
            }
            if path == "/v1/responses" {
                body = json!({"model":"qwen-worker-phases", "stream":true, "input":"a", "max_output_tokens":2});
            }
            let request = axum::http::Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap();
            let response = super::super::super::http::generate(
                State(state.clone()),
                Extension(ingress),
                request,
            )
            .await;
            assert_eq!(response.status(), axum::http::StatusCode::OK, "{path}");
            let incoming = receiver.try_recv().unwrap().remove(0);
            incoming.notify_admission(Ok(()));
            incoming.observe_token(1);
            incoming.emit_token(1);
            incoming.emit(GenerationEvent::Finished {
                prompt_tokens: incoming.prompt.tokens.len(),
                generated_tokens: 1,
                reused_prompt_tokens: 0,
                reason: "length",
                stop_sequence: None,
                termination: crate::output::Termination::OutputLimit,
            });
            drop(incoming);
            let mut reader = response.into_body().into_data_stream();
            assert_eq!(state.model.capacity.available_permits(), 3);
            while state.model.capacity.available_permits() < 4 {
                assert!(reader.next().await.unwrap().is_ok());
            }
            assert_eq!(
                state.ingress_slots.available_permits(),
                3,
                "terminal stream frames lost HTTP admission before delivery"
            );
            if !drop_terminal {
                while let Some(chunk) = reader.next().await {
                    assert!(chunk.is_ok());
                    assert_eq!(
                        state.ingress_slots.available_permits(),
                        3,
                        "terminal stream frames lost HTTP admission before delivery"
                    );
                }
                assert_eq!(
                    state.ingress_slots.available_permits(),
                    4,
                    "completed stream retained HTTP admission after EOF"
                );
            }
            drop(reader);
            assert_eq!(state.ingress_slots.available_permits(), 4);
        }
    }
}
