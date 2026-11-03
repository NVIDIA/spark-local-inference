/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native rank coordination over process-local CUDA executors and control sockets.

#[cfg(test)]
mod tests;
mod wire;

use std::{time::Instant, collections::VecDeque};

use super::{ExecutionManifest, FrameCompletion, GraphExecutor, OwnedUpload, RankStream, Upload};
use crate::{Error, Result, Shutdown, cache::Fence};
use wire::*;

struct Peer {
    channel: Channel,
}

struct Pending {
    fence: Fence,
    graph: usize,
    grammar: bool,
    uploads: Vec<OwnedUpload>,
    dispatched: Instant,
    committed: bool,
    queued: bool,
    ready: Vec<bool>,
    completions: Vec<Option<FrameCompletion>>,
}
impl Pending {
    fn uploads(&self) -> Vec<Upload<'_>> {
        self.uploads.iter().map(|upload| Upload {
            buffer: &upload.buffer, offset: upload.offset, bytes: &upload.bytes,
        }).collect()
    }
}
fn ready_frame(pending: &mut VecDeque<Pending>, rank: usize, fence: Fence, graph: usize) -> Result<()> {
    let frame = pending.iter_mut().find(|frame| frame.fence == fence)
        .ok_or_else(|| invalid("unexpected rank readiness fence"))?;
    if frame.graph != graph || frame.ready[rank] || frame.committed {
        return Err(invalid("rank readiness identity or state differs"));
    }
    frame.ready[rank] = true;
    Ok(())
}
fn completion_frame(pending: &mut VecDeque<Pending>, rank: usize, fence: Fence, graph: usize) -> Result<&mut Pending> {
    let frame = pending.iter_mut().find(|frame| frame.completions[rank + 1].is_none())
        .ok_or_else(|| invalid("unexpected rank completion"))?;
    if !frame.committed || frame.fence != fence || frame.graph != graph {
        return Err(invalid("rank completion order or identity differs"));
    }
    Ok(frame)
}
fn window_available(pending: &VecDeque<Pending>, grammar: bool, manifest: &ExecutionManifest) -> bool {
    pending.len() < 2
        && pending.back().is_none_or(|frame| frame.committed)
        && (!grammar || pending.is_empty())
        && !pending.iter().any(|frame| frame.grammar || !matches!(
            manifest.graphs[frame.graph].kind, super::GraphKind::Decode | super::GraphKind::Speculative))
}

fn identity(fence: Fence, graph: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    number(&mut bytes, fence.0 as usize);
    number(&mut bytes, graph);
    bytes
}
fn read_identity(bytes: &[u8]) -> Result<(Fence, usize)> {
    let mut reader = Reader(bytes);
    let value = (Fence(reader.number()? as u64), reader.number()?);
    reader.end()?;
    Ok(value)
}

/// Rank zero owns request scheduling; peer processes execute identical graph steps.
/// Two frame slots include preparation and unconsumed all-rank completions.
pub struct TensorParallelExecutor {
    local: GraphExecutor,
    peers: Vec<Peer>,
    pending: VecDeque<Pending>,
    poison: Option<Error>,
    shutdown: Option<Shutdown>,
}

impl TensorParallelExecutor {
    /// # Safety
    /// GraphExecutor's ownership contract applies. Each socket must connect to a
    /// distinct prepared rank in order 1..world; every owner includes its b12x IPC
    /// runtime and remains retained until coordinated shutdown confirms drain.
    pub unsafe fn new(manifest: ExecutionManifest, sockets: Vec<RankStream>) -> Result<Self> {
        let world = sockets.len() + 1;
        if world > 16 {
            return Err(invalid("tensor parallelism supports at most sixteen ranks"));
        }
        if world < 2 {
            return Err(invalid("tensor parallelism requires at least two ranks"));
        }
        manifest.validate()?;
        wire::validate_completion_packet(&manifest)?;
        let expected = layout(&manifest);
        let mut peers = Vec::new();
        for (index, socket) in sockets.into_iter().enumerate() {
            let mut channel = Channel::new(socket)?;
            let hello = channel.receive(HELLO)?;
            let mut reader = Reader(&hello);
            if reader.number()? != index + 1 || reader.number()? != world || reader.0 != expected {
                return Err(invalid(
                    "tensor-parallel rank identity, graph I/O schema, or cache ownership layout differs",
                ));
            }
            peers.push(Peer {
                channel,
            });
        }
        // SAFETY: The caller retains all rank-local prepared resources through drain.
        let mut local = unsafe { GraphExecutor::new_unstarted(manifest)? };
        let tuning_started = Instant::now();
        let startup = (|| -> Result<()> {
            for index in local.manifest().referenced_tuning_graphs() {
                let mut bytes = Vec::new();
                number(&mut bytes, index);
                for (sent, peer) in peers.iter_mut().enumerate() {
                    if let Err(error) = peer.channel.send(TUNE, &bytes) {
                        // A failed socket write may have delivered the complete command.
                        eprintln!("startup rank {sent} dispatch is unconfirmed: {error}; terminating the CUDA process");
                        std::process::abort();
                    }
                }
                let mut failure = local.run_tuning_graph(index).err();
                for peer in &mut peers {
                    match peer.channel.receive(TUNED) {
                        Ok(reply) if reply.starts_with(&bytes) => {
                            if reply.len() != bytes.len() {
                                failure.get_or_insert_with(|| invalid(String::from_utf8_lossy(&reply[bytes.len()..]).into_owned()));
                            }
                        }
                        reply => {
                            let message = reply.err().map_or_else(|| "rank tuning completion identity differs".to_string(), |error| error.to_string());
                            eprintln!("startup rank completion is unconfirmed: {message}; terminating the CUDA process");
                            std::process::abort();
                        }
                    }
                }
                if let Some(error) = failure { return Err(error); }
            }
            for peer in &mut peers { peer.channel.send(READY, &[])?; }
            local.finish_startup()?;
            for peer in &mut peers {
                if !peer.channel.receive(STARTED)?.is_empty() {
                    return Err(invalid("rank startup completion payload is not empty"));
                }
            }
            Ok(())
        })();
        if let Err(error) = startup {
            // Tuning completion is confirmed on every rank before ordinary errors.
            let sent: Vec<_> = peers.iter_mut().map(|peer| peer.channel.send(DRAIN, &[]).is_ok()).collect();
            for (peer, sent) in peers.iter_mut().zip(sent) {
                if sent {
                    loop {
                        match peer.channel.wait(false) {
                            Ok((STARTED, bytes)) if bytes.is_empty() => continue,
                            Ok((DRAINED, bytes)) if bytes.is_empty() => {
                                let _ = peer.channel.send(RELEASE, &[]);
                                break;
                            }
                            _ => break,
                        }
                    }
                }
                let _ = peer.channel.stream.shutdown(std::net::Shutdown::Both);
            }
            if !local.close().drained {
                eprintln!("startup coordinator cleanup did not drain; terminating the CUDA process");
                std::process::abort();
            }
            return Err(error);
        }
        local.report_startup(tuning_started);
        Ok(Self {
            local,
            peers,
            pending: VecDeque::with_capacity(2),
            poison: None,
            shutdown: None,
        })
    }

    pub fn outstanding(&self) -> usize {
        self.pending.len()
    }

    pub fn manifest(&self) -> &ExecutionManifest {
        self.local.manifest()
    }

    pub fn can_submit(&self, grammar: bool) -> bool {
        self.poison.is_none()
            && self.shutdown.is_none()
            && window_available(&self.pending, grammar, self.local.manifest())
            && self.local.can_submit(grammar)
    }

    pub fn next_fence(&self) -> Result<Fence> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        if self.shutdown.is_some() {
            return Err(invalid("tensor-parallel executor is closed"));
        }
        match self.pending.back() {
            Some(frame) if !frame.committed => Ok(Fence(frame.fence.0 + 1)),
            _ => self.local.next_fence(),
        }
    }

    pub fn submit(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        grammar: bool,
    ) -> Result<Option<Fence>> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        if self.shutdown.is_some() {
            return Err(invalid("tensor-parallel executor is closed"));
        }
        let barrier = grammar || !matches!(self.local.manifest().graphs.get(graph).map(|g| g.kind),
            Some(super::GraphKind::Decode | super::GraphKind::Speculative));
        if !self.can_submit(barrier) { return Ok(None); }
        let Some(fence) = self.local.validate_submission(graph, uploads, grammar)? else {
            return Ok(None);
        };
        let queued = !self.pending.is_empty();
        self.pending.push_back(Pending {
            fence, graph, grammar, dispatched: Instant::now(), committed: false, queued,
            uploads: uploads.iter().map(|u| OwnedUpload {
                buffer: u.buffer.to_owned(), offset: u.offset, bytes: u.bytes.to_vec(),
            }).collect(),
            ready: vec![false; self.peers.len()],
            completions: (0..=self.peers.len()).map(|_| None).collect(),
        });
        let result = wire::prepare(self.local.manifest(), fence, graph, grammar, uploads, |tag, bytes| {
            for peer in &mut self.peers { peer.channel.send(tag, bytes)?; }
            Ok(())
        });
        if let Err(error) = result {
            self.poison = Some(error.clone());
            return Err(error);
        }
        Ok(Some(fence))
    }

    pub fn poll(&mut self) -> Result<Option<FrameCompletion>> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        if self.shutdown.is_some() {
            return Err(invalid("tensor-parallel executor is closed"));
        }
        match self.poll_ranks() {
            Ok(None) => Ok(None),
            Ok(Some(completion)) => Ok(Some(completion)),
            Err(error) => {
                self.poison = Some(error.clone());
                Err(error)
            }
        }
    }

    fn poll_ranks(&mut self) -> Result<Option<FrameCompletion>> {
        if self.pending.iter().any(|frame| frame.dispatched.elapsed() > TIMEOUT) {
            return Err(invalid("tensor-parallel graph completion deadline expired"));
        }
        if let Some(completion) = self.local.poll()? {
            let frame = self.pending.iter_mut().find(|frame| frame.fence == completion.fence)
                .ok_or_else(|| invalid("unexpected rank-zero completion fence"))?;
            if frame.completions[0].replace(completion).is_some() {
                return Err(invalid("duplicate rank-zero completion"));
            }
        }
        for (index, peer) in self.peers.iter_mut().enumerate() {
            peer.channel.nonblocking(true)?;
            let result = (|| -> Result<()> {
                while let Some((tag, bytes)) = peer.channel.poll()? {
                    if tag == READY {
                        let (fence, graph) = read_identity(&bytes)?;
                        ready_frame(&mut self.pending, index, fence, graph)?;
                    } else if tag == COMPLETE {
                        let mut reader = Reader(&bytes);
                        let fence = Fence(reader.number()? as u64);
                        let graph = reader.number()?;
                        let frame = completion_frame(&mut self.pending, index, fence, graph)?;
                        let timing = wire::timing(&mut reader)?;
                        let proposal_confidence = wire::proposal_confidence(&mut reader, self.local.manifest().request_capacity(), self.local.manifest().completion_token_capacity)?;
                        let speculative_survival = wire::speculative_survival(&mut reader, self.local.manifest().request_capacity(), self.local.manifest().completion_token_capacity)?;
                        if reader.0.len() != self.local.manifest().completion_bytes() {
                            return Err(invalid("rank completion size differs from its ABI"));
                        }
                        frame.completions[index + 1] = Some(FrameCompletion {
                            logprobs: None, prompt_logprobs: None, proposal_confidence, speculative_survival,
                            fence, graph_index: graph, timing,
                            records: super::transport::decode_records(reader.0, fence,
                                self.local.manifest().request_capacity(), self.local.manifest().completion_fields())?,
                        });
                    } else { return Err(invalid("unexpected rank message")); }
                }
                Ok(())
            })();
            peer.channel.nonblocking(false)?;
            result?;
        }
        for frame in &mut self.pending {
            if frame.committed { continue; }
            if !frame.ready.iter().all(|ready| *ready) { break; }
            let uploads = frame.uploads();
            if self.local.validate_submission(frame.graph, &uploads, frame.grammar)? != Some(frame.fence) {
                return Err(invalid("rank-zero reservation changed before commit"));
            }
            for peer in &mut self.peers {
                peer.channel.send(COMMIT, &identity(frame.fence, frame.graph))?;
            }
            if self.local.submit(frame.graph, &uploads, frame.grammar)? != Some(frame.fence) {
                return Err(invalid("rank-zero submission changed after coordinated validation"));
            }
            frame.committed = true;
        }
        if !self.pending.front().is_some_and(|frame| frame.completions.iter().all(Option::is_some)) {
            return Ok(None);
        }
        let mut frame = self.pending.pop_front().unwrap();
        let mut root = frame.completions[0].take().unwrap();
        for (index, completion) in frame.completions.into_iter().skip(1).enumerate() {
            let completion = completion.unwrap();
            let agreement = if self.local.manifest().graphs[root.graph_index].kind == super::GraphKind::Speculative {
                agree_speculative
            } else { agree };
            agreement(&root, &completion).map_err(|error| invalid(format!("rank {}: {error}", index + 1)))?;
            root.timing.merge_rank(index + 1, completion.timing)?;
        }
        root.timing.queued = frame.queued;
        root.timing.wall_ns = frame.dispatched.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        Ok(Some(root))
    }

    pub fn close(&mut self) -> Shutdown {
        if let Some(shutdown) = &self.shutdown {
            return shutdown.clone();
        }
        let result = (|| -> Result<()> {
            while !self.pending.is_empty() {
                self.poll()?;
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
            if let Some(error) = &self.poison {
                return Err(error.clone());
            }
            for peer in &mut self.peers {
                peer.channel.nonblocking(false)?;
                peer.channel.send(DRAIN, &[])?;
            }
            for peer in &mut self.peers {
                if !peer.channel.receive(DRAINED)?.is_empty() {
                    return Err(invalid("unexpected rank drain payload"));
                }
            }
            Ok(())
        })();
        let shutdown = match result {
            Err(error) => self.local.abandon(error),
            Ok(()) => self.local.close(),
        };
        if shutdown.drained {
            // All GPU streams have drained. A lost release can retain a peer's
            // owner, but cannot make any rank free resources still used by CUDA.
            for peer in &mut self.peers {
                let _ = peer.channel.send(RELEASE, &[]);
            }
        }
        for peer in &self.peers {
            let _ = peer.channel.stream.shutdown(std::net::Shutdown::Both);
        }
        self.shutdown = Some(shutdown.clone());
        shutdown
    }
}

impl Drop for TensorParallelExecutor {
    fn drop(&mut self) {
        self.close();
    }
}

pub(super) fn agree(root: &FrameCompletion, peer: &FrameCompletion) -> Result<()> {
    if root.fence != peer.fence || root.graph_index != peer.graph_index {
        return Err(invalid("rank completion identity differs"));
    }
    if root.records.len() != peer.records.len() {
        return Err(invalid("rank completion record counts differ"));
    }
    match (&root.proposal_confidence, &peer.proposal_confidence) {
        (None, None) => {}
        (Some(root_values), Some(peer_values)) => {
            for (values, records) in [(root_values, &root.records), (peer_values, &peer.records)] {
                if values.len() != records.len() || values.iter().zip(records.iter()).any(|(row, record)|
                    record.vector.as_ref().is_none_or(|vector| row.len() + 1 != vector.tokens.len())) {
                    return Err(invalid("proposal confidence geometry differs from completion capacity"));
                }
            }
            // Only the root completion is returned to the shared scheduler.
            // Peer estimates do not select execution or enter the root's curve;
            // their floating-point agreement is not a control-plane invariant.
            // Validate every reported probability and keep exact request-state
            // agreement below, without requiring replicated arithmetic identity.
            for (row, (a, b)) in root_values.iter().zip(peer_values).enumerate() {
                for (draft, (&a, &b)) in a.iter().zip(b).enumerate() {
                    if !(0.0..=1.0).contains(&a) || !(0.0..=1.0).contains(&b) {
                        return Err(invalid(format!(
                            "tensor-parallel proposal confidence is invalid at row {row}, draft {draft}: root {a:?}, peer {b:?}; graph {}, fence {}",
                            root.graph_index, root.fence.0,
                        )));
                    }
                }
            }
        }
        _ => return Err(invalid("tensor-parallel proposal confidence presence differs")),
    }
    match (&root.speculative_survival, &peer.speculative_survival) {
        (None, None) => {}
        (Some(root_values), Some(peer_values)) => {
            super::validate_speculative_survival(root_values, &root.records)?;
            super::validate_speculative_survival(peer_values, &peer.records)?;
            if root_values.iter().flatten().zip(peer_values.iter().flatten())
                .any(|(a, b)| (*a == -1.0) != (*b == -1.0)) {
                return Err(invalid("tensor-parallel speculative survival observation extents differ"));
            }
        }
        _ => return Err(invalid("tensor-parallel speculative survival presence differs")),
    }
    for (lane, (a, b)) in root.records.iter().zip(peer.records.iter()).enumerate() {
        if a.generation != b.generation
            || a.submission_sequence != b.submission_sequence
            || a.consumed_tokens != b.consumed_tokens
            || a.error_code != b.error_code
            || !(a.status == b.status || a.status == 2 && b.status == 1)
        {
            return Err(invalid(format!(
                "tensor-parallel request epochs, consumption, or status differ at graph {}, fence {}, lane {lane}: root {a:?}, peer {b:?}",
                root.graph_index, root.fence.0,
            )));
        }
    }
    Ok(())
}

pub(super) fn agree_speculative(root: &FrameCompletion, peer: &FrameCompletion) -> Result<()> {
    agree(root, peer)?;
    for (a, b) in root.records.iter().zip(peer.records.iter()) {
        // Inactive token/vector payload is undefined. Retain the base
        // transport agreement above, and compare only committed active values.
        if a.status == 0 && b.status == 0 { continue; }
        if a.generation != b.generation || a.submission_sequence != b.submission_sequence
            || a.consumed_tokens != b.consumed_tokens || a.generated_tokens != b.generated_tokens
            || a.token_id != b.token_id || a.status != b.status || a.error_code != b.error_code
            || a.reserved != b.reserved || a.vector != b.vector {
            return Err(invalid("speculative rank completion prefixes or committed state differ"));
        }
    }
    Ok(())
}

fn startup_message(channel: &mut Channel, executor: &mut GraphExecutor, expected: u8) -> Result<Option<Vec<u8>>> {
    let (tag, bytes) = channel.wait(false)?;
    if tag == DRAIN && bytes.is_empty() {
        let shutdown = executor.close();
        if !shutdown.drained {
            return Err(shutdown.error.unwrap_or_else(|| invalid("startup rank did not drain")));
        }
        channel.send(DRAINED, &[])?;
        if !channel.receive(RELEASE)?.is_empty() {
            return Err(invalid("unexpected startup release payload"));
        }
        return Ok(None);
    }
    if tag != expected { return Err(invalid("unexpected startup rank command")); }
    Ok(Some(bytes))
}

/// Run a peer's native control loop until coordinated drain and release.
///
/// # Safety
/// The caller retains all graph owners and b12x IPC mappings when drained is
/// false. A failed collective requires coordinated process termination before
/// reclaiming those resources; this function never synchronizes a stranded graph.
pub unsafe fn serve_rank(
    manifest: ExecutionManifest,
    socket: RankStream,
    rank: usize,
    world: usize,
) -> Shutdown {
    if let Err(error) = manifest.validate().and_then(|_| wire::validate_completion_packet(&manifest)) {
        return Shutdown { drained: true, error: Some(error) };
    }
    let mut executor = match unsafe { GraphExecutor::new_unstarted(manifest) } {
        Ok(executor) => executor,
        Err(error) => {
            return Shutdown {
                drained: true,
                error: Some(error),
            };
        }
    };
    executor.copy_logprobs = false;
    let result = (|| -> Result<()> {
        if world < 2 || !(1..world).contains(&rank) {
            return Err(invalid("invalid tensor-parallel peer identity"));
        }
        let mut channel = Channel::new(socket)?;
        let result = (|| -> Result<()> {
            let mut hello = Vec::new();
            number(&mut hello, rank);
            number(&mut hello, world);
            hello.extend(layout(executor.manifest()));
            channel.send(HELLO, &hello)?;
            for index in executor.manifest().referenced_tuning_graphs() {
                let Some(bytes) = startup_message(&mut channel, &mut executor, TUNE)? else { return Ok(()); };
                let mut reader = Reader(&bytes);
                if reader.number()? != index { return Err(invalid("rank startup tuning order differs")); }
                reader.end()?;
                let mut reply = bytes;
                if let Err(error) = executor.run_tuning_graph(index) {
                    reply.extend(error.to_string().as_bytes());
                }
                channel.send(TUNED, &reply)?;
            }
            let Some(ready) = startup_message(&mut channel, &mut executor, READY)? else { return Ok(()); };
            if !ready.is_empty() {
                return Err(invalid("unexpected coordinator readiness payload"));
            }
            executor.finish_startup()?;
            channel.send(STARTED, &[])?;
            let mut prepared: Option<(Fence, usize, bool, Vec<OwnedUpload>)> = None;
            let mut prepared_at: Option<Instant> = None;
            let mut started = VecDeque::new();
            loop {
                let active = executor.outstanding() != 0 || prepared.is_some();
                channel.nonblocking(active)?;
                let message = if active { channel.poll()? } else { Some(channel.wait(true)?) };
                channel.nonblocking(false)?;
                if let Some((tag, bytes)) = message {
                    match tag {
                        DRAIN if bytes.is_empty() => {
                            if active { return Err(invalid("rank drain arrived with outstanding frames")); }
                            let shutdown = executor.close();
                            if !shutdown.drained {
                                return Err(shutdown.error.unwrap_or_else(|| invalid("rank did not drain")));
                            }
                            channel.send(DRAINED, &[])?;
                            if !channel.receive(RELEASE)?.is_empty() {
                                return Err(invalid("unexpected rank release payload"));
                            }
                            return Ok(());
                        }
                        PREPARE | PREPARE_MORE => {
                            if prepared.is_some() { return Err(invalid("rank already has a prepared frame")); }
                            let (fence, graph, grammar, owned) =
                                receive_prepare(&mut channel, executor.manifest(), tag, bytes)?;
                            let uploads: Vec<_> = owned.iter().map(|upload| Upload {
                                buffer: &upload.buffer, offset: upload.offset, bytes: &upload.bytes,
                            }).collect();
                            if executor.validate_submission(graph, &uploads, grammar)? != Some(fence) {
                                return Err(invalid("rank submission sequence or capacity differs"));
                            }
                            channel.send(READY, &identity(fence, graph))?;
                            prepared = Some((fence, graph, grammar, owned));
                            prepared_at = Some(Instant::now());
                        }
                        COMMIT => {
                            let (fence, graph) = read_identity(&bytes)?;
                            prepared_at = None;
                            let (expected, expected_graph, grammar, owned) = prepared.take()
                                .ok_or_else(|| invalid("rank commit arrived without preparation"))?;
                            if fence != expected || graph != expected_graph {
                                return Err(invalid("rank commit identity differs"));
                            }
                            let uploads: Vec<_> = owned.iter().map(|upload| Upload {
                                buffer: &upload.buffer, offset: upload.offset, bytes: &upload.bytes,
                            }).collect();
                            if executor.submit(graph, &uploads, grammar)? != Some(fence) {
                                return Err(invalid("rank lost its prepared submission"));
                            }
                            started.push_back(Instant::now());
                        }
                        _ => return Err(invalid("unexpected rank command")),
                    }
                }
                if let Some(completed) = executor.poll()? {
                    started.pop_front();
                    channel.send(COMPLETE, &completion(&completed))?;
                }
                if prepared_at.is_some_and(|time| time.elapsed() > TIMEOUT) {
                    return Err(invalid("rank commit deadline expired after preparation"));
                }
                if started.front().is_some_and(|time| time.elapsed() > TIMEOUT) {
                    return Err(invalid("rank graph completion deadline expired"));
                }
                if active { std::thread::yield_now(); }
            }
        })();
        if let Err(error) = &result {
            let _ = channel.nonblocking(false);
            let _ = channel.send(FAILED, error.to_string().as_bytes());
        }
        result
    })();
    match result {
        Ok(()) => executor.close(),
        Err(error) => executor.abandon(error),
    }
}
