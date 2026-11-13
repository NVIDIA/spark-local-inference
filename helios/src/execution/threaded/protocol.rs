/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{
    Command, ExecutionManifest, Fence, FrameCompletion, Peer, Pending, Result, VecDeque, invalid,
};
use crate::execution::GraphKind;

pub(super) fn can_submit_pending(
    pending: &VecDeque<Pending>,
    barrier: bool,
    manifest: &ExecutionManifest,
) -> bool {
    pending.len() < 2
        && pending.back().is_none_or(|frame| frame.committed)
        && (!barrier || pending.is_empty())
        && !pending.iter().any(|frame| {
            frame.step.grammar
                || !matches!(
                    manifest.graphs[frame.step.graph].kind,
                    GraphKind::Decode | GraphKind::Speculative
                )
        })
}

pub(super) fn record_ready(
    pending: &mut VecDeque<Pending>,
    peer: usize,
    fence: Fence,
    graph: usize,
) -> Result<()> {
    let frame = pending
        .iter_mut()
        .find(|frame| frame.step.fence == fence)
        .ok_or_else(|| invalid("unexpected rank readiness fence"))?;
    if frame.step.graph != graph || frame.ready[peer] || frame.committed {
        return Err(invalid("rank readiness identity or state differs"));
    }
    frame.ready[peer] = true;
    Ok(())
}

pub(super) fn record_local_completion(
    pending: &mut VecDeque<Pending>,
    completion: FrameCompletion,
) -> Result<()> {
    let frame = pending
        .iter_mut()
        .find(|frame| frame.step.fence == completion.fence)
        .ok_or_else(|| invalid("unexpected rank-zero completion fence"))?;
    if frame.completions[0].replace(completion).is_some() {
        return Err(invalid("duplicate rank-zero completion"));
    }
    Ok(())
}

pub(super) fn record_peer_completion(
    pending: &mut VecDeque<Pending>,
    peer: usize,
    completion: FrameCompletion,
) -> Result<()> {
    let expected = pending
        .iter_mut()
        .find(|frame| frame.completions[peer + 1].is_none())
        .ok_or_else(|| invalid("unexpected rank completion"))?;
    if !expected.committed
        || expected.step.fence != completion.fence
        || expected.step.graph != completion.graph_index
    {
        return Err(invalid("rank completion order or identity differs"));
    }
    expected.completions[peer + 1] = Some(completion);
    Ok(())
}

pub(super) fn ready_to_commit(frame: &Pending) -> bool {
    !frame.committed && frame.ready.iter().all(|ready| *ready)
}

pub(super) fn dispatch_commit(peers: &[Peer], fence: Fence, graph: usize) -> Result<()> {
    for peer in peers {
        peer.send(Command::Commit(fence, graph))?;
    }
    Ok(())
}

pub(super) fn pop_completed_front(pending: &mut VecDeque<Pending>) -> Option<Pending> {
    if !pending
        .front()
        .is_some_and(|frame| frame.completions.iter().all(Option::is_some))
    {
        return None;
    }
    pending.pop_front()
}
