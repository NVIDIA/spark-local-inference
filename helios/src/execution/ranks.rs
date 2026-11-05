/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Direct scheduler execution on one CUDA rank or a coordinated rank group.

use super::{
    ExecutionManifest, FrameCompletion, GraphExecutor, RankStream, TensorParallelExecutor, Upload,
};
use crate::{Result, Shutdown, cache::{Fence, PageId}};

pub enum RankPeers {
    Processes(Vec<RankStream>),
    Threads(Vec<ExecutionManifest>),
}

pub(crate) enum RankExecutor {
    Threaded(Box<super::threaded::ThreadedExecutor>),
    Local(Box<GraphExecutor>),
    Parallel(Box<TensorParallelExecutor>),
}

impl RankExecutor {
    /// # Safety
    /// Prepared graphs, contexts, buffers, and b12x IPC allocations must remain
    /// retained until coordinated drain. Peer sockets identify ranks 1..world.
    pub(crate) unsafe fn new(
        manifest: ExecutionManifest,
        peers: Option<RankPeers>,
    ) -> Result<Self> {
        match peers {
            Some(RankPeers::Threads(ranks)) => {
                unsafe { super::threaded::ThreadedExecutor::new(manifest, ranks) }
                    .map(|executor| Self::Threaded(Box::new(executor)))
            }
            Some(RankPeers::Processes(peers)) => {
                unsafe { TensorParallelExecutor::new(manifest, peers) }
                    .map(|executor| Self::Parallel(Box::new(executor)))
            }
            None => unsafe { GraphExecutor::new(manifest) }
                .map(|executor| Self::Local(Box::new(executor))),
        }
    }

    pub(crate) fn submit(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        cleared_pages: &[PageId],
        grammar: bool,
    ) -> Result<Option<Fence>> {
        match self {
            Self::Local(executor) => executor.submit(graph, uploads, cleared_pages, grammar),
            Self::Parallel(executor) => executor.submit(graph, uploads, cleared_pages, grammar),
            Self::Threaded(executor) => executor.submit(graph, uploads, cleared_pages, grammar),
        }
    }

    pub(crate) fn poll(&mut self) -> Result<Option<FrameCompletion>> {
        match self {
            Self::Local(executor) => executor.poll(),
            Self::Parallel(executor) => executor.poll(),
            Self::Threaded(executor) => executor.poll(),
        }
    }

    pub(crate) fn outstanding(&self) -> usize {
        match self {
            Self::Local(executor) => executor.outstanding(),
            Self::Parallel(executor) => executor.outstanding(),
            Self::Threaded(executor) => executor.outstanding(),
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn manifest(&self) -> &ExecutionManifest {
        match self {
            Self::Local(executor) => executor.manifest(),
            Self::Parallel(executor) => executor.manifest(),
            Self::Threaded(executor) => executor.manifest(),
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn can_submit(&self, grammar: bool) -> bool {
        match self {
            Self::Local(executor) => executor.can_submit(grammar),
            Self::Parallel(executor) => executor.can_submit(grammar),
            Self::Threaded(executor) => executor.can_submit(grammar),
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn next_fence(&self) -> Result<Fence> {
        match self {
            Self::Local(executor) => executor.next_fence(),
            Self::Parallel(executor) => executor.next_fence(),
            Self::Threaded(executor) => executor.next_fence(),
        }
    }

    pub(crate) fn close(&mut self) -> Shutdown {
        match self {
            Self::Local(executor) => executor.close(),
            Self::Parallel(executor) => executor.close(),
            Self::Threaded(executor) => executor.close(),
        }
    }
}
