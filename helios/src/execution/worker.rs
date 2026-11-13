/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use std::sync::{Mutex, mpsc};
use std::thread::{self, JoinHandle};

use super::{ExecutionManifest, FrameCompletion, RankExecutor, RankStream, Upload};
use crate::{Error, ExecutionError, Failure, Result, Shutdown, cache::Fence};

pub struct OwnedUpload {
    pub buffer: String,
    pub offset: usize,
    pub bytes: Vec<u8>,
}

enum Command {
    Submit {
        graph: usize,
        uploads: Vec<OwnedUpload>,
        grammar: bool,
        reply: mpsc::SyncSender<Result<Option<Fence>>>,
    },
    Poll(mpsc::SyncSender<Result<Option<FrameCompletion>>>),
    Outstanding(mpsc::SyncSender<usize>),
}

struct Control {
    sender: Option<mpsc::SyncSender<Command>>,
    thread: Option<JoinHandle<Shutdown>>,
    shutdown: Option<Shutdown>,
}

/// Synchronous control surface for bootstrap/qualification callers on other threads.
/// The native scheduler owns its rank executor directly without this channel boundary.
pub struct ExecutionWorker {
    control: Mutex<Control>,
}

impl ExecutionWorker {
    /// # Safety
    /// The owner must satisfy GraphExecutor::new's resource lifetime contract.
    pub unsafe fn new(manifest: ExecutionManifest) -> Result<Self> {
        unsafe { Self::start(manifest, None) }
    }

    /// # Safety
    /// TensorParallelExecutor::new's rank and resource lifetime contract applies.
    pub unsafe fn new_parallel(
        manifest: ExecutionManifest,
        peers: Vec<RankStream>,
    ) -> Result<Self> {
        unsafe { Self::start(manifest, Some(super::RankPeers::Processes(peers))) }
    }

    /// # Safety
    /// All rank owners must remain alive until coordinated execution drains.
    pub unsafe fn new_threaded(manifest: ExecutionManifest, ranks: Vec<ExecutionManifest>) -> Result<Self> {
        unsafe { Self::start(manifest, Some(super::RankPeers::Threads(ranks))) }
    }

    unsafe fn start(manifest: ExecutionManifest, peers: Option<super::RankPeers>) -> Result<Self> {
        manifest.validate()?;
        let (sender, receiver) = mpsc::sync_channel(1);
        let (ready_sender, ready) = mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("flare-execution".into())
            .spawn(move || {
                // SAFETY: The caller retains the manifest's graphs and allocations
                // until this worker reports drain. The executor binds its streams
                // to the declared context on this thread and has exclusive access.
                let opened =
                    unsafe { RankExecutor::new(manifest, peers) };
                let mut executor = match opened {
                    Ok(executor) => executor,
                    Err(error) => {
                        let _ = ready_sender.send(Err(error.clone()));
                        return Shutdown {
                            drained: true,
                            error: Some(error),
                        };
                    }
                };
                let _ = ready_sender.send(Ok(()));
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    while let Ok(command) = receiver.recv() {
                        match command {
                            Command::Submit {
                                graph,
                                uploads,
                                grammar,
                                reply,
                            } => {
                                let borrowed: Vec<_> = uploads
                                    .iter()
                                    .map(|upload| Upload {
                                        buffer: &upload.buffer,
                                        offset: upload.offset,
                                        bytes: &upload.bytes,
                                    })
                                    .collect();
                                let _ = reply.send(executor.submit(graph, &borrowed, grammar));
                            }
                            Command::Poll(reply) => {
                                let _ = reply.send(executor.poll());
                            }
                            Command::Outstanding(reply) => {
                                let _ = reply.send(executor.outstanding());
                            }
                        }
                    }
                }));
                let mut shutdown = executor.close();
                if result.is_err() {
                    shutdown.error.get_or_insert_with(|| {
                        Error::execution(ExecutionError::Worker, "native execution worker panicked")
                    });
                }
                shutdown
            })
            .map_err(|error| {
                ExecutionError::Worker(Failure::caused_by(
                    format!("cannot start execution worker: {error}"),
                    error,
                ))
            })?;
        match ready.recv() {
            Ok(Ok(())) => {}
            result => {
                let _ = thread.join();
                return Err(match result {
                    Ok(Err(error)) => error,
                    Err(error) => ExecutionError::Worker(Failure::caused_by(
                        "execution worker exited during startup",
                        error,
                    ))
                    .into(),
                    Ok(Ok(())) => unreachable!("successful startup was handled above"),
                });
            }
        }
        Ok(Self {
            control: Mutex::new(Control {
                sender: Some(sender),
                thread: Some(thread),
                shutdown: None,
            }),
        })
    }

    fn request<T>(&self, command: impl FnOnce(mpsc::SyncSender<T>) -> Command) -> Result<T> {
        let control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let sender = control.sender.as_ref().ok_or_else(|| {
            Error::execution(ExecutionError::Closed, "execution worker is closed")
        })?;
        let (reply, receiver) = mpsc::sync_channel(1);
        sender.send(command(reply)).map_err(|error| {
            ExecutionError::Worker(Failure::caused_by("execution worker disconnected", error))
        })?;
        receiver.recv().map_err(|error| {
            ExecutionError::Worker(Failure::caused_by(
                "execution worker exited without a response",
                error,
            ))
            .into()
        })
    }

    pub fn submit(
        &self,
        graph: usize,
        uploads: Vec<OwnedUpload>,
        grammar: bool,
    ) -> Result<Option<Fence>> {
        self.request(|reply| Command::Submit {
            graph,
            uploads,
            grammar,
            reply,
        })?
    }
    pub fn poll(&self) -> Result<Option<FrameCompletion>> {
        self.request(Command::Poll)?
    }
    pub fn outstanding(&self) -> Result<usize> {
        self.request(Command::Outstanding)
    }
    pub fn close(&self) -> Shutdown {
        let mut control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(shutdown) = &control.shutdown {
            return shutdown.clone();
        }
        control.sender.take();
        let shutdown = match control.thread.take().unwrap().join() {
            Ok(shutdown) => shutdown,
            Err(_) => Shutdown {
                drained: false,
                error: Some(Error::execution(
                    ExecutionError::Drain,
                    "execution worker panicked before reporting drain",
                )),
            },
        };
        control.shutdown = Some(shutdown.clone());
        shutdown
    }
}

impl Drop for ExecutionWorker {
    fn drop(&mut self) {
        self.close();
    }
}
