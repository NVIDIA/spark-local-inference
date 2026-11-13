/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{Bootstrap, Server, ServerConfig, StartFailure};
use crate::{
    Result, Shutdown,
    execution::{ExecutionManifest, RankPeers},
    runtime_core::NativeModel,
};
use std::sync::Mutex;

/// A loaded, exclusively owned core and its validated serving configuration.
/// Multi-host ranks continue to use the explicit rank coordination APIs.
pub struct LoadedSession {
    manifest: ExecutionManifest,
    peers: Option<RankPeers>,
    bootstrap: Bootstrap,
    // CUDA contexts outlive all retained manifest/peer bindings, including
    // registered host storage when a loaded session is dropped before start.
    owner: NativeModel,
}

impl LoadedSession {
    pub fn new(owner: NativeModel, config: serde_json::Value) -> Result<Self> {
        if owner.cluster().is_some() {
            return Err("an owned local session requires every model rank in this process".into());
        }
        let mut manifests = owner.manifests()?.into_iter();
        let manifest = manifests.next().ok_or("loaded core has no rank")?;
        let remaining: Vec<_> = manifests.collect();
        let peers = (!remaining.is_empty()).then_some(RankPeers::Threads(remaining));
        let mut bootstrap = Bootstrap::load(&config.to_string(), &manifest)?;
        bootstrap.server.log_to_stderr = false;
        Ok(Self {
            manifest,
            peers,
            bootstrap,
            owner,
        })
    }

    pub fn server_config_mut(&mut self) -> &mut ServerConfig {
        &mut self.bootstrap.server
    }

    pub fn start(self) -> std::result::Result<ServingSession, StartFailure> {
        let Self {
            owner,
            manifest,
            peers,
            bootstrap,
        } = self;
        // Assume an unconfirmed drain before entering code that may launch GPU work.
        // Unwinding also retains the owner rather than freeing referenced storage.
        let mut owner = DrainGuard::new(owner);
        // SAFETY: The session owns the loaded model and grants its worker exclusive
        // execution. The guard retains the resources when drain is unconfirmed.
        let server = unsafe {
            Server::start(
                manifest,
                bootstrap.model,
                bootstrap.frontend,
                bootstrap.images,
                bootstrap.server,
                peers,
            )
        };
        match server {
            Ok(server) => Ok(ServingSession {
                server,
                owner: Mutex::new(owner),
            }),
            Err(failure) => {
                owner.drained = failure.drained;
                Err(failure)
            }
        }
    }
}

/// Retains the core through server startup, request execution and GPU drain.
pub struct ServingSession {
    server: Server,
    owner: Mutex<DrainGuard<NativeModel>>,
}

impl ServingSession {
    pub fn server(&self) -> &Server {
        &self.server
    }

    /// Joins HTTP and model workers. Drop the session to release a drained model;
    /// an unconfirmed drain retains its resources until process exit.
    pub fn close(&self) -> Shutdown {
        let shutdown = self.server.close();
        let mut owner = self.owner.lock().unwrap_or_else(|e| e.into_inner());
        owner.drained = shutdown.drained;
        shutdown
    }
}

impl Drop for ServingSession {
    fn drop(&mut self) {
        self.close();
    }
}

/// Releasing native storage requires positive drain confirmation, including on unwind.
struct DrainGuard<T> {
    value: Option<T>,
    drained: bool,
}

impl<T> DrainGuard<T> {
    fn new(value: T) -> Self {
        Self {
            value: Some(value),
            drained: false,
        }
    }
}

impl<T> Drop for DrainGuard<T> {
    fn drop(&mut self) {
        if !self.drained {
            if let Some(value) = self.value.take() {
                std::mem::forget(value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct Resource(Arc<AtomicUsize>);
    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn resource_owner_requires_positive_drain_confirmation() {
        for drained in [false, true] {
            let drops = Arc::new(AtomicUsize::new(0));
            let mut owner = DrainGuard::new(Resource(drops.clone()));
            owner.drained = drained;
            drop(owner);
            assert_eq!(drops.load(Ordering::SeqCst), usize::from(drained));
        }
    }

    #[test]
    fn unwinding_before_drain_retains_resources() {
        let drops = Arc::new(AtomicUsize::new(0));
        let observed = drops.clone();
        let _ = std::panic::catch_unwind(move || {
            let _owner = DrainGuard::new(Resource(observed));
            panic!("startup panicked before drain confirmation");
        });
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }
}
