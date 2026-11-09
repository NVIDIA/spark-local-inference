/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

pub mod cache;
#[cfg(all(test, feature = "frontend"))]
#[path = "../tests/support/templates.rs"]
mod template_fixtures;
mod cuda;
#[cfg_attr(not(feature = "runtime-core"), allow(dead_code))]
pub mod engram;
pub mod error;
pub mod execution;
#[cfg(feature = "grammar")]
pub mod grammar;
pub mod model;
pub mod offload;
mod os;
#[cfg(feature = "server")]
pub mod output;
#[cfg(feature = "runtime-core")]
pub mod core;
mod replay;
#[cfg(feature = "runtime-core")]
pub mod runtime_core;
pub mod scheduler;
#[cfg(feature = "server")]
pub mod server;
pub mod startup;

#[cfg(feature = "frontend")]
pub mod frontend;

pub use cuda::current_context;
pub use replay::{BufferSpan, Completion, GraphBundle, ReplayWorker, Shutdown};

pub use error::{AdmissionError, CudaError, Error, ExecutionError, Failure, FrontendError};
pub type Result<T> = std::result::Result<T, Error>;
