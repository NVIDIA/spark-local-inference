/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! The manager's stream and experimental write-only RDMA data channels.

use flare_selene::shared::region::Region;
use std::io;

#[cfg(feature = "rdma")]
#[path = "ibv.rs"]
pub mod ibv;

#[path = "channel.rs"]
pub mod channel;

pub fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
