/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! CPython preparation hooks and native ownership at the core boundary.

pub use helios_core::*;

mod allocator;
mod embedding_tables;
mod export;
mod lifetime;
#[cfg_attr(target_os = "windows", path = "processes_windows.rs")]
mod processes;
mod python;
