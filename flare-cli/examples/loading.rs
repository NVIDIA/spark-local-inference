/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Preview loading output with simulated stages; no model or GPU is used.

use std::{thread, time::Duration};

use flare_helios::startup::StartupProgress;

fn main() {
    let mut progress = StartupProgress::stderr();
    let phase = progress.phase("Prepare model", None);
    thread::sleep(Duration::from_millis(200));
    phase.finish("preview");
    let phase = progress.phase("Initialize GPU", None);
    thread::sleep(Duration::from_millis(300));
    phase.finish("simulated");
    let phase = progress.phase("Load weights", Some(32 << 30));
    for _ in 0..32 {
        thread::sleep(Duration::from_millis(100));
        phase.advance_bytes(1 << 30);
    }
    phase.finish("32.0 GiB · simulated");
    progress.finish_animation();
}
