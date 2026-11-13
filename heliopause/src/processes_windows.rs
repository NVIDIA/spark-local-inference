/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scoped console termination for the model preparation CLI.

use std::{
    io::{self, IsTerminal, Write},
    sync::atomic::{AtomicBool, Ordering},
};
use windows_sys::Win32::System::{Console::*, Threading::ExitProcess};

static CLI_SIGNALS: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn exit_cli(event: u32) -> i32 {
    if !matches!(event, CTRL_C_EVENT | CTRL_BREAK_EVENT) {
        return 0;
    }
    if io::stderr().is_terminal() {
        #[cfg(feature = "server")]
        crate::server::restore_terminal_for_exit();
        let _ = io::stderr().write_all(b"\x1b[0m\x1b[?25h\r\n");
    }
    // Preparation owns the CUDA context; process exit bypasses stranded GPU work.
    unsafe { ExitProcess(130) }
}

pub struct CliSignals;

impl CliSignals {
    pub fn new() -> io::Result<Self> {
        if CLI_SIGNALS.swap(true, Ordering::AcqRel) {
            return Err(io::Error::other(
                "CLI signal handlers are already installed",
            ));
        }
        if unsafe { SetConsoleCtrlHandler(Some(exit_cli), 1) } == 0 {
            CLI_SIGNALS.store(false, Ordering::Release);
            return Err(io::Error::last_os_error());
        }
        Ok(Self)
    }
}

impl Drop for CliSignals {
    fn drop(&mut self) {
        unsafe { SetConsoleCtrlHandler(Some(exit_cli), 0) };
        CLI_SIGNALS.store(false, Ordering::Release);
    }
}
