/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native CLI signal handling and Linux child-rank lifetime supervision.

use std::{
    io,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

static CLI_SIGNALS: AtomicBool = AtomicBool::new(false);
static CLI_TERMINAL: AtomicBool = AtomicBool::new(false);

extern "C" fn exit_cli(signal: libc::c_int) {
    if CLI_TERMINAL.load(Ordering::Relaxed) {
        #[cfg(feature = "server")]
        crate::server::restore_terminal_for_exit();
        // SAFETY: write and _exit are async-signal-safe; the bytes have static storage.
        unsafe {
            let reset = b"\x1b[0m\x1b[?25h\r\n";
            libc::write(libc::STDERR_FILENO, reset.as_ptr().cast(), reset.len());
        }
    }
    unsafe { libc::_exit(128 + signal) }
}

/// Process-wide immediate CLI termination, scoped away from embedded servers.
pub struct CliSignals {
    previous: [libc::sigaction; 2],
}

impl CliSignals {
    pub fn new() -> io::Result<Self> {
        if CLI_SIGNALS.swap(true, Ordering::AcqRel) {
            return Err(io::Error::other(
                "CLI signal handlers are already installed",
            ));
        }
        // SAFETY: sigaction receives initialized handlers and writable previous actions.
        unsafe {
            CLI_TERMINAL.store(libc::isatty(libc::STDERR_FILENO) == 1, Ordering::Relaxed);
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = exit_cli as *const () as usize;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaddset(&mut action.sa_mask, libc::SIGINT);
            libc::sigaddset(&mut action.sa_mask, libc::SIGTERM);
            let mut previous = [std::mem::zeroed(); 2];
            for (index, signal) in [libc::SIGINT, libc::SIGTERM].into_iter().enumerate() {
                if libc::sigaction(signal, &action, &mut previous[index]) != 0 {
                    let error = io::Error::last_os_error();
                    if index == 1 {
                        libc::sigaction(libc::SIGINT, &previous[0], std::ptr::null_mut());
                    }
                    CLI_SIGNALS.store(false, Ordering::Release);
                    return Err(error);
                }
            }
            Ok(Self { previous })
        }
    }
}

impl Drop for CliSignals {
    fn drop(&mut self) {
        for (signal, action) in [libc::SIGINT, libc::SIGTERM]
            .into_iter()
            .zip(&self.previous)
        {
            // SAFETY: The scoped CLI owner restores the actions saved during installation.
            unsafe {
                libc::sigaction(signal, action, std::ptr::null_mut());
            }
        }
        CLI_SIGNALS.store(false, Ordering::Release);
    }
}

#[derive(Default)]
struct Control {
    deadline: Option<Instant>,
    stop: bool,
}

pub struct RankSupervisor {
    peers: Arc<Vec<OwnedFd>>,
    control: Arc<Mutex<Control>>,
    thread: Option<JoinHandle<()>>,
}

fn exited(peer: &OwnedFd) -> io::Result<bool> {
    let mut event = libc::pollfd {
        fd: peer.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: poll receives one initialized, live descriptor and does not retain it.
    let result = unsafe { libc::poll(&mut event, 1, 0) };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(result != 0)
    }
}

fn status(peer: &OwnedFd) -> io::Result<libc::siginfo_t> {
    // SAFETY: waitid initializes siginfo without reaping; Python owns child reaping.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    let result = unsafe {
        libc::waitid(
            libc::P_PIDFD,
            peer.as_raw_fd() as u32,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(info)
    }
}

fn terminate(peers: &[OwnedFd]) {
    for peer in peers {
        // SAFETY: pidfds identify owned children without PID-reuse races.
        unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                peer.as_raw_fd(),
                libc::SIGKILL,
                std::ptr::null::<libc::siginfo_t>(),
                0,
            );
        }
    }
}

fn abort_group(peers: &[OwnedFd], reason: &str) -> ! {
    eprintln!("flare: terminating tensor-parallel rank group: {reason}");
    terminate(peers);
    // SAFETY: CUDA collectives may be stranded; process exit must bypass destructors.
    unsafe { libc::_exit(1) }
}

impl RankSupervisor {
    pub fn new(pids: &[i32]) -> io::Result<Self> {
        if pids.is_empty()
            || pids
                .iter()
                .enumerate()
                .any(|(i, p)| *p <= 0 || pids[..i].contains(p))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "rank PIDs must be positive and unique",
            ));
        }
        let peers: Vec<_> = pids
            .iter()
            .map(|pid| {
                // SAFETY: pidfd_open returns an owned descriptor or -1; flags are zero.
                let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, *pid, 0) };
                if fd < 0 {
                    return Err(io::Error::last_os_error());
                }
                let peer = unsafe { OwnedFd::from_raw_fd(fd as i32) };
                status(&peer)?; // Rejects processes that are not children of this process.
                Ok(peer)
            })
            .collect::<io::Result<_>>()?;
        let peers = Arc::new(peers);
        let control = Arc::new(Mutex::new(Control::default()));
        let watched = peers.clone();
        let state = control.clone();
        let thread = thread::Builder::new()
            .name("flare-ranks".into())
            .spawn(move || {
                loop {
                    {
                        let state = state.lock().unwrap_or_else(|e| e.into_inner());
                        if state.stop {
                            return;
                        }
                        if let Some(deadline) = state.deadline {
                            if Instant::now() >= deadline {
                                abort_group(&watched, "rank shutdown deadline expired");
                            }
                        } else {
                            for (rank, peer) in watched.iter().enumerate() {
                                match exited(peer) {
                                    Ok(false) => (),
                                    Ok(true) => abort_group(
                                        &watched,
                                        &format!(
                                            "peer rank {} exited before coordinated shutdown",
                                            rank + 1
                                        ),
                                    ),
                                    Err(error) => abort_group(
                                        &watched,
                                        &format!("rank monitoring failed: {error}"),
                                    ),
                                }
                            }
                        }
                    }
                    thread::sleep(Duration::from_millis(50));
                }
            })?;
        Ok(Self {
            peers,
            control,
            thread: Some(thread),
        })
    }

    pub fn begin_shutdown(&self, timeout: Duration) -> io::Result<()> {
        if timeout.is_zero() || timeout > Duration::from_secs(300) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "rank shutdown timeout must be in (0,300s]",
            ));
        }
        let mut control = self.control.lock().unwrap_or_else(|e| e.into_inner());
        if control.stop || control.deadline.is_some() {
            return Err(io::Error::other("rank shutdown has already started"));
        }
        control.deadline = Some(Instant::now() + timeout);
        Ok(())
    }

    pub fn finish_shutdown(&mut self) -> io::Result<()> {
        if self
            .control
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .deadline
            .is_none()
        {
            return Err(io::Error::other("rank shutdown has not started"));
        }
        let mut failure = None;
        for (rank, peer) in self.peers.iter().enumerate() {
            while !exited(peer)? {
                thread::sleep(Duration::from_millis(10));
            }
            let info = status(peer)?;
            // SAFETY: waitid supplies the SIGCHLD variant of siginfo.
            let code = unsafe { info.si_status() };
            if info.si_code != libc::CLD_EXITED || code != 0 {
                failure = Some(io::Error::other(format!(
                    "peer rank {} exited with code {code}",
                    rank + 1
                )));
            }
        }
        self.stop();
        failure.map_or(Ok(()), Err)
    }

    pub fn abort(&self, reason: &str) -> ! {
        abort_group(&self.peers, reason)
    }

    fn stop(&mut self) {
        self.control.lock().unwrap_or_else(|e| e.into_inner()).stop = true;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for RankSupervisor {
    fn drop(&mut self) {
        if self.thread.is_some() {
            terminate(&self.peers);
            self.stop();
        }
    }
}

pub fn bind_parent_lifetime(expected_parent: i32) -> io::Result<()> {
    // SAFETY: prctl sets this process's death signal; getppid detects a startup race.
    if unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if unsafe { libc::getppid() } != expected_parent {
        unsafe { libc::_exit(1) }
    }
    Ok(())
}
