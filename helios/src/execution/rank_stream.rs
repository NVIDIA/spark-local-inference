/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Connected rank control sockets: Unix sockets within a host, TCP between hosts.

use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
#[cfg(unix)]
use std::os::fd::OwnedFd;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::time::Duration;

/// One coordinator-to-peer control connection.
///
/// Peers on the coordinator's host use a Unix socket pair. Peers on other hosts
/// connect over TCP; every step exchanges a few small messages, so TCP sends
/// without Nagle coalescing.
pub enum RankStream {
    #[cfg(unix)]
    Unix(UnixStream),
    Tcp(TcpStream),
}

impl RankStream {
    /// Adopt a connected stream socket, choosing its transport from the socket's family.
    #[cfg(unix)]
    pub fn from_socket(socket: OwnedFd) -> io::Result<Self> {
        use std::os::fd::AsRawFd;

        // SAFETY: getsockname writes at most `length` bytes into zeroed storage
        // large enough for every address family and does not retain the descriptor.
        let family = unsafe {
            let mut address: libc::sockaddr_storage = std::mem::zeroed();
            let mut length = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
            if libc::getsockname(
                socket.as_raw_fd(),
                (&mut address as *mut libc::sockaddr_storage).cast(),
                &mut length,
            ) != 0
            {
                return Err(io::Error::last_os_error());
            }
            address.ss_family as libc::c_int
        };
        let stream = match family {
            libc::AF_UNIX => Self::Unix(UnixStream::from(socket)),
            libc::AF_INET | libc::AF_INET6 => Self::Tcp(TcpStream::from(socket)),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "rank control sockets must be Unix or TCP stream sockets",
                ));
            }
        };
        stream.connected()?;
        if let Self::Tcp(stream) = &stream {
            stream.set_nodelay(true)?;
        }
        Ok(stream)
    }

    #[cfg(unix)]
    fn connected(&self) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Self::Unix(stream) => stream.peer_addr().map(drop),
            Self::Tcp(stream) => stream.peer_addr().map(drop),
        }
    }

    pub fn set_nonblocking(&self, enabled: bool) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Self::Unix(stream) => stream.set_nonblocking(enabled),
            Self::Tcp(stream) => stream.set_nonblocking(enabled),
        }
    }

    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Self::Unix(stream) => stream.set_read_timeout(timeout),
            Self::Tcp(stream) => stream.set_read_timeout(timeout),
        }
    }

    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Self::Unix(stream) => stream.set_write_timeout(timeout),
            Self::Tcp(stream) => stream.set_write_timeout(timeout),
        }
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Self::Unix(stream) => stream.shutdown(how),
            Self::Tcp(stream) => stream.shutdown(how),
        }
    }
}

#[cfg(unix)]
impl From<UnixStream> for RankStream {
    fn from(stream: UnixStream) -> Self {
        Self::Unix(stream)
    }
}

impl From<TcpStream> for RankStream {
    fn from(stream: TcpStream) -> Self {
        Self::Tcp(stream)
    }
}

impl Read for RankStream {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        match self {
            #[cfg(unix)]
            Self::Unix(stream) => stream.read(buffer),
            Self::Tcp(stream) => stream.read(buffer),
        }
    }
}

impl Write for RankStream {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        match self {
            #[cfg(unix)]
            Self::Unix(stream) => stream.write(buffer),
            Self::Tcp(stream) => stream.write(buffer),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Self::Unix(stream) => stream.flush(),
            Self::Tcp(stream) => stream.flush(),
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn adopts_unix_and_tcp_sockets_by_family() {
        let (left, _right) = UnixStream::pair().unwrap();
        assert!(matches!(
            RankStream::from_socket(OwnedFd::from(left)).unwrap(),
            RankStream::Unix(_)
        ));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (_server, _) = listener.accept().unwrap();
        match RankStream::from_socket(OwnedFd::from(client)).unwrap() {
            RankStream::Tcp(stream) => assert!(stream.nodelay().unwrap()),
            RankStream::Unix(_) => panic!("a TCP socket was adopted as a Unix socket"),
        }
    }

    #[test]
    fn rejects_unconnected_sockets() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        assert!(RankStream::from_socket(OwnedFd::from(listener)).is_err());
    }
}

#[cfg(test)]
pub(super) fn test_pair() -> std::io::Result<(RankStream, RankStream)> {
    #[cfg(unix)]
    {
        let (a, b) = UnixStream::pair()?;
        Ok((a.into(), b.into()))
    }
    #[cfg(windows)]
    {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let a = TcpStream::connect(listener.local_addr()?)?;
        let (b, _) = listener.accept()?;
        a.set_nodelay(true)?;
        b.set_nodelay(true)?;
        Ok((a.into(), b.into()))
    }
}
