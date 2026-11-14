/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Payload containers, authenticated model metadata, and immutable model storage.

pub mod cache;
pub mod container;
pub mod conversion;
pub mod derivation;
pub mod export;
pub mod inverse;
pub mod naming;
pub mod progress;
pub mod prepared_weights;
pub mod protection;
pub mod reader;
pub mod schema;

/// SHA-256 of exact payload bytes, independent of container framing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PayloadId([u8; 32]);

impl PayloadId {
    pub fn of(bytes: &[u8]) -> Self {
        Self::from_digest(ring::digest::digest(&ring::digest::SHA256, bytes))
    }

    pub fn from_digest(digest: ring::digest::Digest) -> Self {
        let mut bytes = [0; 32];
        bytes.copy_from_slice(digest.as_ref());
        Self(bytes)
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn parse(text: &str) -> crate::Result<Self> {
        if text.len() != 64 {
            return Err(invalid(
                "payload identity must be 64 lowercase hexadecimal digits",
            ));
        }
        let digit = |c| match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            _ => Err(invalid("payload identity must be lowercase hexadecimal")),
        };
        let mut bytes = [0; 32];
        for (output, pair) in bytes.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
            *output = digit(pair[0])? << 4 | digit(pair[1])?;
        }
        Ok(Self(bytes))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        self.to_string()
    }

    pub fn file_name(self) -> String {
        format!("{self}.safetensors")
    }
}

impl std::fmt::Display for PayloadId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

pub(crate) fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::execution(crate::ExecutionError::Manifest, message)
}
