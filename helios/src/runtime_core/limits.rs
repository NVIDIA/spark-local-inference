/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! User limits select physical backing, independently of the prepared graph catalog.

use std::{fmt, str::FromStr};

/// The history capacity to retain in the shared physical page pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvCacheLimit {
    #[default]
    Memory,
    Context,
    Concurrency,
}

impl FromStr for KvCacheLimit {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "memory" => Ok(Self::Memory),
            "context" => Ok(Self::Context),
            "concurrency" => Ok(Self::Concurrency),
            _ => Err("KV cache limit must be memory, context, or concurrency".into()),
        }
    }
}

impl fmt::Display for KvCacheLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Memory => "memory",
            Self::Context => "context",
            Self::Concurrency => "concurrency",
        })
    }
}

/// A per-device serving budget, or the memory to leave available at startup.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemoryLimit {
    Bytes(u64),
    Reserve(u64),
    Fraction(f64),
}

impl MemoryLimit {
    pub(super) fn validate(self) -> crate::Result<()> {
        match self {
            Self::Bytes(0) => Err("occupied memory limit must be positive".into()),
            Self::Fraction(f) if !f.is_finite() || !(0.0 < f && f <= 1.0) => {
                Err("memory fraction must be greater than zero and at most one".into())
            }
            _ => Ok(()),
        }
    }

    pub(super) fn occupied_bytes(self, total: u64) -> Option<u64> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            Self::Fraction(fraction) => Some((total as f64 * fraction).floor() as u64),
            Self::Reserve(_) => None,
        }
    }
}

impl FromStr for MemoryLimit {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        let invalid = || {
            "memory limit must be a positive size (30G), a reserve (-30G), or a fraction greater than zero and at most one (0.8)".to_owned()
        };
        let (reserve, number) = match value.strip_prefix('-') {
            Some(number) => (true, number),
            None => (false, value),
        };
        // Bare numbers are fractions. Byte sizes require a unit to avoid ambiguity at 1.
        if number.chars().any(|c| c.is_ascii_alphabetic()) {
            let bytes = crate::offload::parse_size(number).map_err(|_| invalid())?;
            if bytes == 0 && !reserve {
                return Err(invalid());
            }
            Ok(if reserve {
                Self::Reserve(bytes)
            } else {
                Self::Bytes(bytes)
            })
        } else {
            let fraction = number.parse::<f64>().map_err(|_| invalid())?;
            if reserve
                || !number.chars().all(|c| c.is_ascii_digit() || c == '.')
                || !fraction.is_finite()
                || !(0.0 < fraction && fraction <= 1.0)
            {
                return Err(invalid());
            }
            Ok(Self::Fraction(fraction))
        }
    }
}

impl fmt::Display for MemoryLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes(bytes) | Self::Reserve(bytes) => {
                if matches!(self, Self::Reserve(_)) {
                    f.write_str("-")?;
                }
                let (unit, suffix) = if *bytes >= 1 << 30 {
                    (1u64 << 30, "GiB")
                } else if *bytes >= 1 << 20 {
                    (1u64 << 20, "MiB")
                } else {
                    (1u64 << 10, "KiB")
                };
                write!(f, "{}{suffix}", *bytes as f64 / unit as f64)
            }
            Self::Fraction(fraction) => write!(f, "{fraction}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_limits_parse_and_round_trip() {
        for (text, expected) in [
            ("30g", MemoryLimit::Bytes(30 << 30)),
            ("-30g", MemoryLimit::Reserve(30 << 30)),
            ("-0KiB", MemoryLimit::Reserve(0)),
            ("1.5GiB", MemoryLimit::Bytes(3 << 29)),
            ("0.8", MemoryLimit::Fraction(0.8)),
            ("1", MemoryLimit::Fraction(1.0)),
            ("0.001K", MemoryLimit::Bytes(1)),
        ] {
            let parsed: MemoryLimit = text.parse().unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(parsed.to_string().parse::<MemoryLimit>().unwrap(), parsed);
        }
        for text in [
            "",
            "0",
            "0G",
            "-0.8",
            "1.1",
            "30",
            "NaN",
            "inf",
            "1e-1",
            "--2G",
            "2P",
            "18446744073709551616T",
        ] {
            assert!(text.parse::<MemoryLimit>().is_err(), "{text}");
        }
    }
}
