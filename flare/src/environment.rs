/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashSet;

#[cfg(unix)]
pub fn home_directory() -> std::io::Result<std::path::PathBuf> {
    let mut storage = vec![0u8; 65536];
    let mut record = std::mem::MaybeUninit::<libc::passwd>::uninit();
    let mut result = std::ptr::null_mut();
    // SAFETY: getpwuid_r writes into the supplied record and backing storage.
    let status = unsafe {
        libc::getpwuid_r(
            libc::getuid(),
            record.as_mut_ptr(),
            storage.as_mut_ptr().cast(),
            storage.len(),
            &mut result,
        )
    };
    if status != 0 {
        return Err(std::io::Error::from_raw_os_error(status));
    }
    if result.is_null() {
        return Err(std::io::Error::other(
            "cannot resolve home directory; specify --cache-dir",
        ));
    }
    // SAFETY: The passwd field points into storage, which remains live until copied.
    let home = unsafe { std::ffi::CStr::from_ptr((*result).pw_dir) };
    use std::os::unix::ffi::OsStrExt;
    Ok(std::path::PathBuf::from(std::ffi::OsStr::from_bytes(
        home.to_bytes(),
    )))
}

#[cfg(windows)]
pub use crate::windows::home_directory;

pub fn visibility(devices: &[i32], inherited: Option<&str>) -> Result<String, String> {
    if devices.is_empty()
        || devices.iter().any(|&device| device < 0)
        || devices.iter().collect::<HashSet<_>>().len() != devices.len()
    {
        return Err("--gpus requires distinct nonnegative device indices".into());
    }
    let Some(inherited) = inherited else {
        return Ok(devices
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(","));
    };
    let visible: Vec<_> = inherited.split(',').map(str::trim).collect();
    for device in &visible {
        let index = device.parse::<u32>().is_ok();
        let uuid = device.strip_prefix("GPU-").is_some_and(|suffix| {
            !suffix.is_empty() && suffix.bytes().all(|c| c.is_ascii_hexdigit() || c == b'-')
        });
        if !index && !uuid {
            return Err("CUDA_VISIBLE_DEVICES must list device indices or GPU UUIDs".into());
        }
    }
    if visible.iter().collect::<HashSet<_>>().len() != visible.len() {
        return Err("CUDA_VISIBLE_DEVICES contains duplicate devices".into());
    }
    devices
        .iter()
        .map(|&device| {
            visible.get(device as usize).copied().ok_or_else(|| {
                format!("--gpus {device} is outside CUDA_VISIBLE_DEVICES={inherited}")
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|selected| selected.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_preserves_rank_order_and_inherited_visibility() {
        assert_eq!(visibility(&[4, 7], None).unwrap(), "4,7");
        assert_eq!(visibility(&[2, 0], Some("3,4,7")).unwrap(), "7,3");
        assert_eq!(visibility(&[1], Some("4,7")).unwrap(), "7");
        assert_eq!(
            visibility(&[1, 0], Some("GPU-abcdef01,GPU-12345678")).unwrap(),
            "GPU-12345678,GPU-abcdef01"
        );
    }

    #[test]
    fn invalid_selections_do_not_expand_visibility() {
        for devices in [vec![], vec![-1], vec![4, 4]] {
            assert!(visibility(&devices, None).is_err());
        }
        for mask in ["", "-1", "4,-1,7", "4,", "4,4", "GPU-", "invalid"] {
            assert!(visibility(&[0], Some(mask)).is_err(), "{mask}");
        }
        assert!(visibility(&[2], Some("4,7")).is_err());
    }
}
