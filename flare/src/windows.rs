/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native Windows console and per-user trust-store operations.

use std::{
    fs::File,
    io,
    os::windows::{
        ffi::{OsStrExt, OsStringExt},
        io::{AsRawHandle, FromRawHandle},
    },
    path::{Path, PathBuf},
};
use windows_sys::Win32::{
    Foundation::*,
    Security::{Authorization::*, *},
    Storage::FileSystem::*,
    System::{Com::CoTaskMemFree, Console::*, IO::*},
    UI::Shell::*,
};

fn wide(path: &Path) -> io::Result<Vec<u16>> {
    let mut value: Vec<_> = path.as_os_str().encode_wide().collect();
    if value.contains(&0) {
        return Err(io::ErrorKind::InvalidInput.into());
    }
    value.push(0);
    Ok(value)
}

pub fn trust_path() -> io::Result<PathBuf> {
    Ok(known_folder(&FOLDERID_LocalAppData)?
        .join("Flare")
        .join("trust.toml"))
}

pub fn home_directory() -> io::Result<PathBuf> {
    known_folder(&FOLDERID_Profile)
}

fn known_folder(folder: &windows_sys::core::GUID) -> io::Result<PathBuf> {
    unsafe {
        let mut text = std::ptr::null_mut();
        let status = SHGetKnownFolderPath(folder, 0, std::ptr::null_mut(), &mut text);
        if status < 0 {
            return Err(io::Error::other(format!(
                "resolve user folder: HRESULT {status:#x}"
            )));
        }
        let mut length = 0;
        while *text.add(length) != 0 {
            length += 1;
        }
        let path = PathBuf::from(std::ffi::OsString::from_wide(std::slice::from_raw_parts(
            text, length,
        )));
        CoTaskMemFree(text.cast());
        Ok(path)
    }
}

pub fn private_file(path: &Path, exclusive: bool) -> io::Result<File> {
    let path = wide(path)?;
    let acl: Vec<u16> = "D:P(A;;FA;;;SY)(A;;FA;;;OW)\0".encode_utf16().collect();
    unsafe {
        let mut descriptor = std::ptr::null_mut();
        if ConvertStringSecurityDescriptorToSecurityDescriptorW(
            acl.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            std::ptr::null_mut(),
        ) == 0
        {
            return Err(io::Error::last_os_error());
        }
        let attributes = SECURITY_ATTRIBUTES {
            nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: descriptor,
            bInheritHandle: 0,
        };
        let handle = CreateFileW(
            path.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            &attributes,
            if exclusive { CREATE_NEW } else { OPEN_ALWAYS },
            FILE_ATTRIBUTE_NORMAL,
            std::ptr::null_mut(),
        );
        let error = io::Error::last_os_error();
        LocalFree(descriptor);
        if handle == INVALID_HANDLE_VALUE {
            return Err(error);
        }
        Ok(File::from_raw_handle(handle))
    }
}

pub fn lock(file: &File) -> io::Result<()> {
    let mut operation: OVERLAPPED = unsafe { std::mem::zeroed() };
    if unsafe {
        LockFileEx(
            file.as_raw_handle(),
            LOCKFILE_EXCLUSIVE_LOCK,
            0,
            1,
            0,
            &mut operation,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Try to take the exclusive lock without waiting. Returns false when another
/// holder owns it.
pub fn try_lock(file: &File) -> io::Result<bool> {
    let mut operation: OVERLAPPED = unsafe { std::mem::zeroed() };
    if unsafe {
        LockFileEx(
            file.as_raw_handle(),
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
            0,
            1,
            0,
            &mut operation,
        )
    } == 0
    {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(ERROR_LOCK_VIOLATION as i32) {
            return Ok(false);
        }
        return Err(error);
    }
    Ok(true)
}

pub fn replace(source: &Path, destination: &Path) -> io::Result<()> {
    let source = wide(source)?;
    let destination = wide(destination)?;
    if unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub fn terminal_columns() -> usize {
    let mut info: CONSOLE_SCREEN_BUFFER_INFO = unsafe { std::mem::zeroed() };
    if unsafe { GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &mut info) } == 0 {
        return 80;
    }
    (info.srWindow.Right - info.srWindow.Left + 1) as usize
}

/// Stable volume/file identity and hardlink count for cache ownership accounting.
pub fn file_identity(path: &Path) -> Option<((u64, u64), u64)> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
    };
    let file = File::open(path).ok()?;
    let mut information: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: The file handle is live and information is a writable output buffer.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle().cast(), &mut information) } == 0 {
        return None;
    }
    Some((
        (
            u64::from(information.dwVolumeSerialNumber),
            (u64::from(information.nFileIndexHigh) << 32) | u64::from(information.nFileIndexLow),
        ),
        u64::from(information.nNumberOfLinks),
    ))
}
