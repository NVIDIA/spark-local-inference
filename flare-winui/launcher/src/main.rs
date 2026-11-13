// SPDX-License-Identifier: Apache-2.0
#![windows_subsystem = "windows"]
use std::{
    ffi::OsString,
    os::windows::{ffi::OsStringExt, process::CommandExt},
    path::PathBuf,
    process::Command,
    ptr,
};
use windows_sys::Win32::{
    System::{Com::CoTaskMemFree, Threading::CREATE_NO_WINDOW},
    UI::{
        Shell::{FOLDERID_LocalAppData, SHGetKnownFolderPath},
        WindowsAndMessaging::{MB_ICONERROR, MB_OK, MessageBoxW},
    },
};
const PAYLOAD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/payload.zip"));

fn local_app_data() -> flare_launcher::Result<PathBuf> {
    let mut value = ptr::null_mut();
    let status =
        unsafe { SHGetKnownFolderPath(&FOLDERID_LocalAppData, 0, ptr::null_mut(), &mut value) };
    if status < 0 {
        return Err(format!("Cannot locate Local AppData: HRESULT {status:#x}").into());
    }
    if value.is_null() {
        return Err("Local AppData path is unavailable".into());
    }
    let mut length = 0;
    unsafe {
        while *value.add(length) != 0 {
            length += 1;
        }
    }
    let result = PathBuf::from(OsString::from_wide(unsafe {
        std::slice::from_raw_parts(value, length)
    }));
    unsafe {
        CoTaskMemFree(value.cast());
    }
    Ok(result)
}
fn run() -> flare_launcher::Result<()> {
    let cwd = std::env::current_dir()?;
    let launcher = std::env::current_exe()?;
    if !launcher.is_absolute() {
        return Err("Launcher executable path is not absolute".into());
    }
    let parent = local_app_data()?
        .join("Flare/Desktop")
        .join(env!("FLARE_ARCH"));
    let prepared = flare_launcher::prepare(PAYLOAD, &parent, env!("FLARE_PAYLOAD_SHA256"))?;
    let entrypoint = prepared.directory.join("flare-winui.exe");
    Command::new(&entrypoint)
        .args(std::env::args_os().skip(1))
        .arg("--desktop-launcher")
        .arg(launcher)
        .current_dir(cwd)
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .map_err(|error| format!("Cannot start {}: {error}", entrypoint.display()))?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        let message: Vec<u16> = format!("Flare could not open its desktop application.\n\n{error}")
            .encode_utf16()
            .chain(Some(0))
            .collect();
        let title: Vec<u16> = "Flare".encode_utf16().chain(Some(0)).collect();
        unsafe {
            MessageBoxW(
                ptr::null_mut(),
                message.as_ptr(),
                title.as_ptr(),
                MB_OK | MB_ICONERROR,
            );
        }
        std::process::exit(1);
    }
}
