/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hardware revision selection through NVML without initializing CUDA.

#[path = "hardware_monitor.rs"]
mod monitor;
pub use monitor::{HardwareMonitor, HardwareSnapshot};

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};

use libloading::Library;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn platform(pci_id: u32) -> Option<&'static str> {
    // Single mapping table shared with core export naming.
    flare_helios::core::naming::platform_from_pci(pci_id)
}

pub fn detect(count: usize) -> Result<&'static str> {
    let inherited = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    let indices: Vec<_> = (0..count as i32).collect();
    let visible = crate::environment::visibility(&indices, inherited.as_deref())?;
    // SAFETY: Only the system driver is loaded, and the library remains live through all calls.
    #[cfg(unix)]
    let library = unsafe { Library::new("libnvidia-ml.so.1") };
    #[cfg(windows)]
    let library = unsafe {
        libloading::os::windows::Library::load_with_flags(
            "nvml.dll",
            windows_sys::Win32::System::LibraryLoader::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
        .map(Library::from)
    };
    let library = library.map_err(|error| {
        format!("cannot detect hardware without NVIDIA NVML: {error}; specify --revision")
    })?;
    // SAFETY: These signatures and PciInfo match nvml.h; output pointers are valid.
    unsafe {
        let init = library.get::<unsafe extern "C" fn() -> c_int>(b"nvmlInit_v2\0")?;
        let shutdown = library.get::<unsafe extern "C" fn() -> c_int>(b"nvmlShutdown\0")?;
        check("nvmlInit_v2", init())?;
        let result = (|| {
            let get = library.get::<unsafe extern "C" fn(c_uint, *mut Device) -> c_int>(
                b"nvmlDeviceGetHandleByIndex_v2\0",
            )?;
            let get_by_uuid = library
                .get::<unsafe extern "C" fn(*const c_char, *mut Device) -> c_int>(
                    b"nvmlDeviceGetHandleByUUID\0",
                )?;
            let get_count = library
                .get::<unsafe extern "C" fn(*mut c_uint) -> c_int>(b"nvmlDeviceGetCount_v2\0")?;
            let get_uuid = library
                .get::<unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int>(
                    b"nvmlDeviceGetUUID\0",
                )?;
            let get_pci = library.get::<unsafe extern "C" fn(Device, *mut PciInfo) -> c_int>(
                b"nvmlDeviceGetPciInfo_v3\0",
            )?;
            let mut selected = None;
            for entry in visible.split(',') {
                let mut device = std::ptr::null_mut();
                if let Ok(index) = entry.parse::<u32>() {
                    check("nvmlDeviceGetHandleByIndex_v2", get(index, &mut device))?;
                } else if entry.len() == 40 {
                    let uuid = CString::new(entry)?;
                    check(
                        "nvmlDeviceGetHandleByUUID",
                        get_by_uuid(uuid.as_ptr(), &mut device),
                    )?;
                } else {
                    let mut total = 0;
                    check("nvmlDeviceGetCount_v2", get_count(&mut total))?;
                    let mut candidates = Vec::new();
                    for index in 0..total {
                        let mut candidate = std::ptr::null_mut();
                        check("nvmlDeviceGetHandleByIndex_v2", get(index, &mut candidate))?;
                        let mut uuid = [0; 96];
                        check(
                            "nvmlDeviceGetUUID",
                            get_uuid(candidate, uuid.as_mut_ptr(), 96),
                        )?;
                        candidates.push((
                            CStr::from_ptr(uuid.as_ptr()).to_str()?.to_owned(),
                            candidate,
                        ));
                    }
                    device = candidates[unique_uuid(entry, &candidates)?].1;
                }
                let mut pci: PciInfo = std::mem::zeroed();
                check("nvmlDeviceGetPciInfo_v3", get_pci(device, &mut pci))?;
                let detected = platform(pci.device_id).ok_or_else(|| {
                    format!(
                        "GPU {entry} has unsupported PCI ID {:#010x}; specify --revision",
                        pci.device_id
                    )
                })?;
                if selected.is_some_and(|value| value != detected) {
                    return Err(
                        "selected GPUs span multiple platforms; select one platform or specify --revision"
                            .into(),
                    );
                }
                selected = Some(detected);
            }
            selected.ok_or_else(|| "select at least one GPU or specify --revision".into())
        })();
        let closed = check("nvmlShutdown", shutdown());
        result.and_then(|revision| closed.map(|()| revision))
    }
}

/// Driver-reported GPU metadata; querying NVML creates no CUDA context.
#[derive(Clone, Debug, Default)]
pub struct GpuInfo {
    pub ordinal: u32,
    pub visible_ordinal: Option<u32>,
    pub name: String,
    pub uuid: String,
    pub total_memory_bytes: Option<u64>,
    pub free_memory_bytes: Option<u64>,
    pub architecture: Option<[u32; 2]>,
    pub platform: Option<String>,
}

#[repr(C)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

/// Enumerate actual NVIDIA devices and their inherited visibility without changing process policy.
pub fn inventory() -> Result<Vec<GpuInfo>> {
    #[cfg(unix)]
    let library = unsafe { Library::new("libnvidia-ml.so.1") };
    #[cfg(windows)]
    let library = unsafe {
        libloading::os::windows::Library::load_with_flags(
            "nvml.dll",
            windows_sys::Win32::System::LibraryLoader::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
        .map(Library::from)
    };
    let library =
        library.map_err(|error| format!("NVIDIA device information is unavailable: {error}"))?;
    // SAFETY: These NVML signatures match the system driver. Every output buffer is live and
    // the library remains loaded until the initialized NVML reference has been released.
    unsafe {
        let init = library.get::<unsafe extern "C" fn() -> c_int>(b"nvmlInit_v2\0")?;
        let shutdown = library.get::<unsafe extern "C" fn() -> c_int>(b"nvmlShutdown\0")?;
        let initialized = init();
        if initialized != 0 {
            return Err(format!("NVIDIA device query failed with NVML error {initialized}").into());
        }
        let result = (|| -> Result<Vec<GpuInfo>> {
            let count_fn = library
                .get::<unsafe extern "C" fn(*mut c_uint) -> c_int>(b"nvmlDeviceGetCount_v2\0")?;
            let get = library.get::<unsafe extern "C" fn(c_uint, *mut Device) -> c_int>(
                b"nvmlDeviceGetHandleByIndex_v2\0",
            )?;
            let name_fn = library
                .get::<unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int>(
                    b"nvmlDeviceGetName\0",
                )?;
            let uuid_fn = library
                .get::<unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int>(
                    b"nvmlDeviceGetUUID\0",
                )?;
            let memory_fn = library.get::<unsafe extern "C" fn(Device, *mut NvmlMemory) -> c_int>(
                b"nvmlDeviceGetMemoryInfo\0",
            )?;
            let architecture_fn = library
                .get::<unsafe extern "C" fn(Device, *mut c_int, *mut c_int) -> c_int>(
                    b"nvmlDeviceGetCudaComputeCapability\0",
                )
                .ok();
            let pci_fn = library.get::<unsafe extern "C" fn(Device, *mut PciInfo) -> c_int>(
                b"nvmlDeviceGetPciInfo_v3\0",
            )?;
            let mut count = 0;
            let code = count_fn(&mut count);
            if code != 0 {
                return Err(
                    format!("NVIDIA device enumeration failed with NVML error {code}").into(),
                );
            }
            let mut devices = Vec::new();
            for ordinal in 0..count {
                let mut device = std::ptr::null_mut();
                let code = get(ordinal, &mut device);
                if code != 0 {
                    return Err(
                        format!("Cannot query NVIDIA GPU {ordinal}: NVML error {code}").into(),
                    );
                }
                let mut name = [0 as c_char; 256];
                let mut uuid = [0 as c_char; 96];
                let name = if name_fn(device, name.as_mut_ptr(), name.len() as u32) == 0 {
                    CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned()
                } else {
                    format!("NVIDIA GPU {ordinal}")
                };
                let uuid = if uuid_fn(device, uuid.as_mut_ptr(), uuid.len() as u32) == 0 {
                    CStr::from_ptr(uuid.as_ptr()).to_string_lossy().into_owned()
                } else {
                    String::new()
                };
                let mut memory: NvmlMemory = std::mem::zeroed();
                let memory_known = memory_fn(device, &mut memory) == 0
                    && memory.total > 0
                    && memory.total != u64::MAX;
                let mut major = 0;
                let mut minor = 0;
                let architecture = architecture_fn.as_ref().and_then(|query| {
                    (query(device, &mut major, &mut minor) == 0 && major > 0 && minor >= 0)
                        .then_some([major as u32, minor as u32])
                });
                let mut pci: PciInfo = std::mem::zeroed();
                let platform = (pci_fn(device, &mut pci) == 0)
                    .then(|| platform(pci.device_id))
                    .flatten()
                    .map(str::to_owned);
                // NVML also enumerates accelerators without CUDA capability or a supported GPU identity.
                if architecture.is_none() && platform.is_none() {
                    continue;
                }
                devices.push(GpuInfo {
                    ordinal,
                    visible_ordinal: None,
                    name,
                    uuid,
                    total_memory_bytes: memory_known.then_some(memory.total),
                    free_memory_bytes: (memory_known && memory.free <= memory.total)
                        .then_some(memory.free),
                    architecture,
                    platform,
                });
            }
            assign_visibility(
                &mut devices,
                std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref(),
            )?;
            Ok(devices)
        })();
        let released = shutdown();
        if released != 0 {
            return Err(
                format!("NVIDIA device query shutdown failed with NVML error {released}").into(),
            );
        }
        result
    }
}

fn assign_visibility(devices: &mut [GpuInfo], inherited: Option<&str>) -> Result<()> {
    let Some(inherited) = inherited else {
        for (visible, device) in devices.iter_mut().enumerate() {
            device.visible_ordinal = Some(visible as u32);
        }
        return Ok(());
    };
    let inherited = inherited.trim();
    if inherited.is_empty() || inherited == "-1" {
        return Ok(());
    }
    crate::environment::visibility(&[0], Some(inherited))?;
    for (visible, identifier) in inherited.split(',').map(str::trim).enumerate() {
        let matches = devices
            .iter()
            .enumerate()
            .filter_map(|(index, device)| {
                let matches = identifier.parse::<u32>().map_or_else(
                    |_| !device.uuid.is_empty() && device.uuid.starts_with(identifier),
                    |ordinal| index == ordinal as usize,
                );
                matches.then_some(index)
            })
            .collect::<Vec<_>>();
        let [index] = matches.as_slice() else {
            return Err(format!("CUDA_VISIBLE_DEVICES entry {identifier:?} does not identify one detected NVIDIA GPU").into());
        };
        if devices[*index].visible_ordinal.is_some() {
            return Err("CUDA_VISIBLE_DEVICES lists a GPU more than once".into());
        }
        devices[*index].visible_ordinal = Some(visible as u32);
    }
    Ok(())
}

type Device = *mut c_void;

/// nvmlPciInfo_t as filled by nvmlDeviceGetPciInfo_v3.
#[repr(C)]
struct PciInfo {
    _bus_id_legacy: [c_char; 16],
    _domain: c_uint,
    _bus: c_uint,
    _device: c_uint,
    device_id: c_uint,
    _subsystem_id: c_uint,
    _bus_id: [c_char; 32],
}

fn unique_uuid(prefix: &str, candidates: &[(String, Device)]) -> Result<usize> {
    let mut matches = candidates
        .iter()
        .enumerate()
        .filter(|(_, (uuid, _))| uuid.starts_with(prefix));
    match (matches.next(), matches.next()) {
        (Some((index, _)), None) => Ok(index),
        _ => {
            Err(format!("CUDA_VISIBLE_DEVICES entry {prefix} must identify exactly one GPU").into())
        }
    }
}

fn check(operation: &str, code: c_int) -> Result<()> {
    if code != 0 {
        return Err(format!(
            "hardware detection: {operation} failed with NVML error {code}; specify --revision"
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_inventory_visibility_preserves_indices_and_requires_unique_uuid_prefixes() {
        let mut devices = vec![
            GpuInfo {
                ordinal: 0,
                uuid: "GPU-aaaa".into(),
                ..Default::default()
            },
            GpuInfo {
                ordinal: 1,
                uuid: "GPU-bbbb".into(),
                ..Default::default()
            },
        ];
        assign_visibility(&mut devices, Some("1,0")).unwrap();
        assert_eq!(devices[1].visible_ordinal, Some(0));
        assert_eq!(devices[0].visible_ordinal, Some(1));
        for device in &mut devices {
            device.visible_ordinal = None;
        }
        assign_visibility(&mut devices, Some("GPU-bb")).unwrap();
        assert_eq!(devices[1].visible_ordinal, Some(0));
        assert_eq!(devices[0].visible_ordinal, None);
        for device in &mut devices {
            device.visible_ordinal = None;
        }
        assert!(assign_visibility(&mut devices, Some("GPU-")).is_err());
        assign_visibility(&mut devices, Some("")).unwrap();
        assert!(
            devices
                .iter()
                .all(|device| device.visible_ordinal.is_none())
        );
        assign_visibility(&mut devices, Some("-1")).unwrap();
        assert!(
            devices
                .iter()
                .all(|device| device.visible_ordinal.is_none())
        );
    }

    #[test]
    fn uuid_selection_requires_an_unambiguous_prefix() {
        let candidates = vec![
            ("GPU-abcdef01".into(), std::ptr::null_mut()),
            ("GPU-abcdef02".into(), std::ptr::null_mut()),
            ("GPU-12345678".into(), std::ptr::null_mut()),
        ];
        assert_eq!(unique_uuid("GPU-abcdef01", &candidates).unwrap(), 0);
        assert_eq!(unique_uuid("GPU-123", &candidates).unwrap(), 2);
        assert!(unique_uuid("GPU-abcdef0", &candidates).is_err());
        assert!(unique_uuid("GPU-999", &candidates).is_err());
    }

    #[test]
    fn pci_ids_select_the_chip_without_matching_marketing_names() {
        assert_eq!(platform(0x2e1210de), Some("gb10"));
        assert_eq!(platform(0x2bb410de), Some("gb202"));
        assert_eq!(platform(0x2bb110de), Some("gb202"));
        assert_eq!(platform(0x2bb510de), Some("gb202"));
        assert_eq!(platform(0x2b8510de), Some("gb202"));
        assert_eq!(platform(0x31c310de), Some("gb300"));
        assert_eq!(platform(0x26b110de), None);
        assert_eq!(platform(0x2bb41234), None);
    }
}
