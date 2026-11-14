/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU inventory read through NVML (`libnvidia-ml.so.1`). NVML queries the driver without
//! creating a CUDA context, so the agent can report every GPU on its host, including GPUs that
//! running ranks own.
//!
//! The library is loaded and initialized once per process and stays initialized until exit:
//! every `nvmlInit_v2` and `nvmlShutdown` cycle leaves an event descriptor open, so an agent
//! sampling once a second would exhaust its descriptor limit within minutes.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use libloading::{Library, Symbol};
use serde::{Deserialize, Serialize};

use super::Failure;

const LIBRARY: &str = "libnvidia-ml.so.1";
/// NVML_DEVICE_UUID_V2_BUFFER_SIZE and NVML_DEVICE_NAME_V2_BUFFER_SIZE.
const TEXT_BUFFER: usize = 96;
/// NVML_ERROR_NOT_SUPPORTED.
const NOT_SUPPORTED: c_int = 3;

/// One GPU as the agent reports it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gpu {
    /// NVML enumeration index. It need not equal the CUDA device ordinal; select devices by
    /// [`Gpu::uuid`].
    pub index: u32,
    /// `GPU-xxxxxxxx-...` as NVML reports it, a form `CUDA_VISIBLE_DEVICES` accepts.
    pub uuid: String,
    /// Product name, such as `NVIDIA GB300`.
    pub name: String,
    /// Flare hardware revision: `gb10`, `gb202`, `gb300`, or `sm{major}{minor}` for any other
    /// compute capability.
    pub revision: String,
    /// Device memory in bytes. NVML reports no device memory for an integrated GPU that shares
    /// host memory (GB10); this field then holds the host's `MemTotal`.
    pub memory_total: u64,
    /// Unallocated device memory in bytes, or the host's `MemAvailable` for an integrated GPU.
    pub memory_free: u64,
    /// Percent of the sample period in which a kernel ran; absent when NVML does not report it.
    #[serde(default)]
    pub utilization: Option<u32>,
    /// Percent of the sample period in which the memory controller was busy, the share of the
    /// GPU's peak memory bandwidth in use; absent when NVML does not report it.
    #[serde(default)]
    pub memory_utilization: Option<u32>,
    /// GPU temperature in degrees Celsius; absent when NVML does not report it.
    #[serde(default)]
    pub temperature: Option<u32>,
    /// SM clock in MHz; absent when NVML does not report it.
    #[serde(default)]
    pub clock_mhz: Option<u32>,
}

static NVML: OnceLock<Result<&'static Library, String>> = OnceLock::new();

/// The NVML library, loaded and initialized on first use for the rest of the process.
fn nvml() -> Result<&'static Library, Failure> {
    NVML.get_or_init(|| {
        // SAFETY: Only the system NVML library is loaded, and it is never unloaded.
        let library = unsafe { Library::new(LIBRARY) }
            .map_err(|error| format!("cannot load {LIBRARY}: {error}"))?;
        let library: &'static Library = Box::leak(Box::new(library));
        // SAFETY: The signatures match the NVML ABI in nvml.h.
        unsafe {
            let describe = library
                .get::<ErrorString>(b"nvmlErrorString\0")
                .map_err(|e| e.to_string())?;
            let init = library
                .get::<unsafe extern "C" fn() -> c_int>(b"nvmlInit_v2\0")
                .map_err(|e| e.to_string())?;
            check(&describe, "nvmlInit_v2", init()).map_err(|e| e.to_string())?;
        }
        Ok(library)
    })
    .clone()
    .map_err(Failure::from)
}

/// Every GPU NVML reports, in NVML index order. Fails with an error naming `libnvidia-ml.so.1`
/// when that library cannot be loaded.
pub fn enumerate() -> Result<Vec<Gpu>, Failure> {
    let library = nvml()?;
    // SAFETY: These signatures match the NVML ABI in nvml.h, the structures below match its
    // layouts, and every output pointer is valid for the duration of its call.
    unsafe {
        let describe = library.get::<ErrorString>(b"nvmlErrorString\0")?;
        let get_count = library
            .get::<unsafe extern "C" fn(*mut c_uint) -> c_int>(b"nvmlDeviceGetCount_v2\0")?;
        let get_handle = library.get::<unsafe extern "C" fn(c_uint, *mut Device) -> c_int>(
            b"nvmlDeviceGetHandleByIndex_v2\0",
        )?;
        let get_uuid = library.get::<unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int>(
            b"nvmlDeviceGetUUID\0",
        )?;
        let get_name = library.get::<unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int>(
            b"nvmlDeviceGetName\0",
        )?;
        let get_memory = library.get::<unsafe extern "C" fn(Device, *mut Memory) -> c_int>(
            b"nvmlDeviceGetMemoryInfo\0",
        )?;
        let get_pci = library.get::<unsafe extern "C" fn(Device, *mut PciInfo) -> c_int>(
            b"nvmlDeviceGetPciInfo_v3\0",
        )?;
        let get_capability =
            library.get::<unsafe extern "C" fn(Device, *mut c_int, *mut c_int) -> c_int>(
                b"nvmlDeviceGetCudaComputeCapability\0",
            )?;
        let get_utilization = library
            .get::<unsafe extern "C" fn(Device, *mut Utilization) -> c_int>(
                b"nvmlDeviceGetUtilizationRates\0",
            )?;
        let get_temperature = library
            .get::<unsafe extern "C" fn(Device, c_uint, *mut c_uint) -> c_int>(
                b"nvmlDeviceGetTemperature\0",
            )?;
        let get_clock = library.get::<unsafe extern "C" fn(Device, c_uint, *mut c_uint) -> c_int>(
            b"nvmlDeviceGetClockInfo\0",
        )?;

        let mut count: c_uint = 0;
        check(&describe, "nvmlDeviceGetCount_v2", get_count(&mut count))?;
        let mut gpus = Vec::with_capacity(count as usize);
        for index in 0..count {
            let mut device: Device = std::ptr::null_mut();
            check(
                &describe,
                "nvmlDeviceGetHandleByIndex_v2",
                get_handle(index, &mut device),
            )?;

            let mut uuid = [0u8; TEXT_BUFFER];
            check(
                &describe,
                "nvmlDeviceGetUUID",
                get_uuid(device, uuid.as_mut_ptr().cast(), TEXT_BUFFER as c_uint),
            )?;
            let mut name = [0u8; TEXT_BUFFER];
            check(
                &describe,
                "nvmlDeviceGetName",
                get_name(device, name.as_mut_ptr().cast(), TEXT_BUFFER as c_uint),
            )?;

            let mut pci = PciInfo {
                _bus_id_legacy: [0; 16],
                _domain: 0,
                _bus: 0,
                _device: 0,
                pci_device_id: 0,
                _pci_subsystem_id: 0,
                _bus_id: [0; 32],
            };
            check(
                &describe,
                "nvmlDeviceGetPciInfo_v3",
                get_pci(device, &mut pci),
            )?;
            let (mut major, mut minor) = (0, 0);
            check(
                &describe,
                "nvmlDeviceGetCudaComputeCapability",
                get_capability(device, &mut major, &mut minor),
            )?;

            let mut memory = Memory {
                total: 0,
                free: 0,
                _used: 0,
            };
            let code = get_memory(device, &mut memory);
            let (memory_total, memory_free) = if code == NOT_SUPPORTED {
                host_memory()?
            } else {
                check(&describe, "nvmlDeviceGetMemoryInfo", code)?;
                (memory.total, memory.free)
            };

            // Readings a GPU does not support are absent rather than failing the inventory.
            let optional = |code: c_int, value: u32| -> Result<Option<u32>, Failure> {
                if code == NOT_SUPPORTED {
                    Ok(None)
                } else {
                    check(&describe, "nvml reading", code)?;
                    Ok(Some(value))
                }
            };
            let mut rates = Utilization { gpu: 0, memory: 0 };
            let code = get_utilization(device, &mut rates);
            let utilization = optional(code, rates.gpu)?;
            let memory_utilization = optional(code, rates.memory)?;
            let mut temperature: c_uint = 0;
            // NVML_TEMPERATURE_GPU is sensor 0.
            let temperature = optional(get_temperature(device, 0, &mut temperature), temperature)?;
            let mut clock: c_uint = 0;
            // NVML_CLOCK_SM is clock type 1.
            let clock_mhz = optional(get_clock(device, 1, &mut clock), clock)?;

            gpus.push(Gpu {
                index,
                uuid: text(&uuid),
                name: text(&name),
                revision: revision(pci.pci_device_id, major, minor),
                memory_total,
                memory_free,
                utilization,
                memory_utilization,
                temperature,
                clock_mhz,
            });
        }
        Ok(gpus)
    }
}

/// The Flare hardware revision of a compute capability: `gb10`, `gb202`, `gb300`, or
/// `sm{major}{minor}`.
pub fn capability_revision(major: i32, minor: i32) -> String {
    match (major, minor) {
        (12, 1) => "gb10".into(),
        (12, 0) => "gb202".into(),
        (10, 3) => "gb300".into(),
        _ => format!("sm{major}{minor}"),
    }
}

/// The `CUDA_VISIBLE_DEVICES` value that exposes exactly `uuids`, in that order.
pub fn visible_devices(uuids: &[String]) -> String {
    uuids.join(",")
}

/// nvmlDevice_t, an opaque handle owned by NVML.
type Device = *mut c_void;
type ErrorString = unsafe extern "C" fn(c_int) -> *const c_char;

/// nvmlUtilization_t: percent of the last sample period with a kernel running, and with memory
/// traffic.
#[repr(C)]
struct Utilization {
    gpu: c_uint,
    memory: c_uint,
}

/// nvmlMemory_t, the version 1 layout that nvmlDeviceGetMemoryInfo fills.
#[repr(C)]
struct Memory {
    total: u64,
    free: u64,
    _used: u64,
}

/// nvmlPciInfo_t as nvmlDeviceGetPciInfo_v3 fills it.
#[repr(C)]
struct PciInfo {
    _bus_id_legacy: [c_char; 16],
    _domain: c_uint,
    _bus: c_uint,
    _device: c_uint,
    /// Device ID in the high 16 bits and vendor ID in the low 16 bits, the layout of CUDA's
    /// GPU_PCI_DEVICE_ID attribute.
    pci_device_id: c_uint,
    _pci_subsystem_id: c_uint,
    _bus_id: [c_char; 32],
}

fn check(describe: &Symbol<'_, ErrorString>, call: &str, code: c_int) -> Result<(), Failure> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: nvmlErrorString accepts any return code and returns a static NUL-terminated string.
    let message = unsafe {
        let message = describe(code);
        if message.is_null() {
            String::new()
        } else {
            CStr::from_ptr(message).to_string_lossy().into_owned()
        }
    };
    Err(format!("NVML {call} failed with {code} ({message})").into())
}

/// The NUL-terminated text NVML wrote into `buffer`.
fn text(buffer: &[u8]) -> String {
    let end = buffer
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..end]).into_owned()
}

fn revision(pci_device_id: u32, major: c_int, minor: c_int) -> String {
    if let Some(revision) = platform(pci_device_id) {
        return revision.into();
    }
    match (major, minor) {
        (12, 1) => "gb10".into(),
        (12, 0) => "gb202".into(),
        (10, 3) => "gb300".into(),
        _ => format!("sm{major}{minor}"),
    }
}

/// The PCI ID table of `crate::hardware`, which selects a revision before compute capability.
fn platform(pci_id: u32) -> Option<&'static str> {
    if pci_id & 0xffff != 0x10de {
        return None;
    }
    match pci_id >> 16 {
        0x2e12 => Some("gb10"),
        0x2b85 | 0x2b87 | 0x2b8c | 0x2bb1 | 0x2bb2 | 0x2bb3 | 0x2bb4 | 0x2bb5 | 0x2bb9 | 0x2bbc => {
            Some("gb202")
        }
        _ => None,
    }
}

/// Total and available host memory in bytes, which an integrated GPU allocates from.
fn host_memory() -> Result<(u64, u64), Failure> {
    let text = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| format!("cannot read /proc/meminfo: {error}"))?;
    meminfo(&text).ok_or_else(|| "/proc/meminfo lacks MemTotal or MemAvailable".into())
}

/// `MemTotal` and `MemAvailable` from /proc/meminfo text, converted from KiB to bytes.
fn meminfo(text: &str) -> Option<(u64, u64)> {
    let field = |name: &str| {
        text.lines().find_map(|line| {
            let kib = line
                .strip_prefix(name)?
                .strip_prefix(':')?
                .trim()
                .strip_suffix("kB")?;
            kib.trim().parse::<u64>().ok()?.checked_mul(1024)
        })
    };
    Some((field("MemTotal")?, field("MemAvailable")?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pci_ids_select_the_revision_before_compute_capability() {
        assert_eq!(revision(0x2e1210de, 12, 1), "gb10");
        assert_eq!(revision(0x2bb410de, 12, 0), "gb202");
        assert_eq!(revision(0x2b8510de, 9, 9), "gb202");
    }

    #[test]
    fn unknown_pci_ids_fall_back_to_compute_capability() {
        assert_eq!(revision(0x31c310de, 10, 3), "gb300");
        assert_eq!(revision(0x2e1310de, 12, 1), "gb10");
        assert_eq!(revision(0x2c0010de, 12, 0), "gb202");
        assert_eq!(revision(0x2bb41234, 9, 0), "sm90");
        assert_eq!(revision(0x233010de, 9, 0), "sm90");
        assert_eq!(revision(0x290110de, 10, 0), "sm100");
    }

    #[test]
    fn visible_devices_lists_uuids_in_order() {
        let uuids = ["GPU-b".to_string(), "GPU-a".to_string()];
        assert_eq!(visible_devices(&uuids), "GPU-b,GPU-a");
        assert_eq!(visible_devices(&uuids[..1]), "GPU-b");
        assert_eq!(visible_devices(&[]), "");
    }

    #[test]
    fn meminfo_reports_total_and_available_bytes() {
        let text = "MemTotal:       130535212 kB\nMemFree:         1000 kB\nMemAvailable:   120000000 kB\n";
        assert_eq!(meminfo(text), Some((130535212 * 1024, 120000000 * 1024)));
        assert_eq!(meminfo("MemTotal: 1 kB\n"), None);
    }

    #[test]
    fn enumeration_reports_uuids_and_memory_when_nvml_is_present() {
        // Hosts without the NVIDIA driver have nothing to enumerate.
        let Ok(gpus) = enumerate() else { return };
        for (position, gpu) in gpus.iter().enumerate() {
            assert_eq!(gpu.index as usize, position);
            assert!(gpu.uuid.starts_with("GPU-"), "{gpu:?}");
            assert!(!gpu.name.is_empty(), "{gpu:?}");
            assert!(gpu.memory_total > 0, "{gpu:?}");
            assert!(gpu.memory_free <= gpu.memory_total, "{gpu:?}");
        }
    }
}
