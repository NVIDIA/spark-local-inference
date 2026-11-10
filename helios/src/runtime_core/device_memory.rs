/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Observe memory before creating CUDA contexts, so occupied limits include driver setup.

use std::ffi::{c_char, c_int, c_void};

use super::{NativeResources, driver::Driver, format::invalid, memory::MemorySnapshot};
use crate::Result;

#[repr(C)]
#[derive(Default)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

fn check(operation: &str, status: c_int) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(invalid(format!(
            "occupied memory accounting: {operation} failed with NVML error {status}"
        )))
    }
}

/// CUDA's cuMemGetInfo requires a context. NVML can observe discrete memory without one.
/// Its device is resolved by PCI address so CUDA visibility and NVML ordinals cannot disagree.
unsafe fn discrete_free(pci: &[c_char]) -> Result<u64> {
    #[cfg(unix)]
    let library = unsafe { libloading::Library::new("libnvidia-ml.so.1") };
    #[cfg(windows)]
    let library = unsafe {
        libloading::os::windows::Library::load_with_flags(
            "nvml.dll",
            windows_sys::Win32::System::LibraryLoader::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
        .map(libloading::Library::from)
    };
    let library =
        library.map_err(|e| invalid(format!("occupied memory limits require NVIDIA NVML: {e}")))?;
    // SAFETY: Signatures and NvmlMemory match NVIDIA's NVML ABI. The library remains live,
    // and shutdown balances initialization even if a subsequent query fails.
    unsafe {
        let init = library
            .get::<unsafe extern "C" fn() -> c_int>(b"nvmlInit_v2\0")
            .map_err(|e| e.to_string())?;
        let shutdown = library
            .get::<unsafe extern "C" fn() -> c_int>(b"nvmlShutdown\0")
            .map_err(|e| e.to_string())?;
        let device = library
            .get::<unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> c_int>(
                b"nvmlDeviceGetHandleByPciBusId_v2\0",
            )
            .map_err(|e| e.to_string())?;
        let info = library
            .get::<unsafe extern "C" fn(*mut c_void, *mut NvmlMemory) -> c_int>(
                b"nvmlDeviceGetMemoryInfo\0",
            )
            .map_err(|e| e.to_string())?;
        check("nvmlInit_v2", init())?;
        let result = (|| {
            let mut handle = std::ptr::null_mut();
            check(
                "nvmlDeviceGetHandleByPciBusId_v2",
                device(pci.as_ptr(), &mut handle),
            )?;
            let mut memory = NvmlMemory::default();
            check("nvmlDeviceGetMemoryInfo", info(handle, &mut memory))?;
            Ok(memory.free)
        })();
        shutdown();
        result
    }
}

impl NativeResources {
    pub(super) unsafe fn memory_before_context(ordinal: i32) -> Result<MemorySnapshot> {
        let driver = Driver::load()?;
        unsafe {
            driver.check("cuInit", (driver.init)(0))?;
            let (mut device, mut integrated, mut total) = (0, 0, 0usize);
            driver.check("cuDeviceGet", (driver.device)(&mut device, ordinal))?;
            driver.check(
                "cuDeviceGetAttribute(INTEGRATED)",
                (driver.device_attribute)(&mut integrated, 18, device),
            )?;
            driver.check(
                "cuDeviceTotalMem_v2",
                (driver.device_total_memory)(&mut total, device),
            )?;
            let free = if integrated == 0 {
                let mut pci = [0; 32];
                driver.check(
                    "cuDeviceGetPCIBusId",
                    (driver.device_pci_bus_id)(pci.as_mut_ptr(), pci.len() as c_int, device),
                )?;
                discrete_free(&pci)?.min(total as u64)
            } else {
                #[cfg(unix)]
                {
                    let info =
                        std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
                    super::memory::shared_available(&info, usize::MAX)? as u64
                }
                #[cfg(windows)]
                {
                    use windows_sys::Win32::System::SystemInformation::{
                        GlobalMemoryStatusEx, MEMORYSTATUSEX,
                    };
                    let mut status: MEMORYSTATUSEX = std::mem::zeroed();
                    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
                    if GlobalMemoryStatusEx(&mut status) == 0 {
                        return Err(std::io::Error::last_os_error().to_string().into());
                    }
                    // Windows keeps the GPU carveout outside CPU-visible physical RAM.
                    let mut pci = [0; 32];
                    driver.check(
                        "cuDeviceGetPCIBusId",
                        (driver.device_pci_bus_id)(pci.as_mut_ptr(), pci.len() as c_int, device),
                    )?;
                    // Both remain bounded by the commit this process can obtain.
                    status
                        .ullAvailPhys
                        .saturating_add(discrete_free(&pci)?)
                        .min(total as u64)
                        .min(super::memory::commit_available()?)
                }
            };
            Ok(MemorySnapshot {
                free,
                total: total as u64,
            })
        }
    }
}
