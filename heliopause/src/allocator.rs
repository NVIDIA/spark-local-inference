/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native callbacks used by the preparation process's PyTorch memory pool.

use std::{
    collections::BTreeMap,
    ffi::{CString, c_char, c_void},
    ptr::null_mut,
    sync::Mutex,
};

use crate::runtime_package::driver::{Access, AllocationProperties, Driver, Location};
use crate::{Error, ExecutionError, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Manifest, message)
}

// PyTorch's CPU allocator is mimalloc, which places its 1 GiB arenas at random addresses
// in 2-6 TiB; a peer rank allocates CPU tensors before reserving these ranges.
#[cfg(target_os = "linux")]
const BASE: u64 = 0x80000000000;
#[cfg(target_os = "windows")]
const BASE: u64 = 0x40000000000;
const EXTENT: usize = 1 << 40;
// Keep small Linux exports within the Windows WDDM address range as well.
const HOST_BASE: u64 = 0xc0000000000;

fn host_base(device_base: u64, world: u32) -> u64 {
    if device_base + u64::from(world) * EXTENT as u64 <= HOST_BASE {
        HOST_BASE
    } else {
        0x200000000000
    }
}

#[derive(Default)]
struct RetainedBacking {
    active: bool,
    aborted: bool,
    finished: bool,
    bytes: usize,
    handles: Vec<(u64, usize)>,
}

impl RetainedBacking {
    fn begin(&mut self) -> Result<()> {
        if self.active || self.aborted || !self.handles.is_empty() {
            return Err(invalid(
                "physical retention scope is already active or abandoned",
            ));
        }
        self.active = true;
        self.finished = false;
        Ok(())
    }

    fn defer_free(&self) -> Result<bool> {
        if self.aborted || (!self.active && !self.finished) {
            return Err(invalid(
                "transient free outside a healthy physical retention scope",
            ));
        }
        Ok(self.active)
    }

    fn retain(&mut self, size: usize, acquire: impl FnOnce() -> Result<u64>) -> Result<()> {
        if !self.active || self.aborted {
            return Err(invalid(
                "transient free outside an active physical retention scope",
            ));
        }
        let bytes = self
            .bytes
            .checked_add(size)
            .ok_or_else(|| invalid("retained physical byte count overflow"))?;
        self.handles
            .try_reserve(1)
            .map_err(|e| invalid(format!("retained handle storage: {e}")))?;
        let handle = acquire()?;
        self.handles.push((handle, size));
        self.bytes = bytes;
        Ok(())
    }

    fn finish(&mut self, mut release: impl FnMut(u64) -> Result<()>) -> Result<()> {
        if !self.active || self.aborted {
            return Err(invalid("physical retention scope is not active"));
        }
        while let Some(&(handle, size)) = self.handles.last() {
            release(handle)?;
            self.handles.pop();
            self.bytes -= size;
        }
        self.active = false;
        self.finished = true;
        Ok(())
    }

    fn abort_on_free_error(&mut self, error: Error) -> Option<Error> {
        if self.aborted {
            return None;
        }
        self.abort();
        Some(invalid(format!(
            "{error}; retained bytes={}, handles={}",
            self.bytes,
            self.handles.len()
        )))
    }

    fn abort(&mut self) {
        self.active = false;
        self.aborted = true;
    }

    /// Take a retained handle backing exactly `size` bytes for remapping, so
    /// warmup retention stays bounded by distinct transient sizes instead of
    /// the warmup count. Only an active scope donates handles; aborted scopes
    /// keep their handles untouched.
    fn take(&mut self, size: usize) -> Option<u64> {
        if !self.active {
            return None;
        }
        let index = self.handles.iter().position(|&(_, bytes)| bytes == size)?;
        let (handle, bytes) = self.handles.swap_remove(index);
        self.bytes -= bytes;
        Some(handle)
    }

    /// Return a handle that failed to remap so it stays retained.
    fn put_back(&mut self, size: usize, handle: u64) {
        self.bytes += size;
        self.handles.push((handle, size));
    }
}

struct Arena {
    driver: Driver,
    device: i32,
    granularity: usize,
    gpudirect: bool,
    host_base: u64,
    free: Vec<(u64, usize)>,
    mappings: BTreeMap<u64, usize>,
    retained: RetainedBacking,
}
static ARENA: Mutex<Option<Arena>> = Mutex::new(None);
static ERROR: Mutex<Option<CString>> = Mutex::new(None);

pub(crate) struct MemoryInventory {
    pub device: i32,
    pub free: usize,
    pub total: usize,
    pub mappings: Vec<(u64, usize)>,
}

pub(crate) fn memory_inventory() -> Result<MemoryInventory> {
    let arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let a = arena
        .as_ref()
        .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
    let (_, device) = crate::current_context()?;
    if device != a.device {
        return Err(invalid(
            "memory inventory requires the package rank's CUDA context",
        ));
    }
    let (mut free, mut total) = (0, 0);
    // SAFETY: Preparation retains the current context and both outputs are valid.
    unsafe {
        a.driver
            .check("cuCtxSynchronize", (a.driver.context_sync)())?;
        a.driver.check(
            "cuMemGetInfo_v2",
            (a.driver.memory_info)(&mut free, &mut total),
        )?;
    }
    Ok(MemoryInventory {
        device,
        free,
        total,
        mappings: a
            .mappings
            .iter()
            .map(|(&address, &size)| (address, size))
            .collect(),
    })
}

fn properties(device: i32, gpudirect: bool) -> AllocationProperties {
    AllocationProperties {
        kind: 1,
        handles: i32::from(cfg!(target_os = "linux")),
        location: Location { kind: 1, device },
        windows: null_mut(),
        // cuFile registration requires gpuDirectRDMACapable on VMM allocations.
        flags: [0, u8::from(gpudirect), 0, 0, 0, 0, 0, 0],
    }
}

fn failure(error: impl ToString) {
    *ERROR.lock().unwrap_or_else(|e| e.into_inner()) =
        Some(CString::new(error.to_string().replace('\0', " ")).unwrap());
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_initialize(rank: u32, world: u32, device: i32) -> i32 {
    let initialize = || -> Result<()> {
        let mut arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        if arena.is_some() {
            return Err(invalid("native package allocator is already initialized"));
        }
        if world == 0 || rank >= world || device < 0 {
            return Err(invalid("invalid package rank or device"));
        }
        let host_base = if cfg!(target_os = "windows") {
            HOST_BASE
        } else {
            host_base(BASE, world)
        };
        if u64::from(world) * EXTENT as u64 > BASE.abs_diff(host_base) {
            return Err(invalid(
                "package host reservations would overlap device reservations",
            ));
        }
        let driver = Driver::load()?;
        let (mut address, mut granularity, mut gpudirect) = (0, 0, 0);
        unsafe {
            driver.check(
                "cuDeviceGetAttribute(GPU_DIRECT_RDMA_SUPPORTED)",
                (driver.device_attribute)(&mut gpudirect, 116, device),
            )?;
            driver.check(
                "cuMemGetAllocationGranularity",
                (driver.granularity)(&mut granularity, &properties(device, gpudirect != 0), 0),
            )?;
            let mut reserved = Vec::new();
            for peer in 0..world {
                let base = BASE + u64::from(peer) * EXTENT as u64;
                let status = (driver.reserve)(&mut address, EXTENT, 0, base, 0);
                if status != 0 || address != base {
                    if status == 0 {
                        (driver.release_address)(address, EXTENT);
                    }
                    for address in reserved {
                        (driver.release_address)(address, EXTENT);
                    }
                    driver.check("cuMemAddressReserve", status)?;
                    return Err(invalid(format!(
                        "native package device address range {base:#x}+{EXTENT:#x} is unavailable; CUDA returned {address:#x}"
                    )));
                }
                reserved.push(address);
            }
        }
        *arena = Some(Arena {
            driver,
            device,
            granularity,
            gpudirect: gpudirect != 0,
            host_base: host_base + u64::from(rank) * EXTENT as u64,
            free: vec![(BASE + u64::from(rank) * EXTENT as u64, EXTENT / 2)],
            mappings: BTreeMap::new(),
            retained: RetainedBacking::default(),
        });
        Ok(())
    };
    match initialize() {
        Ok(()) => 0,
        Err(error) => {
            failure(error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_allocate(
    size: usize,
    device: i32,
    _stream: *mut c_void,
) -> *mut c_void {
    let allocate = || -> Result<u64> {
        let mut guard = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        let a = guard
            .as_mut()
            .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
        if device != a.device {
            return Err(invalid("allocation device differs from package rank"));
        }
        let size = size
            .checked_add(a.granularity - 1)
            .ok_or_else(|| invalid("allocation size overflow"))?
            / a.granularity
            * a.granularity;
        let index = a
            .free
            .iter()
            .position(|(_, n)| *n >= size)
            .ok_or_else(|| invalid("native package address range exhausted"))?;
        let (address, available) = a.free[index];
        // During warmup retention, prefer remapping a donated physical backing
        // over fresh storage so retained memory stays bounded by distinct
        // transient sizes rather than the warmup count. A donated handle was
        // retained after its mapping was unmapped, so remapping it preserves
        // the physical allocation the retention workaround keeps alive.
        let donated = a.retained.take(size);
        let mut physical = 0;
        let mapped = if let Some(handle) = donated {
            physical = handle;
            let mapped = unsafe { (a.driver.map)(address, size, 0, handle, 0) };
            if mapped == 0 {
                // The mapping now holds the only reference, as with fresh
                // storage; a later transient free re-derives a handle.
                unsafe { (a.driver.physical_release)(handle) };
            }
            mapped
        } else {
            unsafe {
                a.driver.check(
                    "cuMemCreate",
                    (a.driver.physical_create)(
                        &mut physical,
                        size,
                        &properties(device, a.gpudirect),
                        0,
                    ),
                )?;
                let mapped = (a.driver.map)(address, size, 0, physical, 0);
                (a.driver.physical_release)(physical);
                mapped
            }
        };
        if mapped != 0 {
            if donated.is_some() {
                // The mapping failed, so the handle still holds the only
                // physical reference; keep it retained rather than release it.
                a.retained.put_back(size, physical);
            }
            a.driver.check("cuMemMap", mapped)?;
        }
        unsafe {
            let access = Access {
                location: Location { kind: 1, device },
                flags: 3,
            };
            let status = (a.driver.access)(address, size, &access, 1);
            if status != 0 {
                (a.driver.unmap)(address, size);
            }
            a.driver.check("cuMemSetAccess", status)?;
        }
        if size == available {
            a.free.remove(index);
        } else {
            a.free[index] = (address + size as u64, available - size);
        }
        a.mappings.insert(address, size);
        Ok(address)
    };
    match allocate() {
        Ok(address) => address as *mut c_void,
        // PyTorch may release cached segments and retry a physical allocation.
        Err(Error::Execution(ExecutionError::Driver(error)))
            if error.operation == "cuMemCreate" && error.code == 2 =>
        {
            null_mut()
        }
        Err(error) => {
            failure(error);
            null_mut()
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_free(
    pointer: *mut c_void,
    _size: usize,
    _device: i32,
    _stream: *mut c_void,
) {
    package_free(pointer, false);
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_transient_free(
    pointer: *mut c_void,
    _size: usize,
    _device: i32,
    _stream: *mut c_void,
) {
    package_free(pointer, true);
}

fn package_free(pointer: *mut c_void, transient: bool) {
    let release = || -> Result<()> {
        let mut guard = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        let a = guard
            .as_mut()
            .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
        let address = pointer as u64;
        let size = *a
            .mappings
            .get(&address)
            .ok_or_else(|| invalid("unknown native package allocation"))?;
        if transient && a.retained.defer_free()? {
            let driver = &a.driver;
            a.retained.retain(size, || {
                let mut handle = 0;
                unsafe {
                    driver.check(
                        "cuMemRetainAllocationHandle",
                        (driver.physical_retain)(&mut handle, pointer),
                    )?;
                }
                Ok(handle)
            })?;
        }
        unsafe {
            a.driver
                .check("cuMemUnmap", (a.driver.unmap)(address, size))?;
        }
        a.mappings.remove(&address);
        a.free.push((address, size));
        a.free.sort_unstable();
        let mut merged: Vec<(u64, usize)> = Vec::new();
        for (address, size) in a.free.drain(..) {
            if let Some(previous) = merged.last_mut().filter(|p| p.0 + p.1 as u64 == address) {
                previous.1 += size;
            } else {
                merged.push((address, size));
            }
        }
        a.free = merged;
        Ok(())
    };
    if let Err(error) = release() {
        if transient {
            let mut arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(a) = arena.as_mut() {
                if let Some(error) = a.retained.abort_on_free_error(error) {
                    failure(error);
                }
                return;
            }
        }
        failure(error);
    }
}

fn retention_operation(operation: impl FnOnce(&mut Arena) -> Result<()>) -> i32 {
    let mut guard = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let Some(a) = guard.as_mut() else {
        failure(invalid("native package allocator is uninitialized"));
        return -1;
    };
    match operation(a) {
        Ok(()) => 0,
        Err(error) => {
            failure(invalid(format!(
                "{error}; retained bytes={}, handles={}",
                a.retained.bytes,
                a.retained.handles.len()
            )));
            -1
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_retention_begin() -> i32 {
    retention_operation(|a| a.retained.begin())
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_retention_finish() -> i32 {
    retention_operation(|a| {
        let driver = &a.driver;
        a.retained.finish(|handle| unsafe {
            driver.check("cuMemRelease", (driver.physical_release)(handle))
        })
    })
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_retention_abort() -> i32 {
    retention_operation(|a| {
        a.retained.abort();
        Ok(())
    })
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_snapshot() -> *mut c_char {
    let arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let Some(a) = arena.as_ref() else {
        return null_mut();
    };
    let value = serde_json::json!({
        "granularity": a.granularity, "mappings": a.mappings, "host_base": a.host_base,
        "retained_bytes": a.retained.bytes, "retained_handles": a.retained.handles.len(),
    });
    CString::new(value.to_string()).unwrap().into_raw()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn flare_package_string_free(pointer: *mut c_char) {
    if !pointer.is_null() {
        unsafe {
            drop(CString::from_raw(pointer));
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_error() -> *const c_char {
    ERROR
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(std::ptr::null(), |s| s.as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secondary_aborted_free_preserves_first_driver_error() {
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.retain(4096, || Ok(7)).unwrap();
        let mut published = retained
            .abort_on_free_error(invalid("cuMemUnmap returned CUDA error 715"))
            .unwrap()
            .to_string();
        let first = published.clone();
        let secondary = retained.defer_free().unwrap_err();
        if let Some(error) = retained.abort_on_free_error(secondary) {
            published = error.to_string();
        }
        assert_eq!(published, first);
        assert!(published.contains("cuMemUnmap returned CUDA error 715"));
        assert!(published.contains("retained bytes=4096, handles=1"));
        assert_eq!(retained.handles, vec![(7, 4096)]);
    }

    #[test]
    fn physical_references_preserve_duplicates_and_release_once() {
        let mut retained = RetainedBacking::default();
        assert!(retained.retain(4, || Ok(7)).is_err());
        retained.begin().unwrap();
        assert!(retained.begin().is_err());
        retained.retain(4, || Ok(7)).unwrap();
        retained.retain(8, || Ok(7)).unwrap();
        assert_eq!(retained.bytes, 12);
        let mut released = Vec::new();
        retained
            .finish(|handle| {
                released.push(handle);
                Ok(())
            })
            .unwrap();
        assert_eq!(released, vec![7, 7]);
        assert_eq!(retained.bytes, 0);
        assert!(retained.handles.is_empty());
        assert!(retained.retain(4, || Ok(8)).is_err());
        assert!(!retained.defer_free().unwrap());
    }

    #[test]
    fn physical_reference_errors_preserve_accounting_and_abort_never_releases() {
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        assert!(
            retained
                .retain(4, || Err(invalid("retain failed")))
                .is_err()
        );
        assert_eq!(retained.bytes, 0);
        retained.retain(4, || Ok(7)).unwrap();
        assert!(retained.finish(|_| Err(invalid("release failed"))).is_err());
        assert_eq!(retained.handles, vec![(7, 4)]);
        assert_eq!(retained.bytes, 4);
        retained.abort();
        assert!(retained.defer_free().is_err());
        assert!(
            retained
                .finish(|_| panic!("must not call CUDA after abort"))
                .is_err()
        );
        assert!(retained.begin().is_err());
        assert_eq!(retained.handles, vec![(7, 4)]);
    }

    #[test]
    fn retained_byte_overflow_precedes_driver_call() {
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.bytes = usize::MAX;
        assert!(
            retained
                .retain(1, || panic!("overflow must precede retain"))
                .is_err()
        );
    }

    #[test]
    fn donation_takes_exact_sizes_and_reaccounts_failed_maps() {
        let mut retained = RetainedBacking::default();
        assert_eq!(retained.take(4), None, "inactive scope donates nothing");
        retained.begin().unwrap();
        retained.retain(4, || Ok(7)).unwrap();
        retained.retain(8, || Ok(8)).unwrap();
        assert_eq!(retained.take(16), None, "no best-fit, exact sizes only");
        assert_eq!(retained.take(8), Some(8));
        assert_eq!(retained.bytes, 4);
        // A failed map must leave the handle retained and accounted.
        retained.put_back(8, 8);
        assert_eq!(retained.bytes, 12);
        assert_eq!(retained.take(8), Some(8));
        retained.put_back(8, 8);
        // A complete drain still releases every handle, including donated ones.
        let mut released = Vec::new();
        retained
            .finish(|handle| {
                released.push(handle);
                Ok(())
            })
            .unwrap();
        assert_eq!(released, vec![8, 7]);
        assert_eq!(retained.bytes, 0);
        assert_eq!(retained.take(4), None, "finished scope donates nothing");
    }

    #[test]
    fn small_linux_exports_keep_every_host_range_within_windows_addresses() {
        let device_base = 8 << 40;
        for world in 1..=4 {
            let host = host_base(device_base, world);
            assert_eq!(host, 12 << 40);
            assert!(device_base + u64::from(world) * EXTENT as u64 <= host);
            assert!(host + u64::from(world) * EXTENT as u64 <= 16 << 40);
        }
    }

    #[test]
    fn larger_linux_exports_keep_host_ranges_clear_of_all_device_ranks() {
        let device_base = 8 << 40;
        for world in 5..=24 {
            let host = host_base(device_base, world);
            assert_eq!(host, 32 << 40);
            assert!(device_base + u64::from(world) * EXTENT as u64 <= host);
        }
    }
}
