/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! The GPU virtual address space that fixed-address reservations must fit.
//!
//! Under WDDM, `cuMemAddressReserve` with a fixed address outside the adapter's
//! GPU virtual address space does not return. WDDM reports the address width
//! through `D3DKMTQueryAdapterInfo` (`KMTQAITYPE_QUERY_GPUMMU_CAPS`) for the
//! adapter whose LUID CUDA reports for the device, so fixed reservations are
//! checked against it before they are requested. Where the width is unknown
//! (Linux, or a failed query), no check applies.

use std::ffi::c_int;

/// GPU virtual address width of a CUDA device's WDDM adapter, in bits.
///
/// Returns `None` when the width is unavailable. The CUDA driver must be
/// initialized.
#[cfg(windows)]
pub fn virtual_address_bits(device: c_int) -> Option<u32> {
    use std::ffi::{c_char, c_uint, c_void};

    #[repr(C)]
    struct Luid {
        low: u32,
        high: i32,
    }
    #[repr(C)]
    struct OpenAdapterFromLuid {
        luid: Luid,
        adapter: u32,
    }
    #[repr(C)]
    struct QueryAdapterInfo {
        adapter: u32,
        kind: c_int,
        data: *mut c_void,
        size: c_uint,
    }
    // D3DKMT_QUERY_GPUMMU_CAPS: the physical adapter index, then
    // D3DKMT_GPUMMU_CAPS { Flags, VirtualAddressBitCount }.
    #[repr(C)]
    #[derive(Default)]
    struct GpuMmuCaps {
        physical_adapter_index: u32,
        flags: u32,
        virtual_address_bits: u32,
    }
    #[repr(C)]
    struct CloseAdapter {
        adapter: u32,
    }
    const KMTQAITYPE_QUERY_GPUMMU_CAPS: c_int = 34;

    type GetLuid = unsafe extern "C" fn(*mut c_char, *mut c_uint, c_int) -> c_int;
    type Open = unsafe extern "system" fn(*mut OpenAdapterFromLuid) -> i32;
    type Query = unsafe extern "system" fn(*const QueryAdapterInfo) -> i32;
    type Close = unsafe extern "system" fn(*const CloseAdapter) -> i32;

    // SAFETY: the signatures match cuDeviceGetLuid and the D3DKMT thunks, the
    // structures match d3dkmthk.h, and every pointer outlives its call.
    unsafe {
        let cuda = crate::os::cuda_library().ok()?;
        let get_luid = *cuda.get::<GetLuid>(b"cuDeviceGetLuid\0").ok()?;
        let mut luid = [0u8; 8];
        let mut node_mask: c_uint = 0;
        if get_luid(luid.as_mut_ptr().cast(), &mut node_mask, device) != 0 {
            return None;
        }
        let gdi = libloading::Library::new("gdi32.dll").ok()?;
        let open = *gdi.get::<Open>(b"D3DKMTOpenAdapterFromLuid\0").ok()?;
        let query = *gdi.get::<Query>(b"D3DKMTQueryAdapterInfo\0").ok()?;
        let close = *gdi.get::<Close>(b"D3DKMTCloseAdapter\0").ok()?;
        let mut adapter = OpenAdapterFromLuid {
            luid: Luid {
                low: u32::from_le_bytes(luid[..4].try_into().unwrap()),
                high: i32::from_le_bytes(luid[4..].try_into().unwrap()),
            },
            adapter: 0,
        };
        if open(&mut adapter) != 0 {
            return None;
        }
        let mut caps = GpuMmuCaps::default();
        let info = QueryAdapterInfo {
            adapter: adapter.adapter,
            kind: KMTQAITYPE_QUERY_GPUMMU_CAPS,
            data: (&mut caps as *mut GpuMmuCaps).cast(),
            size: std::mem::size_of::<GpuMmuCaps>() as c_uint,
        };
        let status = query(&info);
        close(&CloseAdapter {
            adapter: adapter.adapter,
        });
        (status == 0 && (1..64).contains(&caps.virtual_address_bits))
            .then_some(caps.virtual_address_bits)
    }
}

/// GPU virtual address width of a CUDA device; unknown outside WDDM.
#[cfg(not(windows))]
pub fn virtual_address_bits(_device: c_int) -> Option<u32> {
    None
}

/// Rejects a fixed reservation of `[address, address + nbytes)` that does not
/// fit a GPU virtual address space of `bits` bits. `what` names the
/// reservation in the error.
pub fn check_fits(what: &str, address: u64, nbytes: u64, bits: Option<u32>) -> Result<(), String> {
    let Some(bits) = bits else {
        return Ok(());
    };
    let limit = 1u64 << bits;
    match address.checked_add(nbytes) {
        Some(end) if end <= limit => Ok(()),
        _ => Err(format!(
            "{what} {address:#x}+{nbytes:#x} exceeds the GPU's {bits}-bit virtual address space \
             (ends at {limit:#x}); CUDA would not return from reserving it"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_fit_or_name_the_limit() {
        assert_eq!(
            check_fits("window", 0x40_0000_0000, 1 << 36, Some(40)),
            Ok(())
        );
        assert_eq!(
            check_fits("window", (1 << 40) - 4096, 4096, Some(40)),
            Ok(())
        );
        let error = check_fits("device window", 0x400_0000_0000, 1 << 40, Some(40)).unwrap_err();
        assert!(
            error.contains("device window 0x40000000000+0x10000000000"),
            "{error}"
        );
        assert!(error.contains("40-bit"), "{error}");
        assert!(check_fits("window", u64::MAX, 2, Some(48)).is_err());
        assert_eq!(check_fits("window", u64::MAX - 1, 1, None), Ok(()));
    }
}
