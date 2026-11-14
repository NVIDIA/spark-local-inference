/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-backed CUDA VMM preserves recorded mailbox addresses on Windows.

use super::{Result, invalid};

fn ranges(
    allocations: impl Iterator<Item = (u64, usize)>,
    granularity: usize,
) -> Result<Vec<(u64, usize)>> {
    let granularity = granularity as u64;
    if !granularity.is_power_of_two() {
        return Err(invalid("invalid host allocation granularity"));
    }
    let mut ranges = allocations
        .map(|(address, bytes)| {
            let end = address
                .checked_add(bytes as u64)
                .and_then(|end| end.checked_add(granularity - 1))
                .ok_or_else(|| invalid("host allocation exceeds address space"))?;
            Ok((address & !(granularity - 1), end & !(granularity - 1)))
        })
        .collect::<Result<Vec<_>>>()?;
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut().filter(|last| start <= last.1) {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    Ok(merged
        .into_iter()
        .map(|(start, end)| (start, (end - start) as usize))
        .collect())
}

#[cfg(windows)]
impl super::NativeResources {
    pub(super) unsafe fn allocate_host(&mut self, core: &super::Core) -> Result<()> {
        use super::driver::Access;
        use super::{AllocationProperties, Location, Memory};

        if !core
            .allocations
            .iter()
            .any(|a| matches!(a.memory, Memory::MappedHost))
        {
            return Ok(());
        }
        let driver = &self.driver;
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            let mut supported = 0;
            driver.check(
                "cuDeviceGetAttribute(HOST_NUMA_VMM)",
                (driver.device_attribute)(&mut supported, 141, self.device),
            )?;
            if supported == 0 {
                return Err(invalid(
                    "mapped core storage requires host-backed CUDA VMM on Windows",
                ));
            }
            let mut numa = 0;
            driver.check(
                "cuDeviceGetAttribute(HOST_NUMA_ID)",
                (driver.device_attribute)(&mut numa, 134, self.device),
            )?;
            let properties = AllocationProperties {
                kind: 1,
                handles: 0,
                location: Location {
                    kind: 3,
                    device: numa.max(0),
                },
                windows: std::ptr::null_mut(),
                flags: [0; 8],
            };
            let mut granularity = 0;
            driver.check(
                "cuMemGetAllocationGranularity(host)",
                (driver.granularity)(&mut granularity, &properties, 0),
            )?;
            let ranges = ranges(
                core
                    .allocations
                    .iter()
                    .filter(|a| matches!(a.memory, Memory::MappedHost))
                    .map(|a| (a.address.unwrap(), a.nbytes)),
                granularity,
            )?;
            let access = [
                Access {
                    location: Location {
                        kind: 3,
                        device: numa.max(0),
                    },
                    flags: 3,
                },
                Access {
                    location: Location {
                        kind: 1,
                        device: self.device,
                    },
                    flags: 3,
                },
            ];
            for (address, bytes) in ranges {
                let mut pointer = 0;
                driver.check(
                    "cuMemAddressReserve(host)",
                    (driver.reserve)(&mut pointer, bytes, granularity, address, 0),
                )?;
                self.reservations.push((pointer, bytes));
                if pointer != address {
                    return Err(invalid(format!(
                        "host CUDA address collision at {address:#x}"
                    )));
                }
                let mut physical = 0;
                driver.check(
                    "cuMemCreate(host)",
                    (driver.physical_create)(&mut physical, bytes, &properties, 0),
                )?;
                let status = (driver.map)(address, bytes, 0, physical, 0);
                (driver.physical_release)(physical);
                driver.check("cuMemMap(host)", status)?;
                self.host_mappings.push((address, bytes));
                driver.check(
                    "cuMemSetAccess(host)",
                    (driver.access)(address, bytes, access.as_ptr(), access.len()),
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_host_pages_share_physical_backing() {
        let base = 0xc00_0000_0000;
        let result = ranges(
            [
                (base + 4096, 4096),
                (base + 8192, 4096),
                (base + (4 << 20), 4096),
            ]
            .into_iter(),
            2 << 20,
        )
        .unwrap();
        assert_eq!(result, [(base, 2 << 20), (base + (4 << 20), 2 << 20)]);
    }

    #[test]
    fn host_range_rounding_rejects_overflow() {
        assert!(ranges([(u64::MAX - 4095, 4096)].into_iter(), 2 << 20).is_err());
        assert!(ranges([(4096, 4096)].into_iter(), 0).is_err());
    }
}
