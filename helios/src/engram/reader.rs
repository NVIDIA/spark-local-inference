/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounded, direct-I/O row gathering into the same buffers used by resident tables.

use super::{Stats, cache::RowCache};
use crate::Result;
use std::{
    fs::File,
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};

#[cfg(test)]
use std::path::Path;

#[cfg(unix)]
mod uring;
#[cfg(windows)]
mod windows;

const BLOCK: usize = 4096;
const CHUNK: usize = 64 << 10;
const DEPTH: usize = 128;
const LOAD_CHUNK: usize = 4 << 20;
const LOAD_READERS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plane {
    pub file: usize,
    pub offset: u64,
    pub nbytes: usize,
    pub row_bytes: usize,
}

impl Plane {
    pub(crate) fn resident_bytes(self) -> Result<usize> {
        ((self.offset % BLOCK as u64) as usize)
            .checked_add(self.nbytes)
            .and_then(|n| n.checked_add(BLOCK - 1))
            .map(|n| (n / BLOCK * BLOCK).max(BLOCK))
            .ok_or_else(|| "resident embedding allocation exceeds host range".into())
    }
}

#[derive(Clone, Copy)]
struct Fragment {
    file: usize,
    source: u64,
    destination: usize,
    nbytes: usize,
}

struct Read {
    file: usize,
    offset: u64,
    nbytes: usize,
    required: usize,
    fragments: std::ops::Range<usize>,
}

struct Aligned(usize, usize);
impl Aligned {
    fn new(nbytes: usize) -> Result<Self> {
        Ok(Self(
            crate::os::allocate_aligned(nbytes).map_err(|e| format!("Engram allocation: {e}"))?,
            nbytes,
        ))
    }
    fn at(&self, slot: usize) -> *mut u8 {
        (self.0 + slot * CHUNK) as *mut u8
    }
}

#[cfg(test)]
pub(super) fn open_file(path: &Path) -> Result<File> {
    crate::os::direct_file(path, false).map_err(|e| format!("open Engram package: {e}").into())
}

pub(crate) struct Resident {
    #[cfg(unix)]
    storage: ResidentMapping,
    #[cfg(windows)]
    storage: Aligned,
    #[cfg(unix)]
    file: File,
    skip: usize,
    nbytes: usize,
}

#[cfg(unix)]
struct ResidentMapping {
    address: usize,
    nbytes: usize,
}

#[cfg(unix)]
impl ResidentMapping {
    fn new(file: &File, nbytes: usize, writable: bool) -> Result<Self> {
        let protection = libc::PROT_READ | if writable { libc::PROT_WRITE } else { 0 };
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                nbytes,
                protection,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(format!(
                "map resident embedding table: {}",
                std::io::Error::last_os_error()
            )
            .into());
        }
        let mapping = Self {
            address: address as usize,
            nbytes,
        };
        // The sealed read-only remap has its own page tables, even after loading
        // populated the backing RAM. Prepare them before row delivery starts.
        if !writable && unsafe { libc::madvise(address, nbytes, libc::MADV_POPULATE_READ) } != 0 {
            let error = std::io::Error::last_os_error();
            return Err(format!("populate resident embedding page tables: {error}").into());
        }
        Ok(mapping)
    }
}

#[cfg(unix)]
impl Drop for ResidentMapping {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.address as *mut libc::c_void, self.nbytes) };
    }
}

#[cfg(unix)]
const RESIDENT_SEALS: i32 =
    libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;

#[cfg(unix)]
fn seal_resident(file: &File) -> Result<()> {
    // The loading threads have joined and their writable mapping is gone.
    // Kernel-held page references can still make F_SEAL_WRITE temporarily busy.
    // Publish only after every required seal succeeds; persistent busy state fails.
    for attempt in 0..4 {
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, RESIDENT_SEALS) } == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EBUSY) || attempt == 3 {
            return Err(format!("seal resident embedding table: {error}").into());
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    unreachable!()
}

impl Resident {
    pub(super) fn load(file: &File, plane: Plane) -> Result<Self> {
        Self::initialize(plane, |address, extent, required| {
            let skip = (plane.offset % BLOCK as u64) as usize;
            let offset = plane.offset - skip as u64;
            std::thread::scope(|scope| -> Result<()> {
                let mut workers = Vec::new();
                for reader in 0..LOAD_READERS.min(extent.div_ceil(LOAD_CHUNK)) {
                    workers.push(
                        std::thread::Builder::new()
                            .name(format!("flare-table-loader-{reader}"))
                            .spawn_scoped(scope, move || -> Result<()> {
                                for begin in (reader * LOAD_CHUNK..extent)
                                    .step_by(LOAD_READERS * LOAD_CHUNK)
                                {
                                    let nbytes = LOAD_CHUNK.min(extent - begin);
                                    let source = offset
                                        .checked_add(begin as u64)
                                        .ok_or("resident Engram file offset exceeds range")?;
                                    // SAFETY: Each reader initializes a disjoint aligned allocation range.
                                    let count = unsafe { crate::os::read_at_raw(file, (address + begin) as *mut u8, nbytes, source) }
                                        .map_err(|e| format!("resident Engram read at {source}: {e}"))?;
                                    let needed = nbytes.min(required - begin);
                                    if count < needed {
                                        return Err(format!(
                                            "resident Engram read at {source} returned {count}, requires {needed} bytes"
                                        ).into());
                                    }
                                }
                                Ok(())
                            })
                            .map_err(|e| format!("start resident Engram loader: {e}"))?,
                    );
                }
                for worker in workers {
                    worker
                        .join()
                        .map_err(|_| "resident Engram loader panicked")??;
                }
                Ok(())
            })
        })
    }

    #[cfg(unix)]
    pub(super) fn load_segments(plane: Plane, segments: &[(File, u64, usize)]) -> Result<Self> {
        if plane.offset != 0 {
            return Err("assembled embedding planes must start at offset zero".into());
        }
        Self::initialize(plane, |address, _, _| {
            let mut destination = 0usize;
            for (file, offset, nbytes) in segments {
                let end = destination
                    .checked_add(*nbytes)
                    .filter(|end| *end <= plane.nbytes)
                    .ok_or("embedding source segments exceed their plane")?;
                let mut copied = 0;
                while copied < *nbytes {
                    let count = (*nbytes - copied).min(LOAD_CHUNK);
                    // SAFETY: Source segments initialize consecutive, disjoint ranges
                    // inside the resident allocation before it becomes immutable.
                    let read = unsafe {
                        crate::os::read_at_raw(
                            file,
                            (address + destination + copied) as *mut u8,
                            count,
                            offset + copied as u64,
                        )
                    }
                    .map_err(|error| format!("read embedding source segment: {error}"))?;
                    if read == 0 {
                        return Err(
                            "embedding source segment ended before its declared size".into()
                        );
                    }
                    copied += read;
                }
                destination = end;
            }
            if destination != plane.nbytes {
                return Err("embedding source segments do not fill their plane".into());
            }
            Ok(())
        })
    }

    fn initialize(
        plane: Plane,
        fill: impl FnOnce(usize, usize, usize) -> Result<()>,
    ) -> Result<Self> {
        let skip = (plane.offset % BLOCK as u64) as usize;
        let required = skip
            .checked_add(plane.nbytes)
            .ok_or("resident Engram plane exceeds host range")?;
        let extent = plane.resident_bytes()?;
        #[cfg(unix)]
        let backing = {
            let descriptor = unsafe {
                libc::memfd_create(
                    c"flare-embedding-table".as_ptr(),
                    libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
                )
            };
            if descriptor < 0 {
                return Err(format!(
                    "create resident embedding table: {}",
                    std::io::Error::last_os_error()
                )
                .into());
            }
            // The descriptor is newly created and owned exclusively until published.
            let backing = unsafe { File::from_raw_fd(descriptor) };
            backing
                .set_len(extent as u64)
                .map_err(|e| format!("size resident embedding table: {e}"))?;
            backing
        };
        #[cfg(unix)]
        let storage = ResidentMapping::new(&backing, extent, true)?;
        #[cfg(unix)]
        let address = storage.address;
        #[cfg(windows)]
        let storage = Aligned::new(extent)?;
        #[cfg(windows)]
        let address = storage.0;
        fill(address, extent, required)?;
        #[cfg(unix)]
        {
            drop(storage);
            seal_resident(&backing)?;
            Self::from_shared(backing, plane)
        }
        #[cfg(windows)]
        {
            Ok(Self {
                storage,
                skip,
                nbytes: plane.nbytes,
            })
        }
    }

    #[cfg(unix)]
    pub(super) fn from_shared(file: File, plane: Plane) -> Result<Self> {
        let seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
        if seals < 0 || seals & RESIDENT_SEALS != RESIDENT_SEALS {
            return Err("resident embedding table must have immutable size and contents".into());
        }
        let skip = (plane.offset % BLOCK as u64) as usize;
        let extent = plane.resident_bytes()?;
        if file
            .metadata()
            .map_err(|e| format!("inspect resident embedding table: {e}"))?
            .len()
            != extent as u64
        {
            return Err("shared resident embedding table extent differs from its plane".into());
        }
        Ok(Self {
            storage: ResidentMapping::new(&file, extent, false)?,
            file,
            skip,
            nbytes: plane.nbytes,
        })
    }

    #[cfg(unix)]
    pub(super) fn descriptor(&self) -> i32 {
        self.file.as_raw_fd()
    }

    fn bytes(&self) -> &[u8] {
        #[cfg(unix)]
        let address = self.storage.address;
        #[cfg(windows)]
        let address = self.storage.0;
        // SAFETY: Every logical byte was initialized by successful reads, and
        // scoped readers joined before this immutable view becomes available.
        unsafe { std::slice::from_raw_parts((address + self.skip) as *const u8, self.nbytes) }
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { crate::os::free_aligned(self.0, self.1) };
    }
}

pub(crate) struct DiskReader {
    #[cfg(unix)]
    uring: uring::Readers,
    #[cfg(windows)]
    windows: windows::Readers,
    fragments: Vec<Fragment>,
    reads: Vec<Read>,
    resident_scales: Option<Resident>,
    stats: Arc<Stats>,
}

// Accumulate locally so timing does not add atomic operations to each completion.
// Drop also records completed phases when a gather exits with an I/O error.
struct DiskTimings<'a> {
    stats: &'a Stats,
    plan: u64,
    submit: u64,
    wait: u64,
    complete: u64,
}

impl Drop for DiskTimings<'_> {
    fn drop(&mut self) {
        for (counter, value) in [
            (&self.stats.disk_plan_ns, self.plan),
            (&self.stats.disk_submit_ns, self.submit),
            (&self.stats.disk_wait_ns, self.wait),
            (&self.stats.disk_complete_ns, self.complete),
        ] {
            counter.fetch_add(value, Ordering::Relaxed);
        }
    }
}

impl DiskReader {
    fn from_files(files: &[Arc<File>], stats: Arc<Stats>, workers: usize) -> Result<Self> {
        #[cfg(unix)]
        let uring = uring::Readers::new(files, workers)?;
        #[cfg(windows)]
        let windows = windows::Readers::new(files, workers)?;
        Ok(Self {
            #[cfg(unix)]
            uring,
            #[cfg(windows)]
            windows,
            fragments: Vec::new(),
            reads: Vec::new(),
            resident_scales: None,
            stats,
        })
    }

    pub fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        let mut timing = DiskTimings {
            stats: &self.stats,
            plan: 0,
            submit: 0,
            wait: 0,
            complete: 0,
        };
        let begin = Instant::now();
        self.fragments.clear();
        let mut base = 0;
        for (ordinal, plane) in planes.iter().enumerate() {
            if ordinal == 1 {
                if let Some(scales) = &self.resident_scales {
                    gather_resident(scales, plane.row_bytes, ids, &mut output[base..]);
                    base += ids.len() * plane.row_bytes;
                    continue;
                }
            }
            for (row, id) in ids.iter().copied().enumerate() {
                let destination = base + row * plane.row_bytes;
                if id < 0 {
                    output[destination..destination + plane.row_bytes].fill(0);
                    continue;
                }
                let source = plane.offset + id as u64 * plane.row_bytes as u64;
                let mut copied = 0;
                while copied < plane.row_bytes {
                    let address = source + copied as u64;
                    let nbytes = (plane.row_bytes - copied).min(BLOCK - address as usize % BLOCK);
                    self.fragments.push(Fragment {
                        file: plane.file,
                        source: address,
                        destination: destination + copied,
                        nbytes,
                    });
                    copied += nbytes;
                }
            }
            base += ids.len() * plane.row_bytes;
        }
        self.fragments.sort_unstable_by_key(|f| (f.file, f.source));
        self.reads.clear();
        for (i, fragment) in self.fragments.iter().enumerate() {
            let block = fragment.source / BLOCK as u64 * BLOCK as u64;
            if let Some(read) = self.reads.last_mut().filter(|r| {
                r.file == fragment.file
                    && block <= r.offset + r.nbytes as u64
                    && block + BLOCK as u64 <= r.offset + CHUNK as u64
            }) {
                read.nbytes = (block - read.offset) as usize + BLOCK;
                read.required = read
                    .required
                    .max((fragment.source - read.offset) as usize + fragment.nbytes);
                read.fragments.end = i + 1;
            } else {
                self.reads.push(Read {
                    file: fragment.file,
                    offset: block,
                    nbytes: BLOCK,
                    required: (fragment.source - block) as usize + fragment.nbytes,
                    fragments: i..i + 1,
                });
            }
        }
        timing.plan = begin.elapsed().as_nanos() as u64;
        #[cfg(unix)]
        self.uring
            .gather(&self.reads, &self.fragments, output, &self.stats)?;
        #[cfg(windows)]
        self.windows
            .gather(&self.reads, &self.fragments, output, &self.stats)?;

        Ok(())
    }
}

struct CachedReader {
    disk: DiskReader,
    cache: RowCache,
    requests: Vec<(i64, usize)>,
    miss_ids: Vec<i64>,
    miss_groups: Vec<(usize, usize)>,
    miss_output: Vec<u8>,
    max_rows: usize,
}

impl CachedReader {
    fn new(disk: DiskReader, planes: &[Plane; 2], budget: usize, max_rows: usize) -> Result<Self> {
        if max_rows == 0 {
            return Err("embedding row cache requires a nonzero gather capacity".into());
        }
        let row_bytes = planes.iter().map(|plane| plane.row_bytes).sum::<usize>();
        let workspace = max_rows
            .checked_mul(
                size_of::<(i64, usize)>()
                    + size_of::<i64>()
                    + size_of::<(usize, usize)>()
                    + row_bytes,
            )
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or("embedding row cache workspace size overflow")?;
        let payload_budget = budget
            .checked_sub(workspace)
            .ok_or("embedding row cache budget cannot hold its bounded miss workspace")?;
        fn allocated<T: Copy>(len: usize, value: T) -> Result<Vec<T>> {
            let mut data = Vec::new();
            data.try_reserve_exact(len)
                .map_err(|e| format!("embedding row cache workspace: {e}"))?;
            data.resize(len, value);
            // Touch the entire workspace before KV sizing, including vectors
            // whose length is cleared after allocation.
            for index in (0..len).step_by((4096 / size_of::<T>()).max(1)) {
                unsafe { data.as_mut_ptr().add(index).write_volatile(data[index]) };
            }
            Ok(data)
        }
        let mut requests = allocated(max_rows, (0i64, 0usize))?;
        let mut miss_ids = allocated(max_rows, 0i64)?;
        let mut miss_groups = allocated(max_rows, (0usize, 0usize))?;
        requests.clear();
        miss_ids.clear();
        miss_groups.clear();
        let miss_output = allocated(max_rows * row_bytes, 0u8)?;
        let cache = RowCache::new(
            payload_budget,
            planes[0].nbytes / planes[0].row_bytes,
            row_bytes,
        )?;
        let bytes = cache.bytes() + workspace;
        disk.stats
            .capacity_rows
            .fetch_add(cache.capacity() as u64, Ordering::Relaxed);
        disk.stats
            .cache_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(Self {
            disk,
            cache,
            requests,
            miss_ids,
            miss_groups,
            miss_output,
            max_rows,
        })
    }

    fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        if ids.len() > self.max_rows {
            return Err("embedding gather exceeds the row cache workspace capacity".into());
        }
        let [weights, scales] = planes.map(|plane| plane.row_bytes);
        let scale_base = ids.len() * weights;
        self.requests.clear();
        self.miss_ids.clear();
        self.miss_groups.clear();
        for (row, &id) in ids.iter().enumerate() {
            if id < 0 {
                output[row * weights..(row + 1) * weights].fill(0);
                output[scale_base + row * scales..scale_base + (row + 1) * scales].fill(0);
            } else {
                self.requests.push((id, row));
            }
        }
        self.requests.sort_unstable_by_key(|&(id, _)| id);
        let mut begin = 0;
        let mut hits = 0;
        while begin < self.requests.len() {
            let id = self.requests[begin].0;
            let end = begin + self.requests[begin..].partition_point(|&(next, _)| next == id);
            if let Some(payload) = self.cache.get(id) {
                hits += 1;
                for &(_, row) in &self.requests[begin..end] {
                    output[row * weights..(row + 1) * weights].copy_from_slice(&payload[..weights]);
                    output[scale_base + row * scales..scale_base + (row + 1) * scales]
                        .copy_from_slice(&payload[weights..]);
                }
            } else {
                self.miss_ids.push(id);
                self.miss_groups.push((begin, end));
            }
            begin = end;
        }
        self.disk.stats.hits.fetch_add(hits, Ordering::Relaxed);
        self.disk
            .stats
            .misses
            .fetch_add(self.miss_ids.len() as u64, Ordering::Relaxed);
        if self.miss_ids.is_empty() {
            return Ok(());
        }
        let count = self.miss_ids.len();
        self.disk.gather(
            planes,
            &self.miss_ids,
            &mut self.miss_output[..count * (weights + scales)],
        )?;
        for (index, (&id, &(begin, end))) in self.miss_ids.iter().zip(&self.miss_groups).enumerate()
        {
            let w = &self.miss_output[index * weights..(index + 1) * weights];
            let s = &self.miss_output
                [count * weights + index * scales..count * weights + (index + 1) * scales];
            for &(_, row) in &self.requests[begin..end] {
                output[row * weights..(row + 1) * weights].copy_from_slice(w);
                output[scale_base + row * scales..scale_base + (row + 1) * scales]
                    .copy_from_slice(s);
            }
            // Only complete successful disk gathers enter the immutable cache.
            let inserted = self.cache.insert(id, w, s);
            self.disk
                .stats
                .resident_rows
                .fetch_add(u64::from(inserted.added), Ordering::Relaxed);
            self.disk
                .stats
                .evictions
                .fetch_add(u64::from(inserted.evicted), Ordering::Relaxed);
        }
        Ok(())
    }
}

enum Data {
    Disk(DiskReader),
    Ram([Arc<Resident>; 2]),
    Cached(Box<CachedReader>),
}

fn gather_resident(data: &Resident, row_bytes: usize, ids: &[i64], output: &mut [u8]) {
    let data = data.bytes();
    for (row, &id) in ids.iter().enumerate() {
        let target = &mut output[row * row_bytes..(row + 1) * row_bytes];
        if id < 0 {
            target.fill(0);
        } else {
            let start = id as usize * row_bytes;
            target.copy_from_slice(&data[start..start + row_bytes]);
        }
    }
}
pub(crate) struct Reader {
    data: Data,
    stats: Arc<Stats>,
}
impl Reader {
    #[cfg(test)]
    fn open(
        path: &Path,
        planes: &[Plane; 2],
        resident: bool,
        cache_bytes: usize,
        max_rows: usize,
        stats: Arc<Stats>,
    ) -> Result<Self> {
        Self::from_files(
            &[Arc::new(open_file(path)?)],
            planes,
            resident,
            cache_bytes,
            max_rows,
            stats,
            1,
            false,
        )
    }
    pub fn from_files(
        files: &[Arc<File>],
        planes: &[Plane; 2],
        resident: bool,
        cache_bytes: usize,
        max_rows: usize,
        stats: Arc<Stats>,
        workers: usize,
        resident_scales: bool,
    ) -> Result<Self> {
        if resident && cache_bytes != 0 {
            return Err("embedding row cache requires disk table storage".into());
        }
        if planes.iter().any(|plane| plane.file >= files.len()) {
            return Err("embedding plane references an unknown file".into());
        }
        let data = if resident {
            Data::Ram([
                Arc::new(Resident::load(&files[planes[0].file], planes[0])?),
                Arc::new(Resident::load(&files[planes[1].file], planes[1])?),
            ])
        } else {
            let mut disk = DiskReader::from_files(files, stats.clone(), workers)?;
            if resident_scales {
                disk.resident_scales = Some(Resident::load(&files[planes[1].file], planes[1])?);
                stats.resident_scale_bytes.fetch_add(planes[1].resident_bytes()? as u64, Ordering::Relaxed);
            }
            if cache_bytes != 0 {
                Data::Cached(Box::new(CachedReader::new(
                    disk,
                    planes,
                    cache_bytes,
                    max_rows,
                )?))
            } else {
                Data::Disk(disk)
            }
        };
        Ok(Self { data, stats })
    }
    #[cfg(unix)]
    pub(super) fn resident(data: [Arc<Resident>; 2], stats: Arc<Stats>) -> Self {
        Self {
            data: Data::Ram(data),
            stats,
        }
    }
    pub fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        if output.len() != ids.len() * planes.iter().map(|p| p.row_bytes).sum::<usize>() {
            return Err("Engram staging size differs from row count".into());
        }
        for plane in planes {
            if ids
                .iter()
                .any(|id| *id < -1 || (*id >= 0 && *id as usize >= plane.nbytes / plane.row_bytes))
            {
                return Err("Engram row exceeds its table".into());
            }
        }
        let begin = Instant::now();
        let result = match &mut self.data {
            Data::Disk(reader) => reader.gather(planes, ids, output),
            Data::Cached(reader) => reader.gather(planes, ids, output),
            Data::Ram(data) => {
                let mut base = 0;
                for (plane, data) in planes.iter().zip(data) {
                    gather_resident(data, plane.row_bytes, ids, &mut output[base..]);
                    base += ids.len() * plane.row_bytes;
                }
                Ok(())
            }
        };
        self.stats.gathers.fetch_add(1, Ordering::Relaxed);
        self.stats.gather_ns.fetch_add(
            begin.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;

    #[test]
    fn resident_scales_preserve_bytes_and_cache_hits_without_reading_the_scale_file() {
        for (weight_row_bytes, scale_row_bytes) in [(256, 8), (80, 10)] {
            check_resident_scale_rows(weight_row_bytes, scale_row_bytes);
        }
    }

    fn check_resident_scale_rows(weight_row_bytes: usize, scale_row_bytes: usize) {
        let root = std::env::temp_dir().join(format!("helios-engram-scales-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let rows = 4097;
        let planes = [
            Plane { file: 0, offset: 4093, nbytes: rows * weight_row_bytes, row_bytes: weight_row_bytes },
            Plane { file: 1, offset: 4093, nbytes: rows * scale_row_bytes, row_bytes: scale_row_bytes },
        ];
        let row_bytes = weight_row_bytes + scale_row_bytes;
        let data: Vec<Vec<u8>> = planes.iter().map(|plane| {
            (0..plane.offset as usize + plane.nbytes).map(|i| (i * 19 + i / 257) as u8).collect()
        }).collect();
        for cache in [0, 65536] {
            let mut files = Vec::new();
            for (i, bytes) in data.iter().enumerate() {
                let path = root.join(i.to_string());
                std::fs::write(&path, bytes).unwrap();
                files.push(Arc::new(open_file(&path).unwrap()));
            }
            let stats = Arc::new(Stats::default());
            let mut reader = Reader::from_files(&files, &planes, false, cache, 8, stats.clone(), 1, true).unwrap();
            assert_eq!(stats.resident_scale_bytes.load(Ordering::Relaxed), planes[1].resident_bytes().unwrap() as u64);
            OpenOptions::new().write(true).open(root.join("1")).unwrap().set_len(0).unwrap();
            for ids in [vec![0, 4096, -1, 0], vec![4096, 0, -1, 0], vec![17, -1, 19]] {
                let mut expected = Vec::new();
                for (plane, bytes) in planes.iter().zip(&data) {
                    for &id in &ids {
                        if id < 0 {
                            expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                        } else {
                            let start = plane.offset as usize + id as usize * plane.row_bytes;
                            expected.extend_from_slice(&bytes[start..start + plane.row_bytes]);
                        }
                    }
                }
                let mut output = vec![0xa5; expected.len()];
                reader.gather(&planes, &ids, &mut output).unwrap();
                assert_eq!(output, expected);
            }
            assert!(stats.disk_reads.load(Ordering::Relaxed) > 0);
            if cache > 0 {
                assert_eq!(stats.hits.load(Ordering::Relaxed), 2);
                assert_eq!(stats.misses.load(Ordering::Relaxed), 4);
                let reads = stats.disk_reads.load(Ordering::Relaxed);
                reader.gather(&planes, &[17, 19], &mut vec![0; row_bytes * 2]).unwrap();
                assert_eq!(stats.disk_reads.load(Ordering::Relaxed), reads);
            }
            assert!(reader.gather(&planes, &[rows as i64], &mut vec![0; row_bytes]).is_err());
            assert!(reader.gather(&planes, &[-2], &mut vec![0; row_bytes]).is_err());
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    fn unsealed_resident_file() -> File {
        let descriptor = unsafe {
            libc::memfd_create(
                c"helios-resident-seal-test".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        assert!(descriptor >= 0);
        let file = unsafe { File::from_raw_fd(descriptor) };
        file.set_len(BLOCK as u64).unwrap();
        file
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn resident_sealing_waits_for_unmapped_pages_held_by_io() {
        use std::io::Read;
        use std::sync::mpsc::{self, RecvTimeoutError};
        use std::time::Duration;

        let file = unsealed_resident_file();
        let mapping = ResidentMapping::new(&file, BLOCK, true).unwrap();
        unsafe { std::ptr::write_bytes(mapping.address as *mut u8, 0x39, BLOCK) };
        let mut descriptors = [-1; 2];
        assert_eq!(
            unsafe { libc::pipe2(descriptors.as_mut_ptr(), libc::O_CLOEXEC) },
            0
        );
        let mut reader = unsafe { File::from_raw_fd(descriptors[0]) };
        let writer = unsafe { File::from_raw_fd(descriptors[1]) };
        let vector = libc::iovec {
            iov_base: mapping.address as *mut libc::c_void,
            iov_len: BLOCK,
        };
        assert_eq!(
            unsafe { libc::vmsplice(writer.as_raw_fd(), &vector, 1, 0) },
            BLOCK as isize
        );
        drop(mapping);
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, RESIDENT_SEALS) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EBUSY)
        );

        std::thread::scope(|scope| {
            let (send, receive) = mpsc::channel();
            let file = &file;
            scope.spawn(move || send.send(seal_resident(file)).unwrap());
            // The I/O reference remains live beyond the kernel's initial pin wait.
            assert!(matches!(
                receive.recv_timeout(Duration::from_millis(250)),
                Err(RecvTimeoutError::Timeout)
            ));
            let mut bytes = [0; BLOCK];
            reader.read_exact(&mut bytes).unwrap();
            assert_eq!(bytes, [0x39; BLOCK]);
            receive.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
        });
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) } & RESIDENT_SEALS,
            RESIDENT_SEALS
        );
        assert!(file.set_len(1).is_err());
        let value = 0u8;
        assert_eq!(
            unsafe { libc::pwrite(file.as_raw_fd(), (&value as *const u8).cast(), 1, 0) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EPERM)
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn resident_sealing_rejects_a_retained_writable_mapping() {
        let file = unsealed_resident_file();
        let initial_seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
        let mapping = ResidentMapping::new(&file, BLOCK, true).unwrap();
        assert!(seal_resident(&file).is_err());
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) },
            initial_seals
        );
        drop(mapping);
        seal_resident(&file).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn shared_resident_rows_are_readable_without_first_access_page_faults() {
        let path = std::env::temp_dir().join(format!(
            "helios-engram-populated-{}.bin",
            std::process::id()
        ));
        let rows = 65536;
        let second = rows * 256;
        let bytes: Vec<_> = (0..second + rows * 8)
            .map(|i| (i * 19 + i / 7) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: 0,
                nbytes: second,
                row_bytes: 256,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        let file = open_file(&path).unwrap();
        let original = planes.map(|plane| Arc::new(Resident::load(&file, plane).unwrap()));
        let shared = std::array::from_fn(|i| {
            Arc::new(
                Resident::from_shared(original[i].file.try_clone().unwrap(), planes[i]).unwrap(),
            )
        });
        std::fs::remove_file(path).unwrap();
        let ids: Vec<_> = (0..128).map(|i| ((i * 509 + 17) % rows) as i64).collect();
        let mut output = vec![0; ids.len() * 264];
        let mut expected = Vec::with_capacity(output.len());
        for plane in &planes {
            for &id in &ids {
                let first = plane.offset as usize + id as usize * plane.row_bytes;
                expected.extend_from_slice(&bytes[first..first + plane.row_bytes]);
            }
        }
        for data in [original, shared] {
            let mut reader = Reader::resident(data, Arc::default());
            // Prime the copy code and output storage without touching the selected table pages.
            reader.gather(&planes, &[0], &mut output[..264]).unwrap();
            output.fill(0xA5);
            let mut before = unsafe { std::mem::zeroed::<libc::rusage>() };
            let mut after = unsafe { std::mem::zeroed::<libc::rusage>() };
            assert_eq!(
                unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut before) },
                0
            );
            reader.gather(&planes, &ids, &mut output).unwrap();
            assert_eq!(
                unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut after) },
                0
            );
            assert_eq!(
                after.ru_minflt, before.ru_minflt,
                "row gathering faulted resident pages"
            );
            assert_eq!(after.ru_majflt, before.ru_majflt);
            assert_eq!(output, expected);
        }
    }

    #[test]
    #[cfg(unix)]
    fn shared_resident_plane_outlives_source_and_loader_and_rejects_mutation() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-shared-{}.bin", std::process::id()));
        let bytes: Vec<_> = (0..12345).map(|i| (i * 19 + i / 7) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let plane = Plane {
            file: 0,
            offset: 97,
            nbytes: 8192,
            row_bytes: 128,
        };
        let original = Resident::load(&open_file(&path).unwrap(), plane).unwrap();
        let peer = Resident::from_shared(original.file.try_clone().unwrap(), plane).unwrap();
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            original.file.metadata().unwrap().ino(),
            peer.file.metadata().unwrap().ino()
        );
        assert!(original.file.set_len(1).is_err());
        let value = 255u8;
        assert_eq!(
            unsafe { libc::pwrite(original.descriptor(), (&value as *const u8).cast(), 1, 97) },
            -1
        );
        let wrong = Plane {
            nbytes: plane.nbytes + BLOCK,
            ..plane
        };
        assert!(Resident::from_shared(original.file.try_clone().unwrap(), wrong).is_err());
        std::fs::remove_file(path).unwrap();
        drop(original);
        assert_eq!(peer.bytes(), &bytes[97..97 + 8192]);
    }

    #[test]
    fn resident_load_covers_parallel_chunks_and_rejects_truncated_planes() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-bulk-{}.bin", std::process::id()));
        let rows = LOAD_CHUNK * 5 / 80 + 1;
        let first = 123;
        let second = first + rows * 80 + 37;
        let bytes: Vec<_> = (0..second + rows * 10)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 80,
                row_bytes: 80,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 10,
                row_bytes: 10,
            },
        ];
        let file = open_file(&path).unwrap();
        for plane in planes {
            let data = Resident::load(&file, plane).unwrap();
            let begin = plane.offset as usize;
            assert_eq!(data.bytes(), &bytes[begin..begin + plane.nbytes]);
        }
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((first + LOAD_CHUNK + 17) as u64)
            .unwrap();
        assert!(
            Resident::load(&file, planes[0])
                .err()
                .unwrap()
                .to_string()
                .contains("requires")
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn warm_rows_avoid_disk_and_failed_misses_never_enter_the_cache() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-cache-{}.bin", std::process::id()));
        let rows = 128;
        let first = 4093;
        let second = first + rows * 80 + 7;
        let bytes: Vec<u8> = (0..second + rows * 10)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 80,
                row_bytes: 80,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 10,
                row_bytes: 10,
            },
        ];
        let stats = Arc::new(Stats::default());
        let mut reader = Reader::open(&path, &planes, false, 65536, 32, stats.clone()).unwrap();
        assert!(stats.cache_bytes.load(Ordering::Relaxed) <= 65536);
        fn check(reader: &mut Reader, planes: &[Plane; 2], bytes: &[u8], ids: &[i64]) {
            let mut expected = Vec::new();
            for plane in planes {
                for &id in ids {
                    if id < 0 {
                        expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                    } else {
                        let start = plane.offset as usize + id as usize * plane.row_bytes;
                        expected.extend_from_slice(&bytes[start..start + plane.row_bytes]);
                    }
                }
            }
            let mut output = vec![0xa5; ids.len() * 90];
            reader.gather(planes, ids, &mut output).unwrap();
            assert_eq!(output, expected);
        }
        check(&mut reader, &planes, &bytes, &[2, 9, 2, -1, 17]);
        assert_eq!(stats.misses.load(Ordering::Relaxed), 3);
        assert_eq!(stats.resident_rows.load(Ordering::Relaxed), 3);
        let reads = stats.disk_reads.load(Ordering::Relaxed);
        assert!(reads > 0);
        check(&mut reader, &planes, &bytes, &[17, 9, -1, 2, 17]);
        assert_eq!(stats.hits.load(Ordering::Relaxed), 3);
        assert_eq!(stats.disk_reads.load(Ordering::Relaxed), reads);
        assert!(reader.gather(&planes, &[128], &mut [0; 90]).is_err());
        assert!(reader.gather(&planes, &[-2], &mut [0; 90]).is_err());
        assert!(reader.gather(&planes, &[2; 33], &mut [0; 33 * 90]).is_err());
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();
        check(&mut reader, &planes, &bytes, &[9, 2]);
        assert!(reader.gather(&planes, &[20], &mut [0; 90]).is_err());
        assert_eq!(stats.resident_rows.load(Ordering::Relaxed), 3);
        std::fs::write(&path, &bytes).unwrap();
        check(&mut reader, &planes, &bytes, &[20]);
        assert_eq!(stats.resident_rows.load(Ordering::Relaxed), 4);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn disk_and_ram_gather_identical_rows_across_unaligned_planes() {
        let path = std::env::temp_dir().join(format!("helios-engram-{}.bin", std::process::id()));
        let rows = 4097;
        let first = 4093;
        let second = first + rows * 256 + 7;
        let bytes: Vec<_> = (0..second + rows * 8)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 256,
                row_bytes: 256,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        let ids: Vec<_> = (0..8192)
            .map(|i| {
                if i % 19 == 0 {
                    -1
                } else {
                    (i * 101 % rows) as i64
                }
            })
            .collect();
        let mut expected = Vec::new();
        for plane in &planes {
            for id in &ids {
                if *id == -1 {
                    expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                } else {
                    let offset = plane.offset as usize + *id as usize * plane.row_bytes;
                    expected.extend_from_slice(&bytes[offset..offset + plane.row_bytes]);
                }
            }
        }
        for (resident, cache_bytes) in [(false, 0), (true, 0), (false, 3 << 20)] {
            let mut reader = Reader::open(
                &path,
                &planes,
                resident,
                cache_bytes,
                ids.len(),
                Arc::default(),
            )
            .unwrap();
            for _ in 0..3 {
                let mut actual = vec![0xA5; expected.len()];
                reader.gather(&planes, &ids, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
            assert!(
                reader
                    .gather(&planes, &[rows as i64], &mut [0; 264])
                    .is_err()
            );
        }
        let mut reader = Reader::open(&path, &planes, false, 0, ids.len(), Arc::default()).unwrap();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(4096)
            .unwrap();
        assert!(reader.gather(&planes, &[100], &mut [0; 264]).is_err());
        std::fs::remove_file(&path).unwrap();
    }
    #[test]
    fn equal_offsets_in_separate_files_never_coalesce() {
        let root = std::env::temp_dir().join(format!("helios-engram-files-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: 4093,
                nbytes: 80 * 128,
                row_bytes: 80,
            },
            Plane {
                file: 1,
                offset: 4093,
                nbytes: 10 * 128,
                row_bytes: 10,
            },
        ];
        let mut files = Vec::new();
        for (i, plane) in planes.iter().enumerate() {
            let path = root.join(i.to_string());
            std::fs::write(
                &path,
                vec![i as u8 + 1; plane.offset as usize + plane.nbytes],
            )
            .unwrap();
            files.push(Arc::new(open_file(&path).unwrap()));
        }
        for (resident, cache) in [(false, 0), (false, 65536), (true, 0)] {
            let mut reader =
                Reader::from_files(&files, &planes, resident, cache, 16, Arc::default(), 1, false).unwrap();
            for _ in 0..3 {
                let mut output = vec![0xa5; 4 * 90];
                reader
                    .gather(&planes, &[17, 2, -1, 17], &mut output)
                    .unwrap();
                assert_eq!(&output[..80], &[1; 80]);
                assert_eq!(&output[160..240], &[0; 80]);
                assert_eq!(&output[320..340], &[2; 20]);
                assert_eq!(&output[340..350], &[0; 10]);
                assert_eq!(&output[350..360], &[2; 10]);
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn parallel_gathers_preserve_rows_and_recover_after_truncation() {
        for (width, scale, rows) in [(80, 10, 65536), (256, 8, 65536), (4109, 37, 2048)] {
            let path = std::env::temp_dir().join(format!(
                "flare-worker-rows-{}-{width}.bin",
                std::process::id()
            ));
            let first = 4093;
            let second = first + rows * width + 7;
            let bytes: Vec<_> = (0..second + rows * scale)
                .map(|i| (i * 13 + i / 257) as u8)
                .collect();
            std::fs::write(&path, &bytes).unwrap();
            let planes = [
                Plane {
                    file: 0,
                    offset: first as u64,
                    nbytes: rows * width,
                    row_bytes: width,
                },
                Plane {
                    file: 0,
                    offset: second as u64,
                    nbytes: rows * scale,
                    row_bytes: scale,
                },
            ];
            let ids: Vec<i64> = (0..1025)
                .map(|i| {
                    if i % 17 == 0 {
                        -1
                    } else if i % 19 == 0 {
                        113
                    } else {
                        ((i * 113) % rows) as i64
                    }
                })
                .collect();
            let mut baseline_reads = None;
            for workers in [1, 2, 4, 8] {
                let stats = Arc::new(Stats::default());
                let file = Arc::new(File::open(&path).unwrap());
                let mut reader = Reader::from_files(
                    std::slice::from_ref(&file),
                    &planes,
                    false,
                    0,
                    ids.len(),
                    stats.clone(),
                    workers,
                    false,
                )
                .unwrap();
                drop(file);
                let check = |reader: &mut Reader, ids: &[i64]| {
                    let mut expected = Vec::new();
                    for plane in planes {
                        for &id in ids {
                            if id < 0 {
                                expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                            } else {
                                let begin = plane.offset as usize + id as usize * plane.row_bytes;
                                expected.extend_from_slice(&bytes[begin..begin + plane.row_bytes]);
                            }
                        }
                    }
                    let mut guarded = vec![0xa5; expected.len() + 64];
                    reader
                        .gather(&planes, ids, &mut guarded[32..32 + expected.len()])
                        .unwrap();
                    assert_eq!(&guarded[32..32 + expected.len()], expected);
                    assert!(
                        guarded[..32]
                            .iter()
                            .chain(&guarded[32 + expected.len()..])
                            .all(|b| *b == 0xa5)
                    );
                };
                for count in [0, 1, 77, 1025, 5, 1025] {
                    check(&mut reader, &ids[..count]);
                }
                let count = stats.disk_reads.load(Ordering::Relaxed);
                if let Some(expected) = baseline_reads {
                    assert_eq!(count, expected);
                } else {
                    baseline_reads = Some(count);
                }
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_len(0)
                    .unwrap();
                let mut output = vec![0; ids.len() * (width + scale)];
                assert!(reader.gather(&planes, &ids, &mut output).is_err());
                std::fs::write(&path, &bytes).unwrap();
                check(&mut reader, &ids);
                check(&mut reader, &[rows as i64 - 1, -1, 0, rows as i64 - 1]);
            }
            std::fs::remove_file(path).unwrap();
        }
    }
    #[test]
    fn parallel_shard_gathers_keep_file_indices_and_recover_after_truncation() {
        let rows = 2048;
        let width = 4109;
        let offset = 4093;
        let planes = std::array::from_fn(|file| Plane {
            file,
            offset: offset as u64,
            nbytes: rows * width,
            row_bytes: width,
        });
        let paths: Vec<_> = (0..2)
            .map(|file| {
                std::env::temp_dir().join(format!(
                    "flare-ring-shard-{}-{file}.bin",
                    std::process::id()
                ))
            })
            .collect();
        let bytes: Vec<Vec<u8>> = (0..2)
            .map(|file| {
                (0..offset + rows * width)
                    .map(|i| ((i * 19 + i / 37 + file * 83) % 251) as u8)
                    .collect()
            })
            .collect();
        for (path, bytes) in paths.iter().zip(&bytes) {
            std::fs::write(path, bytes).unwrap();
        }
        let ids: Vec<i64> = (0..257)
            .map(|i| match i % 31 {
                0 => -1,
                1 => 42,
                _ => ((i * 7) % rows) as i64,
            })
            .collect();
        let mut baseline_reads = None;
        for workers in [1, 2, 4, 8] {
            let files: Vec<_> = paths
                .iter()
                .map(|path| Arc::new(open_file(path).unwrap()))
                .collect();
            let stats = Arc::new(Stats::default());
            let mut reader =
                Reader::from_files(&files, &planes, false, 0, ids.len(), stats.clone(), workers, false)
                    .unwrap();
            drop(files);
            let check = |reader: &mut Reader, ids: &[i64]| {
                let mut expected = Vec::new();
                for plane in planes {
                    for &id in ids {
                        if id < 0 {
                            expected.extend(std::iter::repeat_n(0, width));
                        } else {
                            let start = offset + id as usize * width;
                            expected.extend_from_slice(&bytes[plane.file][start..start + width]);
                        }
                    }
                }
                let mut guarded = vec![0xa5; expected.len() + 64];
                reader
                    .gather(&planes, ids, &mut guarded[32..32 + expected.len()])
                    .unwrap();
                assert_eq!(&guarded[32..32 + expected.len()], expected);
                assert!(
                    guarded[..32]
                        .iter()
                        .chain(&guarded[32 + expected.len()..])
                        .all(|b| *b == 0xa5)
                );
            };
            for count in [0, 3, 257, 17, 257] {
                check(&mut reader, &ids[..count]);
            }
            let count = stats.disk_reads.load(Ordering::Relaxed);
            assert!(count > 2 * DEPTH as u64);
            if let Some(expected) = baseline_reads {
                assert_eq!(count, expected);
            } else {
                baseline_reads = Some(count);
            }
            std::fs::OpenOptions::new()
                .write(true)
                .open(&paths[1])
                .unwrap()
                .set_len(0)
                .unwrap();
            assert!(
                reader
                    .gather(&planes, &ids, &mut vec![0; ids.len() * width * 2])
                    .is_err()
            );
            std::fs::write(&paths[1], &bytes[1]).unwrap();
            check(&mut reader, &ids);
            check(&mut reader, &[rows as i64 - 1, -1, 0]);
        }
        for path in paths {
            std::fs::remove_file(path).unwrap();
        }
    }
}
