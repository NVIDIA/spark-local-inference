/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Immutable resident tables shared across preparation-owned rank processes.

use super::{Binding, Index, Plane, Shared, Table, reader};
use crate::{
    Result,
    execution::{Dtype, ExecutionManifest},
};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    os::fd::FromRawFd,
    path::PathBuf,
    sync::Arc,
};

/// A checkpoint range containing consecutive complete rows of a table plane.
#[derive(Clone, Debug)]
pub struct SourceSegment {
    pub path: PathBuf,
    pub offset: u64,
    pub nbytes: usize,
}

/// Ordered checkpoint ranges assembled into one resident plane without conversion.
#[derive(Clone, Debug)]
pub struct PlaneSource {
    pub segments: Vec<SourceSegment>,
    pub row_bytes: usize,
}

/// Table geometry and the prepared buffers consumed by native row delivery.
#[derive(Clone, Debug)]
pub struct TableSource {
    pub planes: [PlaneSource; 2],
    pub heads: usize,
    pub weights: String,
    pub scales: String,
    pub hashes: String,
    pub event: String,
    pub request: Option<String>,
}

/// Read-only RAM planes. Shared descriptors keep one physical copy across ranks;
/// delivery staging and CUDA events belong to each rank's native binding.
pub struct PreparedTables {
    index: Arc<Index>,
    data: Vec<[Arc<reader::Resident>; 2]>,
}

impl PreparedTables {
    fn index(vocabulary: usize, sources: &[TableSource]) -> Result<Arc<Index>> {
        if vocabulary == 0 || sources.is_empty() {
            return Err("embedding catalog requires a vocabulary and tables".into());
        }
        let mut names = HashSet::new();
        let mut events = HashSet::new();
        let mut tables = Vec::new();
        let mut total_bytes = 0usize;
        for source in sources {
            if source.heads == 0
                || source.heads > 64
                || source.event.is_empty()
                || !events.insert(source.event.clone())
                || [&source.weights, &source.scales, &source.hashes]
                    .into_iter()
                    .chain(source.request.iter())
                    .any(|name| name.is_empty() || !names.insert(name.clone()))
            {
                return Err("invalid embedding table bindings or head count".into());
            }
            let mut planes = Vec::new();
            let mut rows = None;
            for plane in &source.planes {
                if plane.row_bytes == 0 || plane.segments.is_empty() {
                    return Err("invalid embedding table source plane".into());
                }
                let mut nbytes = 0usize;
                for segment in &plane.segments {
                    if segment.path.as_os_str().is_empty()
                        || segment.nbytes == 0
                        || segment.nbytes % plane.row_bytes != 0
                        || segment
                            .offset
                            .checked_add(segment.nbytes as u64)
                            .is_none_or(|end| end > i64::MAX as u64)
                    {
                        return Err("invalid embedding table source segment".into());
                    }
                    nbytes = nbytes
                        .checked_add(segment.nbytes)
                        .ok_or("embedding source plane exceeds host range")?;
                }
                let count = nbytes / plane.row_bytes;
                total_bytes = total_bytes
                    .checked_add(nbytes)
                    .ok_or("resident embedding catalog exceeds host range")?;
                if rows
                    .replace(count)
                    .is_some_and(|previous| previous != count)
                {
                    return Err(
                        "embedding weight and scale planes have different row counts".into(),
                    );
                }
                planes.push(Plane {
                    file: 0,
                    offset: 0,
                    nbytes,
                    row_bytes: plane.row_bytes,
                });
            }
            tables.push(Table {
                planes: [planes[0], planes[1]],
                heads: source.heads,
                weights: source.weights.clone(),
                scales: source.scales.clone(),
                hashes: source.hashes.clone(),
                event: source.event.clone(),
                request: source.request.clone(),
            });
        }
        Ok(Arc::new(Index { tables }))
    }

    /// Load checkpoint spans once into sealed RAM-backed files. Serving never
    /// opens or reads the checkpoint files after this constructor succeeds.
    pub fn load(vocabulary: usize, sources: &[TableSource]) -> Result<Self> {
        let index = Self::index(vocabulary, sources)?;
        let mut data = Vec::new();
        for (source, table) in sources.iter().zip(&index.tables) {
            let mut planes = Vec::new();
            for (source, plane) in source.planes.iter().zip(table.planes) {
                let segments = source
                    .segments
                    .iter()
                    .map(|segment| {
                        let file = File::open(&segment.path)
                            .map_err(|error| format!("open embedding source segment: {error}"))?;
                        Ok((file, segment.offset, segment.nbytes))
                    })
                    .collect::<Result<Vec<_>>>()?;
                planes.push(Arc::new(reader::Resident::load_segments(plane, &segments)?));
            }
            data.push([planes.remove(0), planes.remove(0)]);
        }
        Ok(Self { index, data })
    }

    /// Map shared resident planes received during rank setup. The caller keeps
    /// ownership of its descriptors; this object duplicates and retains them.
    pub fn from_shared(
        vocabulary: usize,
        sources: &[TableSource],
        descriptors: &[i32],
    ) -> Result<Self> {
        let index = Self::index(vocabulary, sources)?;
        if descriptors.len() != index.tables.len() * 2 {
            return Err("shared embedding descriptor count differs from its catalog".into());
        }
        let mut data = Vec::new();
        for (table, handles) in index.tables.iter().zip(descriptors.chunks_exact(2)) {
            let mut planes = Vec::new();
            for (plane, descriptor) in table.planes.into_iter().zip(handles) {
                let owned = unsafe { libc::fcntl(*descriptor, libc::F_DUPFD_CLOEXEC, 0) };
                if owned < 0 {
                    return Err(format!(
                        "duplicate resident embedding descriptor: {}",
                        std::io::Error::last_os_error()
                    )
                    .into());
                }
                let file = unsafe { File::from_raw_fd(owned) };
                planes.push(Arc::new(reader::Resident::from_shared(file, plane)?));
            }
            data.push([planes.remove(0), planes.remove(0)]);
        }
        Ok(Self { index, data })
    }

    /// Borrowed file descriptors, ordered by table and then weight/scale plane.
    pub fn descriptors(&self) -> Vec<i32> {
        self.data
            .iter()
            .flat_map(|planes| planes.iter().map(|plane| plane.descriptor()))
            .collect()
    }

    pub fn nbytes(&self) -> usize {
        self.index
            .tables
            .iter()
            .flat_map(|table| table.planes.iter())
            .map(|plane| plane.nbytes)
            .sum()
    }

    /// Attach process-local native delivery to already captured graph buffers.
    /// Every diagnostic rank has its own staging and completion accounting;
    /// only the immutable resident table bytes are shared across processes.
    pub fn bind(
        &self,
        manifest: &ExecutionManifest,
        events: &HashMap<String, usize>,
    ) -> Result<Binding> {
        manifest.validate()?;
        let tokens = manifest
            .graphs
            .iter()
            .filter(|graph| super::consumes(graph.kind))
            .map(|graph| graph.packed_tokens())
            .max()
            .ok_or("embedding catalog lacks consumer graphs")?;
        let mut handles = Vec::new();
        for table in &self.index.tables {
            let handle = events
                .get(&table.event)
                .copied()
                .filter(|handle| *handle != 0)
                .ok_or("prepared embedding readiness event is missing")?;
            handles.push(handle);
            let rows = tokens
                .checked_mul(table.heads)
                .ok_or("embedding row capacity exceeds host range")?;
            for (name, dtype, height, width) in [
                (
                    &table.weights,
                    Dtype::Uint8,
                    rows,
                    table.planes[0].row_bytes,
                ),
                (&table.scales, Dtype::Uint8, rows, table.planes[1].row_bytes),
                (&table.hashes, Dtype::Int64, tokens, table.heads),
            ] {
                let buffer = manifest.buffer(name)?;
                if buffer.dtype != dtype
                    || buffer.shape.len() != 2
                    || buffer.shape[0] < height
                    || buffer.shape[1] != width
                    || buffer.strides != [width, 1]
                {
                    return Err(
                        "prepared embedding buffer geometry differs from its catalog".into(),
                    );
                }
            }
            if let Some(name) = &table.request {
                let request = manifest.buffer(name)?;
                if request.dtype != Dtype::Int64
                    || request.shape.len() != 1
                    || request.shape[0] <= rows
                    || request.strides != [1]
                    || request.address % 8 != 0
                {
                    return Err("invalid prepared embedding request mailbox".into());
                }
            } else if manifest
                .graphs
                .iter()
                .filter(|g| super::consumes(g.kind))
                .any(|graph| {
                    !graph.input_bindings.iter().any(|binding| {
                        binding.buffer_name == table.hashes
                            && binding.nbytes >= graph.packed_tokens() * table.heads * 8
                    })
                })
            {
                return Err("prepared embedding graph lacks its hash input binding".into());
            }
        }
        let stats = Arc::new(super::Stats::default());
        let readers = self
            .data
            .iter()
            .map(|data| reader::Reader::resident(data.clone(), stats.clone()))
            .collect();
        Ok(Binding {
            shared: Shared::with_readers(
                self.index.clone(),
                readers,
                1,
                tokens,
                manifest.context,
                manifest.device,
                stats,
            )?,
            rank: 0,
            events: handles,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separate_checkpoint_planes_are_shared_without_reopening_sources() {
        let root =
            std::env::temp_dir().join(format!("helios-embedding-sources-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let first = root.join("weights.bin");
        let second = root.join("scales.bin");
        let weights: Vec<_> = (0..9000).map(|i| (i * 13 + (i / 256) * 17) as u8).collect();
        let scales: Vec<_> = (0..1000).map(|i| (i * 17) as u8).collect();
        std::fs::write(&first, &weights).unwrap();
        std::fs::write(&second, &scales).unwrap();
        let sources = [TableSource {
            planes: [
                PlaneSource {
                    segments: vec![
                        SourceSegment {
                            path: first.clone(),
                            offset: 37 + 4096,
                            nbytes: 4096,
                        },
                        SourceSegment {
                            path: first,
                            offset: 37,
                            nbytes: 4096,
                        },
                    ],
                    row_bytes: 256,
                },
                PlaneSource {
                    segments: vec![
                        SourceSegment {
                            path: second.clone(),
                            offset: 43 + 128,
                            nbytes: 128,
                        },
                        SourceSegment {
                            path: second,
                            offset: 43,
                            nbytes: 128,
                        },
                    ],
                    row_bytes: 8,
                },
            ],
            heads: 24,
            weights: "weights".into(),
            scales: "scales".into(),
            hashes: "hashes".into(),
            event: "ready".into(),
            request: Some("requests".into()),
        }];
        let owner = PreparedTables::load(129280, &sources).unwrap();
        assert_eq!(owner.nbytes(), 8448);
        let peer = PreparedTables::from_shared(129280, &sources, &owner.descriptors()).unwrap();
        assert!(PreparedTables::from_shared(129280, &sources, &[-1, -1]).is_err());
        assert!(PreparedTables::from_shared(129280, &sources, &owner.descriptors()[..1]).is_err());
        std::fs::remove_dir_all(root).unwrap();
        drop(owner);
        let mut reader = reader::Reader::resident(peer.data[0].clone(), Arc::default());
        let mut output = vec![0; 3 * (256 + 8)];
        reader
            .gather(&peer.index.tables[0].planes, &[31, -1, 0], &mut output)
            .unwrap();
        assert_eq!(&output[..256], &weights[37 + 15 * 256..37 + 16 * 256]);
        assert_eq!(&output[256..512], &[0; 256]);
        assert_eq!(&output[512..768], &weights[37 + 16 * 256..37 + 17 * 256]);
        assert_eq!(&output[768..776], &scales[43 + 15 * 8..43 + 16 * 8]);
        assert_eq!(&output[776..784], &[0; 8]);
        assert_eq!(&output[784..], &scales[43 + 16 * 8..43 + 17 * 8]);
        assert!(
            reader
                .gather(&peer.index.tables[0].planes, &[32], &mut vec![0; 264])
                .is_err()
        );
    }
}
