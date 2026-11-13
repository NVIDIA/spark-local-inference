/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Preparation-only registration of shared RAM tables for native graph execution.

use crate::engram::Binding;
#[cfg(unix)]
use crate::engram::{PlaneSource, PreparedTables, SourceSegment, TableSource};
#[cfg(unix)]
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::{exceptions::PyKeyError, prelude::*};
#[cfg(unix)]
use std::{collections::HashMap, path::PathBuf};

#[cfg(unix)]
fn sources(value: &Bound<'_, PyAny>) -> PyResult<(usize, Vec<TableSource>)> {
    let vocabulary = value.get_item("vocabulary")?.extract()?;
    let mut tables = Vec::new();
    for table in value.get_item("tables")?.try_iter()? {
        let table = table?;
        let mut planes = Vec::new();
        for plane in table.get_item("planes")?.try_iter()? {
            let plane = plane?;
            let mut segments = Vec::new();
            for segment in plane.get_item("segments")?.try_iter()? {
                let segment = segment?;
                segments.push(SourceSegment {
                    path: PathBuf::from(segment.get_item("path")?.extract::<String>()?),
                    offset: segment.get_item("offset")?.extract()?,
                    nbytes: segment.get_item("nbytes")?.extract()?,
                });
            }
            planes.push(PlaneSource {
                segments,
                row_bytes: plane.get_item("row_bytes")?.extract()?,
            });
        }
        if planes.len() != 2 {
            return Err(PyValueError::new_err(
                "embedding table requires weight and scale planes",
            ));
        }
        tables.push(TableSource {
            planes: [planes.remove(0), planes.remove(0)],
            heads: table.get_item("heads")?.extract()?,
            weights: table.get_item("weights")?.extract()?,
            scales: table.get_item("scales")?.extract()?,
            hashes: table.get_item("hashes")?.extract()?,
            event: table.get_item("event")?.extract()?,
            request: table.get_item("request")?.extract()?,
        });
    }
    Ok((vocabulary, tables))
}

#[pyclass(name = "EmbeddingTables", module = "flare_heliopause")]
pub(crate) struct NativeEmbeddingTables {
    #[cfg(unix)]
    tables: PreparedTables,
}

#[pymethods]
impl NativeEmbeddingTables {
    #[new]
    #[pyo3(signature = (catalog, *, descriptors=None))]
    fn new(
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        descriptors: Option<Vec<i32>>,
    ) -> PyResult<Self> {
        #[cfg(unix)]
        {
            let (vocabulary, sources) = sources(catalog)?;
            let tables = py
                .detach(|| match descriptors {
                    Some(descriptors) => {
                        PreparedTables::from_shared(vocabulary, &sources, &descriptors)
                    }
                    None => PreparedTables::load(vocabulary, &sources),
                })
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            Ok(Self { tables })
        }
        #[cfg(windows)]
        {
            let _ = (py, catalog, descriptors);
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "shared RAM table handoff for ignition debug-serve requires Linux; load an exported core for native RAM serving",
            ))
        }
    }

    #[cfg(unix)]
    #[getter]
    fn descriptors(&self) -> Vec<i32> {
        self.tables.descriptors()
    }

    #[cfg(unix)]
    #[getter]
    fn nbytes(&self) -> usize {
        self.tables.nbytes()
    }

    #[cfg(unix)]
    fn bind(
        &self,
        py: Python<'_>,
        manifest: &Bound<'_, PyAny>,
        events: HashMap<String, usize>,
    ) -> PyResult<NativeEmbeddingBinding> {
        let manifest = crate::python::execution_manifest(manifest)?;
        if manifest.engram.is_some() {
            return Err(PyValueError::new_err(
                "prepared execution already has embedding tables",
            ));
        }
        let binding = py
            .detach(|| self.tables.bind(&manifest, &events))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(NativeEmbeddingBinding { binding })
    }
}

#[pyclass(name = "EmbeddingBinding", module = "flare_heliopause")]
pub(crate) struct NativeEmbeddingBinding {
    binding: Binding,
}

pub(crate) fn binding(value: &Bound<'_, PyAny>) -> PyResult<Option<Binding>> {
    match value.get_item("engram") {
        Ok(value) if value.is_none() => Ok(None),
        Ok(value) => Ok(Some(
            value
                .extract::<PyRef<'_, NativeEmbeddingBinding>>()?
                .binding
                .clone(),
        )),
        Err(error) if error.is_instance_of::<PyKeyError>(value.py()) => Ok(None),
        Err(error) => Err(error),
    }
}
