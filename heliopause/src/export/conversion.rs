/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Convert and re-sign payload-format packages without CUDA initialization.
use super::{ExportResult as Result, invalid};
use super::protection::PublisherRequest;
#[cfg(test)]
use super::protection::Publisher;
use pyo3::{exceptions::PyValueError, prelude::*};
use std::{collections::BTreeSet, path::Path, time::SystemTime};

/// Report which canonical form a package uses, or `legacy` for the blob format.
#[pyfunction]
fn export_payload_form(py: Python<'_>, source: String) -> PyResult<String> {
    py.detach(|| {
        let form = helios_core::package::conversion::form(Path::new(&source))?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
            match form {
                None => "legacy",
                Some(helios_core::package::conversion::Form::Packed) => "packed",
                Some(helios_core::package::conversion::Form::Directory) => "directory",
            }
            .to_string(),
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Convert a payload-format package between its packed and directory forms.
///
/// This is keyless: signed bodies, signatures, compressed executables and payload bytes are
/// copied exactly; only the unsigned location tensor is re-encoded.
#[pyfunction]
fn export_payload_convert(
    py: Python<'_>,
    source: String,
    destination: String,
    form: String,
) -> PyResult<String> {
    py.detach(|| {
        let (source, destination) = (Path::new(&source), Path::new(&destination));
        // Verification and copying of a large package take minutes; report them as the
        // export does, on standard error, and keep standard output for the result lines.
        let progress = helios_core::startup::StartupProgress::stderr_rows();
        match form.as_str() {
            "packed" => helios_core::package::conversion::pack_with(source, destination, None, &progress)?,
            "directory" => {
                helios_core::package::conversion::unpack_with(source, destination, None, &progress)?
            }
            other => {
                return Err(super::invalid(format!(
                    "unknown payload package form {other:?}"
                )));
            }
        }
        let bytes = packed_bytes(destination);
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
            serde_json::json!({
                "form": form,
                "destination": destination.display().to_string(),
                "bytes": bytes,
            })
            .to_string(),
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
fn packed_bytes(path: &Path) -> u64 {
    let Ok(meta) = std::fs::metadata(path) else {
        return 0;
    };
    if meta.is_file() {
        return meta.len();
    }
    let mut total = 0u64;
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        total = total.saturating_add(packed_bytes(&entry.path()));
    }
    total
}

#[derive(serde::Deserialize)]
struct PublicProfile {
    #[serde(default)]
    name: String,
    namespace: String,
    public_key: Vec<u8>,
    #[serde(default)]
    development: bool,
}

pub(crate) fn payload_publishers(profiles: &str) -> Result<Vec<helios_core::runtime_package::Publisher>> {
    let profiles: Vec<PublicProfile> = serde_json::from_str(profiles)?;
    profiles
        .into_iter()
        .map(|profile| {
            Ok(helios_core::runtime_package::Publisher {
                name: if profile.name.is_empty() {
                    profile.namespace.clone()
                } else {
                    profile.name
                },
                namespace: profile.namespace,
                public_key: profile
                    .public_key
                    .as_slice()
                    .try_into()
                    .map_err(|_| super::invalid("publisher public key must be 32 bytes"))?,
                development: profile.development,
            })
        })
        .collect()
}

/// Combine independently prepared TP packages into one signed package.
#[pyfunction]
fn export_payload_combine(
    py: Python<'_>,
    sources: Vec<String>,
    destination: String,
    publisher: String,
    profiles: String,
    accepted: String,
    target_accepted: String,
    form: String,
) -> PyResult<String> {
    let request: PublisherRequest =
        serde_json::from_str(&publisher).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let publishers =
        payload_publishers(&profiles).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if form != "directory" && form != "packed" {
        return Err(PyValueError::new_err(format!(
            "unknown package form {form:?}"
        )));
    }
    py.detach(|| {
        use helios_core::package::conversion;
        let source_paths: Vec<std::path::PathBuf> =
            sources.iter().map(std::path::PathBuf::from).collect();
        let publisher = request.publisher()?;
        let destination = Path::new(&destination);
        // Packed output is published to native staging then converted by the
        // shared keyless container writer; Python never touches the tree.
        let published = if form == "packed" {
            let parent = destination.parent().unwrap_or(Path::new("."));
            std::fs::create_dir_all(parent)?;
            let staging = tempfile::Builder::new()
                .prefix(".flare-packed-")
                .tempdir_in(parent)?;
            let directory = staging.path().join("package");
            let published = conversion::combine(&conversion::CombineRequest {
                sources: &source_paths,
                output: &directory,
                publishers: &publishers,
                accepted: &accepted,
                target_accepted: &target_accepted,
                target_publisher: None,
                key: publisher.signing_key(),
            })?;
            conversion::pack(&published.directory, destination)?;
            published
        } else {
            conversion::combine(&conversion::CombineRequest {
                sources: &source_paths,
                output: destination,
                publishers: &publishers,
                accepted: &accepted,
                target_accepted: &target_accepted,
                target_publisher: None,
                key: publisher.signing_key(),
            })?
        };
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
            serde_json::json!({
                "form": form,
                "destination": destination.display().to_string(),
                "payloads": published.payloads.len(),
            })
            .to_string(),
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Re-sign one payload package under a fresh identity, recompressing executables.
#[pyfunction]
fn export_payload_promote(
    py: Python<'_>,
    source: String,
    destination: String,
    publisher: String,
    profiles: String,
    accepted: String,
    target_accepted: String,
    target_publisher: Option<String>,
) -> PyResult<String> {
    let request: PublisherRequest =
        serde_json::from_str(&publisher).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let publishers =
        payload_publishers(&profiles).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let target_publisher: Option<helios_core::package::conversion::TargetPublisher> =
        target_publisher
            .map(|value| serde_json::from_str(&value))
            .transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    py.detach(|| {
        let publisher = request.publisher()?;
        let published = helios_core::package::conversion::promote(
            Path::new(&source),
            Path::new(&destination),
            &publishers,
            &accepted,
            &target_accepted,
            target_publisher.as_ref(),
            publisher.signing_key(),
        )?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
            serde_json::json!({
                "form": "directory",
                "destination": published.directory.display().to_string(),
                "payloads": published.payloads.len(),
            })
            .to_string(),
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Install a payload package into the shared cache as a committed leased entry,
/// optionally exporting that entry to an exact destination path.
///
/// Cache entries use the directory canonical form. Packed input and a primary
/// file path are accepted directly; no Python or intermediate tree is created.
#[pyfunction]
fn export_payload_convert_cache(
    py: Python<'_>,
    source: String,
    cache: String,
    destination: String,
    profiles: String,
    form: String,
) -> PyResult<String> {
    py.detach(|| convert_cache(&source, &cache, &destination, &profiles, &form))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Install one payload package into the shared cache without selecting it until
/// the optional standalone export has succeeded.
fn convert_cache(
    source: &str,
    cache: &str,
    destination: &str,
    profiles: &str,
    form: &str,
) -> Result<String> {
    convert_cache_at(
        source,
        cache,
        destination,
        profiles,
        form,
        SystemTime::now(),
    )
}

fn convert_cache_at(
    source: &str,
    cache: &str,
    destination: &str,
    profiles: &str,
    form: &str,
    at: SystemTime,
) -> Result<String> {
    let publishers = payload_publishers(profiles)?;
    if form != "directory" {
        return Err(invalid("cache entries use the directory canonical form"));
    }
    use helios_core::package::cache::{PackageSource, Root};
    use helios_core::package::protection::Selection;
    let root = Root::open(cache)?;
    let metadata = helios_core::package::reader::Metadata::open(Path::new(source))?;
    let envelope = helios_core::package::protection::envelope(&metadata.envelope)?;
    let mut selections = Vec::with_capacity(envelope.deployments.len());
    for index in 0..envelope.deployments.len() {
        selections.push(Selection::Deployment {
            deployment: index,
            rank: None,
        });
    }
    if selections.is_empty() {
        return Err(invalid("payload package has no deployment"));
    }
    let mut package_source = PackageSource::open(Path::new(source), &publishers, selections[0])?;
    if selections.len() > 1 {
        package_source = package_source.union(&publishers, &selections[1..])?;
    }
    let model = package_source
        .verified()
        .first()
        .map(|verified| verified.package().model_identity.as_str())
        .filter(|identity| !identity.is_empty())
        .ok_or_else(|| invalid("payload package has no model identity"))?;
    let model = helios_core::package::cache::model_key(model)?;
    let candidate = converted_package_candidate(package_source.verified(), at)?;
    // Reserve the readable name before installation. The reservation stays held
    // through commit, so same-minute conversions receive numeric suffixes rather
    // than racing a caller-chosen entry string.
    let staging = root.stage_named(&model, &candidate)?;
    let plan = root.plan_acquisition(
        package_source,
        &helios_core::package::derivation::Limits::default(),
    )?;
    // Commit the entry without selecting it: a standalone export may still
    // fail, and a failed conversion must leave the prior default untouched.
    let committed = root.install_planned_named(plan, staging)?;
    if !destination.is_empty() {
        if let Err(error) = root.export_entry(&committed, Path::new(destination)) {
            let entry_id = committed.id().to_string();
            let model_key = committed.model().to_string();
            drop(committed);
            return match root.remove(&model_key, &entry_id, false) {
                Ok(()) => Err(error.into()),
                Err(cleanup) => Err(invalid(format!(
                    "{error}; the failed conversion left cache entry {model_key}@{entry_id}, which could not be removed: {cleanup}; remove it with `flare models remove`"
                ))),
            };
        }
    }
    // The entry is complete; this is the only step that selects it. A failure
    // here may already be visible, so no rollback of another caller's ref.
    if let Err(error) = root.publish_entry(&committed) {
        let entry_id = committed.id().to_string();
        let model_key = committed.model().to_string();
        drop(committed);
        return Err(invalid(format!(
            "publishing the default for completed cache entry {model_key}@{entry_id} failed: {error}; the entry is retained and the ref update may already be visible; verify with `flare models list`"
        )));
    }
    Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
        serde_json::json!({
            "form": "directory",
            "model": committed.model(),
            "entry": committed.id(),
            "entry_path": committed.directory().display().to_string(),
            "destination": destination,
        })
        .to_string(),
    )
}

/// Name a converted package from its authenticated deployment identities.
fn converted_package_candidate(
    verified: &[helios_core::package::protection::VerifiedPackage],
    at: SystemTime,
) -> Result<String> {
    use helios_core::package::naming;
    let mut sizes = BTreeSet::new();
    let mut platform: Option<&str> = None;
    for package in verified {
        let deployment = package
            .deployment()
            .ok_or_else(|| invalid("converted package lacks a deployment"))?;
        if deployment.tensor_parallel == 0 {
            return Err(invalid("converted package has a zero tensor-parallel size"));
        }
        sizes.insert(deployment.tensor_parallel);
        let hardware = package.package().hardware_identity.as_str();
        if naming::PLATFORMS.contains(&hardware) {
            if let Some(previous) = platform {
                if previous != hardware {
                    return Err(invalid(format!(
                        "converted deployments span {previous} and {hardware}; names require one device identity"
                    )));
                }
            } else {
                platform = Some(hardware);
            }
        }
    }
    let sizes: Vec<u32> = sizes.into_iter().collect();
    helios_core::package::naming::deployment_candidate(platform, &sizes, at)
        .ok_or_else(|| invalid("converted package naming requires a tensor-parallel size"))
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(export_payload_form, module)?)?;
    module.add_function(wrap_pyfunction!(export_payload_convert, module)?)?;
    module.add_function(wrap_pyfunction!(export_payload_combine, module)?)?;
    module.add_function(wrap_pyfunction!(export_payload_promote, module)?)?;
    module.add_function(wrap_pyfunction!(export_payload_convert_cache, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Publish one signed payload package whose model identity is `t/model`.
    fn payload_package(root: &std::path::Path) -> (std::path::PathBuf, Publisher) {
        use helios_core::package::{PayloadId, container, export as payload_export, schema::wire};
        let publisher = Publisher::from_seed(
            &[7u8; 32],
            "t".into(),
            "https://huggingface.co/t/model".into(),
            false,
            None,
        )
        .unwrap();
        let directory = root.join("package");
        std::fs::create_dir_all(&directory).unwrap();
        let weights = b"convert cache weights".to_vec();
        let weights_id = PayloadId::of(&weights);
        let weights_path = root.join("weights.bin");
        std::fs::write(&weights_path, &weights).unwrap();
        let graph = root.join("graph.bin");
        std::fs::write(&graph, b"graph-bytes").unwrap();
        let runtime = helios_core::runtime_package::wire::ServingContract {
            model_id: "t/model".into(),
            identity: "t/model".into(),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(helios_core::runtime_package::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![helios_core::runtime_package::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let rank = wire::RankBody {
            rank: 0,
            deployment_id: "sm120-tp1".into(),
            context: Some(Default::default()),
            allocations: vec![wire::Allocation {
                name: "device-0".into(),
                nbytes: weights.len() as u64,
                alignment: 4096,
                ..Default::default()
            }],
            initializers: vec![wire::Initializer {
                allocation: "device-0".into(),
                destination_offset: 0,
                source: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: weights_id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: weights.len() as u64,
                    }),
                    role: wire::PayloadRole::Weights as i32,
                }),
            }],
            graphs: vec![wire::Graph {
                name: "decode".into(),
                resource: 0,
                node_count: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let request = payload_export::PublishRequest {
            package: wire::PackageBody {
                model_identity: "t/model".into(),
                hardware_identity: "sm120".into(),
                frontend: Some(helios_core::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    helios_core::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "t".into(),
                    official_source: "https://huggingface.co/t/model".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![payload_export::PublishDeployment {
                payloads: vec![],
                body: wire::DeploymentBody {
                    deployment_id: "sm120-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(runtime),
                    ..Default::default()
                },
                ranks: vec![payload_export::PublishRank {
                    body: rank,
                    payloads: vec![payload_export::PublishPayload {
                        role: wire::PayloadRole::Weights,
                        relative: "weights/layer-000.safetensors".into(),
                        tensor: Some(container::COMPANION_TENSOR.into()),
                        source: container::Slice::new(
                            std::sync::Arc::new(std::fs::File::open(&weights_path).unwrap()),
                            0,
                            weights.len() as u64,
                        )
                        .unwrap(),
                        id: weights_id,
                    }],
                    resources: vec![payload_export::PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph,
                        nbytes: 11,
                    }],
                }],
            }],
            package_id: [51u8; 32],
            public_key: publisher.public_key().try_into().unwrap(),
            accepted: "t".into(),
            no_weights: false,
        };
        payload_export::publish(
            &request,
            publisher.signing_key(),
            payload_export::Destination::Directory(&directory),
        )
        .unwrap();
        (directory.join("model.safetensors"), publisher)
    }

    fn profiles(publisher: &Publisher) -> String {
        serde_json::json!([{
            "name": "t",
            "namespace": "t",
            "public_key": publisher.public_key(),
            "development": false,
        }])
        .to_string()
    }

    fn fixed_time() -> SystemTime {
        std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_792_228_980)
    }

    #[test]
    fn cache_conversion_selects_only_after_a_successful_standalone_export() {
        use helios_core::package::cache::{PackageSource, Root};
        let scratch = tempfile::tempdir().unwrap();
        let (source, publisher) = payload_package(scratch.path());
        let cache = scratch.path().join("cache");
        let model = helios_core::package::cache::model_key("t/model").unwrap();
        let publishers = payload_publishers(&profiles(&publisher)).unwrap();
        let selection = helios_core::package::protection::Selection::Deployment {
            deployment: 0,
            rank: None,
        };
        {
            // The prior default is a verified payload entry.
            let source_package = PackageSource::open(&source, &publishers, selection).unwrap();
            let root = Root::open(&cache).unwrap();
            root.install_package(source_package, &model, "prior")
                .unwrap();
        }

        // A failed standalone export must not select the new entry or void the old one.
        let occupied = scratch.path().join("occupied");
        std::fs::create_dir_all(&occupied).unwrap();
        std::fs::write(occupied.join("keep"), b"keep").unwrap();
        let error = convert_cache_at(
            &source.display().to_string(),
            &cache.display().to_string(),
            &occupied.display().to_string(),
            &profiles(&publisher),
            "directory",
            fixed_time(),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("export destination {} already exists", occupied.display())
        );
        assert_eq!(std::fs::read(occupied.join("keep")).unwrap(), b"keep");
        let root = Root::open(&cache).unwrap();
        assert_eq!(root.ref_entry(&model).unwrap().as_deref(), Some("prior"));
        assert_eq!(root.versions(&model).unwrap(), vec!["prior".to_string()]);

        // A successful conversion selects the converted entry.
        let destination = scratch.path().join("converted");
        convert_cache_at(
            &source.display().to_string(),
            &cache.display().to_string(),
            &destination.display().to_string(),
            &profiles(&publisher),
            "directory",
            fixed_time(),
        )
        .unwrap();
        // The exported tree verifies as a signed package and still matches the
        // installed entry's payload closure.
        let exported = PackageSource::open(
            &destination.join("model.safetensors"),
            &publishers,
            selection,
        )
        .unwrap();
        let root = Root::open(&cache).unwrap();
        let current = root.ref_entry(&model).unwrap().unwrap();
        assert_ne!(current, "prior");
        assert_eq!(current, "unknown-device-tp1-202610170923");
        let selected = root.entry(&model, &current).unwrap();
        root.verify_ready(&selected, &exported).unwrap();
        assert!(root.versions(&model).unwrap().contains(&current));
    }

    #[test]
    fn payload_cache_conversion_reserves_same_minute_names() {
        let scratch = tempfile::tempdir().unwrap();
        let (source, publisher) = payload_package(scratch.path());
        let cache = scratch.path().join("cache");
        let model = helios_core::package::cache::model_key("t/model").unwrap();
        for _ in 0..2 {
            convert_cache_at(
                &source.display().to_string(),
                &cache.display().to_string(),
                "",
                &profiles(&publisher),
                "directory",
                fixed_time(),
            )
            .unwrap();
        }
        let root = helios_core::package::cache::Root::open(&cache).unwrap();
        let mut versions = root.versions(&model).unwrap();
        versions.sort();
        assert_eq!(
            versions,
            vec![
                "unknown-device-tp1-202610170923".to_string(),
                "unknown-device-tp1-202610170923-2".to_string(),
            ]
        );
        assert_eq!(
            root.ref_entry(&model).unwrap().as_deref(),
            Some("unknown-device-tp1-202610170923-2")
        );
    }

}
