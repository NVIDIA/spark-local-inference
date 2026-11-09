//! Metadata-only model details with explicit verification and compatibility limits.
use crate::{
    application::{ModelStore, Source, TrustOptions},
    hardware::GpuInfo,
};
use flare_helios::runtime_core::InspectionOptions;
use serde_json::Value;
use std::collections::BTreeSet;
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Compatibility {
    #[default]
    Unknown = 0,
    Compatible = 1,
    Incompatible = 2,
}
#[derive(Clone, Debug, Default)]
pub struct ModelDetailView {
    pub id: String,
    pub name: String,
    pub path: String,
    pub publisher: String,
    pub official_source: String,
    pub source_revision: String,
    pub metadata_status: String,
    pub signatures_verified: bool,
    pub inspection_error: String,
    pub context_tokens: Option<u64>,
    pub max_concurrency: Option<u64>,
    pub rank_count: Option<u32>,
    pub architecture: String,
    pub prepared_memory_bytes: Option<u64>,
    pub memory_note: String,
    pub compatibility: Compatibility,
    pub compatibility_reason: String,
    pub can_remove_reference: bool,
}
pub fn inspect(
    store: &ModelStore,
    source: &Source,
    options: &TrustOptions,
    id: String,
    name: String,
    gpus: &[GpuInfo],
) -> crate::application::Result<ModelDetailView> {
    let resolved = store.resolve(source)?;
    let mut detail=ModelDetailView { id,name,path:resolved.path.display().to_string(),
        memory_note:"Recorded allocation bytes include host allocations; runtime-sized caches and driver headroom are additional.".into(),..Default::default() };
    let sections = InspectionOptions {
        memory_map: true,
        ..Default::default()
    };
    let mut envelope = None;
    let document = match crate::trust::review(&resolved.path, options) {
        Ok(review) => {
            detail.publisher = format!(
                "{} ({})",
                review.disclosure.publisher, review.disclosure.namespace
            );
            detail.official_source = review.disclosure.official_source.clone();
            detail.signatures_verified = true;
            detail.metadata_status = "Signatures verified; publisher consent is separate".into();
            envelope = Some(review.core().envelope_digest());
            review.core().describe(sections)?
        }
        Err(error) => {
            detail.inspection_error = error.to_string();
            detail.metadata_status = "Unverified core metadata".into();
            flare_helios::runtime_core::unverified_metadata_selected(
                &resolved.path,
                sections,
                options.rank,
                options.tensor_parallel,
            )?
        }
    };
    if detail.publisher.is_empty() {
        detail.publisher = document["core"]["publisher"]["namespace"]
            .as_str()
            .unwrap_or_default()
            .into();
    }
    if detail.official_source.is_empty() {
        detail.official_source = document["core"]["official_source"]
            .as_str()
            .or_else(|| document["core"]["publisher"]["official_source"].as_str())
            .unwrap_or_default()
            .into();
    }
    if let (Some(entry), Some(envelope)) = (resolved.lease.as_ref(), envelope) {
        let root = flare_helios::core::cache::Root::open(store.path())?;
        if let Some(reference) = root.entry_source(entry)? {
            if reference.expected_envelope() == Some(envelope) {
                let selector: Value =
                    serde_json::from_str(reference.selector()).unwrap_or(Value::Null);
                if selector[0] == "huggingface" {
                    let branch = selector[2].as_str().unwrap_or_default();
                    detail.source_revision = if branch == reference.revision() {
                        branch.to_owned()
                    } else {
                        format!("{branch} @ {}", reference.revision())
                    };
                }
            }
        }
    }
    populate(&mut detail, &document, gpus);
    Ok(detail)
}

fn populate(detail: &mut ModelDetailView, document: &Value, gpus: &[GpuInfo]) {
    detail.context_tokens = document["model"]["max_sequence_tokens"]
        .as_u64()
        .filter(|value| *value > 0);
    detail.max_concurrency = document["model"]["max_active"]
        .as_u64()
        .filter(|value| *value > 0);
    let deployments = document["deployments"].as_array();
    let ambiguous = deployments.is_some_and(|values| values.len() > 1)
        && document["core"]["tensor_parallel"].is_null()
        && document["selected_tensor_parallel"].is_null();
    if !ambiguous {
        detail.rank_count = document["rank_count"]
            .as_u64()
            .or_else(|| document["core"]["ranks"].as_u64())
            .or_else(|| document["ranks"].as_array().map(|ranks| ranks.len() as u64))
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0);
    }
    let mut architectures = BTreeSet::new();
    for rank in document["ranks"].as_array().into_iter().flatten() {
        if let Some(value) = architecture(&rank["architecture"]) {
            architectures.insert(value);
        }
    }
    if architectures.is_empty() {
        for deployment in deployments.into_iter().flatten() {
            if let Some(value) = architecture(&deployment["architecture"]) {
                architectures.insert(value);
            }
        }
    }
    detail.architecture = architectures
        .iter()
        .map(|[major, minor]| format!("SM{major}{minor}"))
        .collect::<Vec<_>>()
        .join(" / ");
    let allocations = document["ranks"].as_array().map(|ranks| {
        ranks
            .iter()
            .flat_map(|rank| {
                rank["memory_map"]["allocations"]
                    .as_array()
                    .or_else(|| rank["allocations"].as_array())
                    .into_iter()
                    .flatten()
            })
            .collect::<Vec<_>>()
    });
    detail.prepared_memory_bytes =
        allocations
            .filter(|values| !values.is_empty())
            .and_then(|values| {
                values.into_iter().try_fold(0u64, |total, allocation| {
                    total.checked_add(allocation["nbytes"].as_u64()?)
                })
            });
    detail.compatibility = Compatibility::Unknown;
    detail.compatibility_reason =
        "Full compatibility and available-memory checks run when loading the core.".into();
    if !detail.signatures_verified {
        detail.compatibility_reason =
            "Core signatures have not been verified; recorded capabilities are untrusted.".into();
    } else if detail.rank_count.is_some_and(|count| count != 1) {
        detail.compatibility = Compatibility::Incompatible;
        detail.compatibility_reason =
            "The desktop runtime requires a core prepared for one local GPU.".into();
    } else if let Some(gpu) = gpus.iter().find(|gpu| gpu.visible_ordinal == Some(0)) {
        if let Some(actual) = gpu.architecture {
            if !architectures.is_empty() && !architectures.contains(&actual) {
                detail.compatibility = Compatibility::Incompatible;
                detail.compatibility_reason = format!(
                    "The core requires {}; the selected GPU is SM{}{}.",
                    detail.architecture, actual[0], actual[1]
                );
            }
        }
    } else {
        detail.compatibility_reason =
            "A visible NVIDIA GPU has not been identified; load-time checks are required.".into();
    }
}
fn architecture(value: &Value) -> Option<[u32; 2]> {
    let values = value.as_array()?;
    if values.len() != 2 {
        return None;
    }
    Some([
        u32::try_from(values[0].as_u64()?).ok()?,
        u32::try_from(values[1].as_u64()?).ok()?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn unverified_metadata_never_establishes_compatibility_or_memory_fit() {
        let mut detail = ModelDetailView::default();
        populate(
            &mut detail,
            &json!({"model":{"max_sequence_tokens":8192,"max_active":4},"rank_count":1,
            "ranks":[{"architecture":[12,0],"allocations":[{"nbytes":1024}]}]}),
            &[GpuInfo {
                visible_ordinal: Some(0),
                architecture: Some([12, 0]),
                total_memory_bytes: Some(2048),
                ..Default::default()
            }],
        );
        assert_eq!(detail.context_tokens, Some(8192));
        assert_eq!(detail.prepared_memory_bytes, Some(1024));
        assert_eq!(detail.compatibility, Compatibility::Unknown);
        assert!(!detail.signatures_verified);
    }
    #[test]
    fn authenticated_architecture_mismatch_is_an_explicit_blocker() {
        let mut detail = ModelDetailView {
            signatures_verified: true,
            ..Default::default()
        };
        populate(
            &mut detail,
            &json!({"rank_count":1,"ranks":[{"architecture":[12,0]}]}),
            &[GpuInfo {
                visible_ordinal: Some(0),
                architecture: Some([10, 3]),
                ..Default::default()
            }],
        );
        assert_eq!(detail.compatibility, Compatibility::Incompatible);
        assert!(detail.compatibility_reason.contains("SM103"));
        assert!(detail.prepared_memory_bytes.is_none());
    }
    #[test]
    fn signed_core_details_require_no_acceptance_or_gpu_loading() {
        let temporary = tempfile::tempdir().unwrap();
        let (core, profile, _, _) = crate::hub::tests::signed_payload_fixture(temporary.path());
        let store = ModelStore::open(temporary.path().join("models")).unwrap();
        let trust = temporary.path().join("trust.toml");
        let result = inspect(
            &store,
            &Source::LocalPath(core),
            &TrustOptions {
                publisher_keys: vec![profile],
                hub: crate::options::HubOptions {
                    offline: true,
                    ..Default::default()
                },
                trust_store: Some(trust.clone()),
                ..Default::default()
            },
            "fixture".into(),
            "Fixture".into(),
            &[],
        )
        .unwrap();
        assert!(result.signatures_verified);
        assert_eq!(result.context_tokens, Some(8));
        assert!(!trust.exists());
        assert_eq!(result.compatibility, Compatibility::Unknown);
    }
}
