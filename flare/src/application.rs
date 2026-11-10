//! Model-library workflows shared by terminal and native application frontends.
use std::path::{Path, PathBuf};

use flare_helios::{
    core::{
        cache::{self, Entry, Root},
        progress::AcquisitionProgress,
    },
    runtime_core::{
        AuthorizedCore, CoreClaim, InspectionOptions, Publisher, VerifiedCore,
        unverified_metadata_selected,
    },
    startup::StartupProgress,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub use crate::options::HubOptions;
pub use crate::trust::{
    Acceptance as TrustAcceptance, Disclosure as TrustDisclosure, ReviewOptions as TrustOptions,
};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Source intent is explicit; a missing local path never becomes a network request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Source {
    LocalPath(PathBuf),
    LocalModel {
        model: String,
        version: Option<String>,
    },
    HuggingFace {
        repository: String,
        revision: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VerificationState {
    Unverified,
    SignatureVerified,
    PayloadVerified,
}

#[derive(Debug, Clone, Serialize)]
pub struct LibraryEntry {
    pub model: String,
    pub model_key: String,
    pub version: String,
    pub selected: bool,
    pub leased: bool,
    pub verification: VerificationState,
    /// Remembered acquisition repositories; these records do not establish trust.
    pub hub_repositories: Vec<String>,
    pub metadata: Option<Value>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct InspectOptions {
    pub unverified: bool,
    pub memory_map: bool,
    pub graphs: bool,
    pub frontend: bool,
    pub all: bool,
    pub rank: Option<usize>,
    pub tensor_parallel: Option<u32>,
    pub publisher_keys: Vec<PathBuf>,
    pub trust_store: Option<PathBuf>,
}

/// A resolved managed core retains a cache lease until the operation completes.
pub struct ResolvedModel {
    pub path: PathBuf,
    pub lease: Option<Entry>,
}
pub struct AuthorizedModel {
    pub path: PathBuf,
    pub core: AuthorizedCore,
    pub publisher: Publisher,
    pub lease: Option<Entry>,
}

pub struct TrustReview {
    pub disclosure: TrustDisclosure,
    inner: crate::trust::Review,
    resolved: ResolvedModel,
}
impl TrustReview {
    pub fn core(&self) -> &VerifiedCore {
        self.inner.core()
    }
    pub fn accept(
        self,
        acceptance: &TrustAcceptance,
        verify_weights: bool,
        verify_payloads: bool,
        progress: Option<&StartupProgress>,
    ) -> Result<AuthorizedModel> {
        let (core, publisher) =
            self.inner
                .accept(acceptance, verify_weights, verify_payloads, progress)?;
        Ok(AuthorizedModel {
            path: self.resolved.path,
            core,
            publisher,
            lease: self.resolved.lease,
        })
    }
}

pub struct ModelStore {
    root: Root,
}

fn hub_repositories(root: &Root, entry: &Entry) -> Vec<String> {
    let origin = root.entry_source(entry).ok().flatten();
    let references = root.entry_source_refs(entry).unwrap_or_default();
    origin
        .into_iter()
        .chain(references)
        .filter_map(|reference| {
            let (provider, repository, _, _, _): (
                String,
                String,
                String,
                Option<String>,
                Vec<u32>,
            ) = serde_json::from_str(reference.selector()).ok()?;
            if provider != "huggingface" || crate::hub::repository(&repository).ok()?.1.is_some() {
                return None;
            }
            Some(repository)
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

impl ModelStore {
    pub fn open(path: PathBuf) -> Result<Self> {
        Ok(Self {
            root: Root::open(path)?,
        })
    }
    pub fn path(&self) -> &Path {
        self.root.path()
    }

    /// Report each manifest independently; an unreadable entry does not hide its neighbors.
    pub fn inventory(&self) -> Result<Vec<LibraryEntry>> {
        let mut entries = Vec::new();
        for model_key in self.root.models()? {
            let model = cache::model_identity(&model_key).unwrap_or_else(|_| model_key.clone());
            let selected = self.root.ref_entry(&model_key);
            let versions = self.root.versions(&model_key);
            let (selected, versions) = match (selected, versions) {
                (Ok(selected), Ok(versions)) => (selected, versions),
                (Err(error), _) | (_, Err(error)) => {
                    entries.push(LibraryEntry {
                        model,
                        model_key,
                        version: String::new(),
                        selected: false,
                        leased: false,
                        verification: VerificationState::Unverified,
                        hub_repositories: Vec::new(),
                        metadata: None,
                        error: Some(error.to_string()),
                    });
                    continue;
                }
            };
            for version in versions {
                let report = (|| -> Result<(bool, Value, Vec<String>)> {
                    let leased = self.root.is_leased(&model_key, &version)?;
                    let entry = self.root.entry(&model_key, &version)?;
                    let summary = self.root.summarize_entry(&entry)?;
                    Ok((
                        leased,
                        json!({
                            "form": summary.form, "format_version": summary.format_version,
                            "model_identity": summary.model_identity, "hardware_identity": summary.hardware_identity,
                            "logical_bytes": summary.logical_bytes, "payloads": summary.payloads,
                            "deployments": summary.deployments.iter().map(|deployment| json!({
                                "id": deployment.id, "tensor_parallel": deployment.tensor_parallel,
                                "ready": deployment.ready, "pages": deployment.pages,
                                "state_rows": deployment.state_rows,
                                "max_sequence_tokens": deployment.max_sequence_tokens,
                                "prefill_chunk_size": deployment.prefill_chunk_size,
                            })).collect::<Vec<_>>(),
                        }),
                        hub_repositories(&self.root, &entry),
                    ))
                })();
                let (leased, metadata, hub_repositories, error) = match report {
                    Ok((leased, metadata, repositories)) => {
                        (leased, Some(metadata), repositories, None)
                    }
                    Err(error) => (false, None, Vec::new(), Some(error.to_string())),
                };
                entries.push(LibraryEntry {
                    model: model.clone(),
                    model_key: model_key.clone(),
                    selected: selected.as_deref() == Some(&version),
                    version,
                    leased,
                    verification: VerificationState::Unverified,
                    hub_repositories,
                    metadata,
                    error,
                });
            }
        }
        Ok(entries)
    }

    pub fn resolve(&self, source: &Source) -> Result<ResolvedModel> {
        let (path, lease) = match source {
            Source::LocalPath(path) => {
                let canonical = path
                    .canonicalize()
                    .map_err(|error| format!("cannot access core {}: {error}", path.display()))?;
                let lease = self.root.entry_for_path(&canonical)?;
                (canonical, lease)
            }
            Source::LocalModel { model, version } => {
                let (key, _) = crate::cache_support::resolve_model(&self.root, model)
                    .map_err(|error| error.to_string())?;
                let entry = match version {
                    Some(version) => self.root.entry(&key, version)?,
                    None => self
                        .root
                        .resolve(&key)?
                        .ok_or_else(|| format!("model {model} has no selected default"))?,
                };
                (self.root.primary(&entry)?, Some(entry))
            }
            Source::HuggingFace { .. } => {
                return Err(
                    "Hugging Face sources require acquisition before local resolution".into(),
                );
            }
        };
        Ok(ResolvedModel { path, lease })
    }

    pub fn inspect(&self, source: &Source, options: &InspectOptions) -> Result<Value> {
        let resolved = self.resolve(source)?;
        let sections = InspectionOptions {
            memory_map: options.memory_map,
            graphs: options.graphs,
            frontend: options.frontend,
            all: options.all,
        };
        let mut value = if options.unverified {
            unverified_metadata_selected(
                &resolved.path,
                sections,
                options.rank,
                options.tensor_parallel,
            )?
        } else {
            let claim = CoreClaim::read(&resolved.path)?;
            let mut profiles = crate::trust::publishers(
                &options.publisher_keys,
                claim.development,
                crate::environment::home_directory,
            )?;
            if let Some(profile) =
                crate::trust::remembered_profile(&claim, options.trust_store.as_deref())?
            {
                if !profiles
                    .iter()
                    .any(|publisher| publisher.fingerprint() == claim.signer)
                {
                    profiles.push(profile);
                }
            }
            let verified = match options.tensor_parallel {
                Some(size) => {
                    let rank = options
                        .rank
                        .map(u32::try_from)
                        .transpose()
                        .map_err(|_| "rank exceeds the core rank domain")?;
                    VerifiedCore::open_tensor_parallel(&resolved.path, &profiles, size, rank)
                }
                None => VerifiedCore::open(&resolved.path, &profiles, options.rank),
            }?;
            verified.describe(sections)?
        };
        value["core"]["path"] = Value::String(resolved.path.display().to_string());
        Ok(value)
    }

    pub fn review_trust(&self, source: &Source, options: &TrustOptions) -> Result<TrustReview> {
        let resolved = self.resolve(source)?;
        let inner = crate::trust::review(&resolved.path, options)?;
        Ok(TrustReview {
            disclosure: inner.disclosure.clone(),
            inner,
            resolved,
        })
    }

    /// Install an authenticated core and preserve the selected local version.
    pub fn acquire(
        &self,
        source: &str,
        options: &HubOptions,
        gpu_count: usize,
        publisher_keys: &[PathBuf],
        trust_store: Option<&Path>,
        profiles: &[Publisher],
        progress: AcquisitionProgress,
    ) -> Result<Entry> {
        let (_, revision) = crate::hub::repository(source)?;
        if options.revision.is_none() && revision.is_none() {
            return Err("embedded acquisition requires an explicit hardware revision".into());
        }
        let mut options = options.clone();
        options.core_cache = Some(self.root.path().to_owned());
        crate::hub::acquire_watched(
            source,
            &options,
            gpu_count,
            publisher_keys,
            trust_store,
            profiles,
            progress,
        )
    }

    pub fn select_default(&self, model: &str, version: &str) -> Result<()> {
        let (key, _) = crate::cache_support::resolve_model(&self.root, model)
            .map_err(|error| error.to_string())?;
        let entry = self.root.entry(&key, version)?;
        self.root.publish_entry(&entry)?;
        Ok(())
    }
    pub fn remove(&self, model: &str, version: &str, clear_default: bool) -> Result<()> {
        let (key, _) = crate::cache_support::resolve_model(&self.root, model)
            .map_err(|error| error.to_string())?;
        self.root.remove(&key, version, clear_default)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flare_helios::core::{cache::CoreSource, reader};

    fn installed(parent: &Path) -> (ModelStore, PathBuf, Publisher, String) {
        let (core, profile, publisher, _) = crate::hub::tests::signed_payload_fixture(parent);
        let store = ModelStore::open(parent.join("models")).unwrap();
        let metadata = reader::Metadata::open(&core).unwrap();
        let selections = crate::cache_support::select_deployments(
            &metadata,
            std::slice::from_ref(&publisher),
            &[1],
        )
        .unwrap();
        let source = CoreSource::open(&core, std::slice::from_ref(&publisher), selections[0])
            .unwrap()
            .union(std::slice::from_ref(&publisher), &selections)
            .unwrap();
        let entry = store
            .root
            .install_core_staged(
                source,
                &cache::model_key("owner/offline").unwrap(),
                "fixture",
            )
            .unwrap();
        let version = entry.id().to_owned();
        drop(entry);
        (store, profile, publisher, version)
    }

    #[test]
    fn inventory_recovers_cli_acquisition_repositories_without_matching_model_names() {
        let temp = tempfile::tempdir().unwrap();
        let (store, _, _, version) = installed(temp.path());
        let model = cache::model_key("owner/offline").unwrap();
        let entry = store.root.entry(&model, &version).unwrap();
        assert!(store.inventory().unwrap()[0].hub_repositories.is_empty());
        for selector in [
            json!([
                "huggingface",
                "publisher/Qwen-27B-FLARE",
                "gb10",
                "model.safetensors",
                [1]
            ]),
            json!([
                "huggingface",
                "publisher/Qwen-27B-FLARE",
                "commit",
                null,
                []
            ]),
            json!(["other-provider", "other/model", "main", null, []]),
            json!(["huggingface", "invalid", "main", null, []]),
            json!(["huggingface", "other/model@branch", "main", null, []]),
            json!(["huggingface", "other/model", "main", "wrong tuple"]),
        ] {
            store
                .root
                .publish_source_ref(&selector.to_string(), "commit", &entry, [1; 32])
                .unwrap();
        }
        let legacy_selector = json!(["huggingface", "legacy/model", "gb10", null, []]).to_string();
        let legacy_path = store
            .path()
            .join("sources")
            .join(flare_helios::core::PayloadId::of(legacy_selector.as_bytes()).to_string());
        std::fs::write(
            legacy_path,
            json!([legacy_selector, "commit", model, version]).to_string(),
        )
        .unwrap();
        let inventory = store.inventory().unwrap();
        assert_eq!(
            inventory[0].hub_repositories,
            ["legacy/model", "publisher/Qwen-27B-FLARE"]
        );
        assert!(inventory[0].error.is_none());
        assert_eq!(inventory[0].verification, VerificationState::Unverified);
    }

    #[test]
    fn inventory_preserves_cores_when_provenance_is_missing_or_malformed() {
        let temp = tempfile::tempdir().unwrap();
        let (store, _, _, version) = installed(temp.path());
        let model = cache::model_key("owner/offline").unwrap();
        let entry = store.root.entry(&model, &version).unwrap();
        let selector = json!([
            "huggingface",
            "publisher/retained",
            "gb10",
            "model.safetensors",
            [1]
        ])
        .to_string();
        store
            .root
            .publish_entry_source(&selector, "commit", &entry, [1; 32])
            .unwrap();
        let sources = store.path().join("sources");
        std::fs::create_dir_all(&sources).unwrap();
        std::fs::write(sources.join("0".repeat(64)), "invalid JSON").unwrap();
        let inventory = store.inventory().unwrap();
        assert_eq!(inventory[0].hub_repositories, ["publisher/retained"]);
        assert!(inventory[0].error.is_none());
        let origins = store.path().join("origins");
        for path in std::fs::read_dir(origins).unwrap() {
            std::fs::write(path.unwrap().path(), "invalid JSON").unwrap();
        }
        let inventory = store.inventory().unwrap();
        assert_eq!(inventory.len(), 1);
        assert!(inventory[0].metadata.is_some());
        assert!(inventory[0].error.is_none());
        assert!(inventory[0].hub_repositories.is_empty());
    }

    #[test]
    fn inventory_keeps_corrupt_entries_visible_and_resolution_holds_a_lease() {
        let temp = tempfile::tempdir().unwrap();
        let (store, _, _, version) = installed(temp.path());
        let model = cache::model_key("owner/offline").unwrap();
        std::fs::create_dir_all(store.path().join("models").join(&model).join("corrupt")).unwrap();
        let inventory = store.inventory().unwrap();
        assert_eq!(inventory.len(), 2);
        assert!(
            inventory
                .iter()
                .find(|entry| entry.version == "corrupt")
                .unwrap()
                .error
                .is_some()
        );
        let valid = inventory
            .iter()
            .find(|entry| entry.version == version)
            .unwrap();
        assert_eq!(valid.verification, VerificationState::Unverified);
        assert!(!valid.selected);
        store.select_default("owner/offline", &version).unwrap();
        let source = Source::LocalModel {
            model: "owner/offline".into(),
            version: None,
        };
        let resolved = store.resolve(&source).unwrap();
        assert!(store.remove("owner/offline", &version, true).is_err());
        drop(resolved);
        store.remove("owner/offline", &version, true).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn inspection_displays_non_utf8_local_paths_without_panicking() {
        use std::os::unix::ffi::OsStringExt;
        let temp = tempfile::tempdir().unwrap();
        let (core, _, _, _) = crate::hub::tests::signed_payload_fixture(temp.path());
        let path = temp
            .path()
            .join(std::ffi::OsString::from_vec(b"core-\xff".to_vec()));
        std::fs::rename(core, &path).unwrap();
        let store = ModelStore::open(temp.path().join("models")).unwrap();
        let value = store
            .inspect(
                &Source::LocalPath(path.clone()),
                &InspectOptions {
                    unverified: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            value["core"]["path"],
            path.canonicalize().unwrap().display().to_string()
        );
    }

    #[test]
    fn explicit_local_source_never_resolves_as_a_repository() {
        let temp = tempfile::tempdir().unwrap();
        let store = ModelStore::open(temp.path().join("models")).unwrap();
        assert!(
            store
                .resolve(&Source::LocalPath(temp.path().join("owner/model")))
                .is_err()
        );
        assert!(
            store
                .resolve(&Source::HuggingFace {
                    repository: "owner/model".into(),
                    revision: Some("gb10".into())
                })
                .is_err()
        );
        assert!(store.inventory().unwrap().is_empty());
    }

    #[test]
    fn trust_review_holds_the_entry_and_binds_exact_consent_without_input() {
        let temp = tempfile::tempdir().unwrap();
        let (store, profile, publisher, version) = installed(temp.path());
        let source = Source::LocalModel {
            model: "owner/offline".into(),
            version: Some(version.clone()),
        };
        let options = TrustOptions {
            publisher_keys: vec![profile],
            hub: HubOptions {
                offline: true,
                ..Default::default()
            },
            trust_store: Some(temp.path().join("trust.toml")),
            tensor_parallel: Some(1),
            rank: None,
        };
        let review = store.review_trust(&source, &options).unwrap();
        assert_eq!(review.disclosure.fingerprint, publisher.fingerprint());
        assert!(!review.disclosure.remembered);
        assert!(store.remove("owner/offline", &version, false).is_err());
        let mismatch = TrustAcceptance {
            fingerprint: "0".repeat(64),
            namespace: publisher.namespace.clone(),
        };
        assert!(review.accept(&mismatch, false, true, None).is_err());
        assert!(!options.trust_store.as_ref().unwrap().exists());
        let review = store.review_trust(&source, &options).unwrap();
        let accepted = review
            .accept(
                &TrustAcceptance {
                    fingerprint: publisher.fingerprint(),
                    namespace: publisher.namespace.clone(),
                },
                true,
                true,
                None,
            )
            .unwrap();
        assert!(store.remove("owner/offline", &version, false).is_err());
        drop(accepted);
        let remembered = store.review_trust(&source, &options).unwrap();
        assert!(remembered.disclosure.remembered);
        drop(remembered);
        store.remove("owner/offline", &version, false).unwrap();
    }
}
