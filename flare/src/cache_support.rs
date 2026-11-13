/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-platform prepared-model cache selection and deployment choice.
//!
//! Unix-specific rank transport and locking remain in `core_cache`.

use crate::configuration::{Configuration, expand};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

type Failure = Box<dyn std::error::Error + Send + Sync>;

/// Prepared-model cache root: explicit option, then `paths.core_cache`, then the default.
///
/// Terminal and application callers resolve explicit storage options before host defaults.
pub fn model_root(explicit: Option<&Path>) -> Result<PathBuf, Failure> {
    let home = crate::environment::home_directory()?;
    let config = home.join(".config/flare/config.toml");
    model_root_from(&home, Some(&config), explicit)
}

pub fn model_root_from(
    home: &Path,
    config: Option<&Path>,
    explicit: Option<&Path>,
) -> Result<PathBuf, Failure> {
    if let Some(path) = explicit {
        return Ok(expand(home, path));
    }
    if let Some(config) = config
        && let Some(configured) = Configuration::load(config, false)?.core_cache(home)
    {
        return Ok(configured);
    }
    Ok(default_core_root(home))
}

/// Prefer canonical storage; reuse a legacy tree only when canonical storage is absent.
pub fn default_core_root(home: &Path) -> PathBuf {
    let canonical = home.join(".cache/flare/cores");
    let legacy = home.join(".cache/flare/packages");
    if !canonical.exists() && legacy.is_dir() {
        legacy
    } else {
        canonical
    }
}

/// User Hugging Face cache that prepared-core acquisition may borrow from.
///
/// A `[paths].hub_cache` value reaches acquisition already resolved, through this
/// process's selected configuration, so `--config` is honored and no configuration
/// is reloaded. Without that value the location is the one the Hugging Face
/// library uses by default. The tree is a read-only source: Flare never creates,
/// renames, links into, or changes permissions under it.
pub fn hub_cache_root(explicit: Option<&Path>) -> Result<PathBuf, Failure> {
    if let Some(path) = explicit {
        return Ok(path.to_owned());
    }
    Ok(hub_cache_default(&crate::environment::home_directory()?))
}

pub fn hub_cache_default(home: &Path) -> PathBuf {
    home.join(".cache/huggingface/hub")
}

fn cache_key(input: &str) -> Result<String, Failure> {
    flare_helios::core::cache::model_key(input).map_err(|error| error.to_string().into())
}

/// Accepted user syntax is the model identity (for example `owner/name`); a cache key is
/// accepted only when the identity it decodes to names the same entry, so both spellings resolve
/// deterministically.
pub fn resolve_model(
    root: &flare_helios::core::cache::Root,
    input: &str,
) -> Result<(String, String), Failure> {
    let key = cache_key(input)?;
    if root.ref_entry(&key)?.is_some() || !root.versions(&key)?.is_empty() {
        return Ok((key, input.to_owned()));
    }
    if let Ok(identity) = flare_helios::core::cache::model_identity(input) {
        let decoded = cache_key(&identity)?;
        if decoded != key
            && (root.ref_entry(&decoded)?.is_some() || !root.versions(&decoded)?.is_empty())
        {
            return Ok((decoded, identity));
        }
    }
    Ok((key, input.to_owned()))
}

pub fn explicit_core_path(input: &Path) -> Result<bool, Failure> {
    Ok(input.is_absolute()
        || input.starts_with(".")
        || input.starts_with("..")
        || input.try_exists()?)
}

/// Resolve an immutable local version, holding its lease before releasing the default ref.
pub fn local_model(
    root: &flare_helios::core::cache::Root,
    input: &Path,
) -> Result<Option<(flare_helios::core::cache::Entry, bool)>, Failure> {
    if explicit_core_path(input)? {
        return Ok(None);
    }
    let Some(input) = input.to_str() else {
        return Ok(None);
    };
    let (identity, version) = match input.rsplit_once('@') {
        Some((model, version)) if !model.is_empty() && !version.is_empty() => {
            (model, Some(version))
        }
        Some(_) => return Err("expected a core path or MODEL[@VERSION]".into()),
        None => (input, None),
    };
    let (model, identity) = resolve_model(root, identity)?;
    if root.ref_entry(&model)?.is_none() && root.versions(&model)?.is_empty() {
        return Ok(None);
    }
    let entry = match version {
        Some(version) => root.entry(&model, version)?,
        None => root.resolve(&model)?.ok_or_else(|| format!(
            "model {identity} has no selected default version; use flare models list to see local versions"
        ))?,
    };
    Ok(Some((entry, version.is_some())))
}

/// Normalize requested tensor-parallel sizes: deduplicated, ascending, non-zero.
pub fn requested_tensor_parallel(requested: &[u32], gpu_count: usize) -> Result<Vec<u32>, Failure> {
    let mut sizes: Vec<u32> = if requested.is_empty() {
        vec![u32::try_from(gpu_count.max(1)).map_err(|_| "GPU count is too large")?]
    } else {
        requested.to_vec()
    };
    if sizes.contains(&0) {
        return Err("tensor-parallel sizes must be positive".into());
    }
    sizes.sort_unstable();
    sizes.dedup();
    Ok(sizes)
}

/// Resolve signed deployments for the requested tensor-parallel sizes.
///
/// Topology comes from each authenticated deployment body, never from
/// availability hints: a size that was never exported, appears twice, or has an
/// incomplete payload closure is an error rather than a silently dropped request.
pub fn select_deployments(
    metadata: &flare_helios::core::reader::Metadata,
    publishers: &[flare_helios::runtime_core::Publisher],
    sizes: &[u32],
) -> Result<Vec<flare_helios::core::protection::Selection>, Failure> {
    use flare_helios::core::protection;

    let catalog = protection::verify(
        &metadata.envelope,
        publishers,
        protection::Selection::Catalog,
    )?;
    let count = catalog.deployment_count();
    let mut found: HashMap<u32, Vec<usize>> = HashMap::new();
    for index in 0..count {
        let selection = protection::Selection::Deployment {
            deployment: index,
            rank: None,
        };
        let verified = protection::verify(&metadata.envelope, publishers, selection)?;
        let deployment = verified
            .deployment()
            .ok_or_else(|| format!("deployment {index} has no authenticated body"))?;
        found
            .entry(deployment.tensor_parallel)
            .or_default()
            .push(index);
    }
    let mut selected = Vec::with_capacity(sizes.len());
    for size in sizes {
        let indices = found.get(size).ok_or_else(|| {
            format!("core exports no deployment with tensor-parallel size {size}")
        })?;
        match indices.as_slice() {
            [index] => selected.push(protection::Selection::Deployment {
                deployment: *index,
                rank: None,
            }),
            _ => {
                return Err(format!(
                    "core exports {} deployments with tensor-parallel size {size}",
                    indices.len()
                )
                .into());
            }
        }
    }
    let mut verified = Vec::with_capacity(selected.len());
    for selection in &selected {
        verified.push(protection::verify(
            &metadata.envelope,
            publishers,
            *selection,
        )?);
    }
    flare_helios::core::cache::Root::validate_ready_closure(
        metadata,
        &verified,
        &flare_helios::core::derivation::Limits::default(),
    )
    .map_err(|error| error.to_string())?;
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_core_storage_reuses_legacy_without_moving_data() {
        let home = tempfile::tempdir().unwrap();
        let canonical = home.path().join(".cache/flare/cores");
        let legacy = home.path().join(".cache/flare/packages");
        assert_eq!(default_core_root(home.path()), canonical);
        std::fs::create_dir_all(&legacy).unwrap();
        let sentinel = legacy.join("existing-core");
        std::fs::write(&sentinel, b"retained").unwrap();
        assert_eq!(default_core_root(home.path()), legacy);
        assert!(!canonical.exists());
        std::fs::create_dir_all(&canonical).unwrap();
        assert_eq!(default_core_root(home.path()), canonical);
        assert_eq!(std::fs::read(sentinel).unwrap(), b"retained");
    }

    #[test]
    fn canonical_and_legacy_configuration_paths_override_default_storage() {
        let home = tempfile::tempdir().unwrap();
        let configuration = home.path().join("config.toml");
        std::fs::create_dir_all(home.path().join(".cache/flare/packages")).unwrap();
        for key in ["core_cache", "package_cache"] {
            std::fs::write(&configuration, format!("[paths]\n{key}='~/selected'\n")).unwrap();
            assert_eq!(
                model_root_from(home.path(), Some(&configuration), None).unwrap(),
                home.path().join("selected")
            );
            assert_eq!(
                model_root_from(
                    home.path(),
                    Some(&configuration),
                    Some(Path::new("~/explicit"))
                )
                .unwrap(),
                home.path().join("explicit")
            );
        }
    }

    #[test]
    fn hub_cache_root_prefers_the_selected_value() {
        assert_eq!(
            hub_cache_root(Some(Path::new("/checkpoints"))).unwrap(),
            PathBuf::from("/checkpoints")
        );
    }

    #[test]
    fn hub_cache_default_is_the_hub_library_layout() {
        assert_eq!(
            hub_cache_default(Path::new("/home/test")),
            Path::new("/home/test").join(".cache/huggingface/hub")
        );
    }
}
