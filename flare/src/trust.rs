/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Publisher acceptance and the persistent signing-key trust store.
//!
//! Core review reads its claim, resolves the signer through the publisher repository,
//! verifies signatures, and returns the publisher, source, and license for disclosure.
//! Explicit consent must match the signed fingerprint and publisher namespace. Remembered
//! acceptance suppresses first-use prompting; it never skips the repository check.

#[cfg(unix)]
use std::fs::OpenOptions;
#[cfg(unix)]
use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};

use flare_helios::runtime_core::{AuthorizedCore, CoreClaim, Publisher, VerifiedCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::identity::{self, Resolution};
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SLOT_SIZE: usize = 65536;

pub fn publishers(
    paths: &[PathBuf],
    development: bool,
    home: impl FnOnce() -> io::Result<PathBuf>,
) -> Result<Vec<Publisher>> {
    let default;
    let paths = if development && paths.is_empty() {
        default = home()?.join(".config/flare/development-publisher.json");
        if !default.try_exists()? {
            return Ok(Vec::new());
        }
        std::slice::from_ref(&default)
    } else {
        paths
    };
    let mut publishers = Vec::new();
    for path in paths {
        let mut bytes = Vec::new();
        File::open(path)
            .map_err(|error| format!("cannot read publisher profile {}: {error}", path.display()))?
            .take((SLOT_SIZE + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > SLOT_SIZE {
            return Err(
                format!("publisher profile file exceeds 64 KiB: {}", path.display()).into(),
            );
        }
        publishers.extend(
            serde_json::from_slice::<Vec<Publisher>>(&bytes).map_err(|error| {
                format!("invalid publisher profile {}: {error}", path.display())
            })?,
        );
    }
    unique_profiles(publishers)
}

fn unique_profiles(publishers: Vec<Publisher>) -> Result<Vec<Publisher>> {
    // One key may be listed under several namespaces: a publisher promoting a core keeps the
    // key and changes the namespace. Identity is the (key, namespace) pair, so a differing display
    // name for the same pair is the only real conflict; namespaces never merge into one another.
    let mut unique = std::collections::HashMap::new();
    let mut result = Vec::new();
    for publisher in publishers {
        publisher.validate()?;
        let identity = (publisher.fingerprint(), publisher.namespace.clone());
        if let Some(previous) = unique.insert(identity, publisher.clone()) {
            if previous != publisher {
                return Err(
                    "conflicting identities for the same publisher key and namespace".into(),
                );
            }
        } else {
            result.push(publisher);
        }
    }
    Ok(result)
}

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Store {
    trusted: Vec<Entry>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    fingerprint: String,
    namespace: String,
    /// The verification key, remembered so later loads verify signatures without a profile or
    /// a network request. Entries recorded before it was stored lack it and use profiles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    public_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    /// How the key's identity was established at acceptance: `huggingface.co` when the
    /// publisher's repository listed it, otherwise `profile`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    identity: Option<String>,
}
impl Entry {
    fn profile(&self) -> Option<Publisher> {
        let key = self.public_key.as_deref()?;
        let mut public_key = [0u8; 32];
        for (byte, pair) in public_key.iter_mut().zip(key.as_bytes().chunks(2)) {
            *byte = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
        }
        Some(Publisher {
            name: self
                .name
                .clone()
                .unwrap_or_else(|| "Remembered signer".into()),
            namespace: self.namespace.clone(),
            public_key,
            development: self.namespace == "local-development",
        })
    }
}
/// Public profiles remembered in an explicit trust store. Entries without a
/// stored key cannot verify signatures themselves and are omitted.
pub fn trusted_profiles(store: Option<&Path>) -> Result<Vec<Publisher>> {
    let store = load_store(&store_path(store)?)?;
    Ok(store.trusted.iter().filter_map(Entry::profile).collect())
}

/// The verification key remembered for the claim's signer and namespace when the store holds
/// one, so a core accepted earlier verifies without a profile or a network request.
pub fn remembered_profile(claim: &CoreClaim, store: Option<&Path>) -> Result<Option<Publisher>> {
    let store = load_store(&store_path(store)?)?;
    Ok(store
        .trusted
        .iter()
        .find(|e| e.fingerprint == claim.signer && e.namespace == claim.namespace)
        .and_then(Entry::profile))
}
fn fingerprint_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn store_path(given: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = given {
        return Ok(path.to_owned());
    }
    #[cfg(windows)]
    return Ok(crate::windows::trust_path()?);
    #[cfg(unix)]
    {
        let mut storage = vec![0u8; 65536];
        let mut record = std::mem::MaybeUninit::<libc::passwd>::uninit();
        let mut result = std::ptr::null_mut();
        // The passwd entry is copied before its backing buffer is released.
        let status = unsafe {
            libc::getpwuid_r(
                libc::getuid(),
                record.as_mut_ptr(),
                storage.as_mut_ptr().cast(),
                storage.len(),
                &mut result,
            )
        };
        if status != 0 || result.is_null() {
            return Err("cannot resolve home directory; specify --trust-store".into());
        }
        let home = unsafe { std::ffi::CStr::from_ptr((*result).pw_dir) }.to_str()?;
        Ok(Path::new(home).join(".config/flare/trust.toml"))
    }
}
fn load_store(path: &Path) -> Result<Store> {
    let bytes = match fs::read_to_string(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Store::default()),
        Err(error) => return Err(error.into()),
    };
    let store: Store = toml::from_str(&bytes)?;
    let mut seen = std::collections::HashSet::new();
    for entry in &store.trusted {
        if !fingerprint_valid(&entry.fingerprint)
            || !seen.insert((&entry.fingerprint, &entry.namespace))
            || entry
                .identity
                .as_deref()
                .is_some_and(|i| !matches!(i, "huggingface.co" | "profile"))
        {
            return Err("invalid or duplicate trust-store entry".into());
        }
        match entry.profile() {
            Some(profile) => {
                profile.validate()?;
                // The remembered key must be the one the fingerprint names.
                if hex(&Sha256::digest(profile.public_key)) != entry.fingerprint {
                    return Err("trust-store entry key does not match its fingerprint".into());
                }
            }
            None if entry.public_key.is_some() => {
                return Err("invalid trust-store entry key".into());
            }
            None => Publisher {
                development: false,
                name: "trust entry".into(),
                namespace: entry.namespace.clone(),
                public_key: [1; 32],
            }
            .validate()?,
        }
    }
    Ok(store)
}
fn modify(path: &Path, update: impl FnOnce(&mut Store)) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut lock_name = path.as_os_str().to_owned();
    lock_name.push(".lock");
    #[cfg(windows)]
    let lock = crate::windows::private_file(Path::new(&lock_name), false)?;
    #[cfg(unix)]
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(PathBuf::from(lock_name))?;
    // The lock serializes read/modify/rename across concurrent startup processes.
    #[cfg(unix)]
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    #[cfg(windows)]
    crate::windows::lock(&lock)?;
    let mut store = load_store(path)?;
    update(&mut store);
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(format!(".{}.tmp", std::process::id()));
    let temporary = PathBuf::from(temporary);
    #[cfg(windows)]
    let mut output = crate::windows::private_file(&temporary, true)?;
    #[cfg(unix)]
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary)?;
    let result = (|| -> Result<()> {
        output.write_all(toml::to_string(&store)?.as_bytes())?;
        output.sync_all()?;
        drop(output);
        #[cfg(unix)]
        {
            fs::rename(&temporary, path)?;
            File::open(parent)?.sync_all()?;
        }
        #[cfg(windows)]
        crate::windows::replace(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// How the signing key's identity was established for the disclosure.
#[derive(Debug, Clone, Serialize)]
pub enum Identity {
    /// The publisher's repository lists this key under the claimed namespace.
    Verified,
    /// Signatures verify against a local profile; the publisher's repository did not confirm it.
    Unverified(String),
    Development,
}

/// Everything a core claim establishes before acceptance: the exact claim, the profiles that
/// can verify its signatures, the resolved publisher identity (including revocation), and the
/// remembered trust entry. Acquisition uses this to authenticate a core and bind its metadata
/// without accepting terms or decoding executables; acceptance requires `Review::accept`.
pub struct ResolvedIdentity {
    pub claim: CoreClaim,
    pub profiles: Vec<Publisher>,
    identity: Identity,
    remembered: Option<Entry>,
    /// Trust store this resolution read, where acceptance is recorded.
    pub store_path: PathBuf,
}

pub fn resolve_identity(
    root: &Path,
    publisher_keys: &[PathBuf],
    hub: &crate::options::HubOptions,
    trust_store: Option<&Path>,
) -> Result<ResolvedIdentity> {
    resolve_identity_with(root, publisher_keys, &[], hub, trust_store)
}

/// Resolve identity with profiles carried in an authenticated request. Profiles
/// are public keys, never private material or controller-local paths.
pub fn resolve_identity_with(
    root: &Path,
    publisher_keys: &[PathBuf],
    supplied: &[Publisher],
    hub: &crate::options::HubOptions,
    trust_store: Option<&Path>,
) -> Result<ResolvedIdentity> {
    let claim = CoreClaim::read(root)?;
    let path = store_path(trust_store)?;
    let store = load_store(&path)?;
    let remembered = store
        .trusted
        .iter()
        .find(|e| e.fingerprint == claim.signer && e.namespace == claim.namespace)
        .cloned();
    let mut profiles = publishers(
        publisher_keys,
        claim.development,
        crate::environment::home_directory,
    )?;
    profiles.extend(supplied.iter().cloned());
    let mut profiles = unique_profiles(profiles)?;
    // The publisher's repository is consulted on every load that is not offline, so a key the
    // publisher removed from publisher.json stops loading everywhere, remembered or not. When it
    // cannot be consulted, a previously confirmed identity stands.
    let confirmed_earlier = remembered
        .as_ref()
        .is_some_and(|e| e.identity.as_deref() == Some("huggingface.co"));
    let identity = if claim.development {
        Identity::Development
    } else {
        match identity::resolve(&claim, hub)? {
            Resolution::Verified(profile) => {
                profiles.push(profile);
                Identity::Verified
            }
            _ if confirmed_earlier => Identity::Verified,
            Resolution::Absent => {
                Identity::Unverified(format!("no publisher.json at {}", claim.official_source))
            }
            Resolution::Unreachable(reason) => Identity::Unverified(format!(
                "{}: {reason}",
                identity::profile_url(&claim.official_source)
            )),
        }
    };
    // The remembered key verifies signatures only when no fresher profile names the same key and
    // namespace, so a publisher changing its display name does not conflict with what was
    // remembered, and a key promoted to another namespace still needs a profile for that
    // namespace.
    if !profiles
        .iter()
        .any(|p| p.fingerprint() == claim.signer && p.namespace == claim.namespace)
    {
        if let Some(profile) = remembered.as_ref().and_then(Entry::profile) {
            profiles.push(profile);
        }
    }
    Ok(ResolvedIdentity {
        claim,
        profiles,
        identity,
        remembered,
        store_path: path,
    })
}

/// Metadata and publisher identity retained until the caller makes a trust decision.
pub struct Review {
    core: VerifiedCore,
    resolved: ResolvedIdentity,
    pub disclosure: Disclosure,
}

#[derive(Debug, Clone, Serialize)]
pub struct Disclosure {
    pub fingerprint: String,
    pub namespace: String,
    pub publisher: String,
    pub official_source: String,
    pub license_name: Option<String>,
    pub license_url: Option<String>,
    pub identity: Identity,
    pub remembered: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ReviewOptions {
    pub publisher_keys: Vec<PathBuf>,
    pub hub: crate::options::HubOptions,
    pub trust_store: Option<PathBuf>,
    pub rank: Option<usize>,
    pub tensor_parallel: Option<u32>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Acceptance {
    pub fingerprint: String,
    pub namespace: String,
}

/// Authenticate signed metadata and disclose identity without accepting terms or reading stdin.
pub fn review(root: &Path, options: &ReviewOptions) -> Result<Review> {
    let resolved = resolve_identity(
        root,
        &options.publisher_keys,
        &options.hub,
        options.trust_store.as_deref(),
    )?;
    let core = match options.tensor_parallel {
        Some(size)
            if matches!(
                flare_helios::runtime_core::NativeResources::core_format(root),
                Ok(flare_helios::runtime_core::CoreFormat::Payload)
            ) =>
        {
            let rank = options
                .rank
                .map(u32::try_from)
                .transpose()
                .map_err(|_| "rank exceeds the core rank domain")?;
            VerifiedCore::open_tensor_parallel(root, &resolved.profiles, size, rank)
        }
        _ => VerifiedCore::open(root, &resolved.profiles, options.rank),
    }?;
    let disclosure = Disclosure {
        fingerprint: resolved.claim.signer.clone(),
        namespace: resolved.claim.namespace.clone(),
        publisher: core.publisher().name.clone(),
        official_source: core.official_source().to_owned(),
        license_name: core.license().map(|license| license.name.clone()),
        license_url: core.license().map(|license| license.link.clone()),
        identity: resolved.identity.clone(),
        remembered: resolved.remembered.is_some(),
    };
    Ok(Review {
        core,
        resolved,
        disclosure,
    })
}

impl Review {
    pub fn core(&self) -> &VerifiedCore {
        &self.core
    }

    /// Bind explicit consent to the authenticated fingerprint and exact signed namespace.
    pub fn accept(
        self,
        acceptance: &Acceptance,
        verify_weights: bool,
        verify_payloads: bool,
        progress: Option<&flare_helios::startup::StartupProgress>,
    ) -> Result<(AuthorizedCore, Publisher)> {
        if acceptance.fingerprint != self.resolved.claim.signer
            || acceptance.namespace != self.resolved.claim.namespace
        {
            return Err("publisher acceptance must exactly match the authenticated fingerprint and namespace".into());
        }
        let publisher = self.core.publisher().clone();
        let authorized = self.core.authorize_with_progress(
            &acceptance.namespace,
            verify_weights,
            verify_payloads,
            progress,
        )?;
        let entry = Entry {
            fingerprint: acceptance.fingerprint.clone(),
            namespace: acceptance.namespace.clone(),
            public_key: Some(hex(&publisher.public_key)),
            name: Some(publisher.name.clone()),
            identity: Some(
                match self.resolved.identity {
                    Identity::Verified => "huggingface.co",
                    _ => "profile",
                }
                .into(),
            ),
        };
        let current = self.resolved.remembered.as_ref().is_some_and(|previous| {
            previous.public_key == entry.public_key
                && previous.name == entry.name
                && previous.identity == entry.identity
        });
        if !current {
            modify(&self.resolved.store_path, |store| {
                store.trusted.retain(|previous| {
                    !(previous.fingerprint == entry.fingerprint
                        && previous.namespace == entry.namespace)
                });
                store.trusted.push(entry);
            })?;
        }
        Ok((authorized, publisher))
    }
}

pub fn revoke(fingerprint: &str, given: Option<&Path>) -> Result<()> {
    if !fingerprint_valid(fingerprint) {
        return Err("a full lowercase SHA-256 fingerprint is required".into());
    }
    modify(&store_path(given)?, |store| {
        store
            .trusted
            .retain(|entry| entry.fingerprint != fingerprint)
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn development_publisher() -> Publisher {
        Publisher {
            name: "Local development signer".into(),
            namespace: "local-development".into(),
            public_key: [7; 32],
            development: true,
        }
    }

    #[test]
    fn development_cores_find_the_default_public_profile() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join(".config/flare/development-publisher.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let publisher = development_publisher();
        fs::write(&path, serde_json::to_vec(&[&publisher]).unwrap()).unwrap();
        assert_eq!(
            publishers(&[], true, || Ok(home.path().to_owned())).unwrap(),
            vec![publisher]
        );
    }

    #[test]
    fn one_key_may_hold_several_namespaces_but_not_conflicting_identities() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("publisher.json");
        let mut second = development_publisher();
        second.name = "Example".into();
        second.namespace = "example".into();
        second.development = false;
        let mut promoted = second.clone();
        promoted.namespace = "example-next".into();
        fs::write(&path, serde_json::to_vec(&[&second, &promoted]).unwrap()).unwrap();
        assert_eq!(
            publishers(std::slice::from_ref(&path), false, || unreachable!()).unwrap(),
            vec![second.clone(), promoted]
        );
        // The same key and namespace with different attributes is one ambiguity, not two profiles.
        let mut renamed = second.clone();
        renamed.name = "Someone Else".into();
        fs::write(&path, serde_json::to_vec(&[&second, &renamed]).unwrap()).unwrap();
        assert!(publishers(std::slice::from_ref(&path), false, || unreachable!()).is_err());
    }

    #[test]
    fn explicit_profiles_and_production_cores_skip_the_default_lookup() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("publisher.json");
        let publisher = development_publisher();
        fs::write(&path, serde_json::to_vec(&[&publisher]).unwrap()).unwrap();
        for development in [false, true] {
            assert_eq!(
                publishers(std::slice::from_ref(&path), development, || {
                    panic!("explicit profiles must not resolve the default directory")
                })
                .unwrap(),
                vec![publisher.clone()]
            );
        }
        assert!(
            publishers(&[], false, || {
                panic!("production cores must not resolve the default directory")
            })
            .unwrap()
            .is_empty()
        );
        fs::write(&path, b"[]").unwrap();
        assert!(
            publishers(std::slice::from_ref(&path), true, || {
                panic!("an empty explicit profile must not trigger the default lookup")
            })
            .unwrap()
            .is_empty()
        );
        fs::remove_file(&path).unwrap();
        assert!(
            publishers(&[path], true, || {
                panic!("a missing explicit profile must not trigger the default lookup")
            })
            .is_err()
        );
    }

    #[test]
    fn a_missing_development_profile_leaves_other_keys_available() {
        let home = tempfile::tempdir().unwrap();
        assert!(
            publishers(&[], true, || Ok(home.path().to_owned()))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn malformed_and_oversized_defaults_report_the_profile_path() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join(".config/flare/development-publisher.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        for (bytes, message) in [
            (b"not json".to_vec(), "invalid publisher profile"),
            (vec![b' '; SLOT_SIZE + 1], "exceeds 64 KiB"),
        ] {
            fs::write(&path, bytes).unwrap();
            let error = publishers(&[], true, || Ok(home.path().to_owned()))
                .unwrap_err()
                .to_string();
            assert!(error.contains(message), "{error}");
            assert!(error.contains(path.to_str().unwrap()), "{error}");
        }
    }

    #[test]
    fn default_profiles_validate_keys_and_reject_conflicting_identities() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join(".config/flare/development-publisher.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let publisher = development_publisher();
        let mut invalid = publisher.clone();
        invalid.public_key = [0; 32];
        fs::write(&path, serde_json::to_vec(&[invalid]).unwrap()).unwrap();
        assert!(publishers(&[], true, || Ok(home.path().to_owned())).is_err());
        let mut conflicting = publisher.clone();
        conflicting.name = "Another signer".into();
        fs::write(
            &path,
            serde_json::to_vec(&[publisher, conflicting]).unwrap(),
        )
        .unwrap();
        let error = publishers(&[], true, || Ok(home.path().to_owned()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("conflicting identities"), "{error}");
    }
}

#[cfg(all(test, windows))]
mod windows_tests {
    use super::*;

    #[test]
    fn concurrent_updates_preserve_entries_and_replace_existing_files() {
        let directory =
            std::env::temp_dir().join(format!("flare-trust-用户-{}", std::process::id()));
        let path = directory.join("trust.toml");
        std::thread::scope(|scope| {
            for index in 0..8 {
                let path = &path;
                scope.spawn(move || {
                    modify(path, |store| {
                        store.trusted.push(Entry {
                            fingerprint: format!("{index:064x}"),
                            namespace: "test-publisher".into(),
                            public_key: None,
                            name: None,
                            identity: None,
                        })
                    })
                    .unwrap()
                });
            }
        });
        assert_eq!(load_store(&path).unwrap().trusted.len(), 8);
        modify(&path, |store| store.trusted.clear()).unwrap();
        assert!(load_store(&path).unwrap().trusted.is_empty());
        fs::remove_dir_all(directory).unwrap();
    }
}
