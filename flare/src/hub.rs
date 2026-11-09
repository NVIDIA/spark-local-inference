/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Download prepared packages from hardware-specific Hugging Face revisions.

use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use flare_helios::package::progress::Counter;
use hf_hub::{
    Cache, Repo, RepoType,
    api::sync::{Api, ApiBuilder},
};
use sha2::Digest as _;

use crate::cli::HubOptions;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn is_repository(path: &Path) -> bool {
    !path.exists()
        && !path.is_absolute()
        && !path.starts_with(".")
        && !path.starts_with("..")
        && path.to_str().is_some_and(|text| {
            !text.contains('\\') && !text.ends_with(".safetensors") && repository(text).is_ok()
        })
}

pub(crate) fn repository(text: &str) -> Result<(&str, Option<&str>)> {
    let (id, revision) = match text.split_once('@') {
        Some((id, revision)) if !revision.is_empty() => (id, Some(revision)),
        Some(_) => return Err("Hugging Face revision cannot be empty".into()),
        None => (text, None),
    };
    let parts: Vec<_> = id.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || part.starts_with(['.', '-'])
                || part.ends_with(['.', '-'])
                || part.contains("..")
                || part.contains("--")
                || !part
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
        })
    {
        return Err("expected a Hugging Face repository as ORG/NAME[@REVISION]".into());
    }
    if let Some(revision) = revision {
        validate_relative(revision)?;
    }
    Ok((id, revision))
}

fn validate_relative(value: &str) -> Result<()> {
    if value.is_empty()
        || value.contains(['\\', '\0'])
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == ".." || part.contains(':'))
    {
        return Err(format!("invalid repository-relative path or revision: {value:?}").into());
    }
    Ok(())
}

fn is_companion(filename: &str) -> bool {
    let Some(stem) = filename.strip_suffix(".safetensors") else {
        return false;
    };
    let stem = stem.rsplit('/').next().unwrap_or(stem);
    if stem.len() == 64
        && stem
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return true;
    }
    let Some((_, numbers)) = stem.rsplit_once("-data-") else {
        return false;
    };
    let bytes = numbers.as_bytes();
    bytes.len() == 8
        && &bytes[2..6] == b"-of-"
        && bytes[..2].iter().chain(&bytes[6..]).all(u8::is_ascii_digit)
}

fn choose_file(files: impl IntoIterator<Item = String>, requested: Option<&str>) -> Result<String> {
    if let Some(filename) = requested {
        validate_relative(filename)?;
        if !filename.ends_with(".safetensors") || is_companion(filename) {
            return Err("--filename must name a prepared .safetensors package".into());
        }
        return Ok(filename.to_owned());
    }
    let mut candidates: Vec<_> = files
        .into_iter()
        .filter(|file| file.ends_with(".safetensors") && !is_companion(file))
        .filter(|file| is_canonical_primary(file))
        .collect();
    candidates.sort();
    match candidates.as_slice() {
        [filename] => {
            validate_relative(filename)?;
            Ok(filename.clone())
        }
        [] => Err("revision contains no .safetensors package; choose a revision containing prepared Flare packages".into()),
        _ => Err(format!("revision contains several .safetensors files; select one with --filename:\n  {}", candidates.join("\n  ")).into()),
    }
}

pub fn run_download(args: crate::cli::Download) -> Result<()> {
    crate::environment::configure(&args.gpus)?;
    let entry = download(
        &args.repository,
        &args.hub,
        args.gpus.len(),
        &args.publisher_keys,
        args.trust_store.as_deref(),
        &[],
    )?;
    println!("{}", entry.directory().display());
    Ok(())
}

pub fn download(
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    publisher_keys: &[std::path::PathBuf],
    trust_store: Option<&std::path::Path>,
    supplied_profiles: &[flare_helios::runtime_package::Publisher],
) -> Result<flare_helios::package::cache::Entry> {
    let access_token = if options.offline {
        None
    } else {
        token(options)?
    };
    download_with_token(
        source,
        options,
        gpu_count,
        access_token,
        publisher_keys,
        trust_store,
        supplied_profiles,
    )
}

pub(crate) fn download_with_token(
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    access_token: Option<String>,
    publisher_keys: &[std::path::PathBuf],
    trust_store: Option<&std::path::Path>,
    supplied_profiles: &[flare_helios::runtime_package::Publisher],
) -> Result<flare_helios::package::cache::Entry> {
    download_watched(
        source,
        options,
        gpu_count,
        access_token,
        publisher_keys,
        trust_store,
        supplied_profiles,
        None,
    )
}

/// Acquire a prepared package, reporting into a caller-owned progress handle.
///
/// `Some(counters)` keeps every counter update and suppresses this function's
/// own stderr summary, so a caller such as an agent can present progress itself.
/// `None` retains the local summary. Acquisition never depends on the caller's
/// observer: the handle is push-only, and a failed acquisition returns an error
/// while retaining its resumable transport bytes.
pub(crate) fn download_watched(
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    access_token: Option<String>,
    publisher_keys: &[std::path::PathBuf],
    trust_store: Option<&std::path::Path>,
    supplied_profiles: &[flare_helios::runtime_package::Publisher],
    counters: Option<flare_helios::package::progress::AcquisitionProgress>,
) -> Result<flare_helios::package::cache::Entry> {
    download_with_token_at(
        "https://huggingface.co",
        source,
        options,
        gpu_count,
        access_token,
        publisher_keys,
        trust_store,
        supplied_profiles,
        counters,
    )
}

#[allow(clippy::too_many_arguments)]
fn download_with_token_at(
    endpoint: &str,
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    access_token: Option<String>,
    publisher_keys: &[std::path::PathBuf],
    trust_store: Option<&std::path::Path>,
    supplied_profiles: &[flare_helios::runtime_package::Publisher],
    counters: Option<flare_helios::package::progress::AcquisitionProgress>,
) -> Result<flare_helios::package::cache::Entry> {
    download_for_model_at(
        endpoint,
        source,
        options,
        gpu_count,
        access_token,
        publisher_keys,
        trust_store,
        supplied_profiles,
        None,
        counters,
    )
}

#[allow(clippy::too_many_arguments)]
fn download_for_model_at(
    endpoint: &str,
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    access_token: Option<String>,
    publisher_keys: &[std::path::PathBuf],
    trust_store: Option<&std::path::Path>,
    supplied_profiles: &[flare_helios::runtime_package::Publisher],
    expected_model: Option<&str>,
    counters: Option<flare_helios::package::progress::AcquisitionProgress>,
) -> Result<flare_helios::package::cache::Entry> {
    let (id, given_revision) = repository(source)?;
    let revision = match options.revision.as_deref().or(given_revision) {
        Some(revision) => revision.to_owned(),
        None => crate::hardware::detect_isolated(gpu_count)?,
    };
    validate_relative(&revision)?;
    if let Some(filename) = &options.filename {
        choose_file([], Some(filename))?;
    }
    let model_root = crate::cache_support::model_root(options.model_cache.as_deref())
        .map_err(|error| error.to_string())?;
    eprintln!("Prepared-model cache: {}", model_root.display());
    if options.offline {
        let (entry, expected_envelope) =
            offline_entry_with_expected(&model_root, id, &revision, options, gpu_count)?;
        let primary = entry_primary(&entry)?;
        let resolved = crate::trust::resolve_identity_with(
            &primary,
            publisher_keys,
            supplied_profiles,
            options,
            trust_store,
        )?;
        verify_offline_binding(&entry, expected_envelope, id, &revision, &resolved.profiles)?;
        verify_offline_entry(&entry, options, gpu_count, &resolved.profiles)?;
        flare_helios::package::cache::Root::open(&model_root)?.publish_entry(&entry)?;
        return Ok(entry);
    }
    // Prepared package payloads are fetched into the durable transport shared by
    // every acquisition under `--cache-dir`, or under the prepared-model cache
    // when it is not given. An interrupted transfer resumes from its partial file
    // in a later invocation, and completed content is reused by every repository
    // and revision that has the same bytes.
    let report = AcquisitionReport::with(counters);
    // Local presentation only: an agent caller supplies its own counters and
    // renders its own row, and a silent startup makes these stages inert.
    let startup = if report.summarize {
        flare_helios::startup::StartupProgress::stderr_rows()
    } else {
        flare_helios::startup::StartupProgress::silent()
    };
    let display =
        flare_helios::package::progress::AcquisitionDisplay::new(&startup, &report.progress);
    // Reuse from the user Hugging Face cache is optional: a host without a
    // readable hub-cache setting still downloads normally. The path arrives from
    // this process's selected configuration, never from a reloaded default file.
    let hub_cache = crate::cache_support::hub_cache_root(options.hub_cache.as_deref()).ok();
    let transport = Transport::open(&model_root, options.cache_dir.as_deref(), endpoint)?
        .with_hub_cache(hub_cache);
    let cache = Cache::new(transport.root().to_owned());
    let repo = Repo::with_revision(id.to_owned(), RepoType::Model, revision.clone());
    eprintln!("Package source: {id}@{revision}");
    let (path, installed_folder, installed_commit) = {
        let api = ApiBuilder::from_cache(cache)
            .with_token(access_token.clone())
            .with_endpoint(endpoint.into())
            .with_progress(false)
            .with_retries(0)
            .build()?;
        let token = access_token.as_deref();
        display.stage("Download package", Counter::Network, None);
        let (primary, pinned) = fetch_primary_with_progress(
            &api,
            &transport,
            &repo,
            options.filename.as_deref(),
            token,
            Some(&report.progress),
        )?;
        let filename = primary
            .strip_prefix(transport.snapshot_dir(&pinned.folder_name(), pinned.revision()))?
            .to_str()
            .ok_or("package filename is not UTF-8")?
            .to_owned();
        // The shared resolver supplies remembered and explicit profiles plus the
        // established identity; download never accepts terms or decodes executables.
        let resolved = crate::trust::resolve_identity_with(
            &primary,
            publisher_keys,
            supplied_profiles,
            options,
            trust_store,
        )
        .map_err(|error| transport_hint(transport.root(), error))?;
        let publishers = resolved.profiles;
        let legacy = flare_helios::package::conversion::form(&primary)
            .map_err(|error| error.to_string())?
            .is_none();
        if let Some(expected) = expected_model.filter(|_| !legacy) {
            use flare_helios::package::{cache, protection, reader};
            let metadata = reader::Metadata::open(&primary)?;
            let catalog = protection::verify(
                &metadata.envelope,
                &publishers,
                protection::Selection::Catalog,
            )?;
            let identity = catalog.package().model_identity.as_str();
            let model = cache::model_key(if identity.is_empty() { id } else { identity })
                .map_err(|error| error.to_string())?;
            if model != expected {
                return Err("the Hugging Face update names a different model; the selected version is unchanged".into());
            }
        }
        let entry = if legacy {
            match legacy_warm_entry(
                &primary,
                options,
                &id,
                &revision,
                pinned.revision(),
                &filename,
                gpu_count,
                &publishers,
                &model_root,
            )? {
                Some(entry) => entry,
                None => {
                    let names = dependency_files(&primary, options, gpu_count, &publishers)?;
                    let pending = if names.is_empty() {
                        Some(PendingNetwork {
                            total: Some(0),
                            bound: 0,
                        })
                    } else {
                        pending_network_bytes(&api, &transport, &pinned, &primary, &names, token)
                    };
                    let remaining = pending.as_ref().and_then(|pending| pending.total);
                    report
                        .progress
                        .record_network_total(remaining.map(|remaining| {
                            report
                                .progress
                                .snapshot()
                                .network_bytes
                                .saturating_add(remaining)
                        }));
                    announce_space(
                        &transport,
                        &model_root,
                        pending.as_ref().map(|pending| pending.bound),
                    );
                    if !names.is_empty() {
                        display.stage("Download companions", Counter::Network, remaining);
                    }
                    for name in &names {
                        fetch_dependency_with_progress(
                            &api,
                            &transport,
                            &pinned,
                            &primary,
                            name,
                            token,
                            Some(&report.progress),
                        )?;
                    }
                    // A completed transfer ends at exactly the bytes it pulled,
                    // even when adoption saved more or less than predicted.
                    report
                        .progress
                        .record_network_total(Some(report.progress.snapshot().network_bytes));
                    display.stage("Prepare package", Counter::Materialized, None);
                    install_ready(&id, &primary, options, &model_root, gpu_count, &publishers)
                        .map_err(|error| transport_hint(transport.root(), error))?
                }
            }
        } else {
            // Planning validates cached entries and relocates sources; install
            // materializes and verifies payloads. Each row meters one counter, so
            // verification is presented while planning and materialization while
            // installing.
            display.stage("Verify package", Counter::Verification, None);
            match plan_payload_acquisition(
                &primary,
                options,
                &id,
                &revision,
                pinned.revision(),
                &filename,
                gpu_count,
                &publishers,
                &model_root,
                Some(&report.progress),
            )? {
                PayloadAcquisition::Ready(entry) => entry,
                PayloadAcquisition::Planned { plan, staging } => {
                    let names = plan.remote_files();
                    let pending = if names.is_empty() {
                        Some(PendingNetwork {
                            total: Some(0),
                            bound: 0,
                        })
                    } else {
                        pending_network_bytes(&api, &transport, &pinned, &primary, &names, token)
                    };
                    let remaining = pending.as_ref().and_then(|pending| pending.total);
                    report
                        .progress
                        .record_network_total(remaining.map(|remaining| {
                            report
                                .progress
                                .snapshot()
                                .network_bytes
                                .saturating_add(remaining)
                        }));
                    announce_space(
                        &transport,
                        &model_root,
                        pending.as_ref().map(|pending| pending.bound),
                    );
                    if !names.is_empty() {
                        display.stage("Download companions", Counter::Network, remaining);
                    }
                    for name in &names {
                        fetch_dependency_with_progress(
                            &api,
                            &transport,
                            &pinned,
                            &primary,
                            name,
                            token,
                            Some(&report.progress),
                        )?;
                    }
                    // A completed transfer ends at exactly the bytes it pulled,
                    // even when adoption saved more or less than predicted.
                    report
                        .progress
                        .record_network_total(Some(report.progress.snapshot().network_bytes));
                    display.stage("Prepare package", Counter::Materialized, None);
                    finish_payload_acquisition(&primary, &names, plan, staging, &model_root)
                        .map_err(|error| transport_hint(transport.root(), error))?
                }
            }
        };
        if expected_model.is_some_and(|model| model != entry.model()) {
            return Err("the Hugging Face update names a different model; the selected version is unchanged".into());
        }
        let expected_envelope =
            authenticated_source_envelope(&primary, options, gpu_count, &publishers)?;
        remember_download(
            &model_root,
            id,
            &revision,
            pinned.revision(),
            &filename,
            options,
            gpu_count,
            &entry,
            expected_envelope,
        )?;
        // The default switches only after the source mapping is durable; a failed
        // source recording must not leave a different model version selected.
        flare_helios::package::cache::Root::open(&model_root)?.publish_entry(&entry)?;
        display.finish();
        (entry, pinned.folder_name(), pinned.revision().to_owned())
    };
    let mut report = report;
    report.success = true;
    // The installed package no longer needs its transport copies. Cleanup runs
    // only after this acquisition releases its shared lease, so a concurrent
    // acquisition or reclamation pass makes it skip instead of removing content
    // the other one still reads. A cleanup failure never undoes the install.
    let root = transport.root().to_owned();
    let scope = transport.scope(&installed_folder);
    drop(transport);
    match crate::download_cache::reclaim_installed(
        &root,
        &scope,
        &installed_folder,
        &installed_commit,
    ) {
        Ok(consumed) if consumed.complete_bytes > 0 => eprintln!(
            "Reclaimed {} bytes of consumed Hugging Face transport content",
            consumed.complete_bytes
        ),
        Ok(_) => {}
        Err(error) => eprintln!(
            "Warning: the package is installed, but its transport content was not reclaimed: {error}"
        ),
    }
    Ok(path)
}

struct AcquisitionReport {
    progress: flare_helios::package::progress::AcquisitionProgress,
    success: bool,
    /// Whether this call owns stderr presentation. A caller that supplies its own
    /// counters renders progress and status itself.
    summarize: bool,
}

impl AcquisitionReport {
    fn with(counters: Option<flare_helios::package::progress::AcquisitionProgress>) -> Self {
        match counters {
            Some(progress) => Self {
                progress,
                success: false,
                summarize: false,
            },
            None => Self {
                progress: flare_helios::package::progress::AcquisitionProgress::new(),
                success: false,
                summarize: true,
            },
        }
    }
}

impl Drop for AcquisitionReport {
    fn drop(&mut self) {
        if !self.summarize {
            return;
        }
        let snapshot = self.progress.snapshot();
        let state = if self.success { "completed" } else { "failed" };
        eprintln!(
            "Acquisition {state}: network {} bytes, local source {} bytes, materialized {} bytes, verification {} bytes, elapsed {:.3}s",
            snapshot.network_bytes,
            snapshot.local_source_bytes,
            snapshot.materialized_bytes(),
            snapshot.verification_bytes,
            snapshot.elapsed.as_secs_f64()
        );
    }
}

const TRANSPORT_VERSION: &str = "flare-hub-transport 1";
/// A single body response must finish within this deadline. A deadline that
/// interrupts real progress resumes from the received offset; only attempts
/// that add no bytes count toward `TRANSFER_STALLED_ATTEMPTS`.
const TRANSFER_BODY_DEADLINE: Duration = Duration::from_secs(300);
const TRANSFER_STALLED_ATTEMPTS: usize = 5;
const TRANSFER_BACKOFF_BASE: Duration = Duration::from_millis(250);
const TRANSFER_BACKOFF_LIMIT: Duration = Duration::from_secs(10);
/// A server-requested delay is honored up to this bound so a hostile or
/// misconfigured `Retry-After` cannot stall an acquisition indefinitely.
const TRANSFER_RETRY_AFTER_LIMIT: Duration = Duration::from_secs(60);
const TRANSFER_BUFFER: usize = 1 << 20;
/// A long response syncs its partial at these intervals so a crash keeps the
/// bytes the server already sent.
const PART_SYNC_BYTES: u64 = 64 << 20;
const PART_SYNC_INTERVAL: Duration = Duration::from_secs(5);

/// Persistent Hub transport shared by every acquisition under one parent.
///
/// ```text
/// <parent>/transport/hub/v1/version                          transport generation
/// <parent>/transport/hub/v1/blobs/<scope>/<key>              complete content
/// <parent>/transport/hub/v1/blobs/<scope>/<key>.part         resumable bytes
/// <parent>/transport/hub/v1/blobs/<scope>/<key>.lock         per-content transfer lock
/// <parent>/transport/hub/v1/acquisition.lock                  acquisition lease
/// <parent>/transport/hub/v1/repos/<scope>/<folder>/refs/<revision>
/// <parent>/transport/hub/v1/repos/<scope>/<folder>/snapshots/<commit>/<path>
/// ```
///
/// The scope identifies the endpoint and repository; the key encodes the Hub
/// ETag losslessly. Snapshot entries are hardlinks into
/// `blobs/`, so package files are readable regular files without a second copy.
/// The tree outlives the acquisition: a partial transfer resumes in a later
/// invocation and completed content is reused instead of downloaded again.
struct Transport {
    root: PathBuf,
    /// Endpoint every content key is scoped to. Identical ETags from different
    /// services do not name identical bytes.
    endpoint: String,
    /// Prepared-model cache root, used to find staging trees of older versions.
    model_root: PathBuf,
    /// Explicit `--cache-dir`, whose abandoned temporary trees are also scanned.
    explicit: Option<PathBuf>,
    /// User Hugging Face cache that complete package files may be borrowed from
    /// read-only. Selected by `[paths].hub_cache` or the standard default.
    hub_cache: Option<PathBuf>,
    agent: ureq::Agent,
    /// Shared lease for this acquisition. Reclamation takes the same inode
    /// exclusively, so repository snapshot metadata cannot be removed while an
    /// acquisition still opens files through it.
    _lease: std::fs::File,
}

/// Free bytes on the filesystem that would hold `path`, walking up to the
/// nearest existing ancestor. `None` means the query is unavailable, which is
/// distinct from a filesystem with no free space.
pub(crate) fn available_bytes(path: &Path) -> Option<u64> {
    let mut candidate = path;
    loop {
        if candidate.exists() {
            return free_bytes_at(candidate);
        }
        match candidate.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => candidate = parent,
            _ => return free_bytes_at(Path::new(".")),
        }
    }
}

#[cfg(unix)]
fn free_bytes_at(path: &Path) -> Option<u64> {
    let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is a valid C string and `stat` a valid out-pointer.
    if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    Some(stat.f_bavail as u64 * stat.f_frsize as u64)
}

#[cfg(windows)]
fn free_bytes_at(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let mut available = 0u64;
    // SAFETY: `wide` is NUL-terminated and `available` a valid out-pointer.
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if ok == 0 { None } else { Some(available) }
}

/// One advisory line about space for the transfer. Estimates never reject an
/// acquisition; they describe the transport parent, the model cache, and the
/// known remaining transfer size.
fn space_advisory(
    transport_free: Option<u64>,
    model_free: Option<u64>,
    remaining: Option<u64>,
) -> String {
    use flare_helios::package::progress::format_bytes;
    let free = |bytes: Option<u64>| match bytes {
        Some(bytes) => format_bytes(bytes),
        None => "unknown".to_owned(),
    };
    let remaining = match remaining {
        Some(bytes) => format!("known transfer {}", format_bytes(bytes)),
        None => "known transfer unknown".to_owned(),
    };
    format!(
        "Transport space: {} available, model cache space: {} available, {remaining}",
        free(transport_free),
        free(model_free)
    )
}

fn announce_space(transport: &Transport, model_root: &Path, remaining: Option<u64>) {
    if remaining.is_none() {
        return;
    }
    eprintln!(
        "{}",
        space_advisory(
            available_bytes(transport.root()),
            available_bytes(model_root),
            remaining
        )
    );
}

/// Transport root for a prepared-model cache and optional `--cache-dir`, for
/// callers that inspect free space without opening an acquisition.
pub(crate) fn transport_root(model_root: &Path, explicit: Option<&Path>) -> PathBuf {
    explicit
        .unwrap_or(model_root)
        .join("transport")
        .join("hub")
        .join("v1")
}

/// Declared byte sum of the package files this acquisition still needs, for a
/// stage total. A missing size only omits the total; it never fails a download.
/// Network bytes one acquisition still expects to pull.
///
/// A completed acquisition's network count is exact; this only estimates how
/// much is left so a row can show a fraction and an ETA. Estimates are promises,
/// so anything that may still cause a transfer leaves `total` unknown instead of
/// predicting zero.
struct PendingNetwork {
    /// Bytes that must still arrive over the network, when every remaining
    /// content identity is settled. `None` otherwise.
    total: Option<u64>,
    /// Bytes that may still arrive, for the space advisory.
    bound: u64,
}

fn pending_network_bytes(
    api: &Api,
    transport: &Transport,
    pinned: &Repo,
    primary: &Path,
    names: &[String],
    token: Option<&str>,
) -> Option<PendingNetwork> {
    let remote = api.repo(pinned.clone());
    let folder = pinned.folder_name();
    let snapshot = transport.snapshot_dir(&folder, pinned.revision());
    let base = primary
        .strip_prefix(&snapshot)
        .ok()?
        .parent()
        .map(Path::to_owned)
        .unwrap_or_default();
    let mut pending = PendingNetwork {
        total: Some(0),
        bound: 0,
    };
    // Transfers are keyed by content identity, so several paths that share an
    // ETag are filled by one transfer and counted once.
    let mut contents: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for name in names {
        let relative = base.join(name).to_string_lossy().replace('\\', "/");
        let identity = file_identity(&remote.url(&relative), token).ok()?;
        let size = identity.size();
        if snapshot.join(&relative).is_file() {
            continue;
        }
        let key = blob_key(identity.etag()).ok()?;
        if let Some(known) = contents.get(&key) {
            if *known != size {
                // One identity cannot describe two lengths; stop predicting.
                pending.total = None;
                pending.bound = pending.bound.saturating_add(size.max(*known));
            }
            continue;
        }
        contents.insert(key.clone(), size);

        // Bytes already local are not network work, but only when nothing can
        // still replace them: a complete blob with a stated SHA-256 is verified
        // and refetched when corrupt, and an unvalidated user cache file may fail
        // its identity. A saved resume prefix is always continued from.
        let mut saved = 0u64;
        let mut settled = true;
        if std::fs::symlink_metadata(transport.blob(&folder, &key))
            .is_ok_and(|info| info.is_file() && info.len() == size)
        {
            if content_digest(identity.etag()).is_some() {
                settled = false;
                saved = 0;
            } else {
                saved = size;
            }
        } else if let Ok(info) = std::fs::symlink_metadata(transport.part(&folder, &key))
            && info.is_file()
        {
            saved = info.len().min(size);
        }
        if saved < size
            && cached_identity(identity.etag()).is_some()
            && transport.hub_cache.as_ref().is_some_and(|hub_cache| {
                std::fs::metadata(
                    hub_cache
                        .join(&folder)
                        .join("snapshots")
                        .join(pinned.revision())
                        .join(&relative),
                )
                .is_ok_and(|info| info.is_file() && info.len() == size)
            })
        {
            settled = false;
        }
        let missing = size.saturating_sub(saved);
        pending.bound = pending.bound.saturating_add(missing);
        if !settled {
            pending.total = None;
        } else if let Some(total) = pending.total.as_mut() {
            *total = total.saturating_add(missing);
        }
    }
    Some(pending)
}

impl Transport {
    /// Transport root for a prepared-model cache and optional `--cache-dir`.
    fn path(model_root: &Path, explicit: Option<&Path>) -> PathBuf {
        transport_root(model_root, explicit)
    }

    fn open(model_root: &Path, explicit: Option<&Path>, endpoint: &str) -> Result<Self> {
        let root = Self::path(model_root, explicit);
        std::fs::create_dir_all(root.join("blobs"))?;
        let marker = root.join("version");
        match std::fs::read_to_string(&marker) {
            Ok(text) if text == format!("{TRANSPORT_VERSION}\n") => {}
            Ok(text) => {
                return Err(format!(
                    "{} belongs to {text:?}, not {TRANSPORT_VERSION}; move it aside to select another transport",
                    root.display()
                )
                .into());
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&marker)
                {
                    Ok(mut file) => {
                        file.write_all(format!("{TRANSPORT_VERSION}\n").as_bytes())?;
                        file.sync_all()?;
                        sync_directory(&root)?;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        let text = std::fs::read_to_string(&marker)?;
                        if text != format!("{TRANSPORT_VERSION}\n") {
                            return Err(format!(
                                "{} belongs to {text:?}, not {TRANSPORT_VERSION}",
                                root.display()
                            )
                            .into());
                        }
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            Err(error) => return Err(format!("cannot read {}: {error}", marker.display()).into()),
        }
        let lease = shared_lock(&root.join("acquisition.lock"))?;
        Ok(Self {
            root,
            endpoint: endpoint.to_owned(),
            model_root: model_root.to_owned(),
            explicit: explicit.map(Path::to_owned),
            hub_cache: None,
            agent: transfers(),
            _lease: lease,
        })
    }

    fn with_hub_cache(mut self, hub_cache: Option<PathBuf>) -> Self {
        self.hub_cache = hub_cache;
        self
    }

    fn root(&self) -> &Path {
        &self.root
    }

    /// One content namespace per endpoint and repository. A Hub ETag is only an
    /// identity within the repository that published it, and opaque ETags carry
    /// no content digest at all.
    fn scope(&self, folder: &str) -> String {
        transport_scope(&self.endpoint, folder)
    }

    /// Content namespace for an endpoint and repository folder, for callers
    /// that inspect a transport root without an open acquisition.
    #[cfg(test)]
    fn scope_for(endpoint: &str, folder: &str) -> String {
        transport_scope(endpoint, folder)
    }

    fn blob(&self, folder: &str, key: &str) -> PathBuf {
        self.root.join("blobs").join(self.scope(folder)).join(key)
    }

    fn part(&self, folder: &str, key: &str) -> PathBuf {
        self.root
            .join("blobs")
            .join(self.scope(folder))
            .join(format!("{key}.part"))
    }

    fn lock(&self, folder: &str, key: &str) -> PathBuf {
        self.root
            .join("blobs")
            .join(self.scope(folder))
            .join(format!("{key}.lock"))
    }

    fn snapshot_dir(&self, folder: &str, commit: &str) -> PathBuf {
        self.root
            .join("repos")
            .join(self.scope(folder))
            .join(folder)
            .join("snapshots")
            .join(commit)
    }

    /// Record the commit a requested revision resolved to. Informational: a
    /// failure here does not invalidate content that is already durable.
    fn record_ref(&self, folder: &str, revision: &str, commit: &str) {
        let path = self
            .root
            .join("repos")
            .join(self.scope(folder))
            .join(folder)
            .join("refs")
            .join(revision);
        let write = || -> std::io::Result<()> {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, commit)
        };
        let _ = write();
    }

    /// Adopt resumable bytes an earlier Flare left in staging trees written by
    /// hf-hub. Adoption is strictly additive: legacy files are only read, a
    /// destination that already holds at least as many bytes is left alone, and
    /// both the legacy transfer and the destination lock must be held while the
    /// bytes are copied into a uniquely named file before atomic publication.
    /// A tree whose staging lease or file lock is held is skipped, so a live
    /// older process is never disturbed.
    fn adopt_legacy(&self, folder: &str) -> u64 {
        let mut recovered = 0;
        for (blobs, _lease) in self.legacy_blob_dirs(folder) {
            let Ok(files) = std::fs::read_dir(&blobs) else {
                continue;
            };
            let discovered: Vec<_> = files.flatten().map(|file| file.file_name()).collect();
            for name in discovered {
                let Some(name) = name.to_str() else {
                    continue;
                };
                if name.ends_with(".lock") {
                    continue;
                }
                let (base, partial) = match name.strip_suffix(".part") {
                    Some(base) => (base, true),
                    None => (name, false),
                };
                let Ok(key) = blob_key(base) else {
                    continue;
                };
                let source = blobs.join(name);
                if partial {
                    // hf-hub holds this lock for the whole transfer of the file.
                    let legacy = blobs.join(format!("{base}.lock"));
                    let Ok(Some(_source)) = try_lock_exclusive(&legacy) else {
                        continue;
                    };
                    recovered += self.adopt_partial(folder, &key, &source);
                } else {
                    recovered += self.adopt_complete(folder, &key, &source);
                }
            }
        }
        recovered
    }

    fn legacy_blob_dirs(&self, folder: &str) -> Vec<(PathBuf, Option<std::fs::File>)> {
        let mut directories = Vec::new();
        if let Ok(entries) = std::fs::read_dir(self.model_root.join("staging/trees")) {
            for entry in entries.flatten() {
                let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                    continue;
                };
                let lease = self
                    .model_root
                    .join("locks/staging")
                    .join(format!("{name}.lock"));
                // A failed probe is treated as live; adoption is best effort.
                let Ok(Some(lease)) = try_lock_exclusive(&lease) else {
                    continue;
                };
                let blobs = entry.path().join(folder).join("blobs");
                if blobs.is_dir() {
                    directories.push((blobs, Some(lease)));
                }
            }
        }
        if let Some(parent) = &self.explicit
            && let Ok(entries) = std::fs::read_dir(parent)
        {
            for entry in entries.flatten() {
                if !entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".flare-hub-staging-")
                    || !entry.file_type().is_ok_and(|kind| kind.is_dir())
                {
                    continue;
                }
                let blobs = entry.path().join(folder).join("blobs");
                if blobs.is_dir() {
                    directories.push((blobs, None));
                }
            }
        }
        directories
    }

    fn adopt_complete(&self, folder: &str, key: &str, source: &Path) -> u64 {
        let Some(length) = std::fs::symlink_metadata(source)
            .ok()
            .filter(|info| info.is_file())
            .map(|info| info.len())
        else {
            return 0;
        };
        let destination = self.blob(folder, key);
        if destination.exists() {
            return 0;
        }
        let Some(parent) = destination.parent() else {
            return 0;
        };
        if std::fs::create_dir_all(parent).is_err() {
            return 0;
        }
        if publish_copy(source, &destination).is_err() {
            return 0;
        }
        let _ = sync_directory(parent);
        length
    }

    fn adopt_partial(&self, folder: &str, key: &str, source: &Path) -> u64 {
        let Some(length) = std::fs::symlink_metadata(source)
            .ok()
            .filter(|info| info.is_file())
            .map(|info| info.len())
        else {
            return 0;
        };
        if length == 0 {
            return 0;
        }
        let destination = self.part(folder, key);
        // Serialize with any current Flare transfer of the same content.
        let Ok(Some(_destination_lock)) = try_lock_exclusive(&self.lock(folder, key)) else {
            return 0;
        };
        if let Ok(existing) = std::fs::symlink_metadata(&destination)
            && (existing.len() >= length || !existing.is_file())
        {
            return 0;
        }
        let Some(parent) = destination.parent() else {
            return 0;
        };
        if std::fs::create_dir_all(parent).is_err() {
            return 0;
        }
        if publish_copy(source, &destination).is_err() {
            return 0;
        }
        let _ = sync_directory(parent);
        length
    }
}

/// Copy `source` into a uniquely named temporary file and publish it at
/// `destination` without ever replacing existing content there. The legacy
/// source is only read.
fn publish_copy(source: &Path, destination: &Path) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let Some(parent) = destination.parent() else {
        return Err("published file has no parent directory".into());
    };
    let name = destination
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut temporary = None;
    for _ in 0..64 {
        let candidate = parent.join(format!(
            ".adopt-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut file) => {
                std::io::copy(&mut std::fs::File::open(source)?, &mut file)?;
                file.sync_all()?;
                temporary = Some(candidate);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    let Some(temporary) = temporary else {
        return Err("cannot allocate an adoption temporary file".into());
    };
    // Hardlinking publishes only when no destination exists. When a shorter
    // destination is already resumable, replace it atomically under its lock.
    match std::fs::hard_link(&temporary, destination) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if let Err(error) = std::fs::rename(&temporary, destination) {
                let _ = std::fs::remove_file(&temporary);
                return Err(error.into());
            }
        }
        Err(error) => {
            let _ = std::fs::remove_file(&temporary);
            return Err(error.into());
        }
    }
    let _ = std::fs::remove_file(&temporary);
    Ok(())
}

impl std::fmt::Debug for Transport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Transport")
            .field("root", &self.root)
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

/// One request per transfer attempt, with the Hub's authentication header and
/// without forwarding that header across a redirect to a content host.
fn transfers() -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .max_redirects(0)
        .redirect_auth_headers(ureq::config::RedirectAuthHeaders::Never)
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(120)))
        .timeout_recv_body(Some(TRANSFER_BODY_DEADLINE))
        .build()
        .into()
}

/// Content namespace for one endpoint and repository folder.
fn transport_scope(endpoint: &str, folder: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut context = Sha256::new();
    context.update(endpoint.as_bytes());
    context.update([0u8]);
    context.update(folder.as_bytes());
    format!("{:x}", context.finalize())
}

/// Lossless file name for one Hub ETag.
///
/// `_` escapes first, then every byte outside `[A-Za-z0-9-]` becomes `_XX`, so
/// distinct opaque ETags never share a key. The alphabet excludes `.` so a key
/// can never end in a file-class suffix such as `.part`, `.lock`, or `.corrupt-*`.
/// Path separators and NUL are rejected outright rather than encoded.
fn blob_key(etag: &str) -> Result<String> {
    if etag.is_empty() || etag.bytes().any(|byte| matches!(byte, b'/' | b'\\' | 0)) {
        return Err("Hugging Face returned an invalid file identity".into());
    }
    let mut key = String::with_capacity(etag.len());
    for byte in etag.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' => key.push(byte as char),
            b'_' => key.push_str("_5F"),
            _ => key.push_str(&format!("_{byte:02X}")),
        }
    }
    Ok(key)
}

/// SHA-256 the Hub publishes for a file, when its ETag states one explicitly.
///
/// A bare 64-hex ETag is opaque: without the `sha256:` prefix it is not an
/// authoritative content identity, so no digest is assumed.
fn content_digest(etag: &str) -> Option<[u8; 32]> {
    let hex = etag.strip_prefix("sha256:")?;
    if hex.len() != 64 || !hex.bytes().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let mut digest = [0u8; 32];
    for (index, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(pair).ok()?;
        digest[index] = u8::from_str_radix(text, 16).ok()?;
    }
    Some(digest)
}

fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    std::fs::File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn open_lock(path: &Path) -> Result<std::fs::File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)?
    };
    #[cfg(windows)]
    let file = crate::windows::private_file(path, false)?;
    Ok(file)
}

/// Blocking exclusive lock on `path`, used to serialize one content transfer.
fn content_lock(path: &Path) -> Result<std::fs::File> {
    let file = open_lock(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    #[cfg(windows)]
    if !lock_windows(&file, true, true)? {
        return Err(format!("cannot lock {}", path.display()).into());
    }
    Ok(file)
}

/// Blocking shared lock, used for the acquisition lifetime lease.
fn shared_lock(path: &Path) -> Result<std::fs::File> {
    let file = open_lock(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    #[cfg(windows)]
    if !lock_windows(&file, false, true)? {
        return Err(format!("cannot lock {}", path.display()).into());
    }
    Ok(file)
}

/// Exclusive lock that reports whether another holder already owns `path`.
fn try_lock_exclusive(path: &Path) -> Result<Option<std::fs::File>> {
    let file = open_lock(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return Ok(Some(file));
        }
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::WouldBlock {
            return Ok(None);
        }
        return Err(error.into());
    }
    #[cfg(windows)]
    {
        if lock_windows(&file, true, false)? {
            Ok(Some(file))
        } else {
            Ok(None)
        }
    }
}

/// Byte-range lock over the whole (empty) lock file. Shared locks are compatible
/// with other acquisitions; exclusive locks serialize transfers and reclamation.
#[cfg(windows)]
fn lock_windows(file: &std::fs::File, exclusive: bool, wait: bool) -> std::io::Result<bool> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;
    let mut operation: OVERLAPPED = unsafe { std::mem::zeroed() };
    let mut flags = 0u32;
    if exclusive {
        flags |= LOCKFILE_EXCLUSIVE_LOCK;
    }
    if !wait {
        flags |= LOCKFILE_FAIL_IMMEDIATELY;
    }
    let acquired = unsafe { LockFileEx(file.as_raw_handle(), flags, 0, 1, 0, &mut operation) } != 0;
    if acquired {
        return Ok(true);
    }
    if wait {
        return Err(std::io::Error::last_os_error());
    }
    Ok(false)
}

fn source_selector(
    repository: &str,
    revision: &str,
    filename: Option<&str>,
    sizes: &[u32],
) -> String {
    serde_json::json!(["huggingface", repository, revision, filename, sizes]).to_string()
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct HubOrigin {
    repository: String,
    revision: String,
    commit: String,
    filename: Option<String>,
    sizes: Vec<u32>,
    expected_envelope: [u8; 32],
}

impl HubOrigin {
    fn from_reference(reference: &flare_helios::package::cache::SourceRef) -> Result<Self> {
        let (provider, repository_id, revision, filename, sizes): (
            String,
            String,
            String,
            Option<String>,
            Vec<u32>,
        ) = serde_json::from_str(reference.selector())?;
        if provider != "huggingface" || repository(&repository_id)?.1.is_some() {
            return Err("model origin is not a Hugging Face repository".into());
        }
        validate_relative(&revision)?;
        if let Some(filename) = &filename {
            choose_file([], Some(filename))?;
        }
        let expected_envelope = reference
            .expected_envelope()
            .ok_or_else(|| unbound_source(&repository_id, &revision))?;
        if sizes.is_empty() || sizes.contains(&0) {
            return Err("model origin has no concrete tensor-parallel selection".into());
        }
        Ok(Self {
            repository: repository_id,
            revision,
            commit: reference.revision().into(),
            filename,
            sizes,
            expected_envelope,
        })
    }
}

fn entry_origin(
    store: &flare_helios::package::cache::Root,
    entry: &flare_helios::package::cache::Entry,
) -> Result<Option<HubOrigin>> {
    if let Some(reference) = store.entry_source(entry)? {
        return HubOrigin::from_reference(&reference).map(Some);
    }
    // Source selectors can recover an origin only when they identify one acquisition unambiguously.
    let mut origins = std::collections::BTreeSet::new();
    for reference in store.entry_source_refs(entry)? {
        let value: serde_json::Value = match serde_json::from_str(reference.selector()) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if value.get(0).and_then(serde_json::Value::as_str) != Some("huggingface") {
            continue;
        }
        let fields: (String, String, String, Option<String>, Vec<u32>) =
            serde_json::from_value(value)?;
        if fields.4.is_empty() || fields.3.is_none() {
            continue;
        }
        origins.insert(HubOrigin::from_reference(&reference)?);
    }
    if origins
        .iter()
        .any(|origin| origin.revision != origin.commit)
    {
        origins.retain(|origin| origin.revision != origin.commit);
    }
    if origins.len() > 1 {
        return Err("model version has multiple recorded Hugging Face origins; download the intended source explicitly to record its origin".into());
    }
    let origin = origins.into_iter().next();
    if let Some(origin) = &origin {
        store.publish_entry_source(
            &source_selector(
                &origin.repository,
                &origin.revision,
                origin.filename.as_deref(),
                &origin.sizes,
            ),
            &origin.commit,
            entry,
            origin.expected_envelope,
        )?;
    }
    Ok(origin)
}

pub(crate) fn refresh_model(
    store: &flare_helios::package::cache::Root,
    entry: flare_helios::package::cache::Entry,
    pinned: bool,
    args: &crate::cli::Run,
) -> Result<(flare_helios::package::cache::Entry, Option<String>)> {
    let entry = refresh_model_at(
        "https://huggingface.co",
        store,
        entry,
        pinned,
        &args.hub,
        args.gpus.len().max(1),
        &args.publisher_keys,
        args.trust_store.as_deref(),
        &[],
    )?;
    let repository = if pinned || args.hub.offline {
        None
    } else {
        store
            .entry_source(&entry)?
            .map(|reference| HubOrigin::from_reference(&reference).map(|origin| origin.repository))
            .transpose()?
    };
    Ok((entry, repository))
}

#[allow(clippy::too_many_arguments)]
fn refresh_model_at(
    endpoint: &str,
    store: &flare_helios::package::cache::Root,
    entry: flare_helios::package::cache::Entry,
    pinned: bool,
    options: &HubOptions,
    gpu_count: usize,
    publisher_keys: &[PathBuf],
    trust_store: Option<&Path>,
    supplied_profiles: &[flare_helios::runtime_package::Publisher],
) -> Result<flare_helios::package::cache::Entry> {
    if pinned {
        if options.revision.is_some() || options.filename.is_some() {
            return Err("a local model version is pinned; use models download to select a Hugging Face revision".into());
        }
        return Ok(entry);
    }
    if options.offline && options.revision.is_none() && options.filename.is_none() {
        return Ok(entry);
    }
    let Some(origin) = entry_origin(store, &entry)? else {
        if options.revision.is_some() || options.filename.is_some() {
            return Err("this local model has no Hugging Face origin; download the intended repository explicitly".into());
        }
        return Ok(entry);
    };
    if options.offline {
        if options
            .revision
            .as_ref()
            .is_some_and(|revision| revision != &origin.revision && revision != &origin.commit)
            || options
                .filename
                .as_ref()
                .is_some_and(|filename| Some(filename) != origin.filename.as_ref())
        {
            return Err(
                "the selected local model does not match the requested offline source".into(),
            );
        }
    }
    let mut requested = options.clone();
    requested.model_cache = Some(store.path().to_owned());
    requested.revision.get_or_insert(origin.revision.clone());
    if requested.filename.is_none() {
        requested.filename = origin.filename.clone();
    }
    if requested.tensor_parallel.is_empty() {
        requested.tensor_parallel = origin.sizes.clone();
    }
    let primary = store.primary(&entry)?;
    let resolved = crate::trust::resolve_identity_with(
        &primary,
        publisher_keys,
        supplied_profiles,
        &requested,
        trust_store,
    )?;
    verify_offline_binding(
        &entry,
        origin.expected_envelope,
        &origin.repository,
        &origin.revision,
        &resolved.profiles,
    )?;
    if options.offline {
        return Ok(entry);
    }
    let access_token = token(options)?;
    // The update check never fetches content; it only needs a transport-scoped
    // client so no Hugging Face environment configuration can redirect it.
    let transport_cache = Transport::path(store.path(), options.cache_dir.as_deref());
    let api = ApiBuilder::from_cache(Cache::new(transport_cache))
        .with_endpoint(endpoint.into())
        .with_token(access_token.clone())
        .with_progress(false)
        .with_retries(0)
        .build()?;
    let revision = requested.revision.as_deref().unwrap();
    eprintln!("Checking model origin: {}@{revision}", origin.repository);
    let info = with_transient_retry(|| {
        api.repo(Repo::with_revision(
            origin.repository.clone(),
            RepoType::Model,
            revision.into(),
        ))
        .info()
    })
    .map_err(|error| format!("cannot check {}@{revision} for updates: {error}; the local version is unchanged (use --offline to skip the check)", origin.repository))?;
    if info.sha.len() != 40 || !info.sha.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("Hugging Face returned an invalid commit ID".into());
    }
    if info.sha == origin.commit
        && revision == origin.revision
        && requested.filename == origin.filename
        && requested.tensor_parallel == origin.sizes
    {
        return Ok(entry);
    }
    eprintln!("Updating model from {}@{revision}", origin.repository);
    download_for_model_at(
        endpoint,
        &origin.repository,
        &requested,
        gpu_count,
        access_token,
        publisher_keys,
        trust_store,
        supplied_profiles,
        Some(entry.model()),
        None,
    )
}

fn authenticated_source_envelope(
    primary: &Path,
    options: &HubOptions,
    gpu_count: usize,
    publishers: &[flare_helios::runtime_package::Publisher],
) -> Result<[u8; 32]> {
    use flare_helios::package::{conversion, reader};
    use flare_helios::runtime_package::VerifiedPackage;
    if conversion::form(primary)
        .map_err(|error| error.to_string())?
        .is_none()
    {
        return Ok(VerifiedPackage::open(primary, publishers, None)?.envelope_digest());
    }
    let metadata = reader::Metadata::open(primary)?;
    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    let selections = crate::cache_support::select_deployments(&metadata, publishers, &sizes)
        .map_err(|error| error.to_string())?;
    let selection = *selections
        .first()
        .ok_or("authenticated source has no selected deployment")?;
    // The signed envelope and selected deployment are sufficient to establish the
    // identity. Opening payload companions here would require materializing the
    // Hub snapshot and duplicate the acquisition's closure work.
    Ok(
        flare_helios::package::protection::verify(&metadata.envelope, publishers, selection)?
            .envelope_digest(),
    )
}

pub(crate) fn remember_download(
    root: &Path,
    repository: &str,
    revision: &str,
    commit: &str,
    filename: &str,
    options: &HubOptions,
    gpu_count: usize,
    entry: &flare_helios::package::cache::Entry,
    expected_envelope: [u8; 32],
) -> Result<()> {
    let store = flare_helios::package::cache::Root::open(root)?;
    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    let mut filenames = vec![Some(filename)];
    if options.filename.is_none() {
        filenames.push(None);
    }
    for revision in [revision, commit] {
        for filename in &filenames {
            // The exact selection retains earlier TP entries; the unqualified
            // selector also permits subsets of the most recently acquired union.
            for selection in [sizes.as_slice(), &[]] {
                store.publish_source_ref(
                    &source_selector(repository, revision, *filename, selection),
                    commit,
                    entry,
                    expected_envelope,
                )?;
            }
        }
    }
    store.publish_entry_source(
        &source_selector(repository, revision, Some(filename), &sizes),
        commit,
        entry,
        expected_envelope,
    )?;
    Ok(())
}

#[cfg(test)]
fn offline_entry(
    root: &Path,
    repository: &str,
    revision: &str,
    options: &HubOptions,
    gpu_count: usize,
) -> Result<flare_helios::package::cache::Entry> {
    offline_entry_with_expected(root, repository, revision, options, gpu_count)
        .map(|(entry, _)| entry)
}

fn unbound_source(repository: &str, revision: &str) -> String {
    format!(
        "cached source reference for {repository}@{revision} predates signed-envelope binding; refresh source online or load cached entry by explicit path"
    )
}

fn offline_entry_with_expected(
    root: &Path,
    repository: &str,
    revision: &str,
    options: &HubOptions,
    gpu_count: usize,
) -> Result<(flare_helios::package::cache::Entry, [u8; 32])> {
    let store = flare_helios::package::cache::Root::open(root)?;
    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    let unqualified = source_selector(repository, revision, options.filename.as_deref(), &[]);
    // The unqualified record establishes the latest resolved commit. Its own
    // expected envelope is not used when an exact-TP record supplies the entry.
    let Some(latest) = store.read_source_ref(&unqualified)? else {
        return Err(missing_source(repository, revision, options, &sizes, root).into());
    };
    let commit = latest.revision().to_owned();
    let exact = source_selector(repository, revision, options.filename.as_deref(), &sizes);
    // An exact TP entry is usable only at the last known revision. Preferring it
    // before the unqualified selector would let a moving branch fall back to an
    // older commit merely because that commit had the requested TP cached.
    if let Some((reference, entry)) = store.resolve_source_ref_record(&exact)? {
        if reference.revision() == commit {
            let Some(expected) = reference.expected_envelope() else {
                return Err(unbound_source(repository, revision).into());
            };
            return Ok((entry, expected));
        }
    }
    if let Some((reference, entry)) = store.resolve_source_ref_record(&unqualified)? {
        if reference.revision() == commit {
            let Some(expected) = reference.expected_envelope() else {
                return Err(unbound_source(repository, revision).into());
            };
            return Ok((entry, expected));
        }
    }
    Err(missing_source(repository, revision, options, &sizes, root).into())
}

fn verify_offline_binding(
    entry: &flare_helios::package::cache::Entry,
    expected: [u8; 32],
    repository: &str,
    revision: &str,
    publishers: &[flare_helios::runtime_package::Publisher],
) -> Result<()> {
    use flare_helios::package::{PayloadId, conversion, reader};
    use flare_helios::runtime_package::VerifiedPackage;
    let primary = entry_primary(entry)?;
    let actual = if conversion::form(&primary)
        .map_err(|error| error.to_string())?
        .is_none()
    {
        VerifiedPackage::open(&primary, publishers, None)?.envelope_digest()
    } else {
        let metadata = reader::Metadata::open(&primary)?;
        *PayloadId::of(&metadata.envelope).as_bytes()
    };
    if actual != expected {
        return Err(format!(
            "cached package for {repository}@{revision} does not match the signed envelope recorded for its source; refresh source online or load cached entry by explicit path"
        )
        .into());
    }
    Ok(())
}

fn missing_source(
    repository: &str,
    revision: &str,
    options: &HubOptions,
    sizes: &[u32],
    root: &Path,
) -> String {
    format!(
        "package {repository}@{revision} (filename: {}, TP: {sizes:?}) is not available in the Flare cache {}; download it online first",
        options.filename.as_deref().unwrap_or("automatic"),
        root.display(),
    )
}

fn enforce_legacy_tensor_parallel(
    verified: &flare_helios::runtime_package::VerifiedPackage,
    sizes: &[u32],
) -> Result<()> {
    let ranks = verified
        .identity()
        .map_err(|error| error.to_string())?
        .tensor_parallel;
    if sizes.len() != 1 || sizes[0] != ranks {
        return Err(format!(
            "legacy package provides exactly tensor-parallel size {ranks}, requested {sizes:?}"
        )
        .into());
    }
    Ok(())
}

fn verify_offline_entry(
    entry: &flare_helios::package::cache::Entry,
    options: &HubOptions,
    gpu_count: usize,
    publishers: &[flare_helios::runtime_package::Publisher],
) -> Result<()> {
    use flare_helios::{
        package::{conversion, reader},
        runtime_package::VerifiedPackage,
    };
    let primary = entry_primary(entry)?;
    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    if conversion::form(&primary)?.is_none() {
        let verified = VerifiedPackage::open(&primary, publishers, None)?;
        // A node-side download with neither an explicit TP request nor a GPU
        // count does not constrain a legacy package's signed rank topology.
        if !options.tensor_parallel.is_empty() || gpu_count > 0 {
            enforce_legacy_tensor_parallel(&verified, &sizes)?;
        }
        verified.verify_stored_data()?;
    } else {
        let metadata = reader::Metadata::open(&primary)?;
        let selections = crate::cache_support::select_deployments(&metadata, publishers, &sizes)
            .map_err(|error| error.to_string())?;
        for selection in selections {
            VerifiedPackage::open_selected(&primary, publishers, selection)?
                .verify_stored_data()?;
        }
    }
    Ok(())
}

fn entry_primary(entry: &flare_helios::package::cache::Entry) -> Result<PathBuf> {
    let files = std::fs::read_dir(entry.directory())?
        .map(|file| Ok(file?.file_name().to_string_lossy().into_owned()))
        .collect::<Result<Vec<_>>>()?;
    Ok(entry.directory().join(choose_file(files, None)?))
}

/// Verify stored signatures and compressed executable bytes, select signed deployments,
/// and install the ready union as one leased cache entry. No terms acceptance and no
/// executable decoding happen here; acceptance stays at load time.
/// Resolve a transport package into a local tree of real files.
///
/// Hub snapshots store each file as a symlink into the shared blob cache. The
/// package reader opens components with `O_NOFOLLOW`, so acquisition materializes
/// the signed location closure into acquisition staging: canonical targets are
/// hardlinked (or copied across devices) under the same relative paths, and the
/// tree is removed after installation. Nothing here is a persistent store.
fn resolve_transport(
    primary: &Path,
    names: &[String],
    model_root: &Path,
) -> Result<(PathBuf, Option<PathBuf>)> {
    let directory = primary.parent().unwrap_or(Path::new("."));
    let mut files = vec![
        primary
            .file_name()
            .ok_or("package primary has no file name")?
            .to_string_lossy()
            .into_owned(),
    ];
    files.extend(names.iter().cloned());
    files.sort();
    files.dedup();
    let symlinked = files.iter().any(|name| {
        std::fs::symlink_metadata(directory.join(name))
            .map(|info| info.file_type().is_symlink())
            .unwrap_or(false)
    });
    if !symlinked {
        return Ok((primary.to_owned(), None));
    }
    let staging = model_root.join("staging/acquisition");
    std::fs::create_dir_all(&staging)?;
    // One directory per acquisition, created exclusively so concurrent imports
    // never share a tree and cleanup only ever removes its own.
    let mut unique = 0u64;
    let tree = loop {
        let candidate = staging.join(format!("{}-{}", std::process::id(), unique));
        match std::fs::create_dir(&candidate) {
            Ok(()) => break candidate,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                unique += 1;
                if unique > 1 << 16 {
                    return Err("cannot allocate acquisition staging".into());
                }
            }
            Err(error) => return Err(format!("cannot create acquisition staging: {error}").into()),
        }
    };
    let result = (|| {
        for name in &files {
            let source = directory.join(name);
            let canonical = std::fs::canonicalize(&source)
                .map_err(|error| format!("package file is unavailable: {name}: {error}"))?;
            let target = tree.join(name);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            match std::fs::hard_link(&canonical, &target) {
                Ok(()) => {}
                Err(_) => {
                    std::fs::copy(&canonical, &target)?;
                }
            }
        }
        let name = primary
            .strip_prefix(directory)
            .map_err(|_| "package primary escapes its directory")?;
        Ok(tree.join(name))
    })();
    match result {
        Ok(resolved) => Ok((resolved, Some(tree))),
        Err(error) => {
            let _ = std::fs::remove_dir_all(&tree);
            Err(error)
        }
    }
}

fn legacy_key(repository: &str) -> Result<String> {
    use flare_helios::package::cache;
    Ok(cache::model_key(repository)?)
}

enum PayloadAcquisition<'a> {
    Ready(flare_helios::package::cache::Entry),
    Planned {
        plan: flare_helios::package::cache::AcquisitionPlan<'a>,
        staging: flare_helios::package::cache::Staging,
    },
}

fn acquisition_model_key(
    source: &flare_helios::package::cache::PackageSource<'_>,
    repository: &str,
) -> Result<String> {
    use flare_helios::package::cache;
    match source
        .verified()
        .first()
        .map(|verified| verified.package().model_identity.clone())
    {
        Some(identity) if !identity.is_empty() => {
            Ok(cache::model_key(&identity).map_err(|error| error.to_string())?)
        }
        _ => Ok(cache::model_key(repository).map_err(|error| error.to_string())?),
    }
}

fn signed_hardware(
    source: &flare_helios::package::cache::PackageSource<'_>,
) -> Result<(Option<String>, Vec<u32>)> {
    use flare_helios::package::naming;
    let mut sizes = std::collections::BTreeSet::new();
    let mut package_platform: Option<&str> = None;
    let mut conventional_platform: Option<&str> = None;
    let mut conventional = true;
    for verified in source.verified() {
        let deployment = verified
            .deployment()
            .ok_or("selected deployment lacks an authenticated deployment body")?;
        if deployment.tensor_parallel == 0 {
            return Err("signed deployment has a zero tensor-parallel size".into());
        }
        sizes.insert(deployment.tensor_parallel);

        let signed_platform = verified.package().hardware_identity.as_str();
        if naming::PLATFORMS.contains(&signed_platform) {
            if let Some(previous) = package_platform {
                if previous != signed_platform {
                    return Err(format!(
                        "selected packages span {previous} and {signed_platform}; names require one device identity"
                    )
                    .into());
                }
            } else {
                package_platform = Some(signed_platform);
            }
        }

        let parsed = deployment
            .deployment_id
            .rsplit_once("-tp")
            .and_then(|(device, suffix)| {
                if !naming::PLATFORMS.contains(&device) {
                    return None;
                }
                let tensor_parallel = suffix.parse::<u32>().ok()?;
                (tensor_parallel == deployment.tensor_parallel).then_some(device)
            });
        match parsed {
            Some(device) => {
                if let Some(previous) = conventional_platform {
                    if previous != device {
                        conventional = false;
                    }
                } else {
                    conventional_platform = Some(device);
                }
            }
            None => conventional = false,
        }
    }
    if sizes.is_empty() {
        return Err("selected package lacks an authenticated tensor-parallel size".into());
    }
    let platform =
        package_platform.or_else(|| conventional.then_some(conventional_platform).flatten());
    Ok((platform.map(str::to_owned), sizes.into_iter().collect()))
}

fn acquisition_candidate(platform: Option<&str>, sizes: &[u32]) -> Result<String> {
    flare_helios::package::naming::deployment_candidate(
        platform,
        sizes,
        std::time::SystemTime::now(),
    )
    .ok_or_else(|| "acquisition naming requires at least one non-zero TP size".into())
}

/// Build the shared payload acquisition plan before any dependency fetch.
///
/// The returned plan owns cached object leases and the minimal remote file set.
/// An already-ready immutable entry with a matching remembered source avoids
/// planning entirely. A new entry reserves its readable device/TP/time name
/// before remote transfer and keeps that reservation through install.
#[allow(clippy::too_many_arguments)]
fn plan_payload_acquisition<'a>(
    primary: &Path,
    options: &HubOptions,
    repository: &str,
    revision: &str,
    commit: &str,
    filename: &str,
    gpu_count: usize,
    publishers: &[flare_helios::runtime_package::Publisher],
    model_root: &Path,
    progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
) -> Result<PayloadAcquisition<'a>> {
    use flare_helios::package::{cache, derivation, reader};

    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    // Resolve the primary through transport only. Companions are fetched by the
    // plan's remote set, never by a location-name closure.
    let (resolved, cleanup) = resolve_transport(primary, &[], model_root)?;
    let result = (|| {
        let metadata = reader::Metadata::open(&resolved)?;
        let selections = crate::cache_support::select_deployments(&metadata, publishers, &sizes)
            .map_err(|error| error.to_string())?;
        let source = cache::PackageSource::open(&resolved, publishers, selections[0])?
            .union(publishers, &selections)?;
        let model_key = acquisition_model_key(&source, repository)?;
        let store = cache::Root::open(model_root)?;

        // Warm reuse is keyed by the private source selector, then checked
        // against the signed primary and selected deployments before the hit.
        let mut selectors = Vec::new();
        for source_revision in [revision, commit] {
            for source_filename in [Some(filename), None] {
                for selection in [sizes.as_slice(), &[]] {
                    selectors.push(source_selector(
                        repository,
                        source_revision,
                        source_filename,
                        selection,
                    ));
                }
            }
        }
        for selector in selectors {
            let Ok(Some((stored_revision, entry))) = store.resolve_source_ref(&selector) else {
                continue;
            };
            if stored_revision == commit
                && store.verify_ready_with(&entry, &source, progress).is_ok()
            {
                return Ok(PayloadAcquisition::Ready(entry));
            }
        }

        let (platform, signed_sizes) = signed_hardware(&source)?;
        let candidate = acquisition_candidate(platform.as_deref(), &signed_sizes)?;
        let staging = store.stage_named(&model_key, &candidate)?;
        let plan = store.plan_acquisition_with(source, &derivation::Limits::default(), progress)?;
        Ok(PayloadAcquisition::Planned { plan, staging })
    })();
    if let Some(tree) = cleanup {
        let _ = std::fs::remove_dir_all(tree);
    }
    result
}

fn finish_payload_acquisition(
    primary: &Path,
    names: &[String],
    mut plan: flare_helios::package::cache::AcquisitionPlan<'_>,
    staging: flare_helios::package::cache::Staging,
    model_root: &Path,
) -> Result<flare_helios::package::cache::Entry> {
    use flare_helios::package::cache;
    let (resolved, cleanup) = resolve_transport(primary, names, model_root)?;
    let result = (|| {
        plan.relocate_source(&resolved)?;
        cache::Root::open(model_root)?.install_planned_named(plan, staging)
    })();
    if let Some(tree) = cleanup {
        let _ = std::fs::remove_dir_all(tree);
    }
    result.map_err(|error| error.to_string().into())
}

fn file_digest(path: &Path) -> Result<[u8; 32]> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut context = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        context.update(&buffer[..count]);
    }
    Ok(context.finalize().into())
}

fn legacy_warm_entry(
    primary: &Path,
    options: &HubOptions,
    repository: &str,
    revision: &str,
    commit: &str,
    filename: &str,
    gpu_count: usize,
    publishers: &[flare_helios::runtime_package::Publisher],
    model_root: &Path,
) -> Result<Option<flare_helios::package::cache::Entry>> {
    use flare_helios::package::cache;
    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    let (resolved, cleanup) = resolve_transport(primary, &[], model_root)?;
    let require_tp = !options.tensor_parallel.is_empty() || gpu_count > 0;
    let result = (|| {
        let store = cache::Root::open(model_root)?;
        let model_key = legacy_key(repository)?;
        let digest = file_digest(&resolved)?;

        // Preserve existing package-hash legacy entries without fetching companions,
        // but apply the same signed closure and TP checks as named source refs.
        let old_id = format!(
            "{}-legacy",
            flare_helios::package::PayloadId::from_bytes(digest)
        );
        if let Ok(entry) = store.entry(&model_key, &old_id) {
            if let Ok(entry_primary) = entry_primary(&entry) {
                if file_digest(&entry_primary).is_ok_and(|candidate| candidate == digest) {
                    if let Ok(entry_verified) = flare_helios::runtime_package::VerifiedPackage::open(
                        &entry_primary,
                        publishers,
                        None,
                    ) {
                        let tp_ok = !require_tp
                            || entry_verified
                                .identity()
                                .is_ok_and(|identity| sizes == [identity.tensor_parallel]);
                        if tp_ok && entry_verified.verify_stored_data().is_ok() {
                            return Ok(Some(entry));
                        }
                    }
                }
            }
        }

        let mut selectors = Vec::new();
        for source_revision in [revision, commit] {
            for source_filename in [Some(filename), None] {
                for selection in [sizes.as_slice(), &[]] {
                    selectors.push(source_selector(
                        repository,
                        source_revision,
                        source_filename,
                        selection,
                    ));
                }
            }
        }
        for selector in selectors {
            let Ok(Some((stored_revision, entry))) = store.resolve_source_ref(&selector) else {
                continue;
            };
            if stored_revision != commit {
                continue;
            }
            let Ok(entry_primary) = entry_primary(&entry) else {
                continue;
            };
            if !file_digest(&entry_primary).is_ok_and(|candidate| candidate == digest) {
                continue;
            }
            let Ok(entry_verified) = flare_helios::runtime_package::VerifiedPackage::open(
                &entry_primary,
                publishers,
                None,
            ) else {
                continue;
            };
            let tp_ok = !require_tp
                || entry_verified
                    .identity()
                    .is_ok_and(|identity| sizes == [identity.tensor_parallel]);
            if tp_ok && entry_verified.verify_stored_data().is_ok() {
                return Ok(Some(entry));
            }
        }
        Ok(None)
    })();
    if let Some(tree) = cleanup {
        let _ = std::fs::remove_dir_all(tree);
    }
    result
}

fn install_ready(
    repository: &str,
    path: &Path,
    options: &HubOptions,
    model_root: &Path,
    gpu_count: usize,
    publishers: &[flare_helios::runtime_package::Publisher],
) -> Result<flare_helios::package::cache::Entry> {
    use flare_helios::package::{cache, conversion};

    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    if !conversion::form(path)
        .map_err(|error| error.to_string())?
        .is_none()
    {
        return match plan_payload_acquisition(
            path,
            options,
            repository,
            options.revision.as_deref().unwrap_or(""),
            "",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model.safetensors"),
            gpu_count,
            publishers,
            model_root,
            None,
        )? {
            PayloadAcquisition::Ready(entry) => Ok(entry),
            PayloadAcquisition::Planned { plan, staging } => {
                let names = plan.remote_files();
                finish_payload_acquisition(path, &names, plan, staging, model_root)
            }
        };
    }

    let names = flare_helios::runtime_package::package_companions(path)?;
    let (resolved, cleanup) = resolve_transport(path, &names, model_root)?;
    let install = (|| {
        let verified =
            flare_helios::runtime_package::VerifiedPackage::open(&resolved, publishers, None)?;
        if !options.tensor_parallel.is_empty() || gpu_count > 0 {
            enforce_legacy_tensor_parallel(&verified, &sizes)?;
        }
        let signed = verified.identity().map_err(|error| error.to_string())?;
        let platform =
            if flare_helios::package::naming::PLATFORMS.contains(&signed.hardware.as_str()) {
                Some(signed.hardware.as_str())
            } else {
                None
            };
        let candidate = acquisition_candidate(platform, &[signed.tensor_parallel])?;
        let store = cache::Root::open(model_root).map_err(|error| error.to_string())?;
        let model_key = legacy_key(repository)?;
        // Existing legacy entries keep their historical package-hash names.
        let primary_bytes = std::fs::read(&resolved).map_err(|error| error.to_string())?;
        let old_id = format!(
            "{}-legacy",
            flare_helios::package::PayloadId::of(&primary_bytes)
        );
        if let Ok(entry) = store.entry(&model_key, &old_id) {
            return Ok(entry);
        }
        let staging = store
            .stage_named(&model_key, &candidate)
            .map_err(|error| error.to_string())?;
        let entry = store
            .import_legacy_named(&resolved, staging)
            .map_err(|error| error.to_string())?;
        Ok(entry)
    })();
    if let Some(tree) = cleanup {
        let _ = std::fs::remove_dir_all(tree);
    }
    install
}

/// The access token named by `--token-file`, for private repositories.
pub(crate) fn token(options: &HubOptions) -> Result<Option<String>> {
    options
        .token_file
        .as_ref()
        .map(|path| {
            let value = std::fs::read_to_string(path)
                .map_err(|error| format!("cannot read token file {}: {error}", path.display()))?;
            let token = value.trim();
            if token.is_empty() {
                return Err("Hugging Face token file is empty".into());
            }
            Ok(token.to_owned())
        })
        .transpose()
}

#[cfg(test)]
fn fetch_primary(
    api: &Api,
    transport: &Transport,
    repo: &Repo,
    requested: Option<&str>,
) -> Result<(PathBuf, Repo)> {
    fetch_primary_with_progress(api, transport, repo, requested, None, None)
}

fn fetch_primary_with_progress(
    api: &Api,
    transport: &Transport,
    repo: &Repo,
    requested: Option<&str>,
    token: Option<&str>,
    progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
) -> Result<(PathBuf, Repo)> {
    let info = with_transient_retry(|| api.repo(repo.clone()).info())?;
    if info.sha.len() != 40 || !info.sha.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("Hugging Face returned an invalid commit ID".into());
    }
    let filename = choose_file(
        info.siblings.into_iter().map(|file| file.rfilename),
        requested,
    )?;
    let pinned = Repo::with_revision(repo.url(), RepoType::Model, info.sha.clone());
    eprintln!("Resolved package: {}@{} / {filename}", repo.url(), info.sha);
    let recovered = transport.adopt_legacy(&pinned.folder_name());
    if recovered > 0 {
        eprintln!(
            "Recovered {recovered} bytes of interrupted Hugging Face transfers from an older staging tree"
        );
    }
    let path = fetch_file_with_progress(api, transport, &pinned, &filename, token, progress)?;
    transport.record_ref(&repo.folder_name(), repo.revision(), &info.sha);
    Ok((path, pinned))
}

/// Fetch one file of an already resolved revision.
#[cfg(test)]
fn fetch_dependency(
    api: &Api,
    transport: &Transport,
    repo: &Repo,
    primary: &Path,
    name: &str,
) -> Result<()> {
    fetch_dependency_with_progress(api, transport, repo, primary, name, None, None)
}

fn fetch_dependency_with_progress(
    api: &Api,
    transport: &Transport,
    repo: &Repo,
    primary: &Path,
    name: &str,
    token: Option<&str>,
    progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
) -> Result<()> {
    let filename = primary
        .strip_prefix(transport.snapshot_dir(&repo.folder_name(), repo.revision()))
        .unwrap_or(primary);
    let base = Path::new(filename).parent().unwrap_or(Path::new(""));
    let relative = base.join(name).to_string_lossy().replace('\\', "/");
    fetch_file_with_progress(api, transport, repo, &relative, token, progress)?;
    Ok(())
}

/// Files the downloaded primary depends on.
///
/// Payload-format packages name their companions through signed location records,
/// so only the selected closure is fetched. Legacy packages keep their own
/// companion enumeration.
fn dependency_files(
    primary: &Path,
    options: &HubOptions,
    gpu_count: usize,
    publishers: &[flare_helios::runtime_package::Publisher],
) -> Result<Vec<String>> {
    use flare_helios::package::{derivation, protection, reader};

    let Ok(metadata) = reader::Metadata::open(primary) else {
        return Ok(flare_helios::runtime_package::package_companions(primary)?);
    };
    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    let selections = crate::cache_support::select_deployments(&metadata, publishers, &sizes)
        .map_err(|error| error.to_string())?;
    let placed: std::collections::BTreeMap<Vec<u8>, String> = metadata
        .locations
        .locations
        .iter()
        .map(|location| (location.payload_id.clone(), location.file.clone()))
        .collect();
    // The shared cache closure decides which payload identities this selection
    // needs: its required payloads plus the derivation sources they depend on,
    // authorized only by the selected deployments' catalogs.
    let mut selected = Vec::with_capacity(selections.len());
    for selection in &selections {
        selected.push(
            protection::verify(&metadata.envelope, publishers, *selection)
                .map_err(|error| error.to_string())?,
        );
    }
    let needed = flare_helios::package::cache::selected_derivation_closure(
        &metadata,
        &selected,
        &derivation::Limits::default(),
    )
    .map_err(|error| error.to_string())?;
    let mut files: Vec<String> = needed
        .iter()
        .filter_map(|id| placed.get(id))
        .filter(|file| !file.is_empty())
        .cloned()
        .collect();
    files.sort();
    files.dedup();
    Ok(files)
}

/// Reuse the transport's content-addressed file even when this revision has no
/// snapshot entry yet.
#[cfg(test)]
fn fetch_file(api: &Api, transport: &Transport, pinned: &Repo, filename: &str) -> Result<PathBuf> {
    fetch_file_with_progress(api, transport, pinned, filename, None, None)
}

fn fetch_file_with_progress(
    api: &Api,
    transport: &Transport,
    pinned: &Repo,
    filename: &str,
    token: Option<&str>,
    progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
) -> Result<PathBuf> {
    let path = transport
        .snapshot_dir(&pinned.folder_name(), pinned.revision())
        .join(filename);
    if path.is_file() {
        return Ok(path);
    }
    let remote = api.repo(pinned.clone());
    let metadata = with_identity_retry(|| file_identity(&remote.url(filename), token))?;
    if metadata.commit_hash() != pinned.revision() {
        return Err("Hugging Face returned an invalid file identity".into());
    }
    let key = blob_key(metadata.etag())?;
    let folder = pinned.folder_name();
    let blob = transport.blob(&folder, &key);
    // Serialize the lookup and transfer of one content identity. Concurrent
    // downloads of the same bytes keep one inode and one transfer.
    let _lock = content_lock(&transport.lock(&folder, &key))?;
    if path.is_file() {
        return Ok(path);
    }
    match std::fs::symlink_metadata(&blob) {
        Ok(info) => {
            if !info.is_file() || info.len() != metadata.size() {
                return Err(format!("invalid cached Hugging Face blob: {}", blob.display()).into());
            }
            // A stated SHA-256 identity is checked before the bytes are reused,
            // so a size-correct but corrupt transport file cannot poison an
            // install. Opaque identities are checked by signed validation.
            if let Some(digest) = content_digest(metadata.etag())
                && !blob_matches(&blob, metadata.size(), digest, progress)?
            {
                quarantine(&blob)?;
                eprintln!(
                    "Discarded corrupt transport content for {filename}: its bytes do not match the Hub digest"
                );
                eprintln!("Downloading model file: {filename}...");
                transfer_blob(
                    &transport.agent,
                    &transport.part(&folder, &key),
                    &blob,
                    &remote.url(filename),
                    token,
                    filename,
                    &metadata,
                    progress,
                )?;
            } else {
                eprintln!("Reusing cached model file: {filename}");
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let adopted = match (cached_identity(metadata.etag()), &transport.hub_cache) {
                (Some(identity), Some(hub_cache)) => adopt_hub_file(
                    hub_cache,
                    &folder,
                    pinned.revision(),
                    filename,
                    metadata.size(),
                    identity,
                    &key,
                    &blob,
                    progress,
                )?,
                _ => false,
            };
            if !adopted {
                eprintln!("Downloading model file: {filename}...");
                transfer_blob(
                    &transport.agent,
                    &transport.part(&folder, &key),
                    &blob,
                    &remote.url(filename),
                    token,
                    filename,
                    &metadata,
                    progress,
                )?;
            }
        }
        Err(error) => return Err(error.into()),
    }
    link_snapshot(
        &transport.snapshot_dir(&folder, pinned.revision()),
        filename,
        &blob,
    )?;
    Ok(path)
}

/// Content identity a Hugging Face cache file must satisfy before reuse.
///
/// The Hub states a file's identity in its ETag: large files carry the LFS
/// SHA-256, either bare or `sha256:`-prefixed, and repository files tracked by Git
/// carry the Git blob identity. Anything else is opaque: it certifies nothing, so
/// the file is downloaded instead of reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CachedIdentity {
    Sha256([u8; 32]),
    GitBlob([u8; 20]),
}

/// Identity a Hub ETag states for one file, when it states one Flare can check.
fn cached_identity(etag: &str) -> Option<CachedIdentity> {
    let etag = etag.trim().trim_matches('"');
    if let Some(hex) = etag.strip_prefix("sha256:") {
        return parse_identity(hex, 32).map(|bytes| {
            let mut digest = [0u8; 32];
            digest.copy_from_slice(&bytes);
            CachedIdentity::Sha256(digest)
        });
    }
    if etag.len() == 64 {
        return parse_identity(etag, 32).map(|bytes| {
            let mut digest = [0u8; 32];
            digest.copy_from_slice(&bytes);
            CachedIdentity::Sha256(digest)
        });
    }
    if etag.len() == 40 {
        return parse_identity(etag, 20).map(|bytes| {
            let mut digest = [0u8; 20];
            digest.copy_from_slice(&bytes);
            CachedIdentity::GitBlob(digest)
        });
    }
    None
}

fn parse_identity(hex: &str, bytes: usize) -> Option<Vec<u8>> {
    if hex.len() != bytes * 2 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let mut digest = Vec::with_capacity(bytes);
    for pair in hex.as_bytes().chunks_exact(2) {
        let text = std::str::from_utf8(pair).ok()?;
        digest.push(u8::from_str_radix(text, 16).ok()?);
    }
    Some(digest)
}

/// Hashes a cache file the way its stated identity is defined.
enum ReuseHasher {
    Sha256(ring::digest::Context),
    GitBlob(ring::digest::Context),
}

impl ReuseHasher {
    fn new(identity: CachedIdentity, size: u64) -> Self {
        match identity {
            CachedIdentity::Sha256(_) => {
                Self::Sha256(ring::digest::Context::new(&ring::digest::SHA256))
            }
            // A Git blob identity covers the header and the complete content.
            CachedIdentity::GitBlob(_) => {
                let mut context =
                    ring::digest::Context::new(&ring::digest::SHA1_FOR_LEGACY_USE_ONLY);
                context.update(format!("blob {size}\0").as_bytes());
                Self::GitBlob(context)
            }
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        match self {
            Self::Sha256(context) | Self::GitBlob(context) => context.update(bytes),
        }
    }

    fn matches(self, identity: CachedIdentity) -> bool {
        match (self, identity) {
            (Self::Sha256(context), CachedIdentity::Sha256(expected)) => {
                context.finish().as_ref() == expected
            }
            (Self::GitBlob(context), CachedIdentity::GitBlob(expected)) => {
                context.finish().as_ref() == expected
            }
            _ => false,
        }
    }
}

/// Borrow one complete prepared-package file from the user's Hugging Face cache.
///
/// The cache is a read-only source: this opens the pinned commit's snapshot entry
/// for reading and never creates, renames, links, locks, or changes permissions
/// under the cache directory. The file must have the size the Hub reports and
/// hash to the identity its ETag states. Verified bytes are copied beside the
/// transport blob and renamed into place, so snapshot linking and every
/// signature check treat them exactly like a downloaded body.
fn adopt_hub_file(
    hub_cache: &Path,
    folder: &str,
    commit: &str,
    filename: &str,
    size: u64,
    identity: CachedIdentity,
    key: &str,
    blob: &Path,
    progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
) -> Result<bool> {
    let source = hub_cache
        .join(folder)
        .join("snapshots")
        .join(commit)
        .join(filename);
    match std::fs::metadata(&source) {
        Ok(metadata) if metadata.is_file() && metadata.len() == size => {}
        _ => return Ok(false),
    }
    let mut input = match std::fs::File::open(&source) {
        Ok(input) => input,
        Err(_) => return Ok(false),
    };
    // Never `<key>.part`: that partial preserves a failed transfer of this same
    // content and resumes from it in a later invocation.
    let mut copy = AdoptedCopy::create(blob, key)?;
    let mut hasher = ReuseHasher::new(identity, size);
    let mut buffer = vec![0u8; TRANSFER_BUFFER];
    let mut written = 0u64;
    loop {
        let count = match input.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Ok(false),
        };
        if written + count as u64 > size {
            return Ok(false);
        }
        hasher.update(&buffer[..count]);
        copy.file.write_all(&buffer[..count])?;
        written += count as u64;
        if let Some(progress) = progress {
            progress.record_local_source_bytes(count as u64);
            progress.record_verification_bytes(count as u64);
        }
    }
    if written != size || !hasher.matches(identity) {
        eprintln!(
            "Ignoring Hugging Face cache file {filename}: it does not match its published identity"
        );
        return Ok(false);
    }
    copy.publish(blob)?;
    eprintln!("Reusing Hugging Face cache file: {filename}");
    Ok(true)
}

/// One adopted copy being written beside its final blob.
///
/// Dropping an unpublished copy removes it, so a failure never leaves bytes that
/// a transport collector cannot attribute to a content key.
struct AdoptedCopy {
    file: std::fs::File,
    path: PathBuf,
}

impl AdoptedCopy {
    fn create(blob: &Path, key: &str) -> Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = blob
            .parent()
            .ok_or("adopted payload has no parent directory")?;
        std::fs::create_dir_all(directory)?;
        loop {
            // The key identifies the content for collection and diagnosis; a
            // crashed copy is reclaimed under the same content lock.
            let path = directory.join(format!(
                ".{key}.adopt-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => return Ok(Self { file, path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
    }

    fn publish(self, blob: &Path) -> Result<()> {
        self.file.sync_all()?;
        std::fs::rename(&self.path, blob)?;
        if let Some(parent) = blob.parent() {
            sync_directory(parent)?;
        }
        Ok(())
    }
}

impl Drop for AdoptedCopy {
    fn drop(&mut self) {
        if std::fs::symlink_metadata(&self.path).is_ok() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Append-only state for one resumable transfer.
struct Resume {
    file: std::fs::File,
    offset: u64,
    size: u64,
    digest: Option<sha2::Sha256>,
}

impl Resume {
    fn open(
        path: &Path,
        size: u64,
        verify: bool,
        progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
    ) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(path)?;
        let mut resume = Self {
            file,
            offset: 0,
            size,
            digest: verify.then(sha2::Sha256::new),
        };
        let length = resume.file.metadata()?.len();
        if length > size {
            // A partial larger than its declared content is not ours to
            // truncate; the caller quarantines it before opening this resume.
            return Err(format!(
                "partial {} holds {length} bytes for content declared as {size} bytes",
                path.display()
            )
            .into());
        }
        resume.offset = length;
        if resume.offset > 0
            && let Some(digest) = resume.digest.as_mut()
        {
            let mut buffer = vec![0u8; TRANSFER_BUFFER];
            let mut remaining = resume.offset;
            while remaining > 0 {
                let limit = remaining.min(TRANSFER_BUFFER as u64) as usize;
                let count = resume.file.read(&mut buffer[..limit])?;
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
                if let Some(progress) = progress {
                    progress.record_verification_bytes(count as u64);
                }
                remaining -= count as u64;
            }
            if remaining > 0 {
                return Err(format!(
                    "partial {} shrank while verifying its resume prefix",
                    path.display()
                )
                .into());
            }
        }
        Ok(resume)
    }

    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        self.file.write_all(bytes)?;
        if let Some(digest) = self.digest.as_mut() {
            digest.update(bytes);
        }
        self.offset += bytes.len() as u64;
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file.sync_data()?;
        Ok(())
    }

    fn take_digest(&mut self) -> Option<[u8; 32]> {
        self.digest.take().map(|digest| digest.finalize().into())
    }
}

/// Why one transfer attempt ended without completing the file.
enum Attempt {
    /// The server or content cannot satisfy the request; retrying cannot help.
    Fatal(String),
    /// The attempt made no usable progress but may succeed when resumed.
    Retry {
        message: String,
        /// `Retry-After` requested by the server, for throttling responses.
        retry_after: Option<Duration>,
    },
}

impl Attempt {
    fn retry(message: impl Into<String>) -> Self {
        Self::Retry {
            message: message.into(),
            retry_after: None,
        }
    }
}

/// Delay before the next attempt: bounded exponential backoff, extended to the
/// server's `Retry-After` when it asks for a longer pause.
fn retry_delay(failures: usize, retry_after: Option<Duration>) -> Duration {
    let exponent = failures.saturating_sub(1).min(8) as u32;
    let backoff =
        (TRANSFER_BACKOFF_BASE * 2u32.saturating_pow(exponent)).min(TRANSFER_BACKOFF_LIMIT);
    match retry_after {
        Some(after) => after.min(TRANSFER_RETRY_AFTER_LIMIT).max(backoff),
        None => backoff,
    }
}

/// Whether a streaming transfer should sync its partial now.
fn part_sync_due(synced: u64, offset: u64, since_sync: Duration) -> bool {
    offset.saturating_sub(synced) >= PART_SYNC_BYTES || since_sync >= PART_SYNC_INTERVAL
}

/// The delay a `Retry-After` header requests: either delta-seconds or an
/// HTTP-date. A date at or before now requests no wait.
fn retry_after(response: &ureq::http::Response<ureq::Body>) -> Option<Duration> {
    let value = response.headers().get("retry-after")?.to_str().ok()?;
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs() as i64;
    retry_after_date(value, now)
}

/// `Retry-After: Sun, 06 Nov 1994 08:49:37 GMT` as a delay from `now_unix`.
///
/// The header is attacker-controlled, so parsing and date arithmetic belong to
/// `httpdate`, which validates fields and rejects overflow. A date at or before
/// `now_unix` requests no wait.
fn retry_after_date(value: &str, now_unix: i64) -> Option<Duration> {
    let date = httpdate::parse_http_date(value).ok()?;
    let target = date.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
    let target = i128::from(target);
    let now = i128::from(now_unix);
    Some(Duration::from_secs(
        (target - now).clamp(0, i128::from(u64::MAX)) as u64,
    ))
}

fn attempt_error(error: ureq::Error) -> Attempt {
    use ureq::Error;
    match error {
        Error::Timeout(_) | Error::Io(_) | Error::HostNotFound | Error::ConnectionFailed => {
            Attempt::retry(transport_error(&error))
        }
        Error::BadUri(_) => Attempt::Fatal(transport_error(&error)),
        error => Attempt::Fatal(transport_error(&error)),
    }
}

/// A transport error description that never repeats a request URL.
///
/// Request URLs can carry presigned query strings or embedded credentials, so a
/// rejected URI is reported by category rather than by value.
fn transport_error(error: &ureq::Error) -> String {
    match error {
        ureq::Error::BadUri(_) => "invalid request URL".to_owned(),
        error => error.to_string(),
    }
}

fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let range = value.trim().strip_prefix("bytes ")?;
    let (range, total) = range.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    Some((
        start.trim().parse().ok()?,
        end.trim().parse().ok()?,
        total.trim().parse().ok()?,
    ))
}

/// Whether the token may be sent to `current` for a request that started at
/// `origin`: same scheme, host, and port, as parsed URLs.
fn authenticated_redirect(origin: &str, current: &str) -> bool {
    match (url::Url::parse(origin), url::Url::parse(current)) {
        (Ok(origin), Ok(current)) => {
            same_origin(&origin, &current)
                && current.username().is_empty()
                && current.password().is_none()
        }
        _ => false,
    }
}

/// Scheme, host, and effective port equality.
fn same_origin(left: &url::Url, right: &url::Url) -> bool {
    left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

/// Resolve a redirect target with `url::Url::join` and accept only `http` or
/// `https` targets without embedded credentials.
fn redirect_target(current: &str, location: &str) -> Option<String> {
    let base = url::Url::parse(current).ok()?;
    let target = base.join(location).ok()?;
    if !matches!(target.scheme(), "http" | "https") {
        return None;
    }
    if !target.username().is_empty() || target.password().is_some() {
        return None;
    }
    Some(target.into())
}

/// One ranged request, following redirects itself so authentication is explicit:
/// the token goes to the original host only and is never sent to a content host.
fn ranged_get(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    offset: u64,
) -> std::result::Result<ureq::http::Response<ureq::Body>, Attempt> {
    let mut current = url.to_owned();
    for redirect in 0..=5 {
        let mut request = agent
            .get(&current)
            .header("Range", &format!("bytes={offset}-"))
            .header("Accept-Encoding", "identity");
        if let Some(token) = token
            && authenticated_redirect(url, &current)
        {
            request = request.header("Authorization", &format!("Bearer {token}"));
        }
        let response = request.call().map_err(attempt_error)?;
        let status = response.status().as_u16();
        if (300..400).contains(&status) {
            if redirect == 5 {
                return Err(Attempt::retry("too many redirects while fetching content"));
            }
            let Some(location) = response
                .headers()
                .get("location")
                .and_then(|value| value.to_str().ok())
            else {
                return Err(Attempt::retry(
                    "server returned a redirect without a Location header",
                ));
            };
            let Some(target) = redirect_target(&current, location) else {
                return Err(Attempt::Fatal(
                    "server redirected to an unsupported target".to_owned(),
                ));
            };
            current = target;
            continue;
        }
        return Ok(response);
    }
    Err(Attempt::retry("too many redirects while fetching content"))
}

/// Fetch the missing suffix of one content file and append it to its partial.
///
/// The response must describe exactly the bytes it contains: a resumed request
/// requires `206` with a `Content-Range` that starts at the current offset and
/// names the file's declared length. A response that cannot be appended is
/// reported with the saved offset; the partial bytes are never discarded here.
fn attempt_transfer(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    resume: &mut Resume,
    progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
) -> std::result::Result<(), Attempt> {
    let offset = resume.offset;
    let local = |error: String| Attempt::Fatal(format!("cannot write the partial file: {error}"));
    let response = ranged_get(agent, url, token, offset)?;
    let status = response.status().as_u16();
    let body_length = match status {
        206 => {
            let header = response
                .headers()
                .get("content-range")
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| Attempt::retry("206 response without Content-Range"))?;
            let Some((start, end, total)) = parse_content_range(header) else {
                return Err(Attempt::retry(format!(
                    "malformed Content-Range {header:?}"
                )));
            };
            if total != resume.size {
                return Err(Attempt::retry(format!(
                    "server declared {total} bytes for content declared as {} bytes",
                    resume.size
                )));
            }
            if end < start || end >= total {
                return Err(Attempt::retry(format!(
                    "content range {header:?} is not satisfiable"
                )));
            }
            if start != offset {
                return Err(Attempt::retry(format!(
                    "server answered with bytes {start}-{end} of {total} instead of resuming at the requested byte {offset}; kept {offset} saved bytes"
                )));
            }
            end - start + 1
        }
        200 => {
            let declared = response.headers().get("content-length").and_then(|value| {
                value
                    .to_str()
                    .ok()
                    .and_then(|text| text.parse::<u64>().ok())
            });
            if declared != Some(resume.size) {
                return Err(Attempt::retry(format!(
                    "server sent the complete content without a declared length of {} bytes",
                    resume.size
                )));
            }
            if offset > 0 {
                return Err(Attempt::retry(format!(
                    "server sent the complete {declared:?}-byte body instead of resuming at byte {offset}; kept {offset} saved bytes"
                )));
            }
            resume.size
        }
        416 => {
            return Err(Attempt::retry(format!(
                "server rejected the resume range at byte {offset}"
            )));
        }
        // Throttling responses may name the pause they want; 5xx and request
        // timeouts use bounded exponential backoff.
        429 | 503 => {
            return Err(Attempt::Retry {
                message: format!("HTTP {status}"),
                retry_after: retry_after(&response),
            });
        }
        408 | 500..=599 => return Err(Attempt::retry(format!("HTTP {status}"))),
        status => return Err(Attempt::Fatal(format!("HTTP {status}"))),
    };
    if status == 206
        && let Some(length) = response.headers().get("content-length").and_then(|value| {
            value
                .to_str()
                .ok()
                .and_then(|text| text.parse::<u64>().ok())
        })
        && length != body_length
    {
        return Err(Attempt::retry(format!(
            "Content-Length {length} does not match the declared range of {body_length} bytes"
        )));
    }
    let body = response.into_body();
    let mut reader = body.into_reader();
    let mut buffer = vec![0u8; TRANSFER_BUFFER];
    let mut written = 0u64;
    let mut synced = resume.offset;
    let mut synced_at = std::time::Instant::now();
    loop {
        let count = match reader.read(&mut buffer) {
            Ok(count) => count,
            Err(error) => {
                return Err(Attempt::retry(format!(
                    "body interrupted after {written} of {body_length} bytes: {error}"
                )));
            }
        };
        if count == 0 {
            break;
        }
        if written + count as u64 > body_length {
            return Err(Attempt::retry(format!(
                "body exceeds the {body_length} bytes its response declared"
            )));
        }
        resume
            .append(&buffer[..count])
            .map_err(|error| local(error.to_string()))?;
        written += count as u64;
        if let Some(progress) = progress {
            progress.record_network_bytes(count as u64);
        }
        if part_sync_due(synced, resume.offset, synced_at.elapsed()) {
            resume.sync().map_err(|error| local(error.to_string()))?;
            synced = resume.offset;
            synced_at = std::time::Instant::now();
        }
    }
    if written != body_length {
        return Err(Attempt::retry(format!(
            "body ended after {written} of {body_length} bytes"
        )));
    }
    Ok(())
}

/// Identity and size the Hub states for one repository file.
struct FileIdentity {
    commit: String,
    etag: String,
    size: u64,
}

impl FileIdentity {
    fn commit_hash(&self) -> &str {
        &self.commit
    }

    fn etag(&self) -> &str {
        &self.etag
    }

    fn size(&self) -> u64 {
        self.size
    }
}

/// One file identity probe that never panics on a malformed response.
#[derive(Debug)]
struct IdentityFailure {
    message: String,
    transient: bool,
}

impl IdentityFailure {
    fn permanent(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            transient: false,
        }
    }
}

/// Read the size and identity the Hub states for `url`.
///
/// The request is the Hub's own probe (`Range: bytes=0-0`). Redirects are
/// followed one at a time with a bound. The canonical identity (`x-linked-etag`
/// or `etag`, plus `x-repo-commit`) is captured only from the repository origin
/// and never overwritten: a content host may state a different ETag for the same
/// bytes, and trusting it would change cache identity. The declared size comes
/// from the final response's `Content-Range`. A redirect without `Location` is
/// reported as an error rather than trusted, and the token is only sent to the
/// original host.
fn file_identity(
    url: &str,
    token: Option<&str>,
) -> std::result::Result<FileIdentity, IdentityFailure> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .max_redirects(0)
        .redirect_auth_headers(ureq::config::RedirectAuthHeaders::Never)
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(120)))
        .timeout_global(Some(Duration::from_secs(120)))
        .build()
        .into();
    let base = url::Url::parse(url).map_err(|error| {
        IdentityFailure::permanent(format!("invalid file identity URL: {error}"))
    })?;
    let mut current = url.to_owned();
    let mut commit = None;
    let mut etag = None;
    let mut identity_captured = false;
    for redirect in 0..=5 {
        let mut request = agent
            .get(&current)
            .header("Range", "bytes=0-0")
            .header("Accept-Encoding", "identity");
        if current
            .parse::<url::Url>()
            .is_ok_and(|target| same_origin(&base, &target))
            && let Some(token) = token
        {
            request = request.header("Authorization", &format!("Bearer {token}"));
        }
        let response = request.call().map_err(|error| {
            use ureq::Error;
            let transient = matches!(
                error,
                Error::Timeout(_) | Error::Io(_) | Error::HostNotFound | Error::ConnectionFailed
            ) || matches!(error, Error::StatusCode(status) if matches!(status, 408 | 429 | 500..=599));
            IdentityFailure {
                message: format!("file identity request failed: {}", transport_error(&error)),
                transient,
            }
        })?;
        let status = response.status().as_u16();
        // Only the repository response states the canonical identity. Content
        // hosts (and off-origin redirect targets) may send their own ETag.
        if !identity_captured
            && current
                .parse::<url::Url>()
                .is_ok_and(|target| same_origin(&base, &target))
            && let Some(value) = response
                .headers()
                .get("x-linked-etag")
                .or_else(|| response.headers().get("etag"))
                .and_then(|value| value.to_str().ok())
        {
            etag = Some(value.trim().replace('"', ""));
            if let Some(value) = response
                .headers()
                .get("x-repo-commit")
                .and_then(|value| value.to_str().ok())
            {
                commit = Some(value.trim().to_owned());
            }
            identity_captured = true;
        }
        if (300..400).contains(&status) {
            if redirect == 5 {
                return Err(IdentityFailure::permanent(
                    "Hugging Face redirected too many times while resolving a file identity",
                ));
            }
            let Some(location) = response
                .headers()
                .get("location")
                .and_then(|value| value.to_str().ok())
            else {
                return Err(IdentityFailure::permanent(
                    "Hugging Face returned a redirect without a Location header while resolving a file identity",
                ));
            };
            let Some(target) = redirect_target(&current, location) else {
                return Err(IdentityFailure::permanent(
                    "Hugging Face redirected a file identity to an unsupported target",
                ));
            };
            current = target;
            continue;
        }
        if !(200..300).contains(&status) {
            return Err(IdentityFailure {
                message: format!("Hugging Face returned HTTP {status} for a file identity"),
                transient: matches!(status, 408 | 429 | 500..=599),
            });
        }
        let size = response
            .headers()
            .get("content-range")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| {
                value
                    .rsplit_once('/')
                    .map(|(_, total)| total.trim().to_owned())
            })
            .and_then(|total| total.parse::<u64>().ok());
        let (Some(commit), Some(etag), Some(size)) = (commit, etag, size) else {
            return Err(IdentityFailure::permanent(
                "Hugging Face returned an incomplete file identity",
            ));
        };
        return Ok(FileIdentity { commit, etag, size });
    }
    unreachable!("the bounded redirect loop returns or errors")
}

/// Probe one file identity with bounded backoff for transient transport failures.
fn with_identity_retry(
    mut operation: impl FnMut() -> std::result::Result<FileIdentity, IdentityFailure>,
) -> Result<FileIdentity> {
    let mut attempt = 0usize;
    loop {
        match operation() {
            Ok(identity) => return Ok(identity),
            Err(failure) => {
                attempt += 1;
                if attempt >= TRANSFER_STALLED_ATTEMPTS || !failure.transient {
                    return Err(failure.message.into());
                }
                eprintln!(
                    "Retrying after a transient Hugging Face error: {}",
                    failure.message
                );
                std::thread::sleep(retry_delay(attempt, None));
            }
        }
    }
}

/// Errors from Hub resolution or file identity that may succeed when retried.
/// Authentication, missing repositories, missing files, and malformed responses
/// are permanent and are never retried, so the signed binding is unchanged.
fn transient_hub_error(error: &hf_hub::api::sync::ApiError) -> bool {
    use hf_hub::api::sync::ApiError;
    let ApiError::RequestError(request) = error else {
        return false;
    };
    match request.as_ref() {
        ureq::Error::Timeout(_)
        | ureq::Error::Io(_)
        | ureq::Error::HostNotFound
        | ureq::Error::ConnectionFailed => true,
        ureq::Error::StatusCode(status) => matches!(status, 408 | 429 | 500..=599),
        _ => false,
    }
}

/// Run one Hub metadata operation with bounded backoff. The operation is
/// idempotent: resolution and file identity are read-only requests.
fn with_transient_retry<T>(
    mut operation: impl FnMut() -> std::result::Result<T, hf_hub::api::sync::ApiError>,
) -> Result<T> {
    let mut attempt = 0usize;
    loop {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error) => {
                attempt += 1;
                if attempt >= TRANSFER_STALLED_ATTEMPTS || !transient_hub_error(&error) {
                    return Err(error.into());
                }
                eprintln!("Retrying after a transient Hugging Face error: {error}");
                std::thread::sleep(retry_delay(attempt, None));
            }
        }
    }
}

/// Whether an error reports that acquired content failed verification, which is
/// the only evidence that this acquisition's transport copy may be unusable.
fn content_failure(error: &str) -> bool {
    [
        "payload digest",
        "does not match its declared identity",
        "differs from its commitment",
        "package signature verification failed",
    ]
    .iter()
    .any(|phrase| error.contains(phrase))
}

/// Leave every transport byte intact and tell the operator how to reclaim the
/// suspect copy. Identification is not assumed: only an error that names failed
/// verification gets this guidance.
fn transport_hint(root: &Path, error: Box<dyn std::error::Error>) -> Box<dyn std::error::Error> {
    let text = error.to_string();
    if !content_failure(&text) {
        return error;
    }
    format!(
        "{text}; no transport bytes were removed. Inspect {} with `flare models downloads` and reclaim retained copies with `flare models gc --downloads`",
        root.display()
    )
    .into()
}

/// Whether a complete transport file hashes to `digest`. The read is recorded as
/// verification work, not network traffic.
fn blob_matches(
    blob: &Path,
    size: u64,
    digest: [u8; 32],
    progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
) -> Result<bool> {
    let mut file = std::fs::File::open(blob)?;
    let mut hasher = sha2::Sha256::new();
    let mut buffer = vec![0u8; TRANSFER_BUFFER];
    let mut read = 0u64;
    loop {
        let count = match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        };
        hasher.update(&buffer[..count]);
        read += count as u64;
        if let Some(progress) = progress {
            progress.record_verification_bytes(count as u64);
        }
        if read > size {
            return Ok(false);
        }
    }
    Ok(read == size && hasher.finalize().as_slice() == digest)
}

/// Move a partial out of the way without losing its bytes.
///
/// Quarantined files stay beside the transport for diagnosis and are removed by
/// `flare models gc --downloads`; a transfer never truncates saved content.
fn quarantine(part: &Path) -> Result<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = part
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let target = part.with_file_name(format!(
        ".{name}.corrupt-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::rename(part, &target)?;
    if let Some(parent) = target.parent() {
        sync_directory(parent)?;
    }
    Ok(target)
}

/// Complete one content file, resuming its partial and validating its length
/// and, when the Hub publishes a SHA-256 ETag, its content digest.
fn transfer_blob(
    agent: &ureq::Agent,
    part: &Path,
    blob: &Path,
    url: &str,
    token: Option<&str>,
    filename: &str,
    metadata: &FileIdentity,
    progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
) -> Result<()> {
    let size = metadata.size();
    let expected = content_digest(metadata.etag());
    if std::fs::symlink_metadata(part).is_ok_and(|info| info.len() > size) {
        let quarantined = quarantine(part)?;
        eprintln!(
            "Partial for {filename} is larger than its declared {size} bytes; moved it to {}",
            quarantined.display()
        );
    }
    let mut resume = Resume::open(part, size, expected.is_some(), progress)?;
    let mut stalled = 0usize;
    let mut retried_after_digest = false;
    let mut failure = String::from("no transfer attempted");
    loop {
        if resume.offset == resume.size {
            if let Some(expected) = expected {
                let actual = resume.take_digest();
                if actual != Some(expected) {
                    // Downloaded bytes are authoritative only if they match the
                    // published digest. Keep them, and retry the source once.
                    let quarantined = quarantine(part)?;
                    let actual = actual
                        .map(|digest| {
                            flare_helios::package::PayloadId::from_bytes(digest).to_string()
                        })
                        .unwrap_or_else(|| "<unavailable>".into());
                    if retried_after_digest {
                        return Err(format!(
                            "{filename}: content digest {actual} still does not match the Hub ETag {}; kept the quarantined bytes in {}",
                            flare_helios::package::PayloadId::from_bytes(expected),
                            quarantined.display()
                        )
                        .into());
                    }
                    retried_after_digest = true;
                    eprintln!(
                        "Downloaded content for {filename} does not match the Hub digest; kept it in {} and retrying the source once",
                        quarantined.display()
                    );
                    resume = Resume::open(part, size, true, progress)?;
                    stalled = 0;
                    failure = format!("content digest {actual} did not match the Hub ETag");
                    continue;
                }
            }
            resume.sync()?;
            publish_blob(part, blob, size)?;
            return Ok(());
        }
        let before = resume.offset;
        let mut retry_after = None;
        match attempt_transfer(agent, url, token, &mut resume, progress) {
            Ok(()) => {}
            Err(Attempt::Fatal(message)) => return Err(format!("{filename}: {message}").into()),
            Err(Attempt::Retry {
                message,
                retry_after: after,
            }) => {
                failure = message;
                retry_after = after;
            }
        }
        if resume.offset > before {
            stalled = 0;
        } else {
            stalled += 1;
        }
        resume.sync()?;
        if stalled >= TRANSFER_STALLED_ATTEMPTS {
            break;
        }
        std::thread::sleep(retry_delay(stalled, retry_after));
    }
    Err(format!(
        "{filename}: download failed after {TRANSFER_STALLED_ATTEMPTS} attempts without progress: {failure}; kept {} of {size} bytes for resume in {}",
        resume.offset,
        part.display()
    )
    .into())
}

fn publish_blob(part: &Path, blob: &Path, size: u64) -> Result<()> {
    match std::fs::symlink_metadata(blob) {
        Ok(info) => {
            if !info.is_file() || info.len() != size {
                return Err(format!("invalid cached Hugging Face blob: {}", blob.display()).into());
            }
            let _ = std::fs::remove_file(part);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::rename(part, blob)?;
            if let Some(parent) = blob.parent() {
                sync_directory(parent)?;
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn link_snapshot(snapshot: &Path, filename: &str, blob: &Path) -> Result<()> {
    let path = snapshot.join(filename);
    if path.is_file() {
        return Ok(());
    }
    let parent = path
        .parent()
        .ok_or("package file has no parent directory")?;
    std::fs::create_dir_all(parent)?;
    match std::fs::hard_link(blob, &path) {
        Ok(()) => sync_directory(parent),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(format!(
            "cannot link cached package file {}: {error}",
            path.display()
        )
        .into()),
    }
}

/// Cached primary inside the Hub snapshot, without any second export tree.
/// Payload companions are not candidates for the package primary.
fn is_canonical_primary(file: &str) -> bool {
    let segments: Vec<&str> = file.split('/').collect();
    segments.len() == 1 || (segments.len() == 2 && segments[1] == "model.safetensors")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        thread,
        time::Duration,
    };

    const COMMIT: &str = "0123456789012345678901234567890123456789";

    /// Build a minimal signed payload package plus its trusted public profile.
    ///
    /// Used by Orbit's offline success fixture so the node verifies a real
    /// signature and a real payload closure without HOME or embedded profiles.
    pub(crate) fn signed_payload_fixture(
        parent: &std::path::Path,
    ) -> (
        std::path::PathBuf,
        std::path::PathBuf,
        flare_helios::runtime_package::Publisher,
        String,
    ) {
        signed_payload_fixture_with(parent, "gb10-tp1", "sm120")
    }

    fn signed_payload_fixture_with(
        parent: &std::path::Path,
        deployment_id: &str,
        hardware_identity: &str,
    ) -> (
        std::path::PathBuf,
        std::path::PathBuf,
        flare_helios::runtime_package::Publisher,
        String,
    ) {
        signed_payload_fixture_identity(parent, deployment_id, hardware_identity, "owner/offline")
    }

    fn signed_payload_fixture_identity(
        parent: &Path,
        deployment_id: &str,
        hardware_identity: &str,
        model: &str,
    ) -> (
        PathBuf,
        PathBuf,
        flare_helios::runtime_package::Publisher,
        String,
    ) {
        use flare_helios::package::{PayloadId, container, export, schema::wire};
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        let package = parent.join("package");
        std::fs::create_dir_all(&package).unwrap();
        let weights = b"offline orbit weights".repeat(4);
        let id = PayloadId::of(&weights);
        let raw = parent.join("weights.bin");
        std::fs::write(&raw, &weights).unwrap();
        let graph = parent.join("graph.bin");
        std::fs::write(&graph, b"graph-bytes").unwrap();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Offline Orbit".into(),
            namespace: "local-development".into(),
            public_key,
            development: true,
        };
        let contract = flare_helios::runtime_package::wire::ServingContract {
            model_id: "orbit-offline".into(),
            identity: "orbit-offline-sm120".into(),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(flare_helios::runtime_package::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![flare_helios::runtime_package::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let rank = wire::RankBody {
            rank: 0,
            deployment_id: deployment_id.into(),
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
                        payload_id: id.as_bytes().to_vec(),
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
        let request = export::PublishRequest {
            package: wire::PackageBody {
                model_identity: model.into(),
                hardware_identity: hardware_identity.into(),
                frontend: Some(flare_helios::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    flare_helios::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "local-development".into(),
                    official_source: "local-development".into(),
                    development: true,
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![export::PublishDeployment {
                payloads: vec![],
                body: wire::DeploymentBody {
                    deployment_id: deployment_id.into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(contract),
                    ..Default::default()
                },
                ranks: vec![export::PublishRank {
                    body: rank,
                    payloads: vec![export::PublishPayload {
                        role: wire::PayloadRole::Weights,
                        relative: "weights/layer-000.safetensors".into(),
                        tensor: Some(container::COMPANION_TENSOR.into()),
                        source: container::Slice::new(
                            std::sync::Arc::new(std::fs::File::open(&raw).unwrap()),
                            0,
                            weights.len() as u64,
                        )
                        .unwrap(),
                        id,
                    }],
                    resources: vec![export::PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph,
                        nbytes: 11,
                    }],
                }],
            }],
            package_id: [23u8; 32],
            public_key,
            accepted: "local-development".into(),
            no_weights: false,
        };
        export::publish(&request, &key, export::Destination::Directory(&package)).unwrap();
        let profile = parent.join("publisher.json");
        std::fs::write(
            &profile,
            serde_json::to_vec(&vec![publisher.clone()]).unwrap(),
        )
        .unwrap();
        (package, profile, publisher, COMMIT.to_owned())
    }

    pub(crate) fn fixture(companions: &[&str]) -> Vec<u8> {
        fn bytes(field: u8, body: &[u8]) -> Vec<u8> {
            let mut out = vec![field * 8 + 2];
            let mut n = body.len();
            while n >= 128 {
                out.push((n as u8 & 127) | 128);
                n >>= 7;
            }
            out.push(n as u8);
            out.extend_from_slice(body);
            out
        }
        // ProtectedIndex -> PublisherMetadata -> ModelIndex, with unsigned discovery metadata.
        let index = [8, if companions.is_empty() { 4 } else { 5 }];
        let metadata = bytes(1, &index);
        let mut envelope = vec![8, 4];
        for (kind, body) in [(1, metadata), (2, vec![]), (3, vec![])] {
            envelope.extend(bytes(4, &[vec![8, kind], bytes(2, &body)].concat()));
        }
        let mut data = Vec::new();
        for name in companions {
            data.extend(bytes(4, &bytes(3, name.as_bytes())));
        }
        let rank = [bytes(2, &[]), bytes(5, &data)].concat();
        envelope.extend(bytes(5, &[vec![8, 4], bytes(2, &rank)].concat()));
        let header = serde_json::json!({
            "__metadata__": {"format": "flare"},
            "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]},
            "flare.metadata": {"dtype": "U8", "shape": [envelope.len()], "data_offsets": [0, envelope.len()]},
        });
        let mut result = 4088u64.to_le_bytes().to_vec();
        result.extend(serde_json::to_vec(&header).unwrap());
        result.resize(4096, b' ');
        result.extend(envelope);
        result
    }

    fn origin_fixture(
        parent: &Path,
    ) -> (
        flare_helios::package::cache::Root,
        flare_helios::package::cache::Entry,
        flare_helios::runtime_package::Publisher,
        HubOptions,
    ) {
        let (package, _, publisher, _) = signed_payload_fixture(parent);
        let options = HubOptions {
            model_cache: Some(parent.join("models")),
            ..Default::default()
        };
        let store = flare_helios::package::cache::Root::open(options.model_cache.as_ref().unwrap())
            .unwrap();
        let entry = install_ready(
            "publisher/prepared",
            &package,
            &options,
            store.path(),
            1,
            std::slice::from_ref(&publisher),
        )
        .unwrap();
        let digest =
            authenticated_source_envelope(&package, &options, 1, std::slice::from_ref(&publisher))
                .unwrap();
        remember_download(
            store.path(),
            "publisher/prepared",
            "gb10",
            COMMIT,
            "model.safetensors",
            &options,
            1,
            &entry,
            digest,
        )
        .unwrap();
        store.publish_entry(&entry).unwrap();
        (store, entry, publisher, options)
    }

    #[test]
    fn model_refresh_checks_recorded_repository_without_downloading_unchanged_payloads() {
        let temp = tempfile::tempdir().unwrap();
        let (store, entry, publisher, options) = origin_fixture(temp.path());
        assert_ne!(
            entry.model(),
            flare_helios::package::cache::model_key("publisher/prepared").unwrap()
        );
        let original = entry.id().to_owned();
        let (endpoint, worker) = mock_hub(HubRoutes::default().document(
            "/api/models/publisher/prepared/revision/gb10",
            metadata_response_with(&["model.safetensors"]),
        ));
        let selected = refresh_model_at(
            &endpoint,
            &store,
            entry,
            false,
            &options,
            1,
            &[],
            None,
            &[publisher],
        )
        .unwrap();
        worker.join().unwrap();
        assert_eq!(selected.id(), original);
        assert!(store.evict(selected.model(), selected.id()).is_err());
    }

    #[test]
    fn model_refresh_pinned_and_offline_never_contact_origin() {
        let temp = tempfile::tempdir().unwrap();
        let (store, entry, publisher, mut options) = origin_fixture(temp.path());
        let model = entry.model().to_owned();
        let version = entry.id().to_owned();
        options.token_file = Some(temp.path().join("missing-token"));
        for (pinned, offline) in [(true, false), (false, true)] {
            options.offline = offline;
            let selected = refresh_model_at(
                "http://127.0.0.1:0",
                &store,
                store.entry(&model, &version).unwrap(),
                pinned,
                &options,
                1,
                &[],
                None,
                std::slice::from_ref(&publisher),
            )
            .unwrap();
            assert_eq!(selected.id(), version);
        }
        options.offline = false;
        options.revision = Some("different".into());
        assert!(
            refresh_model_at(
                "http://127.0.0.1:0",
                &store,
                entry,
                true,
                &options,
                1,
                &[],
                None,
                &[publisher]
            )
            .err()
            .unwrap()
            .to_string()
            .contains("pinned")
        );
    }

    #[test]
    fn model_refresh_installs_update_and_preserves_previous_version_origin() {
        let temp = tempfile::tempdir().unwrap();
        let (store, old, publisher, options) = origin_fixture(&temp.path().join("original"));
        let old_id = old.id().to_owned();
        // The selected entry was acquired before the commit now served by the branch.
        let old_commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let origin = entry_origin(&store, &old).unwrap().unwrap();
        store
            .publish_entry_source(
                &source_selector(
                    "publisher/prepared",
                    "gb10",
                    Some("model.safetensors"),
                    &[1],
                ),
                old_commit,
                &old,
                origin.expected_envelope,
            )
            .unwrap();
        let (package, _, updated_publisher, _) =
            signed_payload_fixture(&temp.path().join("updated"));
        let routes = package_routes(&package, "publisher/prepared", "gb10");
        let (endpoint, worker, _) = mock_hub_counted(routes);
        let selected = refresh_model_at(
            &endpoint,
            &store,
            old,
            false,
            &options,
            1,
            &[],
            None,
            &[publisher, updated_publisher],
        )
        .unwrap();
        worker.join().unwrap();
        assert_ne!(selected.id(), old_id);
        assert_eq!(
            store.resolve(selected.model()).unwrap().unwrap().id(),
            selected.id()
        );
        assert_eq!(
            entry_origin(&store, &selected).unwrap().unwrap().commit,
            COMMIT
        );
        let previous = store.entry(selected.model(), &old_id).unwrap();
        assert_eq!(
            entry_origin(&store, &previous).unwrap().unwrap().commit,
            old_commit
        );
    }

    #[test]
    fn model_refresh_rejects_changed_identity_and_untrusted_update_without_selecting_it() {
        for (model, trusted, expected_error) in [
            ("owner/other", true, "different model"),
            ("owner/offline", false, "publisher"),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let (store, old, publisher, options) = origin_fixture(&temp.path().join("original"));
            let key = old.model().to_owned();
            let id = old.id().to_owned();
            let origin = entry_origin(&store, &old).unwrap().unwrap();
            store
                .publish_entry_source(
                    &source_selector(
                        "publisher/prepared",
                        "gb10",
                        Some("model.safetensors"),
                        &[1],
                    ),
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    &old,
                    origin.expected_envelope,
                )
                .unwrap();
            let (package, _, other, _) = signed_payload_fixture_identity(
                &temp.path().join("updated"),
                "gb10-tp1",
                "sm120",
                model,
            );
            let routes = package_routes(&package, "publisher/prepared", "gb10");
            let (endpoint, worker, _) = mock_hub_counted(routes);
            let mut profiles = vec![publisher];
            if trusted {
                profiles.push(other);
            }
            let error = refresh_model_at(
                &endpoint,
                &store,
                old,
                false,
                &options,
                1,
                &[],
                None,
                &profiles,
            )
            .err()
            .unwrap()
            .to_string();
            worker.join().unwrap();
            assert!(error.contains(expected_error), "{error}");
            assert_eq!(store.resolve(&key).unwrap().unwrap().id(), id);
        }
    }

    #[test]
    fn model_origin_recovers_hub_refs_and_ignores_rank_transfer_refs() {
        let temp = tempfile::tempdir().unwrap();
        let (store, entry, _, _) = origin_fixture(temp.path());
        for record in std::fs::read_dir(store.path().join("origins")).unwrap() {
            std::fs::remove_file(record.unwrap().path()).unwrap();
        }
        store
            .publish_source_ref("distributed-export-rank0", "export", &entry, [1; 32])
            .unwrap();
        let origin = entry_origin(&store, &entry).unwrap().unwrap();
        assert_eq!(origin.repository, "publisher/prepared");
        assert_eq!(origin.revision, "gb10");
        assert_eq!(origin.sizes, vec![1]);
        for record in std::fs::read_dir(store.path().join("sources")).unwrap() {
            std::fs::remove_file(record.unwrap().path()).unwrap();
        }
        assert_eq!(entry_origin(&store, &entry).unwrap(), Some(origin));
    }

    #[test]
    fn model_refresh_failed_update_keeps_default() {
        let temp = tempfile::tempdir().unwrap();
        let (store, entry, publisher, options) = origin_fixture(temp.path());
        let model = entry.model().to_owned();
        let version = entry.id().to_owned();
        let response =
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec();
        let (endpoint, worker) = mock_hub(
            HubRoutes::default().document("/api/models/publisher/prepared/revision/gb10", response),
        );
        let error = refresh_model_at(
            &endpoint,
            &store,
            entry,
            false,
            &options,
            1,
            &[],
            None,
            &[publisher],
        )
        .err()
        .unwrap()
        .to_string();
        worker.join().unwrap();
        assert!(error.contains("--offline"), "{error}");
        assert_eq!(store.resolve(&model).unwrap().unwrap().id(), version);
    }

    #[test]
    fn local_paths_and_repository_revisions_are_distinct() {
        assert!(is_repository(Path::new("owner/model")));
        assert!(is_repository(Path::new("owner/model@release/gb202")));
        for path in [
            "model.safetensors",
            "dir/model.safetensors",
            "./owner/model",
            "/owner/model",
            "../owner/model",
            "C:\\models\\model.safetensors",
        ] {
            assert!(!is_repository(Path::new(path)), "{path}");
        }
        assert_eq!(
            repository("owner/model@gb10").unwrap(),
            ("owner/model", Some("gb10"))
        );
        for source in [
            "owner",
            "owner/model/extra",
            "../model",
            "owner/model@",
            "owner/model@../../other",
        ] {
            assert!(repository(source).is_err(), "{source}");
        }
    }

    #[test]
    fn package_selection_rejects_ambiguous_and_escaping_filenames() {
        assert_eq!(
            choose_file(["README.md".into(), "model.safetensors".into()], None).unwrap(),
            "model.safetensors"
        );
        let error = choose_file(["tp4.safetensors".into(), "tp2.safetensors".into()], None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--filename"));
        assert!(error.contains("tp2.safetensors\n  tp4.safetensors"));
        for file in [
            "../model.safetensors",
            "/model.safetensors",
            "C:\\model.safetensors",
            "weights.json",
        ] {
            assert!(choose_file([], Some(file)).is_err(), "{file}");
        }
    }

    /// One content file the mock Hub serves with real HTTP range semantics.
    #[derive(Clone)]
    struct MockFile {
        commit: String,
        etag: String,
        body: Vec<u8>,
        /// Scripted answers for successive content requests, for fault tests.
        faults: std::collections::VecDeque<MockFault>,
        /// Scripted answers for successive file identity probes.
        metadata_faults: std::collections::VecDeque<MockFault>,
    }

    impl MockFile {
        fn new(etag: &str, body: impl Into<Vec<u8>>) -> Self {
            Self {
                commit: COMMIT.to_owned(),
                etag: etag.to_owned(),
                body: body.into(),
                faults: std::collections::VecDeque::new(),
                metadata_faults: std::collections::VecDeque::new(),
            }
        }

        fn identity_faults(mut self, faults: impl IntoIterator<Item = MockFault>) -> Self {
            self.metadata_faults = faults.into_iter().collect();
            self
        }

        fn with_faults(mut self, faults: impl IntoIterator<Item = MockFault>) -> Self {
            self.faults = faults.into_iter().collect();
            self
        }

        fn at_commit(mut self, commit: &str) -> Self {
            self.commit = commit.to_owned();
            self
        }

        fn script(mut self, faults: impl IntoIterator<Item = MockFault>) -> Self {
            self.faults = faults.into_iter().collect();
            self
        }
    }

    /// A scripted deviation from a conforming content response.
    #[derive(Clone, Copy)]
    enum MockFault {
        /// Declare the requested range but send only these body bytes, then close.
        Short(usize),
        /// Answer `200` with the whole body, ignoring the requested range.
        IgnoreRange,
        /// Answer with this status, no body, and a `Retry-After` in seconds.
        Throttle(u16, u64),
        /// Answer with this status and a `Retry-After` one second from now as an
        /// IMF-fixdate, exercising the date form.
        ThrottleDate(u16),
        /// Answer with this status and no body.
        Status(u16),
        /// Answer `206` from this offset instead of the requested one.
        Shifted(u64),
        /// Declare fewer bytes than the body actually contains.
        Overrun(usize),
        /// Declare the remaining range, send nothing, then wait this long.
        Stall(Duration),
        /// Answer this file identity probe with this status and no body.
        MetadataStatus(u16),
    }

    #[derive(Clone)]
    enum HubRoute {
        /// Canned responses written verbatim; the last one repeats.
        Document(std::collections::VecDeque<Vec<u8>>),
        /// Repository content with `Range`, `Content-Range`, and length handling.
        File(MockFile),
    }

    #[derive(Clone, Default)]
    struct HubRoutes(std::collections::HashMap<String, HubRoute>);

    impl HubRoutes {
        fn document(mut self, path: impl Into<String>, response: Vec<u8>) -> Self {
            self.0.insert(
                path.into(),
                HubRoute::Document(std::collections::VecDeque::from([response])),
            );
            self
        }

        /// Serve these responses in order for successive requests to one path.
        fn scripted_document(mut self, path: impl Into<String>, responses: Vec<Vec<u8>>) -> Self {
            assert!(!responses.is_empty());
            self.0
                .insert(path.into(), HubRoute::Document(responses.into()));
            self
        }

        fn file(mut self, path: impl Into<String>, file: MockFile) -> Self {
            self.0.insert(path.into(), HubRoute::File(file));
            self
        }
    }

    #[derive(Default, Debug)]
    struct TransferLog {
        /// Content responses served per path, including resumed and failed attempts.
        files: std::collections::BTreeMap<String, usize>,
        /// Content requests received, including scripted failures.
        requests: usize,
        /// Document responses served per path, including scripted failures.
        documents: std::collections::BTreeMap<String, usize>,
        /// File identity probes received, including scripted failures.
        metadata_requests: usize,
        /// Credential sent with each path's last request, if any. Test-only.
        auth: std::collections::BTreeMap<String, String>,
        /// Range start of every content request, in order. Test-only.
        ranges: Vec<(String, u64)>,
        /// Body bytes actually written, which never include a resumed prefix twice.
        bytes: usize,
    }

    /// Longest quiet period before the mock stops accepting, in idle ticks. A
    /// test that keeps the server alive across a stalled request uses
    /// `u64::MAX` and never joins the worker.
    fn serve_hub(
        routes: HubRoutes,
        log: Option<std::sync::Arc<std::sync::Mutex<TransferLog>>>,
        idle_limit: u64,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let routes = std::sync::Arc::new(std::sync::Mutex::new(routes));
        let worker = thread::spawn(move || {
            let mut idle = 0;
            let mut served = 0;
            loop {
                let (socket, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        if served > 0 {
                            idle += 1;
                            // Exceed the longest retry backoff this suite uses.
                            if idle > idle_limit {
                                break;
                            }
                        }
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("accept failed: {error}"),
                };
                idle = 0;
                served += 1;
                // One thread per connection: a scripted stall must not block the
                // client that resumes the transfer.
                let routes = std::sync::Arc::clone(&routes);
                let log = log.clone();
                thread::spawn(move || serve_connection(socket, &routes, log.as_ref()));
            }
        });
        (endpoint, worker)
    }

    fn serve_connection(
        mut socket: std::net::TcpStream,
        routes: &std::sync::Mutex<HubRoutes>,
        log: Option<&std::sync::Arc<std::sync::Mutex<TransferLog>>>,
    ) {
        socket.set_nonblocking(false).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut reader = BufReader::new(&mut socket);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let mut parts = line.split_whitespace();
        parts.next().expect("request method");
        let path = parts.next().expect("request path").to_owned();
        let mut range = None;
        let mut auth = None;
        loop {
            line.clear();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" || line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                if name.eq_ignore_ascii_case("range") {
                    range = Some(value.trim().to_owned());
                }
                if name.eq_ignore_ascii_case("authorization") {
                    auth = Some(value.trim().to_owned());
                }
            }
        }
        if let Some(log) = log
            && let Some(auth) = auth
        {
            log.lock().unwrap().auth.insert(path.clone(), auth);
        }
        // Decide the response while holding the route lock, then release it
        // before writing a body that may be scripted to stall.
        enum Planned {
            Document(Vec<u8>),
            File(MockFile, Option<MockFault>),
        }
        let planned = {
            let mut routes = routes.lock().unwrap();
            match routes
                .0
                .get_mut(&path)
                .unwrap_or_else(|| panic!("unexpected request {path}"))
            {
                HubRoute::Document(responses) => {
                    if let Some(log) = log {
                        *log.lock()
                            .unwrap()
                            .documents
                            .entry(path.clone())
                            .or_insert(0) += 1;
                    }
                    let response = if responses.len() > 1 {
                        responses.pop_front().unwrap()
                    } else {
                        responses.front().unwrap().clone()
                    };
                    Planned::Document(response)
                }
                HubRoute::File(file) => {
                    let fault = if range.as_deref() == Some("bytes=0-0") {
                        file.metadata_faults.pop_front()
                    } else {
                        file.faults.pop_front()
                    };
                    Planned::File(file.clone(), fault)
                }
            }
        };
        match planned {
            Planned::Document(response) => socket.write_all(&response).unwrap(),
            Planned::File(file, fault) => {
                serve_file(&mut socket, &path, &file, fault, range.as_deref(), log)
            }
        }
    }

    /// Answer one content request the way the Hub and its CDN do: a metadata
    /// probe is `bytes=0-0`, and a transfer asks for an open-ended suffix.
    fn serve_file(
        socket: &mut std::net::TcpStream,
        path: &str,
        file: &MockFile,
        fault: Option<MockFault>,
        range: Option<&str>,
        log: Option<&std::sync::Arc<std::sync::Mutex<TransferLog>>>,
    ) {
        let size = file.body.len();
        let requested = range.and_then(|value| value.strip_prefix("bytes="));
        let (start, end) = match requested.and_then(|value| value.split_once('-')) {
            Some((start, end)) => (
                start.parse::<usize>().unwrap_or(0),
                end.trim().parse::<usize>().ok(),
            ),
            None => (0, None),
        };
        let metadata = range == Some("bytes=0-0");
        if metadata && let Some(log) = log.as_ref() {
            log.lock().unwrap().metadata_requests += 1;
        }
        if metadata && let Some(MockFault::MetadataStatus(status)) = fault {
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 {status} Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    )
                    .as_bytes(),
                )
                .unwrap();
            return;
        }
        if !metadata && start >= size {
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{size}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    )
                    .as_bytes(),
                )
                .unwrap();
            return;
        }
        let fault = if metadata {
            None
        } else {
            if let Some(log) = log {
                log.lock().unwrap().requests += 1;
                log.lock()
                    .unwrap()
                    .ranges
                    .push((path.to_owned(), start as u64));
            }
            fault
        };
        let (status, body_start, body_end, write_limit) = match fault {
            None => ("206 Partial Content", start, end.unwrap_or(size - 1), None),
            // A conforming header with a body that stops early.
            Some(MockFault::Short(count)) => (
                "206 Partial Content",
                start,
                end.unwrap_or(size - 1),
                Some(count),
            ),
            // A complete, unranged response that cannot be appended to a prefix.
            Some(MockFault::IgnoreRange) => ("200 OK", 0, size - 1, None),
            Some(MockFault::ThrottleDate(status)) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64;
                let date = imf_fixdate(now + 1);
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 {status} Error\r\nRetry-After: {date}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )
                        .as_bytes(),
                    )
                    .unwrap();
                return;
            }
            Some(MockFault::Throttle(status, seconds)) => {
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 {status} Error\r\nRetry-After: {seconds}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )
                        .as_bytes(),
                    )
                    .unwrap();
                return;
            }
            Some(MockFault::Status(status)) => {
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 {status} Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )
                        .as_bytes(),
                    )
                    .unwrap();
                return;
            }
            // A different offset than the one that was requested.
            Some(MockFault::Shifted(offset)) => {
                ("206 Partial Content", offset as usize, size - 1, None)
            }
            // A body that continues past the range it declares.
            Some(MockFault::MetadataStatus(status)) => {
                // Only reachable if a script puts one in the content queue.
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 {status} Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )
                        .as_bytes(),
                    )
                    .unwrap();
                return;
            }
            Some(MockFault::Stall(wait)) => {
                let declared = size - start;
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 206 Partial Content\r\nx-repo-commit: {}\r\netag: {}\r\nContent-Range: bytes {start}-{}/{size}\r\nContent-Length: {declared}\r\nConnection: close\r\n\r\n",
                            file.commit,
                            file.etag,
                            size - 1
                        )
                        .as_bytes(),
                    )
                    .unwrap();
                std::thread::sleep(wait);
                return;
            }
            Some(MockFault::Overrun(extra)) => (
                "206 Partial Content",
                start,
                (start + extra - 1).min(size - 1),
                None,
            ),
        };
        let body_end = body_end.min(size - 1);
        let declared = body_end - body_start + 1;
        let content_range = if metadata {
            format!("Content-Range: bytes 0-0/{size}\r\n")
        } else {
            format!("Content-Range: bytes {body_start}-{body_end}/{size}\r\n")
        };
        let mut response = format!(
            "HTTP/1.1 {status}\r\nx-repo-commit: {}\r\netag: {}\r\n{content_range}Content-Length: {declared}\r\nConnection: close\r\n\r\n",
            file.commit, file.etag
        )
        .into_bytes();
        if metadata {
            response.push(b'p');
        } else {
            let written = write_limit.unwrap_or(declared).min(declared);
            response.extend_from_slice(&file.body[body_start..body_start + written]);
            if matches!(fault, Some(MockFault::Overrun(_))) {
                response
                    .extend_from_slice(&file.body[body_start..(body_start + declared).min(size)]);
            }
            if let Some(log) = log {
                let mut log = log.lock().unwrap();
                *log.files.entry(path.to_owned()).or_insert(0) += 1;
                log.bytes += written;
            }
        }
        socket.write_all(&response).unwrap();
    }

    fn mock_hub(routes: HubRoutes) -> (String, thread::JoinHandle<()>) {
        serve_hub(routes, None, 600)
    }

    /// Serve a Hub and record content responses by path and byte count.
    fn mock_hub_counted(
        routes: HubRoutes,
    ) -> (
        String,
        thread::JoinHandle<()>,
        std::sync::Arc<std::sync::Mutex<TransferLog>>,
    ) {
        let log = std::sync::Arc::new(std::sync::Mutex::new(TransferLog::default()));
        let (endpoint, worker) = serve_hub(routes, Some(std::sync::Arc::clone(&log)), 600);
        (endpoint, worker, log)
    }

    /// A Hub that keeps accepting for the whole test, for a request that stalls
    /// across a subprocess kill. The worker is deliberately never joined.
    fn mock_hub_counted_patient(
        routes: HubRoutes,
    ) -> (
        String,
        thread::JoinHandle<()>,
        std::sync::Arc<std::sync::Mutex<TransferLog>>,
    ) {
        let log = std::sync::Arc::new(std::sync::Mutex::new(TransferLog::default()));
        let (endpoint, worker) = serve_hub(routes, Some(std::sync::Arc::clone(&log)), u64::MAX);
        (endpoint, worker, log)
    }

    fn package_files(root: &Path) -> Vec<String> {
        fn walk(root: &Path, directory: &Path, files: &mut Vec<String>) {
            for entry in std::fs::read_dir(directory).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.is_dir() {
                    walk(root, &path, files);
                } else {
                    files.push(
                        path.strip_prefix(root)
                            .unwrap()
                            .to_string_lossy()
                            .replace('\\', "/"),
                    );
                }
            }
        }
        let mut files = Vec::new();
        walk(root, root, &mut files);
        files.sort();
        files
    }

    fn package_routes(root: &Path, repository: &str, revision: &str) -> HubRoutes {
        use sha2::{Digest, Sha256};
        let files = package_files(root);
        let names: Vec<&str> = files.iter().map(String::as_str).collect();
        let mut routes = HubRoutes::default().document(
            format!("/api/models/{repository}/revision/{revision}"),
            metadata_response_with(&names),
        );
        for file in files {
            let body = std::fs::read(root.join(&file)).unwrap();
            let etag = format!("{:x}", Sha256::digest(&body));
            routes = routes.file(
                format!("/{repository}/resolve/{COMMIT}/{file}"),
                MockFile::new(&etag, body),
            );
        }
        routes
    }

    fn counted_download(
        package: &Path,
        repository: &str,
        revision: &str,
        options: &HubOptions,
        publisher: &flare_helios::runtime_package::Publisher,
    ) -> (TransferLog, String) {
        let routes = package_routes(package, repository, revision);
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let entry = download_with_token_at(
            &endpoint,
            &format!("{repository}@{revision}"),
            options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(publisher),
            None,
        )
        .unwrap();
        let entry_id = entry.id().to_owned();
        drop(entry);
        worker.join().unwrap();
        let log = log.lock().unwrap();
        (
            TransferLog {
                files: log.files.clone(),
                requests: log.requests,
                documents: log.documents.clone(),
                metadata_requests: log.metadata_requests,
                auth: log.auth.clone(),
                ranges: log.ranges.clone(),
                bytes: log.bytes,
            },
            entry_id,
        )
    }

    fn counted_download_entry(
        endpoint: &str,
        repository: &str,
        options: &HubOptions,
        publisher: &flare_helios::runtime_package::Publisher,
        log: &std::sync::Arc<std::sync::Mutex<TransferLog>>,
    ) -> (flare_helios::package::cache::Entry, TransferLog) {
        let entry = download_with_token_at(
            endpoint,
            &format!("{repository}@gb10"),
            options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(publisher),
            None,
        )
        .unwrap();
        let snapshot = {
            let log = log.lock().unwrap();
            TransferLog {
                files: log.files.clone(),
                requests: log.requests,
                documents: log.documents.clone(),
                metadata_requests: log.metadata_requests,
                auth: log.auth.clone(),
                ranges: log.ranges.clone(),
                bytes: log.bytes,
            }
        };
        (entry, snapshot)
    }

    fn rewrite_payload_primary(
        directory: &Path,
        locations: Vec<flare_helios::package::schema::wire::PayloadLocation>,
        derivations: Option<&flare_helios::package::schema::wire::DerivationSet>,
        tensors: &[(&str, &[u8])],
    ) {
        use flare_helios::package::{container, reader, schema::wire};
        use prost::Message;
        let metadata = reader::Metadata::open(directory).unwrap();
        let manifest = metadata.primary().tensor(reader::MANIFEST).unwrap();
        let replaced: std::collections::BTreeSet<Vec<u8>> = locations
            .iter()
            .map(|location| location.payload_id.clone())
            .collect();
        let mut preserved: Vec<wire::PayloadLocation> = metadata
            .locations
            .locations
            .iter()
            .filter(|location| !replaced.contains(&location.payload_id))
            .cloned()
            .collect();
        preserved.extend(locations);
        let locations = wire::LocationSet {
            version: metadata.locations.version,
            locations: preserved,
            available_deployments: metadata.locations.available_deployments.clone(),
        };
        let path = directory.join("model.safetensors");
        let temporary = directory.join("model.rebuilt");
        let mut output = std::fs::File::create(&temporary).unwrap();
        let mut writer = container::Writer::new(&mut output, container::PAYLOAD_FORMAT).unwrap();
        writer
            .add(
                reader::MANIFEST,
                manifest.len(),
                container::Source::Slice(manifest),
            )
            .unwrap();
        let encoded = locations.encode_to_vec();
        writer
            .add(
                reader::LOCATIONS,
                encoded.len() as u64,
                container::Source::Bytes(encoded),
            )
            .unwrap();
        if let Some(derivations) = derivations {
            let encoded = derivations.encode_to_vec();
            writer
                .add(
                    reader::DERIVATIONS,
                    encoded.len() as u64,
                    container::Source::Bytes(encoded),
                )
                .unwrap();
        }
        for name in metadata.primary().tensor_names() {
            if name == reader::MANIFEST
                || name == reader::LOCATIONS
                || name == reader::DERIVATIONS
                || name.starts_with(container::ALIGNMENT_PREFIX)
                || tensors.iter().any(|(replacement, _)| *replacement == name)
            {
                continue;
            }
            let tensor = metadata.primary().tensor(name).unwrap();
            writer
                .add(name, tensor.len(), container::Source::Slice(tensor))
                .unwrap();
        }
        for (name, bytes) in tensors {
            writer
                .add(
                    name,
                    bytes.len() as u64,
                    container::Source::Bytes(bytes.to_vec()),
                )
                .unwrap();
        }
        writer.finish().unwrap();
        output.sync_all().unwrap();
        // Windows cannot replace an open or read-only file. Close both files and
        // remove only this directory's link, so a shared cache object stays read-only.
        drop(output);
        drop(metadata);
        std::fs::remove_file(&path).unwrap();
        std::fs::rename(temporary, path).unwrap();
    }

    fn location_bytes(
        directory: &Path,
        location: &flare_helios::package::schema::wire::PayloadLocation,
    ) -> Vec<u8> {
        use flare_helios::package::container;
        let container = if location.file.is_empty() {
            container::Container::open(&directory.join("model.safetensors")).unwrap()
        } else {
            container::Container::open(&directory.join(&location.file)).unwrap()
        };
        let slice = container.tensor(&location.tensor).unwrap();
        let mut bytes = vec![0; slice.len() as usize];
        slice.read_at(0, &mut bytes).unwrap();
        bytes
    }

    fn blob_count(root: &Path) -> usize {
        std::fs::read_dir(root.join("blobs"))
            .unwrap()
            .flatten()
            .filter(|entry| entry.path().is_file())
            .count()
    }

    #[test]
    fn warm_download_ignores_malformed_optional_hints_without_dependency_transfer() {
        for variant in ["malformed", "overlimit"] {
            let temp = tempfile::tempdir().unwrap();
            let (package, _profile, publisher, _commit) = signed_payload_fixture(temp.path());
            let metadata = flare_helios::package::reader::Metadata::open(&package).unwrap();
            let locations = metadata.locations.locations.clone();
            let hints = if variant == "malformed" {
                flare_helios::package::schema::wire::DerivationSet {
                    version: flare_helios::package::derivation::DERIVATION_VERSION + 1,
                    derivations: Vec::new(),
                }
            } else {
                let count =
                    flare_helios::package::derivation::Limits::default().max_derivations + 1;
                flare_helios::package::schema::wire::DerivationSet {
                    version: flare_helios::package::derivation::DERIVATION_VERSION,
                    derivations: (0..count)
                        .map(|_| flare_helios::package::schema::wire::Derivation {
                            target_payload_id: Vec::new(),
                            operations: Vec::new(),
                        })
                        .collect(),
                }
            };
            rewrite_payload_primary(&package, locations, Some(&hints), &[]);
            let model_root = temp.path().join("models");
            let options = HubOptions {
                revision: Some("gb10".into()),
                tensor_parallel: vec![1],
                model_cache: Some(model_root.clone()),
                ..Default::default()
            };
            let routes = package_routes(&package, "owner/offline", "gb10");
            let (endpoint, worker, log) = mock_hub_counted(routes);
            let (cold_entry, cold) =
                counted_download_entry(&endpoint, "owner/offline", &options, &publisher, &log);
            worker.join().unwrap();
            let cold_id = cold_entry.id().to_owned();
            let blobs = blob_count(&model_root);
            drop(cold_entry);

            let routes = package_routes(&package, "owner/offline", "gb10");
            let (endpoint, worker, log) = mock_hub_counted(routes);
            let (warm_entry, warm) =
                counted_download_entry(&endpoint, "owner/offline", &options, &publisher, &log);
            assert_eq!(warm_entry.id(), cold_id, "{variant}: warm entry changed");
            assert!(
                warm.files
                    .keys()
                    .all(|file| file.ends_with("/model.safetensors")),
                "{variant}: optional hints caused dependency transfer: {warm:?}"
            );
            assert_eq!(
                blob_count(&model_root),
                blobs,
                "{variant}: optional hints caused materialization"
            );
            let key = flare_helios::package::cache::model_key("owner/offline").unwrap();
            assert_eq!(
                flare_helios::package::cache::Root::open(&model_root)
                    .unwrap()
                    .ref_entry(&key)
                    .unwrap()
                    .as_deref(),
                Some(cold_id.as_str()),
                "{variant}: warm success changed the selected entry unexpectedly"
            );
            drop(warm_entry);
            worker.join().unwrap();
            let _ = cold;
        }
    }

    #[test]
    fn offline_resolution_rejects_corrupt_embedded_payload_before_default_update() {
        let (_source_scratch, primary, publishers) = selected_tp_package();
        let source_root = primary.parent().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("tp1".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let routes = package_routes(source_root, "sel/model", "tp1");
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (entry, _) =
            counted_download_entry(&endpoint, "sel/model", &options, &publishers[0], &log);
        let selected = entry.id().to_owned();
        worker.join().unwrap();

        let locations = flare_helios::package::reader::Metadata::open(entry.directory())
            .unwrap()
            .locations
            .locations;
        rewrite_payload_primary(
            entry.directory(),
            locations,
            None,
            &[("frontend.tokenizer", b"corrupt embedded frontend")],
        );
        drop(entry);

        let offline = HubOptions {
            offline: true,
            ..options
        };
        let error = match download_with_token_at(
            "http://unused",
            "sel/model@gb10",
            &offline,
            1,
            None,
            &[],
            None,
            &publishers,
            None,
        ) {
            Ok(_) => panic!("offline resolution accepted corrupt embedded frontend bytes"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("payload")
                || error.to_string().contains("digest")
                || error.to_string().contains("verification"),
            "unexpected offline failure: {error}"
        );
        let key = flare_helios::package::cache::model_key("sel").unwrap();
        assert_eq!(
            flare_helios::package::cache::Root::open(&model_root)
                .unwrap()
                .ref_entry(&key)
                .unwrap()
                .as_deref(),
            Some(selected.as_str()),
            "offline verification failure changed the default"
        );
    }

    #[test]
    fn offline_source_ref_rejects_a_different_valid_envelope_without_mutation() {
        let (_source_scratch, primary, publishers) = selected_tp_package();
        let (_other_scratch, other_primary, other_publishers) = selected_tp_package();
        let source_root = primary.parent().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("tp1".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let routes = package_routes(source_root, "sel/model", "tp1");
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (entry, _) =
            counted_download_entry(&endpoint, "sel/model", &options, &publishers[0], &log);
        let selected = entry.id().to_owned();
        worker.join().unwrap();
        let ref_selector = source_selector("sel/model", "tp1", None, &[]);
        let expected = flare_helios::package::cache::Root::open(&model_root)
            .unwrap()
            .read_source_ref(&ref_selector)
            .unwrap()
            .unwrap()
            .expected_envelope()
            .unwrap();
        let installed_primary = entry.directory().join("model.safetensors");
        let mut permissions = std::fs::metadata(&installed_primary).unwrap().permissions();
        permissions.set_readonly(false);
        std::fs::set_permissions(&installed_primary, permissions).unwrap();
        std::fs::copy(&other_primary, &installed_primary).unwrap();
        drop(entry);

        let mut all_publishers = publishers.clone();
        all_publishers.extend(other_publishers);
        let offline = HubOptions {
            offline: true,
            ..options
        };
        let error = match download_with_token_at(
            "http://unused",
            "sel/model@gb10",
            &offline,
            1,
            None,
            &[],
            None,
            &all_publishers,
            None,
        ) {
            Ok(_) => panic!("offline resolution accepted a different signed envelope"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("does not match the signed envelope recorded for its source"),
            "unexpected offline failure: {error}"
        );
        let key = flare_helios::package::cache::model_key("sel").unwrap();
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(selected.as_str()),
            "envelope mismatch changed the default"
        );
        assert_eq!(
            store
                .read_source_ref(&ref_selector)
                .unwrap()
                .unwrap()
                .expected_envelope(),
            Some(expected),
            "envelope mismatch changed the source reference"
        );
    }

    fn replace_source_ref_with_legacy_record(
        root: &Path,
        repository: &str,
        revision: &str,
        entry: &flare_helios::package::cache::Entry,
    ) -> PathBuf {
        let selector = source_selector(repository, revision, None, &[]);
        let path = root
            .join("sources")
            .join(flare_helios::package::PayloadId::of(selector.as_bytes()).to_string());
        std::fs::write(
            &path,
            serde_json::to_vec(&(selector, revision, entry.model(), entry.id())).unwrap(),
        )
        .unwrap();
        path
    }

    #[test]
    fn legacy_unbound_source_ref_fails_offline_and_refreshes_online_without_dependencies() {
        let (_source_scratch, primary, publishers) = selected_tp_package();
        let source_root = primary.parent().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("tp1".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let routes = package_routes(source_root, "sel/model", "tp1");
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (entry, _) =
            counted_download_entry(&endpoint, "sel/model", &options, &publishers[0], &log);
        let selected = entry.id().to_owned();
        worker.join().unwrap();
        let ref_path =
            replace_source_ref_with_legacy_record(&model_root, "sel/model", "tp1", &entry);
        drop(entry);

        let offline = HubOptions {
            offline: true,
            ..options.clone()
        };
        let error = match download_with_token_at(
            "http://unused",
            "sel/model@gb10",
            &offline,
            1,
            None,
            &[],
            None,
            &publishers,
            None,
        ) {
            Ok(_) => panic!("offline resolution accepted an unbound source reference"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains(
                "refresh source online or load cached entry by explicit path"
            ),
            "unexpected offline failure: {error}"
        );
        let key = flare_helios::package::cache::model_key("sel").unwrap();
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(selected.as_str()),
            "unbound offline failure changed the default"
        );
        assert_eq!(
            store
                .read_source_ref(&source_selector("sel/model", "tp1", None, &[]))
                .unwrap()
                .unwrap()
                .expected_envelope(),
            None
        );
        assert!(ref_path.is_file());

        // Online resolution reuses the existing entry and writes a bound source
        // record without fetching dependency payloads.
        let routes = package_routes(source_root, "sel/model", "tp1");
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (refreshed, warm) =
            counted_download_entry(&endpoint, "sel/model", &options, &publishers[0], &log);
        assert_eq!(refreshed.id(), selected);
        assert!(
            warm.files
                .keys()
                .all(|file| file.ends_with("/model.safetensors")),
            "source refresh fetched dependencies: {warm:?}"
        );
        worker.join().unwrap();
        drop(refreshed);
        let bound = store
            .read_source_ref(&source_selector("sel/model", "tp1", None, &[]))
            .unwrap()
            .unwrap();
        assert!(bound.expected_envelope().is_some());

        let selected_offline = download_with_token_at(
            "http://unused",
            "sel/model@gb10",
            &offline,
            1,
            None,
            &[],
            None,
            &publishers,
            None,
        )
        .expect("bound source reference must resolve offline");
        assert_eq!(selected_offline.id(), selected);
    }

    #[test]
    fn offline_binding_keeps_the_selected_ref_across_a_concurrent_update() {
        let (_first_scratch, first_primary, first_publishers) = selected_tp_package();
        let (_second_scratch, second_primary, second_publishers) = selected_tp_package();
        let first_root = first_primary.parent().unwrap();
        let second_root = second_primary.parent().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let first_options = HubOptions {
            revision: Some("tp1".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let second_options = HubOptions {
            revision: Some("tp1".into()),
            tensor_parallel: vec![2],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let routes = package_routes(first_root, "sel/model", "tp1");
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (first, _) = counted_download_entry(
            &endpoint,
            "sel/model",
            &first_options,
            &first_publishers[0],
            &log,
        );
        let first_id = first.id().to_owned();
        worker.join().unwrap();

        // Capture the exact TP1 selection and its expected identity under its lease
        // before a concurrent acquisition updates the unqualified selector.
        let (leased, expected) =
            offline_entry_with_expected(&model_root, "sel/model", "tp1", &first_options, 1)
                .unwrap();
        assert_eq!(leased.id(), first_id);
        drop(first);

        let routes = package_routes(second_root, "sel/model", "tp1");
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (second, _) = counted_download_entry(
            &endpoint,
            "sel/model",
            &second_options,
            &second_publishers[0],
            &log,
        );
        let second_id = second.id().to_owned();
        assert_ne!(first_id, second_id);
        worker.join().unwrap();
        drop(second);

        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        let unqualified = source_selector("sel/model", "tp1", None, &[]);
        assert_eq!(
            store
                .read_source_ref(&unqualified)
                .unwrap()
                .unwrap()
                .entry(),
            second_id
        );
        // The captured identity belongs to the exact TP1 entry, not the now-current
        // unqualified TP2 record.
        verify_offline_binding(&leased, expected, "sel/model", "tp1", &first_publishers).unwrap();

        let selected = download_with_token_at(
            "http://unused",
            "sel/model@gb10",
            &HubOptions {
                offline: true,
                ..first_options
            },
            1,
            None,
            &[],
            None,
            &first_publishers,
            None,
        )
        .expect("exact TP1 source binding must survive a concurrent unqualified update");
        assert_eq!(selected.id(), first_id);
        let key = flare_helios::package::cache::model_key("sel").unwrap();
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(first_id.as_str())
        );
    }

    #[test]
    fn warm_download_repairs_corrupt_entry_from_signed_source() {
        let (_scratch, primary, publishers) = selected_tp_package();
        let source_root = primary.parent().unwrap();
        let source_metadata = flare_helios::package::reader::Metadata::open(&primary).unwrap();
        let source_envelope = source_metadata.envelope.clone();
        let source_primary = std::fs::read(&primary).unwrap();
        let other = tempfile::tempdir().unwrap();
        let (other_package, _, _, _) = signed_payload_fixture(other.path());
        let other_primary = other_package.join("model.safetensors");
        let key = flare_helios::package::cache::model_key("sel").unwrap();

        for tamper in ["embedded", "tensor", "envelope"] {
            let temp = tempfile::tempdir().unwrap();
            let model_root = temp.path().join("models");
            let options = HubOptions {
                revision: Some("tp1".into()),
                tensor_parallel: vec![1],
                model_cache: Some(model_root.clone()),
                ..Default::default()
            };
            let routes = package_routes(source_root, "sel/model", "tp1");
            let (endpoint, worker, log) = mock_hub_counted(routes);
            let (first, cold) =
                counted_download_entry(&endpoint, "sel/model", &options, &publishers[0], &log);
            worker.join().unwrap();
            let first_id = first.id().to_owned();
            let blobs = blob_count(&model_root);
            let entry_directory = first.directory().to_owned();

            match tamper {
                "embedded" => {
                    let locations = flare_helios::package::reader::Metadata::open(&entry_directory)
                        .unwrap()
                        .locations
                        .locations;
                    rewrite_payload_primary(
                        &entry_directory,
                        locations,
                        None,
                        &[("frontend.tokenizer", b"tampered")],
                    );
                }
                "tensor" => {
                    let source = flare_helios::package::cache::PackageSource::open(
                        &primary,
                        std::slice::from_ref(&publishers[0]),
                        flare_helios::package::protection::Selection::Deployment {
                            deployment: 0,
                            rank: None,
                        },
                    )
                    .unwrap();
                    let required: std::collections::BTreeSet<Vec<u8>> = source
                        .verified()
                        .iter()
                        .flat_map(|verified| {
                            flare_helios::package::reader::required_payloads(verified).unwrap()
                        })
                        .collect();
                    let mut locations =
                        flare_helios::package::reader::Metadata::open(&entry_directory)
                            .unwrap()
                            .locations
                            .locations;
                    let location = locations
                        .iter_mut()
                        .find(|location| {
                            !location.file.is_empty() && required.contains(&location.payload_id)
                        })
                        .expect("external required location");
                    location.tensor = "other".into();
                    rewrite_payload_primary(&entry_directory, locations, None, &[]);
                }
                "envelope" => {
                    std::fs::remove_file(entry_directory.join("model.safetensors")).unwrap();
                    std::fs::copy(&other_primary, entry_directory.join("model.safetensors"))
                        .unwrap();
                }
                _ => unreachable!(),
            }

            let routes = package_routes(source_root, "sel/model", "tp1");
            let (endpoint, worker, log) = mock_hub_counted(routes);
            let (repaired, warm) =
                counted_download_entry(&endpoint, "sel/model", &options, &publishers[0], &log);
            assert_ne!(
                repaired.id(),
                first_id,
                "{tamper}: corrupt entry was returned"
            );
            assert_eq!(
                flare_helios::package::reader::Metadata::open(repaired.directory())
                    .unwrap()
                    .envelope,
                source_envelope,
                "{tamper}: repaired entry changed signed semantics"
            );
            assert_eq!(
                flare_helios::package::cache::Root::open(&model_root)
                    .unwrap()
                    .ref_entry(&key)
                    .unwrap()
                    .as_deref(),
                Some(repaired.id()),
                "{tamper}: repaired entry was not selected"
            );
            let entry_metadata =
                flare_helios::package::reader::Metadata::open(repaired.directory()).unwrap();
            for location in &entry_metadata.locations.locations {
                let Some(source_location) = source_metadata
                    .locations
                    .locations
                    .iter()
                    .find(|source| source.payload_id == location.payload_id)
                else {
                    continue;
                };
                assert_eq!(
                    location_bytes(repaired.directory(), location),
                    location_bytes(source_root, source_location),
                    "{tamper}: payload {} changed",
                    flare_helios::package::PayloadId::from_bytes(
                        location.payload_id.as_slice().try_into().unwrap(),
                    )
                );
            }
            assert!(
                warm.files
                    .keys()
                    .all(|file| file.ends_with("/model.safetensors")),
                "{tamper}: warm repair downloaded dependency payloads: {warm:?}"
            );
            assert!(
                !warm.files.is_empty() || cold.files.is_empty(),
                "{tamper}: unexpected warm transfer accounting"
            );
            assert_eq!(
                blob_count(&model_root),
                blobs,
                "{tamper}: warm repair created new payload objects"
            );
            drop(first);
            drop(repaired);
            worker.join().unwrap();
        }
        assert_eq!(std::fs::read(&primary).unwrap(), source_primary);
    }

    /// IMF-fixdate for a Unix timestamp, for exercising date-form `Retry-After`.
    fn imf_fixdate(unix: i64) -> String {
        httpdate::fmt_http_date(std::time::UNIX_EPOCH + Duration::from_secs(unix as u64))
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(bytes))
    }

    fn transport_fixture() -> (tempfile::TempDir, Transport) {
        let temp = tempfile::tempdir().unwrap();
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        (temp, transport)
    }

    fn transport_api(endpoint: String) -> Api {
        ApiBuilder::from_cache(Cache::new(
            std::env::temp_dir().join("flare-test-unused-hub"),
        ))
        .with_token(None)
        .with_endpoint(endpoint)
        .with_progress(false)
        .build()
        .unwrap()
    }

    fn retry_repo() -> Repo {
        Repo::with_revision("owner/retry".into(), RepoType::Model, COMMIT.into())
    }

    fn seed_part(transport: &Transport, folder: &str, key: &str, bytes: &[u8]) {
        let part = transport.part(folder, key);
        std::fs::create_dir_all(part.parent().unwrap()).unwrap();
        std::fs::write(part, bytes).unwrap();
    }

    fn retry_folder() -> String {
        retry_repo().folder_name()
    }

    fn retry_path() -> String {
        format!("/owner/retry/resolve/{COMMIT}/model.safetensors")
    }

    #[test]
    fn transport_resumes_an_interrupted_body_from_the_received_offset() {
        use flare_helios::package::progress::AcquisitionProgress;
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let key = blob_key(&etag).unwrap();
        let routes = HubRoutes::default().file(
            retry_path(),
            MockFile::new(&etag, body.clone()).with_faults([MockFault::Short(32)]),
        );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let progress = AcquisitionProgress::new();
        let path = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            Some(&progress),
        )
        .unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        let log = log.lock().unwrap();
        assert_eq!(log.files[&retry_path()], 2);
        assert_eq!(log.bytes, body.len());
        assert_eq!(progress.snapshot().network_bytes, body.len() as u64);
        assert!(transport.part(&retry_folder(), &key).exists() == false);
        assert!(transport.blob(&retry_folder(), &key).is_file());
    }

    #[test]
    fn transport_resumes_across_invocations_without_recounting_the_prefix() {
        use flare_helios::package::progress::AcquisitionProgress;
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let key = blob_key(&etag).unwrap();
        let routes = HubRoutes::default().file(
            retry_path(),
            MockFile::new(&etag, body.clone()).with_faults([
                MockFault::Short(16),
                MockFault::Status(503),
                MockFault::Status(503),
                MockFault::Status(503),
                MockFault::Status(503),
                MockFault::Status(503),
            ]),
        );
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let temp = tempfile::tempdir().unwrap();
        let first = Transport::open(temp.path(), None, "http://mock").unwrap();
        let error = fetch_file_with_progress(
            &transport_api(endpoint),
            &first,
            &retry_repo(),
            "model.safetensors",
            None,
            Some(&AcquisitionProgress::new()),
        )
        .unwrap_err()
        .to_string();
        worker.join().unwrap();
        assert!(error.contains("kept 16 of 64 bytes"), "{error}");
        assert_eq!(
            std::fs::read(first.part(&retry_folder(), &key)).unwrap(),
            body[..16]
        );
        drop(first);

        let resumed = HubRoutes::default().file(retry_path(), MockFile::new(&etag, body.clone()));
        let (endpoint, worker, log) = mock_hub_counted(resumed);
        let second = Transport::open(temp.path(), None, "http://mock").unwrap();
        let progress = AcquisitionProgress::new();
        let path = fetch_file_with_progress(
            &transport_api(endpoint),
            &second,
            &retry_repo(),
            "model.safetensors",
            None,
            Some(&progress),
        )
        .unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        let log = log.lock().unwrap();
        assert_eq!(
            log.bytes,
            body.len() - 16,
            "the resumed prefix must not be refetched"
        );
        assert_eq!(progress.snapshot().network_bytes, body.len() as u64 - 16);
    }

    #[test]
    fn transport_never_restarts_a_saved_prefix_when_the_server_ignores_the_range() {
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let key = blob_key(&etag).unwrap();
        let (_temp, transport) = transport_fixture();
        let prefix = [0xffu8; 16];
        seed_part(&transport, &retry_folder(), &key, &prefix);
        let faults: Vec<MockFault> = (0..TRANSFER_STALLED_ATTEMPTS)
            .map(|_| MockFault::IgnoreRange)
            .collect();
        let routes = HubRoutes::default().file(
            retry_path(),
            MockFile::new(&etag, body.clone()).with_faults(faults),
        );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let error = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        worker.join().unwrap();
        assert!(error.contains("kept 16 saved bytes"), "{error}");
        assert_eq!(
            std::fs::read(transport.part(&retry_folder(), &key)).unwrap(),
            prefix,
            "saved bytes are never discarded to satisfy an unranged response"
        );
        assert!(!transport.blob(&retry_folder(), &key).exists());
        assert!(log.lock().unwrap().files[&retry_path()] >= 1);
    }

    #[test]
    fn transport_rejects_a_misaligned_content_range_without_touching_the_partial() {
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let key = blob_key(&etag).unwrap();
        let (_temp, transport) = transport_fixture();
        let prefix = [0xa5u8; 5];
        seed_part(&transport, &retry_folder(), &key, &prefix);
        let faults: Vec<MockFault> = (0..5).map(|_| MockFault::Shifted(3)).collect();
        let routes = HubRoutes::default().file(
            retry_path(),
            MockFile::new(&etag, body.clone()).with_faults(faults),
        );
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let error = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        worker.join().unwrap();
        assert!(
            error.contains("instead of resuming at the requested byte 5"),
            "{error}"
        );
        assert_eq!(
            std::fs::read(transport.part(&retry_folder(), &key)).unwrap(),
            prefix
        );
        assert!(!transport.blob(&retry_folder(), &key).exists());
    }

    #[test]
    fn transport_stops_after_repeated_attempts_without_progress() {
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let key = blob_key(&etag).unwrap();
        let (_temp, transport) = transport_fixture();
        seed_part(&transport, &retry_folder(), &key, &[0xa5u8; 3]);
        let faults: Vec<MockFault> = (0..TRANSFER_STALLED_ATTEMPTS)
            .map(|_| MockFault::Status(503))
            .collect();
        let routes = HubRoutes::default().file(
            retry_path(),
            MockFile::new(&etag, body.clone()).with_faults(faults),
        );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let error = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        worker.join().unwrap();
        assert!(error.contains("without progress"), "{error}");
        assert!(error.contains("HTTP 503"), "{error}");
        assert_eq!(
            log.lock().unwrap().requests,
            TRANSFER_STALLED_ATTEMPTS,
            "one attempt per stalled response"
        );
        assert_eq!(
            std::fs::read(transport.part(&retry_folder(), &key)).unwrap(),
            [0xa5u8; 3]
        );
    }

    #[test]
    fn redirect_targets_resolve_as_rfc3986_references() {
        let base = format!("/owner/retry/resolve/{COMMIT}/model.safetensors");
        let base = format!("https://huggingface.co{base}");
        assert_eq!(
            redirect_target(&base, "https://cdn.example/x").unwrap(),
            "https://cdn.example/x"
        );
        assert_eq!(
            redirect_target(&base, "//cdn.example/x").unwrap(),
            "https://cdn.example/x"
        );
        assert_eq!(
            redirect_target(&base, "/other/model.safetensors").unwrap(),
            "https://huggingface.co/other/model.safetensors"
        );
        assert_eq!(
            redirect_target(&base, "../cdn/model.safetensors").unwrap(),
            format!("https://huggingface.co/owner/retry/resolve/cdn/model.safetensors")
        );
        assert_eq!(
            redirect_target(&base, "./sibling.safetensors").unwrap(),
            format!("https://huggingface.co/owner/retry/resolve/{COMMIT}/sibling.safetensors")
        );
        assert_eq!(
            redirect_target(&base, "?download=1").unwrap(),
            format!("{base}?download=1")
        );
        assert_eq!(
            redirect_target(&base, "#fragment").unwrap(),
            format!("{base}#fragment")
        );
        assert_eq!(
            redirect_target(&base, "cdn%20dir/model%2Bsafe.safetensors").unwrap(),
            format!(
                "https://huggingface.co/owner/retry/resolve/{COMMIT}/cdn%20dir/model%2Bsafe.safetensors"
            ),
            "percent-encoded paths are preserved"
        );
        assert!(redirect_target(&base, "ftp://example.com/x").is_none());
        assert!(redirect_target(&base, "javascript:alert(1)").is_none());
        assert!(
            redirect_target(&base, "https://user:password@cdn.example/x").is_none(),
            "a redirect target never introduces credentials"
        );
    }

    #[test]
    fn redirect_authentication_requires_the_same_origin() {
        assert!(authenticated_redirect(
            "https://huggingface.co/a/b",
            "https://huggingface.co/c"
        ));
        assert!(authenticated_redirect(
            "https://huggingface.co/a",
            "https://huggingface.co:443/c"
        ));
        assert!(!authenticated_redirect(
            "https://huggingface.co/a",
            "https://cdn.huggingface.co/a"
        ));
        assert!(!authenticated_redirect(
            "https://huggingface.co/a",
            "http://huggingface.co/a"
        ));
        assert!(!authenticated_redirect(
            "https://huggingface.co/a",
            "https://huggingface.co:8443/a"
        ));
        assert!(!authenticated_redirect(
            "https://huggingface.co/a",
            "https://user@huggingface.co/a"
        ));
    }

    #[test]
    fn relative_redirect_resolves_against_the_current_directory() {
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let resolved = format!("/owner/retry/resolve/cdn/model.safetensors");
        let redirect = b"HTTP/1.1 302 Found\r\nLocation: ../cdn/model.safetensors\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec();
        let routes = HubRoutes::default()
            .document(retry_path(), redirect)
            .file(resolved.clone(), MockFile::new(&etag, body.clone()));
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let path = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            Some("secret"),
            None,
        )
        .unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert_eq!(
            log.lock().unwrap().auth.get(&resolved).map(String::as_str),
            Some("Bearer secret"),
            "a same-host relative redirect keeps the credential"
        );
    }

    #[test]
    fn unsupported_redirect_scheme_is_rejected() {
        let redirect = b"HTTP/1.1 302 Found\r\nLocation: ftp://example.com/model.safetensors\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec();
        let (endpoint, worker, log) =
            mock_hub_counted(HubRoutes::default().document(retry_path(), redirect));
        let (_temp, transport) = transport_fixture();
        let error = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        worker.join().unwrap();
        assert!(error.contains("unsupported target"), "{error}");
        assert_eq!(log.lock().unwrap().documents[&retry_path()], 1);
        assert!(!transport.snapshot_dir(&retry_folder(), COMMIT).exists());
    }

    #[test]
    fn same_host_redirect_keeps_the_bearer_token() {
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let cdn = format!("/cdn/{COMMIT}/model.safetensors");
        let redirect = format!(
            "HTTP/1.1 302 Found\r\nLocation: {cdn}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
        .into_bytes();
        let routes = HubRoutes::default()
            .document(retry_path(), redirect)
            .file(cdn.clone(), MockFile::new(&etag, body.clone()));
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let path = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            Some("secret"),
            None,
        )
        .unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        let log = log.lock().unwrap();
        assert_eq!(
            log.auth.get(&cdn).map(String::as_str),
            Some("Bearer secret"),
            "a same-host redirect keeps the credential"
        );
    }

    #[test]
    fn cross_host_redirect_drops_the_bearer_token() {
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let cdn = format!("/cdn/{COMMIT}/model.safetensors");
        let (cdn_endpoint, cdn_worker, cdn_log) = mock_hub_counted(
            HubRoutes::default().file(cdn.clone(), MockFile::new(&etag, body.clone())),
        );
        let redirect = format!(
            "HTTP/1.1 302 Found\r\nLocation: {cdn_endpoint}{cdn}\r\nx-linked-etag: sha256:{etag}\r\nx-repo-commit: {COMMIT}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
        .into_bytes();
        let (endpoint, worker, log) =
            mock_hub_counted(HubRoutes::default().document(retry_path(), redirect));
        let (_temp, transport) = transport_fixture();
        let path = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            Some("secret"),
            None,
        )
        .unwrap();
        worker.join().unwrap();
        cdn_worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert_eq!(
            log.lock()
                .unwrap()
                .auth
                .get(&retry_path())
                .map(String::as_str),
            Some("Bearer secret"),
            "the origin receives the credential"
        );
        assert!(
            cdn_log.lock().unwrap().auth.get(&cdn).is_none(),
            "a content host never receives the credential"
        );
    }

    #[test]
    fn repository_identity_wins_over_a_content_host_etag() {
        let body: Vec<u8> = (0..64u8).collect();
        let repository_identity = sha256_etag(&body);
        let cdn = format!("/cdn/{COMMIT}/model.safetensors");
        // The content host states a different ETag for the same bytes, as real
        // CDNs may. It must not become the Flare cache identity.
        let (cdn_endpoint, cdn_worker, _cdn_log) = mock_hub_counted(
            HubRoutes::default().file(cdn.clone(), MockFile::new("cdn-etag", body.clone())),
        );
        let redirect = format!(
            "HTTP/1.1 302 Found\r\nLocation: {cdn_endpoint}{cdn}\r\nx-linked-etag: {repository_identity}\r\nx-repo-commit: {COMMIT}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
        .into_bytes();
        let (endpoint, worker, _log) =
            mock_hub_counted(HubRoutes::default().document(retry_path(), redirect));
        let (_temp, transport) = transport_fixture();
        let api = transport_api(endpoint);
        let remote = api.repo(retry_repo());
        let identity = file_identity(&remote.url("model.safetensors"), None).unwrap();
        assert_eq!(identity.etag(), repository_identity);
        assert_eq!(identity.commit_hash(), COMMIT);
        assert_eq!(identity.size(), body.len() as u64);
        let path = fetch_file_with_progress(
            &api,
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap();
        worker.join().unwrap();
        cdn_worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        let folder = retry_folder();
        assert!(
            transport
                .blob(&folder, &blob_key(&repository_identity).unwrap())
                .is_file(),
            "the repository identity addresses the cached bytes"
        );
        assert!(
            !transport
                .blob(&folder, &blob_key("cdn-etag").unwrap())
                .exists(),
            "a content host ETag never becomes cache identity"
        );
    }

    #[test]
    fn transient_file_identity_failures_are_retried() {
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let routes = HubRoutes::default().file(
            retry_path(),
            MockFile::new(&etag, body.clone()).identity_faults([
                MockFault::MetadataStatus(503),
                MockFault::MetadataStatus(429),
            ]),
        );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let path = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        let log = log.lock().unwrap();
        assert_eq!(log.metadata_requests, 3, "two transient probes are retried");
        assert_eq!(log.files[&retry_path()], 1, "the body is transferred once");
    }

    #[test]
    fn permanent_file_identity_failures_are_not_retried() {
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let routes = HubRoutes::default().file(
            retry_path(),
            MockFile::new(&etag, body).identity_faults([MockFault::MetadataStatus(401)]),
        );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let error = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        worker.join().unwrap();
        assert!(error.contains("401"), "{error}");
        let log = log.lock().unwrap();
        assert_eq!(
            log.metadata_requests, 1,
            "a permanent failure is not retried"
        );
        assert!(log.files.is_empty());
        assert!(!transport.snapshot_dir(&retry_folder(), COMMIT).exists());
    }

    #[test]
    fn malformed_redirect_is_reported_without_panicking() {
        let routes = HubRoutes::default().document(
            retry_path(),
            b"HTTP/1.1 302 Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
        );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let error = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        worker.join().unwrap();
        assert!(error.contains("without a Location header"), "{error}");
        assert_eq!(log.lock().unwrap().documents[&retry_path()], 1);
        assert!(!transport.snapshot_dir(&retry_folder(), COMMIT).exists());
        assert!(
            !transport
                .part(&retry_folder(), "any")
                .parent()
                .unwrap()
                .exists()
                || std::fs::read_dir(transport.part(&retry_folder(), "any").parent().unwrap())
                    .unwrap()
                    .next()
                    .is_none()
        );
    }

    #[test]
    fn transient_resolution_failures_are_retried() {
        let routes = HubRoutes::default()
            .scripted_document(
                "/api/models/owner/model/revision/gb10",
                vec![
                    b"HTTP/1.1 503 Busy\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
                    b"HTTP/1.1 429 Slow\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
                    metadata_response(),
                ],
            )
            .file(
                format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
                MockFile::new(&sha256_hex(&fixture(&[])), fixture(&[])),
            );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let path = fetch_primary(&transport_api(endpoint), &transport, &repo, None)
            .unwrap()
            .0;
        worker.join().unwrap();
        assert_eq!(std::fs::read(path).unwrap(), fixture(&[]));
        assert_eq!(
            log.lock().unwrap().documents["/api/models/owner/model/revision/gb10"],
            3
        );
    }

    #[test]
    fn permanent_resolution_failures_are_not_retried() {
        let routes = HubRoutes::default().scripted_document(
            "/api/models/owner/model/revision/gb10",
            vec![
                b"HTTP/1.1 404 Missing\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
            ],
        );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let error = fetch_primary(&transport_api(endpoint), &transport, &repo, None)
            .unwrap_err()
            .to_string();
        worker.join().unwrap();
        assert!(error.contains("404"), "{error}");
        assert_eq!(
            log.lock().unwrap().documents["/api/models/owner/model/revision/gb10"],
            1
        );
    }

    #[test]
    fn transport_keys_encode_opaque_etags_losslessly() {
        let cases = [
            ("sha256:abc", "sha256_3Aabc"),
            ("a-b", "a-b"),
            ("a_b", "a_5Fb"),
            ("a.part", "a_2Epart"),
            ("W\"weak\"", "W_22weak_22"),
        ];
        for (etag, expected) in cases {
            assert_eq!(blob_key(etag).unwrap(), expected);
        }
        let mut keys = std::collections::BTreeSet::new();
        for etag in [
            "a:b",
            "a-b",
            "a_3Ab",
            "sha256:00",
            "sha256-00",
            "x.part",
            "x.lock",
            ".part.corrupt-1-2",
        ] {
            let key = blob_key(etag).unwrap();
            assert!(
                key.bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"-_".contains(&byte)),
                "{etag} produced an unsafe key {key}"
            );
            assert!(
                !key.contains('.'),
                "{etag} produced a key that looks like a file-class suffix: {key}"
            );
            assert!(keys.insert(key), "{etag} collided with another opaque ETag");
        }
        assert!(blob_key("").is_err());
        assert!(blob_key("a/b").is_err());
        assert!(blob_key("a\\b").is_err());
        assert!(blob_key("a\0b").is_err());
    }

    #[test]
    fn transport_verifies_only_an_explicit_sha256_identity() {
        let digest = sha256_hex(b"content");
        assert!(
            content_digest(&digest).is_none(),
            "a bare 64-hex ETag is opaque"
        );
        assert!(content_digest(&format!("sha256:{digest}")).is_some());
        assert!(content_digest("sha256:not-hex").is_none());
        assert!(content_digest("md5:00").is_none());
    }

    #[test]
    fn transport_does_not_treat_a_bare_hex_etag_as_a_content_digest() {
        let body: Vec<u8> = (0..64u8).collect();
        // Without the `sha256:` prefix this 64-hex value is opaque, so it must
        // not block publication of content it does not describe.
        let etag = sha256_hex(b"opaque identity");
        let routes = HubRoutes::default().file(retry_path(), MockFile::new(&etag, body.clone()));
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let path = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert!(
            transport
                .blob(&retry_folder(), &blob_key(&etag).unwrap())
                .is_file()
        );
    }

    #[test]
    fn transport_retry_backoff_is_bounded_and_honours_retry_after() {
        assert_eq!(retry_delay(1, None), Duration::from_millis(250));
        assert_eq!(retry_delay(2, None), Duration::from_millis(500));
        assert_eq!(retry_delay(4, None), Duration::from_secs(2));
        assert_eq!(
            retry_delay(usize::MAX, None),
            TRANSFER_BACKOFF_LIMIT,
            "backoff is bounded"
        );
        assert_eq!(
            retry_delay(1, Some(Duration::from_secs(7))),
            Duration::from_secs(7),
            "a server pause longer than the backoff is honored"
        );
        assert_eq!(
            retry_delay(4, Some(Duration::from_secs(1))),
            Duration::from_secs(2),
            "backoff wins when the server asks for less"
        );
        assert_eq!(
            retry_delay(1, Some(Duration::from_secs(3600))),
            TRANSFER_RETRY_AFTER_LIMIT,
            "an unreasonable server pause is capped"
        );
    }

    #[test]
    fn transport_retry_after_accepts_seconds_and_http_dates() {
        // 1994-11-06T08:49:37Z, the RFC 7231 example date.
        const DATE: i64 = 784_111_777;
        assert_eq!(imf_fixdate(DATE), "Sun, 06 Nov 1994 08:49:37 GMT");
        assert_eq!(
            retry_after_date("Sun, 06 Nov 1994 08:49:37 GMT", DATE),
            Some(Duration::ZERO),
            "a date that has already passed requests no wait"
        );
        assert_eq!(
            retry_after_date("Sun, 06 Nov 1994 08:49:37 GMT", DATE - 7),
            Some(Duration::from_secs(7))
        );
        assert_eq!(
            retry_after_date(&imf_fixdate(DATE + 90), DATE),
            Some(Duration::from_secs(90))
        );
        // `httpdate` also accepts the obsolete RFC 850 and asctime forms.
        assert_eq!(
            retry_after_date("Sunday, 06-Nov-94 08:49:37 GMT", DATE - 5),
            Some(Duration::from_secs(5))
        );
        // Malformed, out-of-range, and overflowing values are rejected.
        for malformed in [
            "Sun, 06 Nov 1994 08:49:37 PST",
            "Sun, 32 Nov 1994 08:49:37 GMT",
            "Sun, 06 Foo 1994 08:49:37 GMT",
            "Sun, 06 Nov 1994 24:00:00 GMT",
            "Sun, 06 Nov 1994 08:60:00 GMT",
            "Sun, 06 Nov 1994 -1:00:00 GMT",
            "Sun, 06 Nov 99999999999999999999 08:49:37 GMT",
            "Sun, 06 Nov -1 08:49:37 GMT",
            "not a date",
            "",
        ] {
            assert!(
                retry_after_date(malformed, DATE).is_none(),
                "accepted malformed Retry-After {malformed:?}"
            );
        }
        // The latest date the HTTP-date grammar allows still yields a bounded
        // delay instead of overflowing.
        let latest = imf_fixdate(253_402_300_799);
        assert_eq!(
            retry_delay(1, retry_after_date(&latest, DATE)),
            TRANSFER_RETRY_AFTER_LIMIT,
            "{latest}"
        );
        // A date beyond the cap is bounded like a numeric pause.
        assert_eq!(
            retry_delay(1, retry_after_date(&imf_fixdate(DATE + 3_600), DATE)),
            TRANSFER_RETRY_AFTER_LIMIT
        );
    }

    #[test]
    fn transport_honours_an_http_date_retry_after() {
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let routes = HubRoutes::default().file(
            retry_path(),
            MockFile::new(&etag, body.clone()).with_faults([MockFault::ThrottleDate(503)]),
        );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let started = std::time::Instant::now();
        let path = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap();
        let elapsed = started.elapsed();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert!(
            elapsed >= Duration::from_millis(900),
            "the acquisition waited for the date form of Retry-After: {elapsed:?}"
        );
        assert_eq!(log.lock().unwrap().requests, 2);
    }

    #[test]
    fn transport_syncs_a_streaming_partial_periodically() {
        assert!(!part_sync_due(0, PART_SYNC_BYTES - 1, Duration::ZERO));
        assert!(part_sync_due(0, PART_SYNC_BYTES, Duration::ZERO));
        assert!(part_sync_due(1024, 1024, PART_SYNC_INTERVAL));
        assert!(!part_sync_due(
            1024,
            1024,
            PART_SYNC_INTERVAL - Duration::from_millis(1)
        ));
    }

    #[test]
    fn transport_honours_retry_after_from_a_throttled_response() {
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let routes = HubRoutes::default().file(
            retry_path(),
            MockFile::new(&etag, body.clone()).with_faults([MockFault::Throttle(429, 1)]),
        );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let started = std::time::Instant::now();
        let path = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap();
        let elapsed = started.elapsed();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert!(
            elapsed >= Duration::from_millis(900),
            "the acquisition waited for the server's Retry-After: {elapsed:?}"
        );
        assert_eq!(log.lock().unwrap().requests, 2);
    }

    #[test]
    fn transport_rejects_a_body_that_exceeds_its_declared_range() {
        let body: Vec<u8> = (0..64u8).collect();
        let etag = sha256_hex(&body);
        let routes = HubRoutes::default().file(
            retry_path(),
            MockFile::new(&etag, body.clone()).with_faults([MockFault::Overrun(8)]),
        );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let path = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        let log = log.lock().unwrap();
        assert!(
            log.files[&retry_path()] > 1,
            "the overrun response is rejected"
        );
        assert_eq!(log.bytes, body.len(), "no byte is accepted twice");
    }

    #[test]
    fn transport_rejects_content_that_does_not_match_the_hub_digest() {
        let body: Vec<u8> = (0..64u8).collect();
        let etag = format!("sha256:{}", sha256_hex(b"some other content"));
        let routes = HubRoutes::default().file(retry_path(), MockFile::new(&etag, body.clone()));
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (_temp, transport) = transport_fixture();
        let error = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &retry_repo(),
            "model.safetensors",
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        worker.join().unwrap();
        assert!(
            error.contains("still does not match the Hub ETag"),
            "{error}"
        );
        assert_eq!(log.lock().unwrap().files[&retry_path()], 2);
        assert!(
            !transport
                .blob(&retry_folder(), &blob_key(&etag).unwrap())
                .exists()
        );
        let quarantined: Vec<_> = std::fs::read_dir(
            transport
                .part(&retry_folder(), &blob_key(&etag).unwrap())
                .parent()
                .unwrap(),
        )
        .unwrap()
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().contains(".corrupt-"))
        .collect();
        assert_eq!(quarantined.len(), 2, "each suspect download is preserved");
        for entry in quarantined {
            assert_eq!(std::fs::read(entry.path()).unwrap(), body);
        }
    }

    #[test]
    fn transport_adopts_abandoned_staging_content_but_not_a_live_transfer() {
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("cache");
        std::fs::create_dir_all(model_root.join("locks/staging")).unwrap();
        let tree = model_root.join("staging/trees/crashed");
        let blobs = tree.join("models--owner--model/blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let complete = b"complete content".to_vec();
        let partial = b"resumable partial".to_vec();
        let live = b"still downloading".to_vec();
        let complete_key = blob_key(&sha256_hex(&complete)).unwrap();
        let partial_key = blob_key(&sha256_hex(b"partial content")).unwrap();
        let live_key = blob_key(&sha256_hex(b"live content")).unwrap();
        std::fs::write(blobs.join(&sha256_hex(&complete)), &complete).unwrap();
        std::fs::write(
            blobs.join(format!("{}.part", sha256_hex(b"partial content"))),
            &partial,
        )
        .unwrap();
        std::fs::write(
            blobs.join(format!("{}.part", sha256_hex(b"live content"))),
            &live,
        )
        .unwrap();
        let held =
            content_lock(&blobs.join(format!("{}.lock", sha256_hex(b"live content")))).unwrap();
        let transport = Transport::open(&model_root, None, "http://mock").unwrap();
        let recovered = transport.adopt_legacy("models--owner--model");
        assert_eq!(recovered as usize, complete.len() + partial.len());
        assert_eq!(
            std::fs::read(transport.blob("models--owner--model", &complete_key)).unwrap(),
            complete
        );
        assert_eq!(
            std::fs::read(transport.part("models--owner--model", &partial_key)).unwrap(),
            partial
        );
        assert!(!transport.part("models--owner--model", &live_key).exists());
        assert!(
            tree.is_dir(),
            "adoption must leave the legacy tree to the collector"
        );
        drop(held);

        let routes = HubRoutes::default().file(
            format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
            MockFile::new(&sha256_hex(&complete), complete.clone()),
        );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
        let path = fetch_file_with_progress(
            &transport_api(endpoint),
            &transport,
            &repo,
            "model.safetensors",
            None,
            None,
        )
        .unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(path).unwrap(), complete);
        assert!(
            log.lock().unwrap().files.is_empty(),
            "adopted content must be reused without a body transfer"
        );
    }

    #[test]
    fn transport_lease_excludes_reclamation_while_an_acquisition_is_open() {
        let temp = tempfile::tempdir().unwrap();
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        let lease = transport.root().join("acquisition.lock");
        assert!(
            try_lock_exclusive(&lease).unwrap().is_none(),
            "an open acquisition must block reclamation"
        );
        // A concurrent acquisition shares the lifetime lease instead of waiting.
        let second = Transport::open(temp.path(), None, "http://mock").unwrap();
        assert!(second.blob("models--owner--model", "key").ends_with("key"));
        drop(second);
        drop(transport);
        let reclamation = try_lock_exclusive(&lease)
            .unwrap()
            .expect("reclamation proceeds once every acquisition has finished");
        drop(reclamation);
    }

    #[test]
    fn transport_rejects_an_unknown_generation() {
        let temp = tempfile::tempdir().unwrap();
        let root = Transport::path(temp.path(), None);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("version"), "flare-hub-transport 0\n").unwrap();
        let error = Transport::open(temp.path(), None, "http://mock")
            .unwrap_err()
            .to_string();
        assert!(error.contains("flare-hub-transport 0"), "{error}");
    }

    #[test]
    fn multi_tp_entry_names_are_readable() {
        let name = acquisition_candidate(Some("gb202"), &[1, 2]).unwrap();
        assert!(name.starts_with("gb202-tp1+tp2-"), "{name}");
        assert!(!name.chars().all(|c| c.is_ascii_hexdigit()), "{name}");
        let single = acquisition_candidate(Some("gb10"), &[1]).unwrap();
        assert!(single.starts_with("gb10-tp1-"), "{single}");
        let deduplicated = acquisition_candidate(Some("gb202"), &[2, 1, 2]).unwrap();
        assert!(
            deduplicated.starts_with("gb202-tp1+tp2-"),
            "shared naming must deduplicate and sort union labels: {deduplicated}"
        );
        let unknown = acquisition_candidate(None, &[2, 1]).unwrap();
        assert!(
            unknown.starts_with("unknown-device-tp1+tp2-"),
            "signed-device fallback must use the shared unknown-device union: {unknown}"
        );
        assert!(acquisition_candidate(Some("gb202"), &[]).is_err());
        assert!(acquisition_candidate(Some("gb202"), &[0]).is_err());
    }

    fn inverse_tp_package(
        directory: &Path,
        canonical: &[u8],
        derived: &[u8],
    ) -> flare_helios::runtime_package::Publisher {
        use flare_helios::package::{PayloadId, conversion, derivation, export, schema::wire};
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        std::fs::create_dir_all(directory).unwrap();
        let graph = directory.join("graph.bin");
        std::fs::write(&graph, b"graph-bytes").unwrap();
        let canonical_path = directory.join("payload-canonical");
        std::fs::write(&canonical_path, canonical).unwrap();
        let derived_path = directory.join("payload-derived");
        std::fs::write(&derived_path, derived).unwrap();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let publisher = Publisher {
            name: "Inverse".into(),
            namespace: "inverse".into(),
            public_key: key.public_key().as_ref().try_into().unwrap(),
            development: false,
        };
        let payload = |relative: &str, path: &Path, bytes: &[u8]| export::PublishPayload {
            role: wire::PayloadRole::Weights,
            relative: relative.into(),
            source: export::pin_file(path, bytes.len() as u64).unwrap(),
            id: PayloadId::of(bytes),
            tensor: None,
        };
        let rank = |rank: u32,
                    deployment: &str,
                    target: PayloadId,
                    nbytes: u64,
                    payloads: Vec<export::PublishPayload>| export::PublishRank {
            body: wire::RankBody {
                rank,
                deployment_id: deployment.into(),
                context: Some(Default::default()),
                allocations: vec![wire::Allocation {
                    name: "weights".into(),
                    nbytes: 32,
                    alignment: 4096,
                    address: Some(0x1000_0000),
                    memory: flare_helios::runtime_package::wire::Memory::Device as i32,
                    ..Default::default()
                }],
                initializers: vec![wire::Initializer {
                    allocation: "weights".into(),
                    destination_offset: 0,
                    source: Some(wire::PayloadUse {
                        slice: Some(wire::PayloadSlice {
                            payload_id: target.as_bytes().to_vec(),
                            offset: 0,
                            nbytes,
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
            },
            payloads,
            resources: vec![export::PublishResource {
                kind: wire::ExecutableKind::Graph,
                path: graph.clone(),
                nbytes: 11,
            }],
        };
        let canonical_id = PayloadId::of(canonical);
        let derived_id = PayloadId::of(derived);
        let request = export::PublishRequest {
            package: wire::PackageBody {
                package_id: vec![19; 32],
                model_identity: "inverse/tp".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(wire::PublisherMetadata {
                    namespace: publisher.namespace.clone(),
                    official_source: "https://huggingface.co/inverse/tp".into(),
                    ..Default::default()
                }),
                frontend: Some(Default::default()),
                sampling_defaults: Some(Default::default()),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![
                export::PublishDeployment {
                    body: wire::DeploymentBody {
                        deployment_id: "gb10-tp1".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 1,
                        contract: Some(Default::default()),
                        ..Default::default()
                    },
                    ranks: vec![rank(
                        0,
                        "gb10-tp1",
                        canonical_id,
                        canonical.len() as u64,
                        vec![payload(
                            "weights/canonical.safetensors",
                            &canonical_path,
                            canonical,
                        )],
                    )],
                    payloads: vec![],
                },
                export::PublishDeployment {
                    body: wire::DeploymentBody {
                        deployment_id: "gb10-tp2".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 2,
                        contract: Some(Default::default()),
                        ..Default::default()
                    },
                    ranks: vec![
                        rank(
                            0,
                            "gb10-tp2",
                            derived_id,
                            derived.len() as u64,
                            vec![payload(
                                "weights/derived.safetensors",
                                &derived_path,
                                derived,
                            )],
                        ),
                        rank(
                            1,
                            "gb10-tp2",
                            derived_id,
                            derived.len() as u64,
                            vec![payload(
                                "weights/derived-rank1.safetensors",
                                &derived_path,
                                derived,
                            )],
                        ),
                    ],
                    payloads: vec![],
                },
            ],
            package_id: [19; 32],
            public_key: publisher.public_key,
            accepted: publisher.namespace.clone(),
            no_weights: false,
        };
        export::publish(&request, &key, export::Destination::Directory(directory)).unwrap();
        conversion::emit_derivations(
            directory,
            &[derivation::Derivation {
                target: derived_id,
                length: derived.len() as u64,
                operations: vec![
                    derivation::Operation::Copy(derivation::Copy {
                        source: canonical_id,
                        source_offset: 0,
                        destination_offset: 8,
                        contiguous: 8,
                        outer: Vec::new(),
                    }),
                    derivation::Operation::Copy(derivation::Copy {
                        source: canonical_id,
                        source_offset: 8,
                        destination_offset: 0,
                        contiguous: 8,
                        outer: Vec::new(),
                    }),
                ],
            }],
            &derivation::Limits::default(),
        )
        .unwrap();
        publisher
    }

    #[test]
    fn tp2_target_cache_recovers_tp1_over_http_without_canonical_transfer() {
        let temp = tempfile::tempdir().unwrap();
        let package = temp.path().join("package");
        let canonical: Vec<u8> = (0..16).collect();
        let mut derived = canonical[8..16].to_vec();
        derived.extend_from_slice(&canonical[..8]);
        let publisher = inverse_tp_package(&package, &canonical, &derived);
        let model_root = temp.path().join("models");
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        let derived_id = flare_helios::package::PayloadId::of(&derived);
        store
            .insert_payload(derived_id, derived.len() as u64, derived.as_slice())
            .unwrap();
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (log, entry_id) =
            counted_download(&package, "inverse/tp", "gb10", &options, &publisher);
        assert!(entry_id.starts_with("gb10-tp1-"), "{entry_id}");
        assert!(
            !log.files
                .keys()
                .any(|path| path.contains("weights/canonical")),
            "canonical payload was transferred: {log:?}"
        );
        assert!(
            !log.files
                .keys()
                .any(|path| path.contains("weights/derived")),
            "cached TP2 target was transferred again: {log:?}"
        );
    }

    #[test]
    #[ignore = "requires FLARE_LEGACY_FIXTURE with model.safetensors and publisher.json"]
    fn legacy_fixture_cold_and_warm_use_friendly_names_and_default() {
        let package = PathBuf::from(
            std::env::var_os("FLARE_LEGACY_FIXTURE")
                .expect("FLARE_LEGACY_FIXTURE must name a signed legacy package directory"),
        );
        let profile = package.join("publisher.json");
        assert!(
            package.join("model.safetensors").is_file(),
            "legacy fixture lacks model.safetensors: {}",
            package.display()
        );
        assert!(
            profile.is_file(),
            "legacy fixture lacks publisher.json: {}",
            package.display()
        );
        let publishers: Vec<flare_helios::runtime_package::Publisher> =
            serde_json::from_slice(&std::fs::read(profile).unwrap()).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("custom-branch".into()),
            tensor_parallel: vec![2],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (cold, cold_id) = counted_download(
            &package,
            "legacy/cluster",
            "custom-branch",
            &options,
            &publishers[0],
        );
        assert!(
            cold_id.starts_with("unknown-device-tp2-"),
            "legacy missing-device name is not explicit: {cold_id}"
        );
        assert!(!cold_id.contains("sm120"), "{cold_id}");
        assert!(
            cold.files.keys().any(|path| path.contains("model-data-")),
            "cold legacy download did not transfer companions: {cold:?}"
        );

        let (warm, warm_id) = counted_download(
            &package,
            "legacy/cluster",
            "custom-branch",
            &options,
            &publishers[0],
        );
        assert_eq!(
            warm_id, cold_id,
            "warm legacy download did not reuse the named entry"
        );
        assert!(
            !warm.files.keys().any(|path| path.contains("model-data-")),
            "warm legacy download transferred companions: {warm:?}"
        );
        let key = flare_helios::package::cache::model_key("legacy/cluster").unwrap();
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(warm_id.as_str()),
            "warm legacy download did not update the default"
        );
    }

    #[test]
    #[ignore = "requires FLARE_LEGACY_FIXTURE with model.safetensors and publisher.json"]
    fn legacy_historical_hash_warm_validates_signed_tp() {
        let package = PathBuf::from(
            std::env::var_os("FLARE_LEGACY_FIXTURE")
                .expect("FLARE_LEGACY_FIXTURE must name a signed legacy package directory"),
        );
        let primary = package.join("model.safetensors");
        let publishers: Vec<flare_helios::runtime_package::Publisher> =
            serde_json::from_slice(&std::fs::read(package.join("publisher.json")).unwrap())
                .unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        let key = flare_helios::package::cache::model_key("legacy/cluster").unwrap();
        let primary_bytes = std::fs::read(&primary).unwrap();
        let old_id = format!(
            "{}-legacy",
            flare_helios::package::PayloadId::of(&primary_bytes)
        );
        store.import_legacy_staged(&primary, &key, &old_id).unwrap();

        let exact = HubOptions {
            tensor_parallel: vec![2],
            ..Default::default()
        };
        let warm = legacy_warm_entry(
            &primary,
            &exact,
            "legacy/cluster",
            "custom-branch",
            COMMIT,
            "model.safetensors",
            1,
            &publishers,
            &model_root,
        )
        .unwrap();
        assert_eq!(warm.unwrap().id(), old_id);

        for requested in [vec![4], vec![1, 2]] {
            let wrong = HubOptions {
                tensor_parallel: requested,
                ..Default::default()
            };
            let warm = legacy_warm_entry(
                &primary,
                &wrong,
                "legacy/cluster",
                "custom-branch",
                COMMIT,
                "model.safetensors",
                1,
                &publishers,
                &model_root,
            )
            .unwrap();
            assert!(
                warm.is_none(),
                "historical hash warm accepted unsupported TP {:?}",
                wrong.tensor_parallel
            );
        }
    }

    #[test]
    #[ignore = "requires FLARE_LEGACY_FIXTURE with model.safetensors and publisher.json"]
    fn legacy_warm_rejects_mismatched_explicit_tp_without_default_change() {
        let package = PathBuf::from(
            std::env::var_os("FLARE_LEGACY_FIXTURE")
                .expect("FLARE_LEGACY_FIXTURE must name a signed legacy package directory"),
        );
        let profile = package.join("publisher.json");
        let publishers: Vec<flare_helios::runtime_package::Publisher> =
            serde_json::from_slice(&std::fs::read(profile).unwrap()).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("custom-branch".into()),
            tensor_parallel: vec![2],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (_log, entry_id) = counted_download(
            &package,
            "legacy/cluster",
            "custom-branch",
            &options,
            &publishers[0],
        );
        let key = flare_helios::package::cache::model_key("legacy/cluster").unwrap();
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(entry_id.as_str())
        );

        let mut wrong = options.clone();
        wrong.tensor_parallel = vec![1, 2];
        let warm = legacy_warm_entry(
            &package.join("model.safetensors"),
            &wrong,
            "legacy/cluster",
            "custom-branch",
            COMMIT,
            "model.safetensors",
            1,
            &publishers,
            &model_root,
        )
        .unwrap();
        assert!(
            warm.is_none(),
            "legacy warm path accepted a mismatched explicit TP"
        );
        assert!(
            install_ready(
                "legacy/cluster",
                &package.join("model.safetensors"),
                &wrong,
                &model_root,
                1,
                &publishers,
            )
            .is_err(),
            "legacy cold install accepted a mismatched explicit TP"
        );
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(entry_id.as_str()),
            "failed TP mismatch changed the default"
        );
    }

    #[test]
    fn custom_revision_uses_signed_device_identity() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _commit) = signed_payload_fixture(temp.path());
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("custom-branch".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (_log, entry_id) = counted_download(
            &package,
            "owner/offline",
            "custom-branch",
            &options,
            &publisher,
        );
        assert!(
            entry_id.starts_with("gb10-tp1-"),
            "custom revision must use the signed gb10 identity, got {entry_id}"
        );
    }

    #[test]
    fn signed_custom_cluster_id_uses_package_hardware_identity() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _commit) =
            signed_payload_fixture_with(temp.path(), "cluster-west", "gb10");
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("custom-branch".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (_log, entry_id) = counted_download(
            &package,
            "owner/offline",
            "custom-branch",
            &options,
            &publisher,
        );
        assert!(
            entry_id.starts_with("gb10-tp1-"),
            "custom cluster ID must use signed gb10 hardware, got {entry_id}"
        );
    }

    #[test]
    fn old_sm_only_package_gets_unknown_device_label() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _commit) =
            signed_payload_fixture_with(temp.path(), "sm120-tp1", "sm120");
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("gb202".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (_log, entry_id) =
            counted_download(&package, "owner/offline", "gb202", &options, &publisher);
        assert!(
            entry_id.starts_with("unknown-device-tp1-"),
            "old SM-only package must get an explicit unknown-device label, got {entry_id}"
        );
        assert!(!entry_id.contains("sm120"), "{entry_id}");
        assert!(!entry_id.starts_with("gb202"), "{entry_id}");
    }

    #[test]
    fn cold_and_warm_downloads_report_actual_transfers() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _commit) = signed_payload_fixture(temp.path());
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (cold, cold_id) =
            counted_download(&package, "owner/offline", "gb10", &options, &publisher);
        assert!(
            cold.files
                .keys()
                .any(|name| name.contains("weights/layer-000.safetensors")),
            "{cold:?}"
        );
        assert!(cold.bytes > 0, "{cold:?}");
        assert!(cold_id.starts_with("gb10-tp1-"), "{cold_id}");
        assert!(
            !cold_id.chars().all(|c| c.is_ascii_hexdigit()),
            "entry name must not be a package hash: {cold_id}"
        );
        let selector = source_selector("owner/offline", "gb10", Some("model.safetensors"), &[1]);
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert!(
            store.read_source_ref(&selector).unwrap().is_some(),
            "cold download did not record its source selector"
        );

        let (warm, warm_id) =
            counted_download(&package, "owner/offline", "gb10", &options, &publisher);
        assert_eq!(
            warm_id, cold_id,
            "warm download did not reuse the ready entry"
        );
        assert!(
            !warm.files.keys().any(|name| name.contains("/weights/")),
            "warm download transferred companions {warm:?}"
        );
        assert!(
            !warm.files.keys().any(|name| name.contains("deployments/")),
            "warm download transferred executables {warm:?}"
        );
        let key = flare_helios::package::cache::model_key("owner/offline").unwrap();
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(warm_id.as_str()),
            "warm download did not update the default after success"
        );
    }

    /// Serve a scripted sequence that tolerates repeated metadata calls: after
    /// the first two requests for a path, the body response is used.
    fn metadata_response_with(files: &[&str]) -> Vec<u8> {
        let siblings: Vec<String> = files
            .iter()
            .map(|name| format!("{{\"rfilename\":\"{name}\"}}"))
            .collect();
        let body = format!(
            "{{\"sha\":\"{COMMIT}\",\"siblings\":[{}]}}",
            siblings.join(",")
        );
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\netag: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            "0".repeat(64),
            body.len()
        )
        .into_bytes()
    }

    fn metadata_response() -> Vec<u8> {
        revision_response(COMMIT)
    }

    fn revision_response_with(commit: &str, files: &[&str]) -> Vec<u8> {
        let siblings: Vec<String> = files
            .iter()
            .map(|name| format!("{{\"rfilename\":\"{name}\"}}"))
            .collect();
        let body = format!(
            "{{\"sha\":\"{commit}\",\"siblings\":[{}]}}",
            siblings.join(",")
        );
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\netag: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            "0".repeat(64),
            body.len()
        )
        .into_bytes()
    }

    fn revision_response(commit: &str) -> Vec<u8> {
        let body =
            format!(r#"{{"sha":"{commit}","siblings":[{{"rfilename":"model.safetensors"}}]}}"#);
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\netag: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            "0".repeat(64),
            body.len()
        ).into_bytes()
    }

    #[cfg(unix)]
    fn assert_same_file(left: &Path, right: &Path) {
        use std::os::unix::fs::MetadataExt;
        let left = std::fs::metadata(left).unwrap();
        let right = std::fs::metadata(right).unwrap();
        assert_eq!((left.dev(), left.ino()), (right.dev(), right.ino()));
    }

    /// Two deployments with derivations; a selecting TP1 must fetch only TP1's
    /// dependency source, never the unselected TP2 source or embedded frontend.
    fn selected_tp_package() -> (
        tempfile::TempDir,
        std::path::PathBuf,
        Vec<flare_helios::runtime_package::Publisher>,
    ) {
        use flare_helios::package::{
            PayloadId, container, conversion, derivation, export, schema::wire,
        };
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        let scratch = tempfile::tempdir().unwrap();
        let directory = scratch.path().join("package");
        std::fs::create_dir_all(&directory).unwrap();
        let s1 = b"tp1 canonical source".to_vec();
        let t1 = s1[..s1.len() - 3].to_vec();
        let s2 = b"tp2 canonical source".to_vec();
        let t2 = s2[..s2.len() - 3].to_vec();
        let tokenizer = b"tokenizer bytes".to_vec();
        let (s1_id, t1_id) = (PayloadId::of(&s1), PayloadId::of(&t1));
        let (s2_id, t2_id) = (PayloadId::of(&s2), PayloadId::of(&t2));
        let tokenizer_id = PayloadId::of(&tokenizer);
        let write = |name: &str, bytes: &[u8]| {
            let path = scratch.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        };
        let s1_path = write("s1.bin", &s1);
        let t1_path = write("t1.bin", &t1);
        let s2_path = write("s2.bin", &s2);
        let t2_path = write("t2.bin", &t2);
        let tokenizer_path = write("tokenizer.bin", &tokenizer);
        let graph = write("graph.bin", b"graph-bytes");
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Sel".into(),
            namespace: "sel".into(),
            public_key,
            development: false,
        };
        let payload = |id: PayloadId, path: &std::path::Path, nbytes: usize, relative: &str| {
            export::PublishPayload {
                role: wire::PayloadRole::Weights,
                relative: relative.into(),
                tensor: Some(container::COMPANION_TENSOR.into()),
                source: container::Slice::new(
                    std::sync::Arc::new(std::fs::File::open(path).unwrap()),
                    0,
                    nbytes as u64,
                )
                .unwrap(),
                id,
            }
        };
        let contract = |id: &str| flare_helios::runtime_package::wire::ServingContract {
            model_id: "sel".into(),
            identity: format!("sel-{id}"),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(flare_helios::runtime_package::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![flare_helios::runtime_package::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let rank = |deployment: &str, rank: u32, target: PayloadId, nbytes: usize| wire::RankBody {
            rank,
            deployment_id: deployment.into(),
            context: Some(Default::default()),
            allocations: vec![wire::Allocation {
                name: "device-0".into(),
                nbytes: nbytes as u64,
                alignment: 4096,
                ..Default::default()
            }],
            initializers: vec![wire::Initializer {
                allocation: "device-0".into(),
                destination_offset: 0,
                source: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: target.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: nbytes as u64,
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
        let request = export::PublishRequest {
            package: wire::PackageBody {
                model_identity: "sel".into(),
                hardware_identity: "sm120".into(),
                frontend: Some(flare_helios::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    flare_helios::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "sel".into(),
                    official_source: "https://huggingface.co/sel/model".into(),
                    ..Default::default()
                }),
                assets: vec![wire::FrontendAsset {
                    name: "tokenizer".into(),
                    payload: Some(wire::PayloadUse {
                        slice: Some(wire::PayloadSlice {
                            payload_id: tokenizer_id.as_bytes().to_vec(),
                            offset: 0,
                            nbytes: tokenizer.len() as u64,
                        }),
                        role: wire::PayloadRole::Frontend as i32,
                    }),
                    sha256: flare_helios::package::protection::sha256(&tokenizer).to_vec(),
                }],
                ..Default::default()
            },
            payloads: vec![export::PublishPayload {
                role: wire::PayloadRole::Frontend,
                relative: String::new(),
                tensor: Some("frontend.tokenizer".into()),
                source: container::Slice::new(
                    std::sync::Arc::new(std::fs::File::open(&tokenizer_path).unwrap()),
                    0,
                    tokenizer.len() as u64,
                )
                .unwrap(),
                id: tokenizer_id,
            }],
            deployments: vec![
                export::PublishDeployment {
                    payloads: vec![],
                    body: wire::DeploymentBody {
                        deployment_id: "sm120-tp1".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 1,
                        contract: Some(contract("tp1")),
                        ..Default::default()
                    },
                    ranks: vec![export::PublishRank {
                        body: rank("sm120-tp1", 0, t1_id, t1.len()),
                        payloads: vec![
                            payload(s1_id, &s1_path, s1.len(), "weights/tp1-source.safetensors"),
                            payload(t1_id, &t1_path, t1.len(), "weights/tp1-target.safetensors"),
                        ],
                        resources: vec![export::PublishResource {
                            kind: wire::ExecutableKind::Graph,
                            path: graph.clone(),
                            nbytes: 11,
                        }],
                    }],
                },
                export::PublishDeployment {
                    payloads: vec![],
                    body: wire::DeploymentBody {
                        deployment_id: "sm120-tp2".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 2,
                        contract: Some(contract("tp2")),
                        ..Default::default()
                    },
                    ranks: vec![
                        export::PublishRank {
                            body: rank("sm120-tp2", 0, t2_id, t2.len()),
                            payloads: vec![
                                payload(
                                    s2_id,
                                    &s2_path,
                                    s2.len(),
                                    "weights/tp2-source.safetensors",
                                ),
                                payload(
                                    t2_id,
                                    &t2_path,
                                    t2.len(),
                                    "weights/tp2-target.safetensors",
                                ),
                            ],
                            resources: vec![export::PublishResource {
                                kind: wire::ExecutableKind::Graph,
                                path: graph.clone(),
                                nbytes: 11,
                            }],
                        },
                        export::PublishRank {
                            body: wire::RankBody {
                                rank: 1,
                                deployment_id: "sm120-tp2".into(),
                                context: Some(Default::default()),
                                allocations: vec![wire::Allocation {
                                    name: "device-1".into(),
                                    nbytes: t2.len() as u64,
                                    alignment: 4096,
                                    ..Default::default()
                                }],
                                graphs: vec![wire::Graph {
                                    name: "decode".into(),
                                    resource: 0,
                                    node_count: 1,
                                    ..Default::default()
                                }],
                                ..Default::default()
                            },
                            payloads: vec![],
                            resources: vec![export::PublishResource {
                                kind: wire::ExecutableKind::Graph,
                                path: graph,
                                nbytes: 11,
                            }],
                        },
                    ],
                },
            ],
            package_id: [43u8; 32],
            public_key,
            accepted: "sel".into(),
            no_weights: false,
        };
        export::publish(&request, &key, export::Destination::Directory(&directory)).unwrap();
        let recipe = |target: PayloadId, source: PayloadId, length: usize| derivation::Derivation {
            target,
            length: length as u64,
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source,
                source_offset: 0,
                destination_offset: 0,
                contiguous: length as u64,
                outer: Vec::new(),
            })],
        };
        conversion::emit_derivations(
            &directory,
            &[
                recipe(t1_id, s1_id, t1.len()),
                recipe(t2_id, s2_id, t2.len()),
            ],
            &derivation::Limits::default(),
        )
        .unwrap();
        (
            scratch,
            directory.join("model.safetensors"),
            vec![publisher],
        )
    }

    #[test]
    fn selected_tp_fetches_only_its_own_dependency_companions() {
        let (_scratch, primary, publishers) = selected_tp_package();
        let options = HubOptions {
            tensor_parallel: vec![1],
            ..Default::default()
        };
        let files = dependency_files(&primary, &options, 1, &publishers).unwrap();
        assert_eq!(
            files,
            vec![
                "deployments/sm120-tp1/rank-0/graphs.safetensors".to_string(),
                "weights/tp1-source.safetensors".to_string(),
            ],
            "selected TP fetches its own executables and dependency source only"
        );
        assert!(!files.iter().any(|file| file.contains("tp2")), "{files:?}");
        assert!(
            !files.iter().any(|file| file.contains("tokenizer")),
            "{files:?}"
        );
    }

    #[test]
    fn payload_download_fetches_only_the_selected_closure_over_http() {
        use flare_helios::package::{PayloadId, container, export, reader, schema::wire};
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        std::fs::create_dir_all(&source).unwrap();
        let weights = b"selected closure weights".repeat(4);
        let id = PayloadId::of(&weights);
        let raw = temp.path().join("weights.bin");
        std::fs::write(&raw, &weights).unwrap();
        let graph = temp.path().join("graph.bin");
        std::fs::write(&graph, b"graph-bytes").unwrap();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Test".into(),
            namespace: "test".into(),
            public_key,
            development: false,
        };
        let runtime = flare_helios::runtime_package::wire::ServingContract {
            model_id: "closure".into(),
            identity: "closure-sm120".into(),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(flare_helios::runtime_package::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![flare_helios::runtime_package::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let rank = wire::RankBody {
            rank: 0,
            deployment_id: "gb10-tp1".into(),
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
                        payload_id: id.as_bytes().to_vec(),
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
        let request = export::PublishRequest {
            package: wire::PackageBody {
                model_identity: "owner/closure".into(),
                hardware_identity: "sm120".into(),
                frontend: Some(flare_helios::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    flare_helios::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "test".into(),
                    official_source: "https://huggingface.co/test/closure".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![export::PublishDeployment {
                payloads: vec![],
                body: wire::DeploymentBody {
                    deployment_id: "gb10-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(runtime),
                    ..Default::default()
                },
                ranks: vec![export::PublishRank {
                    body: rank,
                    payloads: vec![export::PublishPayload {
                        role: wire::PayloadRole::Weights,
                        relative: "weights/layer-000.safetensors".into(),
                        tensor: Some(container::COMPANION_TENSOR.into()),
                        source: flare_helios::package::container::Slice::new(
                            std::sync::Arc::new(std::fs::File::open(&raw).unwrap()),
                            0,
                            (weights.len() as u64) as u64,
                        )
                        .unwrap(),
                        id,
                    }],
                    resources: vec![export::PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph,
                        nbytes: (11) as u64,
                    }],
                }],
            }],
            package_id: [17u8; 32],
            public_key,
            accepted: "test".into(),
            no_weights: false,
        };
        export::publish(&request, &key, export::Destination::Directory(&source)).unwrap();
        let metadata = reader::Metadata::open(&source).unwrap();
        let files: Vec<String> = std::iter::once("model.safetensors".to_string())
            .chain(
                metadata
                    .locations
                    .locations
                    .iter()
                    .filter(|location| !location.file.is_empty())
                    .map(|location| location.file.clone()),
            )
            .collect();

        // The snapshot and blob cache are regular files on the same volume. Hard
        // links exercise the same content/closure contract without requiring
        // Windows symlink privilege.
        let snapshot = temp.path().join("snapshot");
        std::fs::create_dir_all(snapshot.join("blobs")).unwrap();
        for name in &files {
            let blob = snapshot.join("blobs").join(name.replace('/', "_"));
            std::fs::copy(source.join(name), &blob).unwrap();
            let link = snapshot.join(name);
            std::fs::create_dir_all(link.parent().unwrap()).unwrap();
            std::fs::hard_link(&blob, &link).unwrap();
        }
        let decoy = snapshot.join("weights/unselected.safetensors");
        std::fs::write(&decoy, b"decoy").unwrap();

        let repo = Repo::with_revision("owner/closure".into(), RepoType::Model, COMMIT.into());
        let resolve = |name: &str| format!("/owner/closure/resolve/{COMMIT}/{}", name);
        use sha2::{Digest, Sha256};
        let mut routes = HubRoutes::default().document(
            format!("/api/models/owner/closure/revision/{COMMIT}"),
            metadata_response_with(&[
                "model.safetensors",
                files[1].as_str(),
                "weights/unselected.safetensors",
            ]),
        );
        for name in &files {
            let body = std::fs::read(source.join(name)).unwrap();
            let etag = format!("{:x}", Sha256::digest(&body));
            routes = routes.file(resolve(name), MockFile::new(&etag, body));
        }
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let api = transport_api(endpoint);
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            ..Default::default()
        };
        let (primary, _pinned) = fetch_primary(&api, &transport, &repo, None).unwrap();
        for name in dependency_files(&primary, &options, 1, &[publisher.clone()]).unwrap() {
            fetch_dependency(
                &api,
                &transport,
                &Repo::with_revision(repo.url(), RepoType::Model, COMMIT.into()),
                &primary,
                &name,
            )
            .unwrap();
        }
        worker.join().unwrap();
        // Only the selected closure was requested: the decoy is not in the snapshot.
        assert!(
            !transport
                .snapshot_dir(&repo.folder_name(), COMMIT)
                .join("weights/unselected.safetensors")
                .exists()
        );
        let requested: Vec<String> = log.lock().unwrap().files.keys().cloned().collect();
        assert!(
            requested.iter().all(|path| !path.contains("unselected")),
            "unselected content was transferred: {requested:?}"
        );
        let entry = install_ready(
            "owner/closure",
            &primary,
            &options,
            &model_root,
            1,
            &[publisher.clone()],
        )
        .unwrap();
        assert!(
            entry
                .directory()
                .join("weights/layer-000.safetensors")
                .is_file()
        );
        // Repository names need not match the model identity in signed metadata.
        let expected_envelope =
            authenticated_source_envelope(&primary, &options, 1, std::slice::from_ref(&publisher))
                .unwrap();
        remember_download(
            &model_root,
            "mirror/closure",
            "gb202",
            COMMIT,
            "model.safetensors",
            &options,
            1,
            &entry,
            expected_envelope,
        )
        .unwrap();
        // Windows cannot delete the acquisition lease while this transport holds it.
        let transport_root = transport.root().to_owned();
        drop(transport);
        std::fs::remove_dir_all(&transport_root).unwrap();
        let offline = HubOptions {
            offline: true,
            revision: Some("gb202".into()),
            model_cache: Some(model_root.clone()),
            token_file: Some(temp.path().join("nonexistent-token")),
            ..options.clone()
        };
        for revision in ["gb202", COMMIT] {
            for filename in [None, Some("model.safetensors")] {
                let requested = HubOptions {
                    filename: filename.map(str::to_owned),
                    ..offline.clone()
                };
                let hit =
                    offline_entry(&model_root, "mirror/closure", revision, &requested, 1).unwrap();
                assert_eq!(hit.directory(), entry.directory());
                verify_offline_entry(&hit, &requested, 1, &[publisher.clone()]).unwrap();
                assert!(verify_offline_entry(&hit, &requested, 1, &[]).is_err());
            }
        }
        assert!(!transport_root.join("flare-packages").exists());
        let wrong_file = HubOptions {
            filename: Some("other.safetensors".into()),
            ..offline.clone()
        };
        assert!(offline_entry(&model_root, "mirror/closure", "gb202", &wrong_file, 1).is_err());
        assert!(offline_entry(&model_root, "mirror/closure", "gb10", &offline, 1).is_err());
        let wrong_tp = HubOptions {
            tensor_parallel: vec![2],
            ..offline.clone()
        };
        let hit = offline_entry(&model_root, "mirror/closure", "gb202", &wrong_tp, 1).unwrap();
        assert!(verify_offline_entry(&hit, &wrong_tp, 1, &[publisher.clone()]).is_err());
        drop(hit);

        let payload = entry.directory().join("weights/layer-000.safetensors");
        let mut permissions = std::fs::metadata(&payload).unwrap().permissions();
        permissions.set_readonly(false);
        std::fs::set_permissions(&payload, permissions).unwrap();
        let mut corrupted = std::fs::read(&payload).unwrap();
        *corrupted.last_mut().unwrap() ^= 1;
        std::fs::write(&payload, corrupted).unwrap();
        assert!(verify_offline_entry(&entry, &offline, 1, &[publisher]).is_err());

        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert!(store.evict(entry.model(), entry.id()).is_err());
        let (model, id) = (entry.model().to_owned(), entry.id().to_owned());
        drop(entry);
        store.evict(&model, &id).unwrap();
        assert!(offline_entry(&model_root, "mirror/closure", "gb202", &offline, 1).is_err());
    }

    #[test]
    fn transport_root_is_persistent_and_scoped_by_endpoint_and_repository() {
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("cache");
        let first = Transport::open(&model_root, None, "https://huggingface.co").unwrap();
        let root = first.root().to_owned();
        assert!(root.ends_with("transport/hub/v1"));
        assert_eq!(
            std::fs::read_to_string(root.join("version")).unwrap(),
            format!("{TRANSPORT_VERSION}\n")
        );
        let folder = "models--owner--model";
        let key = blob_key(&"a".repeat(64)).unwrap();
        std::fs::create_dir_all(first.part(folder, &key).parent().unwrap()).unwrap();
        std::fs::write(first.part(folder, &key), b"saved prefix").unwrap();
        drop(first);
        // A later invocation reuses the same durable root and its saved bytes.
        let second = Transport::open(&model_root, None, "https://huggingface.co").unwrap();
        assert_eq!(second.root(), root);
        assert_eq!(
            std::fs::read(second.part(folder, &key)).unwrap(),
            b"saved prefix"
        );
        assert_ne!(
            second.scope(folder),
            second.scope("models--owner--other"),
            "each repository owns a content namespace"
        );
        let mirror = Transport::open(&model_root, None, "https://mirror.example").unwrap();
        assert_ne!(
            second.scope(folder),
            mirror.scope(folder),
            "each endpoint owns a content namespace"
        );
        assert!(second.root().join("blobs").is_dir());
        let explicit = temp.path().join("explicit");
        let elsewhere =
            Transport::open(&model_root, Some(&explicit), "https://huggingface.co").unwrap();
        assert!(elsewhere.root().starts_with(&explicit));
    }

    #[test]
    fn requested_rank_count_mismatch_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _commit) = signed_payload_fixture(temp.path());
        let verified = flare_helios::runtime_package::VerifiedPackage::open(
            &package,
            std::slice::from_ref(&publisher),
            None,
        )
        .unwrap();
        assert!(enforce_legacy_tensor_parallel(&verified, &[1]).is_ok());
        let error = enforce_legacy_tensor_parallel(&verified, &[2])
            .unwrap_err()
            .to_string();
        assert!(error.contains("requested [2]"), "{error}");
        let error = enforce_legacy_tensor_parallel(&verified, &[1, 2])
            .unwrap_err()
            .to_string();
        assert!(error.contains("requested [1, 2]"), "{error}");
    }

    /// Routes whose file ETags are produced by `etag` from each file's bytes.
    fn routes_with(
        root: &Path,
        repository: &str,
        revision: &str,
        etag: impl Fn(&[u8]) -> String,
    ) -> HubRoutes {
        let files = package_files(root);
        let names: Vec<&str> = files.iter().map(String::as_str).collect();
        let mut routes = HubRoutes::default().document(
            format!("/api/models/{repository}/revision/{revision}"),
            metadata_response_with(&names),
        );
        for file in files {
            let body = std::fs::read(root.join(&file)).unwrap();
            routes = routes.file(
                format!("/{repository}/resolve/{COMMIT}/{file}"),
                MockFile::new(&etag(&body), body),
            );
        }
        routes
    }

    /// The LFS identity the Hub publishes for a large file, prefixed.
    fn sha256_etag(body: &[u8]) -> String {
        format!("sha256:{:x}", sha2::Sha256::digest(body))
    }

    /// The LFS identity the Hub publishes for a large file, unprefixed.
    fn lfs_etag(body: &[u8]) -> String {
        format!("{:x}", sha2::Sha256::digest(body))
    }

    /// The identity the Hub publishes for a file tracked by Git.
    fn git_blob_etag(body: &[u8]) -> String {
        let mut context = ring::digest::Context::new(&ring::digest::SHA1_FOR_LEGACY_USE_ONLY);
        context.update(format!("blob {}\0", body.len()).as_bytes());
        context.update(body);
        context
            .finish()
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    /// Lay out the repository cache the Hugging Face library builds for one commit.
    fn write_hub_cache(root: &Path, folder: &str, commit: &str, package: &Path) -> PathBuf {
        let blobs = root.join(folder).join("blobs");
        let snapshot = root.join(folder).join("snapshots").join(commit);
        std::fs::create_dir_all(&blobs).unwrap();
        for file in package_files(package) {
            let body = std::fs::read(package.join(&file)).unwrap();
            let digest = format!("{:x}", sha2::Sha256::digest(&body));
            std::fs::write(blobs.join(&digest), &body).unwrap();
            let entry = snapshot.join(&file);
            std::fs::create_dir_all(entry.parent().unwrap()).unwrap();
            std::fs::write(&entry, &body).unwrap();
        }
        snapshot
    }

    /// Every entry under a tree, with its bytes and permission bits.
    fn tree_contents(root: &Path) -> Vec<(String, Vec<u8>, u32)> {
        fn walk(root: &Path, directory: &Path, out: &mut Vec<(String, Vec<u8>, u32)>) {
            for entry in std::fs::read_dir(directory).unwrap() {
                let path = entry.unwrap().path();
                let relative = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                let metadata = std::fs::symlink_metadata(&path).unwrap();
                #[cfg(unix)]
                let mode = {
                    use std::os::unix::fs::PermissionsExt;
                    metadata.permissions().mode()
                };
                #[cfg(not(unix))]
                let mode = 0;
                if metadata.is_dir() {
                    out.push((relative, Vec::new(), mode));
                    walk(root, &path, out);
                } else {
                    out.push((relative, std::fs::read(&path).unwrap(), mode));
                }
            }
        }
        let mut entries = Vec::new();
        walk(root, root, &mut entries);
        entries.sort();
        entries
    }

    /// Fetch one repository through a transport that may borrow from `hub_cache`.
    fn fetch_with_hub_cache(
        package: &Path,
        repository: &str,
        hub_cache: PathBuf,
        routes: HubRoutes,
    ) -> (PathBuf, Transport, Option<usize>) {
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let api = transport_api(endpoint);
        let transport = Transport::open(package.parent().unwrap(), None, "http://mock")
            .unwrap()
            .with_hub_cache(Some(hub_cache));
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let (primary, _pinned) = fetch_primary(&api, &transport, &repo, None).unwrap();
        worker.join().unwrap();
        let bytes = log.lock().unwrap().bytes;
        (primary, transport, Some(bytes))
    }

    #[test]
    fn complete_hugging_face_cache_files_are_reused_without_a_transfer() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let repository = "owner/offline";
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let hub_cache = temp.path().join("hf");
        write_hub_cache(&hub_cache, &repo.folder_name(), COMMIT, &package);
        let cached = tree_contents(&hub_cache);
        let routes = routes_with(&package, repository, COMMIT, sha256_etag);
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let api = transport_api(endpoint);
        let transport = Transport::open(temp.path(), None, "http://mock")
            .unwrap()
            .with_hub_cache(Some(hub_cache.clone()));
        let (primary, _pinned) = fetch_primary(&api, &transport, &repo, None).unwrap();
        for name in
            dependency_files(&primary, &options, 1, std::slice::from_ref(&publisher)).unwrap()
        {
            fetch_dependency(&api, &transport, &repo, &primary, &name).unwrap();
        }
        worker.join().unwrap();
        {
            let log = log.lock().unwrap();
            assert_eq!(log.bytes, 0, "reused files transferred {:?}", log.files);
            assert!(log.files.is_empty(), "content requests: {:?}", log.files);
        }
        let entry = install_ready(
            repository,
            &primary,
            &options,
            &model_root,
            1,
            std::slice::from_ref(&publisher),
        )
        .unwrap();
        assert!(entry.directory().join("model.safetensors").is_file());
        assert_eq!(
            tree_contents(&hub_cache),
            cached,
            "the Hugging Face cache must stay unchanged"
        );
    }

    #[test]
    fn unprefixed_lfs_identity_reuses_a_hugging_face_cache_file() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, _publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let repository = "owner/offline";
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let hub_cache = temp.path().join("hf");
        write_hub_cache(&hub_cache, &repo.folder_name(), COMMIT, &package);
        let cached = tree_contents(&hub_cache);
        let routes = routes_with(&package, repository, COMMIT, lfs_etag);
        let (primary, transport, bytes) =
            fetch_with_hub_cache(&package, repository, hub_cache.clone(), routes);
        assert_eq!(bytes, Some(0), "an LFS ETag states the file's SHA-256");
        assert!(transport.root().is_dir());
        assert_eq!(
            std::fs::read(&primary).unwrap(),
            std::fs::read(package.join("model.safetensors")).unwrap()
        );
        assert_eq!(
            tree_contents(&hub_cache),
            cached,
            "the Hugging Face cache must stay unchanged"
        );
    }

    #[test]
    fn git_blob_identity_reuses_a_hugging_face_cache_file() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, _publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let repository = "owner/offline";
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let hub_cache = temp.path().join("hf");
        write_hub_cache(&hub_cache, &repo.folder_name(), COMMIT, &package);
        let cached = tree_contents(&hub_cache);
        let routes = routes_with(&package, repository, COMMIT, git_blob_etag);
        let (primary, _transport, bytes) =
            fetch_with_hub_cache(&package, repository, hub_cache.clone(), routes);
        assert_eq!(bytes, Some(0), "a Git blob identity is verifiable");
        assert_eq!(
            std::fs::read(&primary).unwrap(),
            std::fs::read(package.join("model.safetensors")).unwrap()
        );
        assert_eq!(tree_contents(&hub_cache), cached);
    }

    #[test]
    fn corrupt_hugging_face_cache_file_is_ignored_and_downloaded() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, _publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let repository = "owner/offline";
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let hub_cache = temp.path().join("hf");
        write_hub_cache(&hub_cache, &repo.folder_name(), COMMIT, &package);
        let primary_entry = hub_cache
            .join(repo.folder_name())
            .join("snapshots")
            .join(COMMIT)
            .join("model.safetensors");
        let mut bytes = std::fs::read(&primary_entry).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&primary_entry, &bytes).unwrap();
        let cached = tree_contents(&hub_cache);
        let routes = routes_with(&package, repository, COMMIT, lfs_etag);
        let (primary, _transport, transferred) =
            fetch_with_hub_cache(&package, repository, hub_cache.clone(), routes);
        assert!(
            transferred.unwrap_or_default() > 0,
            "a file that fails its stated identity must be downloaded"
        );
        assert_eq!(
            std::fs::read(&primary).unwrap(),
            std::fs::read(package.join("model.safetensors")).unwrap(),
            "the transport copy must hold the downloaded, verified bytes"
        );
        assert_eq!(
            tree_contents(&hub_cache),
            cached,
            "the Hugging Face cache must stay unchanged"
        );
    }

    /// Seed one transport blob with corrupt bytes of the expected length.
    fn seed_corrupt_blob(
        root: &Path,
        scope: &str,
        folder: &str,
        key: &str,
        corrupt: &[u8],
    ) -> std::path::PathBuf {
        let _ = folder;
        let blob = root.join("blobs").join(scope).join(key);
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, corrupt).unwrap();
        blob
    }

    #[test]
    fn corrupt_reused_content_with_a_stated_digest_is_quarantined_and_refetched() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _) = signed_payload_fixture(temp.path());
        let model_root = temp.path().join("models");
        let downloads = temp.path().join("downloads");
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            cache_dir: Some(downloads.clone()),
            ..Default::default()
        };
        let routes = routes_with(&package, "owner/model", "gb10", sha256_etag);
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let folder =
            Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into()).folder_name();
        let scope = Transport::scope_for(&endpoint, &folder);
        let root = Transport::path(&model_root, Some(&downloads));
        // A size-correct but corrupt copy of the primary is already in the transport.
        let good = std::fs::read(package.join("model.safetensors")).unwrap();
        let mut corrupt = good.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0xff;
        let key = blob_key(&sha256_etag(&good)).unwrap();
        seed_corrupt_blob(&root, &scope, &folder, &key, &corrupt);
        let entry = download_with_token_at(
            &endpoint,
            "owner/model@gb10",
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(&publisher),
            None,
        )
        .unwrap();
        worker.join().unwrap();
        // The stated digest proved the bytes wrong, so exactly the primary was
        // transferred again and the suspect bytes were preserved.
        assert_eq!(
            log.lock().unwrap().files[&format!("/owner/model/resolve/{COMMIT}/model.safetensors")],
            1
        );
        let quarantined: Vec<_> = std::fs::read_dir(root.join("blobs").join(&scope))
            .unwrap()
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().contains(".corrupt-"))
            .collect();
        assert_eq!(quarantined.len(), 1, "the suspect bytes are preserved");
        assert_eq!(std::fs::read(quarantined[0].path()).unwrap(), corrupt);
        let primary = entry_primary(&entry).unwrap();
        let verified = flare_helios::runtime_package::VerifiedPackage::open(
            &primary,
            std::slice::from_ref(&publisher),
            None,
        )
        .unwrap();
        verified.verify_stored_data().unwrap();
    }

    #[test]
    fn corrupt_opaque_reuse_is_preserved_and_reported_not_silently_retried() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _) = signed_payload_fixture(temp.path());
        let names = package_files(&package);
        let companion = names
            .iter()
            .find(|name| name.as_str() != "model.safetensors")
            .expect("fixture has a companion")
            .clone();
        let model_root = temp.path().join("models");
        let downloads = temp.path().join("downloads");
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            cache_dir: Some(downloads.clone()),
            ..Default::default()
        };
        // Bare 64-hex ETags are opaque for mandatory verification.
        let routes = routes_with(&package, "owner/model", "gb10", lfs_etag);
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let folder =
            Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into()).folder_name();
        let scope = Transport::scope_for(&endpoint, &folder);
        let root = Transport::path(&model_root, Some(&downloads));
        let good = std::fs::read(package.join(&companion)).unwrap();
        let mut corrupt = good.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0xff;
        let key = blob_key(&lfs_etag(&good)).unwrap();
        let blob = seed_corrupt_blob(&root, &scope, &folder, &key, &corrupt);
        let error = match download_with_token_at(
            &endpoint,
            "owner/model@gb10",
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(&publisher),
            None,
        ) {
            Ok(_) => panic!("corrupt opaque content must not install"),
            Err(error) => error.to_string(),
        };
        worker.join().unwrap();
        assert!(
            error.contains("payload digest") || error.contains("declared identity"),
            "{error}"
        );
        assert!(
            error.contains("no transport bytes were removed"),
            "the failure names the retained copy: {error}"
        );
        assert_eq!(
            std::fs::read(&blob).unwrap(),
            corrupt,
            "an unproven suspect is preserved byte for byte"
        );
        let quarantined: Vec<_> = std::fs::read_dir(root.join("blobs").join(&scope))
            .unwrap()
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().contains(".corrupt-"))
            .collect();
        assert!(
            quarantined.is_empty(),
            "an unproven suspect is never quarantined or re-downloaded"
        );
    }

    #[test]
    fn opaque_file_identity_does_not_reuse_a_hugging_face_cache_file() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, _publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let repository = "owner/offline";
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let hub_cache = temp.path().join("hf");
        write_hub_cache(&hub_cache, &repo.folder_name(), COMMIT, &package);
        let routes = routes_with(&package, repository, COMMIT, |_| {
            "opaquetag0123456789abcdef012345".into()
        });
        let (_primary, _transport, transferred) =
            fetch_with_hub_cache(&package, repository, hub_cache, routes);
        assert!(
            transferred.unwrap_or_default() > 0,
            "an opaque ETag certifies no digest, so the file must be downloaded"
        );
    }

    #[test]
    fn configured_acquisition_reuses_a_matching_bare_lfs_cache_file_without_a_transfer() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let repository = "owner/offline";
        let model_root = temp.path().join("models");
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let hub_cache = temp.path().join("hf");
        write_hub_cache(&hub_cache, &repo.folder_name(), COMMIT, &package);
        let cached = tree_contents(&hub_cache);

        // Options come from an explicitly selected configuration file, exactly as
        // `flare --config PATH download` builds them. The default configuration
        // location is elsewhere, so any reload of it would lose the hub cache and
        // transfer the files again.
        let config_path = temp.path().join("explicit.toml");
        std::fs::write(
            &config_path,
            format!("[paths]\nhub_cache = '{}'\n", hub_cache.display()),
        )
        .unwrap();
        use clap::{CommandFactory, FromArgMatches};
        let matches = crate::cli::Cli::command()
            .try_get_matches_from([
                "flare",
                "--config",
                config_path.to_str().unwrap(),
                "download",
                "owner/offline@gb10",
                "--revision",
                "gb10",
                "--gpus",
                "1",
                "--model-cache",
                model_root.to_str().unwrap(),
            ])
            .unwrap();
        let mut parsed = crate::cli::Cli::from_arg_matches(&matches).unwrap();
        crate::config::Configuration::load(&config_path, true)
            .unwrap()
            .apply(&mut parsed, &matches, temp.path())
            .unwrap();
        let options = match parsed.command {
            crate::cli::Command::Download(args) => args.hub,
            _ => panic!("download command"),
        };
        assert_eq!(options.hub_cache.as_deref(), Some(hub_cache.as_path()));
        assert_eq!(options.model_cache.as_deref(), Some(model_root.as_path()));

        // The acquisition resolves the requested branch, then fetches the commit.
        let routes = routes_with(&package, repository, "gb10", lfs_etag);
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let entry = download_for_model_at(
            &endpoint,
            "owner/offline@gb10",
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(&publisher),
            None,
            None,
        )
        .unwrap();
        worker.join().unwrap();
        assert!(entry.directory().join("model.safetensors").is_file());
        {
            let log = log.lock().unwrap();
            assert_eq!(log.bytes, 0, "reused files transferred {:?}", log.files);
            assert!(log.files.is_empty(), "content requests: {:?}", log.files);
        }
        assert_eq!(
            tree_contents(&hub_cache),
            cached,
            "the Hugging Face cache must stay unchanged"
        );
    }

    #[test]
    fn pending_network_bytes_excludes_saved_prefixes_and_cached_files() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, _publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let repository = "owner/offline";
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let folder = repo.folder_name();
        // The helper meters every named file, so the primary participates here as
        // the treating-cache wiring itself would see it.
        let names = package_files(&package);
        assert!(names.len() >= 3, "fixture files: {names:?}");
        assert!(names.iter().any(|name| name == "model.safetensors"));
        let routes = routes_with(&package, repository, COMMIT, lfs_etag);
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let api = transport_api(endpoint);
        let hub_cache = temp.path().join("hf");
        let transport = Transport::open(temp.path(), None, "http://mock")
            .unwrap()
            .with_hub_cache(Some(hub_cache.clone()));

        // The primary is complete in the transport snapshot.
        let complete = "model.safetensors".to_owned();
        let complete_path = transport.snapshot_dir(&folder, COMMIT).join(&complete);
        std::fs::create_dir_all(complete_path.parent().unwrap()).unwrap();
        std::fs::write(
            &complete_path,
            std::fs::read(package.join(&complete)).unwrap(),
        )
        .unwrap();

        // One companion has half of its bytes saved as a resume prefix.
        let resumed = names
            .iter()
            .find(|name| name.as_str() != complete)
            .unwrap()
            .clone();
        let resumed_body = std::fs::read(package.join(&resumed)).unwrap();
        let half = (resumed_body.len() / 2) as u64;
        let key = blob_key(&lfs_etag(&resumed_body)).unwrap();
        let part = transport.part(&folder, &key);
        std::fs::create_dir_all(part.parent().unwrap()).unwrap();
        std::fs::write(&part, &resumed_body[..half as usize]).unwrap();

        let primary = transport
            .snapshot_dir(&folder, COMMIT)
            .join("model.safetensors");
        // The third companion has no local bytes yet, so only the completed file
        // and the saved prefix reduce the network work.
        let other = names
            .iter()
            .find(|name| name.as_str() != complete && **name != resumed)
            .map(|name| std::fs::metadata(package.join(name)).unwrap().len())
            .unwrap();
        let missing = resumed_body.len() as u64 - half + other;
        let pending =
            pending_network_bytes(&api, &transport, &repo, &primary, &names, None).unwrap();
        assert_eq!(
            pending.total,
            Some(missing),
            "only the bytes the network must still deliver are pending"
        );
        assert_eq!(pending.bound, missing, "nothing else is unaccounted for");

        // The remaining companion exists only in the user Hugging Face cache.
        // Its size alone proves nothing, so the network volume is unknown until
        // adoption validates it instead of being promised as zero.
        let cached = names
            .iter()
            .find(|name| name.as_str() != complete && **name != resumed)
            .unwrap()
            .clone();
        let cached_body = std::fs::read(package.join(&cached)).unwrap();
        let cached_path = hub_cache
            .join(&folder)
            .join("snapshots")
            .join(COMMIT)
            .join(&cached);
        std::fs::create_dir_all(cached_path.parent().unwrap()).unwrap();
        std::fs::write(&cached_path, &cached_body).unwrap();
        let pending =
            pending_network_bytes(&api, &transport, &repo, &primary, &names, None).unwrap();
        worker.join().unwrap();
        assert_eq!(
            pending.total, None,
            "an unvalidated cache file makes the network total unknown"
        );
        assert_eq!(
            pending.bound, missing,
            "the advisory bound still counts bytes that may have to be fetched"
        );
    }

    #[test]
    fn opaque_identity_still_counts_a_saved_resume_prefix() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, _publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let repository = "owner/offline";
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let folder = repo.folder_name();
        // Distinct opaque ETags: content addressing keys every file separately.
        let next = std::sync::atomic::AtomicUsize::new(0);
        let routes = routes_with(&package, repository, COMMIT, |_| {
            format!(
                "opaque-{}",
                next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            )
        });
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let api = transport_api(endpoint);
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        let names = package_files(&package);
        let resumed = names[0].clone();
        let body = std::fs::read(package.join(&resumed)).unwrap();
        let half = (body.len() / 2) as u64;
        let key = blob_key("opaque-0").unwrap();
        let part = transport.part(&folder, &key);
        std::fs::create_dir_all(part.parent().unwrap()).unwrap();
        std::fs::write(&part, &body[..half as usize]).unwrap();
        let primary = transport
            .snapshot_dir(&folder, COMMIT)
            .join("model.safetensors");
        let pending =
            pending_network_bytes(&api, &transport, &repo, &primary, &names, None).unwrap();
        worker.join().unwrap();
        let other: u64 = names[1..]
            .iter()
            .map(|name| std::fs::metadata(package.join(name)).unwrap().len())
            .sum();
        assert_eq!(
            pending.total,
            Some(body.len() as u64 - half + other),
            "a saved prefix resumes even when the ETag states no digest"
        );
    }

    #[test]
    fn rejected_redirects_and_invalid_urls_do_not_echo_credentials() {
        const SECRET: &str = "topsecret-credential";
        for location in [
            format!("https://user:{SECRET}@content.example/file?sig={SECRET}"),
            format!("file:///srv/file?sig={SECRET}"),
        ] {
            let response = format!(
                "HTTP/1.1 302 Found\r\nlocation: {location}\r\n\
                 Content-Length: 0\r\nConnection: close\r\n\r\n"
            );
            let routes =
                HubRoutes::default().document("/owner/model/resolve/x", response.into_bytes());
            let (endpoint, worker) = mock_hub(routes);
            let url = format!("{endpoint}/owner/model/resolve/x");
            let body_message = match ranged_get(&transfers(), &url, None, 0) {
                Err(Attempt::Fatal(message)) => message,
                _ => panic!("expected the redirect to be rejected"),
            };
            let identity = match file_identity(&url, None) {
                Err(failure) => failure.message,
                Ok(_) => panic!("expected the redirect to be rejected"),
            };
            worker.join().unwrap();
            for message in [body_message, identity] {
                assert!(
                    !message.contains(SECRET),
                    "an error must never repeat redirect credentials: {message}"
                );
                assert!(
                    message.contains("unsupported target"),
                    "the rejection must be reported by kind: {message}"
                );
            }
        }

        // A URI rejected by the transport carries no user-visible value either.
        let bad = ureq::Error::BadUri(format!("https://user:{SECRET}@host/file"));
        let message = transport_error(&bad);
        assert!(!message.contains(SECRET), "{message}");
        assert_eq!(message, "invalid request URL");
        match attempt_error(ureq::Error::BadUri(format!("https://host/{SECRET}"))) {
            Attempt::Fatal(message) => assert!(!message.contains(SECRET), "{message}"),
            _ => panic!("an invalid URI must be fatal"),
        }
    }

    #[test]
    fn pending_network_bytes_reports_unknown_for_an_unverified_sha256_blob() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, _publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let repository = "owner/offline";
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let folder = repo.folder_name();
        let routes = routes_with(&package, repository, COMMIT, sha256_etag);
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let api = transport_api(endpoint);
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        let names = package_files(&package);
        let cached = names[0].clone();
        let body = std::fs::read(package.join(&cached)).unwrap();
        let key = blob_key(&sha256_etag(&body)).unwrap();
        let blob = transport.blob(&folder, &key);
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, &body).unwrap();
        let primary = transport
            .snapshot_dir(&folder, COMMIT)
            .join("model.safetensors");
        let pending =
            pending_network_bytes(&api, &transport, &repo, &primary, &names, None).unwrap();
        worker.join().unwrap();
        assert_eq!(
            pending.total, None,
            "a stated SHA-256 is checked before reuse, so its bytes may still transfer"
        );
        let rest: u64 = names[1..]
            .iter()
            .map(|name| std::fs::metadata(package.join(name)).unwrap().len())
            .sum();
        assert_eq!(
            pending.bound,
            body.len() as u64 + rest,
            "the advisory bound still covers a possible refetch"
        );
    }

    #[test]
    fn pending_network_bytes_counts_a_shared_identity_once() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, _publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let repository = "owner/offline";
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let mut names = package_files(&package);
        // Two paths that the Hub serves with identical bytes: one content
        // identity, one transfer.
        let shared = names[0].clone();
        let other = names[1].clone();
        let shared_body = std::fs::read(package.join(&shared)).unwrap();
        let mut routes = routes_with(&package, repository, COMMIT, lfs_etag);
        routes = routes.file(
            format!("/{repository}/resolve/{COMMIT}/{other}"),
            MockFile::new(&lfs_etag(&shared_body), shared_body.clone()),
        );
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let api = transport_api(endpoint);
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        let primary = transport
            .snapshot_dir(&repo.folder_name(), COMMIT)
            .join("model.safetensors");
        names.truncate(2);
        let pending =
            pending_network_bytes(&api, &transport, &repo, &primary, &names, None).unwrap();
        worker.join().unwrap();
        assert_eq!(
            pending.total,
            Some(shared_body.len() as u64),
            "one identity is one transfer, whatever number of paths name it"
        );
        assert_eq!(pending.bound, shared_body.len() as u64);
    }

    #[test]
    fn corrupt_transport_blob_progress_stays_unknown_then_downloads() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let repository = "owner/offline";
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let folder = repo.folder_name();
        let routes = routes_with(&package, repository, "gb10", sha256_etag);
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let model_root = temp.path().join("models");
        let cache_dir = temp.path().join("downloads");
        let transport = Transport::open(&model_root, Some(&cache_dir), "http://mock").unwrap();
        // A same-sized corrupt blob: only digest validation can reject it.
        let primary_body = std::fs::read(package.join("model.safetensors")).unwrap();
        let key = blob_key(&sha256_etag(&primary_body)).unwrap();
        let blob = transport.blob(&folder, &key);
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        let mut corrupt = primary_body.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0xff;
        std::fs::write(&blob, &corrupt).unwrap();
        let api = transport_api(endpoint.clone());
        let pending = pending_network_bytes(
            &api,
            &transport,
            &repo,
            &transport
                .snapshot_dir(&folder, COMMIT)
                .join("model.safetensors"),
            &["model.safetensors".to_owned()],
            None,
        )
        .unwrap();
        assert_eq!(
            pending.total, None,
            "an unverified blob is not counted as saved"
        );
        drop(transport);

        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            cache_dir: Some(cache_dir.clone()),
            ..Default::default()
        };
        let counters = flare_helios::package::progress::AcquisitionProgress::new();
        let entry = download_with_token_at(
            &endpoint,
            "owner/offline@gb10",
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(&publisher),
            Some(counters.clone()),
        )
        .unwrap();
        worker.join().unwrap();
        assert!(entry.directory().join("model.safetensors").is_file());
        let snapshot = counters.snapshot();
        let declared: u64 = package_files(&package)
            .iter()
            .map(|name| std::fs::metadata(package.join(name)).unwrap().len())
            .sum();
        assert_eq!(
            snapshot.network_bytes, declared,
            "the corrupt blob is refetched with the rest of the package"
        );
        assert!(!log.lock().unwrap().files.is_empty());
        assert_eq!(
            snapshot.network_total,
            Some(snapshot.network_bytes),
            "a completed acquisition reports no outstanding fraction"
        );
    }

    #[test]
    fn corrupt_hugging_face_cache_progress_stays_unknown_then_downloads() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let repository = "owner/offline";
        let repo = Repo::with_revision(repository.into(), RepoType::Model, COMMIT.into());
        let folder = repo.folder_name();
        let hub_cache = temp.path().join("hf");
        write_hub_cache(&hub_cache, &folder, COMMIT, &package);
        let cached = hub_cache
            .join(&folder)
            .join("snapshots")
            .join(COMMIT)
            .join("model.safetensors");
        let mut bytes = std::fs::read(&cached).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&cached, &bytes).unwrap();
        let routes = routes_with(&package, repository, "gb10", lfs_etag);
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let model_root = temp.path().join("models");
        let cache_dir = temp.path().join("downloads");
        // Before the run the estimate cannot promise the corrupt file's bytes, so
        // it reports an unknown network total instead of zero.
        let transport = Transport::open(&model_root, Some(&cache_dir), "http://mock")
            .unwrap()
            .with_hub_cache(Some(hub_cache.clone()));
        let api = transport_api(endpoint.clone());
        let pending = pending_network_bytes(
            &api,
            &transport,
            &repo,
            &transport
                .snapshot_dir(&folder, COMMIT)
                .join("model.safetensors"),
            &["model.safetensors".to_owned()],
            None,
        )
        .unwrap();
        assert_eq!(
            pending.total, None,
            "an unvalidated cache file is not saved work"
        );
        drop(transport);
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            cache_dir: Some(cache_dir.clone()),
            hub_cache: Some(hub_cache.clone()),
            ..Default::default()
        };
        let counters = flare_helios::package::progress::AcquisitionProgress::new();
        let entry = download_with_token_at(
            &endpoint,
            "owner/offline@gb10",
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(&publisher),
            Some(counters.clone()),
        )
        .unwrap();
        worker.join().unwrap();
        assert!(entry.directory().join("model.safetensors").is_file());
        let snapshot = counters.snapshot();
        let corrupt_size = std::fs::metadata(package.join("model.safetensors"))
            .unwrap()
            .len();
        assert_eq!(
            snapshot.network_bytes, corrupt_size,
            "only the file that failed its identity is downloaded; certified companions are reused"
        );
        assert_eq!(
            snapshot.network_total,
            Some(snapshot.network_bytes),
            "an unvalidated cache file never becomes a promised zero"
        );
    }

    #[test]
    fn offline_download_never_consults_the_hugging_face_cache() {
        let temp = tempfile::tempdir().unwrap();
        let (_package, _profile, publisher, _revision) =
            signed_payload_fixture(&temp.path().join("package"));
        let model_root = temp.path().join("models");
        let options = HubOptions {
            offline: true,
            revision: Some("gb10".into()),
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let error = download_with_token_at(
            "http://127.0.0.1:9",
            "owner/offline@gb10",
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(&publisher),
            None,
        )
        .err()
        .expect("an empty prepared-model cache cannot satisfy an offline download")
        .to_string();
        assert!(
            error.contains("not available in the Flare cache"),
            "offline resolution must report the Flare cache, not a Hub failure: {error}"
        );
        assert!(
            !model_root.join("transport").exists(),
            "offline resolution returns before a transport exists, so no Hub cache is opened"
        );
    }

    #[test]
    fn offline_download_never_imports_hf_snapshots() {
        let temp = tempfile::tempdir().unwrap();
        let transport = Transport::open(temp.path(), None, "https://huggingface.co").unwrap();
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let directory = transport.snapshot_dir(&repo.folder_name(), COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), fixture(&[])).unwrap();
        transport.record_ref(&repo.folder_name(), "gb202", COMMIT);
        let options = HubOptions {
            offline: true,
            revision: Some("gb202".into()),
            model_cache: Some(temp.path().join("flare")),
            token_file: Some(temp.path().join("nonexistent-token")),
            ..Default::default()
        };
        let error = download("owner/model", &options, 1, &[], None, &[])
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("not available in the Flare cache"),
            "{error}"
        );
        assert_eq!(
            std::fs::read_dir(temp.path().join("flare/models"))
                .unwrap()
                .count(),
            0
        );
        // Windows cannot delete the acquisition lease while this transport holds it.
        let transport_root = transport.root().to_owned();
        drop(transport);
        std::fs::remove_dir_all(&transport_root).unwrap();
        let error = download("owner/model", &options, 1, &[], None, &[])
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("not available in the Flare cache"),
            "{error}"
        );
        assert!(!transport_root.exists());
    }

    #[test]
    fn offline_branch_does_not_select_an_older_commit_for_its_tp() {
        use flare_helios::package::cache::Root;
        let temp = tempfile::tempdir().unwrap();
        let root = Root::open(temp.path()).unwrap();
        let first = root.stage("model", "first").unwrap().commit().unwrap();
        let second = root.stage("model", "second").unwrap().commit().unwrap();
        let tp1 = HubOptions {
            tensor_parallel: vec![1],
            ..Default::default()
        };
        let tp2 = HubOptions {
            tensor_parallel: vec![2],
            ..Default::default()
        };
        remember_download(
            temp.path(),
            "owner/model",
            "gb202",
            COMMIT,
            "model.safetensors",
            &tp1,
            1,
            &first,
            [0; 32],
        )
        .unwrap();
        let other_commit = "abcdefabcdefabcdefabcdefabcdefabcdefabcd";
        remember_download(
            temp.path(),
            "owner/model",
            "gb202",
            other_commit,
            "model.safetensors",
            &tp2,
            1,
            &second,
            [0; 32],
        )
        .unwrap();
        let branch = offline_entry(temp.path(), "owner/model", "gb202", &tp1, 1).unwrap();
        assert_eq!(branch.id(), second.id());
        let pinned = offline_entry(temp.path(), "owner/model", COMMIT, &tp1, 1).unwrap();
        assert_eq!(pinned.id(), first.id());
        drop(pinned);
        drop(branch);
        let first_id = first.id().to_owned();
        drop(first);
        root.evict("model", &first_id).unwrap();
        assert_eq!(
            offline_entry(temp.path(), "owner/model", "gb202", &tp1, 1)
                .unwrap()
                .id(),
            second.id()
        );
        assert!(offline_entry(temp.path(), "owner/model", COMMIT, &tp1, 1).is_err());
    }

    #[test]
    fn offline_download_selects_an_already_installed_nondefault_entry() {
        use flare_helios::package::cache::{Root, model_key};
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, commit) = signed_payload_fixture(temp.path());
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("gb202".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            offline: true,
            ..Default::default()
        };
        let key = model_key("owner/offline").unwrap();
        let root = Root::open(&model_root).unwrap();
        let prior = root.stage_named(&key, "prior").unwrap().commit().unwrap();
        root.publish_entry(&prior).unwrap();

        // Install the signed entry but leave the prior default selected.
        let installed = install_ready(
            "owner/offline",
            &package,
            &options,
            &model_root,
            1,
            &[publisher.clone()],
        )
        .unwrap();
        let installed_id = installed.id().to_owned();
        assert_ne!(
            root.ref_entry(&key).unwrap().as_deref(),
            Some(installed_id.as_str())
        );
        let expected_envelope =
            authenticated_source_envelope(&package, &options, 1, std::slice::from_ref(&publisher))
                .unwrap();
        remember_download(
            &model_root,
            "owner/offline",
            "gb202",
            &commit,
            "model.safetensors",
            &options,
            1,
            &installed,
            expected_envelope,
        )
        .unwrap();
        drop(installed);

        // The public offline download path resolves the installed source reference
        // and selects that existing entry without installing or downloading again.
        let selected = download_with_token(
            "owner/offline@gb202",
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(&publisher),
        )
        .unwrap();
        assert_eq!(selected.id(), installed_id);
        assert_eq!(
            root.ref_entry(&key).unwrap().as_deref(),
            Some(installed_id.as_str())
        );
        assert!(root.entry(&key, &installed_id).is_ok());
        drop(selected);
        drop(prior);
        let _ = std::fs::remove_dir_all(temp.path());
    }

    #[test]
    fn failed_source_ref_recording_preserves_the_prior_default() {
        use flare_helios::package::cache::{Root, model_key};
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let root = Root::open(&model_root).unwrap();
        let key = model_key("owner/closure").unwrap();
        // A previously acquired and selected version.
        let prior = root.stage_named(&key, "prior").unwrap().commit().unwrap();
        root.publish_entry(&prior).unwrap();
        // The newly installed but not yet selected entry, as install_ready leaves it.
        let fresh = root.stage_named(&key, "fresh").unwrap().commit().unwrap();
        // Force the first source-reference publish to fail: its destination is a directory.
        let options = HubOptions {
            tensor_parallel: vec![1],
            ..Default::default()
        };
        let selector = source_selector("owner/closure", "gb202", Some("model.safetensors"), &[1]);
        std::fs::create_dir_all(
            model_root
                .join("sources")
                .join(flare_helios::package::PayloadId::of(selector.as_bytes()).to_string()),
        )
        .unwrap();
        assert!(
            remember_download(
                &model_root,
                "owner/closure",
                "gb202",
                COMMIT,
                "model.safetensors",
                &options,
                1,
                &fresh,
                [0; 32],
            )
            .is_err()
        );
        // The prior default is untouched and the failed acquisition is not selected.
        assert_eq!(root.ref_entry(&key).unwrap().as_deref(), Some(prior.id()));
        assert_ne!(prior.id(), fresh.id());
        let fresh_id = fresh.id().to_owned();
        drop(fresh);
        root.remove(&key, &fresh_id, false).unwrap();
        let _ = std::fs::remove_dir_all(temp.path());
    }

    #[test]
    fn offline_exact_tp_entry_survives_eviction_of_the_latest_union() {
        use flare_helios::package::cache::Root;
        let temp = tempfile::tempdir().unwrap();
        let root = Root::open(temp.path()).unwrap();
        let tp1_entry = root.stage("model", "tp1").unwrap().commit().unwrap();
        let tp2_entry = root.stage("model", "tp2").unwrap().commit().unwrap();
        let tp1 = HubOptions {
            tensor_parallel: vec![1],
            ..Default::default()
        };
        let tp2 = HubOptions {
            tensor_parallel: vec![2],
            ..Default::default()
        };
        remember_download(
            temp.path(),
            "owner/model",
            "gb202",
            COMMIT,
            "model.safetensors",
            &tp1,
            1,
            &tp1_entry,
            [1; 32],
        )
        .unwrap();
        remember_download(
            temp.path(),
            "owner/model",
            "gb202",
            COMMIT,
            "model.safetensors",
            &tp2,
            1,
            &tp2_entry,
            [2; 32],
        )
        .unwrap();
        let tp2_id = tp2_entry.id().to_owned();
        drop(tp2_entry);
        root.evict("model", &tp2_id).unwrap();
        // The unqualified target is gone, but the last resolved commit is known
        // and the same-commit exact TP1 entry is still ready.
        let (hit, expected) =
            offline_entry_with_expected(temp.path(), "owner/model", "gb202", &tp1, 1).unwrap();
        assert_eq!(hit.id(), tp1_entry.id());
        assert_eq!(expected, [1; 32]);
    }

    /// Names in one transport scope that are not lock inodes.
    fn retained_transport_files(root: &Path, scope: &str) -> Vec<String> {
        let blobs = root.join("blobs").join(scope);
        let Ok(entries) = std::fs::read_dir(blobs) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .flatten()
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| !name.ends_with(".lock"))
            .collect();
        names.sort();
        names
    }

    /// Download a signed fixture through the transport and return its entry and
    /// the transport root the acquisition used.
    fn fixture_download(
        temp: &Path,
        package: &Path,
        publisher: &flare_helios::runtime_package::Publisher,
    ) -> (
        flare_helios::package::cache::Entry,
        std::path::PathBuf,
        String,
        String,
        String,
    ) {
        let model_root = temp.join("models");
        let downloads = temp.join("downloads");
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            cache_dir: Some(downloads.clone()),
            ..Default::default()
        };
        let routes = package_routes(package, "owner/model", "gb10");
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let entry = download_with_token_at(
            &endpoint,
            "owner/model@gb10",
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(publisher),
            None,
        )
        .unwrap();
        worker.join().unwrap();
        let folder =
            Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into()).folder_name();
        let scope = Transport::scope_for(&endpoint, &folder);
        let root = Transport::path(&model_root, Some(&downloads));
        (entry, root, scope, folder, endpoint)
    }

    #[test]
    fn local_acquisition_reports_advisory_space_without_rejecting() {
        use flare_helios::package::progress::format_bytes;
        let temp = tempfile::tempdir().unwrap();
        let existing = temp.path().join("cache");
        std::fs::create_dir_all(&existing).unwrap();
        let missing = temp.path().join("not/created/yet");
        assert!(
            available_bytes(&existing).is_some_and(|free| free > 0),
            "free space is reported for an existing path"
        );
        assert!(
            available_bytes(&missing).is_some_and(|free| free > 0),
            "free space is estimated from the nearest existing ancestor"
        );
        assert!(
            free_bytes_at(Path::new("")).is_none(),
            "an unavailable filesystem query is unknown, not zero"
        );
        assert!(
            !temp.path().join("not").exists(),
            "estimates create nothing"
        );
        let known = space_advisory(
            available_bytes(&existing),
            available_bytes(&missing),
            Some(8 * 1024 * 1024),
        );
        assert!(known.contains(&format_bytes(8 * 1024 * 1024)), "{known}");
        assert!(known.contains("known transfer"), "{known}");
        assert!(known.contains("available"), "{known}");
        let full = space_advisory(Some(0), Some(0), Some(0));
        assert!(full.contains(&format_bytes(0)), "{full}");
        assert!(
            !full.contains("unknown available"),
            "a full filesystem is reported as zero free space: {full}"
        );
        let unknown = space_advisory(None, Some(1024), None);
        assert!(unknown.contains("known transfer unknown"), "{unknown}");
        assert!(
            unknown.contains("Transport space: unknown available"),
            "an unavailable query renders as unknown: {unknown}"
        );
        assert!(unknown.contains(&format_bytes(1024)), "{unknown}");
    }

    /// Child process for the interrupted-acquisition test. Only selected
    /// explicitly by the parent; it reads its endpoint and paths from the
    /// test's environment, never from production configuration.
    #[test]
    #[ignore = "spawned by the interrupted-acquisition test"]
    fn download_child_process() {
        let Ok(endpoint) = std::env::var("FLARE_TEST_DOWNLOAD_ENDPOINT") else {
            return;
        };
        let repository = std::env::var("FLARE_TEST_DOWNLOAD_REPOSITORY").unwrap();
        let revision = std::env::var("FLARE_TEST_DOWNLOAD_REVISION").unwrap();
        let model_cache = std::env::var("FLARE_TEST_DOWNLOAD_MODEL_CACHE").unwrap();
        let cache_dir = std::env::var("FLARE_TEST_DOWNLOAD_CACHE_DIR").unwrap();
        let profile: flare_helios::runtime_package::Publisher =
            serde_json::from_str(&std::env::var("FLARE_TEST_DOWNLOAD_PUBLISHER").unwrap()).unwrap();
        let options = HubOptions {
            revision: Some(revision.clone()),
            tensor_parallel: vec![1],
            model_cache: Some(model_cache.into()),
            cache_dir: Some(cache_dir.into()),
            ..Default::default()
        };
        download_with_token_at(
            &endpoint,
            &format!("{repository}@{revision}"),
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(&profile),
            None,
        )
        .unwrap();
    }

    /// Kills and reaps a spawned acquisition on every path, including a panic,
    /// so a failed assertion never leaves an orphan downloader running.
    struct ChildGuard(Option<std::process::Child>);

    impl ChildGuard {
        fn new(child: std::process::Child) -> Self {
            Self(Some(child))
        }

        fn kill(mut self) {
            self.stop();
        }

        fn stop(&mut self) {
            if let Some(mut child) = self.0.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    impl Drop for ChildGuard {
        fn drop(&mut self) {
            self.stop();
        }
    }

    /// Wait until `part` reaches `bytes`, then report its length.
    fn wait_for_partial(part: &Path, bytes: u64, timeout: Duration) -> u64 {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let length = std::fs::symlink_metadata(part)
                .map(|info| info.len())
                .unwrap_or(0);
            if length >= bytes || std::time::Instant::now() >= deadline {
                return length;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn a_killed_acquisition_resumes_the_partial_and_reuses_completed_files() {
        use flare_helios::package::cache::Root;
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _) =
            signed_payload_fixture(&temp.path().join("package"));
        let names = package_files(&package);
        let interrupted = names
            .iter()
            .rev()
            .find(|name| name.as_str() != "model.safetensors")
            .expect("fixture has a companion")
            .clone();
        let interrupted_body = std::fs::read(package.join(&interrupted)).unwrap();
        let half = (interrupted_body.len() / 2) as u64;
        let etag = sha256_hex(&interrupted_body);
        let routes = package_routes(&package, "owner/model", "gb10").file(
            format!("/owner/model/resolve/{COMMIT}/{interrupted}"),
            MockFile::new(&etag, interrupted_body.clone()).with_faults([
                MockFault::Short(half as usize),
                MockFault::Stall(Duration::from_secs(10)),
            ]),
        );
        let (endpoint, _worker, log) = mock_hub_counted_patient(routes);
        let model_root = temp.path().join("models");
        let cache_dir = temp.path().join("downloads");
        let folder =
            Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into()).folder_name();
        let scope = Transport::scope_for(&endpoint, &folder);
        let root = Transport::path(&model_root, Some(&cache_dir));
        let part = root
            .join("blobs")
            .join(&scope)
            .join(format!("{}.part", blob_key(&etag).unwrap()));

        let child = ChildGuard::new(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "hub::tests::download_child_process",
                    "--ignored",
                    "--nocapture",
                ])
                .env("FLARE_TEST_DOWNLOAD_ENDPOINT", &endpoint)
                .env("FLARE_TEST_DOWNLOAD_REPOSITORY", "owner/model")
                .env("FLARE_TEST_DOWNLOAD_REVISION", "gb10")
                .env("FLARE_TEST_DOWNLOAD_MODEL_CACHE", &model_root)
                .env("FLARE_TEST_DOWNLOAD_CACHE_DIR", &cache_dir)
                .env(
                    "FLARE_TEST_DOWNLOAD_PUBLISHER",
                    serde_json::to_string(&publisher).unwrap(),
                )
                .spawn()
                .unwrap(),
        );
        let observed = wait_for_partial(&part, half, Duration::from_secs(30));
        // The child holds the shared acquisition lease and the content lock.
        let collected = crate::download_cache::reclaim(&cache_dir, &model_root).unwrap();
        assert!(
            collected.acquisition_active,
            "reclamation must observe the live child acquisition"
        );
        assert_eq!(collected.complete_files, 0, "an active child is protected");
        assert_eq!(
            observed, half,
            "the child stopped after a controlled partial"
        );
        child.kill();
        assert_eq!(std::fs::read(&part).unwrap().len() as u64, half);

        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            cache_dir: Some(cache_dir.clone()),
            ..Default::default()
        };
        let counters = flare_helios::package::progress::AcquisitionProgress::new();
        let entry = download_with_token_at(
            &endpoint,
            "owner/model@gb10",
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(&publisher),
            Some(counters.clone()),
        )
        .unwrap();
        // A resumed acquisition must not report the saved prefix or the already
        // completed files as outstanding work: the declared total ends at exactly
        // the bytes it pulled, so the settled row is complete and carries no ETA.
        let snapshot = counters.snapshot();
        assert_eq!(
            snapshot.network_total,
            Some(snapshot.network_bytes),
            "a completed resume must not leave an incomplete fraction"
        );
        let row = flare_helios::package::progress::format_progress(
            snapshot.network_bytes,
            snapshot.network_total,
            Some(1.0),
        );
        assert!(!row.contains("ETA"), "{row}");

        // The restarted acquisition resumes exactly at the saved offset and does
        // not transfer any completed file again.
        let log = log.lock().unwrap();
        let ranges = |name: &str| -> Vec<u64> {
            log.ranges
                .iter()
                .filter(|(path, _)| path.ends_with(name))
                .map(|(_, start)| *start)
                .collect()
        };
        assert_eq!(ranges("model.safetensors"), vec![0]);
        for name in names.iter().filter(|name| **name != interrupted) {
            if name == "model.safetensors" {
                continue;
            }
            assert_eq!(ranges(name), vec![0], "{name} is transferred once");
        }
        let interrupted_ranges = ranges(&interrupted);
        assert_eq!(interrupted_ranges.first(), Some(&0));
        assert!(
            interrupted_ranges.len() >= 2
                && interrupted_ranges[1..].iter().all(|start| *start == half),
            "the killed attempt and the restart both resume from the saved offset: {interrupted_ranges:?}"
        );
        let expected: u64 = names
            .iter()
            .map(|name| std::fs::metadata(package.join(name)).unwrap().len())
            .sum();
        assert_eq!(
            log.bytes as u64, expected,
            "every content byte is transferred exactly once"
        );
        drop(log);

        let primary = entry_primary(&entry).unwrap();
        let verified = flare_helios::runtime_package::VerifiedPackage::open(
            &primary,
            std::slice::from_ref(&publisher),
            None,
        )
        .unwrap();
        verified.verify_stored_data().unwrap();
        let store = Root::open(&model_root).unwrap();
        let key = flare_helios::package::cache::model_key("owner/offline").unwrap();
        assert_eq!(store.ref_entry(&key).unwrap().as_deref(), Some(entry.id()));
    }

    #[test]
    fn successful_install_reclaims_consumed_transport_content() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _) = signed_payload_fixture(temp.path());
        let (entry, root, scope, folder, _endpoint) =
            fixture_download(temp.path(), &package, &publisher);
        assert!(
            retained_transport_files(&root, &scope).is_empty(),
            "consumed transport copies are reclaimed after install"
        );
        assert!(
            !root
                .join("repos")
                .join(&scope)
                .join(&folder)
                .join("snapshots")
                .join(COMMIT)
                .exists(),
            "the installed commit's snapshot tree is reclaimed"
        );
        // The installed package verifies from the model cache alone.
        let primary = entry_primary(&entry).unwrap();
        let verified = flare_helios::runtime_package::VerifiedPackage::open(
            &primary,
            std::slice::from_ref(&publisher),
            None,
        )
        .unwrap();
        verified.verify_stored_data().unwrap();
    }

    #[test]
    fn a_changed_commit_fetches_new_weights_and_reuses_cached_companions() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _) = signed_payload_fixture(&temp.path().join("first"));
        let (_, root, _, _, endpoint) =
            fixture_download(&temp.path().join("first"), &package, &publisher);
        // The successful install reclaimed its transport content, so the update
        // can only reuse payload objects already published in the model cache.
        assert!(
            retained_transport_files(
                &root,
                &Transport::scope_for(&endpoint, "models--owner--model")
            )
            .is_empty()
        );
        // A second signed package supplies a new primary (new weights and a new
        // signature) while its companion payload bytes are unchanged.
        let (updated, _, updated_publisher, _) =
            signed_payload_fixture(&temp.path().join("second"));
        let names = package_files(&updated);
        let other_commit = "abcdef0123456789012345678901234567890123";
        let names_ref: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut routes = HubRoutes::default().document(
            "/api/models/owner/model/revision/gb202",
            revision_response_with(other_commit, &names_ref),
        );
        for name in &names {
            let body = std::fs::read(updated.join(name)).unwrap();
            routes = routes.file(
                format!("/owner/model/resolve/{other_commit}/{name}"),
                MockFile::new(&sha256_hex(&body), body).at_commit(other_commit),
            );
        }
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let model_root = temp.path().join("first/models");
        let downloads = temp.path().join("first/downloads");
        let options = HubOptions {
            revision: Some("gb202".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            cache_dir: Some(downloads.clone()),
            ..Default::default()
        };
        let entry = download_with_token_at(
            &endpoint,
            "owner/model@gb202",
            &options,
            1,
            None,
            &[],
            None,
            &[publisher, updated_publisher.clone()],
            None,
        )
        .unwrap();
        worker.join().unwrap();
        let mut requested: Vec<String> = log.lock().unwrap().files.keys().cloned().collect();
        requested.sort();
        assert_eq!(
            requested,
            vec![format!(
                "/owner/model/resolve/{other_commit}/model.safetensors"
            )],
            "the changed weights are fetched and the unchanged companion is reused"
        );
        let primary = entry_primary(&entry).unwrap();
        let verified = flare_helios::runtime_package::VerifiedPackage::open(
            &primary,
            std::slice::from_ref(&updated_publisher),
            None,
        )
        .unwrap();
        verified.verify_stored_data().unwrap();
    }

    #[test]
    fn failed_install_preserves_resumable_transport_content() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _) = signed_payload_fixture(temp.path());
        let names = package_files(&package);
        let companion = names
            .iter()
            .find(|name| name.as_str() != "model.safetensors")
            .expect("fixture has a companion")
            .clone();
        let primary_bytes = std::fs::read(package.join("model.safetensors")).unwrap();
        let companion_bytes = std::fs::read(package.join(&companion)).unwrap();
        let companion_etag = sha256_hex(&companion_bytes);
        let routes = package_routes(&package, "owner/model", "gb10").file(
            format!("/owner/model/resolve/{COMMIT}/{companion}"),
            MockFile::new(&companion_etag, companion_bytes)
                .script([MockFault::Short(4), MockFault::Status(400)]),
        );
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let model_root = temp.path().join("models");
        let downloads = temp.path().join("downloads");
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            cache_dir: Some(downloads.clone()),
            ..Default::default()
        };
        let error = match download_with_token_at(
            &endpoint,
            "owner/model@gb10",
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(&publisher),
            None,
        ) {
            Ok(_) => panic!("a failed companion transfer must not install"),
            Err(error) => error.to_string(),
        };
        worker.join().unwrap();
        assert!(error.contains("HTTP 400"), "{error}");
        let folder =
            Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into()).folder_name();
        let scope = Transport::scope_for(&endpoint, &folder);
        let root = Transport::path(&model_root, Some(&downloads));
        let primary_key = blob_key(&sha256_hex(&primary_bytes)).unwrap();
        assert!(
            root.join("blobs").join(&scope).join(&primary_key).is_file(),
            "the transferred primary stays for the next attempt"
        );
        let partial = root
            .join("blobs")
            .join(&scope)
            .join(format!("{}.part", blob_key(&companion_etag).unwrap()));
        assert_eq!(
            std::fs::read(&partial).unwrap().len(),
            4,
            "the failed companion keeps its received bytes"
        );
        assert!(
            !root
                .join("repos")
                .join(&scope)
                .join(&folder)
                .join("snapshots")
                .join(COMMIT)
                .join(&companion)
                .exists(),
            "an incomplete file is never published into the snapshot"
        );
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert!(
            store
                .ref_entry(&flare_helios::package::cache::model_key("owner/offline").unwrap())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn concurrent_acquisition_defers_reclamation_until_it_finishes() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _) = signed_payload_fixture(temp.path());
        let model_root = temp.path().join("models");
        let downloads = temp.path().join("downloads");
        // A second acquisition of the same transport root is active.
        let holder = Transport::open(
            &model_root,
            Some(&downloads),
            "http://placeholder", // replaced after the endpoint is known
        )
        .unwrap();
        drop(holder);
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            cache_dir: Some(downloads.clone()),
            ..Default::default()
        };
        let routes = package_routes(&package, "owner/model", "gb10");
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let holder = Transport::open(&model_root, Some(&downloads), &endpoint).unwrap();
        let entry = download_with_token_at(
            &endpoint,
            "owner/model@gb10",
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(&publisher),
            None,
        )
        .unwrap();
        worker.join().unwrap();
        let folder =
            Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into()).folder_name();
        let scope = Transport::scope_for(&endpoint, &folder);
        let root = Transport::path(&model_root, Some(&downloads));
        assert!(
            !retained_transport_files(&root, &scope).is_empty(),
            "an active acquisition prevents reclamation"
        );
        assert!(entry_primary(&entry).is_ok());
        // Once no acquisition is active the retained copies are reclaimable.
        drop(holder);
        let consumed =
            crate::download_cache::reclaim_installed(&root, &scope, &folder, COMMIT).unwrap();
        assert!(!consumed.acquisition_active);
        assert!(consumed.complete_bytes > 0);
        assert!(retained_transport_files(&root, &scope).is_empty());
    }

    #[test]
    fn download_resolves_hardware_branch_then_fetches_the_exact_commit() {
        let temp = tempfile::tempdir().unwrap();
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let package = fixture(&[]);
        let etag = sha256_hex(&package);
        let routes = HubRoutes::default()
            .document(
                "/api/models/owner/model/revision/gb202",
                metadata_response(),
            )
            .file(
                format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
                MockFile::new(&etag, package.clone()),
            );
        let (endpoint, worker) = mock_hub(routes);
        let path = fetch_primary(&transport_api(endpoint), &transport, &repo, None)
            .unwrap()
            .0;
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), package);
        assert_eq!(
            std::fs::read_to_string(
                transport
                    .root()
                    .join("repos")
                    .join(transport.scope(&repo.folder_name()))
                    .join(repo.folder_name())
                    .join("refs/gb202")
            )
            .unwrap(),
            COMMIT
        );
    }

    #[test]
    fn online_cache_hit_resolves_revision_without_redownloading_weights() {
        let temp = tempfile::tempdir().unwrap();
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let directory = transport.snapshot_dir(&repo.folder_name(), COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), fixture(&[])).unwrap();
        let routes = HubRoutes::default()
            .document("/api/models/owner/model/revision/gb10", metadata_response());
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let path = fetch_primary(&transport_api(endpoint), &transport, &repo, None)
            .unwrap()
            .0;
        worker.join().unwrap();
        assert_eq!(std::fs::read(path).unwrap(), fixture(&[]));
        assert!(log.lock().unwrap().files.is_empty());
    }

    #[test]
    fn companions_download_from_the_primary_commit() {
        let temp = tempfile::tempdir().unwrap();
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let name = "model-data-01-of-01.safetensors";
        let directory = transport.snapshot_dir(&repo.folder_name(), COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), fixture(&[name])).unwrap();
        let body = b"data".to_vec();
        let etag = sha256_hex(&body);
        let routes = HubRoutes::default()
            .document(
                "/api/models/owner/model/revision/gb202",
                metadata_response(),
            )
            .file(
                format!("/owner/model/resolve/{COMMIT}/{name}"),
                MockFile::new(&etag, body.clone()),
            );
        let (endpoint, worker) = mock_hub(routes);
        let api = transport_api(endpoint);
        let (path, pinned) = fetch_primary(&api, &transport, &repo, None).unwrap();
        fetch_dependency(&api, &transport, &pinned, &path, name).unwrap();
        worker.join().unwrap();
        assert_eq!(
            std::fs::read(path.parent().unwrap().join(name)).unwrap(),
            body
        );
    }

    #[test]
    fn cached_primary_blob_is_linked_without_downloading_its_body() {
        let temp = tempfile::tempdir().unwrap();
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let package = fixture(&[]);
        let etag = sha256_hex(&package);
        let key = blob_key(&etag).unwrap();
        let folder = repo.folder_name();
        let blob = transport.blob(&folder, &key);
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, &package).unwrap();
        let routes = HubRoutes::default()
            .document("/api/models/owner/model/revision/gb10", metadata_response())
            .file(
                format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
                MockFile::new(&etag, package.clone()),
            );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let path = fetch_primary(&transport_api(endpoint), &transport, &repo, None)
            .unwrap()
            .0;
        worker.join().unwrap();
        assert_eq!(
            path,
            transport
                .snapshot_dir(&folder, COMMIT)
                .join("model.safetensors")
        );
        assert_eq!(std::fs::read(&path).unwrap(), package);
        assert!(
            log.lock().unwrap().files.is_empty(),
            "a published blob must not be transferred again"
        );
        #[cfg(unix)]
        assert_same_file(&blob, &path);
    }

    #[test]
    fn revisions_share_downloaded_payload_without_replacing_the_blob() {
        let temp = tempfile::tempdir().unwrap();
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        let other_commit = "abcdef0123456789012345678901234567890123";
        let first_repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let second_repo =
            Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let name = format!("{}.safetensors", "a".repeat(64));
        for commit in [COMMIT, other_commit] {
            let directory = transport.snapshot_dir(&first_repo.folder_name(), commit);
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join("model.safetensors"), fixture(&[&name])).unwrap();
        }
        let body = b"data".to_vec();
        let etag = sha256_hex(&body);
        let routes = HubRoutes::default()
            .document("/api/models/owner/model/revision/gb10", metadata_response())
            .document(
                "/api/models/owner/model/revision/gb202",
                revision_response(other_commit),
            )
            .file(
                format!("/owner/model/resolve/{COMMIT}/{name}"),
                MockFile::new(&etag, body.clone()),
            )
            .file(
                format!("/owner/model/resolve/{other_commit}/{name}"),
                MockFile::new(&etag, body.clone()).at_commit(other_commit),
            );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let api = transport_api(endpoint);
        let (first, pinned) = fetch_primary(&api, &transport, &first_repo, None).unwrap();
        fetch_dependency(&api, &transport, &pinned, &first, &name).unwrap();
        let transfers = log.lock().unwrap().files.len();
        let (second, pinned) = fetch_primary(&api, &transport, &second_repo, None).unwrap();
        fetch_dependency(&api, &transport, &pinned, &second, &name).unwrap();
        worker.join().unwrap();
        assert_eq!(
            second,
            transport
                .snapshot_dir(&second_repo.folder_name(), other_commit)
                .join("model.safetensors")
        );
        assert_eq!(log.lock().unwrap().files.len(), transfers);
        assert_eq!(transfers, 1, "the shared payload is transferred once");
        let blob = transport.blob(&first_repo.folder_name(), &blob_key(&etag).unwrap());
        assert_eq!(std::fs::read(&blob).unwrap(), body);
        #[cfg(unix)]
        {
            assert_same_file(&blob, &first.parent().unwrap().join(&name));
            assert_same_file(&blob, &second.parent().unwrap().join(&name));
        }
    }

    #[test]
    fn concurrent_fetches_transfer_the_payload_once() {
        let temp = tempfile::tempdir().unwrap();
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
        let package = fixture(&[]);
        let etag = sha256_hex(&package);
        let key = blob_key(&etag).unwrap();
        let folder = repo.folder_name();
        let blob = transport.blob(&folder, &key);
        let lock = content_lock(&transport.lock(&folder, &key)).unwrap();
        let routes = HubRoutes::default().file(
            format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
            MockFile::new(&etag, package.clone()),
        );
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let clients: Vec<_> = (0..2)
            .map(|_| {
                let repo = repo.clone();
                let endpoint = endpoint.clone();
                let api = transport_api(endpoint.clone());
                let folder = folder.clone();
                let key = key.clone();
                let root = temp.path().to_owned();
                thread::spawn(move || {
                    let transport = Transport::open(&root, None, "http://mock").unwrap();
                    assert_eq!(transport.part(&folder, &key).parent().is_some(), true);
                    fetch_file(&api, &transport, &repo, "model.safetensors")
                        .map_err(|error| error.to_string())
                })
            })
            .collect();
        // Both callers fetched the file identity before either could publish the
        // blob; the waiting caller must reuse the completed transfer.
        std::thread::sleep(Duration::from_millis(200));
        drop(lock);
        for client in clients {
            let path = client.join().unwrap().unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), package);
            #[cfg(unix)]
            assert_same_file(&blob, &path);
        }
        worker.join().unwrap();
        assert_eq!(log.lock().unwrap().files.len(), 1);
    }

    #[test]
    fn truncated_cached_blobs_are_rejected_without_modifying_them() {
        let temp = tempfile::tempdir().unwrap();
        let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
        let etag = "fixture";
        let key = blob_key(etag).unwrap();
        let folder = repo.folder_name();
        let blob = transport.blob(&folder, &key);
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, b"truncated").unwrap();
        let routes = HubRoutes::default().file(
            format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
            MockFile::new(etag, vec![0u8; 100]),
        );
        let (endpoint, worker, _log) = mock_hub_counted(routes);
        let error = fetch_file(
            &transport_api(endpoint),
            &transport,
            &repo,
            "model.safetensors",
        )
        .unwrap_err();
        worker.join().unwrap();
        assert!(
            error
                .to_string()
                .contains("invalid cached Hugging Face blob")
        );
        assert_eq!(std::fs::read(blob).unwrap(), b"truncated");
        assert!(!transport.snapshot_dir(&folder, COMMIT).exists());
    }

    #[test]
    fn cache_lookup_rejects_escaping_etags_and_unpinned_commits() {
        for (commit, etag) in [
            (COMMIT, "../outside"),
            (COMMIT, ""),
            ("other-commit", "fixture"),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let transport = Transport::open(temp.path(), None, "http://mock").unwrap();
            let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
            let mut file = MockFile::new(etag, vec![0u8; 100]);
            file.commit = commit.to_owned();
            let routes = HubRoutes::default().file(
                format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
                file,
            );
            let (endpoint, worker, _log) = mock_hub_counted(routes);
            let error = fetch_file(
                &transport_api(endpoint),
                &transport,
                &repo,
                "model.safetensors",
            )
            .unwrap_err();
            worker.join().unwrap();
            assert!(
                error.to_string().contains("invalid file identity"),
                "{error}"
            );
            assert!(!transport.snapshot_dir(&repo.folder_name(), COMMIT).exists());
        }
    }
}
