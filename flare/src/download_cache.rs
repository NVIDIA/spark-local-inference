/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inspection and reclamation for the Hugging Face download transport cache.
//!
//! `flare download` fetches prepared cores through one shared tree per download
//! parent. The root lives at `<parent>/transport/hub/v1` and contains a version
//! marker, blobs scoped by endpoint and repository, per-content transfer locks,
//! resumable partials, quarantined suspect bytes, and repository metadata whose
//! snapshot entries are hardlinks into the scoped blob store. Older Flare
//! releases created Hugging Face trees directly in `--cache-dir` and under the
//! model cache's `staging/trees`; those remain recognizable and recoverable.
//!
//! Inspection is read-only. Reclamation removes only data whose transfer lock is
//! free, preserves the transport root and its marker, and never removes lock
//! inodes.

#[cfg(unix)]
use std::fs::OpenOptions;
use std::{
    collections::HashSet,
    fs::{self, File},
    io,
    path::{Path, PathBuf},
};

use serde::Serialize;

use flare_helios::core::cache::is_legacy_hub_transport;

const VERSION: &[u8] = b"flare-hub-transport 1\n";
const LEGACY_PREFIX: &str = ".flare-hub-staging-";
const ROOT: [&str; 3] = ["transport", "hub", "v1"];

type Failure = Box<dyn std::error::Error>;

/// Shared transport root below one download parent.
pub fn root(parent: &Path) -> PathBuf {
    ROOT.iter()
        .fold(parent.to_path_buf(), |path, name| path.join(name))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Marker {
    Missing,
    Valid,
    Foreign,
}

impl Marker {
    fn read(root: &Path) -> io::Result<Self> {
        match fs::read(root.join("version")) {
            Ok(bytes) if bytes == VERSION => Ok(Self::Valid),
            Ok(_) => Ok(Self::Foreign),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Self::Missing),
            Err(error) => Err(error),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Valid => "valid",
            Self::Foreign => "foreign",
        }
    }
}

/// Read-only description of one download parent's transport cache.
#[derive(Debug, Serialize)]
pub struct Inspection {
    pub parent: String,
    pub transport: TransportInspection,
    pub legacy: LegacyInspection,
}

#[derive(Debug, Serialize)]
pub struct TransportInspection {
    pub root: String,
    pub present: bool,
    pub marker: String,
    /// A shared acquisition lease or an exclusive reclamation lease is held.
    pub acquisition_active: bool,
    /// Totals across every scope below `blobs/`.
    pub blobs: BlobInspection,
    pub scopes: Vec<ScopeInspection>,
    pub repositories: Vec<RepositoryInspection>,
    /// Apparent bytes of unique inodes below the root, including lock inodes.
    pub retained_bytes: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct BlobInspection {
    pub complete_files: usize,
    pub complete_bytes: u64,
    pub partial_files: usize,
    pub partial_bytes: u64,
    pub quarantined_files: usize,
    pub quarantined_bytes: u64,
    pub adopted_files: usize,
    pub adopted_bytes: u64,
    pub lock_files: usize,
    pub unknown_files: usize,
    pub active_files: usize,
    pub active_bytes: u64,
}

impl BlobInspection {
    fn add(&mut self, other: &Self) {
        self.complete_files += other.complete_files;
        self.complete_bytes += other.complete_bytes;
        self.partial_files += other.partial_files;
        self.partial_bytes += other.partial_bytes;
        self.quarantined_files += other.quarantined_files;
        self.quarantined_bytes += other.quarantined_bytes;
        self.adopted_files += other.adopted_files;
        self.adopted_bytes += other.adopted_bytes;
        self.lock_files += other.lock_files;
        self.unknown_files += other.unknown_files;
        self.active_files += other.active_files;
        self.active_bytes += other.active_bytes;
    }
}

#[derive(Debug, Serialize)]
pub struct ScopeInspection {
    pub scope: String,
    pub blobs: BlobInspection,
    /// Apparent bytes of unique inodes in this scope.
    pub retained_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct RepositoryInspection {
    pub scope: String,
    pub folder: String,
    pub refs: Vec<RefInspection>,
    pub snapshots: Vec<SnapshotInspection>,
}

#[derive(Debug, Serialize)]
pub struct RefInspection {
    pub revision: String,
    pub commit: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SnapshotInspection {
    pub commit: String,
    pub files: usize,
    /// Apparent bytes; snapshot entries are hardlinks into the blob store.
    pub bytes: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct LegacyInspection {
    pub trees: Vec<LegacyTreeInspection>,
    pub complete_bytes: u64,
    pub partial_bytes: u64,
    pub active_trees: usize,
}

#[derive(Debug, Serialize)]
pub struct LegacyTreeInspection {
    pub path: String,
    pub kind: String,
    pub active: bool,
    pub complete_files: usize,
    pub complete_bytes: u64,
    pub partial_files: usize,
    pub partial_bytes: u64,
    pub unknown_files: usize,
}

/// Reclamation totals and the bytes retained in each surviving scope.
#[derive(Debug, Default, Serialize)]
pub struct Reclaimed {
    pub present: bool,
    pub marker: String,
    /// An acquisition lease was held, so new-root reclamation was skipped.
    pub acquisition_active: bool,
    pub repository_refs: usize,
    pub snapshot_files: usize,
    pub complete_files: usize,
    pub complete_bytes: u64,
    pub partial_files: usize,
    pub partial_bytes: u64,
    pub quarantined_files: usize,
    pub quarantined_bytes: u64,
    pub adopted_files: usize,
    pub adopted_bytes: u64,
    /// Blobs, partials, or transient files skipped because their lock is held.
    pub active_files: usize,
    pub active_bytes: u64,
    pub retained: Vec<RetainedScope>,
    pub legacy_trees: usize,
    pub legacy_bytes: u64,
    pub legacy_active: usize,
    /// Legacy trees preserved because they contain files this command does not recognize.
    pub legacy_unknown_trees: usize,
    /// Explicit legacy directories preserved because they have no operation lease.
    pub legacy_unverified: usize,
    /// Apparent bytes of unique inodes that remain below the transport root.
    pub retained_bytes: u64,
}

/// Transport copies reclaimed after one successful installation.
#[derive(Debug, Default, Serialize)]
pub struct Consumed {
    /// An acquisition lease was held, so this commit was deferred.
    pub acquisition_active: bool,
    /// The commit was recorded for reclamation by a later successful cleanup.
    pub deferred: bool,
    pub snapshot_files: usize,
    pub complete_files: usize,
    pub complete_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct RetainedScope {
    pub scope: String,
    pub retained_bytes: u64,
    pub active_files: usize,
    pub active_bytes: u64,
}

/// Inspect the transport cache below `parent` and legacy Hugging Face trees below
/// `model_root` and `parent`.
pub fn inspect(parent: &Path, model_root: &Path) -> Result<Inspection, Failure> {
    Ok(Inspection {
        parent: parent.display().to_string(),
        transport: inspect_transport(parent)?,
        legacy: inspect_legacy(parent, model_root)?,
    })
}

fn acquisition_lease(root: &Path) -> PathBuf {
    root.join("acquisition.lock")
}

fn inspect_transport(parent: &Path) -> Result<TransportInspection, Failure> {
    let root = root(parent);
    let present = root.is_dir();
    let marker = if present {
        Marker::read(&root)?
    } else {
        Marker::Missing
    };
    let acquisition_active = present
        && matches!(
            lock_free(&acquisition_lease(&root), false)?,
            LockState::Held
        );
    let (blobs, scopes) = inspect_scopes(&root)?;
    let repositories = inspect_repositories(&root)?;
    let retained_bytes = if present { unique_bytes(&root)? } else { 0 };
    Ok(TransportInspection {
        root: root.display().to_string(),
        present,
        marker: marker.name().to_owned(),
        acquisition_active,
        blobs,
        scopes,
        repositories,
        retained_bytes,
    })
}

/// Content-addressed scopes are keyed by endpoint and repository.
fn valid_scope(name: &str) -> bool {
    name.len() == 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn scopes(root: &Path) -> Result<Vec<(String, PathBuf)>, Failure> {
    let directory = root.join("blobs");
    let mut scopes = Vec::new();
    if !directory.is_dir() {
        return Ok(scopes);
    }
    for entry in fs::read_dir(&directory)? {
        let entry = entry?;
        if !entry_metadata(&entry)?.is_some_and(|metadata| metadata.is_dir()) {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if valid_scope(&name) {
            scopes.push((name, entry.path()));
        }
    }
    scopes.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(scopes)
}

fn inspect_scopes(root: &Path) -> Result<(BlobInspection, Vec<ScopeInspection>), Failure> {
    let mut totals = BlobInspection::default();
    let mut inspected = Vec::new();
    for (scope, directory) in scopes(root)? {
        let blobs = inspect_scope(&directory)?;
        totals.add(&blobs);
        inspected.push(ScopeInspection {
            scope,
            retained_bytes: unique_bytes(&directory)?,
            blobs,
        });
    }
    Ok((totals, inspected))
}

fn inspect_scope(directory: &Path) -> Result<BlobInspection, Failure> {
    let mut blobs = BlobInspection::default();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let Some(metadata) = entry_metadata(&entry)? else {
            continue;
        };
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            blobs.unknown_files += 1;
            continue;
        };
        if name.ends_with(".lock") {
            blobs.lock_files += 1;
            continue;
        }
        let (key, kind) = match quarantine_key(name) {
            Some(key) => (key, TransferKind::Quarantined),
            None => match adopted_key(name) {
                Some(key) => (key, TransferKind::Adopted),
                None => match name.strip_suffix(".part") {
                    Some(key) => (key, TransferKind::Partial),
                    None => (name, TransferKind::Complete),
                },
            },
        };
        if !valid_key(key) {
            blobs.unknown_files += 1;
            continue;
        }
        let active = matches!(
            lock_free(&directory.join(format!("{key}.lock")), false)?,
            LockState::Held
        );
        match kind {
            TransferKind::Complete => {
                blobs.complete_files += 1;
                blobs.complete_bytes += metadata.len();
            }
            TransferKind::Partial => {
                blobs.partial_files += 1;
                blobs.partial_bytes += metadata.len();
            }
            TransferKind::Quarantined => {
                blobs.quarantined_files += 1;
                blobs.quarantined_bytes += metadata.len();
            }
            TransferKind::Adopted => {
                blobs.adopted_files += 1;
                blobs.adopted_bytes += metadata.len();
            }
        }
        if active {
            blobs.active_files += 1;
            blobs.active_bytes += metadata.len();
        }
    }
    Ok(blobs)
}

fn inspect_repositories(root: &Path) -> Result<Vec<RepositoryInspection>, Failure> {
    let directory = root.join("repos");
    let mut repositories = Vec::new();
    if !directory.is_dir() {
        return Ok(repositories);
    }
    for entry in fs::read_dir(&directory)? {
        let entry = entry?;
        if !entry_metadata(&entry)?.is_some_and(|metadata| metadata.is_dir()) {
            continue;
        }
        let Some(scope) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !valid_scope(&scope) {
            continue;
        }
        for folder in fs::read_dir(entry.path())? {
            let folder = folder?;
            if !entry_metadata(&folder)?.is_some_and(|metadata| metadata.is_dir()) {
                continue;
            }
            let Some(name) = folder.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let mut refs = Vec::new();
            for (revision, path) in files_with_prefix(&folder.path().join("refs"))?.0 {
                let commit = fs::read_to_string(&path)
                    .ok()
                    .map(|text| text.trim().to_owned())
                    .filter(|text| !text.is_empty() && text.len() <= 128);
                refs.push(RefInspection { revision, commit });
            }
            refs.sort_by(|left, right| left.revision.cmp(&right.revision));
            let mut snapshots = Vec::new();
            let snapshot_root = folder.path().join("snapshots");
            if snapshot_root.is_dir() {
                for entry in fs::read_dir(&snapshot_root)? {
                    let entry = entry?;
                    if !entry_metadata(&entry)?.is_some_and(|metadata| metadata.is_dir()) {
                        continue;
                    }
                    let Some(commit) = entry.file_name().to_str().map(str::to_owned) else {
                        continue;
                    };
                    let (files, bytes) = file_usage(&entry.path())?;
                    snapshots.push(SnapshotInspection {
                        commit,
                        files,
                        bytes,
                    });
                }
            }
            snapshots.sort_by(|left, right| left.commit.cmp(&right.commit));
            repositories.push(RepositoryInspection {
                scope: scope.clone(),
                folder: name,
                refs,
                snapshots,
            });
        }
    }
    repositories
        .sort_by(|left, right| (&left.scope, &left.folder).cmp(&(&right.scope, &right.folder)));
    Ok(repositories)
}

fn inspect_legacy(parent: &Path, model_root: &Path) -> Result<LegacyInspection, Failure> {
    let mut legacy = LegacyInspection::default();
    for tree in legacy_trees(parent, model_root)? {
        let active = tree.active()?;
        let usage = tree.usage()?;
        legacy.complete_bytes += usage.complete_bytes;
        legacy.partial_bytes += usage.partial_bytes;
        legacy.active_trees += usize::from(active);
        legacy.trees.push(LegacyTreeInspection {
            path: tree.path.display().to_string(),
            kind: tree.kind.name().to_owned(),
            active,
            complete_files: usage.complete_files,
            complete_bytes: usage.complete_bytes,
            partial_files: usage.partial_files,
            partial_bytes: usage.partial_bytes,
            unknown_files: usage.unknown_files,
        });
    }
    Ok(legacy)
}

/// Reclaim the download transport cache below `parent` and abandoned legacy
/// Hugging Face trees. Active transfers and unknown files are preserved.
pub fn reclaim(parent: &Path, model_root: &Path) -> Result<Reclaimed, Failure> {
    let root = root(parent);
    let mut reclaimed = Reclaimed::default();
    if !root.is_dir() {
        reclaimed.marker = Marker::Missing.name().to_owned();
        let legacy = reclaim_legacy(parent, model_root)?;
        reclaimed.legacy_trees = legacy.legacy_trees;
        reclaimed.legacy_bytes = legacy.legacy_bytes;
        reclaimed.legacy_active = legacy.legacy_active;
        reclaimed.legacy_unknown_trees = legacy.legacy_unknown_trees;
        reclaimed.legacy_unverified = legacy.legacy_unverified;
        return Ok(reclaimed);
    }
    let marker = Marker::read(&root)?;
    reclaimed.present = true;
    reclaimed.marker = marker.name().to_owned();
    // New-root reclamation excludes every acquisition. The lease is held from
    // before repository metadata is removed until the retained-byte pass ends.
    let mut _lease = None;
    if marker == Marker::Valid {
        match lock_free(&acquisition_lease(&root), true)? {
            LockState::Held => reclaimed.acquisition_active = true,
            LockState::Free(lease) => {
                _lease = lease;
                reclaim_repositories(&root, &mut reclaimed)?;
                reclaim_blobs(&root, &mut reclaimed)?;
            }
        }
    }
    reclaimed.retained_bytes = unique_bytes(&root)?;
    drop(_lease);
    let legacy = reclaim_legacy(parent, model_root)?;
    reclaimed.legacy_trees = legacy.legacy_trees;
    reclaimed.legacy_bytes = legacy.legacy_bytes;
    reclaimed.legacy_active = legacy.legacy_active;
    reclaimed.legacy_unknown_trees = legacy.legacy_unknown_trees;
    reclaimed.legacy_unverified = legacy.legacy_unverified;
    Ok(reclaimed)
}

/// Reclaim one successfully installed acquisition's transport copies.
///
/// The acquisition must have released its shared lease first. This pass takes
/// the exclusive lease so it cannot race an active acquisition. When another
/// acquisition holds the lease, the commit is recorded as installed and the
/// next successful cleanup processes it together with its own commit. Only
/// recorded commits' snapshot entries and complete blobs whose last hardlink
/// they held are removed, so failed partials, other commits, quarantined bytes,
/// unknown files, and lock inodes remain.
pub fn reclaim_installed(
    root: &Path,
    scope: &str,
    folder: &str,
    commit: &str,
) -> Result<Consumed, Failure> {
    let mut consumed = Consumed::default();
    if !root.is_dir() || Marker::read(root)? != Marker::Valid {
        return Ok(consumed);
    }
    if !valid_scope(scope) || !valid_component(folder) || !valid_component(commit) {
        return Err("installed transport cleanup names an invalid scope, folder, or commit".into());
    }
    let LockState::Free(_lease) = lock_free(&acquisition_lease(root), true)? else {
        consumed.acquisition_active = true;
        consumed.deferred = record_installed(root, scope, folder, commit)?;
        return Ok(consumed);
    };
    let mut commits = recorded_installs(root, scope, folder)?;
    if !commits.iter().any(|recorded| recorded == commit) {
        commits.push(commit.to_owned());
    }
    for commit in commits {
        reclaim_installed_commit(root, scope, folder, &commit, &mut consumed)?;
        clear_recorded_install(root, scope, folder, &commit)?;
    }
    Ok(consumed)
}

fn installed_directory(root: &Path, scope: &str, folder: &str) -> PathBuf {
    root.join("repos")
        .join(scope)
        .join(folder)
        .join("installed")
}

fn record_installed(root: &Path, scope: &str, folder: &str, commit: &str) -> Result<bool, Failure> {
    let directory = installed_directory(root, scope, folder);
    fs::create_dir_all(&directory)?;
    fs::write(directory.join(commit), commit.as_bytes())?;
    Ok(true)
}

fn recorded_installs(root: &Path, scope: &str, folder: &str) -> Result<Vec<String>, Failure> {
    let directory = installed_directory(root, scope, folder);
    let mut commits = Vec::new();
    for entry in match fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(commits),
        Err(error) => return Err(error.into()),
    } {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let Some(commit) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if valid_component(&commit) {
            commits.push(commit);
        }
    }
    commits.sort();
    Ok(commits)
}

fn clear_recorded_install(
    root: &Path,
    scope: &str,
    folder: &str,
    commit: &str,
) -> Result<(), Failure> {
    let path = installed_directory(root, scope, folder).join(commit);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn reclaim_installed_commit(
    root: &Path,
    scope: &str,
    folder: &str,
    commit: &str,
    consumed: &mut Consumed,
) -> Result<(), Failure> {
    let snapshots = root
        .join("repos")
        .join(scope)
        .join(folder)
        .join("snapshots")
        .join(commit);
    if !snapshots.is_dir() {
        return Ok(());
    }
    let mut installed = HashSet::new();
    snapshot_inodes(&snapshots, &mut installed)?;
    consumed.snapshot_files += count_files(&snapshots)?;
    match fs::remove_dir_all(&snapshots) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let blobs = root.join("blobs").join(scope);
    if !blobs.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(&blobs)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let Some(metadata) = entry_metadata(&entry)? else {
            continue;
        };
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.ends_with(".lock")
            || name.ends_with(".part")
            || quarantine_key(name).is_some()
            || !valid_key(name)
        {
            continue;
        }
        if !installed.contains(&identity(&metadata, &entry.path()))
            || links(&metadata, &entry.path()) > 1
        {
            continue;
        }
        let lock = blobs.join(format!("{name}.lock"));
        let LockState::Free(_holding) = lock_free(&lock, true)? else {
            continue;
        };
        match fs::remove_file(entry.path()) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        }
        consumed.complete_files += 1;
        consumed.complete_bytes += metadata.len();
    }
    Ok(())
}

fn valid_component(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.bytes().any(|byte| matches!(byte, b'/' | b'\\' | 0))
}

fn snapshot_inodes(path: &Path, identities: &mut HashSet<(u64, u64)>) -> Result<(), Failure> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            snapshot_inodes(&entry.path(), identities)?;
        } else if file_type.is_file() {
            let metadata = entry.metadata()?;
            identities.insert(identity(&metadata, &entry.path()));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn links(metadata: &fs::Metadata, _path: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    metadata.nlink()
}

#[cfg(windows)]
fn links(_metadata: &fs::Metadata, path: &Path) -> u64 {
    windows_file_identity(path)
        .map(|(_, links)| links)
        .unwrap_or(u64::MAX)
}

fn reclaim_repositories(root: &Path, reclaimed: &mut Reclaimed) -> Result<(), Failure> {
    let directory = root.join("repos");
    if !directory.is_dir() {
        return Ok(());
    }
    for scope in fs::read_dir(&directory)? {
        let scope = scope?;
        if !entry_metadata(&scope)?.is_some_and(|metadata| metadata.is_dir()) {
            continue;
        }
        let Some(name) = scope.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !valid_scope(&name) {
            continue;
        }
        for folder in fs::read_dir(scope.path())? {
            let folder = folder?;
            if !entry_metadata(&folder)?.is_some_and(|metadata| metadata.is_dir()) {
                continue;
            }
            let path = folder.path();
            reclaimed.repository_refs += count_files(&path.join("refs"))?;
            reclaimed.snapshot_files += count_files(&path.join("snapshots"))?;
            for name in ["refs", "snapshots", "installed"] {
                match fs::remove_dir_all(path.join(name)) {
                    Ok(()) => {}
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
        }
    }
    Ok(())
}

fn reclaim_blobs(root: &Path, reclaimed: &mut Reclaimed) -> Result<(), Failure> {
    for (scope, directory) in scopes(root)? {
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let Some(metadata) = entry_metadata(&entry)? else {
                continue;
            };
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if name.ends_with(".lock") {
                continue;
            }
            let (key, kind) = match quarantine_key(name) {
                Some(key) => (key, TransferKind::Quarantined),
                None => match adopted_key(name) {
                    Some(key) => (key, TransferKind::Adopted),
                    None => match name.strip_suffix(".part") {
                        Some(key) => (key, TransferKind::Partial),
                        None => (name, TransferKind::Complete),
                    },
                },
            };
            if !valid_key(key) {
                continue;
            }
            let lock = directory.join(format!("{key}.lock"));
            let LockState::Free(_holding) = lock_free(&lock, true)? else {
                reclaimed.active_files += 1;
                reclaimed.active_bytes += metadata.len();
                continue;
            };
            match fs::remove_file(entry.path()) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.into()),
            }
            match kind {
                TransferKind::Complete => {
                    reclaimed.complete_files += 1;
                    reclaimed.complete_bytes += metadata.len();
                }
                TransferKind::Partial => {
                    reclaimed.partial_files += 1;
                    reclaimed.partial_bytes += metadata.len();
                }
                TransferKind::Quarantined => {
                    reclaimed.quarantined_files += 1;
                    reclaimed.quarantined_bytes += metadata.len();
                }
                TransferKind::Adopted => {
                    reclaimed.adopted_files += 1;
                    reclaimed.adopted_bytes += metadata.len();
                }
            }
        }
        let inspection = inspect_scope(&directory)?;
        reclaimed.retained.push(RetainedScope {
            scope,
            retained_bytes: unique_bytes(&directory)?,
            active_files: inspection.active_files,
            active_bytes: inspection.active_bytes,
        });
    }
    Ok(())
}

fn reclaim_legacy(parent: &Path, model_root: &Path) -> Result<Reclaimed, Failure> {
    let mut reclaimed = Reclaimed::default();
    for tree in legacy_trees(parent, model_root)? {
        let access = tree.acquire()?;
        let guards = match access {
            LegacyAccess::Active => {
                reclaimed.legacy_active += 1;
                continue;
            }
            LegacyAccess::Unverified => {
                reclaimed.legacy_unverified += 1;
                continue;
            }
            LegacyAccess::Gone => continue,
            LegacyAccess::Held { _lease, _locks } => (_lease, _locks),
        };
        let usage = tree.usage()?;
        if usage.unknown_files > 0 {
            reclaimed.legacy_unknown_trees += 1;
            continue;
        }
        match fs::remove_dir_all(&tree.path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        }
        reclaimed.legacy_bytes += usage.bytes;
        reclaimed.legacy_trees += 1;
        drop(guards);
    }
    Ok(reclaimed)
}

#[derive(Clone, Copy)]
enum TreeKind {
    CacheStaging,
    ExplicitStaging,
}

impl TreeKind {
    fn name(self) -> &'static str {
        match self {
            Self::CacheStaging => "cache_staging",
            Self::ExplicitStaging => "explicit_staging",
        }
    }
}

struct LegacyTree {
    path: PathBuf,
    kind: TreeKind,
    lock: Option<PathBuf>,
}

/// Liveness result for one legacy tree that reclamation can act on.
enum LegacyAccess {
    /// Every applicable lock is held by this pass; the tree can be removed before
    /// the guards are dropped.
    Held { _lease: File, _locks: Vec<File> },
    /// A staging lease or transfer lock is held by another process.
    Active,
    /// Safe inactivity cannot be established, so the tree is preserved.
    Unverified,
    /// The tree disappeared before this pass acquired its lease.
    Gone,
}

impl LegacyTree {
    /// Read-only liveness probe used by inspection.
    fn active(&self) -> Result<bool, Failure> {
        if let Some(lock) = &self.lock
            && matches!(lock_free(lock, false)?, LockState::Held)
        {
            return Ok(true);
        }
        for lock in lock_files(&self.path)? {
            if matches!(lock_free(&lock, false)?, LockState::Held) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Acquire every lock that establishes this tree's inactivity and keep them
    /// held until the caller finishes deleting it.
    ///
    /// A cache staging tree has an operation lease that no new writer can take
    /// while it is held. An explicit legacy directory has only per-file locks,
    /// and an old writer can create a new lock at any time, so its inactivity
    /// cannot be established and it is never reclaimed.
    fn acquire(&self) -> Result<LegacyAccess, Failure> {
        if !self.path.is_dir() {
            return Ok(LegacyAccess::Gone);
        }
        let Some(lease) = &self.lock else {
            return Ok(LegacyAccess::Unverified);
        };
        // The lease inode is stable and exists for the whole staging transaction.
        // Creating a missing one establishes inactivity against any future writer
        // that would use the same transaction token.
        let lease = match lock_free(lease, true)? {
            LockState::Free(Some(lease)) => lease,
            LockState::Free(None) => return Ok(LegacyAccess::Unverified),
            LockState::Held => return Ok(LegacyAccess::Active),
        };
        let mut locks = Vec::new();
        let mut held = std::collections::HashSet::new();
        for lock in lock_files(&self.path)? {
            if !held.insert(lock.clone()) {
                continue;
            }
            match lock_free(&lock, false)? {
                LockState::Free(Some(file)) => locks.push(file),
                LockState::Free(None) => {}
                LockState::Held => return Ok(LegacyAccess::Active),
            }
        }
        // The lease prevents another staging transaction from starting, but a
        // lock file that appeared while this pass was scanning is still probed.
        for lock in lock_files(&self.path)? {
            if !held.insert(lock.clone()) {
                continue;
            }
            match lock_free(&lock, false)? {
                LockState::Free(Some(file)) => locks.push(file),
                LockState::Free(None) => {}
                LockState::Held => return Ok(LegacyAccess::Active),
            }
        }
        Ok(LegacyAccess::Held {
            _lease: lease,
            _locks: locks,
        })
    }

    fn usage(&self) -> Result<TreeUsage, Failure> {
        legacy_usage(&self.path)
    }
}

fn legacy_trees(parent: &Path, model_root: &Path) -> Result<Vec<LegacyTree>, Failure> {
    let mut trees = Vec::new();
    if parent.is_dir() {
        for entry in fs::read_dir(parent)? {
            let entry = entry?;
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if name.starts_with(LEGACY_PREFIX)
                && entry_metadata(&entry)?.is_some_and(|metadata| metadata.is_dir())
            {
                trees.push(LegacyTree {
                    path: entry.path(),
                    kind: TreeKind::ExplicitStaging,
                    lock: None,
                });
            }
        }
    }
    let staging = model_root.join("staging/trees");
    if staging.is_dir() {
        for entry in fs::read_dir(&staging)? {
            let entry = entry?;
            if !entry_metadata(&entry)?.is_some_and(|metadata| metadata.is_dir())
                || !is_legacy_hub_transport(&entry.path())
            {
                continue;
            }
            let Some(token) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            trees.push(LegacyTree {
                path: entry.path(),
                kind: TreeKind::CacheStaging,
                lock: Some(
                    model_root
                        .join("locks/staging")
                        .join(format!("{token}.lock")),
                ),
            });
        }
    }
    trees.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(trees)
}

enum LockState {
    /// No transfer holds the lock. The file keeps the exclusive lock while it
    /// stays alive, so reclamation holds it until the guarded file is removed.
    Free(Option<File>),
    Held,
}

/// Probe the exclusive transfer lock at `path`.
///
/// Inspection passes `create = false` and treats a missing lock inode as free.
/// Reclamation passes `create = true` and keeps the returned lock until the
/// guarded file is gone.
fn lock_free(path: &Path, create: bool) -> Result<LockState, Failure> {
    let file = match open_lock(path, create) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(LockState::Free(None));
        }
        Err(error) => return Err(error.into()),
    };
    if try_exclusive(&file)? {
        Ok(LockState::Free(Some(file)))
    } else {
        Ok(LockState::Held)
    }
}

#[cfg(unix)]
fn open_lock(path: &Path, create: bool) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    if create {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)
    } else {
        File::open(path)
    }
}

#[cfg(not(unix))]
fn open_lock(path: &Path, create: bool) -> io::Result<File> {
    if create {
        crate::windows::private_file(path, false)
    } else {
        File::open(path)
    }
}

#[cfg(unix)]
fn try_exclusive(file: &File) -> io::Result<bool> {
    use std::os::fd::AsRawFd;
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::WouldBlock {
        Ok(false)
    } else {
        Err(error)
    }
}

#[cfg(not(unix))]
fn try_exclusive(file: &File) -> io::Result<bool> {
    crate::windows::try_lock(file)
}

/// Metadata for a directory entry that another pass may have removed.
fn entry_metadata(entry: &fs::DirEntry) -> Result<Option<fs::Metadata>, Failure> {
    match entry.metadata() {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[derive(Clone, Copy)]
enum TransferKind {
    Complete,
    Partial,
    Quarantined,
    Adopted,
}

fn transient_key<'a>(name: &'a str, marker: &str) -> Option<&'a str> {
    let rest = name.strip_prefix('.')?;
    let (key, suffix) = rest.split_once(marker)?;
    let (process, count) = suffix.split_once('-')?;
    if key.is_empty()
        || !valid_key(key)
        || process.is_empty()
        || count.is_empty()
        || !process.bytes().all(|byte| byte.is_ascii_digit())
        || !count.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    Some(key)
}

/// Quarantined suspect bytes are Flare-owned files named
/// `.<key>.part.corrupt-<pid>-<n>`.
fn quarantine_key(name: &str) -> Option<&str> {
    transient_key(name, ".part.corrupt-")
}

/// A transient adoption copy is named `.<key>.adopt-<pid>-<n>` and exists only
/// while `<key>.lock` is held; a crash can leave one behind.
fn adopted_key(name: &str) -> Option<&str> {
    transient_key(name, ".adopt-")
}

fn valid_key(key: &str) -> bool {
    !key.is_empty()
        && key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

fn lock_files(tree: &Path) -> Result<Vec<PathBuf>, Failure> {
    let mut locks = Vec::new();
    let mut stack = vec![tree.to_path_buf()];
    while let Some(directory) = stack.pop() {
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        for entry in entries {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                stack.push(entry.path());
            } else if file_type.is_file() {
                let name = entry.file_name();
                let Some(name) = name.to_str() else {
                    continue;
                };
                if name.ends_with(".lock") || name.ends_with(".flare.lock") {
                    locks.push(entry.path());
                }
            }
        }
    }
    locks.sort();
    Ok(locks)
}

#[derive(Default)]
struct TreeUsage {
    complete_files: usize,
    complete_bytes: u64,
    partial_files: usize,
    partial_bytes: u64,
    unknown_files: usize,
    bytes: u64,
}

fn legacy_usage(tree: &Path) -> Result<TreeUsage, Failure> {
    let mut usage = TreeUsage::default();
    let mut seen = HashSet::new();
    let mut stack = vec![tree.to_path_buf()];
    while let Some(directory) = stack.pop() {
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        for entry in entries {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                stack.push(entry.path());
                continue;
            }
            if !file_type.is_file() {
                if file_type.is_symlink() {
                    let relative = entry
                        .path()
                        .strip_prefix(tree)
                        .unwrap_or(&entry.path())
                        .to_string_lossy()
                        .replace('\\', "/");
                    if !relative.starts_with("refs/") && !relative.starts_with("snapshots/") {
                        usage.unknown_files += 1;
                    }
                }
                continue;
            }
            let metadata = fs::symlink_metadata(entry.path())?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                usage.unknown_files += 1;
                continue;
            };
            if name.ends_with(".lock") || name.ends_with(".flare.lock") {
                continue;
            }
            if !seen.insert(identity(&metadata, &entry.path())) {
                continue;
            }
            usage.bytes += metadata.len();
            let relative = entry
                .path()
                .strip_prefix(tree)
                .unwrap_or(&entry.path())
                .to_string_lossy()
                .replace('\\', "/");
            let in_blobs = entry
                .path()
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|parent| parent == "blobs");
            if !in_blobs {
                if relative.starts_with("refs/") || relative.starts_with("snapshots/") {
                    continue;
                }
                usage.unknown_files += 1;
                continue;
            }
            if name.ends_with(".part") {
                usage.partial_files += 1;
                usage.partial_bytes += metadata.len();
            } else {
                usage.complete_files += 1;
                usage.complete_bytes += metadata.len();
            }
        }
    }
    Ok(usage)
}

fn files_with_prefix(directory: &Path) -> Result<(Vec<(String, PathBuf)>, usize), Failure> {
    let mut found = Vec::new();
    let mut count = 0;
    if !directory.is_dir() {
        return Ok((found, count));
    }
    let mut stack = vec![directory.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = match fs::read_dir(&current) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        for entry in entries {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                stack.push(entry.path());
                continue;
            }
            if !file_type.is_file() && !file_type.is_symlink() {
                continue;
            }
            count += 1;
            let relative = entry
                .path()
                .strip_prefix(directory)
                .unwrap_or(&entry.path())
                .to_string_lossy()
                .replace('\\', "/");
            found.push((relative, entry.path()));
        }
    }
    found.sort_by(|left, right| left.0.cmp(&right.0));
    Ok((found, count))
}

fn file_usage(path: &Path) -> Result<(usize, u64), Failure> {
    let (files, _) = files_with_prefix(path)?;
    let mut see = HashSet::new();
    let mut bytes = 0;
    for (_, path) in files {
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        if !metadata.is_file() {
            continue;
        }
        if see.insert(identity(&metadata, &path)) {
            bytes += metadata.len();
        }
    }
    Ok((count_files(path)?, bytes))
}

fn count_files(path: &Path) -> Result<usize, Failure> {
    let mut count = 0;
    let mut stack = vec![path.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = match fs::read_dir(&current) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        for entry in entries {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                stack.push(entry.path());
            } else {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn unique_bytes(path: &Path) -> Result<u64, Failure> {
    let (files, _) = files_with_prefix(path)?;
    let mut seen = HashSet::new();
    let mut bytes = 0;
    for (_, path) in files {
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        if metadata.is_file() && seen.insert(identity(&metadata, &path)) {
            bytes += metadata.len();
        }
    }
    Ok(bytes)
}

#[cfg(unix)]
fn identity(metadata: &fs::Metadata, _path: &Path) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (metadata.dev(), metadata.ino())
}

#[cfg(windows)]
fn identity(_metadata: &fs::Metadata, path: &Path) -> (u64, u64) {
    if let Some((identity, _)) = windows_file_identity(path) {
        return identity;
    }
    // An unreadable identity cannot establish that two paths share an object.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    (u64::MAX, hasher.finish())
}

#[cfg(windows)]
use crate::windows::file_identity as windows_file_identity;

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;

    fn hold(path: &Path) -> File {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .unwrap();
        assert_eq!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        file
    }

    fn hold_shared(path: &Path) -> File {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .unwrap();
        assert_eq!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) },
            0
        );
        file
    }

    fn write(path: &Path, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, bytes).unwrap();
    }

    const SCOPE: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn transport(parent: &Path) -> PathBuf {
        let root = root(parent);
        fs::create_dir_all(root.join("blobs")).unwrap();
        fs::create_dir_all(root.join("repos")).unwrap();
        write(&root.join("version"), VERSION);
        root
    }

    fn scoped(root: &Path) -> PathBuf {
        root.join("blobs").join(SCOPE)
    }

    fn repository(root: &Path, folder: &str) -> PathBuf {
        root.join("repos").join(SCOPE).join(folder)
    }

    #[test]
    fn inspection_classifies_transport_files() {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path();
        let root = transport(parent);
        let blobs_dir = scoped(&root);
        let repos = repository(&root, "models--o--n");
        write(&blobs_dir.join("sha256-aaa"), b"complete");
        write(&blobs_dir.join("sha256-bbb.part"), b"part");
        write(&blobs_dir.join("sha256-ccc.lock"), b"");
        write(&blobs_dir.join(".sha256-eee.part.corrupt-1-0"), b"bad");
        write(&blobs_dir.join(".sha256-fff.adopt-1-0"), b"adopt");
        write(&blobs_dir.join("notes.txt"), b"unknown");
        write(&blobs_dir.join("sha256-ddd"), b"live");
        let _active = hold(&blobs_dir.join("sha256-ddd.lock"));
        write(&repos.join("refs/main"), b"commit\n");
        write(
            &repos.join("snapshots/commit/model.safetensors"),
            b"snapshot",
        );

        let inspection =
            inspect(parent, &tempfile::tempdir().unwrap().path().to_path_buf()).unwrap();
        assert_eq!(inspection.transport.marker, "valid");
        assert!(!inspection.transport.acquisition_active);
        let blobs = &inspection.transport.blobs;
        assert_eq!(blobs.complete_files, 2);
        assert_eq!(
            blobs.complete_bytes,
            b"complete".len() as u64 + b"live".len() as u64
        );
        assert_eq!(blobs.partial_files, 1);
        assert_eq!(blobs.partial_bytes, b"part".len() as u64);
        assert_eq!(blobs.quarantined_files, 1);
        assert_eq!(blobs.quarantined_bytes, b"bad".len() as u64);
        assert_eq!(blobs.adopted_files, 1);
        assert_eq!(blobs.adopted_bytes, b"adopt".len() as u64);
        assert_eq!(blobs.lock_files, 2);
        assert_eq!(blobs.unknown_files, 1);
        assert_eq!(blobs.active_files, 1);
        assert_eq!(blobs.active_bytes, b"live".len() as u64);
        assert_eq!(inspection.transport.scopes.len(), 1);
        assert_eq!(inspection.transport.scopes[0].scope, SCOPE);
        let repository = &inspection.transport.repositories[0];
        assert_eq!(repository.scope, SCOPE);
        assert_eq!(repository.folder, "models--o--n");
        assert_eq!(repository.refs[0].revision, "main");
        assert_eq!(repository.refs[0].commit.as_deref(), Some("commit"));
        assert_eq!(repository.snapshots[0].files, 1);
        assert!(inspection.legacy.trees.is_empty());
        let document = serde_json::to_value(&inspection).unwrap();
        assert_eq!(document["transport"]["marker"], "valid");
        assert_eq!(document["transport"]["blobs"]["complete_files"], 2);
        assert_eq!(
            document["transport"]["repositories"][0]["folder"],
            "models--o--n"
        );
        assert!(document["legacy"]["trees"].is_array());
    }

    #[test]
    fn reclaim_removes_metadata_and_unlocked_files_only() {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path();
        let root = transport(parent);
        let blobs_dir = scoped(&root);
        let repos = repository(&root, "models--o--n");
        write(&blobs_dir.join("sha256-aaa"), b"complete");
        write(&blobs_dir.join("sha256-bbb.part"), b"partial");
        write(&blobs_dir.join("sha256-ccc"), b"live");
        write(&blobs_dir.join("sha256-ccc.part"), b"live-part");
        write(&blobs_dir.join(".sha256-ddd.part.corrupt-1-0"), b"bad");
        write(&blobs_dir.join(".sha256-eee.adopt-1-0"), b"adopt");
        write(&blobs_dir.join(".sha256-ccc.adopt-1-0"), b"live-adopt");
        write(&blobs_dir.join("notes.txt"), b"unknown");
        let _active = hold(&blobs_dir.join("sha256-ccc.lock"));
        fs::create_dir_all(repos.join("snapshots/commit")).unwrap();
        fs::hard_link(
            blobs_dir.join("sha256-aaa"),
            repos.join("snapshots/commit/model.safetensors"),
        )
        .unwrap();
        write(&repos.join("refs/main"), b"commit\n");

        let reclaimed =
            reclaim(parent, &tempfile::tempdir().unwrap().path().to_path_buf()).unwrap();
        assert_eq!(reclaimed.complete_files, 1);
        assert_eq!(reclaimed.complete_bytes, b"complete".len() as u64);
        assert_eq!(reclaimed.partial_files, 1);
        assert_eq!(reclaimed.partial_bytes, b"partial".len() as u64);
        assert_eq!(reclaimed.quarantined_files, 1);
        assert_eq!(reclaimed.quarantined_bytes, b"bad".len() as u64);
        assert_eq!(reclaimed.adopted_files, 1);
        assert_eq!(reclaimed.adopted_bytes, b"adopt".len() as u64);
        assert_eq!(reclaimed.repository_refs, 1);
        assert_eq!(reclaimed.snapshot_files, 1);
        assert_eq!(reclaimed.active_files, 3);
        assert_eq!(
            reclaimed.active_bytes,
            (b"live".len() + b"live-part".len() + b"live-adopt".len()) as u64
        );
        assert_eq!(reclaimed.retained.len(), 1);
        assert_eq!(reclaimed.retained[0].scope, SCOPE);
        assert!(root.join("version").is_file());
        assert!(blobs_dir.join("sha256-ccc.lock").is_file());
        assert!(blobs_dir.join("sha256-ccc").is_file());
        assert!(blobs_dir.join("sha256-ccc.part").is_file());
        assert!(blobs_dir.join(".sha256-ccc.adopt-1-0").is_file());
        assert!(blobs_dir.join("notes.txt").is_file());
        assert!(!blobs_dir.join("sha256-aaa").exists());
        assert!(!blobs_dir.join("sha256-bbb.part").exists());
        assert!(!blobs_dir.join(".sha256-ddd.part.corrupt-1-0").exists());
        assert!(!blobs_dir.join(".sha256-eee.adopt-1-0").exists());
        assert!(root.join("repos").is_dir());
        assert!(root.join("repos").join(SCOPE).is_dir());
        assert!(repos.is_dir());
        assert!(!repos.join("refs").exists());
        assert!(!repos.join("snapshots").exists());
    }

    #[test]
    fn inspection_does_not_create_a_missing_cache() {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path().join("downloads");
        let model_root = temporary.path().join("models");
        let inspection = inspect(&parent, &model_root).unwrap();
        assert!(!inspection.transport.present);
        assert!(inspection.legacy.trees.is_empty());
        assert!(!parent.exists());
        assert!(!model_root.exists());
    }

    #[test]
    fn reclaim_leaves_foreign_transport_roots_untouched() {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path();
        let root = transport(parent);
        write(&root.join("version"), b"some-other-cache 1\n");
        write(&scoped(&root).join("sha256-aaa.part"), b"partial");

        let reclaimed =
            reclaim(parent, &tempfile::tempdir().unwrap().path().to_path_buf()).unwrap();
        assert_eq!(reclaimed.marker, "foreign");
        assert_eq!(reclaimed.complete_files, 0);
        assert_eq!(reclaimed.partial_files, 0);
        assert!(scoped(&root).join("sha256-aaa.part").is_file());
    }

    #[test]
    fn reclaim_of_absent_transport_preserves_unverified_legacy_staging() {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path().join("downloads");
        let model_root = temporary.path().join("models");
        fs::create_dir_all(&parent).unwrap();
        let explicit = parent.join(".flare-hub-staging-abc");
        write(&explicit.join("models--o--n/blobs/sha256:a"), b"complete");
        write(
            &explicit.join("models--o--n/blobs/sha256:a.part"),
            b"partial",
        );

        let reclaimed = reclaim(&parent, &model_root).unwrap();
        assert_eq!(reclaimed.marker, "missing");
        assert!(!reclaimed.present);
        assert_eq!(reclaimed.legacy_trees, 0);
        assert_eq!(reclaimed.legacy_unverified, 1);
        assert_eq!(reclaimed.legacy_bytes, 0);
        assert!(explicit.is_dir());
    }

    #[test]
    fn legacy_cache_staging_is_reclaimed_only_when_its_locks_are_free() {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path().join("downloads");
        let model_root = temporary.path().join("models");
        let active = model_root.join("staging/trees/transfer-live");
        write(&active.join("models--o--n/blobs/sha256:a"), b"live");
        let _lease = hold(&model_root.join("locks/staging/transfer-live.lock"));
        let transfer = model_root.join("staging/trees/transfer-busy");
        write(&transfer.join("models--o--n/blobs/sha256:b"), b"busy");
        let _blob = hold(&transfer.join("models--o--n/blobs/sha256:b.lock"));
        let abandoned = model_root.join("staging/trees/transfer-idle");
        write(&abandoned.join("models--o--n/blobs/sha256:c"), b"idle");
        write(&abandoned.join("models--o--n/blobs/sha256:c.lock"), b"");
        let export = model_root.join("staging/trees/export-writer");
        write(&export.join("model.safetensors"), b"export");
        let explicit_busy = parent.join(".flare-hub-staging-busy");
        write(&explicit_busy.join("models--o--n/blobs/sha256:d"), b"busy");
        let _explicit = hold(&explicit_busy.join("models--o--n/blobs/sha256:d.lock"));
        let unknown = parent.join(".flare-hub-staging-unknown");
        write(&unknown.join("models--o--n/blobs/sha256:e"), b"unknown");
        write(&unknown.join("unexpected.txt"), b"keep");
        let cache_unknown = model_root.join("staging/trees/transfer-unknown");
        write(
            &cache_unknown.join("models--o--n/blobs/sha256:f"),
            b"unknown",
        );
        write(&cache_unknown.join("stray.txt"), b"keep");
        write(&model_root.join("locks/staging/transfer-unknown.lock"), b"");

        let inspection = inspect(&parent, &model_root).unwrap();
        assert_eq!(inspection.legacy.trees.len(), 6);
        assert_eq!(inspection.legacy.active_trees, 3);
        let complete: usize = inspection
            .legacy
            .trees
            .iter()
            .map(|tree| tree.complete_files)
            .sum();
        assert_eq!(complete, 6);

        let reclaimed = reclaim(&parent, &model_root).unwrap();
        assert_eq!(reclaimed.legacy_active, 2);
        assert_eq!(reclaimed.legacy_unverified, 2);
        assert_eq!(reclaimed.legacy_unknown_trees, 1);
        assert_eq!(reclaimed.legacy_trees, 1);
        assert_eq!(reclaimed.legacy_bytes, b"idle".len() as u64);
        assert!(active.is_dir());
        assert!(transfer.is_dir());
        assert!(!abandoned.exists());
        assert!(explicit_busy.is_dir());
        assert!(unknown.is_dir());
        assert!(cache_unknown.is_dir());
        assert!(export.is_dir());
        assert!(
            model_root
                .join("locks/staging/transfer-live.lock")
                .is_file()
        );
    }

    #[test]
    fn installed_cleanup_reclaims_only_consumed_orphans() {
        let temporary = tempfile::tempdir().unwrap();
        let root = transport(temporary.path());
        let blobs = scoped(&root);
        let snapshots = repository(&root, "models--o--n").join("snapshots");
        write(&blobs.join("sha256-aaa"), b"shared");
        write(&blobs.join("sha256-bbb"), b"only");
        write(&blobs.join("sha256-ccc"), b"failed-complete");
        write(&blobs.join("sha256-ddd.part"), b"failed-part");
        write(&blobs.join(".sha256-eee.part.corrupt-1-0"), b"suspect");
        write(&blobs.join("sha256-fff.lock"), b"");
        fs::create_dir_all(snapshots.join("commit-1")).unwrap();
        fs::create_dir_all(snapshots.join("commit-2")).unwrap();
        fs::hard_link(
            blobs.join("sha256-aaa"),
            snapshots.join("commit-1/a.safetensors"),
        )
        .unwrap();
        fs::hard_link(
            blobs.join("sha256-bbb"),
            snapshots.join("commit-1/b.safetensors"),
        )
        .unwrap();
        fs::hard_link(
            blobs.join("sha256-aaa"),
            snapshots.join("commit-2/a.safetensors"),
        )
        .unwrap();

        let consumed = reclaim_installed(&root, SCOPE, "models--o--n", "commit-1").unwrap();
        assert!(!consumed.acquisition_active);
        assert_eq!(consumed.snapshot_files, 2);
        assert_eq!(consumed.complete_files, 1);
        assert_eq!(consumed.complete_bytes, b"only".len() as u64);
        assert!(!snapshots.join("commit-1").exists());
        assert!(snapshots.join("commit-2/a.safetensors").is_file());
        assert!(blobs.join("sha256-aaa").is_file());
        assert!(!blobs.join("sha256-bbb").exists());
        assert!(blobs.join("sha256-ccc").is_file());
        assert!(blobs.join("sha256-ddd.part").is_file());
        assert!(blobs.join(".sha256-eee.part.corrupt-1-0").is_file());
        assert!(blobs.join("sha256-fff.lock").is_file());
    }

    #[test]
    fn concurrent_cleanup_preserves_an_active_acquisition() {
        let temporary = tempfile::tempdir().unwrap();
        let root = transport(temporary.path());
        let blobs = scoped(&root);
        let snapshots = repository(&root, "models--o--n").join("snapshots");
        write(&blobs.join("sha256-aaa"), b"consumed");
        fs::create_dir_all(snapshots.join("commit")).unwrap();
        fs::hard_link(
            blobs.join("sha256-aaa"),
            snapshots.join("commit/model.safetensors"),
        )
        .unwrap();
        let lease = hold_shared(&acquisition_lease(&root));
        let consumed = std::thread::scope(|scope| {
            let root = root.clone();
            scope
                .spawn(move || reclaim_installed(&root, SCOPE, "models--o--n", "commit").unwrap())
                .join()
                .unwrap()
        });
        assert!(consumed.acquisition_active);
        assert!(consumed.deferred);
        assert_eq!(consumed.snapshot_files, 0);
        assert_eq!(consumed.complete_files, 0);
        assert!(
            root.join("repos")
                .join(SCOPE)
                .join("models--o--n/installed/commit")
                .is_file()
        );
        assert!(snapshots.join("commit/model.safetensors").is_file());
        assert!(blobs.join("sha256-aaa").is_file());

        drop(lease);
        let consumed = reclaim_installed(&root, SCOPE, "models--o--n", "commit").unwrap();
        assert!(!consumed.acquisition_active);
        assert!(!consumed.deferred);
        assert_eq!(consumed.snapshot_files, 1);
        assert_eq!(consumed.complete_files, 1);
        assert!(!snapshots.join("commit").exists());
        assert!(!blobs.join("sha256-aaa").exists());
        assert!(
            !root
                .join("repos")
                .join(SCOPE)
                .join("models--o--n/installed/commit")
                .exists()
        );
    }

    #[test]
    fn deferred_cleanup_runs_on_the_next_successful_install() {
        let temporary = tempfile::tempdir().unwrap();
        let root = transport(temporary.path());
        let blobs = scoped(&root);
        let snapshots = repository(&root, "models--o--n").join("snapshots");
        write(&blobs.join("sha256-aaa"), b"first");
        write(&blobs.join("sha256-bbb"), b"second");
        write(&blobs.join("sha256-ccc"), b"failed");
        for (commit, key) in [
            ("commit-1", "sha256-aaa"),
            ("commit-2", "sha256-bbb"),
            ("commit-failed", "sha256-ccc"),
        ] {
            fs::create_dir_all(snapshots.join(commit)).unwrap();
            fs::hard_link(
                blobs.join(key),
                snapshots.join(commit).join("model.safetensors"),
            )
            .unwrap();
        }
        let lease = hold_shared(&acquisition_lease(&root));
        let deferred = reclaim_installed(&root, SCOPE, "models--o--n", "commit-1").unwrap();
        assert!(deferred.acquisition_active);
        assert!(deferred.deferred);
        assert!(snapshots.join("commit-1/model.safetensors").is_file());
        drop(lease);

        let consumed = reclaim_installed(&root, SCOPE, "models--o--n", "commit-2").unwrap();
        assert!(!consumed.acquisition_active);
        assert_eq!(consumed.complete_files, 2);
        assert_eq!(
            consumed.complete_bytes,
            (b"first".len() + b"second".len()) as u64
        );
        assert!(!snapshots.join("commit-1").exists());
        assert!(!snapshots.join("commit-2").exists());
        assert!(!blobs.join("sha256-aaa").exists());
        assert!(!blobs.join("sha256-bbb").exists());
        assert!(snapshots.join("commit-failed/model.safetensors").is_file());
        assert!(blobs.join("sha256-ccc").is_file());
        assert!(
            !root
                .join("repos")
                .join(SCOPE)
                .join("models--o--n/installed/commit-1")
                .exists()
        );
    }

    #[test]
    fn concurrent_installed_cleanup_reclaims_once() {
        let temporary = tempfile::tempdir().unwrap();
        let root = transport(temporary.path());
        let blobs = scoped(&root);
        let snapshots = repository(&root, "models--o--n").join("snapshots");
        write(&blobs.join("sha256-aaa"), b"consumed");
        fs::create_dir_all(snapshots.join("commit")).unwrap();
        fs::hard_link(
            blobs.join("sha256-aaa"),
            snapshots.join("commit/model.safetensors"),
        )
        .unwrap();
        let (snapshot_files, complete_files): (usize, usize) = std::thread::scope(|scope| {
            let passes: Vec<_> = (0..4)
                .map(|_| {
                    let root = root.clone();
                    scope.spawn(move || {
                        let consumed =
                            reclaim_installed(&root, SCOPE, "models--o--n", "commit").unwrap();
                        (consumed.snapshot_files, consumed.complete_files)
                    })
                })
                .collect();
            passes
                .into_iter()
                .map(|pass| pass.join().unwrap())
                .fold((0, 0), |total, pass| (total.0 + pass.0, total.1 + pass.1))
        });
        assert_eq!(snapshot_files, 1);
        assert_eq!(complete_files, 1);
        assert!(!snapshots.join("commit").exists());
        assert!(!blobs.join("sha256-aaa").exists());
    }

    #[test]
    fn legacy_acquisition_holds_the_lease_until_the_tree_is_gone() {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path().join("downloads");
        let model_root = temporary.path().join("models");
        let tree = model_root.join("staging/trees/transfer-idle");
        write(&tree.join("models--o--n/blobs/sha256:a"), b"idle");
        write(&model_root.join("locks/staging/transfer-idle.lock"), b"");
        let trees = legacy_trees(&parent, &model_root).unwrap();
        let guard = trees[0].acquire().unwrap();
        assert!(matches!(guard, LegacyAccess::Held { .. }));
        assert!(matches!(trees[0].acquire().unwrap(), LegacyAccess::Active));
        drop(guard);
        assert!(matches!(
            trees[0].acquire().unwrap(),
            LegacyAccess::Held { .. }
        ));

        let explicit = parent.join(".flare-hub-staging-leftover");
        write(&explicit.join("models--o--n/blobs/sha256:b"), b"leftover");
        let trees = legacy_trees(&parent, &model_root).unwrap();
        let leftover = trees
            .iter()
            .find(|tree| tree.path == explicit)
            .expect("explicit legacy tree");
        assert!(matches!(
            leftover.acquire().unwrap(),
            LegacyAccess::Unverified
        ));
    }

    #[test]
    fn concurrent_gc_preserves_an_active_acquisition() {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path().join("downloads");
        let model_root = temporary.path().join("models");
        let root = transport(&parent);
        let blobs_dir = scoped(&root);
        let repos = repository(&root, "models--o--n");
        write(&blobs_dir.join("sha256-aaa"), b"complete");
        write(&blobs_dir.join("sha256-bbb.part"), b"partial");
        fs::create_dir_all(repos.join("snapshots/commit")).unwrap();
        fs::hard_link(
            blobs_dir.join("sha256-aaa"),
            repos.join("snapshots/commit/model.safetensors"),
        )
        .unwrap();
        write(&repos.join("refs/main"), b"commit\n");
        let lease = hold_shared(&acquisition_lease(&root));

        let inspection = inspect(&parent, &model_root).unwrap();
        assert!(inspection.transport.acquisition_active);
        let reclaimed = std::thread::scope(|scope| {
            let parent = parent.clone();
            let model_root = model_root.clone();
            scope
                .spawn(move || reclaim(&parent, &model_root).unwrap())
                .join()
                .unwrap()
        });
        assert!(reclaimed.acquisition_active);
        assert_eq!(reclaimed.complete_files, 0);
        assert_eq!(reclaimed.partial_files, 0);
        assert_eq!(reclaimed.repository_refs, 0);
        assert_eq!(reclaimed.snapshot_files, 0);
        assert!(blobs_dir.join("sha256-aaa").is_file());
        assert!(blobs_dir.join("sha256-bbb.part").is_file());
        assert!(repos.join("snapshots/commit/model.safetensors").is_file());
        assert!(repos.join("refs/main").is_file());

        drop(lease);
        let reclaimed = reclaim(&parent, &model_root).unwrap();
        assert!(!reclaimed.acquisition_active);
        assert_eq!(reclaimed.complete_files, 1);
        assert_eq!(reclaimed.partial_files, 1);
        assert_eq!(reclaimed.repository_refs, 1);
        assert_eq!(reclaimed.snapshot_files, 1);
        assert!(!blobs_dir.join("sha256-aaa").exists());
        assert!(!blobs_dir.join("sha256-bbb.part").exists());
        assert!(!repos.join("snapshots").exists());
        assert!(!repos.join("refs").exists());
    }

    #[test]
    fn concurrent_reclaim_preserves_an_active_blob() {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path().join("downloads");
        let model_root = temporary.path().join("models");
        let root = transport(&parent);
        let blobs_dir = scoped(&root);
        write(&blobs_dir.join("sha256-aaa"), b"live");
        let lock = blobs_dir.join("sha256-aaa.lock");
        let active = hold(&lock);
        std::thread::scope(|scope| {
            let parent = parent.clone();
            let model_root = model_root.clone();
            let pass = scope.spawn(move || reclaim(&parent, &model_root).unwrap());
            let reclaimed = pass.join().unwrap();
            assert_eq!(reclaimed.active_files, 1);
            assert_eq!(reclaimed.complete_files, 0);
        });
        assert!(blobs_dir.join("sha256-aaa").is_file());
        drop(active);
        let reclaimed = reclaim(&parent, &model_root).unwrap();
        assert_eq!(reclaimed.complete_files, 1);
        assert!(!blobs_dir.join("sha256-aaa").exists());
        assert!(lock.is_file());
    }

    #[test]
    fn concurrent_reclaim_of_the_transport_root_is_idempotent() {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path().join("downloads");
        let model_root = temporary.path().join("models");
        let root = transport(&parent);
        let blobs_dir = scoped(&root);
        let repos = repository(&root, "models--o--n");
        write(&blobs_dir.join("sha256-aaa"), b"complete");
        write(&blobs_dir.join("sha256-bbb.part"), b"partial");
        fs::create_dir_all(repos.join("snapshots/commit")).unwrap();
        fs::hard_link(
            blobs_dir.join("sha256-aaa"),
            repos.join("snapshots/commit/model.safetensors"),
        )
        .unwrap();
        write(&repos.join("refs/main"), b"commit\n");
        let (complete, partial): (usize, usize) = std::thread::scope(|scope| {
            let passes: Vec<_> = (0..4)
                .map(|_| {
                    let parent = parent.clone();
                    let model_root = model_root.clone();
                    scope.spawn(move || {
                        let reclaimed = reclaim(&parent, &model_root).unwrap();
                        (reclaimed.complete_files, reclaimed.partial_files)
                    })
                })
                .collect();
            passes
                .into_iter()
                .map(|pass| pass.join().unwrap())
                .fold((0, 0), |total, pass| (total.0 + pass.0, total.1 + pass.1))
        });
        assert_eq!((complete, partial), (1, 1));
        assert!(root.join("version").is_file());
        assert!(!blobs_dir.join("sha256-aaa").exists());
        assert!(!blobs_dir.join("sha256-bbb.part").exists());
    }

    #[test]
    fn concurrent_reclaim_deletes_a_cache_staging_tree_once() {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path().join("downloads");
        let model_root = temporary.path().join("models");
        let tree = model_root.join("staging/trees/transfer-idle");
        write(&tree.join("models--o--n/blobs/sha256:a"), b"idle");
        write(&model_root.join("locks/staging/transfer-idle.lock"), b"");
        let removed: usize = std::thread::scope(|scope| {
            let passes: Vec<_> = (0..4)
                .map(|_| {
                    let parent = parent.clone();
                    let model_root = model_root.clone();
                    scope.spawn(move || reclaim(&parent, &model_root).unwrap().legacy_trees)
                })
                .collect();
            passes.into_iter().map(|pass| pass.join().unwrap()).sum()
        });
        assert_eq!(removed, 1);
        assert!(!tree.exists());
    }
}

#[cfg(all(test, windows))]
mod windows_tests {
    use super::*;

    #[test]
    fn hardlinks_share_an_identity_and_are_counted_once() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        let alias = temporary.path().join("alias");
        fs::write(&source, b"content").unwrap();
        fs::hard_link(&source, &alias).unwrap();
        let source_metadata = fs::metadata(&source).unwrap();
        let alias_metadata = fs::metadata(&alias).unwrap();
        assert_eq!(
            identity(&source_metadata, &source),
            identity(&alias_metadata, &alias)
        );
        assert_eq!(links(&source_metadata, &source), 2);
        assert_eq!(unique_bytes(temporary.path()).unwrap(), 7);
    }
}
