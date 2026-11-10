// SPDX-License-Identifier: Apache-2.0
//! Verified, versioned extraction of the native application's embedded files.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Cursor, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};
use zip::{CompressionMethod, ZipArchive};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
const REPARSE: u32 = 0x400;
const OPEN_REPARSE: u32 = 0x0020_0000;
const MAX_FILES: usize = 16384;
const MAX_BYTES: u64 = 16 * 1024 * 1024 * 1024;
static UNIQUE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: u32,
    entrypoint: String,
    files: Vec<ManifestFile>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestFile {
    path: String,
    size: u64,
    sha256: String,
}
struct Bundle<'a> {
    bytes: &'a [u8],
    manifest: Manifest,
    manifest_bytes: Vec<u8>,
}
pub struct Prepared {
    pub directory: PathBuf,
    _lock: File,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate_name(value: &str) -> Result<String> {
    if value.is_empty() || value.len() > 32768 || value.contains('\\') || value.starts_with('/') {
        return Err(format!("Invalid bundle path: {value:?}").into());
    }
    let mut depth = 0;
    for part in value.split('/') {
        depth += 1;
        if depth > 64
            || part.is_empty()
            || matches!(part, "." | "..")
            || part.ends_with(['.', ' '])
            || part.encode_utf16().count() > 255
            || part.chars().any(|c| c < ' ' || "<>:\"|?*".contains(c))
        {
            return Err(format!("Invalid bundle path: {value:?}").into());
        }
        let device = part
            .split('.')
            .next()
            .unwrap()
            .trim_end_matches(' ')
            .to_uppercase();
        let numbered = device
            .strip_prefix("COM")
            .or_else(|| device.strip_prefix("LPT"));
        if matches!(
            device.as_str(),
            "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$" | "CLOCK$"
        ) || numbered.is_some_and(|v| {
            matches!(
                v,
                "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        }) {
            return Err(format!("Windows device path is not allowed: {value:?}").into());
        }
    }
    Ok(value.to_lowercase())
}

impl<'a> Bundle<'a> {
    fn open(bytes: &'a [u8]) -> Result<Self> {
        let mut zip = ZipArchive::new(Cursor::new(bytes))?;
        if zip.len() > MAX_FILES + 1 {
            return Err("Bundle has too many entries".into());
        }
        let mut names = BTreeMap::new();
        for index in 0..zip.len() {
            let file = zip.by_index(index)?;
            let name = std::str::from_utf8(file.name_raw())?;
            let key = validate_name(name.trim_end_matches('/'))?;
            if !matches!(
                file.compression(),
                CompressionMethod::Stored | CompressionMethod::Deflated
            ) || file
                .unix_mode()
                .is_some_and(|mode| mode & 0o170000 == 0o120000)
            {
                return Err(format!("Unsupported bundle entry: {name}").into());
            }
            if names
                .insert(key, (name.to_owned(), file.size(), file.is_dir()))
                .is_some()
            {
                return Err("Duplicate bundle path".into());
            }
        }
        let mut manifest_bytes = Vec::new();
        zip.by_name("bundle.json")?
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut manifest_bytes)?;
        if manifest_bytes.len() > 8 * 1024 * 1024 {
            return Err("Bundle manifest is too large".into());
        }
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        if manifest.format != 1
            || manifest.entrypoint != "flare-winui.exe"
            || manifest.files.is_empty()
            || manifest.files.len() > MAX_FILES
        {
            return Err("Unsupported native bundle manifest".into());
        }
        let mut expected = BTreeSet::new();
        let mut total = 0u64;
        for file in &manifest.files {
            let key = validate_name(&file.path)?;
            if key == "bundle.json" || !valid_digest(&file.sha256) || !expected.insert(key.clone())
            {
                return Err("Invalid or duplicate manifest file".into());
            }
            total = total
                .checked_add(file.size)
                .filter(|n| *n <= MAX_BYTES)
                .ok_or("Bundle is too large")?;
            let Some((name, size, directory)) = names.get(&key) else {
                return Err(format!("Missing bundle file: {}", file.path).into());
            };
            if name != &file.path || *size != file.size || *directory {
                return Err("ZIP entry differs from manifest".into());
            }
        }
        if !expected.contains("flare-winui.exe") {
            return Err("Bundle entrypoint is missing".into());
        }
        for key in &expected {
            let mut path = key.as_str();
            while let Some((parent, _)) = path.rsplit_once('/') {
                if expected.contains(parent) {
                    return Err("A bundle file is also a directory".into());
                }
                path = parent;
            }
        }
        for (key, (_, _, directory)) in names {
            if key != "bundle.json" && !directory && !expected.contains(&key) {
                return Err("Bundle contains an unlisted file".into());
            }
        }
        Ok(Self {
            bytes,
            manifest,
            manifest_bytes,
        })
    }
}

fn plain_metadata(path: &Path) -> Result<Option<fs::Metadata>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_attributes() & REPARSE != 0 {
                return Err(format!(
                    "Reparse points are not allowed in the desktop cache: {}",
                    path.display()
                )
                .into());
            }
            Ok(Some(metadata))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}
fn plain_ancestors(path: &Path, create: bool) -> Result<()> {
    if !path.is_absolute() {
        return Err("Desktop cache path must be absolute".into());
    }
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component);
        if !current.has_root() {
            continue;
        }
        if create && plain_metadata(&current)?.is_none() {
            match fs::create_dir(&current) {
                Ok(()) => (),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
                Err(error) => return Err(error.into()),
            }
        }
        match plain_metadata(&current)? {
            Some(metadata) if metadata.is_dir() => (),
            _ => {
                return Err(format!(
                    "Desktop cache ancestor is not a directory: {}",
                    current.display()
                )
                .into());
            }
        }
    }
    Ok(())
}
fn open_plain(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(OPEN_REPARSE)
        .open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.file_attributes() & REPARSE != 0 {
        return Err("Desktop cache contains a non-regular file".into());
    }
    Ok(file)
}
fn tree_files(directory: &Path, prefix: &str, found: &mut BTreeSet<String>) -> Result<()> {
    for item in fs::read_dir(directory)? {
        let item = item?;
        let name = item.file_name();
        let name = name
            .to_str()
            .ok_or("Desktop cache contains an invalid filename")?;
        let relative = if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix}/{name}")
        };
        validate_name(&relative)?;
        let metadata =
            plain_metadata(&item.path())?.ok_or("Desktop cache changed while checking it")?;
        if metadata.is_dir() {
            tree_files(&item.path(), &relative, found)?;
        } else if metadata.is_file() {
            if !found.insert(validate_name(&relative)?) {
                return Err("Duplicate desktop cache file".into());
            }
        } else {
            return Err("Desktop cache contains an unsupported file".into());
        }
    }
    Ok(())
}
impl Bundle<'_> {
    fn valid(&self, directory: &Path) -> Result<bool> {
        let Some(metadata) = plain_metadata(directory)? else {
            return Ok(false);
        };
        if !metadata.is_dir() {
            return Ok(false);
        }
        let mut found = BTreeSet::new();
        tree_files(directory, "", &mut found)?;
        let expected: BTreeSet<_> = self
            .manifest
            .files
            .iter()
            .map(|f| f.path.to_lowercase())
            .chain(std::iter::once("bundle.json".into()))
            .collect();
        if found != expected {
            return Ok(false);
        }
        if fs::read(directory.join("bundle.json"))? != self.manifest_bytes {
            return Ok(false);
        }
        for expected in &self.manifest.files {
            let path = directory.join(&expected.path);
            plain_ancestors(path.parent().unwrap(), false)?;
            let mut file = open_plain(&path)?;
            if file.metadata()?.len() != expected.size {
                return Ok(false);
            }
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 128 * 1024];
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                hash.update(&buffer[..count]);
            }
            if format!("{:x}", hash.finalize()) != expected.sha256 {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn extract(&self, directory: &Path) -> Result<()> {
        let mut zip = ZipArchive::new(Cursor::new(self.bytes))?;
        for expected in &self.manifest.files {
            let target = directory.join(&expected.path);
            plain_ancestors(target.parent().unwrap(), true)?;
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .custom_flags(OPEN_REPARSE)
                .open(&target)?;
            let mut input = zip.by_name(&expected.path)?;
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 128 * 1024];
            let mut total = 0u64;
            loop {
                let count = input.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                total = total
                    .checked_add(count as u64)
                    .filter(|n| *n <= expected.size)
                    .ok_or("Bundle file exceeds its declared length")?;
                output.write_all(&buffer[..count])?;
                hash.update(&buffer[..count]);
            }
            if total != expected.size || format!("{:x}", hash.finalize()) != expected.sha256 {
                return Err(format!("Bundle file verification failed: {}", expected.path).into());
            }
            output.sync_all()?;
        }
        let mut manifest = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("bundle.json"))?;
        manifest.write_all(&self.manifest_bytes)?;
        manifest.sync_all()?;
        Ok(())
    }
}
fn lock(parent: &Path, identity: &str) -> Result<File> {
    let path = parent.join(format!("{identity}.lock"));
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        plain_ancestors(parent, false)?;
        plain_metadata(&path)?;
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .share_mode(0)
            .custom_flags(OPEN_REPARSE)
            .open(&path)
        {
            Ok(file) => {
                if file.metadata()?.file_attributes() & REPARSE != 0 {
                    return Err("Desktop extraction lock is a reparse point".into());
                }
                return Ok(file);
            }
            Err(error) if error.raw_os_error() == Some(32) && Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(40))
            }
            Err(error) => return Err(format!("Cannot lock desktop extraction: {error}").into()),
        }
    }
}
fn unique_path(parent: &Path, identity: &str, label: &str) -> PathBuf {
    parent.join(format!(
        ".{identity}.{label}-{}-{}",
        std::process::id(),
        UNIQUE.fetch_add(1, Ordering::Relaxed)
    ))
}
struct Stage(PathBuf);
impl Stage {
    fn create(parent: &Path, identity: &str) -> Result<Self> {
        for _ in 0..100 {
            let path = unique_path(parent, identity, "stage");
            match fs::create_dir(&path) {
                Ok(()) => {
                    plain_ancestors(&path, false)?;
                    return Ok(Self(path));
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Err("Cannot create a unique desktop staging directory".into())
    }
}
fn remove_plain(path: &Path) -> Result<()> {
    let Some(metadata) = plain_metadata(path)? else {
        return Ok(());
    };
    if metadata.is_dir() {
        let mut found = BTreeSet::new();
        tree_files(path, "", &mut found)?;
        fs::remove_dir_all(path)?;
    } else {
        fs::remove_file(path)?;
    }
    Ok(())
}
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = remove_plain(&self.0);
    }
}
/// Keep the extraction lock until the caller has spawned the verified entrypoint.
pub fn prepare(bytes: &[u8], parent: &Path, identity: &str) -> Result<Prepared> {
    if !valid_digest(identity) || digest(bytes) != identity {
        return Err("Embedded desktop payload identity is invalid".into());
    }
    let bundle = Bundle::open(bytes)?;
    plain_ancestors(parent, true)?;
    let lock = lock(parent, identity)?;
    let directory = parent.join(identity);
    if bundle.valid(&directory)? {
        return Ok(Prepared {
            directory,
            _lock: lock,
        });
    }
    let stage = Stage::create(parent, identity)?;
    bundle.extract(&stage.0)?;
    if !bundle.valid(&stage.0)? {
        return Err("Staged desktop payload verification failed".into());
    }
    let previous = if plain_metadata(&directory)?.is_some() {
        let previous = unique_path(parent, identity, "replaced");
        fs::rename(&directory, &previous)?;
        Some(previous)
    } else {
        None
    };
    if let Err(error) = fs::rename(&stage.0, &directory) {
        if let Some(previous) = &previous {
            let _ = fs::rename(previous, &directory);
        }
        return Err(format!("Cannot publish desktop extraction: {error}").into());
    }
    if let Some(previous) = previous {
        let _ = remove_plain(&previous);
    }
    Ok(Prepared {
        directory,
        _lock: lock,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use zip::{ZipWriter, write::SimpleFileOptions};
    fn fixture(files: &[(&str, &[u8])], mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        let mut manifest = serde_json::json!({"format":1,"entrypoint":"flare-winui.exe","files":files.iter().map(|(p,b)|serde_json::json!({"path":p,"size":b.len(),"sha256":digest(b)})).collect::<Vec<_>>()});
        mutate(&mut manifest);
        let mut z = ZipWriter::new(Cursor::new(Vec::new()));
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        z.start_file("bundle.json", options).unwrap();
        z.write_all(&serde_json::to_vec(&manifest).unwrap())
            .unwrap();
        for (p, b) in files {
            z.start_file(*p, options).unwrap();
            z.write_all(b).unwrap();
        }
        z.finish().unwrap().into_inner()
    }
    pub(super) fn native_bundle() -> Vec<u8> {
        fixture(
            &[
                ("flare-winui.exe", b"native GUI"),
                ("runtime/support.dll", b"support"),
                ("resources.pri", b"resources"),
            ],
            |_| {},
        )
    }
    #[test]
    fn cache_reuse_and_repair() {
        let temp = tempfile::tempdir().unwrap();
        let bytes = native_bundle();
        let hash = digest(&bytes);
        let directory = {
            let held = prepare(&bytes, temp.path(), &hash).unwrap();
            held.directory.clone()
        };
        let before = fs::metadata(directory.join("flare-winui.exe"))
            .unwrap()
            .last_write_time();
        {
            let held = prepare(&bytes, temp.path(), &hash).unwrap();
            assert_eq!(held.directory, directory);
            assert_eq!(
                fs::metadata(directory.join("flare-winui.exe"))
                    .unwrap()
                    .last_write_time(),
                before
            );
        }
        fs::write(directory.join("runtime/support.dll"), b"corrupt").unwrap();
        fs::remove_file(directory.join("resources.pri")).unwrap();
        fs::write(directory.join("unexpected.dll"), b"unlisted").unwrap();
        let held = prepare(&bytes, temp.path(), &hash).unwrap();
        assert_eq!(
            fs::read(held.directory.join("runtime/support.dll")).unwrap(),
            b"support"
        );
        assert_eq!(
            fs::read(held.directory.join("resources.pri")).unwrap(),
            b"resources"
        );
        assert!(!held.directory.join("unexpected.dll").exists());
        assert!(
            Bundle::open(&bytes)
                .unwrap()
                .valid(&held.directory)
                .unwrap()
        );
    }
    #[test]
    fn failed_verification_never_publishes_partial_files() {
        let temp = tempfile::tempdir().unwrap();
        let bytes = fixture(&[("flare-winui.exe", b"exe")], |m| {
            m["files"][0]["sha256"] = "0".repeat(64).into()
        });
        let hash = digest(&bytes);
        assert!(prepare(&bytes, temp.path(), &hash).is_err());
        assert!(!temp.path().join(&hash).exists());
        assert!(
            fs::read_dir(temp.path())
                .unwrap()
                .all(|e| !e.unwrap().file_type().unwrap().is_dir())
        );
    }
    #[test]
    fn malformed_paths_and_duplicates_are_rejected() {
        for invalid in [
            "../outside",
            "/absolute",
            "C:/outside",
            "a\\b",
            "a:stream",
            "a//b",
            "a/./b",
            "a/../b",
            "AUX.txt",
            "folder/COM1",
            "LPT¹.bin",
            "NUL .txt",
            "file.",
            "file ",
            "a\0b",
        ] {
            assert!(validate_name(invalid).is_err(), "accepted {invalid:?}");
        }
        let duplicate = fixture(
            &[
                ("flare-winui.exe", b"exe"),
                ("Support.dll", b"a"),
                ("support.dll", b"b"),
            ],
            |_| {},
        );
        assert!(Bundle::open(&duplicate).is_err());
        let traversal = fixture(
            &[("flare-winui.exe", b"exe"), ("../escape", b"bad")],
            |_| {},
        );
        assert!(Bundle::open(&traversal).is_err());
        assert!(validate_name("Assets/模型 logo.png").is_ok());
    }
    #[test]
    fn concurrent_first_preparation_of_same_absent_base_is_serialized() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("absent/deep/cache");
        let bytes = Arc::new(native_bundle());
        let barrier = Arc::new(Barrier::new(6));
        let handles: Vec<_> = (0..6)
            .map(|_| {
                let bytes = bytes.clone();
                let barrier = barrier.clone();
                let parent = parent.clone();
                thread::spawn(move || {
                    barrier.wait();
                    let held = prepare(&bytes, &parent, &digest(&bytes)).unwrap();
                    held.directory.clone()
                })
            })
            .collect();
        let directories: Vec<_> = handles.into_iter().map(|t| t.join().unwrap()).collect();
        assert!(directories.iter().all(|d| d == &directories[0]));
        assert!(
            Bundle::open(&bytes)
                .unwrap()
                .valid(&directories[0])
                .unwrap()
        );
        assert_eq!(
            fs::read_dir(parent)
                .unwrap()
                .filter(|e| e.as_ref().unwrap().file_type().unwrap().is_dir())
                .count(),
            1
        );
    }
}

#[cfg(test)]
mod reparse_tests {
    use super::*;
    use std::os::windows::{ffi::OsStrExt, io::AsRawHandle};
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn DeviceIoControl(
            device: *mut std::ffi::c_void,
            code: u32,
            input: *const std::ffi::c_void,
            input_size: u32,
            output: *mut std::ffi::c_void,
            output_size: u32,
            returned: *mut u32,
            overlapped: *mut std::ffi::c_void,
        ) -> i32;
    }
    fn junction(link: &Path, target: &Path) {
        fs::create_dir(link).unwrap();
        let print: Vec<u16> = target.as_os_str().encode_wide().collect();
        let substitute: Vec<u16> = std::ffi::OsStr::new("\\??\\")
            .encode_wide()
            .chain(print.iter().copied())
            .collect();
        let mut buffer = Vec::new();
        buffer.extend_from_slice(&0xa0000003u32.to_le_bytes());
        buffer.extend_from_slice(
            &((8 + (substitute.len() + print.len() + 2) * 2) as u16).to_le_bytes(),
        );
        buffer.extend_from_slice(&0u16.to_le_bytes());
        for n in [
            0,
            substitute.len() * 2,
            (substitute.len() + 1) * 2,
            print.len() * 2,
        ] {
            buffer.extend_from_slice(&(n as u16).to_le_bytes());
        }
        for n in substitute
            .into_iter()
            .chain(Some(0))
            .chain(print)
            .chain(Some(0))
        {
            buffer.extend_from_slice(&n.to_le_bytes());
        }
        let handle = OpenOptions::new()
            .write(true)
            .custom_flags(OPEN_REPARSE | 0x02000000)
            .open(link)
            .unwrap();
        let mut returned = 0;
        let result = unsafe {
            DeviceIoControl(
                handle.as_raw_handle(),
                0x000900a4,
                buffer.as_ptr().cast(),
                buffer.len() as u32,
                std::ptr::null_mut(),
                0,
                &mut returned,
                std::ptr::null_mut(),
            )
        };
        assert_ne!(result, 0, "{}", std::io::Error::last_os_error());
    }
    #[test]
    fn reparse_cache_ancestors_are_rejected_without_writing_the_target() {
        let temp = tempfile::tempdir().unwrap();
        let outside = temp.path().join("outside");
        fs::create_dir(&outside).unwrap();
        let link = temp.path().join("linked");
        junction(&link, &outside);
        let bytes = super::tests::native_bundle();
        assert!(prepare(&bytes, &link.join("cache"), &digest(&bytes)).is_err());
        assert!(!outside.join("cache").exists());
    }
}

#[cfg(test)]
mod payload_tests {
    use super::*;
    #[test]
    fn different_versions_can_prepare_the_same_initially_absent_parent() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("absent/cache");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let threads: Vec<_> = (0..2)
            .map(|index| {
                let parent = parent.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    let mut bytes = super::tests::native_bundle();
                    bytes.extend_from_slice(&[index]);
                    barrier.wait();
                    let prepared = prepare(&bytes, &parent, &digest(&bytes)).unwrap();
                    prepared.directory.clone()
                })
            })
            .collect();
        let paths: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_ne!(paths[0], paths[1]);
        assert!(
            paths
                .iter()
                .all(|path| path.join("flare-winui.exe").is_file())
        );
    }
    #[test]
    fn real_embedded_payload_can_be_prepared_without_launching() {
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/payload.zip"));
        if bytes.is_empty() {
            eprintln!("No packaged payload present; fixture tests remain available");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let prepared = prepare(bytes, temp.path(), &digest(bytes)).unwrap();
        assert!(prepared.directory.join("flare-winui.exe").is_file());
        assert!(
            Bundle::open(bytes)
                .unwrap()
                .valid(&prepared.directory)
                .unwrap()
        );
    }
}
