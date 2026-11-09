//! Terminal adapters for prepared-core acquisition.
use crate::cli::HubOptions;
#[cfg(unix)]
pub use flare::hub::{available_bytes, transport_root};
pub use flare::hub::{is_repository, repository};
use std::path::{Path, PathBuf};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[cfg(unix)]
pub fn token(options: &HubOptions) -> Result<Option<String>> {
    flare::hub::token(&options.into())
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
    publisher_keys: &[PathBuf],
    trust_store: Option<&Path>,
    profiles: &[flare_helios::runtime_core::Publisher],
) -> Result<flare_helios::core::cache::Entry> {
    let mut resolved = flare::options::HubOptions::from(options);
    if resolved.revision.is_none() && repository(source)?.1.is_none() {
        resolved.revision = Some(crate::hardware::detect_isolated(gpu_count)?);
    }
    let progress = flare_helios::core::progress::AcquisitionProgress::new();
    let startup = flare_helios::startup::StartupProgress::stderr_rows();
    let display = flare_helios::core::progress::AcquisitionDisplay::new(&startup, &progress);
    display.stage(
        "Acquire core",
        flare_helios::core::progress::Counter::Network,
        None,
    );
    let cache = flare::cache_support::model_root(options.core_cache.as_deref())
        .map_err(|error| error.to_string())?;
    eprintln!("Prepared-model cache: {}", cache.display());
    if !resolved.offline {
        let (repository, source_revision) = repository(source)?;
        let revision = resolved.revision.as_deref().or(source_revision).unwrap();
        eprintln!("Core source: {repository}@{revision}");
    }
    let result = (|| -> Result<flare_helios::core::cache::Entry> {
        let entry = flare::hub::acquire_watched(
            source,
            &resolved,
            gpu_count,
            publisher_keys,
            trust_store,
            profiles,
            progress.clone(),
        )?;
        flare_helios::core::cache::Root::open(cache)?.publish_entry(&entry)?;
        Ok(entry)
    })();
    display.finish();
    let snapshot = progress.snapshot();
    let state = if result.is_ok() {
        "completed"
    } else {
        "failed"
    };
    eprintln!(
        "Acquisition {state}: network {} bytes, local source {} bytes, materialized {} bytes, verification {} bytes, elapsed {:.3}s",
        snapshot.network_bytes,
        snapshot.local_source_bytes,
        snapshot.materialized_bytes(),
        snapshot.verification_bytes,
        snapshot.elapsed.as_secs_f64()
    );
    result
}

#[cfg(unix)]
pub fn download_watched(
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    token: Option<String>,
    publisher_keys: &[PathBuf],
    trust_store: Option<&Path>,
    profiles: &[flare_helios::runtime_core::Publisher],
    progress: Option<flare_helios::core::progress::AcquisitionProgress>,
) -> Result<flare_helios::core::cache::Entry> {
    flare::hub::download_watched(
        source,
        &options.into(),
        gpu_count,
        token,
        publisher_keys,
        trust_store,
        profiles,
        progress,
    )
}
pub fn refresh_model(
    store: &flare_helios::core::cache::Root,
    entry: flare_helios::core::cache::Entry,
    pinned: bool,
    args: &crate::cli::Run,
) -> Result<(flare_helios::core::cache::Entry, Option<String>)> {
    flare::hub::refresh_model(
        store,
        entry,
        pinned,
        &(&args.hub).into(),
        args.gpus.len().max(1),
        &args.publisher_keys,
        args.trust_store.as_deref(),
    )
}
#[cfg(test)]
#[path = "hub_fixtures.rs"]
pub(crate) mod tests;

#[cfg(all(test, unix))]
pub fn remember_download(
    root: &Path,
    repository: &str,
    revision: &str,
    commit: &str,
    filename: &str,
    options: &HubOptions,
    gpu_count: usize,
    entry: &flare_helios::core::cache::Entry,
    expected_envelope: [u8; 32],
) -> Result<()> {
    flare::hub::remember_download(
        root,
        repository,
        revision,
        commit,
        filename,
        &options.into(),
        gpu_count,
        entry,
        expected_envelope,
    )
}
