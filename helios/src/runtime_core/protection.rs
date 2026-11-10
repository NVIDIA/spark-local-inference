/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Publisher signatures and acceptance-bound access to executable resources.
//!
//! Signature verification and publisher acceptance are important to user safety because
//! model cores contain executable GPU code. Signatures authenticate core contents
//! against a signing key; they do not sandbox GPU code.
//! Resolving who a signing key belongs to is the caller's task, from a `CoreClaim`.

use super::{archive::Archive, format::invalid, protobuf::wire};
use crate::Result;
use crate::os::FileReadAt;
use crate::core::{protection as payload_auth, reader as payload_reader};
use prost::Message;
use ring::{aead, digest, hkdf, signature};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

pub(super) const CHUNK: usize = 4 << 20;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Publisher {
    pub name: String,
    pub namespace: String,
    pub public_key: [u8; 32],
    #[serde(default)]
    pub development: bool,
}
impl Publisher {
    pub fn fingerprint(&self) -> String {
        hex(&sha256(&self.public_key))
    }
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty()
            || self.name.len() > 256
            || self.name.chars().any(char::is_control)
            || !namespace_valid(&self.namespace)
            || self.public_key == [0; 32]
            || (self.development && self.namespace != "local-development")
        {
            return Err(invalid("invalid publisher profile"));
        }
        Ok(())
    }
}
pub(super) fn namespace_valid(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
/// The official source is a repository under the publisher's Hugging Face namespace, or the
/// literal `local-development` marker for development cores.
pub(super) fn source_valid(namespace: &str, development: bool, source: &str) -> bool {
    if development {
        return namespace == "local-development" && source == "local-development";
    }
    let Some(repository) = source.strip_prefix(&format!("https://huggingface.co/{namespace}/"))
    else {
        return false;
    };
    !repository.is_empty()
        && repository.len() <= 96
        && repository.as_bytes()[0].is_ascii_alphanumeric()
        && repository
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}
/// Links are printed to a terminal before acceptance: HTTPS, bounded, and printable ASCII only.
fn link_valid(link: &str) -> bool {
    link.len() <= 512 && link.starts_with("https://") && link.bytes().all(|b| b.is_ascii_graphic())
}

/// The license governing use of the model, as the publisher recorded it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreLicense {
    pub name: String,
    /// Where the license text is published; empty when the publisher recorded none.
    pub link: String,
}
impl CoreLicense {
    fn validated(license: &wire::License) -> Result<Self> {
        if license.name.is_empty()
            || license.name.len() > 128
            || license.name.chars().any(char::is_control)
            || !(license.link.is_empty() || link_valid(&license.link))
        {
            return Err(invalid("invalid core license metadata"));
        }
        Ok(Self {
            name: license.name.clone(),
            link: license.link.clone(),
        })
    }
}
/// What a core claims before any signature is verified: which key signed it and where it says
/// it was published. The caller resolves the signer's identity from the claim, then verifies the
/// core with the resolved profile; nothing here is trusted until that verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreClaim {
    /// Lowercase SHA-256 fingerprint of the signing key.
    pub signer: String,
    pub namespace: String,
    pub official_source: String,
    pub development: bool,
}
impl CoreClaim {
    pub fn read(path: &Path) -> Result<Self> {
        // Dispatch on the container format rather than requiring callers to know which core
        // generation they hold. Both formats expose the same untrusted claim fields.
        let primary = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.to_owned()
        };
        if crate::core::reader::declared_format(&primary)?
            .is_some_and(|format| format == crate::core::container::PAYLOAD_FORMAT)
        {
            let metadata = payload_reader::Metadata::open(path)?;
            let claim = payload_auth::Claim::read(&metadata.envelope)?;
            return Ok(Self {
                signer: claim.signer,
                namespace: claim.namespace,
                official_source: claim.official_source,
                development: claim.development,
            });
        }
        let (_, _, _, bytes) = Archive::envelope(path)?;
        let envelope = wire::ProtectedIndex::decode(bytes.as_slice())
            .map_err(|e| invalid(format!("invalid signed envelope: {e}")))?;
        ExecutableEncoding::from_version(envelope.version)?;
        if envelope.signer.len() != 32 || envelope.groups.is_empty() {
            return Err(invalid(
                "invalid signed core envelope",
            ));
        }
        let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())
            .map_err(|e| invalid(format!("invalid signed metadata: {e}")))?;
        if !namespace_valid(&metadata.namespace)
            || !source_valid(
                &metadata.namespace,
                metadata.development,
                &metadata.official_source,
            )
        {
            return Err(invalid(
                "core publisher namespace or official source mismatch",
            ));
        }
        Ok(Self {
            signer: hex(&envelope.signer),
            namespace: metadata.namespace,
            official_source: metadata.official_source,
            development: metadata.development,
        })
    }
}

pub(super) fn sha256(bytes: &[u8]) -> Vec<u8> {
    digest::digest(&digest::SHA256, bytes).as_ref().to_vec()
}
pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn fields(values: &[&[u8]]) -> Vec<u8> {
    let mut result = Vec::new();
    for value in values {
        result.extend_from_slice(&(value.len() as u64).to_le_bytes());
        result.extend_from_slice(value);
    }
    result
}
#[derive(Clone, Copy)]
pub(super) enum ExecutableEncoding {
    EncryptedV4,
    CompressedV6,
}
impl ExecutableEncoding {
    pub(super) fn from_version(version: u32) -> Result<Self> {
        match version {
            4 => Ok(Self::EncryptedV4),
            6 => Ok(Self::CompressedV6),
            _ => Err(invalid(format!(
                "unsupported signed core format {version}"
            ))),
        }
    }
    fn signature_message(self, core: &[u8], signer: &[u8], kind: u32, body: &[u8]) -> Vec<u8> {
        // Signed-core authentication domains remain byte-exact across identifier changes.
        let (domain, version): (&[u8], u32) = match self {
            Self::EncryptedV4 => (b"flare-package-signature-v4", 4),
            Self::CompressedV6 => (b"flare-package-signature-v6", 6),
        };
        fields(&[
            domain,
            &version.to_le_bytes(),
            core,
            signer,
            &kind.to_le_bytes(),
            body,
        ])
    }
}

pub struct LegacyVerified {
    pub(super) archive: Archive,
    pub(super) index: wire::ModelIndex,
    publisher: Publisher,
    official_source: String,
}
pub struct AuthorizedCore {
    pub(super) kind: AuthorizedKind,
}

impl AuthorizedCore {
    /// Canonical model ID followed by serving aliases from the signed serving contract.
    pub fn model_names(&self) -> Result<Vec<&str>> {
        let contract = match &self.kind {
            AuthorizedKind::Legacy { index, .. } => index.contract.as_ref(),
            AuthorizedKind::Payload(core) => core.source.authorized.verified()
                .deployment().and_then(|deployment| deployment.contract.as_ref()),
        }.ok_or_else(|| invalid("core has no serving contract"))?;
        Ok(std::iter::once(contract.model_id.as_str())
            .chain(contract.served_names.iter().map(String::as_str)).collect())
    }

    /// SHA-256 of the exact signed envelope bytes this copy verified.
    pub fn envelope_digest(&self) -> [u8; 32] {
        match &self.kind {
            AuthorizedKind::Legacy { archive, .. } => archive.envelope_digest(),
            AuthorizedKind::Payload(core) => core.envelope_digest(),
        }
    }

    /// The legacy model index when this core uses the compatibility format.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn legacy_index(&self) -> Option<&wire::ModelIndex> {
        match &self.kind {
            AuthorizedKind::Legacy { index, .. } => Some(index),
            AuthorizedKind::Payload(_) => None,
        }
    }
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn legacy_archive(&self) -> Option<&Archive> {
        match &self.kind {
            AuthorizedKind::Legacy { archive, .. } => Some(archive),
            AuthorizedKind::Payload(_) => None,
        }
    }
}

pub(super) enum AuthorizedKind {
    Legacy {
        archive: Archive,
        index: wire::ModelIndex,
    },
    Payload(PayloadAuthorizedCore),
}

/// One read surface over whichever backend authorized this core. Every model-loading stage
/// takes this type, so the loader is written once.
pub(super) enum FilesRef {
    Legacy(Archive),
    Payload(PayloadSource),
}

/// Distinct companion containers a closure needs, plus the primary. Opens are refused before any
/// companion is opened when the process limit cannot cover them.
fn open_file_budget(current: u64, required: u64, limit: u64) -> Result<()> {
    if current.saturating_add(required) > limit {
        return Err(invalid(format!(
            "selected deployment needs {required} more core files, with {current} already open, but the process open-file limit is {limit}"
        )));
    }
    Ok(())
}

/// Descriptors this process currently holds, when the platform can report them.
fn open_descriptors() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_dir("/proc/self/fd")
            .ok()
            .map(|entries| entries.filter_map(|entry| entry.ok()).count() as u64)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn preflight_open_files(
    metadata: &payload_reader::Metadata,
    ids: &std::collections::BTreeSet<Vec<u8>>,
) -> Result<()> {
    let mut files = std::collections::BTreeSet::new();
    files.insert(String::new());
    for location in &metadata.locations.locations {
        if ids.contains(&location.payload_id) {
            files.insert(location.file.clone());
        }
    }
    // Each selected companion needs one opened container and one direct-I/O handle, and every
    // payload inside it shares that handle.
    let required = (files.len() as u64).saturating_mul(2);
    #[cfg(unix)]
    {
        let mut limit: libc::rlimit = unsafe { std::mem::zeroed() };
        // SAFETY: getrlimit writes into the provided struct.
        if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } == 0 {
            // Count what the process already holds rather than assuming fixed headroom.
            let current = open_descriptors().unwrap_or(0);
            return open_file_budget(current, required, limit.rlim_cur as u64);
        }
        Ok(())
    }
    #[cfg(windows)]
    {
        Ok(())
    }
}

/// Resolve a prepared deployment. An unrequested size is accepted only when the core holds one
/// deployment; a requested size must match exactly one.
fn select_deployment_index(available: &[u32], requested: Option<u32>) -> Result<usize> {
    if available.is_empty() {
        return Err(invalid("core holds no prepared deployment"));
    }
    let matches: Vec<usize> = match requested {
        None => (0..available.len()).collect(),
        Some(size) => available
            .iter()
            .enumerate()
            .filter(|(_, value)| **value == size)
            .map(|(index, _)| index)
            .collect(),
    };
    match (requested, matches.as_slice()) {
        (None, [index]) => Ok(*index),
        (None, _) => Err(invalid(format!(
            "core holds {} deployments; select a tensor-parallel size (available: {available:?})",
            available.len()
        ))),
        (Some(_size), [index]) => Ok(*index),
        (Some(size), []) => Err(invalid(format!(
            "core has no tensor-parallel size {size} deployment (available: {available:?})"
        ))),
        (Some(size), _) => Err(invalid(format!(
            "core holds more than one tensor-parallel size {size} deployment"
        ))),
    }
}

fn select_deployment(
    verified: &payload_auth::VerifiedCore,
    tensor_parallel: Option<u32>,
) -> Result<usize> {
    let available = (0..verified.deployment_count())
        .map(|index| {
            verified
                .deployment_body(index)
                .map(|body| body.tensor_parallel)
        })
        .collect::<Result<Vec<_>>>()?;
    select_deployment_index(&available, tensor_parallel)
}

/// Position of a prepared rank id inside the verified rank list. A single-rank selection stores
/// one core, so the original rank id must be mapped, not used as an index.
pub(super) fn payload_rank_position(
    ranks: &[crate::core::schema::wire::RankBody],
    rank: usize,
) -> Result<usize> {
    ranks
        .iter()
        .position(|body| body.rank as usize == rank)
        .ok_or_else(|| invalid("selected rank is not part of the verified deployment"))
}

impl PayloadSource {
    pub(super) fn rank_position(&self, rank: usize) -> Result<usize> {
        payload_rank_position(self.authorized.verified().ranks(), rank)
    }
    fn token_span(&self, use_: &crate::core::schema::wire::PayloadUse) -> Result<wire::Span> {
        let slice = use_
            .slice
            .as_ref()
            .ok_or_else(|| invalid("payload use has no slice"))?;
        let placement = self
            .placements
            .get(&slice.payload_id)
            .ok_or_else(|| invalid("payload has no physical placement"))?;
        Ok(wire::Span {
            filename: placement.token.clone(),
            offset: placement
                .offset
                .checked_add(slice.offset)
                .ok_or_else(|| invalid("payload span overflow"))?,
            nbytes: slice.nbytes,
        })
    }
}

impl FilesRef {
    pub(super) fn rank_position(&self, rank: usize) -> Result<usize> {
        match self {
            FilesRef::Legacy(archive) => archive.protection.position(rank),
            FilesRef::Payload(source) => source.rank_position(rank),
        }
    }
}

impl super::payload::Files for FilesRef {
    fn span<'a>(&self, core: &'a super::format::Core, key: &str) -> Result<&'a wire::Span> {
        match self {
            FilesRef::Legacy(archive) => super::payload::Files::span(archive, core, key),
            FilesRef::Payload(source) => super::payload::Files::span(source, core, key),
        }
    }
    fn source(&self, span: &wire::Span) -> Result<(&super::archive::PayloadFile, u64)> {
        match self {
            FilesRef::Legacy(archive) => super::payload::Files::source(archive, span),
            FilesRef::Payload(source) => super::payload::Files::source(source, span),
        }
    }
    fn read(&self, core: &super::format::Core, key: &str) -> Result<Vec<u8>> {
        match self {
            FilesRef::Legacy(archive) => super::payload::Files::read(archive, core, key),
            FilesRef::Payload(source) => super::payload::Files::read(source, core, key),
        }
    }
    fn records<'a>(
        &'a self,
        core: &super::format::Core,
        key: &str,
    ) -> Result<super::payload::RecordReader<'a>> {
        match self {
            FilesRef::Legacy(archive) => super::payload::Files::records(archive, core, key),
            FilesRef::Payload(source) => super::payload::Files::records(source, core, key),
        }
    }
    fn path(&self) -> &Path {
        match self {
            FilesRef::Legacy(archive) => super::payload::Files::path(archive),
            FilesRef::Payload(source) => super::payload::Files::path(source),
        }
    }
    fn source_index(&self, span: &wire::Span) -> Result<usize> {
        match self {
            FilesRef::Legacy(archive) => super::payload::Files::source_index(archive, span),
            FilesRef::Payload(source) => super::payload::Files::source_index(source, span),
        }
    }
    fn asset_path(&self, span: &wire::Span) -> Result<PathBuf> {
        match self {
            FilesRef::Legacy(archive) => super::payload::Files::asset_path(archive, span),
            FilesRef::Payload(source) => super::payload::Files::asset_path(source, span),
        }
    }
    fn release_page_cache(&self) {
        match self {
            FilesRef::Legacy(archive) => super::payload::Files::release_page_cache(archive),
            FilesRef::Payload(source) => super::payload::Files::release_page_cache(source),
        }
    }
    fn direct_files(&self) -> Box<dyn Iterator<Item = Result<Arc<File>>> + '_> {
        match self {
            FilesRef::Legacy(archive) => super::payload::Files::direct_files(archive),
            FilesRef::Payload(source) => super::payload::Files::direct_files(source),
        }
    }
    fn retained_files(&self) -> Result<Vec<Arc<File>>> {
        match self {
            FilesRef::Legacy(archive) => super::payload::Files::retained_files(archive),
            FilesRef::Payload(source) => super::payload::Files::retained_files(source),
        }
    }
    fn validate(&self, core: &super::format::Core) -> Result<()> {
        match self {
            FilesRef::Legacy(archive) => super::payload::Files::validate(archive, core),
            FilesRef::Payload(source) => super::payload::Files::validate(source, core),
        }
    }
    fn resource_reader<'a>(
        &'a self,
        core: &super::format::Core,
        key: &str,
    ) -> Result<Box<dyn Read + 'a>> {
        match self {
            FilesRef::Legacy(archive) => {
                super::payload::Files::resource_reader(archive, core, key)
            }
            FilesRef::Payload(source) => {
                super::payload::Files::resource_reader(source, core, key)
            }
        }
    }
}
fn publisher_profiles(publishers: &[Publisher]) -> Result<Vec<Publisher>> {
    let mut profiles = super::publishers::embedded()?;
    for publisher in publishers {
        publisher.validate()?;
        if let Some(previous) = profiles
            .iter()
            .find(|p| p.public_key == publisher.public_key)
        {
            if previous != publisher {
                return Err(invalid("conflicting identities for the same publisher key"));
            }
        } else {
            profiles.push(publisher.clone());
        }
    }
    Ok(profiles)
}

/// Authenticate exact envelope bytes against publisher profiles without opening payload files.
/// The payload length is the signed logical primary-file extent.
pub fn verify_core_metadata(
    bytes: &[u8],
    publishers: &[Publisher],
    payload_length: u64,
) -> Result<(wire::ModelIndex, Publisher)> {
    let profiles = publisher_profiles(publishers)?;
    let (protection, index) = Protection::verify(bytes, &profiles, payload_length, None)?;
    Ok((index, protection.publisher))
}

impl LegacyVerified {
    pub(super) fn verified_rank(&self) -> Option<usize> {
        self.archive.protection.rank
    }

    /// Authenticate the core's index and publisher binding, for every rank or only `rank` of
    /// a multi-host core, which a copy holding that rank's bytes alone verifies completely.
    pub fn open(path: &Path, publishers: &[Publisher], rank: Option<usize>) -> Result<Self> {
        let profiles = publisher_profiles(publishers)?;
        let (archive, index) = Archive::open(path, &profiles, rank)?;
        let publisher = archive.protection.publisher.clone();
        let official_source = archive.protection.official_source.clone();
        Ok(Self {
            archive,
            index,
            publisher,
            official_source,
        })
    }
    pub fn publisher(&self) -> &Publisher {
        &self.publisher
    }
    pub fn official_source(&self) -> &str {
        &self.official_source
    }
    pub fn license(&self) -> Option<&CoreLicense> {
        self.archive.protection.license.as_ref()
    }
    /// SHA-256 of the exact signed envelope bytes this copy verified.
    pub fn envelope_digest(&self) -> [u8; 32] {
        self.archive.envelope_digest()
    }
    /// Digest the stored data of every verified rank against the signed index.
    pub fn verify_stored_data(&self) -> Result<()> {
        self.archive.protection.verify_data(&self.archive)
    }
    /// Require explicit acceptance of the authenticated publisher namespace.
    pub fn authorize(mut self, accepted: &str, verify_weights: bool) -> Result<AuthorizedCore> {
        if accepted != self.archive.protection.publisher.namespace {
            return Err(invalid("publisher namespace acceptance does not match"));
        }
        if verify_weights {
            self.archive.protection.verify_data(&self.archive)?;
        }
        self.archive.protection.accepted = Some(accepted.to_owned());
        Ok(AuthorizedCore {
            kind: AuthorizedKind::Legacy {
                archive: self.archive,
                index: self.index,
            },
        })
    }
}

/// An authenticated prepared core, whichever on-disk format it uses.
pub enum VerifiedCore {
    Legacy(LegacyVerified),
    Payload(PayloadVerifiedCore),
}

impl VerifiedCore {
    /// Read-only identity of this core's selected deployment.
    ///
    /// Extracted from verified typed metadata after signature verification, before
    /// any publisher acceptance or executable decoding, so acquisition can match
    /// a plan's model, tensor-parallel size, and signed cluster topology.
    pub fn identity(&self) -> Result<super::model::CoreIdentity> {
        match self {
            Self::Payload(core) => {
                let (model, hardware, tensor_parallel, architecture) = core.identity_parts()?;
                Ok(super::model::CoreIdentity {
                    model,
                    hardware,
                    tensor_parallel,
                    architecture,
                    payload_format: true,
                    cluster: core.cluster_identity(),
                })
            }
            Self::Legacy(core) => super::model::legacy_identity(&core.index),
        }
    }
    /// SHA-256 of the exact signed envelope bytes this copy verified.
    pub fn envelope_digest(&self) -> [u8; 32] {
        match self {
            Self::Legacy(core) => core.envelope_digest(),
            Self::Payload(core) => core.envelope_digest(),
        }
    }

    /// Open a core for one rank. Payload cores must be unambiguous: exactly one deployment,
    /// and either an explicit rank or a core holding a single rank.
    pub fn open(path: &Path, publishers: &[Publisher], rank: Option<usize>) -> Result<Self> {
        match crate::runtime_core::NativeResources::core_format(path)? {
            crate::runtime_core::CoreFormat::Legacy => {
                Ok(Self::Legacy(LegacyVerified::open(path, publishers, rank)?))
            }
            crate::runtime_core::CoreFormat::Payload => {
                let rank = rank
                    .map(|rank| {
                        u32::try_from(rank)
                            .map_err(|_| invalid("rank id exceeds the core rank domain"))
                    })
                    .transpose()?;
                let metadata = payload_reader::Metadata::open(path)?;
                let verified = payload_auth::verify(
                    &metadata.envelope,
                    publishers,
                    payload_auth::Selection::Catalog,
                )?;
                let deployment = select_deployment(&verified, None)?;
                Self::open_selected(
                    path,
                    publishers,
                    payload_auth::Selection::Deployment { deployment, rank },
                )
            }
        }
    }
    /// Open one deployment selected by its tensor-parallel size. Ambiguous or absent sizes are
    /// rejected rather than resolved by position.
    pub fn open_tensor_parallel(
        path: &Path,
        publishers: &[Publisher],
        tensor_parallel: u32,
        rank: Option<u32>,
    ) -> Result<Self> {
        if tensor_parallel == 0 {
            return Err(invalid("tensor-parallel size must be positive"));
        }
        let metadata = payload_reader::Metadata::open(path)?;
        let verified = payload_auth::verify(
            &metadata.envelope,
            publishers,
            payload_auth::Selection::Catalog,
        )?;
        let deployment = select_deployment(&verified, Some(tensor_parallel))?;
        Self::open_selected(
            path,
            publishers,
            payload_auth::Selection::Deployment { deployment, rank },
        )
    }
    /// Number of authenticated ranks: legacy index rank count or the selected
    /// payload deployment's ranks.
    pub fn rank_count(&self) -> usize {
        match self {
            Self::Legacy(core) => core.index.ranks.len(),
            Self::Payload(core) => core.verified.ranks().len(),
        }
    }
    pub fn open_selected(
        path: &Path,
        publishers: &[Publisher],
        selection: payload_auth::Selection,
    ) -> Result<Self> {
        Ok(Self::Payload(PayloadVerifiedCore::open(
            path, publishers, selection,
        )?))
    }
    pub fn publisher(&self) -> &Publisher {
        match self {
            Self::Legacy(core) => core.publisher(),
            Self::Payload(core) => core.publisher(),
        }
    }
    pub fn official_source(&self) -> &str {
        match self {
            Self::Legacy(core) => core.official_source(),
            Self::Payload(core) => core.official_source(),
        }
    }
    pub fn license(&self) -> Option<&CoreLicense> {
        match self {
            Self::Legacy(core) => core.license(),
            Self::Payload(core) => core.license_ref(),
        }
    }
    pub fn verify_stored_data(&self) -> Result<()> {
        self.verify_stored_data_with_progress(None)
    }

    /// Report hashed bytes to `phase` where the on-disk format can count them. Legacy cores
    /// hash through one archive pass and keep their own progress reporting.
    pub fn verify_stored_data_with_progress(
        &self,
        phase: Option<&crate::startup::Phase>,
    ) -> Result<()> {
        match self {
            Self::Legacy(core) => core.verify_stored_data(),
            Self::Payload(core) => core.verify_stored_data_with_progress(phase),
        }
    }
    pub fn authorize(self, accepted: &str, verify_weights: bool) -> Result<AuthorizedCore> {
        self.authorize_with_progress(accepted, verify_weights, true, None)
    }

    /// Authorize signed metadata without reading payload bytes. A controller that mirrors only a
    /// node-held core's index leaves payload verification to the serving ranks.
    pub fn authorize_metadata(
        self,
        accepted: &str,
        verify_weights: bool,
    ) -> Result<AuthorizedCore> {
        self.authorize_with_progress(accepted, verify_weights, false, None)
    }

    /// Authorize the core while reporting mandatory and requested stored-data verification.
    pub fn authorize_with_progress(
        self,
        accepted: &str,
        verify_weights: bool,
        verify_payloads: bool,
        progress: Option<&crate::startup::StartupProgress>,
    ) -> Result<AuthorizedCore> {
        match self {
            Self::Legacy(core) => {
                let verify = verify_weights && verify_payloads;
                let phase = progress
                    .filter(|_| verify)
                    .map(|progress| progress.phase("Verify stored data", None));
                let authorized = core.authorize(accepted, verify)?;
                if let Some(phase) = phase {
                    phase.finish("stored-data contents");
                }
                Ok(authorized)
            }
            Self::Payload(core) => Ok(AuthorizedCore {
                kind: AuthorizedKind::Payload(core.authorize_with_progress(
                    accepted,
                    verify_weights,
                    verify_payloads,
                    progress,
                )?),
            }),
        }
    }
    pub fn describe(
        &self,
        options: crate::runtime_core::InspectionOptions,
    ) -> Result<serde_json::Value> {
        match self {
            Self::Legacy(core) => core.describe(options),
            Self::Payload(core) => core.describe(options),
        }
    }
}

pub(super) struct Protection {
    publisher: Publisher,
    official_source: String,
    license: Option<CoreLicense>,
    /// The explicitly accepted publisher namespace.
    accepted: Option<String>,
    encoding: ExecutableEncoding,
    core_id: Vec<u8>,
    resources: BTreeMap<u64, (bool, wire::ProtectedResource)>,
    /// Stored-data ranges: what every rank shares, then each verified rank's in index order.
    data: Vec<wire::StoredData>,
    /// The one rank of a multi-host core verified, whose bundle alone the index then holds.
    rank: Option<usize>,
}
/// Initializer detail for memory-map inspection: destination placement plus the
/// payload source range and its container location.
fn initializer_detail(
    body: &crate::core::schema::wire::RankBody,
    locations: &crate::core::schema::wire::LocationSet,
) -> Vec<serde_json::Value> {
    let placed: std::collections::BTreeMap<Vec<u8>, (&str, &str)> = locations
        .locations
        .iter()
        .map(|location| {
            (
                location.payload_id.clone(),
                (location.file.as_str(), location.tensor.as_str()),
            )
        })
        .collect();
    let mut detail = Vec::new();
    for initializer in &body.initializers {
        let Some(slice) = initializer
            .source
            .as_ref()
            .and_then(|use_| use_.slice.as_ref())
        else {
            continue;
        };
        if slice.nbytes == 0 {
            continue;
        }
        let (file, tensor) = placed.get(&slice.payload_id).copied().unwrap_or(("", ""));
        detail.push(serde_json::json!({
            "allocation": initializer.allocation,
            "destination_offset": initializer.destination_offset,
            "payload_id": hex(&slice.payload_id),
            "source_offset": slice.offset,
            "nbytes": slice.nbytes,
            "file": file,
            "tensor": tensor,
        }));
    }
    detail
}

fn initializer_spans(
    body: &crate::core::schema::wire::RankBody,
) -> std::collections::BTreeMap<String, Vec<wire::Span>> {
    let mut spans: std::collections::BTreeMap<String, Vec<wire::Span>> = Default::default();
    for initializer in &body.initializers {
        let Some(slice) = initializer
            .source
            .as_ref()
            .and_then(|use_| use_.slice.as_ref())
        else {
            continue;
        };
        if slice.nbytes == 0 {
            continue;
        }
        spans
            .entry(initializer.allocation.clone())
            .or_default()
            .push(wire::Span {
                offset: initializer.destination_offset,
                nbytes: slice.nbytes,
                filename: String::new(),
            });
    }
    spans
}

fn add_catalog(
    resources: &mut BTreeMap<u64, (bool, wire::ProtectedResource)>,
    catalog: wire::ResourceCatalog,
    executable: bool,
    encoding: ExecutableEncoding,
    payload_size: u64,
) -> Result<()> {
    for entry in catalog.resources {
        let maximum = crate::core::schema::MAX_EXECUTABLE_CHUNKS * CHUNK as u64;
        let (count, stored_size_valid) = match (executable, encoding) {
            (true, ExecutableEncoding::EncryptedV4) => {
                let count = entry.plain_bytes.div_ceil(CHUNK as u64);
                let stored = count
                    .checked_mul(16)
                    .and_then(|tags| entry.plain_bytes.checked_add(tags));
                (count, stored == Some(entry.stored_bytes))
            }
            (true, ExecutableEncoding::CompressedV6) => {
                (entry.stored_bytes.div_ceil(CHUNK as u64), true)
            }
            (false, _) => (1, entry.stored_bytes == entry.plain_bytes),
        };
        if entry.plain_bytes == 0
            || entry.stored_bytes == 0
            || entry.plain_bytes > maximum
            || entry.stored_bytes > maximum
            || !stored_size_valid
            || count != entry.sha256.len() as u64
            || entry.sha256.iter().any(|h| h.len() != 32)
            || entry
                .offset
                .checked_add(entry.stored_bytes)
                .is_none_or(|end| end > payload_size)
            || resources
                .insert(entry.offset, (executable, entry))
                .is_some()
        {
            return Err(invalid("invalid protected resource catalog"));
        }
    }
    Ok(())
}

fn stored_data(
    mut data: wire::StoredData,
    payload_size: u64,
    version: u32,
) -> Result<wire::StoredData> {
    if data.sha256.len() != 32 {
        return Err(invalid("invalid stored-data signature group"));
    }
    if data.parts.is_empty() {
        if data
            .offset
            .checked_add(data.nbytes)
            .is_none_or(|end| end > payload_size)
        {
            return Err(invalid("invalid stored-data signature group"));
        }
        if data.nbytes > 0 {
            data.parts.push(wire::Span {
                offset: data.offset,
                nbytes: data.nbytes,
                filename: String::new(),
            });
        }
    } else if data.offset != 0
        || data
            .parts
            .iter()
            .try_fold(0u64, |sum, part| sum.checked_add(part.nbytes))
            != Some(data.nbytes)
    {
        return Err(invalid("invalid stored-data parts"));
    }
    for part in &data.parts {
        if !part.filename.is_empty() {
            super::archive::validate_filename(&part.filename)?;
            if version != 5 {
                return Err(invalid("companion references require model index 5"));
            }
        }
        if part.nbytes == 0
            || part
                .offset
                .checked_add(part.nbytes)
                .is_none_or(|end| part.filename.is_empty() && end > payload_size)
        {
            return Err(invalid("invalid stored-data part"));
        }
    }
    Ok(data)
}

impl Protection {
    /// Authenticate the envelope: what every rank shares, then every rank, or only `rank` of a
    /// multi-host core, whose host holds that rank's bytes alone.
    pub fn verify(
        bytes: &[u8],
        publishers: &[Publisher],
        payload_size: u64,
        rank: Option<usize>,
    ) -> Result<(Self, wire::ModelIndex)> {
        let envelope = wire::ProtectedIndex::decode(bytes)
            .map_err(|e| invalid(format!("invalid signed envelope: {e}")))?;
        let encoding = ExecutableEncoding::from_version(envelope.version)?;
        if envelope.core_id.len() != 32
            || envelope.signer.len() != 32
            || envelope.groups.len() != 3
            || envelope.ranks.is_empty()
        {
            return Err(invalid(
                "invalid signed core envelope",
            ));
        }
        let publisher = publishers
            .iter()
            .find(|p| sha256(&p.public_key) == envelope.signer)
            .ok_or_else(|| {
                invalid(
                    "unknown core publisher; supply a matching publisher verification profile",
                )
            })?
            .clone();
        publisher.validate()?;
        let authenticate = |kind: u32, group: &wire::SignedGroup, label: String| -> Result<()> {
            if group.kind != kind {
                return Err(invalid("missing, duplicate, or unordered signature group"));
            }
            signature::UnparsedPublicKey::new(&signature::ED25519, publisher.public_key)
                .verify(
                    &encoding.signature_message(
                        &envelope.core_id,
                        &envelope.signer,
                        group.kind,
                        &group.body,
                    ),
                    &group.signature,
                )
                .map_err(|_| invalid(format!("publisher signature failed for {label}")))
        };
        for (i, group) in envelope.groups.iter().enumerate() {
            authenticate(i as u32 + 1, group, format!("group {}", i + 1))?;
        }
        let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())
            .map_err(|e| invalid(format!("invalid signed metadata: {e}")))?;
        if metadata.group_hashes.len() != 2
            || metadata
                .group_hashes
                .iter()
                .zip(&envelope.groups[1..])
                .any(|(hash, group)| *hash != sha256(&group.body))
        {
            return Err(invalid("signature group commitment mismatch"));
        }
        if metadata.namespace != publisher.namespace
            || metadata.development != publisher.development
            || !source_valid(
                &publisher.namespace,
                publisher.development,
                &metadata.official_source,
            )
        {
            return Err(invalid(
                "core publisher namespace or official source mismatch",
            ));
        }
        // Cores exported before the field existed carry no license; it stays absent.
        let license = metadata
            .license
            .as_ref()
            .map(CoreLicense::validated)
            .transpose()?;
        let mut index = metadata
            .index
            .ok_or_else(|| invalid("missing signed model index"))?;
        let count = metadata.ranks as usize;
        if !matches!(index.version, 4 | 5)
            || !index.ranks.is_empty()
            || envelope.ranks.len() != count
            || index
                .cluster
                .as_ref()
                .is_some_and(|c| c.world as usize != count)
        {
            return Err(invalid("unsupported model index"));
        }
        let verified = match rank {
            None => 0..count,
            Some(rank) if index.cluster.is_some() && rank < count => rank..rank + 1,
            Some(rank) => {
                return Err(invalid(format!(
                    "rank {rank} is not a separately served rank of this core"
                )));
            }
        };
        let mut resources = BTreeMap::new();
        let shared = wire::ResourceCatalog::decode(envelope.groups[1].body.as_slice())
            .map_err(|e| invalid(e.to_string()))?;
        add_catalog(&mut resources, shared, false, encoding, payload_size)?;
        let shared = wire::StoredData::decode(envelope.groups[2].body.as_slice())
            .map_err(|e| invalid(e.to_string()))?;
        let mut data = vec![stored_data(shared, payload_size, index.version)?];
        for rank in verified {
            let group = &envelope.ranks[rank];
            authenticate(4, group, format!("rank {rank}"))?;
            let signed = wire::RankProtection::decode(group.body.as_slice())
                .map_err(|e| invalid(format!("invalid signed rank: {e}")))?;
            let (Some(bundle), Some(stored)) = (signed.bundle, signed.data) else {
                return Err(invalid("incomplete signed rank"));
            };
            if signed.rank as usize != rank {
                return Err(invalid("signed ranks out of order"));
            }
            add_catalog(
                &mut resources,
                signed.executables.unwrap_or_default(),
                true,
                encoding,
                payload_size,
            )?;
            add_catalog(
                &mut resources,
                signed.support.unwrap_or_default(),
                false,
                encoding,
                payload_size,
            )?;
            data.push(stored_data(stored, payload_size, index.version)?);
            index.ranks.push(bundle);
        }
        let protection = Self {
            publisher,
            official_source: metadata.official_source,
            license,
            accepted: None,
            encoding,
            core_id: envelope.core_id,
            resources,
            data,
            rank,
        };
        protection.validate_membership(&index, payload_size)?;
        Ok((protection, index))
    }
    /// The index position of `rank`'s bundle: the rank itself, or the only one held when a host
    /// verified its rank alone.
    pub fn position(&self, rank: usize) -> Result<usize> {
        match self.rank {
            None => Ok(rank),
            Some(only) if only == rank => Ok(0),
            Some(only) => Err(invalid(format!(
                "only rank {only} of this core was verified"
            ))),
        }
    }
    /// Payload ranges of the stored data that serving the bundle at `position` reads: what every
    /// rank shares, then that rank's.
    pub fn stored_ranges(&self, position: usize) -> Vec<wire::Span> {
        [&self.data[0], &self.data[position + 1]]
            .into_iter()
            .flat_map(|range| range.parts.iter().cloned())
            .collect()
    }
    fn validate_membership(&self, index: &wire::ModelIndex, payload_size: u64) -> Result<()> {
        let ranges: Vec<&wire::Span> = self.data.iter().flat_map(|range| &range.parts).collect();
        let within = |range: &wire::StoredData, span: &wire::Span| -> Result<()> {
            if !range.parts.iter().any(|part| {
                span.filename == part.filename
                    && span.offset >= part.offset
                    && span
                        .offset
                        .checked_add(span.nbytes)
                        .is_some_and(|end| end <= part.offset + part.nbytes)
            }) {
                return Err(invalid("stored allocation outside its data range"));
            }
            Ok(())
        };
        let mut spans: Vec<(u32, &wire::Span)> = Vec::new();
        for (position, rank) in index.ranks.iter().enumerate() {
            let mut roles = HashMap::new();
            let mut add = |name: &str, kind: u32| -> Result<()> {
                if !rank.resources.contains_key(name)
                    || roles
                        .insert(name.to_owned(), kind)
                        .is_some_and(|old| old != kind)
                {
                    return Err(invalid("missing or conflicting resource classification"));
                }
                Ok(())
            };
            for module in &rank.modules {
                add(&module.file, 2)?;
                for global in &module.globals {
                    add(&global.file, 3)?;
                }
            }
            for graph in &rank.graphs {
                add(&graph.resource, 2)?;
            }
            for allocation in &rank.allocations {
                if allocation
                    .data_spans
                    .iter()
                    .any(|span| !span.filename.is_empty())
                {
                    return Err(invalid("allocation destination spans cannot name a file"));
                }
                if let Some(file) = &allocation.file {
                    add(file, 4)?;
                }
            }
            if roles.len() != rank.resources.len() {
                return Err(invalid("unclassified core resource"));
            }
            for (name, span) in &rank.resources {
                if roles[name] == 4 {
                    within(&self.data[position + 1], span)?;
                }
                spans.push((roles[name], span));
            }
        }
        for asset in &index.assets {
            let span = asset
                .payload
                .as_ref()
                .ok_or_else(|| invalid("missing asset range"))?;
            let (_, resource) = self
                .resources
                .get(&span.offset)
                .ok_or_else(|| invalid("unsigned asset"))?;
            if resource.sha256.len() != 1 || asset.sha256 != hex(&resource.sha256[0]) {
                return Err(invalid("asset digest differs from publisher catalog"));
            }
            spans.push((3, span));
        }
        for (owner, span) in super::engram::spans(index, self.rank)? {
            let position = match owner {
                None => 0,
                Some(rank) => self.position(rank)? + 1,
            };
            let data = self
                .data
                .get(position)
                .ok_or_else(|| invalid("embedding partition owner is absent"))?;
            within(data, span)?;
            spans.push((4, span));
        }
        spans.sort_by_key(|(_, s)| (&s.filename, s.offset));
        let mut end = 0;
        let mut protected = 0;
        let mut filename = "";
        let mut primary_end = 0;
        for (kind, span) in spans {
            if span.filename != filename {
                end = 0;
                filename = &span.filename;
            }
            if !span.filename.is_empty() {
                super::archive::validate_filename(&span.filename)?;
                if index.version != 5 || kind != 4 {
                    return Err(invalid(
                        "only stored data in model index 5 may reference a companion",
                    ));
                }
            }
            if span.nbytes == 0 || span.offset < end {
                return Err(invalid("empty or overlapping core resources"));
            }
            end = span
                .offset
                .checked_add(span.nbytes)
                .filter(|n| !span.filename.is_empty() || *n <= payload_size)
                .ok_or_else(|| invalid("resource exceeds core"))?;
            if span.filename.is_empty() {
                primary_end = end;
            }
            if kind != 4 {
                let (executable, entry) = self
                    .resources
                    .get(&span.offset)
                    .ok_or_else(|| invalid("unsigned executable or supporting resource"))?;
                if *executable != (kind == 2) || entry.stored_bytes != span.nbytes {
                    return Err(invalid("resource signature classification mismatch"));
                }
                if ranges.iter().any(|range| {
                    range.filename == span.filename
                        && span.offset < range.offset + range.nbytes
                        && range.offset < end
                }) {
                    return Err(invalid(
                        "executable or supporting resource inside stored data",
                    ));
                }
                protected += 1;
            }
        }
        let mut ordered = ranges.clone();
        ordered.sort_by_key(|range| (&range.filename, range.offset));
        if ordered.windows(2).any(|pair| {
            pair[1].filename == pair[0].filename && pair[1].offset < pair[0].offset + pair[0].nbytes
        }) {
            return Err(invalid("overlapping stored-data ranges"));
        }
        // A host verifying one rank cannot see the others' resources, which complete the payload.
        if protected != self.resources.len() || (self.rank.is_none() && primary_end != payload_size)
        {
            return Err(invalid("unreferenced core resources"));
        }
        Ok(())
    }
    pub fn data_parts(&self) -> impl Iterator<Item = &wire::Span> {
        self.data.iter().flat_map(|data| &data.parts)
    }
    fn verify_data(&self, archive: &Archive) -> Result<()> {
        let mut block = vec![0; CHUNK];
        for range in &self.data {
            let mut hash = digest::Context::new(&digest::SHA256);
            for part in &range.parts {
                let mut pos = 0;
                while pos < part.nbytes {
                    let count = (part.nbytes - pos).min(block.len() as u64) as usize;
                    archive.read_logical(part, pos, &mut block[..count])?;
                    hash.update(&block[..count]);
                    pos += count as u64;
                }
            }
            if hash.finish().as_ref() != range.sha256 {
                return Err(invalid("stored-data content verification failed"));
            }
        }
        Ok(())
    }
    pub fn reader(&self, file: Arc<File>, start: u64, span: &wire::Span) -> Result<ResourceReader> {
        let (executable, entry) = self
            .resources
            .get(&span.offset)
            .ok_or_else(|| invalid("resource lacks publisher authentication"))?;
        if !span.filename.is_empty() || span.nbytes != entry.stored_bytes {
            return Err(invalid("resource length mismatch"));
        }
        if self.accepted.as_deref() != Some(self.publisher.namespace.as_str()) {
            return Err(invalid(
                "publisher acceptance required before resource access",
            ));
        }
        let inner = if *executable && matches!(self.encoding, ExecutableEncoding::EncryptedV4) {
            ResourceContents::Encrypted(EncryptedResourceReader {
                file,
                start,
                entry: entry.clone(),
                core_id: self.core_id.clone(),
                key: encrypted_resource_key(
                    &self.publisher.public_key,
                    &self.core_id,
                    entry.offset,
                    self.publisher.namespace.as_str(),
                )?,
                chunk: 0,
                buffer: Vec::new(),
                position: 0,
            })
        } else if *executable {
            ResourceContents::Compressed(payload_auth::open_compressed_reader(
                PinnedSlice::new(&file, start, entry.stored_bytes)?,
                entry.stored_bytes,
                entry.plain_bytes,
                &entry.sha256,
            )?)
        } else {
            let mut bytes = vec![0; super::protobuf::size(entry.stored_bytes)?];
            file.read_exact_at(&mut bytes, start)
                .map_err(|error| error.to_string())?;
            if entry.sha256.len() != 1 || sha256(&bytes) != entry.sha256[0] {
                return Err(invalid("publisher resource content verification failed"));
            }
            ResourceContents::Plain(io::Cursor::new(bytes))
        };
        Ok(ResourceReader {
            length: entry.plain_bytes,
            inner,
        })
    }
}

enum ResourceContents {
    Encrypted(EncryptedResourceReader),
    Compressed(payload_auth::ResourceReader<PinnedSlice>),
    Plain(io::Cursor<Vec<u8>>),
}

pub(super) struct ResourceReader {
    length: u64,
    inner: ResourceContents,
}
impl ResourceReader {
    pub fn len(&self) -> u64 {
        self.length
    }
}
impl Read for ResourceReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        match &mut self.inner {
            ResourceContents::Encrypted(reader) => reader.read(output),
            ResourceContents::Compressed(reader) => reader.read(output),
            ResourceContents::Plain(reader) => reader.read(output),
        }
    }
}

fn encrypted_resource_key(
    public_key: &[u8],
    core: &[u8],
    offset: u64,
    namespace: &str,
) -> Result<aead::LessSafeKey> {
    let info = fields(&[
        b"flare-executable-v3",
        &offset.to_le_bytes(),
        namespace.as_bytes(),
    ]);
    let prk = hkdf::Salt::new(hkdf::HKDF_SHA256, core).extract(public_key);
    let parts = [info.as_slice()];
    let mut key = [0; 32];
    prk.expand(&parts, hkdf::HKDF_SHA256)
        .map_err(|_| invalid("derive executable key"))?
        .fill(&mut key)
        .map_err(|_| invalid("derive executable key"))?;
    Ok(aead::LessSafeKey::new(
        aead::UnboundKey::new(&aead::AES_256_GCM, &key)
            .map_err(|_| invalid("invalid executable key"))?,
    ))
}
fn encrypted_aad(core: &[u8], offset: u64, size: u64, chunk: u64) -> Vec<u8> {
    fields(&[
        b"flare-executable-chunk-v3",
        core,
        &offset.to_le_bytes(),
        &size.to_le_bytes(),
        &chunk.to_le_bytes(),
    ])
}

struct EncryptedResourceReader {
    file: Arc<File>,
    start: u64,
    entry: wire::ProtectedResource,
    core_id: Vec<u8>,
    key: aead::LessSafeKey,
    chunk: usize,
    buffer: Vec<u8>,
    position: usize,
}
impl EncryptedResourceReader {
    fn refill(&mut self) -> Result<bool> {
        if self.chunk == self.entry.sha256.len() {
            return Ok(false);
        }
        let plain_offset = self.chunk as u64 * CHUNK as u64;
        let count = (self.entry.plain_bytes - plain_offset).min(CHUNK as u64) + 16;
        self.buffer.resize(super::protobuf::size(count)?, 0);
        self.file
            .read_exact_at(
                &mut self.buffer,
                self.start + plain_offset + self.chunk as u64 * 16,
            )
            .map_err(|e| e.to_string())?;
        if sha256(&self.buffer) != self.entry.sha256[self.chunk] {
            return Err(invalid("publisher resource content verification failed"));
        }
        let key = &self.key;
        let mut nonce = [0; 12];
        nonce[4..].copy_from_slice(&(self.chunk as u64).to_be_bytes());
        let length = key
            .open_in_place(
                aead::Nonce::assume_unique_for_key(nonce),
                aead::Aad::from(encrypted_aad(
                    &self.core_id,
                    self.entry.offset,
                    self.entry.plain_bytes,
                    self.chunk as u64,
                )),
                &mut self.buffer,
            )
            .map_err(|_| invalid("executable decryption failed"))?
            .len();
        self.buffer.truncate(length);
        self.chunk += 1;
        self.position = 0;
        Ok(true)
    }
}
impl Read for EncryptedResourceReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if self.position == self.buffer.len()
            && !self.refill().map_err(|e| io::Error::other(e.to_string()))?
        {
            return Ok(0);
        }
        let n = output.len().min(self.buffer.len() - self.position);
        output[..n].copy_from_slice(&self.buffer[self.position..self.position + n]);
        self.position += n;
        Ok(n)
    }
}

#[cfg(test)]
#[path = "protection/legacy_tests.rs"]
mod legacy_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialized_primary_spans_and_stored_ranges_keep_their_legacy_meaning() {
        let span = wire::Span::decode(&b"\x08\x09\x10\x07"[..]).unwrap();
        assert_eq!(
            (span.offset, span.nbytes, span.filename.as_str()),
            (9, 7, "")
        );
        let mut bytes = vec![8, 7, 18, 32];
        bytes.extend([0; 32]);
        bytes.extend([24, 9]);
        let data = wire::StoredData::decode(bytes.as_slice()).unwrap();
        assert!(data.parts.is_empty());
        let normalized = stored_data(data.clone(), 16, 4).unwrap();
        assert_eq!(normalized.parts, vec![span]);
        assert!(stored_data(data, 15, 4).is_err());
        let empty = wire::StoredData {
            sha256: sha256(&[]).to_vec(),
            ..Default::default()
        };
        assert!(stored_data(empty, 0, 4).unwrap().parts.is_empty());
    }
}

// ------------------------------------------------- payload-format core loading

use crate::core::schema::wire as payload_wire;

/// Bounded reader over one pinned container range. Reads through the file handle the core
/// reader opened, so a payload or resource slice is consumed without materializing its container.
pub(super) struct PinnedSlice {
    file: Arc<File>,
    start: u64,
    len: u64,
    position: u64,
}

impl PinnedSlice {
    fn new(file: &Arc<File>, start: u64, len: u64) -> Result<Self> {
        start
            .checked_add(len)
            .ok_or_else(|| invalid("pinned slice extent overflow"))?;
        Ok(Self {
            file: file.clone(),
            start,
            len,
            position: 0,
        })
    }
}

impl Read for PinnedSlice {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let remaining = self.len - self.position;
        if remaining == 0 || output.is_empty() {
            return Ok(0);
        }
        let count = output
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        FileReadAt::read_exact_at(
            &*self.file,
            &mut output[..count],
            self.start + self.position,
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        self.position += count as u64;
        Ok(count)
    }
}

/// Read surface over payload containers the core reader opened and pinned.
///
/// Resource keys come from [`super::protobuf::payload_rank`]. Plaintext keys resolve to a byte
/// range inside one pinned payload. Executable frames are authenticated and decompressed without
/// buffering a combined executable payload.
pub struct PayloadSource {
    path: PathBuf,
    primary: Arc<File>,
    #[cfg_attr(unix, allow(dead_code))]
    relative: BTreeMap<String, String>,
    placements: BTreeMap<Vec<u8>, super::format::Placement>,
    files: BTreeMap<String, super::archive::PayloadFile>,
    executables: BTreeMap<String, payload_wire::ExecutableResource>,
    authorized: Arc<payload_auth::AuthorizedCore>,
}

const HASH_CHUNK: usize = 4 << 20;

/// Hash each payload once, spreading independent payloads across a bounded worker set. Every
/// byte is still checked; only the order of independent reads changes. `hash` returns the bytes
/// it verified so an active phase can report progress while the cache is being read.
fn verify_payload_ids(
    ids: std::collections::BTreeSet<Vec<u8>>,
    hash: impl Fn(&[u8]) -> Result<u64> + Sync,
    phase: Option<&crate::startup::Phase>,
) -> Result<()> {
    let ids: Vec<Vec<u8>> = ids.into_iter().collect();
    if ids.is_empty() {
        return Ok(());
    }
    let workers = std::thread::available_parallelism()
        .map_or(8, |count| count.get())
        .max(8)
        .min(16)
        .min(ids.len());
    let hash = |id: &[u8]| -> Result<()> {
        let bytes = hash(id)?;
        if let Some(phase) = phase {
            phase.advance_bytes(bytes);
        }
        Ok(())
    };
    if workers <= 1 {
        for id in &ids {
            hash(id)?;
        }
        return Ok(());
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let failure = std::sync::Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(id) = ids.get(index) else { break };
                if let Err(error) = hash(id) {
                    let mut slot = failure.lock().unwrap();
                    if slot.is_none() {
                        *slot = Some(error);
                    }
                    break;
                }
            });
        }
    });
    match failure.into_inner().unwrap() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

impl PayloadSource {
    fn payload_file(&self, id: &[u8]) -> Result<(&super::archive::PayloadFile, u64)> {
        let placement = self
            .placements
            .get(id)
            .ok_or_else(|| invalid("payload has no physical placement"))?;
        let file = self
            .files
            .get(&placement.token)
            .ok_or_else(|| invalid("payload container is not open"))?;
        Ok((file, placement.nbytes))
    }
    /// Hash one whole payload by streaming it in bounded chunks. Weight, state, and table planes
    /// are hashed this way, never by allocating the payload.
    pub(super) fn hash_payload(&self, id: &[u8]) -> Result<()> {
        let (file, nbytes) = self.payload_file(id)?;
        let mut context = ring::digest::Context::new(&ring::digest::SHA256);
        let mut buffer = vec![0u8; HASH_CHUNK];
        let mut offset = 0u64;
        while offset < nbytes {
            let count = usize::try_from((nbytes - offset).min(HASH_CHUNK as u64))
                .map_err(|_| invalid("payload hash length overflow"))?;
            FileReadAt::read_exact_at(&*file.file, &mut buffer[..count], file.start + offset)
                .map_err(|error| invalid(format!("read payload: {error}")))?;
            context.update(&buffer[..count]);
            offset += count as u64;
        }
        if context.finish().as_ref() != id {
            return Err(invalid("payload content verification failed"));
        }
        Ok(())
    }
    fn supporting_payload_ids(&self) -> Result<std::collections::BTreeSet<Vec<u8>>> {
        let mut ids = std::collections::BTreeSet::new();
        for rank in self.authorized.verified().ranks() {
            for module in &rank.modules {
                for global in &module.globals {
                    if let Some(slice) = global
                        .contents
                        .as_ref()
                        .and_then(|use_| use_.slice.as_ref())
                    {
                        ids.insert(slice.payload_id.clone());
                    }
                }
            }
        }
        for asset in &self.authorized.verified().core().assets {
            let Some(slice) = asset.payload.as_ref().and_then(|use_| use_.slice.as_ref()) else {
                return Err(invalid("frontend asset has no payload slice"));
            };
            if asset.sha256.as_slice() != slice.payload_id.as_slice() {
                return Err(invalid("frontend asset digest differs from its payload"));
            }
            ids.insert(slice.payload_id.clone());
        }
        Ok(ids)
    }
    /// Supporting resources and optional weight checks share one distinct payload set.
    fn verification_payloads(&self, verify_weights: bool) -> Result<payload_reader::PayloadFiles> {
        let mut ids = self.supporting_payload_ids()?;
        if verify_weights {
            for rank in self.authorized.verified().ranks() {
                for initializer in &rank.initializers {
                    if let Some(slice) = initializer
                        .source
                        .as_ref()
                        .and_then(|use_| use_.slice.as_ref())
                    {
                        ids.insert(slice.payload_id.clone());
                    }
                }
            }
            if let Some(engram) = self
                .authorized
                .verified()
                .deployment()
                .and_then(|deployment| deployment.engram.as_ref())
            {
                for table in &engram.tables {
                    for plane in [&table.weights, &table.scales] {
                        if let Some(slice) = plane.as_ref().and_then(|use_| use_.slice.as_ref()) {
                            ids.insert(slice.payload_id.clone());
                        }
                    }
                    // Only planes the verified ranks own are part of this host's closure; a
                    // rank-selective copy does not hold its siblings' partitions.
                    let verified_ranks = self.authorized.verified().ranks();
                    for shard in table
                        .shards
                        .iter()
                        .filter(|shard| verified_ranks.iter().any(|rank| rank.rank == shard.rank))
                    {
                        for plane in [&shard.weights, &shard.scales] {
                            if let Some(slice) = plane.as_ref().and_then(|use_| use_.slice.as_ref()) {
                                ids.insert(slice.payload_id.clone());
                            }
                        }
                    }
                }
            }
        }
        let mut slices = BTreeMap::new();
        for id in ids {
            let (file, nbytes) = self.payload_file(&id)?;
            slices.insert(
                id,
                crate::core::container::Slice::new(file.file.clone(), file.start, nbytes)?,
            );
        }
        Ok(payload_reader::PayloadFiles::from_slices(slices))
    }
    fn executable_reader<'a>(
        &'a self,
        resource: &'a payload_wire::ExecutableResource,
    ) -> Result<crate::core::protection::ResourceReader<PinnedSlice>> {
        let slice = resource
            .stored
            .as_ref()
            .ok_or_else(|| invalid("executable resource has no compressed bytes"))?;
        let (file, nbytes) = self.payload_file(&slice.payload_id)?;
        let end = slice
            .offset
            .checked_add(slice.nbytes)
            .ok_or_else(|| invalid("executable slice overflow"))?;
        if end > nbytes {
            return Err(invalid("executable slice exceeds its payload"));
        }
        let source = PinnedSlice::new(&file.file, file.start + slice.offset, slice.nbytes)?;
        self.authorized.open_reader(resource, source)
    }
}

impl super::payload::Files for PayloadSource {
    fn span<'a>(&self, core: &'a super::format::Core, key: &str) -> Result<&'a wire::Span> {
        core
            .resources
            .get(key)
            .ok_or_else(|| invalid(format!("missing core resource {key}")))
    }
    fn source(&self, span: &wire::Span) -> Result<(&super::archive::PayloadFile, u64)> {
        let file = self
            .files
            .get(&span.filename)
            .ok_or_else(|| invalid("payload container is not open"))?;
        let offset = file
            .start
            .checked_add(span.offset)
            .ok_or_else(|| invalid("payload offset overflow"))?;
        Ok((file, offset))
    }
    fn read(&self, core: &super::format::Core, key: &str) -> Result<Vec<u8>> {
        if let Some(resource) = self.executables.get(key) {
            // A cubin needs one contiguous plaintext image for the driver; compressed bytes are
            // streamed per chunk rather than materialized as a combined payload.
            let mut reader = self.executable_reader(resource)?;
            let mut plaintext = Vec::new();
            reader
                .read_to_end(&mut plaintext)
                .map_err(|error| invalid(format!("read executable: {error}")))?;
            return Ok(plaintext);
        }
        let span = self.span(core, key)?;
        let (file, offset) = self.source(span)?;
        let mut bytes = vec![0u8; super::protobuf::size(span.nbytes)?];
        FileReadAt::read_exact_at(&*file.file, &mut bytes, offset)
            .map_err(|error| invalid(format!("read resource: {error}")))?;
        Ok(bytes)
    }
    fn records<'a>(
        &'a self,
        core: &super::format::Core,
        key: &str,
    ) -> Result<super::payload::RecordReader<'a>> {
        if let Some(resource) = self.executables.get(key) {
            let reader = self.executable_reader(resource)?;
            return Ok(super::payload::RecordReader::Stream(Box::new(reader)));
        }
        let bytes = self.read(core, key)?;
        Ok(super::payload::RecordReader::Slice { bytes, position: 0 })
    }
    fn path(&self) -> &Path {
        &self.path
    }
    fn source_index(&self, span: &wire::Span) -> Result<usize> {
        self.files
            .keys()
            .position(|token| token == &span.filename)
            .ok_or_else(|| invalid("payload container is not open"))
    }
    fn asset_path(&self, span: &wire::Span) -> Result<PathBuf> {
        let (file, _) = self.source(span)?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            Ok(PathBuf::from(format!(
                "/proc/self/fd/{}",
                file.file.as_raw_fd()
            )))
        }
        #[cfg(windows)]
        {
            let relative = self
                .relative
                .get(&span.filename)
                .ok_or_else(|| invalid("payload container has no relative name"))?;
            // An empty relative name means the payload is stored in the primary container.
            Ok(if relative.is_empty() {
                self.path.clone()
            } else {
                self.path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .join(relative)
            })
        }
    }
    fn release_page_cache(&self) {}
    fn direct_files(&self) -> Box<dyn Iterator<Item = Result<Arc<File>>> + '_> {
        Box::new(self.files.values().map(|file| file.direct()))
    }
    fn retained_files(&self) -> Result<Vec<Arc<File>>> {
        self.files
            .values()
            .map(|payload| {
                #[cfg(windows)]
                {
                    return Ok(Arc::new(
                        crate::os::direct_read_guard(&payload.file).map_err(|error| {
                            invalid(format!("retain read-only core: {error}"))
                        })?,
                    ));
                }
                #[cfg(unix)]
                {
                    Ok(payload.file.clone())
                }
            })
            .collect()
    }
    fn resource_reader<'a>(
        &'a self,
        core: &super::format::Core,
        key: &str,
    ) -> Result<Box<dyn Read + 'a>> {
        if let Some(resource) = self.executables.get(key) {
            return Ok(Box::new(self.executable_reader(resource)?));
        }
        let span = self.span(core, key)?;
        let (file, offset) = self.source(span)?;
        Ok(Box::new(PinnedSlice::new(&file.file, offset, span.nbytes)?))
    }
    fn validate(&self, core: &super::format::Core) -> Result<()> {
        for allocation in &core.allocations {
            if allocation.is_payload_bound() {
                for (source, span) in allocation.sources.iter().zip(&allocation.data_spans) {
                    let payload = self.span(core, &source.resource)?;
                    if source.nbytes != span[1] as u64
                        || source
                            .offset
                            .checked_add(source.nbytes)
                            .is_none_or(|end| end > payload.nbytes)
                    {
                        return Err(invalid("allocation source exceeds its payload resource"));
                    }
                }
            } else if let Some(key) = &allocation.file
                && self.span(core, key)?.nbytes != allocation.stored_bytes() as u64
            {
                return Err(invalid("allocation payload size differs from its spans"));
            }
        }
        Ok(())
    }
}

/// An authenticated payload-format core with one prepared deployment adapted for the loader.
pub struct PayloadVerifiedCore {
    metadata: payload_reader::Metadata,
    verified: payload_auth::VerifiedCore,
    cores: Vec<super::format::Core>,
    placements: BTreeMap<Vec<u8>, super::format::Placement>,
    files: BTreeMap<String, super::archive::PayloadFile>,
    #[cfg_attr(unix, allow(dead_code))]
    relative: BTreeMap<String, String>,
    license: Option<CoreLicense>,
}

fn same_peer_readers(left: &[u32], right: &[u32]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

fn push_peer(
    peers: &mut Vec<wire::PeerMemory>,
    indices: &mut BTreeMap<u64, usize>,
    peer: &wire::PeerMemory,
) -> Result<()> {
    if let Some(&index) = indices.get(&peer.address) {
        let existing = &peers[index];
        if existing.owner != peer.owner
            || existing.nbytes != peer.nbytes
            || !same_peer_readers(&existing.readers, &peer.readers)
        {
            return Err(invalid("conflicting peer memory records"));
        }
    } else {
        indices.insert(peer.address, peers.len());
        peers.push(peer.clone());
    }
    Ok(())
}

fn build_descriptor<F: Fn(&crate::core::schema::wire::PayloadUse) -> Result<wire::Span>>(
    verified: &payload_auth::VerifiedCore,
    token_span: F,
) -> Result<wire::ModelIndex> {
    let core = verified.core();
    let deployment = verified
        .deployment()
        .ok_or_else(|| invalid("payload core authenticates no deployment"))?;
    let mut assets = Vec::new();
    for asset in &core.assets {
        let use_ = asset
            .payload
            .as_ref()
            .ok_or_else(|| invalid("frontend asset has no payload"))?;
        assets.push(wire::Asset {
            name: asset.name.clone(),
            payload: Some(token_span(use_)?),
            sha256: asset
                .sha256
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        });
    }
    let engram = deployment
        .engram
        .as_ref()
        .map(|engram| -> Result<wire::Engram> {
            let mut tables = Vec::new();
            for table in &engram.tables {
                let plane = |value: &Option<crate::core::schema::wire::PayloadUse>| {
                    value.as_ref().map(|use_| token_span(use_)).transpose()
                };
                tables.push(wire::EngramTable {
                    layer: table.layer,
                    rows: table.rows,
                    embedding_width: table.embedding_width,
                    scale_block: table.scale_block,
                    weights: plane(&table.weights)?,
                    scales: plane(&table.scales)?,
                    hash: table.hash.as_ref().map(|hash| wire::EngramHash {
                        method: hash.method,
                        orders: hash.orders.clone(),
                        multipliers: hash.multipliers.clone(),
                        primes: hash.primes.clone(),
                        offsets: hash.offsets.clone(),
                    }),
                    weight_buffer: table.weight_buffer.clone(),
                    scale_buffer: table.scale_buffer.clone(),
                    hash_buffer: table.hash_buffer.clone(),
                    ready_event: table.ready_event.clone(),
                    encoding: table.encoding,
                    request_buffer: table.request_buffer.clone(),
                    global_scale: table.global_scale,
                    shards: table
                        .shards
                        .iter()
                        .map(|shard| {
                            Ok(wire::EngramTableShard {
                                rank: shard.rank,
                                first_row: shard.first_row,
                                rows: shard.rows,
                                weights: shard
                                    .weights
                                    .as_ref()
                                    .map(|use_| token_span(use_))
                                    .transpose()?,
                                scales: shard
                                    .scales
                                    .as_ref()
                                    .map(|use_| token_span(use_))
                                    .transpose()?,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                });
            }
            Ok(wire::Engram {
                token_map: engram.token_map.clone(),
                pad_token_id: engram.pad_token_id,
                compressed_vocabulary: engram.compressed_vocabulary,
                tables,
            })
        })
        .transpose()?;
    let mut peers = Vec::new();
    let mut peer_indices = BTreeMap::new();
    let mut ranks = Vec::new();
    for body in verified.ranks() {
        for peer in &body.peers {
            push_peer(&mut peers, &mut peer_indices, peer)?;
        }
        // The loader descriptor describes topology and transports only; inspection
        // detail is added by the caller from the authenticated rank bodies.
        ranks.push(wire::BundleIndex {
            architecture: deployment.architecture.clone(),
            roce: body.roce.clone(),
            ..Default::default()
        });
    }
    Ok(wire::ModelIndex {
        version: 5,
        ranks,
        peer_memory: peers,
        assets,
        engram,
        sampling_defaults: core.sampling_defaults.clone(),
        frontend: core.frontend.clone(),
        cluster: deployment.cluster.clone(),
        contract: deployment.contract.clone(),
        ..Default::default()
    })
}

impl PayloadVerifiedCore {
    /// Read-only identity of the selected deployment, from verified metadata only.
    pub fn cluster_identity(&self) -> Option<super::model::ClusterIdentity> {
        self.verified
            .deployment()
            .and_then(|deployment| deployment.cluster.as_ref())
            .map(|cluster| super::model::ClusterIdentity {
                world: cluster.world,
                export: cluster.export_id.clone(),
            })
    }
    /// SHA-256 of the exact signed envelope bytes this core verified.
    pub fn envelope_digest(&self) -> [u8; 32] {
        self.verified.envelope_digest()
    }
    pub(crate) fn identity_parts(&self) -> Result<(String, String, u32, Vec<i32>)> {
        let body = self.verified.core();
        let deployment = self
            .verified
            .deployment()
            .ok_or_else(|| invalid("verified payload core has no deployment"))?;
        Ok((
            body.model_identity.clone(),
            body.hardware_identity.clone(),
            deployment.tensor_parallel,
            deployment.architecture.clone(),
        ))
    }

    fn token_span(&self, use_: &crate::core::schema::wire::PayloadUse) -> Result<wire::Span> {
        let slice = use_
            .slice
            .as_ref()
            .ok_or_else(|| invalid("payload use has no slice"))?;
        let placement = self
            .placements
            .get(&slice.payload_id)
            .ok_or_else(|| invalid("payload has no physical placement"))?;
        Ok(wire::Span {
            filename: placement.token.clone(),
            offset: placement
                .offset
                .checked_add(slice.offset)
                .ok_or_else(|| invalid("payload span overflow"))?,
            nbytes: slice.nbytes,
        })
    }

    /// Inspection descriptor: the loader descriptor enriched from authenticated rank
    /// bodies. Stored destination spans come from initializers, which are the
    /// authoritative placement for payload cores; adapted allocation records do
    /// not carry them, and a destination offset is never a payload source offset.
    fn inspection_descriptor(&self) -> Result<wire::ModelIndex> {
        let mut index = self.descriptor()?;
        for (bundle, (body, core)) in index
            .ranks
            .iter_mut()
            .zip(self.verified.ranks().iter().zip(&self.cores))
        {
            let relocations = |values: &[super::format::Relocation]| {
                values
                    .iter()
                    .map(|relocation| wire::Relocation {
                        offset: relocation.offset as u64,
                        target: Some(wire::Reference {
                            allocation: relocation.target.allocation.clone(),
                            offset: relocation.target.offset as u64,
                        }),
                        kind: match relocation.kind {
                            super::format::RelocationKind::Pointer => 0,
                            super::format::RelocationKind::TensorMap => 1,
                        },
                    })
                    .collect::<Vec<_>>()
            };
            let spans = initializer_spans(body);
            bundle.execution = core.execution.clone();
            bundle.allocations = core
                .allocations
                .iter()
                .map(|allocation| wire::Allocation {
                    name: allocation.name.clone(),
                    nbytes: allocation.nbytes as u64,
                    alignment: allocation.alignment as u64,
                    file: allocation.file.clone(),
                    data_spans: spans.get(&allocation.name).cloned().unwrap_or_default(),
                    relocations: relocations(&allocation.relocations),
                    address: allocation.address,
                    memory: match allocation.memory {
                        super::format::Memory::Device => 0,
                        super::format::Memory::MappedHost => 1,
                        super::format::Memory::Peer => 2,
                    },
                    sizing: allocation.sizing.clone(),
                })
                .collect();
            bundle.modules = core
                .modules
                .iter()
                .map(|module| wire::Module {
                    name: module.name.clone(),
                    file: module.file.clone(),
                    globals: module
                        .globals
                        .iter()
                        .map(|global| wire::Global {
                            symbol: global.symbol.clone(),
                            file: global.file.clone(),
                            nbytes: global.nbytes as u64,
                            relocations: relocations(&global.relocations),
                        })
                        .collect(),
                })
                .collect();
            bundle.functions = core
                .functions
                .iter()
                .map(|function| wire::Function {
                    module: function.module.clone(),
                    symbol: function.symbol.clone(),
                    parameter_sizes: function
                        .parameter_sizes
                        .iter()
                        .map(|size| *size as u64)
                        .collect(),
                    attributes: function
                        .attributes
                        .iter()
                        .map(|[id, value]| wire::FunctionAttribute {
                            id: *id,
                            value: *value,
                        })
                        .collect(),
                })
                .collect();
            bundle.graphs = core
                .graphs
                .iter()
                .map(|graph| wire::GraphIndex {
                    name: graph.name.clone(),
                    flags: graph.flags,
                    resource: graph.resource.clone(),
                    node_count: graph.node_count as u64,
                    edge_count: graph.edge_count as u64,
                    functions: graph.functions.iter().map(|index| *index as u64).collect(),
                })
                .collect();
            bundle.resources = core.resources.clone();
            bundle.reservations = core
                .reservations
                .iter()
                .map(|reservation| wire::Reservation {
                    address: reservation.address,
                    nbytes: reservation.nbytes as u64,
                })
                .collect();
            bundle.events = core.events.clone();
        }
        Ok(index)
    }

    /// Metadata-only descriptor for the shared loader and inspection. No acceptance or payload
    /// opening is required: every referenced payload has a placement from the selected closure.
    fn descriptor(&self) -> Result<wire::ModelIndex> {
        build_descriptor(&self.verified, |use_| self.token_span(use_))
    }

    pub fn open(
        path: &Path,
        publishers: &[Publisher],
        selection: payload_auth::Selection,
    ) -> Result<Self> {
        let metadata = payload_reader::Metadata::open(path)?;
        let verified = payload_auth::verify(&metadata.envelope, publishers, selection)?;
        let deployment = verified
            .deployment()
            .ok_or_else(|| invalid("catalog verification selects no deployment"))?;
        let ids = payload_reader::required_payloads(&verified)?;
        preflight_open_files(&metadata, &ids)?;
        let payloads = metadata.payloads(verified.catalog(), ids.iter().map(|id| id.as_slice()))?;
        let mut files = BTreeMap::new();
        let mut placements = BTreeMap::new();
        // Payloads in one packed container share a single cached direct-I/O handle.
        let mut direct: BTreeMap<usize, std::sync::Arc<std::sync::Mutex<Option<Arc<File>>>>> =
            BTreeMap::new();
        for (id, slice) in payloads.iter() {
            let token = hex(id);
            let key = std::sync::Arc::as_ptr(slice.file()) as usize;
            let handle = direct
                .entry(key)
                .or_insert_with(|| std::sync::Arc::new(std::sync::Mutex::new(None)))
                .clone();
            files.insert(
                token.clone(),
                super::archive::PayloadFile::new_shared(
                    slice.file().clone(),
                    slice.offset(),
                    slice.len(),
                    handle,
                ),
            );
            placements.insert(
                id.to_vec(),
                super::format::Placement {
                    token,
                    offset: 0,
                    nbytes: slice.len(),
                },
            );
        }
        let architecture: [i32; 2] = deployment
            .architecture
            .as_slice()
            .try_into()
            .map_err(|_| invalid("deployment architecture must name a compute capability"))?;
        let mut cores = Vec::with_capacity(verified.ranks().len());
        for rank in verified.ranks() {
            cores.push(super::protobuf::payload_rank(
                rank,
                architecture,
                &placements,
            )?);
        }
        if cores.is_empty() {
            return Err(invalid("payload core authenticates no rank"));
        }
        let mut relative = BTreeMap::new();
        for location in &metadata.locations.locations {
            relative.insert(hex(&location.payload_id), location.file.clone());
        }
        let license = verified
            .core()
            .publisher
            .as_ref()
            .filter(|publisher| !publisher.license_name.is_empty())
            .map(|publisher| CoreLicense {
                name: publisher.license_name.clone(),
                link: publisher.license_link.clone(),
            });
        if cores.is_empty() {
            return Err(invalid("payload core authenticates no rank"));
        }
        Ok(Self {
            metadata,
            verified,
            cores,
            placements,
            files,
            relative,
            license,
        })
    }
    pub fn claim(&self) -> payload_auth::Claim {
        self.verified.claim()
    }
    pub fn publisher(&self) -> &Publisher {
        self.verified.publisher()
    }
    pub fn official_source(&self) -> &str {
        self.verified
            .core()
            .publisher
            .as_ref()
            .map_or("", |publisher| publisher.official_source.as_str())
    }
    pub fn core_id(&self) -> [u8; 32] {
        self.verified.core_id()
    }
    pub fn deployment_id(&self) -> &str {
        self.verified
            .deployment()
            .map_or("", |deployment| deployment.deployment_id.as_str())
    }
    pub fn rank_ids(&self) -> Vec<u32> {
        self.verified.ranks().iter().map(|rank| rank.rank).collect()
    }
    pub fn license_ref(&self) -> Option<&CoreLicense> {
        self.license.as_ref()
    }
    pub fn license(&self) -> Option<CoreLicense> {
        let publisher = self.verified.core().publisher.as_ref()?;
        if publisher.license_name.is_empty() {
            return None;
        }
        Some(CoreLicense {
            name: publisher.license_name.clone(),
            link: publisher.license_link.clone(),
        })
    }
    /// Hash one payload by streaming it in bounded chunks, comparing with its signed descriptor.
    pub(super) fn hash_payload(&self, id: &[u8]) -> Result<()> {
        let placement = self
            .placements
            .get(id)
            .ok_or_else(|| invalid("payload has no physical placement"))?;
        let file = self
            .files
            .get(&placement.token)
            .ok_or_else(|| invalid("payload container is not open"))?;
        const CHUNK: usize = 4 << 20;
        let mut context = ring::digest::Context::new(&ring::digest::SHA256);
        let mut buffer = vec![0u8; CHUNK];
        let mut offset = 0u64;
        while offset < placement.nbytes {
            let count = super::protobuf::size((placement.nbytes - offset).min(CHUNK as u64))?;
            FileReadAt::read_exact_at(&*file.file, &mut buffer[..count], file.start + offset)
                .map_err(|error| invalid(format!("read payload: {error}")))?;
            context.update(&buffer[..count]);
            offset += count as u64;
        }
        if context.finish().as_ref() != id {
            return Err(invalid("payload content verification failed"));
        }
        Ok(())
    }
    /// Hash every stored payload the selected deployment references against its signed
    /// descriptor. This includes initializers plus selected frontend assets, module globals,
    /// offloaded table planes, and executable resources.
    pub fn verify_stored_data(&self) -> Result<()> {
        self.verify_stored_data_with_progress(None)
    }

    /// Hash the stored payloads while advancing `phase` by the bytes each payload verifies, so a
    /// launch that must rehash a cached rank reports progress instead of appearing stuck.
    pub fn verify_stored_data_with_progress(
        &self,
        phase: Option<&crate::startup::Phase>,
    ) -> Result<()> {
        verify_payload_ids(
            payload_reader::required_payloads(&self.verified)?,
            |id| {
                let placement = self
                    .placements
                    .get(id)
                    .ok_or_else(|| invalid("payload has no physical placement"))?;
                self.hash_payload(id)?;
                Ok(placement.nbytes)
            },
            phase,
        )
    }
    pub fn describe(
        &self,
        options: crate::runtime_core::InspectionOptions,
    ) -> Result<serde_json::Value> {
        let core = self.verified.core();
        let deployment = self.verified.deployment();
        // A single verified rank (a selected rank copy) keeps its own id; a full
        // deployment must let each bundle carry its own rank instead of applying
        // the first rank to every bundle.
        let selected_ranks = self.verified.ranks();
        let rank = match selected_ranks {
            [single] => Some(single.rank as usize),
            _ => None,
        };
        // The shared describer supplies the same model, cache, speculation, sampling, and frontend
        // sections a legacy core reports; the payload document adds identity and storage.
        let descriptor = self.inspection_descriptor()?;
        let mut document = super::describe::describe_index(&descriptor, options, rank)?;
        // Bundle indices carry no rank id; report the authenticated body ids, and
        // keep initializer source/destination placement distinct under memory maps.
        if let Some(entries) = document["ranks"].as_array_mut() {
            for (entry, body) in entries.iter_mut().zip(selected_ranks) {
                entry["rank"] = serde_json::json!(body.rank);
                if (options.memory_map || options.all) && entry.get("memory_map").is_some() {
                    entry["memory_map"]["initializers"] =
                        serde_json::json!(initializer_detail(body, &self.metadata.locations));
                }
            }
        }
        let primary_bytes = self
            .metadata
            .primary()
            .file()
            .metadata()
            .map_err(|error| invalid(format!("cannot stat the core: {error}")))?
            .len();
        // Only containers opened for the selected closure are companions of this copy.
        // Locations name the whole catalog, so iterating them would report unselected
        // ranks and deployments as zero-byte files. Several payloads may share one
        // opened container; its length is the same whichever payload is visited first.
        let mut companions: std::collections::BTreeMap<String, u64> = Default::default();
        for (token, file) in &self.files {
            let Some(relative) = self.relative.get(token) else {
                continue;
            };
            if relative.is_empty() || companions.contains_key(relative) {
                continue;
            }
            let length = file
                .file
                .metadata()
                .map_err(|error| invalid(format!("cannot stat companion {relative}: {error}")))?
                .len();
            companions.insert(relative.clone(), length);
        }
        let mut deployments = Vec::new();
        for index in 0..self.verified.deployment_count() {
            let body = self.verified.deployment_body(index)?;
            deployments.push(serde_json::json!({
                "id": body.deployment_id,
                "tensor_parallel": body.tensor_parallel,
                "architecture": body.architecture,
                "ranks": body.ranks.iter().map(|rank| rank.rank).collect::<Vec<_>>(),
                "selected": deployment.is_some_and(|selected| selected.deployment_id == body.deployment_id),
            }));
        }
        let payload_bytes: u64 = self
            .verified
            .catalog()
            .ids()
            .into_iter()
            .filter_map(|id| self.verified.catalog().length(id).ok())
            .sum();
        document["core"] = serde_json::json!({
            "signatures_verified": true,
            "format": "payload",
            "format_version": core.format_version,
            "core_id": hex(&self.verified.core_id()),
            "envelope_sha256": hex(&self.verified.envelope_digest()),
            "bytes": primary_bytes,
            "companions": companions
                .into_iter()
                .map(|(name, bytes)| serde_json::json!({"name": name, "bytes": bytes}))
                .collect::<Vec<_>>(),
            "ranks": self.verified.ranks().len(),
            "tensor_parallel": deployment.map(|deployment| deployment.tensor_parallel),
            "cluster": deployment.and_then(|deployment| deployment.cluster.as_ref()).map(|cluster| serde_json::json!({
                "world": cluster.world,
                "export_id": cluster.export_id,
            })),
            "publisher": {
                "name": self.verified.publisher().name,
                "namespace": self.verified.publisher().namespace,
                "fingerprint": self.verified.publisher().fingerprint(),
                "development": self.verified.publisher().development,
            },
            "official_source": core.publisher.as_ref().map(|publisher| publisher.official_source.clone()),
            "license": self.license.as_ref().map(|license| serde_json::json!({
                "name": license.name,
                "link": license.link,
            })),
        });
        document["model_identity"] = serde_json::json!(core.model_identity);
        document["hardware_identity"] = serde_json::json!(core.hardware_identity);
        document["deployments"] = serde_json::json!(deployments);
        document["payloads"] = serde_json::json!({
            "count": self.verified.catalog().len(),
            "bytes": payload_bytes,
            "assets": core.assets.iter().map(|asset| asset.name.clone()).collect::<Vec<_>>(),
        });
        document["engram_tables"] = serde_json::json!(
            deployment
                .and_then(|deployment| deployment.engram.as_ref())
                .map(|engram| engram.tables.len())
                .unwrap_or(0)
        );
        // Rank detail comes from the shared describer so payload and legacy documents
        // report the same keys, counts by default and arrays only under detail flags.
        if options.all {
            document["payload_ids"] = serde_json::json!(
                self.verified
                    .catalog()
                    .ids()
                    .into_iter()
                    .map(hex)
                    .collect::<Vec<_>>()
            );
        }
        Ok(document)
    }

    /// Record acceptance and check supporting resources, plus stored weights and tables when
    /// full verification is requested.
    pub fn authorize(
        self,
        accepted: &str,
        verify_weights: bool,
    ) -> Result<PayloadAuthorizedCore> {
        self.authorize_with_progress(accepted, verify_weights, true, None)
    }

    fn authorize_with_progress(
        self,
        accepted: &str,
        verify_weights: bool,
        verify_payloads: bool,
        progress: Option<&crate::startup::StartupProgress>,
    ) -> Result<PayloadAuthorizedCore> {
        let Self {
            metadata,
            verified,
            cores,
            placements,
            files,
            relative,
            license: _,
        } = self;
        let mut executables = BTreeMap::new();
        for rank in verified.ranks() {
            for resource in &rank.resources {
                executables.insert(
                    format!("executable/{}", resource.resource_id),
                    resource.clone(),
                );
            }
        }
        let authorized = Arc::new(verified.authorize(accepted)?);
        let source = PayloadSource {
            path: metadata.path.clone(),
            primary: metadata.primary().file().clone(),
            relative,
            placements,
            files,
            executables,
            authorized,
        };
        if verify_payloads {
            let payloads = source.verification_payloads(verify_weights)?;
            let total = payloads.iter().try_fold(0u64, |total, (_, slice)| {
                total
                    .checked_add(slice.len())
                    .ok_or_else(|| invalid("verification byte count overflow"))
            })?;
            let phase = progress.map(|progress| progress.phase("Verify stored data", Some(total)));
            payloads.verify_with_progress(&|bytes| {
                if let Some(phase) = &phase {
                    phase.advance_bytes(bytes);
                }
            })?;
            if let Some(phase) = phase {
                phase.finish(format!("{:.1} GiB", total as f64 / (1u64 << 30) as f64));
            }
        }
        Ok(PayloadAuthorizedCore { source, cores })
    }
}

impl PayloadAuthorizedCore {
    pub(super) fn cluster(&self) -> Option<wire::Cluster> {
        self.source
            .authorized
            .verified()
            .deployment()
            .and_then(|deployment| deployment.cluster.clone())
    }
    pub(super) fn reserve_bytes(&self) -> u64 {
        self.source
            .authorized
            .verified()
            .deployment()
            .and_then(|deployment| deployment.contract.as_ref())
            .and_then(|contract| contract.reserve_bytes)
            .unwrap_or(0)
    }
    pub(super) fn rank_needs(&self) -> Vec<crate::runtime_core::RankNeeds> {
        let architecture = self
            .source
            .authorized
            .verified()
            .deployment()
            .map(|deployment| {
                [
                    deployment.architecture.first().copied().unwrap_or(0),
                    deployment.architecture.get(1).copied().unwrap_or(0),
                ]
            })
            .unwrap_or_default();
        self.cores
            .iter()
            .map(|core| crate::runtime_core::RankNeeds {
                architecture,
                device_bytes: core
                    .allocations
                    .iter()
                    .filter(|allocation| matches!(allocation.memory, super::format::Memory::Device))
                    .map(|allocation| allocation.nbytes as u64)
                    .sum(),
            })
            .collect()
    }
}

/// An accepted payload core ready to load one adapted rank.
pub struct PayloadAuthorizedCore {
    source: PayloadSource,
    cores: Vec<super::format::Core>,
}

impl PayloadAuthorizedCore {
    /// Verified identity of the selected deployment: model, hardware, TP, architecture.
    /// Signed cluster topology of the authorized deployment, when declared.
    pub fn cluster_identity(&self) -> Option<super::model::ClusterIdentity> {
        self.source
            .authorized
            .verified()
            .deployment()
            .and_then(|deployment| deployment.cluster.as_ref())
            .map(|cluster| super::model::ClusterIdentity {
                world: cluster.world,
                export: cluster.export_id.clone(),
            })
    }
    /// SHA-256 of the exact signed envelope bytes this core verified.
    pub fn envelope_digest(&self) -> [u8; 32] {
        self.source.authorized.verified().envelope_digest()
    }

    pub(crate) fn identity_parts(&self) -> Result<(String, String, u32, Vec<i32>)> {
        let verified = self.source.authorized.verified();
        let body = verified.core();
        let deployment = verified
            .deployment()
            .ok_or_else(|| invalid("authorized payload core has no deployment"))?;
        Ok((
            body.model_identity.clone(),
            body.hardware_identity.clone(),
            deployment.tensor_parallel,
            deployment.architecture.clone(),
        ))
    }

    pub fn source(&self) -> &PayloadSource {
        &self.source
    }
    pub fn rank_count(&self) -> usize {
        self.cores.len()
    }
    /// Hash every stored payload the selected deployment references, streaming from pinned
    /// handles. This includes initializers plus selected frontend assets, module globals,
    /// offloaded table planes, and executable resources.
    pub fn verify_stored_data(&self) -> Result<()> {
        verify_payload_ids(
            payload_reader::required_payloads(self.source.authorized.verified())?,
            |id| {
                let (_, nbytes) = self.source.payload_file(id)?;
                self.source.hash_payload(id)?;
                Ok(nbytes)
            },
            None,
        )
    }
    pub(super) fn descriptor(&self) -> Result<wire::ModelIndex> {
        build_descriptor(self.source.authorized.verified(), |use_| {
            self.source.token_span(use_)
        })
    }
    #[cfg_attr(not(any(test, feature = "test-tools")), allow(dead_code))]
    pub(super) fn rank_cores(&self) -> &[super::format::Core] {
        &self.cores
    }
    pub(super) fn into_parts(self) -> (PayloadSource, Vec<super::format::Core>) {
        (self.source, self.cores)
    }
    /// Files a host serving `rank` needs, transferred whole because payload containers are
    /// immutable content-addressed objects. The primary always travels with the rank.
    pub fn rank_ranges(&self, rank: usize) -> Result<Vec<super::archive::CoreFile>> {
        let body = self
            .source
            .authorized
            .verified()
            .ranks()
            .iter()
            .find(|body| body.rank as usize == rank)
            .ok_or_else(|| invalid("rank is not part of the verified deployment"))?;
        let mut ids: std::collections::BTreeSet<Vec<u8>> = Default::default();
        for initializer in &body.initializers {
            if let Some(slice) = initializer
                .source
                .as_ref()
                .and_then(|use_| use_.slice.as_ref())
            {
                ids.insert(slice.payload_id.clone());
            }
        }
        for resource in &body.resources {
            if let Some(slice) = resource.stored.as_ref() {
                ids.insert(slice.payload_id.clone());
            }
        }
        for module in &body.modules {
            for global in &module.globals {
                if let Some(slice) = global
                    .contents
                    .as_ref()
                    .and_then(|use_| use_.slice.as_ref())
                {
                    ids.insert(slice.payload_id.clone());
                }
            }
        }
        for asset in &self.source.authorized.verified().core().assets {
            if let Some(slice) = asset.payload.as_ref().and_then(|use_| use_.slice.as_ref()) {
                ids.insert(slice.payload_id.clone());
            }
        }
        if let Some(engram) = self
            .source
            .authorized
            .verified()
            .deployment()
            .and_then(|deployment| deployment.engram.as_ref())
        {
            for table in &engram.tables {
                for plane in [&table.weights, &table.scales] {
                    if let Some(slice) = plane.as_ref().and_then(|use_| use_.slice.as_ref()) {
                        ids.insert(slice.payload_id.clone());
                    }
                }
                // Only the requested rank's partition belongs to this host's closure; a
                // rank-selective copy does not hold its siblings' partitions.
                for shard in table.shards.iter().filter(|shard| shard.rank == body.rank) {
                    for plane in [&shard.weights, &shard.scales] {
                        if let Some(slice) = plane.as_ref().and_then(|use_| use_.slice.as_ref()) {
                            ids.insert(slice.payload_id.clone());
                        }
                    }
                }
            }
        }
        let mut files: Vec<super::archive::CoreFile> = Vec::new();
        // Transfer uses the handle the reader opened; the path is never reopened, and the primary
        // is preseede in the dedup set because a packed primary also holds payload tensors.
        let primary = self.source.primary.clone();
        let mut seen = std::collections::BTreeSet::from([Arc::as_ptr(&primary) as usize]);
        let primary_len = primary.metadata().map_err(|error| error.to_string())?.len();
        files.push(super::archive::CoreFile {
            filename: String::new(),
            file: primary,
            ranges: vec![(0, primary_len)],
        });
        for id in ids {
            let placement = self
                .source
                .placements
                .get(&id)
                .ok_or_else(|| invalid("payload has no physical placement"))?;
            let file = self
                .source
                .files
                .get(&placement.token)
                .ok_or_else(|| invalid("payload container is not open"))?;
            if !seen.insert(Arc::as_ptr(&file.file) as usize) {
                continue;
            }
            let length = file
                .file
                .metadata()
                .map_err(|error| error.to_string())?
                .len();
            files.push(super::archive::CoreFile {
                filename: self
                    .source
                    .relative
                    .get(&placement.token)
                    .cloned()
                    .ok_or_else(|| invalid("payload container has no signed relative name"))?,
                file: file.file.clone(),
                ranges: vec![(0, length)],
            });
        }
        Ok(files)
    }
    /// Load one rank into a live CUDA context.
    ///
    /// # Safety
    /// The core must be trusted and the selected device must match its architecture.
    pub unsafe fn load(mut self, rank: usize, ordinal: i32) -> Result<super::NativeResources> {
        if rank >= self.cores.len() {
            return Err(invalid("payload rank is outside the deployment"));
        }
        let mut cores = std::mem::take(&mut self.cores);
        let core = cores.swap_remove(rank);
        unsafe { super::NativeResources::load_prepared(&self.source, core, ordinal) }
    }
}

#[cfg(test)]
mod payload_rank_tests {
    use super::*;
    use crate::core::schema::wire as payload_wire;

    fn body(rank: u32) -> payload_wire::RankBody {
        payload_wire::RankBody {
            rank,
            ..Default::default()
        }
    }

    #[test]
    fn deployment_selection_rejects_ambiguity() {
        assert_eq!(select_deployment_index(&[1], None).unwrap(), 0);
        assert!(select_deployment_index(&[1, 2], None).is_err());
        assert!(select_deployment_index(&[], None).is_err());
        assert_eq!(select_deployment_index(&[1, 2], Some(2)).unwrap(), 1);
        assert!(select_deployment_index(&[1, 2], Some(4)).is_err());
        assert!(select_deployment_index(&[2, 2], Some(2)).is_err());
    }

    #[test]
    fn open_file_budget_accounts_for_open_and_required_descriptors() {
        assert!(open_file_budget(10, 20, 64).is_ok());
        assert!(open_file_budget(10, 54, 64).is_ok());
        assert!(open_file_budget(10, 55, 64).is_err());
        assert!(open_file_budget(64, 1, 64).is_err());
    }

    #[test]
    fn selected_rank_ids_map_to_stored_positions() {
        // A rank-1-only selection stores one core; the original rank id is not its index.
        assert_eq!(payload_rank_position(&[body(1)], 1).unwrap(), 0);
        assert!(payload_rank_position(&[body(1)], 0).is_err());
        assert_eq!(payload_rank_position(&[body(0), body(1)], 1).unwrap(), 1);
        assert_eq!(payload_rank_position(&[body(2), body(1)], 1).unwrap(), 1);
        assert!(payload_rank_position(&[], 0).is_err());
    }
}

#[cfg(test)]
mod format_sniff_tests {
    use super::super::{NativeResources, CoreFormat};
    use std::io::Write;

    fn write_core(
        name: &str,
        format: &str,
        tensor_offsets: [[u64; 2]; 2],
    ) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "flare-format-{}-{name}.safetensors",
            std::process::id()
        ));
        let header = serde_json::json!({
            "__metadata__": { "format": format },
            "flare.payload": { "dtype": "U8", "shape": [0], "data_offsets": tensor_offsets[0] },
            "flare.metadata": { "dtype": "U8", "shape": [4], "data_offsets": tensor_offsets[1] },
            "flare.layout": { "dtype": "U8", "shape": [4], "data_offsets": [8, 12] },
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        path
    }

    #[test]
    fn format_detection_does_not_apply_the_payload_parser() {
        // A legacy header with metadata tensors that are not aligned still dispatches to the
        // legacy reader instead of failing the payload container parser.
        let legacy = write_core("legacy", "flare-snapshot", [[0, 0], [0, 4]]);
        assert_eq!(
            NativeResources::core_format(&legacy).unwrap(),
            CoreFormat::Legacy
        );
        let payload = write_core("payload", "flare-payload", [[0, 0], [0, 4]]);
        assert_eq!(
            NativeResources::core_format(&payload).unwrap(),
            CoreFormat::Payload
        );
        let unrelated = write_core("unknown", "something-else", [[0, 0], [0, 4]]);
        assert_eq!(
            NativeResources::core_format(&unrelated).unwrap(),
            CoreFormat::Legacy
        );
        let _ = std::fs::remove_file(legacy);
        let _ = std::fs::remove_file(payload);
        let _ = std::fs::remove_file(unrelated);
    }
}

#[cfg(test)]
mod payload_core_tests {
    use super::*;
    use crate::core::PayloadId;
    use crate::core::export::{
        self, Destination, PublishDeployment, PublishPayload, PublishRank, PublishRequest,
        PublishResource,
    };
    use crate::core::schema::wire as pw;
    use crate::runtime_core::wire::PeerMemory;
    use ring::signature::KeyPair;

    fn write(dir: &std::path::Path, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn rank_body(
        deployment: &str,
        rank: u32,
        weight_id: &[u8],
        weight_nbytes: u64,
    ) -> pw::RankBody {
        pw::RankBody {
            rank,
            deployment_id: deployment.into(),
            context: Some(crate::runtime_core::wire::Context::default()),
            reservations: vec![],
            allocations: vec![pw::Allocation {
                name: format!("device-{rank}"),
                nbytes: weight_nbytes,
                alignment: 256,
                address: Some(0x1000),
                memory: 0,
                sizing: None,
                relocations: vec![],
            }],
            initializers: vec![pw::Initializer {
                allocation: format!("device-{rank}"),
                destination_offset: 0,
                source: Some(pw::PayloadUse {
                    slice: Some(pw::PayloadSlice {
                        payload_id: weight_id.to_vec(),
                        offset: 0,
                        nbytes: weight_nbytes,
                    }),
                    role: pw::PayloadRole::Weights as i32,
                }),
            }],
            resources: vec![],
            modules: vec![pw::Module {
                name: "kernel".into(),
                cubin: 0,
                globals: vec![],
            }],
            functions: vec![pw::Function {
                module: "kernel".into(),
                symbol: "decode".into(),
                parameter_sizes: vec![],
                attributes: vec![],
            }],
            graphs: vec![pw::Graph {
                name: "decode".into(),
                flags: 0,
                resource: 1,
                node_count: 1,
                edge_count: 0,
                functions: vec![0],
            }],
            execution: Some(crate::runtime_core::wire::Execution {
                abi_version: 2,
                max_active: 4,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Publish a one-rank core whose single allocation is filled by two
    /// initializers with distinct, non-zero source and destination offsets.
    fn two_initializer_core() -> (std::path::PathBuf, Vec<Publisher>) {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("flare-two-init-{}-{unique}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Two init".into(),
            namespace: "two-init".into(),
            public_key,
            development: false,
        };

        // 16 stored bytes; initializers read (2..6) and (9..12) into the same
        // 32-byte allocation at destination offsets 5 and 12.
        let weights = (0..16u8).collect::<Vec<u8>>();
        let weights_id = PayloadId::of(&weights);
        let weights_path = write(&root, "w.prepared", &weights);
        let cubin = b"fake cubin".to_vec();
        let graph = b"fake graph".to_vec();
        let cubin_path = write(&root, "k.plain", &cubin);
        let graph_path = write(&root, "g.plain", &graph);

        let initializer =
            |source_offset: u64, destination_offset: u64, nbytes: u64| pw::Initializer {
                allocation: "device-0".into(),
                destination_offset,
                source: Some(pw::PayloadUse {
                    slice: Some(pw::PayloadSlice {
                        payload_id: weights_id.as_bytes().to_vec(),
                        offset: source_offset,
                        nbytes,
                    }),
                    role: pw::PayloadRole::Weights as i32,
                }),
            };
        let rank = pw::RankBody {
            rank: 0,
            deployment_id: "sm120-tp1".into(),
            context: Some(crate::runtime_core::wire::Context::default()),
            allocations: vec![pw::Allocation {
                name: "device-0".into(),
                nbytes: 32,
                alignment: 256,
                address: Some(0x1000),
                memory: 0,
                sizing: None,
                relocations: vec![],
            }],
            initializers: vec![initializer(2, 5, 4), initializer(9, 12, 3)],
            modules: vec![pw::Module {
                name: "kernel".into(),
                cubin: 0,
                globals: vec![],
            }],
            functions: vec![pw::Function {
                module: "kernel".into(),
                symbol: "decode".into(),
                parameter_sizes: vec![],
                attributes: vec![],
            }],
            graphs: vec![pw::Graph {
                name: "decode".into(),
                flags: 0,
                resource: 1,
                node_count: 1,
                edge_count: 0,
                functions: vec![0],
            }],
            execution: Some(crate::runtime_core::wire::Execution {
                abi_version: 2,
                max_active: 4,
                ..Default::default()
            }),
            ..Default::default()
        };
        let request = PublishRequest {
            core: pw::CoreBody {
                model_identity: "two-init/model".into(),
                hardware_identity: "sm120".into(),
                frontend: Some(crate::runtime_core::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    crate::runtime_core::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(pw::PublisherMetadata {
                    namespace: "two-init".into(),
                    official_source: "https://huggingface.co/two-init/model".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![PublishDeployment {
                payloads: vec![],
                body: pw::DeploymentBody {
                    deployment_id: "sm120-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(crate::runtime_core::wire::ServingContract {
                        model_id: "two-init/model".into(),
                        identity: "two-init".into(),
                        max_sequence_tokens: 8,
                        prefill_chunk_size: 1,
                        pages: 1,
                        state_rows: 1,
                        page_tokens: 1,
                        model_contract: Some(crate::runtime_core::wire::ModelContract {
                            vocabulary: 8,
                            graphs: vec![crate::runtime_core::wire::GraphContract {
                                graph: "decode".into(),
                                ..Default::default()
                            }],
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                ranks: vec![PublishRank {
                    body: rank,
                    payloads: vec![PublishPayload {
                        role: pw::PayloadRole::Weights,
                        relative: "weights/layer.safetensors".into(),
                        source: crate::core::export::pin_file(
                            &weights_path,
                            weights.len() as u64,
                        )
                        .unwrap(),
                        id: weights_id,
                        tensor: None,
                    }],
                    resources: vec![
                        PublishResource {
                            kind: pw::ExecutableKind::Cubin,
                            path: cubin_path,
                            nbytes: cubin.len() as u64,
                        },
                        PublishResource {
                            kind: pw::ExecutableKind::Graph,
                            path: graph_path,
                            nbytes: graph.len() as u64,
                        },
                    ],
                }],
            }],
            core_id: [31u8; 32],
            public_key,
            accepted: "two-init".into(),
            no_weights: false,
        };
        crate::core::export::publish(&request, &key, Destination::Directory(&root)).unwrap();
        (root, vec![publisher])
    }

    /// Publish one three-rank core whose rank bodies carry the supplied peer records.
    fn peer_core(
        peers: [Vec<PeerMemory>; 3],
    ) -> (std::path::PathBuf, Vec<Publisher>) {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("flare-peer-{}-{unique}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let core_id = [73u8; 32];
        let weights = b"peer rank weights".to_vec();
        let weight_id = PayloadId::of(&weights);
        let weight_len = weights.len() as u64;
        let weight_path = write(&root, "weights.prepared", &weights);
        let cubin = write(&root, "k.plain", b"fake cubin");
        let graph = write(&root, "g.plain", b"fake graph");
        let resources = || {
            vec![
                PublishResource {
                    kind: pw::ExecutableKind::Cubin,
                    path: cubin.clone(),
                    nbytes: 10,
                },
                PublishResource {
                    kind: pw::ExecutableKind::Graph,
                    path: graph.clone(),
                    nbytes: 10,
                },
            ]
        };
        let ranks = (0..3u32)
            .map(|rank| {
                let mut body = rank_body("sm120-tp3", rank, weight_id.as_bytes(), weight_len);
                body.peers = peers[rank as usize].clone();
                PublishRank {
                    body,
                    payloads: vec![PublishPayload {
                        role: pw::PayloadRole::Weights,
                        relative: format!("weights/rank-{rank}.safetensors"),
                        source: crate::core::export::pin_file(&weight_path, weight_len).unwrap(),
                        id: weight_id,
                        tensor: None,
                    }],
                    resources: resources(),
                }
            })
            .collect();
        let request = PublishRequest {
            core: pw::CoreBody {
                format_version: 1,
                core_id: core_id.to_vec(),
                model_identity: "peer-model".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(pw::PublisherMetadata {
                    namespace: "peer".into(),
                    official_source: "https://huggingface.co/peer/model".into(),
                    ..Default::default()
                }),
                frontend: Some(crate::runtime_core::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    crate::runtime_core::wire::DefaultSamplingParameters::default(),
                ),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![PublishDeployment {
                body: pw::DeploymentBody {
                    deployment_id: "sm120-tp3".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 3,
                    contract: Some(crate::runtime_core::wire::ServingContract {
                        model_contract: Some(crate::runtime_core::wire::ModelContract::default()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                ranks,
                payloads: vec![],
            }],
            core_id,
            public_key,
            accepted: "peer".into(),
            no_weights: false,
        };
        export::publish(&request, &key, Destination::Directory(&root)).unwrap();
        let publishers = vec![Publisher {
            name: "Peer".into(),
            namespace: "peer".into(),
            public_key,
            development: false,
        }];
        (root, publishers)
    }

    /// Publish a real two-deployment core: tp1 with one rank and tp2 with two, where rank 0 of
    /// tp2 owns an extra payload that rank 1 must not be asked to transfer.
    fn two_deployment_core() -> (std::path::PathBuf, Vec<Publisher>, [u8; 32]) {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("flare-two-tp-{}-{unique}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let core_id = [9u8; 32];

        let weight1 = b"rank one weights".to_vec();
        let extra = b"rank zero only".to_vec();
        let cubin = b"fake cubin".to_vec();
        let graph = b"fake graph".to_vec();
        let weight1_id = PayloadId::of(&weight1);
        let extra_id = PayloadId::of(&extra);
        let weight1_path = write(&root, "w1.prepared", &weight1);
        let extra_path = write(&root, "extra.prepared", &extra);
        let cubin_path = write(&root, "k.plain", &cubin);
        let graph_path = write(&root, "g.plain", &graph);

        let rank_payload =
            |id: PayloadId, path: &std::path::Path, nbytes: u64, name: &str| PublishPayload {
                role: pw::PayloadRole::Weights,
                relative: name.into(),
                source: crate::core::export::pin_file(path, nbytes).unwrap(),
                id,
                tensor: None,
            };
        let resources = || {
            vec![
                PublishResource {
                    kind: pw::ExecutableKind::Cubin,
                    path: cubin_path.clone(),
                    nbytes: cubin.len() as u64,
                },
                PublishResource {
                    kind: pw::ExecutableKind::Graph,
                    path: graph_path.clone(),
                    nbytes: graph.len() as u64,
                },
            ]
        };
        let deployment = |id: &str, tp: u32, ranks: Vec<PublishRank>| PublishDeployment {
            body: pw::DeploymentBody {
                deployment_id: id.into(),
                architecture: vec![12, 0],
                tensor_parallel: tp,
                contract: Some(crate::runtime_core::wire::ServingContract {
                    model_contract: Some(crate::runtime_core::wire::ModelContract::default()),
                    ..Default::default()
                }),
                cluster: None,
                rebuild: None,
                engram: None,
                payloads: vec![],
                ranks: vec![],
            },
            payloads: vec![],
            ranks,
        };
        let weight1_len = weight1.len() as u64;
        let extra_len = extra.len() as u64;
        let request = PublishRequest {
            core: pw::CoreBody {
                format_version: 1,
                core_id: core_id.to_vec(),
                model_identity: "model".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(pw::PublisherMetadata {
                    namespace: "example".into(),
                    official_source: "https://huggingface.co/example/model".into(),
                    development: false,
                    license_name: "Example".into(),
                    license_link: String::new(),
                    ranks: 0,
                }),
                frontend: Some(crate::runtime_core::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    crate::runtime_core::wire::DefaultSamplingParameters::default(),
                ),
                payloads: vec![],
                assets: vec![],
                deployments: vec![],
            },
            payloads: vec![],
            deployments: vec![
                deployment(
                    "sm120-tp1",
                    1,
                    vec![PublishRank {
                        body: rank_body("sm120-tp1", 0, weight1_id.as_bytes(), weight1_len),
                        payloads: vec![rank_payload(
                            weight1_id,
                            &weight1_path,
                            weight1_len,
                            "weights/tp1.safetensors",
                        )],
                        resources: resources(),
                    }],
                ),
                deployment(
                    "sm120-tp2",
                    2,
                    vec![
                        PublishRank {
                            body: rank_body("sm120-tp2", 0, extra_id.as_bytes(), extra_len),
                            payloads: vec![rank_payload(
                                extra_id,
                                &extra_path,
                                extra_len,
                                "weights/rank0-only.safetensors",
                            )],
                            resources: resources(),
                        },
                        PublishRank {
                            body: rank_body("sm120-tp2", 1, weight1_id.as_bytes(), weight1_len),
                            payloads: vec![rank_payload(
                                weight1_id,
                                &weight1_path,
                                weight1_len,
                                "weights/rank1.safetensors",
                            )],
                            resources: resources(),
                        },
                    ],
                ),
            ],
            core_id,
            public_key,
            accepted: "example".into(),
            no_weights: false,
        };
        export::publish(&request, &key, Destination::Directory(&root)).unwrap();
        let publishers = vec![Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        }];
        (root, publishers, core_id)
    }

    /// Publish a real TP2 core with one shared Engram table and one partitioned table.
    /// Each rank also has an initializer, module global, and graph executable.
    fn engram_core() -> (std::path::PathBuf, Vec<Publisher>) {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("flare-engram-tp-{}-{unique}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let core_id = [11u8; 32];
        let use_of = |id: PayloadId, nbytes: u64, role: pw::PayloadRole| pw::PayloadUse {
            slice: Some(pw::PayloadSlice {
                payload_id: id.as_bytes().to_vec(),
                offset: 0,
                nbytes,
            }),
            role: role as i32,
        };
        let payload = |name: &str, bytes: &[u8], role: pw::PayloadRole| {
            let path = write(&root, &name.replace('/', "-"), bytes);
            let id = PayloadId::of(bytes);
            (
                id,
                bytes.len() as u64,
                PublishPayload {
                    role,
                    relative: name.into(),
                    source: crate::core::export::pin_file(&path, bytes.len() as u64).unwrap(),
                    id,
                    tensor: None,
                },
            )
        };
        let cubin = write(&root, "k.plain", b"fake cubin");
        let graph = write(&root, "g.plain", b"fake graph");
        let resources = || {
            vec![
                PublishResource {
                    kind: pw::ExecutableKind::Cubin,
                    path: cubin.clone(),
                    nbytes: 10,
                },
                PublishResource {
                    kind: pw::ExecutableKind::Graph,
                    path: graph.clone(),
                    nbytes: 10,
                },
            ]
        };

        let (asset_id, asset_len, mut asset_payload) = payload(
            "assets/tokenizer.safetensors",
            b"tokenizer payload",
            pw::PayloadRole::Frontend,
        );
        asset_payload.relative.clear();
        asset_payload.tensor = Some("frontend.tokenizer".into());
        let mut ranks = Vec::new();
        for rank in 0..2u32 {
            let (weight_id, weight_len, weight) = payload(
                &format!("weights/rank-{rank}.safetensors"),
                format!("initializer rank {rank}").as_bytes(),
                pw::PayloadRole::Weights,
            );
            let (global_id, global_len, global) = payload(
                &format!("globals/rank-{rank}.safetensors"),
                format!("module global rank {rank}").as_bytes(),
                pw::PayloadRole::InitialState,
            );
            let mut body = rank_body("sm120-tp2", rank, weight_id.as_bytes(), weight_len);
            body.modules[0].globals.push(pw::ModuleGlobal {
                symbol: format!("global_{rank}"),
                contents: Some(use_of(global_id, global_len, pw::PayloadRole::InitialState)),
                nbytes: global_len,
                relocations: vec![],
            });
            ranks.push(PublishRank {
                body,
                payloads: vec![weight, global],
                resources: resources(),
            });
        }

        let mut deployment_payloads = Vec::new();
        let (shared_w_id, shared_w_len, shared_w) = payload(
            "engram/shared-weights.safetensors",
            b"shared-weights",
            pw::PayloadRole::Weights,
        );
        deployment_payloads.push(shared_w);
        let (shared_s_id, shared_s_len, shared_s) = payload(
            "engram/shared-scales.safetensors",
            b"shared-scales",
            pw::PayloadRole::TunedWeights,
        );
        deployment_payloads.push(shared_s);
        let mut shards = Vec::new();
        for rank in 0..2u32 {
            let (weight_id, weight_len, weight) = payload(
                &format!("engram/shard-{rank}-weights.safetensors"),
                format!("shard weights {rank}").as_bytes(),
                pw::PayloadRole::Weights,
            );
            deployment_payloads.push(weight);
            let (scale_id, scale_len, scale) = payload(
                &format!("engram/shard-{rank}-scales.safetensors"),
                format!("shard scales {rank}").as_bytes(),
                pw::PayloadRole::TunedWeights,
            );
            deployment_payloads.push(scale);
            shards.push(pw::EngramTableShard {
                rank,
                first_row: rank as u64 * 2,
                rows: 2,
                weights: Some(use_of(weight_id, weight_len, pw::PayloadRole::Weights)),
                scales: Some(use_of(scale_id, scale_len, pw::PayloadRole::TunedWeights)),
            });
        }
        let engram = pw::Engram {
            token_map: vec![1, 2],
            pad_token_id: 0,
            compressed_vocabulary: 8,
            tables: vec![
                pw::EngramTable {
                    layer: 1,
                    rows: 1,
                    embedding_width: 2,
                    scale_block: 16,
                    weights: Some(use_of(shared_w_id, shared_w_len, pw::PayloadRole::Weights)),
                    scales: Some(use_of(
                        shared_s_id,
                        shared_s_len,
                        pw::PayloadRole::TunedWeights,
                    )),
                    hash: Some(pw::EngramHash {
                        method: pw::engram_hash::Method::XorModulo as i32,
                        ..Default::default()
                    }),
                    weight_buffer: "shared_w".into(),
                    scale_buffer: "shared_s".into(),
                    hash_buffer: "shared_h".into(),
                    ready_event: "shared_ready".into(),
                    encoding: pw::engram_table::Encoding::Fp8E4m3E8m0 as i32,
                    global_scale: 1.0,
                    shards: vec![],
                    ..Default::default()
                },
                pw::EngramTable {
                    layer: 2,
                    rows: 4,
                    embedding_width: 2,
                    scale_block: 16,
                    hash: Some(pw::EngramHash {
                        method: pw::engram_hash::Method::XorModulo as i32,
                        ..Default::default()
                    }),
                    weight_buffer: "shard_w".into(),
                    scale_buffer: "shard_s".into(),
                    hash_buffer: "shard_h".into(),
                    ready_event: "shard_ready".into(),
                    request_buffer: "shard_request".into(),
                    encoding: pw::engram_table::Encoding::Fp8E4m3E8m0 as i32,
                    global_scale: 1.0,
                    shards,
                    ..Default::default()
                },
            ],
        };
        let deployment = PublishDeployment {
            body: pw::DeploymentBody {
                deployment_id: "sm120-tp2".into(),
                architecture: vec![12, 0],
                tensor_parallel: 2,
                contract: Some(crate::runtime_core::wire::ServingContract {
                    model_contract: Some(crate::runtime_core::wire::ModelContract::default()),
                    ..Default::default()
                }),
                engram: Some(engram),
                ..Default::default()
            },
            payloads: deployment_payloads,
            ranks,
        };
        let request = PublishRequest {
            core: pw::CoreBody {
                format_version: 1,
                core_id: core_id.to_vec(),
                model_identity: "engram-model".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(pw::PublisherMetadata {
                    namespace: "example".into(),
                    official_source: "https://huggingface.co/example/model".into(),
                    development: false,
                    license_name: "Example".into(),
                    license_link: String::new(),
                    ranks: 0,
                }),
                frontend: Some(crate::runtime_core::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    crate::runtime_core::wire::DefaultSamplingParameters::default(),
                ),
                assets: vec![pw::FrontendAsset {
                    name: "tokenizer".into(),
                    payload: Some(use_of(asset_id, asset_len, pw::PayloadRole::Frontend)),
                    sha256: asset_id.as_bytes().to_vec(),
                }],
                ..Default::default()
            },
            payloads: vec![asset_payload],
            deployments: vec![deployment],
            core_id,
            public_key,
            accepted: "example".into(),
            no_weights: false,
        };
        export::publish(&request, &key, Destination::Directory(&root)).unwrap();
        let publishers = vec![Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        }];
        (root, publishers)
    }

    fn peer(readers: Vec<u32>) -> PeerMemory {
        PeerMemory {
            owner: 0,
            address: 0x1000,
            nbytes: b"peer rank weights".len() as u64,
            readers,
        }
    }

    fn descriptor_peers_with_rank(
        peers: [Vec<PeerMemory>; 3],
        rank: Option<u32>,
    ) -> (Vec<PeerMemory>, Vec<u32>) {
        let (root, publishers) = peer_core(peers);
        let selected = PayloadVerifiedCore::open(
            &root,
            &publishers,
            payload_auth::Selection::Deployment {
                deployment: 0,
                rank,
            },
        )
        .unwrap();
        let rank_ids = selected.rank_ids();
        let descriptor = selected.descriptor().unwrap();
        drop(selected);
        let _ = std::fs::remove_dir_all(root);
        (descriptor.peer_memory, rank_ids)
    }

    fn descriptor_peers(peers: [Vec<PeerMemory>; 3]) -> Vec<PeerMemory> {
        descriptor_peers_with_rank(peers, None).0
    }

    #[test]
    fn peer_descriptor_coalesces_owner_inclusive_records_by_reader_membership() {
        let peers = descriptor_peers([
            vec![peer(vec![2, 1, 0])],
            vec![peer(vec![0, 1, 2])],
            vec![peer(vec![0, 1, 2])],
        ]);
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].owner, 0);
        assert_eq!(peers[0].nbytes, b"peer rank weights".len() as u64);
        let mut readers = peers[0].readers.clone();
        readers.sort_unstable();
        assert_eq!(readers, vec![0, 1, 2]);
    }

    #[test]
    fn peer_descriptor_keeps_reader_only_imports_without_renumbering() {
        let (peers, rank_ids) = descriptor_peers_with_rank(
            [
                vec![],
                vec![peer(vec![0, 1, 2])],
                vec![peer(vec![0, 1, 2])],
            ],
            Some(2),
        );
        assert_eq!(rank_ids, vec![2]);
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].owner, 0);
        assert_eq!(
            peers[0]
                .readers
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([0, 1, 2])
        );
    }

    #[test]
    fn peer_descriptor_rejects_same_address_conflicts() {
        for peers in [
            [vec![peer(vec![0, 1, 2])], vec![PeerMemory { owner: 1, ..peer(vec![0, 1, 2]) }], vec![]],
            [vec![peer(vec![0, 1, 2])], vec![PeerMemory { nbytes: 1, ..peer(vec![0, 1, 2]) }], vec![]],
            [vec![peer(vec![0, 1, 2])], vec![peer(vec![0, 1])], vec![]],
        ] {
            let (root, publishers) = peer_core(peers);
            let selected = PayloadVerifiedCore::open(
                &root,
                &publishers,
                payload_auth::Selection::Deployment {
                    deployment: 0,
                    rank: None,
                },
            )
            .unwrap();
            let error = selected.descriptor().unwrap_err();
            assert!(error.to_string().contains("conflicting peer memory records"), "{error}");
            drop(selected);
            let _ = std::fs::remove_dir_all(root);
        }
    }

    fn corrupt_payload_companion(path: &std::path::Path) {
        use crate::core::container::{self, COMPANION_TENSOR, Container, Source, Writer};
        let container = Container::open(path).unwrap();
        let slice = container.tensor(COMPANION_TENSOR).unwrap();
        let mut bytes = vec![0; slice.len() as usize];
        slice.read_at(0, &mut bytes).unwrap();
        bytes[0] ^= 0xff;
        let temporary = path.with_extension("corrupt");
        let mut output = std::fs::File::create(&temporary).unwrap();
        let mut writer = Writer::new(&mut output, container::PAYLOAD_FORMAT).unwrap();
        writer
            .add(COMPANION_TENSOR, bytes.len() as u64, Source::Bytes(bytes))
            .unwrap();
        writer.finish().unwrap();
        output.sync_all().unwrap();
        std::fs::rename(temporary, path).unwrap();
    }

    fn corrupt_primary_tensor(path: &std::path::Path, tensor: &str) {
        use crate::core::container::{self, Container, Source, Writer};
        let path = path.join("model.safetensors");
        let container = Container::open(&path).unwrap();
        let mut bytes = vec![0; container.tensor(tensor).unwrap().len() as usize];
        container
            .tensor(tensor)
            .unwrap()
            .read_at(0, &mut bytes)
            .unwrap();
        bytes[0] ^= 0xff;
        let temporary = path.with_extension("corrupt");
        let mut output = std::fs::File::create(&temporary).unwrap();
        let mut writer = Writer::new(&mut output, container::PAYLOAD_FORMAT).unwrap();
        for name in container.tensor_names() {
            if name == tensor || name.starts_with(container::ALIGNMENT_PREFIX) {
                continue;
            }
            let slice = container.tensor(name).unwrap();
            writer.add(name, slice.len(), Source::Slice(slice)).unwrap();
        }
        writer
            .add(tensor, bytes.len() as u64, Source::Bytes(bytes))
            .unwrap();
        writer.finish().unwrap();
        output.sync_all().unwrap();
        std::fs::rename(temporary, path).unwrap();
    }

    #[test]
    fn payload_stored_data_hashes_every_selected_role_and_respects_rank_scope() {
        use crate::core::reader::Metadata;
        for role in ["frontend", "global", "engram", "executable"] {
            let (root, publishers) = engram_core();
            let baseline =
                VerifiedCore::open_tensor_parallel(&root, &publishers, 2, None).unwrap();
            baseline.verify_stored_data().unwrap();
            drop(baseline);
            let metadata = Metadata::open(&root).unwrap();
            let needle = match role {
                "frontend" => "assets/tokenizer",
                "global" => "globals/rank-0",
                "engram" => "engram/shared-weights",
                "executable" => "deployments/sm120-tp2/rank-0/graphs",
                _ => unreachable!(),
            };
            let location = metadata
                .locations
                .locations
                .iter()
                .find(|location| {
                    if role == "frontend" {
                        location.file.is_empty() && location.tensor == "frontend.tokenizer"
                    } else {
                        location.file.contains(needle)
                    }
                })
                .unwrap_or_else(|| panic!("missing {role} location"));
            if location.file.is_empty() {
                corrupt_primary_tensor(&root, &location.tensor);
            } else {
                corrupt_payload_companion(&root.join(&location.file));
            }
            let core =
                VerifiedCore::open_tensor_parallel(&root, &publishers, 2, None).unwrap();
            assert!(
                core.verify_stored_data().is_err(),
                "corrupt {role} payload was accepted"
            );
            let _ = std::fs::remove_dir_all(root);
        }

        // A selected rank must not require another rank's private payload bytes.
        let (root, publishers) = engram_core();
        let baseline =
            VerifiedCore::open_tensor_parallel(&root, &publishers, 2, Some(1)).unwrap();
        baseline.verify_stored_data().unwrap();
        drop(baseline);
        let metadata = Metadata::open(&root).unwrap();
        let location = metadata
            .locations
            .locations
            .iter()
            .find(|location| location.file.contains("globals/rank-0"))
            .unwrap();
        corrupt_payload_companion(&root.join(&location.file));
        let rank_one =
            VerifiedCore::open_tensor_parallel(&root, &publishers, 2, Some(1)).unwrap();
        rank_one.verify_stored_data().unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn authorization_checks_supporting_resources_and_optional_weight_and_table_bytes() {
        for (name, optional) in [
            ("frontend.tokenizer", false),
            ("globals/rank-1", false),
            ("engram/shared-weights", true),
            ("engram/shared-scales", true),
            ("engram/shard-1-weights", true),
            ("engram/shard-1-scales", true),
            ("weights/rank-1", true),
        ] {
            for verify_weights in [false, true] {
                let (root, publishers) = engram_core();
                let metadata = payload_reader::Metadata::open(&root).unwrap();
                let location = metadata.locations.locations.iter().find(|location| {
                    location.file.contains(name) || location.tensor == name
                }).unwrap();
                if location.file.is_empty() {
                    corrupt_primary_tensor(&root, &location.tensor);
                } else {
                    corrupt_payload_companion(&root.join(&location.file));
                }
                let core = VerifiedCore::open_tensor_parallel(&root, &publishers, 2, Some(1)).unwrap();
                let result = core.authorize_with_progress(
                    "example",
                    verify_weights,
                    true,
                    Some(&crate::startup::StartupProgress::silent()),
                );
                assert_eq!(result.is_err(), !optional || verify_weights, "{name}, verify_weights={verify_weights}");
                drop(result);
                drop(metadata);
                let _ = std::fs::remove_dir_all(root);
            }
        }

        let (root, publishers) = engram_core();
        for name in ["globals/rank-0", "engram/shard-0-weights", "weights/rank-0"] {
            let metadata = payload_reader::Metadata::open(&root).unwrap();
            let location = metadata.locations.locations.iter().find(|location| location.file.contains(name)).unwrap();
            corrupt_payload_companion(&root.join(&location.file));
        }
        let core = VerifiedCore::open_tensor_parallel(&root, &publishers, 2, Some(1)).unwrap();
        let authorized = core.authorize("example", true).unwrap();
        drop(authorized);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn engram_rank_closure_keeps_shared_planes_and_only_the_requested_shard() {
        let (root, publishers) = engram_core();
        let selected = VerifiedCore::open_tensor_parallel(&root, &publishers, 2, None).unwrap();
        let authorized = selected.authorize("example", false).unwrap();
        let AuthorizedKind::Payload(payload) = &authorized.kind else {
            panic!("expected the payload backend");
        };
        let files = payload.rank_ranges(1).unwrap();
        let names: Vec<&str> = files.iter().map(|file| file.filename.as_str()).collect();
        assert!(
            names.contains(&"engram/shared-weights.safetensors"),
            "{names:?}"
        );
        assert!(
            names.contains(&"engram/shared-scales.safetensors"),
            "{names:?}"
        );
        assert!(
            names.contains(&"engram/shard-1-weights.safetensors"),
            "{names:?}"
        );
        assert!(
            names.contains(&"engram/shard-1-scales.safetensors"),
            "{names:?}"
        );
        assert!(
            !names.iter().any(|name| name.contains("shard-0")),
            "{names:?}"
        );
        assert!(names.contains(&"weights/rank-1.safetensors"), "{names:?}");
        assert!(names.contains(&"globals/rank-1.safetensors"), "{names:?}");
        let ranks = payload.source.authorized.verified().ranks();
        let metadata = crate::core::reader::Metadata::open(&root).unwrap();
        for (first, selected) in ranks[0].resources.iter().zip(&ranks[1].resources) {
            let shared = first.stored.as_ref().unwrap();
            let selected = selected.stored.as_ref().unwrap();
            assert_eq!(shared.payload_id, selected.payload_id);
            let location = metadata.locations.locations.iter()
                .find(|location| location.payload_id == selected.payload_id).unwrap();
            assert!(names.contains(&location.file.as_str()), "{names:?}");
        }
        assert!(!names.contains(&"weights/rank-0.safetensors"), "{names:?}");
        assert!(!names.contains(&"globals/rank-0.safetensors"), "{names:?}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn ambiguous_deployments_require_a_selection_and_rank_transfer_is_rank_local() {
        let (root, publishers, _) = two_deployment_core();
        assert!(
            VerifiedCore::open(&root, &publishers, Some(0)).is_err(),
            "a core holding several deployments must not silently pick one"
        );
        let selected =
            VerifiedCore::open_tensor_parallel(&root, &publishers, 2, Some(1)).unwrap();
        let authorized = selected.authorize("example", false).unwrap();
        let AuthorizedKind::Payload(payload) = &authorized.kind else {
            panic!("expected the payload backend");
        };
        let files = payload.rank_ranges(1).unwrap();
        let names: Vec<String> = files.iter().map(|file| file.filename.clone()).collect();
        // Rank 1's weight bytes are identical to tp1's, so the exporter deduplicated them to one
        // published relative name; what matters is that exactly one weight container travels and
        // rank 0's private payload does not.
        assert_eq!(
            names
                .iter()
                .filter(|name| name.starts_with("weights/"))
                .count(),
            1,
            "{names:?}"
        );
        assert!(
            !names.iter().any(|name| name.contains("rank0-only")),
            "rank 1 must not transfer another rank's payload: {names:?}"
        );
        // The primary travels once and carries the pinned handle, not a reopened path.
        assert_eq!(names.iter().filter(|name| name.is_empty()).count(), 1);
        assert!(
            VerifiedCore::open_tensor_parallel(&root, &publishers, 4, Some(0)).is_err(),
            "an unexported tensor-parallel size must be rejected"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn asset_paths_read_initializer_bytes_in_directory_and_packed_cores() {
        use crate::runtime_core::payload::Files;

        let (root, publishers) = two_initializer_core();
        let packed = root.with_extension("safetensors");
        crate::core::conversion::pack(&root, &packed).unwrap();
        for path in [&root, &packed] {
            let verified = VerifiedCore::open_tensor_parallel(path, &publishers, 1, None).unwrap();
            let authorized = verified.authorize("two-init", false).unwrap();
            let AuthorizedKind::Payload(payload) = &authorized.kind else {
                panic!("expected a payload core");
            };
            let core = &payload.cores[0];
            let allocation = &core.allocations[0];
            assert_eq!(allocation.sources.len(), 2);
            for (source, expected) in allocation
                .sources
                .iter()
                .zip([vec![2, 3, 4, 5], vec![9, 10, 11]])
            {
                let span = payload.source.span(core, &source.resource).unwrap();
                let (_, offset) = payload.source.source(span).unwrap();
                let file = File::open(payload.source.asset_path(span).unwrap()).unwrap();
                let mut bytes = vec![0; source.nbytes as usize];
                FileReadAt::read_exact_at(&file, &mut bytes, offset + source.offset).unwrap();
                assert_eq!(bytes, expected, "{}", path.display());
            }
        }
        std::fs::remove_file(packed).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn memory_map_reports_initializer_source_and_destination_offsets_exactly() {
        let (root, publishers) = two_initializer_core();
        let verified = VerifiedCore::open_tensor_parallel(&root, &publishers, 1, None).unwrap();
        let described = verified
            .describe(crate::runtime_core::InspectionOptions {
                memory_map: true,
                graphs: false,
                frontend: false,
                all: false,
            })
            .unwrap();
        let rank = &described["ranks"][0];
        // Two initializers feed one allocation: 4 + 3 stored bytes.
        let allocation = rank["memory_map"]["allocations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|allocation| allocation["name"] == "device-0")
            .expect("device-0 allocation");
        assert_eq!(allocation["stored_bytes"], 7, "{allocation}");
        let initializers = rank["memory_map"]["initializers"].as_array().unwrap();
        assert_eq!(initializers.len(), 2, "{initializers:?}");
        let pairs: Vec<(u64, u64, u64)> = initializers
            .iter()
            .map(|entry| {
                (
                    entry["source_offset"].as_u64().unwrap(),
                    entry["destination_offset"].as_u64().unwrap(),
                    entry["nbytes"].as_u64().unwrap(),
                )
            })
            .collect();
        assert_eq!(pairs, vec![(2, 5, 4), (9, 12, 3)]);
        for entry in initializers {
            assert_eq!(entry["allocation"], "device-0");
            assert_eq!(entry["file"], "weights/layer.safetensors");
            assert_eq!(entry["tensor"], crate::core::container::COMPANION_TENSOR);
            assert!(
                entry["payload_id"]
                    .as_str()
                    .is_some_and(|id| id.len() == 64)
            );
        }
        // Destination offsets are not payload source offsets.
        assert!(
            pairs
                .iter()
                .all(|(source, destination, _)| source != destination)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn authenticated_handles_survive_rename_and_replacement() {
        let (root, publishers, _) = two_deployment_core();
        let selected =
            VerifiedCore::open_tensor_parallel(&root, &publishers, 2, Some(1)).unwrap();
        // Inspection reports the same allocation and graph detail as the legacy document.
        let described = selected
            .describe(crate::runtime_core::InspectionOptions {
                memory_map: true,
                graphs: true,
                frontend: true,
                all: true,
            })
            .unwrap();
        // Detail lives where the shared describer puts it, not in payload-only keys.
        assert!(
            described["ranks"][0]["memory_map"]["allocations"]
                .as_array()
                .is_some()
        );
        assert!(
            described["ranks"][0]["execution_graphs"]
                .as_array()
                .is_some()
        );
        assert_eq!(described["ranks"][0]["rank"], 1);
        assert_eq!(described["ranks"][0]["max_active"], 4);
        assert!(
            described["deployments"]
                .as_array()
                .is_some_and(|list| list.len() == 2)
        );
        assert_eq!(described["core"]["signatures_verified"], true);
        let companions = described["core"]["companions"].as_array().unwrap();
        let names: Vec<&str> = companions
            .iter()
            .map(|entry| entry["name"].as_str().unwrap())
            .collect();
        assert!(
            companions
                .iter()
                .all(|entry| entry["bytes"].as_u64().is_some_and(|bytes| bytes > 0)),
            "{companions:?}"
        );
        assert!(names.contains(&"weights/tp1.safetensors"), "{names:?}");
        assert!(
            !names.iter().any(|name| name.contains("rank0-only")),
            "unselected companions must not be listed: {names:?}"
        );
        assert!(described["model"].is_object());
        assert!(described["cache"].is_object());
        assert!(described["sampling_defaults"].is_object());
        assert_eq!(described["core"]["tensor_parallel"], 2);
        // Stored bytes come from initializer destinations, not from adapted
        // allocation records that carry no destination spans.
        let allocations = described["ranks"][0]["memory_map"]["allocations"]
            .as_array()
            .unwrap();
        assert!(
            allocations
                .iter()
                .any(|allocation| allocation["stored_bytes"].as_u64().unwrap_or(0) > 0),
            "{allocations:?}"
        );
        // Initializer detail keeps the destination offset and the payload source
        // offset separate, and names the payload's container location.
        let initializers = described["ranks"][0]["memory_map"]["initializers"]
            .as_array()
            .unwrap();
        assert!(!initializers.is_empty());
        let first = &initializers[0];
        assert_eq!(first["destination_offset"], 0);
        assert_eq!(first["source_offset"], 0);
        assert!(
            first["payload_id"]
                .as_str()
                .is_some_and(|id| id.len() == 64)
        );
        assert!(
            first["file"]
                .as_str()
                .is_some_and(|file| file.starts_with("weights/")),
            "{first}"
        );

        // A full deployment reports each authenticated rank id once.
        let full = VerifiedCore::open_tensor_parallel(&root, &publishers, 2, None).unwrap();
        let full = full
            .describe(crate::runtime_core::InspectionOptions::default())
            .unwrap();
        let ids: Vec<u64> = full["ranks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["rank"].as_u64().unwrap())
            .collect();
        assert_eq!(
            ids,
            vec![0, 1],
            "rank ids must come from the authenticated bodies"
        );

        // The default document reports counts only, and capacity falls back to the
        // captured execution maximum the contract omits.
        let default = selected
            .describe(crate::runtime_core::InspectionOptions::default())
            .unwrap();
        assert_eq!(default["model"]["max_active"], 4);
        assert!(
            default["ranks"][0]["graphs"].is_u64(),
            "{}",
            default["ranks"][0]
        );
        assert!(default["ranks"][0]["allocations"].is_u64());
        assert!(default["ranks"][0].get("memory_map").is_none());
        assert!(default["ranks"][0].get("execution_graphs").is_none());
        let authorized = selected.authorize("example", false).unwrap();
        let AuthorizedKind::Payload(payload) = &authorized.kind else {
            panic!("expected the payload backend");
        };
        // Move the core directory and replace one companion with different bytes at the same
        // relative path. Authenticated reads must keep using the handles opened before acceptance.
        let moved = root.with_extension("moved");
        let _ = std::fs::remove_dir_all(&moved);
        if cfg!(windows) {
            // Windows core handles deny renaming or replacing the core while it is
            // open (docs/windows.md), so the replacement cannot occur there.
            assert!(std::fs::rename(&root, &moved).is_err());
            payload.verify_stored_data().unwrap();
            drop(authorized);
            let _ = std::fs::remove_dir_all(root);
            return;
        }
        std::fs::rename(&root, &moved).unwrap();
        let weight = moved.join("weights/tp1.safetensors");
        std::fs::remove_file(&weight).unwrap();
        std::fs::write(&weight, b"replacement contents").unwrap();
        payload.verify_stored_data().unwrap();
        let files = payload.rank_ranges(1).unwrap();
        assert!(
            files
                .iter()
                .any(|file| file.filename == "weights/tp1.safetensors")
        );
        let _ = std::fs::remove_dir_all(moved);
    }
}
