/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Authenticated payload-package envelopes and payload-relative executable access.
//!
//! One envelope carries a shared body, one signed body per prepared deployment, and one signed
//! body per rank. Every signature covers the exact serialized bytes of its body, and each parent
//! body commits to its children with SHA-256, so verification never parses and reserializes
//! before checking. A host serving one rank authenticates the shared body, its deployment, and
//! its own rank without reading another rank's bodies or payloads.
//!
//! Executable resources are one zstd frame per resource. Compressed bytes are authenticated by
//! signed compressed-chunk commitments and the payload catalog before the decoder consumes them.
//! The decoder enforces the signed decoded length, a bounded window, one complete frame, and no
//! trailing bytes.

use super::{
    invalid,
    schema::{self, wire},
};
use crate::Result;
use crate::runtime_package::Publisher;
use prost::Message;
use ring::{digest, signature};
use std::collections::BTreeSet;
use std::io::{self, BufRead, Read, Write};

/// Compressed chunk size shared with executable resource records.
pub const CHUNK: usize = schema::MAX_CHUNK;
/// Bound on the in-memory convenience compression/open helpers. Streaming paths are bounded by
/// their buffers instead.
pub const MAX_IN_MEMORY: usize = 128 << 20;
/// Bound on compressed chunks accepted for one resource (4 TiB at the fixed chunk size).
const MAX_CHUNKS: u64 = 1 << 20;
/// Bound on compressed and decoded resource bytes.
const MAX_RESOURCE_BYTES: u64 = MAX_CHUNKS * CHUNK as u64;
/// Fixed zstd level and maximum decoder back-reference window.
const COMPRESSION_LEVEL: i32 = 3;
const WINDOW_LOG: u32 = 23;
/// Bounded input buffer used while feeding the zstd encoder.
const INPUT_BUFFER: usize = 1 << 20;

const KIND_PACKAGE: u32 = 1;
const KIND_DEPLOYMENT: u32 = 2;
const KIND_RANK: u32 = 3;

fn fields(values: &[&[u8]]) -> Vec<u8> {
    let mut result = Vec::new();
    for value in values {
        result.extend_from_slice(&(value.len() as u64).to_le_bytes());
        result.extend_from_slice(value);
    }
    result
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let digest = digest::digest(&digest::SHA256, bytes);
    let mut result = [0; 32];
    result.copy_from_slice(digest.as_ref());
    result
}

/// The signature identity of a publisher profile: SHA-256 of its Ed25519 public key.
pub fn fingerprint(public_key: &[u8; 32]) -> [u8; 32] {
    sha256(public_key)
}

/// Domain-separated message for one signed body.
fn signature_message(
    package_id: &[u8],
    signer: &[u8],
    kind: u32,
    first: u32,
    second: u32,
    body: &[u8],
) -> Vec<u8> {
    fields(&[
        b"flare-package-signature-v5",
        &schema::FORMAT_VERSION.to_le_bytes(),
        package_id,
        signer,
        &kind.to_le_bytes(),
        &first.to_le_bytes(),
        &second.to_le_bytes(),
        body,
    ])
}

fn verify_signature(public_key: &[u8; 32], message: &[u8], signature: &[u8]) -> Result<()> {
    if signature.len() != 64 {
        return Err(invalid("package signature has the wrong length"));
    }
    signature::UnparsedPublicKey::new(&signature::ED25519, public_key)
        .verify(message, signature)
        .map_err(|_| invalid("package signature verification failed"))
}

fn check_commitment(commitment: &[u8], body: &[u8], what: &str) -> Result<()> {
    if commitment != sha256(body).as_slice() {
        return Err(invalid(format!("{what} differs from its commitment")));
    }
    Ok(())
}

/// Decode and bound-check an envelope without trusting anything it says.
pub fn envelope(bytes: &[u8]) -> Result<wire::ProtectedPackage> {
    if bytes.len() > schema::MAX_ENVELOPE {
        return Err(invalid("package envelope exceeds its limit"));
    }
    let envelope = wire::ProtectedPackage::decode(bytes)
        .map_err(|_| invalid("invalid payload package envelope"))?;
    if envelope.version != schema::FORMAT_VERSION {
        return Err(invalid("unsupported payload package format version"));
    }
    if envelope.package_id.len() != 32 || envelope.signer.len() != 32 {
        return Err(invalid(
            "package identity or signer fingerprint is malformed",
        ));
    }
    if envelope.package_body.len() > schema::MAX_BODY {
        return Err(invalid("package body exceeds its limit"));
    }
    if envelope.deployments.len() > 4096 || envelope.ranks.len() > 65536 {
        return Err(invalid("package envelope holds too many bodies"));
    }
    Ok(envelope)
}

/// What a package claims before any signature is verified: which key signed it and where it says
/// it was published. Callers resolve the signer's identity from this claim, then verify; nothing
/// here is trusted on its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    /// Lowercase SHA-256 fingerprint of the signing key.
    pub signer: String,
    pub package_id: [u8; 32],
    pub namespace: String,
    pub official_source: String,
    pub development: bool,
    pub model_identity: String,
    pub hardware_identity: String,
}

impl Claim {
    pub fn read(bytes: &[u8]) -> Result<Self> {
        let envelope = envelope(bytes)?;
        let body = wire::PackageBody::decode(envelope.package_body.as_slice())
            .map_err(|_| invalid("invalid package body"))?;
        let publisher = body
            .publisher
            .as_ref()
            .ok_or_else(|| invalid("package has no publisher metadata"))?;
        schema::validate_publisher_metadata(publisher)?;
        let mut package_id = [0; 32];
        package_id.copy_from_slice(&envelope.package_id);
        Ok(Self {
            signer: hex(&envelope.signer),
            package_id,
            namespace: publisher.namespace.clone(),
            official_source: publisher.official_source.clone(),
            development: publisher.development,
            model_identity: body.model_identity,
            hardware_identity: body.hardware_identity,
        })
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// What one verification authenticates.
///
/// [`Selection::Catalog`] authenticates only the shared body and its payload catalog. It is
/// discovery, never execution: [`VerifiedPackage::authorize`] rejects it, because no prepared
/// deployment, rank, or executable has been authenticated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    Catalog,
    /// One prepared deployment. `rank: None` verifies every rank of that deployment, which is
    /// what a single-host multi-rank load needs; `Some(rank)` verifies one rank alone, which is
    /// what a host holding one rank's copy needs.
    Deployment {
        deployment: usize,
        rank: Option<u32>,
    },
}

/// Resolve the verification key by fingerprint. Namespace and name in a profile are claims; the
/// signed package body fixes the namespace, so a profile list may carry the same key under the
/// several namespaces a publisher promoted a package through. Profiles that disagree on the key
/// itself cannot match, and a namespace that no profile claims is still rejected by the caller's
/// identity comparison.
fn resolve_publisher<'a>(
    publishers: &'a [Publisher],
    signer: &[u8],
    namespace: Option<&str>,
) -> Result<&'a Publisher> {
    let mut found: Option<&Publisher> = None;
    for publisher in publishers {
        if fingerprint(&publisher.public_key) != signer {
            continue;
        }
        publisher.validate()?;
        if let Some(namespace) = namespace
            && publisher.namespace != namespace
        {
            continue;
        }
        match found {
            Some(previous) if previous.namespace != publisher.namespace => {
                return Err(invalid(
                    "conflicting publisher profiles for the signing key",
                ));
            }
            Some(_) => {}
            None => found = Some(publisher),
        }
    }
    found.ok_or_else(|| invalid("package signer is not a configured publisher"))
}

/// A package whose shared body and selected deployment bodies are authenticated.
pub struct VerifiedPackage {
    envelope: wire::ProtectedPackage,
    package: wire::PackageBody,
    catalog: schema::Catalog,
    publisher: Publisher,
    deployment: Option<wire::DeploymentBody>,
    ranks: Vec<wire::RankBody>,
    selection: Selection,
    /// SHA-256 of the exact envelope bytes this package was verified from.
    envelope_digest: [u8; 32],
}

impl VerifiedPackage {
    pub fn claim(&self) -> Claim {
        let publisher = self.package.publisher.clone().unwrap_or_default();
        Claim {
            signer: hex(&self.envelope.signer),
            package_id: self.package_id(),
            namespace: publisher.namespace,
            official_source: publisher.official_source,
            development: publisher.development,
            model_identity: self.package.model_identity.clone(),
            hardware_identity: self.package.hardware_identity.clone(),
        }
    }
    pub fn publisher(&self) -> &Publisher {
        &self.publisher
    }
    pub fn package(&self) -> &wire::PackageBody {
        &self.package
    }
    pub fn package_id(&self) -> [u8; 32] {
        let mut id = [0; 32];
        id.copy_from_slice(&self.envelope.package_id);
        id
    }
    pub fn catalog(&self) -> &schema::Catalog {
        &self.catalog
    }
    /// Verify one deployment body without its ranks. Used to resolve a requested tensor-parallel
    /// size before any rank or payload is opened.
    pub fn deployment_body(&self, index: usize) -> Result<wire::DeploymentBody> {
        let commitment = self
            .package
            .deployments
            .get(index)
            .ok_or_else(|| invalid("deployment is outside the package"))?
            .clone();
        let body = signed_deployment(&self.envelope, index)?;
        check_commitment(&commitment.body_digest, body, "deployment body")?;
        verify_signature(
            &self.publisher.public_key,
            &signature_message(
                &self.envelope.package_id,
                &self.envelope.signer,
                KIND_DEPLOYMENT,
                index as u32,
                0,
                body,
            ),
            signed_deployment_signature(&self.envelope, index)?,
        )?;
        let decoded =
            wire::DeploymentBody::decode(body).map_err(|_| invalid("invalid deployment body"))?;
        let mut catalog = self.catalog.clone();
        schema::validate_deployment(&decoded, &mut catalog)?;
        if decoded.deployment_id != commitment.deployment_id {
            return Err(invalid("deployment body disagrees with its commitment"));
        }
        Ok(decoded)
    }

    /// The digest of the exact envelope bytes this package was verified from.
    pub fn envelope_digest(&self) -> [u8; 32] {
        self.envelope_digest
    }
    /// True when `bytes` are the envelope this package was verified from. Cache and acquisition
    /// use this to bind authenticated metadata to one primary instead of mixing packages.
    pub fn matches_envelope(&self, bytes: &[u8]) -> bool {
        sha256(bytes) == self.envelope_digest
    }
    pub fn selection(&self) -> Selection {
        self.selection
    }
    pub fn deployment_count(&self) -> usize {
        self.package.deployments.len()
    }
    pub fn deployment(&self) -> Option<&wire::DeploymentBody> {
        self.deployment.as_ref()
    }
    /// Verified rank bodies in rank order. Empty for [`Selection::Catalog`].
    pub fn ranks(&self) -> &[wire::RankBody] {
        &self.ranks
    }
    pub fn rank(&self, rank: u32) -> Option<&wire::RankBody> {
        self.ranks.iter().find(|body| body.rank == rank)
    }
    /// True when a prepared deployment and at least one rank were authenticated.
    pub fn is_executable(&self) -> bool {
        self.deployment.is_some() && !self.ranks.is_empty()
    }
    /// Hash actual bytes against a signed payload digest. This is the authentication step for
    /// plaintext payloads that are not executable resources: module globals, frontend assets, and
    /// weight bytes. A signed digest without this check is not evidence about local bytes.
    pub fn verify_payload(&self, id: &[u8], bytes: &[u8]) -> Result<()> {
        let length = self.catalog.length(id)?;
        if length != bytes.len() as u64 {
            return Err(invalid("payload length differs from its signed descriptor"));
        }
        if sha256(bytes).as_slice() != id {
            return Err(invalid("payload content verification failed"));
        }
        Ok(())
    }
    /// Record the string the user applied to accept the publisher's terms. It must equal the
    /// namespace in the signed package metadata; executable access is granted only after that
    /// explicit equality check.
    pub fn authorize(self, accepted: &str) -> Result<AuthorizedPackage> {
        if !self.is_executable() {
            return Err(invalid(
                "catalog-only verification cannot authorize executable access",
            ));
        }
        if accepted.is_empty() || accepted.len() > 256 || accepted.chars().any(char::is_control) {
            return Err(invalid("invalid acceptance string"));
        }
        let namespace = self
            .package
            .publisher
            .as_ref()
            .ok_or_else(|| invalid("package has no publisher metadata"))?
            .namespace
            .as_str();
        if accepted != namespace {
            return Err(invalid(
                "accepted namespace does not match the signed package namespace",
            ));
        }
        Ok(AuthorizedPackage { verified: self })
    }
}

/// Authenticate the shared body and the selected deployment and rank bodies.
pub fn verify(
    bytes: &[u8],
    publishers: &[Publisher],
    selection: Selection,
) -> Result<VerifiedPackage> {
    let envelope = envelope(bytes)?;
    let unsigned = wire::PackageBody::decode(envelope.package_body.as_slice())
        .map_err(|_| invalid("invalid package body"))?;
    let expected_namespace = unsigned
        .publisher
        .as_ref()
        .map(|publisher| publisher.namespace.as_str());
    let publisher = resolve_publisher(publishers, &envelope.signer, expected_namespace)?.clone();
    verify_signature(
        &publisher.public_key,
        &signature_message(
            &envelope.package_id,
            &envelope.signer,
            KIND_PACKAGE,
            0,
            0,
            &envelope.package_body,
        ),
        &envelope.package_signature,
    )?;
    let package = unsigned;
    let mut catalog = schema::validate_package(&package)?;
    if package.package_id.as_slice() != envelope.package_id.as_slice() {
        return Err(invalid(
            "envelope and body disagree on the package identity",
        ));
    }
    let metadata = package
        .publisher
        .as_ref()
        .ok_or_else(|| invalid("package has no publisher metadata"))?;
    if publisher.namespace != metadata.namespace || publisher.development != metadata.development {
        return Err(invalid(
            "publisher profile disagrees with the signed package identity",
        ));
    }

    let mut deployment = None;
    let mut ranks = Vec::new();
    if let Selection::Deployment {
        deployment: index,
        rank,
    } = selection
    {
        let commitment = package
            .deployments
            .get(index)
            .ok_or_else(|| invalid("selected deployment is outside the package"))?
            .clone();
        let body = signed_deployment(&envelope, index)?;
        check_commitment(&commitment.body_digest, body, "deployment body")?;
        verify_signature(
            &publisher.public_key,
            &signature_message(
                &envelope.package_id,
                &envelope.signer,
                KIND_DEPLOYMENT,
                index as u32,
                0,
                body,
            ),
            signed_deployment_signature(&envelope, index)?,
        )?;
        let decoded =
            wire::DeploymentBody::decode(body).map_err(|_| invalid("invalid deployment body"))?;
        let count = schema::validate_deployment(&decoded, &mut catalog)?;
        if decoded.deployment_id != commitment.deployment_id {
            return Err(invalid("deployment body disagrees with its commitment"));
        }
        let selected: Vec<u32> = match rank {
            Some(rank) => {
                if rank >= count {
                    return Err(invalid("selected rank is outside the deployment"));
                }
                vec![rank]
            }
            None => (0..count).collect(),
        };
        // Freeze the shared and deployment catalogs before validating any rank. Each rank is
        // checked against that frozen view plus its own descriptors, so an earlier rank's payload
        // catalog can never authorize a later rank's slices.
        let frozen = catalog.clone();
        let mut resource_ids = BTreeSet::new();
        let mut decoded_metadata_nbytes = 0u64;
        for rank_id in selected {
            let rank_commitment = decoded
                .ranks
                .get(rank_id as usize)
                .ok_or_else(|| invalid("selected rank is outside the deployment"))?;
            let body = signed_rank(&envelope, index, rank_id)?;
            check_commitment(&rank_commitment.body_digest, body, "rank body")?;
            verify_signature(
                &publisher.public_key,
                &signature_message(
                    &envelope.package_id,
                    &envelope.signer,
                    KIND_RANK,
                    index as u32,
                    rank_id,
                    body,
                ),
                signed_rank_signature(&envelope, index, rank_id)?,
            )?;
            let decoded_rank =
                wire::RankBody::decode(body).map_err(|_| invalid("invalid rank body"))?;
            if decoded_rank.rank != rank_id || decoded_rank.deployment_id != decoded.deployment_id {
                return Err(invalid("rank body disagrees with its deployment"));
            }
            decoded_metadata_nbytes = decoded_metadata_nbytes.checked_add(
                super::prepared_weights::reconstructed_rank_nbytes(&decoded_rank, body.len())?
            ).ok_or_else(|| invalid("selected rank metadata length overflow"))?;
            if decoded_metadata_nbytes > schema::MAX_SELECTED_RANK_METADATA {
                return Err(invalid("selected reconstructed rank metadata exceeds its limit"));
            }
            ranks.push(decoded_rank);
        }
        for decoded_rank in &ranks {
            let mut local = frozen.clone();
            schema::validate_rank(decoded_rank, count, &mut local, &mut resource_ids)?;
            if decoded_rank.compressed_prepared_weights.is_some()
                && let Some(weights) = super::prepared_weights::catalog(decoded_rank)?
            {
                schema::validate_rank_prepared_weights(decoded_rank, &weights)?;
            }
            catalog.insert_all(&decoded_rank.payloads)?;
        }
        deployment = Some(decoded);
    }

    Ok(VerifiedPackage {
        envelope,
        package,
        catalog,
        publisher,
        deployment,
        ranks,
        selection,
        envelope_digest: sha256(bytes),
    })
}

fn signed_deployment_signature(envelope: &wire::ProtectedPackage, index: usize) -> Result<&[u8]> {
    let signed = signed_deployment_entry(envelope, index)?;
    Ok(&signed.signature)
}

fn signed_deployment(envelope: &wire::ProtectedPackage, index: usize) -> Result<&[u8]> {
    let signed = signed_deployment_entry(envelope, index)?;
    Ok(&signed.body)
}

fn signed_deployment_entry(
    envelope: &wire::ProtectedPackage,
    index: usize,
) -> Result<&wire::SignedDeployment> {
    let mut found = None;
    for signed in &envelope.deployments {
        if signed.index as usize != index {
            continue;
        }
        if found.is_some() {
            return Err(invalid("duplicate signed deployment body"));
        }
        found = Some(signed);
    }
    let signed = found.ok_or_else(|| invalid("missing signed deployment body"))?;
    if signed.body.len() > schema::MAX_BODY || signed.signature.len() != 64 {
        return Err(invalid("malformed signed deployment body"));
    }
    Ok(signed)
}

fn signed_rank_signature(
    envelope: &wire::ProtectedPackage,
    deployment: usize,
    rank: u32,
) -> Result<&[u8]> {
    let signed = signed_rank_entry(envelope, deployment, rank)?;
    Ok(&signed.signature)
}

fn signed_rank(envelope: &wire::ProtectedPackage, deployment: usize, rank: u32) -> Result<&[u8]> {
    let signed = signed_rank_entry(envelope, deployment, rank)?;
    Ok(&signed.body)
}

fn signed_rank_entry<'a>(
    envelope: &'a wire::ProtectedPackage,
    deployment: usize,
    rank: u32,
) -> Result<&'a wire::SignedRank> {
    let mut found = None;
    for signed in &envelope.ranks {
        if signed.deployment as usize != deployment || signed.rank != rank {
            continue;
        }
        if found.is_some() {
            return Err(invalid("duplicate signed rank body"));
        }
        found = Some(signed);
    }
    let signed = found.ok_or_else(|| invalid("missing signed rank body"))?;
    if signed.body.len() > schema::MAX_BODY || signed.signature.len() != 64 {
        return Err(invalid("malformed signed rank body"));
    }
    Ok(signed)
}

/// A package whose publisher terms were accepted, granting access to executable resources.
pub struct AuthorizedPackage {
    verified: VerifiedPackage,
}

impl AuthorizedPackage {
    pub fn verified(&self) -> &VerifiedPackage {
        &self.verified
    }
    pub fn verify_payload(&self, id: &[u8], bytes: &[u8]) -> Result<()> {
        self.verified.verify_payload(id, bytes)
    }
    /// Open one stored executable resource. Compressed bytes are authenticated by their signed
    /// chunk commitments before the decoder sees them.
    pub fn open_reader<R: Read>(
        &self,
        resource: &wire::ExecutableResource,
        source: R,
    ) -> Result<ResourceReader<R>> {
        let stored = resource
            .stored
            .as_ref()
            .ok_or_else(|| invalid("executable resource has no stored bytes"))?;
        if resource.chunk_nbytes as usize != CHUNK {
            return Err(invalid(
                "executable resource uses an unsupported chunk size",
            ));
        }
        open_compressed_reader(
            source,
            stored.nbytes,
            resource.decoded_nbytes,
            &resource.chunk_sha256,
        )
    }

    /// Open one complete executable resource into memory. This is only for graphs and cubins,
    /// not for bulk weights, and is bounded by [`MAX_IN_MEMORY`].
    pub fn open_resource(
        &self,
        resource: &wire::ExecutableResource,
        stored: &[u8],
    ) -> Result<Vec<u8>> {
        let slice = resource
            .stored
            .as_ref()
            .ok_or_else(|| invalid("executable resource has no stored bytes"))?;
        if slice.nbytes != stored.len() as u64 {
            return Err(invalid("executable stored length is inconsistent"));
        }
        if resource.decoded_nbytes > MAX_IN_MEMORY as u64 {
            return Err(invalid(
                "in-memory executable open is limited to graphs and cubins; use open_reader",
            ));
        }
        let mut plaintext = Vec::with_capacity(schema::size(resource.decoded_nbytes)?);
        let mut reader = self.open_reader(resource, std::io::Cursor::new(stored))?;
        reader
            .read_to_end(&mut plaintext)
            .map_err(|error| invalid(format!("read executable: {error}")))?;
        if plaintext.len() as u64 != resource.decoded_nbytes {
            return Err(invalid("executable decoded length is inconsistent"));
        }
        Ok(plaintext)
    }

    /// Stream one executable resource into `sink` with bounded memory, requiring the exact
    /// declared decoded length.
    pub fn write_plaintext<R: Read, W: Write>(
        &self,
        resource: &wire::ExecutableResource,
        source: R,
        sink: &mut W,
    ) -> Result<u64> {
        let mut reader = self.open_reader(resource, source)?;
        let copied = std::io::copy(&mut reader, sink)
            .map_err(|error| invalid(format!("write executable plaintext: {error}")))?;
        if copied != resource.decoded_nbytes {
            return Err(invalid("executable decoded length is inconsistent"));
        }
        Ok(copied)
    }
}

/// Which body a signature covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    Package,
    Deployment(u32),
    Rank { deployment: u32, rank: u32 },
}

impl BodyKind {
    fn parts(self) -> (u32, u32, u32) {
        match self {
            BodyKind::Package => (KIND_PACKAGE, 0, 0),
            BodyKind::Deployment(index) => (KIND_DEPLOYMENT, index, 0),
            BodyKind::Rank { deployment, rank } => (KIND_RANK, deployment, rank),
        }
    }
}

/// Sign one serialized body. The envelope carries the signer fingerprint, so the caller supplies
/// the public key the fingerprint is taken from; the private key never leaves the caller.
pub fn sign_body(
    key: &signature::Ed25519KeyPair,
    package_id: &[u8; 32],
    public_key: &[u8; 32],
    body: BodyKind,
    bytes: &[u8],
) -> Result<Vec<u8>> {
    let (kind, first, second) = body.parts();
    let signer = fingerprint(public_key);
    Ok(key
        .sign(&signature_message(
            package_id, &signer, kind, first, second, bytes,
        ))
        .as_ref()
        .to_vec())
}

/// Writer side: assemble bodies and sign them into one envelope. Resource identifiers and payload
/// identities must be assigned by the caller before this step.
pub struct EnvelopeBuilder {
    package_id: [u8; 32],
    public_key: [u8; 32],
    package_body: Vec<u8>,
    deployments: Vec<(u32, Vec<u8>)>,
    ranks: Vec<(u32, u32, Vec<u8>)>,
}

impl EnvelopeBuilder {
    pub fn new(package_body: Vec<u8>, public_key: [u8; 32], package_id: [u8; 32]) -> Result<Self> {
        if package_body.len() > schema::MAX_BODY {
            return Err(invalid("package body exceeds its limit"));
        }
        Ok(Self {
            package_id,
            public_key,
            package_body,
            deployments: Vec::new(),
            ranks: Vec::new(),
        })
    }
    pub fn deployment(&mut self, index: u32, body: Vec<u8>) -> Result<&mut Self> {
        if body.len() > schema::MAX_BODY {
            return Err(invalid("deployment body exceeds its limit"));
        }
        self.deployments.push((index, body));
        Ok(self)
    }
    pub fn rank(&mut self, deployment: u32, rank: u32, body: Vec<u8>) -> Result<&mut Self> {
        if body.len() > schema::MAX_BODY {
            return Err(invalid("rank body exceeds its limit"));
        }
        self.ranks.push((deployment, rank, body));
        Ok(self)
    }
    /// Sign every body and serialize the envelope. The signer fingerprint, not the verification
    /// key, is what the envelope carries.
    pub fn sign(self, key: &signature::Ed25519KeyPair) -> Result<Vec<u8>> {
        let signer = fingerprint(&self.public_key);
        let sign = |body: BodyKind, bytes: &[u8]| -> Result<Vec<u8>> {
            sign_body(key, &self.package_id, &self.public_key, body, bytes)
        };
        let envelope = wire::ProtectedPackage {
            version: schema::FORMAT_VERSION,
            package_id: self.package_id.to_vec(),
            signer: signer.to_vec(),
            package_signature: sign(BodyKind::Package, &self.package_body)?,
            package_body: self.package_body,
            deployments: self
                .deployments
                .into_iter()
                .map(|(index, body)| {
                    Ok(wire::SignedDeployment {
                        index,
                        signature: sign(BodyKind::Deployment(index), &body)?,
                        body,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            ranks: self
                .ranks
                .into_iter()
                .map(|(deployment, rank, body)| {
                    Ok(wire::SignedRank {
                        deployment,
                        rank,
                        signature: sign(BodyKind::Rank { deployment, rank }, &body)?,
                        body,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        };
        Ok(wire::ProtectedPackage::encode_to_vec(&envelope))
    }
}

/// Compression metadata for one executable resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedResource {
    pub stored_nbytes: u64,
    pub chunk_sha256: Vec<Vec<u8>>,
}

fn validate_decoded(decoded_nbytes: u64) -> Result<()> {
    if decoded_nbytes == 0 || decoded_nbytes > MAX_RESOURCE_BYTES {
        return Err(invalid("invalid executable decoded length"));
    }
    Ok(())
}

fn validate_stored(stored_nbytes: u64) -> Result<u64> {
    if stored_nbytes == 0 || stored_nbytes > MAX_RESOURCE_BYTES {
        return Err(invalid("invalid executable stored length"));
    }
    let chunks = stored_nbytes.div_ceil(CHUNK as u64);
    if chunks == 0 || chunks > MAX_CHUNKS {
        return Err(invalid(
            "executable resource has too many compressed chunks",
        ));
    }
    Ok(chunks)
}

fn validate_chunks(stored_nbytes: u64, chunk_sha256: &[Vec<u8>]) -> Result<()> {
    let chunks = validate_stored(stored_nbytes)?;
    if chunk_sha256.len() as u64 != chunks || chunk_sha256.iter().any(|digest| digest.len() != 32) {
        return Err(invalid(
            "executable compressed chunk commitments are inconsistent",
        ));
    }
    Ok(())
}

/// Streams compressed bytes to `sink` and hashes each fixed-size compressed chunk.
struct ChunkHasher<'a, W: Write + ?Sized> {
    sink: &'a mut W,
    buffer: Vec<u8>,
    chunk_sha256: Vec<Vec<u8>>,
    stored_nbytes: u64,
}

impl<'a, W: Write + ?Sized> ChunkHasher<'a, W> {
    fn new(sink: &'a mut W) -> Self {
        Self {
            sink,
            buffer: Vec::with_capacity(CHUNK),
            chunk_sha256: Vec::new(),
            stored_nbytes: 0,
        }
    }

    fn flush_chunk(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let nbytes = self.buffer.len() as u64;
        self.stored_nbytes = self
            .stored_nbytes
            .checked_add(nbytes)
            .ok_or_else(|| io::Error::other("compressed executable length overflow"))?;
        if self.stored_nbytes > MAX_RESOURCE_BYTES {
            return Err(io::Error::other("compressed executable exceeds its limit"));
        }
        self.chunk_sha256.push(sha256(&self.buffer).to_vec());
        self.sink
            .write_all(&self.buffer)
            .map_err(|error| io::Error::other(format!("write compressed executable: {error}")))?;
        self.buffer.clear();
        Ok(())
    }

    fn finish(mut self) -> Result<CompressedResource> {
        self.flush_chunk()
            .map_err(|error| invalid(error.to_string()))?;
        validate_chunks(self.stored_nbytes, &self.chunk_sha256)?;
        Ok(CompressedResource {
            stored_nbytes: self.stored_nbytes,
            chunk_sha256: self.chunk_sha256,
        })
    }
}

impl<W: Write + ?Sized> Write for ChunkHasher<'_, W> {
    fn write(&mut self, mut bytes: &[u8]) -> io::Result<usize> {
        let total = bytes.len();
        while !bytes.is_empty() {
            let room = CHUNK - self.buffer.len();
            let count = room.min(bytes.len());
            self.buffer.extend_from_slice(&bytes[..count]);
            bytes = &bytes[count..];
            if self.buffer.len() == CHUNK {
                self.flush_chunk()?;
            }
        }
        Ok(total)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.sink.flush()
    }
}

/// Compress exactly `decoded_nbytes` bytes from `source` into one zstd frame.
pub fn compress_stream<R: Read, W: Write + ?Sized>(
    mut source: R,
    decoded_nbytes: u64,
    sink: &mut W,
) -> Result<CompressedResource> {
    validate_decoded(decoded_nbytes)?;
    let mut chunks = ChunkHasher::new(sink);
    {
        let mut encoder = zstd::stream::write::Encoder::new(&mut chunks, COMPRESSION_LEVEL)
            .map_err(|error| invalid(format!("create executable compressor: {error}")))?;
        encoder
            .set_pledged_src_size(Some(decoded_nbytes))
            .map_err(|error| invalid(format!("set executable decoded length: {error}")))?;
        encoder
            .include_checksum(true)
            .map_err(|error| invalid(format!("enable executable checksum: {error}")))?;
        encoder
            .include_contentsize(true)
            .map_err(|error| invalid(format!("enable executable content size: {error}")))?;
        encoder
            .window_log(WINDOW_LOG)
            .map_err(|error| invalid(format!("set executable window: {error}")))?;
        let mut buffer = vec![0u8; INPUT_BUFFER];
        let mut remaining = decoded_nbytes;
        while remaining > 0 {
            let count = remaining.min(buffer.len() as u64) as usize;
            source
                .read_exact(&mut buffer[..count])
                .map_err(|error| invalid(format!("read executable plaintext: {error}")))?;
            encoder
                .write_all(&buffer[..count])
                .map_err(|error| invalid(format!("write executable plaintext: {error}")))?;
            remaining -= count as u64;
        }
        encoder
            .finish()
            .map_err(|error| invalid(format!("finish executable compressor: {error}")))?;
    }
    chunks.finish()
}

/// Compress one in-memory executable resource.
pub fn compress(plaintext: &[u8]) -> Result<(Vec<u8>, CompressedResource)> {
    let mut stored = Vec::new();
    let resource = compress_stream(
        std::io::Cursor::new(plaintext),
        plaintext.len() as u64,
        &mut stored,
    )?;
    Ok((stored, resource))
}

/// A bounded, authenticated source of compressed chunks.
struct VerifiedChunkSource<R> {
    source: R,
    stored_remaining: u64,
    chunk_sha256: Vec<Vec<u8>>,
    next_chunk: usize,
    buffer: Vec<u8>,
    position: usize,
    error: Option<String>,
}

impl<R: Read> VerifiedChunkSource<R> {
    fn new(source: R, stored_nbytes: u64, chunk_sha256: Vec<Vec<u8>>) -> Self {
        Self {
            source,
            stored_remaining: stored_nbytes,
            chunk_sha256,
            next_chunk: 0,
            buffer: Vec::with_capacity(CHUNK),
            position: 0,
            error: None,
        }
    }

    fn fail(&mut self, message: &str) -> io::Error {
        self.buffer.clear();
        self.position = 0;
        if self.error.is_none() {
            self.error = Some(message.to_owned());
        }
        io::Error::new(io::ErrorKind::InvalidData, message)
    }

    fn unread(&self) -> u64 {
        (self.buffer.len() - self.position) as u64 + self.stored_remaining
    }
}

impl<R: Read> BufRead for VerifiedChunkSource<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if let Some(message) = &self.error {
            return Err(io::Error::new(io::ErrorKind::InvalidData, message.clone()));
        }
        if self.position == self.buffer.len() {
            if self.next_chunk == self.chunk_sha256.len() {
                if self.stored_remaining != 0 {
                    return Err(self.fail("executable compressed length is inconsistent"));
                }
                self.buffer.clear();
                self.position = 0;
                return Ok(&[]);
            }
            let count = self.stored_remaining.min(CHUNK as u64) as usize;
            self.buffer.resize(count, 0);
            if let Err(error) = self.source.read_exact(&mut self.buffer) {
                let message = format!("read executable compressed chunk: {error}");
                return Err(self.fail(&message));
            }
            if self.chunk_sha256[self.next_chunk].as_slice() != sha256(&self.buffer).as_slice() {
                return Err(self.fail("executable compressed chunk content verification failed"));
            }
            self.stored_remaining -= count as u64;
            self.next_chunk += 1;
            self.position = 0;
        }
        Ok(&self.buffer[self.position..])
    }

    fn consume(&mut self, amount: usize) {
        let available = self.buffer.len() - self.position;
        if amount > available {
            let _ = self.fail("executable compressed reader consumed past its chunk");
            return;
        }
        self.position += amount;
    }
}

impl<R: Read> Read for VerifiedChunkSource<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        let buffer = self.fill_buf()?;
        if buffer.is_empty() {
            return Ok(0);
        }
        let count = output.len().min(buffer.len());
        output[..count].copy_from_slice(&buffer[..count]);
        self.consume(count);
        Ok(count)
    }
}

/// Bounded zstd reader over one authenticated executable resource.
pub struct ResourceReader<R> {
    decoder: Option<zstd::stream::read::Decoder<'static, VerifiedChunkSource<R>>>,
    remaining: u64,
    finished: bool,
    error: Option<String>,
}

impl<R: Read> ResourceReader<R> {
    fn finish_frame(&mut self) -> io::Result<()> {
        let decoder = self.decoder.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "executable stream is closed")
        })?;
        let source = decoder.finish();
        if source.unread() != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "executable compressed stream has trailing bytes",
            ));
        }
        Ok(())
    }
}

/// Open one compressed executable stream. The source is bounded by `stored_nbytes`; each
/// compressed chunk is authenticated before zstd consumes it.
pub fn open_compressed_reader<R: Read>(
    source: R,
    stored_nbytes: u64,
    decoded_nbytes: u64,
    chunk_sha256: &[Vec<u8>],
) -> Result<ResourceReader<R>> {
    validate_decoded(decoded_nbytes)?;
    validate_chunks(stored_nbytes, chunk_sha256)?;
    let verified = VerifiedChunkSource::new(source, stored_nbytes, chunk_sha256.to_vec());
    let mut decoder = zstd::stream::read::Decoder::with_buffer(verified)
        .map_err(|error| invalid(format!("create executable decompressor: {error}")))?;
    decoder = decoder.single_frame();
    decoder
        .window_log_max(WINDOW_LOG)
        .map_err(|error| invalid(format!("set executable window: {error}")))?;
    Ok(ResourceReader {
        decoder: Some(decoder),
        remaining: decoded_nbytes,
        finished: false,
        error: None,
    })
}

impl<R: Read> ResourceReader<R> {
    fn read_decoded(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if self.finished {
            return Ok(0);
        }
        if self.remaining == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "executable decoded length is inconsistent",
            ));
        }
        let limit = usize::try_from(self.remaining.min(output.len() as u64)).unwrap_or(usize::MAX);
        let count = self
            .decoder
            .as_mut()
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "executable stream is closed")
            })?
            .read(&mut output[..limit])?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "executable decoded length is inconsistent",
            ));
        }
        self.remaining -= count as u64;
        if self.remaining == 0 {
            let mut extra = [0u8; 1];
            let extra_count = self
                .decoder
                .as_mut()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "executable stream is closed")
                })?
                .read(&mut extra)?;
            if extra_count != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "executable decoded length is inconsistent",
                ));
            }
            self.finish_frame()?;
            self.finished = true;
        }
        Ok(count)
    }
}

impl<R: Read> Read for ResourceReader<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if let Some(message) = &self.error {
            return Err(io::Error::new(io::ErrorKind::InvalidData, message.clone()));
        }
        if output.is_empty() || self.finished {
            return Ok(0);
        }
        match self.read_decoded(output) {
            Ok(count) => Ok(count),
            Err(error) => {
                if self.error.is_none() {
                    self.error = Some(error.to_string());
                }
                self.decoder.take();
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_package::wire as runtime;
    use ring::rand::SystemRandom;
    use ring::signature::KeyPair;

    struct Fixture {
        publishers: Vec<Publisher>,
        signing_key: signature::Ed25519KeyPair,
        envelope: Vec<u8>,
        package_id: [u8; 32],
    }

    fn payload(id: u8, nbytes: u64) -> wire::PayloadDescriptor {
        wire::PayloadDescriptor {
            id: vec![id; 32],
            nbytes,
        }
    }
    fn slice(id: u8, offset: u64, nbytes: u64) -> wire::PayloadSlice {
        wire::PayloadSlice {
            payload_id: vec![id; 32],
            offset,
            nbytes,
        }
    }
    fn use_of(id: u8, nbytes: u64, role: wire::PayloadRole) -> wire::PayloadUse {
        wire::PayloadUse {
            slice: Some(slice(id, 0, nbytes)),
            role: role as i32,
        }
    }

    const FIXTURE_EXECUTABLE: &[u8] = b"fixture executable\0\0\0\0\0";

    fn fixture_executable() -> (Vec<u8>, CompressedResource) {
        compress(FIXTURE_EXECUTABLE).unwrap()
    }

    fn fixture_executable_nbytes() -> u64 {
        fixture_executable().0.len() as u64
    }

    fn rank_body(rank: u32) -> wire::RankBody {
        wire::RankBody {
            rank,
            deployment_id: "sm120-tp2".into(),
            context: Some(runtime::Context::default()),
            reservations: vec![],
            allocations: vec![wire::Allocation {
                name: "weights".into(),
                nbytes: 4096,
                alignment: 4096,
                address: Some(0x1000_0000),
                memory: runtime::Memory::Device as i32,
                sizing: None,
                relocations: vec![],
            }],
            initializers: vec![wire::Initializer {
                allocation: "weights".into(),
                destination_offset: 0,
                source: Some(use_of(
                    1,
                    fixture_executable_nbytes(),
                    wire::PayloadRole::Weights,
                )),
            }],
            payloads: vec![],
            resources: vec![{
                let (stored, compressed) = fixture_executable();
                wire::ExecutableResource {
                    resource_id: 100 + rank as u64,
                    kind: wire::ExecutableKind::Graph as i32,
                    stored: Some(slice(1, 0, stored.len() as u64)),
                    decoded_nbytes: FIXTURE_EXECUTABLE.len() as u64,
                    chunk_nbytes: CHUNK as u32,
                    chunk_sha256: compressed.chunk_sha256,
                }
            }],
            modules: vec![],
            functions: vec![],
            graphs: vec![wire::Graph {
                name: "decode".into(),
                flags: 0,
                resource: 0,
                node_count: 1,
                edge_count: 0,
                functions: vec![],
            }],
            events: vec![],
            peers: vec![],
            execution: Some(runtime::Execution::default()),
            roce: None,
            prepared_weights: None,
            compressed_prepared_weights: None,
        }
    }

    fn fixture() -> Fixture {
        let rng = SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let signing_key = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = signing_key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        };
        let package_id = [5u8; 32];
        let package_body = wire::PackageBody {
            format_version: schema::FORMAT_VERSION,
            package_id: package_id.to_vec(),
            model_identity: "model".into(),
            hardware_identity: "sm120".into(),
            publisher: Some(wire::PublisherMetadata {
                namespace: "example".into(),
                official_source: "https://huggingface.co/example/model".into(),
                development: false,
                license_name: "Example".into(),
                license_link: String::new(),
                ranks: 0,
            }),
            frontend: Some(runtime::FrontendSettings::default()),
            sampling_defaults: Some(runtime::DefaultSamplingParameters::default()),
            payloads: vec![payload(1, fixture_executable_nbytes())],
            assets: vec![],
            deployments: vec![wire::DeploymentCommitment {
                deployment_id: "sm120-tp2".into(),
                body_digest: vec![0; 32],
            }],
        };
        let mut body = package_body.clone();
        let ranks: Vec<(u32, Vec<u8>)> = (0..2)
            .map(|rank| (rank, wire::RankBody::encode_to_vec(&rank_body(rank))))
            .collect();
        let deployment_ranks: Vec<wire::RankCommitment> = ranks
            .iter()
            .map(|(rank, body)| wire::RankCommitment {
                rank: *rank,
                body_digest: sha256(body).to_vec(),
            })
            .collect();
        let deployment = wire::DeploymentBody {
            deployment_id: "sm120-tp2".into(),
            architecture: vec![12, 0],
            tensor_parallel: 2,
            contract: Some(runtime::ServingContract::default()),
            cluster: None,
            rebuild: None,
            engram: None,
            payloads: vec![payload(1, fixture_executable_nbytes())],
            ranks: deployment_ranks,
        };
        body.deployments[0].body_digest =
            sha256(&wire::DeploymentBody::encode_to_vec(&deployment)).to_vec();
        let mut builder = EnvelopeBuilder::new(
            wire::PackageBody::encode_to_vec(&body),
            public_key,
            package_id,
        )
        .unwrap();
        builder
            .deployment(0, wire::DeploymentBody::encode_to_vec(&deployment))
            .unwrap();
        for (rank, rank_body) in &ranks {
            builder.rank(0, *rank, rank_body.clone()).unwrap();
        }
        Fixture {
            publishers: vec![publisher],
            envelope: builder.sign(&signing_key).unwrap(),
            signing_key,
            package_id,
        }
    }

    #[test]
    fn claim_names_the_signer_fingerprint_and_identity() {
        let fixture = fixture();
        let claim = Claim::read(&fixture.envelope).unwrap();
        assert_eq!(
            claim.signer,
            hex(&fingerprint(&fixture.publishers[0].public_key))
        );
        assert_eq!(claim.namespace, "example");
        assert_eq!(
            claim.official_source,
            "https://huggingface.co/example/model"
        );
        assert_eq!(claim.package_id, fixture.package_id);
        assert_eq!(claim.model_identity, "model");
        assert!(!claim.development);
    }

    #[test]
    fn signer_must_be_the_key_fingerprint_not_the_key() {
        let fixture = fixture();
        let mut decoded = envelope(&fixture.envelope).unwrap();
        decoded.signer = fixture.publishers[0].public_key.to_vec();
        let bytes = wire::ProtectedPackage::encode_to_vec(&decoded);
        assert!(verify(&bytes, &fixture.publishers, Selection::Catalog).is_err());
        assert!(Claim::read(&bytes).is_ok());
    }

    #[test]
    fn publisher_profile_must_match_the_signed_identity() {
        let fixture = fixture();
        let mut other = fixture.publishers.clone();
        other[0].namespace = "someone-else".into();
        assert!(verify(&fixture.envelope, &other, Selection::Catalog).is_err());
        let mut development = fixture.publishers.clone();
        development[0].development = true;
        assert!(verify(&fixture.envelope, &development, Selection::Catalog).is_err());
    }

    #[test]
    fn catalog_selection_is_discovery_only() {
        let fixture = fixture();
        let verified = verify(&fixture.envelope, &fixture.publishers, Selection::Catalog).unwrap();
        assert!(verified.deployment().is_none());
        assert!(verified.ranks().is_empty());
        assert!(!verified.is_executable());
        assert!(verified.authorize("example").is_err());
    }

    #[test]
    fn whole_deployment_and_single_rank_selections_verify() {
        let fixture = fixture();
        let verified = verify(
            &fixture.envelope,
            &fixture.publishers,
            Selection::Deployment {
                deployment: 0,
                rank: Some(1),
            },
        )
        .unwrap();
        assert_eq!(verified.ranks().len(), 1);
        assert_eq!(verified.rank(1).unwrap().rank, 1);
        assert!(verified.is_executable());
        assert_eq!(verified.deployment().unwrap().deployment_id, "sm120-tp2");

        let all = verify(
            &fixture.envelope,
            &fixture.publishers,
            Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        assert_eq!(all.ranks().len(), 2);
        assert_eq!(all.ranks()[0].rank, 0);
        assert_eq!(all.ranks()[1].rank, 1);
        assert_eq!(all.catalog().len(), 1);
    }

    #[test]
    fn a_single_rank_copy_needs_no_sibling_body() {
        let fixture = fixture();
        let mut decoded = envelope(&fixture.envelope).unwrap();
        decoded.ranks.retain(|signed| signed.rank == 1);
        let bytes = wire::ProtectedPackage::encode_to_vec(&decoded);
        let verified = verify(
            &bytes,
            &fixture.publishers,
            Selection::Deployment {
                deployment: 0,
                rank: Some(1),
            },
        )
        .unwrap();
        assert_eq!(verified.ranks().len(), 1);
        assert!(
            verify(
                &bytes,
                &fixture.publishers,
                Selection::Deployment {
                    deployment: 0,
                    rank: None
                }
            )
            .is_err()
        );
    }

    #[test]
    fn corrupt_bodies_commitments_and_identity_are_rejected() {
        let fixture = fixture();
        let mut decoded = envelope(&fixture.envelope).unwrap();
        let last = decoded.package_body.len() - 1;
        decoded.package_body[last] ^= 1;
        let bytes = wire::ProtectedPackage::encode_to_vec(&decoded);
        assert!(verify(&bytes, &fixture.publishers, Selection::Catalog).is_err());

        let mut decoded = envelope(&fixture.envelope).unwrap();
        decoded.deployments[0].body[0] ^= 1;
        let bytes = wire::ProtectedPackage::encode_to_vec(&decoded);
        assert!(
            verify(
                &bytes,
                &fixture.publishers,
                Selection::Deployment {
                    deployment: 0,
                    rank: Some(0)
                }
            )
            .is_err()
        );

        let mut decoded = envelope(&fixture.envelope).unwrap();
        decoded.deployments.clear();
        let bytes = wire::ProtectedPackage::encode_to_vec(&decoded);
        assert!(
            verify(
                &bytes,
                &fixture.publishers,
                Selection::Deployment {
                    deployment: 0,
                    rank: Some(0)
                }
            )
            .is_err()
        );
        assert!(
            verify(
                &fixture.envelope,
                &fixture.publishers,
                Selection::Deployment {
                    deployment: 3,
                    rank: Some(0)
                }
            )
            .is_err()
        );
    }

    #[test]
    fn rank_body_must_name_the_selected_deployment() {
        let fixture = fixture();
        let mut body = rank_body(0);
        body.deployment_id = "other-tp2".into();
        let bytes = wire::RankBody::encode_to_vec(&body);
        let mut package =
            wire::PackageBody::decode(envelope(&fixture.envelope).unwrap().package_body.as_slice())
                .unwrap();
        let mut decoded = envelope(&fixture.envelope).unwrap();
        let mut deployment =
            wire::DeploymentBody::decode(decoded.deployments[0].body.as_slice()).unwrap();
        deployment.ranks[0].body_digest = sha256(&bytes).to_vec();
        deployment.ranks[1].body_digest = sha256(&bytes).to_vec();
        let deployment_bytes = wire::DeploymentBody::encode_to_vec(&deployment);
        decoded.deployments[0].body = deployment_bytes.clone();
        package.deployments[0].body_digest = sha256(&deployment_bytes).to_vec();
        let public_key = fixture.publishers[0].public_key;
        let mut builder = EnvelopeBuilder::new(
            wire::PackageBody::encode_to_vec(&package),
            public_key,
            fixture.package_id,
        )
        .unwrap();
        builder.deployment(0, deployment_bytes).unwrap();
        builder.rank(0, 0, bytes.clone()).unwrap();
        builder.rank(0, 1, bytes).unwrap();
        let signed = builder.sign(&fixture.signing_key).unwrap();
        assert!(
            verify(
                &signed,
                &fixture.publishers,
                Selection::Deployment {
                    deployment: 0,
                    rank: None
                }
            )
            .is_err()
        );
    }

    #[test]
    fn a_rank_cannot_use_another_ranks_payload_descriptor() {
        let fixture = fixture();
        let mut declaring = rank_body(0);
        declaring.payloads = vec![payload(9, 32)];
        let mut borrowing = rank_body(1);
        borrowing.initializers[0]
            .source
            .as_mut()
            .unwrap()
            .slice
            .as_mut()
            .unwrap()
            .payload_id = vec![9; 32];
        let rank0 = wire::RankBody::encode_to_vec(&declaring);
        let rank1 = wire::RankBody::encode_to_vec(&borrowing);
        let mut package =
            wire::PackageBody::decode(envelope(&fixture.envelope).unwrap().package_body.as_slice())
                .unwrap();
        let decoded = envelope(&fixture.envelope).unwrap();
        let mut deployment =
            wire::DeploymentBody::decode(decoded.deployments[0].body.as_slice()).unwrap();
        deployment.ranks[0].body_digest = sha256(&rank0).to_vec();
        deployment.ranks[1].body_digest = sha256(&rank1).to_vec();
        let deployment_bytes = wire::DeploymentBody::encode_to_vec(&deployment);
        package.deployments[0].body_digest = sha256(&deployment_bytes).to_vec();
        let mut builder = EnvelopeBuilder::new(
            wire::PackageBody::encode_to_vec(&package),
            fixture.publishers[0].public_key,
            fixture.package_id,
        )
        .unwrap();
        builder.deployment(0, deployment_bytes).unwrap();
        builder.rank(0, 0, rank0).unwrap();
        builder.rank(0, 1, rank1).unwrap();
        let signed = builder.sign(&fixture.signing_key).unwrap();
        // The borrowing rank is rejected whether verified alone or alongside its sibling: rank 0
        // declaring the descriptor does not extend rank 1's authority.
        for rank in [Some(1), None] {
            assert!(
                verify(
                    &signed,
                    &fixture.publishers,
                    Selection::Deployment {
                        deployment: 0,
                        rank
                    }
                )
                .is_err(),
                "{rank:?}"
            );
        }
        // The declaring rank itself remains loadable.
        assert!(
            verify(
                &signed,
                &fixture.publishers,
                Selection::Deployment {
                    deployment: 0,
                    rank: Some(0)
                }
            )
            .is_ok()
        );
    }

    #[test]
    fn duplicate_executable_resource_ids_across_ranks_are_rejected() {
        let fixture = fixture();
        let mut body = rank_body(1);
        body.resources[0].resource_id = 100; // same as rank 0
        let mut package =
            wire::PackageBody::decode(envelope(&fixture.envelope).unwrap().package_body.as_slice())
                .unwrap();
        let decoded = envelope(&fixture.envelope).unwrap();
        let mut deployment =
            wire::DeploymentBody::decode(decoded.deployments[0].body.as_slice()).unwrap();
        let bytes = wire::RankBody::encode_to_vec(&body);
        deployment.ranks[1].body_digest = sha256(&bytes).to_vec();
        let deployment_bytes = wire::DeploymentBody::encode_to_vec(&deployment);
        package.deployments[0].body_digest = sha256(&deployment_bytes).to_vec();
        let mut builder = EnvelopeBuilder::new(
            wire::PackageBody::encode_to_vec(&package),
            fixture.publishers[0].public_key,
            fixture.package_id,
        )
        .unwrap();
        builder.deployment(0, deployment_bytes).unwrap();
        builder
            .rank(0, 0, wire::RankBody::encode_to_vec(&rank_body(0)))
            .unwrap();
        builder.rank(0, 1, bytes).unwrap();
        let signed = builder.sign(&fixture.signing_key).unwrap();
        assert!(
            verify(
                &signed,
                &fixture.publishers,
                Selection::Deployment {
                    deployment: 0,
                    rank: None
                }
            )
            .is_err()
        );
        assert!(
            verify(
                &signed,
                &fixture.publishers,
                Selection::Deployment {
                    deployment: 0,
                    rank: Some(0)
                }
            )
            .is_ok()
        );
    }

    #[test]
    fn plaintext_payloads_are_authenticated_by_reading_their_bytes() {
        let fixture = fixture();
        let mut verified = verify(
            &fixture.envelope,
            &fixture.publishers,
            Selection::Deployment {
                deployment: 0,
                rank: Some(0),
            },
        )
        .unwrap();
        let bytes = [3u8; 32];
        let digest = sha256(&bytes);
        // A digest the signed catalog does not describe is not authenticated by presenting it.
        assert!(verified.verify_payload(&digest, &bytes).is_err());
        // A signed descriptor authenticates exactly its length and those bytes.
        verified
            .catalog
            .insert(&wire::PayloadDescriptor {
                id: digest.to_vec(),
                nbytes: 32,
            })
            .unwrap();
        assert!(verified.verify_payload(&digest, &bytes).is_ok());
        assert!(verified.verify_payload(&digest, &bytes[..31]).is_err());
        assert!(verified.verify_payload(&digest, &[0u8; 32]).is_err());
        assert!(verified.verify_payload(&vec![1u8; 32], &bytes).is_err());
    }

    fn read_stored(
        stored: &[u8],
        decoded_nbytes: u64,
        chunk_sha256: &[Vec<u8>],
    ) -> io::Result<Vec<u8>> {
        let mut reader = open_compressed_reader(
            std::io::Cursor::new(stored),
            stored.len() as u64,
            decoded_nbytes,
            chunk_sha256,
        )
        .map_err(io::Error::other)?;
        let mut plaintext = Vec::new();
        reader.read_to_end(&mut plaintext)?;
        Ok(plaintext)
    }

    fn pseudo_random(len: usize) -> Vec<u8> {
        let mut state = 0x9e37_79b9u32;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect()
    }

    #[test]
    fn zstd_round_trip_streams_one_bounded_frame() {
        let plaintext = pseudo_random(CHUNK + 37);
        let (stored, compressed) = compress(&plaintext).unwrap();
        assert_eq!(compressed.stored_nbytes, stored.len() as u64);
        assert!(!compressed.chunk_sha256.is_empty());
        assert_eq!(
            compressed.chunk_sha256.len() as u64,
            stored.len().div_ceil(CHUNK) as u64
        );
        for (chunk, digest) in stored.chunks(CHUNK).zip(&compressed.chunk_sha256) {
            assert_eq!(digest.as_slice(), sha256(chunk).as_slice());
        }
        assert_eq!(
            read_stored(&stored, plaintext.len() as u64, &compressed.chunk_sha256).unwrap(),
            plaintext
        );

        let mut streamed = Vec::new();
        let streamed_resource = compress_stream(
            std::io::Cursor::new(plaintext.clone()),
            plaintext.len() as u64,
            &mut streamed,
        )
        .unwrap();
        assert_eq!(streamed, stored);
        assert_eq!(streamed_resource, compressed);
    }

    #[test]
    fn short_input_and_empty_resources_are_rejected() {
        assert!(compress(&[]).is_err());
        let mut sink = Vec::new();
        assert!(compress_stream(std::io::Cursor::new(b"abc"), 4, &mut sink).is_err());
        assert!(open_compressed_reader(std::io::Cursor::new([]), 0, 1, &[]).is_err());
        assert!(
            open_compressed_reader(std::io::Cursor::new([0u8; 1]), 1, 0, &[vec![0; 32]]).is_err()
        );
    }

    #[test]
    fn truncated_extra_frames_and_trailing_bytes_are_rejected() {
        let (stored, resource) = compress(b"one authenticated zstd frame").unwrap();
        let decoded = b"one authenticated zstd frame".len() as u64;

        let truncated = &stored[..stored.len() - 1];
        let hashes = vec![sha256(truncated).to_vec()];
        assert!(read_stored(truncated, decoded, &hashes).is_err());

        let mut trailing = stored.clone();
        trailing.push(0x5a);
        let hashes = vec![sha256(&trailing[..CHUNK.min(trailing.len())]).to_vec()];
        assert!(read_stored(&trailing, decoded, &hashes).is_err());

        let mut extra_frame = stored.clone();
        extra_frame.extend_from_slice(&stored);
        let hashes = extra_frame
            .chunks(CHUNK)
            .map(|chunk| sha256(chunk).to_vec())
            .collect::<Vec<_>>();
        assert!(read_stored(&extra_frame, decoded, &hashes).is_err());
        let _ = resource;
    }

    #[test]
    fn take_through_decoded_length_rejects_trailing_and_extra_frames() {
        let plaintext = b"archive consumer frame";
        let (stored, compressed) = compress(plaintext).unwrap();
        let decoded = plaintext.len() as u64;

        let valid_reader = open_compressed_reader(
            std::io::Cursor::new(&stored),
            stored.len() as u64,
            decoded,
            &compressed.chunk_sha256,
        )
        .unwrap();
        let mut valid = valid_reader.take(decoded);
        let mut output = vec![0u8; decoded as usize];
        valid.read_exact(&mut output).unwrap();
        assert_eq!(output, plaintext);
        assert_eq!(valid.read(&mut [0u8; 1]).unwrap(), 0);

        let mut trailing = stored.clone();
        trailing.push(0x5a);
        let trailing_hashes = trailing
            .chunks(CHUNK)
            .map(|chunk| sha256(chunk).to_vec())
            .collect::<Vec<_>>();
        let trailing_reader = open_compressed_reader(
            std::io::Cursor::new(&trailing),
            trailing.len() as u64,
            decoded,
            &trailing_hashes,
        )
        .unwrap();
        let mut trailing_take = trailing_reader.take(decoded);
        let mut output = vec![0u8; decoded as usize];
        assert!(trailing_take.read_exact(&mut output).is_err());
        assert!(trailing_take.read(&mut [0u8; 1]).is_err());

        let mut extra = stored.clone();
        extra.extend_from_slice(&stored);
        let extra_hashes = extra
            .chunks(CHUNK)
            .map(|chunk| sha256(chunk).to_vec())
            .collect::<Vec<_>>();
        let extra_reader = open_compressed_reader(
            std::io::Cursor::new(&extra),
            extra.len() as u64,
            decoded,
            &extra_hashes,
        )
        .unwrap();
        let mut extra_take = extra_reader.take(decoded);
        let mut output = vec![0u8; decoded as usize];
        assert!(extra_take.read_exact(&mut output).is_err());
        assert!(extra_take.read(&mut [0u8; 1]).is_err());
    }

    #[test]
    fn decoded_length_must_match_the_signed_value() {
        let plaintext = b"decoded length";
        let (stored, resource) = compress(plaintext).unwrap();
        assert!(read_stored(&stored, plaintext.len() as u64 + 1, &resource.chunk_sha256).is_err());
        assert!(read_stored(&stored, plaintext.len() as u64 - 1, &resource.chunk_sha256).is_err());
    }

    #[test]
    fn compressed_chunks_are_authenticated_before_decoding() {
        let (stored, resource) = compress(b"chunk authentication").unwrap();
        let mut corrupted = stored.clone();
        corrupted[0] ^= 1;
        let error = read_stored(&corrupted, 20, &resource.chunk_sha256).unwrap_err();
        assert!(error.to_string().contains("chunk"), "{error}");
    }

    #[test]
    fn authenticated_source_errors_are_terminal() {
        let plaintext = b"terminal source";
        let (stored, compressed) = compress(plaintext).unwrap();
        let mut corrupted = stored.clone();
        corrupted[0] ^= 1;
        let mut reader = open_compressed_reader(
            std::io::Cursor::new(&corrupted),
            corrupted.len() as u64,
            plaintext.len() as u64,
            &compressed.chunk_sha256,
        )
        .unwrap();
        let mut output = [0u8; 32];
        assert!(reader.read(&mut output).is_err());
        assert_eq!(output, [0u8; 32]);
        assert!(reader.read(&mut output).is_err());

        let mut reader = open_compressed_reader(
            std::io::Cursor::new(&stored[..stored.len() - 1]),
            stored.len() as u64,
            plaintext.len() as u64,
            &compressed.chunk_sha256,
        )
        .unwrap();
        assert!(reader.read(&mut output).is_err());
        assert_eq!(output, [0u8; 32]);
        assert!(reader.read(&mut output).is_err());
    }

    #[test]
    fn later_compressed_chunks_are_authenticated_before_decode() {
        let plaintext = pseudo_random(CHUNK * 2 + 17);
        let (stored, compressed) = compress(&plaintext).unwrap();
        assert!(compressed.chunk_sha256.len() >= 2);
        let mut corrupted = stored.clone();
        corrupted[CHUNK] ^= 1;
        let mut reader = open_compressed_reader(
            std::io::Cursor::new(&corrupted),
            corrupted.len() as u64,
            plaintext.len() as u64,
            &compressed.chunk_sha256,
        )
        .unwrap();
        let mut output = Vec::new();
        assert!(reader.read_to_end(&mut output).is_err());
        let mut output = [0u8; 16];
        assert!(reader.read(&mut output).is_err());
    }

    #[test]
    fn zstd_checksum_corruption_is_rejected() {
        let plaintext = pseudo_random(1024);
        let (stored, _resource) = compress(&plaintext).unwrap();
        let mut corrupted = stored.clone();
        let index = corrupted.len() - 2;
        corrupted[index] ^= 1;
        let hashes = corrupted
            .chunks(CHUNK)
            .map(|chunk| sha256(chunk).to_vec())
            .collect::<Vec<_>>();
        assert!(read_stored(&corrupted, plaintext.len() as u64, &hashes).is_err());
    }

    #[test]
    fn compressed_metadata_bounds_are_checked() {
        let (stored, resource) = compress(b"metadata").unwrap();
        assert!(
            open_compressed_reader(
                std::io::Cursor::new(&stored),
                stored.len() as u64,
                8,
                &resource.chunk_sha256[..resource.chunk_sha256.len() - 1],
            )
            .is_err()
        );
        assert!(
            open_compressed_reader(
                std::io::Cursor::new(&stored),
                stored.len() as u64,
                8,
                &vec![vec![0; 31]],
            )
            .is_err()
        );
        assert!(
            open_compressed_reader(
                std::io::Cursor::new(&stored),
                MAX_RESOURCE_BYTES + 1,
                8,
                &resource.chunk_sha256,
            )
            .is_err()
        );
        assert!(
            open_compressed_reader(
                std::io::Cursor::new(&stored),
                stored.len() as u64,
                MAX_RESOURCE_BYTES + 1,
                &resource.chunk_sha256,
            )
            .is_err()
        );
    }

    #[test]
    fn decoder_rejects_frames_above_the_window_cap() {
        let plaintext = pseudo_random(CHUNK * 2 + 1);
        let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), COMPRESSION_LEVEL).unwrap();
        encoder
            .set_pledged_src_size(Some(plaintext.len() as u64))
            .unwrap();
        encoder.include_checksum(true).unwrap();
        encoder.include_contentsize(true).unwrap();
        encoder.window_log(WINDOW_LOG + 1).unwrap();
        encoder.write_all(&plaintext).unwrap();
        let stored = encoder.finish().unwrap();
        let hashes = stored
            .chunks(CHUNK)
            .map(|chunk| sha256(chunk).to_vec())
            .collect::<Vec<_>>();
        let mut reader = open_compressed_reader(
            std::io::Cursor::new(&stored),
            stored.len() as u64,
            plaintext.len() as u64,
            &hashes,
        )
        .unwrap();
        let mut output = Vec::new();
        assert!(reader.read_to_end(&mut output).is_err());
    }

    #[test]
    fn authorized_package_opens_the_zstd_resource() {
        let fixture = fixture();
        let verified = |accepted: &str| {
            verify(
                &fixture.envelope,
                &fixture.publishers,
                Selection::Deployment {
                    deployment: 0,
                    rank: Some(0),
                },
            )
            .unwrap()
            .authorize(accepted)
        };
        assert!(verified("other").is_err());
        let authorized = verified("example").unwrap();
        let plaintext = b"authorized executable";
        let (stored, compressed) = compress(plaintext).unwrap();
        let resource = wire::ExecutableResource {
            resource_id: 7,
            kind: wire::ExecutableKind::Cubin as i32,
            stored: Some(slice(1, 0, stored.len() as u64)),
            decoded_nbytes: plaintext.len() as u64,
            chunk_nbytes: CHUNK as u32,
            chunk_sha256: compressed.chunk_sha256.clone(),
        };
        assert_eq!(
            authorized.open_resource(&resource, &stored).unwrap(),
            plaintext
        );
        let mut streamed = Vec::new();
        authorized
            .write_plaintext(&resource, std::io::Cursor::new(&stored), &mut streamed)
            .unwrap();
        assert_eq!(streamed, plaintext);

        let mut reader = authorized
            .open_reader(&resource, std::io::Cursor::new(&stored))
            .unwrap();
        let mut read = Vec::new();
        reader.read_to_end(&mut read).unwrap();
        assert_eq!(read, plaintext);

        let mut wrong = resource.clone();
        wrong.decoded_nbytes += 1;
        assert!(authorized.open_resource(&wrong, &stored).is_err());
        let mut wrong_chunk = resource.clone();
        wrong_chunk.chunk_nbytes = (CHUNK - 1) as u32;
        assert!(authorized.open_resource(&wrong_chunk, &stored).is_err());
        let mut bad_chunk = stored.clone();
        bad_chunk[0] ^= 1;
        assert!(authorized.open_resource(&resource, &bad_chunk).is_err());
    }
}
