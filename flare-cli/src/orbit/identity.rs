/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Node and controller identities: one Ed25519 key each, stored as a 32-byte seed.
//!
//! A key signs launch grants and controller-set records, and authenticates the session handshake
//! by signing its transcript. The fingerprint is the lowercase hex SHA-256 of the public key, the
//! convention core publishers use.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use ring::signature::{ED25519, Ed25519KeyPair, KeyPair, UnparsedPublicKey};

use super::{Failure, hex, sha256, unhex32};

/// A private identity.
pub struct Key {
    pair: Ed25519KeyPair,
    seed: [u8; 32],
}

impl Key {
    pub fn generate() -> Result<Self, Failure> {
        Self::from_seed(&super::random(32)?.try_into().unwrap())
    }

    pub fn from_seed(seed: &[u8; 32]) -> Result<Self, Failure> {
        let pair = Ed25519KeyPair::from_seed_unchecked(seed).map_err(|e| e.to_string())?;
        Ok(Self { pair, seed: *seed })
    }

    /// Read the key at `path`, or create it there with owner-only permissions.
    pub fn load_or_create(path: &Path) -> Result<Self, Failure> {
        match fs::metadata(path) {
            Ok(metadata) => {
                if metadata.permissions().mode() & 0o077 != 0 {
                    return Err(
                        format!("key {} must not be readable by others", path.display()).into(),
                    );
                }
                let text = fs::read_to_string(path)?;
                Self::from_seed(&unhex32(text.trim())?)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let key = Self::generate()?;
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                    fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
                }
                let temporary = path.with_extension("tmp");
                fs::write(&temporary, format!("{}\n", hex(&key.seed)))?;
                fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600))?;
                fs::rename(&temporary, path)?;
                Ok(key)
            }
            Err(error) => Err(format!("cannot read key {}: {error}", path.display()).into()),
        }
    }

    pub fn public(&self) -> PublicKey {
        PublicKey(self.pair.public_key().as_ref().try_into().unwrap())
    }

    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        self.pair.sign(message).as_ref().try_into().unwrap()
    }
}

/// A public identity.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct PublicKey([u8; 32]);

impl PublicKey {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Failure> {
        Ok(Self(
            bytes.try_into().map_err(|_| "a public key is 32 bytes")?,
        ))
    }

    pub fn from_hex(text: &str) -> Result<Self, Failure> {
        Ok(Self(unhex32(text)?))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex(&self.0)
    }

    /// Lowercase hex SHA-256 of the public key.
    pub fn fingerprint(&self) -> String {
        hex(&sha256(&self.0))
    }

    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), Failure> {
        UnparsedPublicKey::new(&ED25519, &self.0)
            .verify(message, signature)
            .map_err(|_| "signature does not verify".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signatures_verify_and_fingerprints_are_sha256_of_the_public_key() {
        let key = Key::from_seed(&[7; 32]).unwrap();
        let signature = key.sign(b"grant");
        key.public().verify(b"grant", &signature).unwrap();
        assert!(key.public().verify(b"other", &signature).is_err());
        assert!(
            Key::from_seed(&[8; 32])
                .unwrap()
                .public()
                .verify(b"grant", &signature)
                .is_err()
        );
        assert_eq!(
            key.public().fingerprint(),
            hex(&sha256(key.public().as_bytes()))
        );
        assert_eq!(
            PublicKey::from_hex(&key.public().to_hex()).unwrap(),
            key.public()
        );
    }

    #[test]
    fn keys_are_created_private_and_reloaded() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("agent/node.key");
        let created = Key::load_or_create(&path).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let reloaded = Key::load_or_create(&path).unwrap();
        assert_eq!(created.public(), reloaded.public());
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(Key::load_or_create(&path).is_err());
    }
}
