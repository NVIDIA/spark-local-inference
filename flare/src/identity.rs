/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Who a core's signing key belongs to, resolved from the publisher's own repository.
//!
//! A core claims a signer fingerprint, a namespace, and an official source under that
//! namespace on huggingface.co. The publisher proves the claim by publishing `publisher.json`,
//! the profile `ignition keygen` writes, at the root of that repository. Every publisher is
//! verified the same way; no key is privileged. A key file that exists but omits the signer is
//! an impersonation or a revocation and fails loading outright.

use flare_helios::runtime_core::{CoreClaim, Publisher};
use std::time::Duration;

use crate::options::HubOptions;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const PROFILE_LIMIT: u64 = 65536;

/// What the publisher's repository said about the signing key.
#[derive(Debug, PartialEq, Eq)]
pub enum Resolution {
    /// The key file at the official source lists the signer under the claimed namespace.
    Verified(Publisher),
    /// The official source publishes no key file.
    Absent,
    /// The official source could not be consulted: offline, network failure, or a server error.
    Unreachable(String),
}

/// The URL of the key file: the official source's default branch, root `publisher.json`.
pub fn profile_url(official_source: &str) -> String {
    format!("{official_source}/resolve/main/publisher.json")
}

/// What one request for the key file produced, before interpretation.
pub enum Fetched {
    Body(Vec<u8>),
    Status(u16),
    Failed(String),
}

pub fn resolve(claim: &CoreClaim, options: &HubOptions) -> Result<Resolution> {
    if options.offline {
        return Ok(Resolution::Unreachable("not consulted (--offline)".into()));
    }
    let url = profile_url(&claim.official_source);
    let fetched = fetch(&url, crate::hub::token(options)?.as_deref());
    classify(claim, &url, fetched)
}

fn fetch(url: &str, token: Option<&str>) -> Fetched {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(10)))
        .build()
        .into();
    let mut request = agent.get(url);
    if let Some(token) = token {
        request = request.header("Authorization", &format!("Bearer {token}"));
    }
    match request.call() {
        Ok(mut response) => {
            let status = response.status().as_u16();
            if !(200..300).contains(&status) {
                return Fetched::Status(status);
            }
            match response
                .body_mut()
                .with_config()
                .limit(PROFILE_LIMIT)
                .read_to_vec()
            {
                Ok(body) => Fetched::Body(body),
                Err(error) => Fetched::Failed(error.to_string()),
            }
        }
        Err(error) => Fetched::Failed(error.to_string()),
    }
}

/// Interpret the key file. A readable file decides the claim one way or the other; only the
/// absence of a file or of a connection leaves the identity unresolved.
pub fn classify(claim: &CoreClaim, url: &str, fetched: Fetched) -> Result<Resolution> {
    let body = match fetched {
        Fetched::Body(body) => body,
        Fetched::Status(404) => return Ok(Resolution::Absent),
        // huggingface.co answers 401 for a repository that does not exist or is not public.
        Fetched::Status(status @ (401 | 403)) => {
            return Ok(Resolution::Unreachable(format!(
                "HTTP {status}, the repository does not exist or is not public"
            )));
        }
        Fetched::Status(status) => return Ok(Resolution::Unreachable(format!("HTTP {status}"))),
        Fetched::Failed(reason) => return Ok(Resolution::Unreachable(reason)),
    };
    let profiles: Vec<Publisher> = serde_json::from_slice(&body).map_err(|error| {
        format!("core rejected: {url} is not a Flare publisher profile: {error}")
    })?;
    for profile in &profiles {
        profile.validate().map_err(|error| {
            format!("core rejected: {url} holds an invalid publisher profile: {error}")
        })?;
    }
    let Some(profile) = profiles.iter().find(|p| p.fingerprint() == claim.signer) else {
        return Err(format!(
            "core rejected: signing key {} is not among the keys {} publishes at {url}; the core was not \
             signed by its claimed publisher, or the key was revoked",
            claim.signer, claim.namespace
        )
        .into());
    };
    if profile.namespace != claim.namespace || profile.development {
        return Err(format!(
            "core rejected: {url} publishes signing key {} under namespace {}, not the core's {}",
            claim.signer, profile.namespace, claim.namespace
        )
        .into());
    }
    Ok(Resolution::Verified(profile.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim(signer: &str) -> CoreClaim {
        CoreClaim {
            signer: signer.into(),
            namespace: "example".into(),
            official_source: "https://huggingface.co/example/model".into(),
            development: false,
        }
    }
    fn profile(namespace: &str, key: u8) -> Publisher {
        Publisher {
            name: "Example".into(),
            namespace: namespace.into(),
            public_key: [key; 32],
            development: false,
        }
    }
    fn body(profiles: &[Publisher]) -> Fetched {
        Fetched::Body(serde_json::to_vec(profiles).unwrap())
    }
    const URL: &str = "https://huggingface.co/example/model/resolve/main/publisher.json";

    #[test]
    fn a_listed_key_verifies_and_rotation_lists_remain_valid() {
        let current = profile("example", 7);
        let previous = profile("example", 9);
        let verified = classify(
            &claim(&previous.fingerprint()),
            URL,
            body(&[current.clone(), previous.clone()]),
        )
        .unwrap();
        assert_eq!(verified, Resolution::Verified(previous));
        assert_eq!(
            classify(
                &claim(&current.fingerprint()),
                URL,
                body(&[current.clone()])
            )
            .unwrap(),
            Resolution::Verified(current)
        );
    }

    #[test]
    fn an_unlisted_key_fails_outright() {
        let listed = profile("example", 7);
        let error = classify(
            &claim(&profile("example", 8).fingerprint()),
            URL,
            body(&[listed]),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not among the keys example publishes"),
            "{error}"
        );
    }

    #[test]
    fn a_key_listed_under_another_namespace_fails_outright() {
        let other = profile("someone-else", 7);
        let error = classify(&claim(&other.fingerprint()), URL, body(&[other])).unwrap_err();
        assert!(
            error.to_string().contains("under namespace someone-else"),
            "{error}"
        );
    }

    #[test]
    fn a_file_that_is_not_a_profile_fails_outright() {
        let error = classify(
            &claim("00"),
            URL,
            Fetched::Body(b"{\"unrelated\": true}".to_vec()),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("not a Flare publisher profile"),
            "{error}"
        );
    }

    #[test]
    fn absence_and_unreachability_leave_identity_unresolved() {
        assert_eq!(
            classify(&claim("00"), URL, Fetched::Status(404)).unwrap(),
            Resolution::Absent
        );
        assert_eq!(
            classify(&claim("00"), URL, Fetched::Status(503)).unwrap(),
            Resolution::Unreachable("HTTP 503".into())
        );
        assert_eq!(
            classify(&claim("00"), URL, Fetched::Failed("dns".into())).unwrap(),
            Resolution::Unreachable("dns".into())
        );
        assert_eq!(
            classify(&claim("00"), URL, Fetched::Status(401)).unwrap(),
            Resolution::Unreachable(
                "HTTP 401, the repository does not exist or is not public".into()
            )
        );
    }

    #[test]
    fn the_key_file_is_the_default_branch_root_of_the_official_source() {
        assert_eq!(profile_url("https://huggingface.co/example/model"), URL);
    }
}
