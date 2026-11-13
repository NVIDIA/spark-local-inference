/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Built-in publisher identities and additional release-provisioned profiles.

use super::{Publisher, format::invalid};
use crate::Result;

const SLOT_SIZE: usize = 65536;
const MAGIC: &[u8] = b"FLARE-PUBLISHERS-V3";
const fn empty_slot() -> [u8; SLOT_SIZE] {
    let mut result = [0; SLOT_SIZE];
    let mut i = 0;
    while i < MAGIC.len() {
        result[i] = MAGIC[i];
        i += 1;
    }
    result
}
#[used]
#[unsafe(link_section = ".flare_publishers")]
static EMBEDDED: [u8; SLOT_SIZE] = empty_slot();

pub(super) fn embedded() -> Result<Vec<Publisher>> {
    // The release stamper fills this ELF section after linking. Volatile reads prevent
    // the compiler from replacing its contents with the source-build placeholders.
    let pointer = std::ptr::addr_of!(EMBEDDED).cast::<u8>();
    let slot: Vec<u8> = (0..SLOT_SIZE)
        .map(|i| unsafe { std::ptr::read_volatile(pointer.add(i)) })
        .collect();
    if &slot[..MAGIC.len()] != MAGIC {
        return Err(invalid("invalid embedded publisher section"));
    }
    let length = u32::from_le_bytes(slot[32..36].try_into().unwrap()) as usize;
    if length > SLOT_SIZE - 36 {
        return Err(invalid("embedded publisher section exceeds capacity"));
    }
    let mut publishers: Vec<Publisher> = serde_json::from_slice(include_bytes!("publishers.json"))
        .map_err(|e| invalid(e.to_string()))?;
    if length != 0 {
        let additional: Vec<Publisher> =
            serde_json::from_slice(&slot[36..36 + length]).map_err(|e| invalid(e.to_string()))?;
        for publisher in additional {
            if let Some(previous) = publishers
                .iter()
                .find(|p| p.public_key == publisher.public_key)
            {
                if previous != &publisher {
                    return Err(invalid(
                        "conflicting embedded identities for the same publisher key",
                    ));
                }
            } else {
                publishers.push(publisher);
            }
        }
    }
    for publisher in &publishers {
        publisher.validate()?;
    }
    Ok(publishers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_build_has_no_embedded_publishers() {
        assert!(embedded().unwrap().is_empty());
    }
}
