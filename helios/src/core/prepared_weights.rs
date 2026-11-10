/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lossless encoding of prepared weight catalogs inside signed rank metadata.

use super::{
    invalid,
    schema::{self, wire},
};
use crate::Result;
use prost::Message;
use std::{
    borrow::Cow,
    io::{Cursor, Read, Write},
};

const WINDOW_LOG: u32 = 23;

/// Check the catalog representation without decompressing untrusted bytes.
pub fn validate_wrapper(body: &wire::RankBody) -> Result<()> {
    if let Some(compressed) = &body.compressed_prepared_weights {
        if body.prepared_weights.is_some() {
            return Err(invalid(
                "rank has both inline and compressed prepared weight catalogs",
            ));
        }
        if compressed.zstd.is_empty()
            || compressed.zstd.len() > schema::MAX_BODY
            || compressed.decoded_nbytes == 0
            || compressed.decoded_nbytes > schema::MAX_BODY as u64
        {
            return Err(invalid("invalid compressed prepared weight catalog length"));
        }
    }
    Ok(())
}

fn field_nbytes(tag: u32, payload: u64) -> Result<u64> {
    let prefix =
        prost::encoding::key_len(tag) as u64 + prost::encoding::encoded_len_varint(payload) as u64;
    payload
        .checked_add(prefix)
        .ok_or_else(|| invalid("prepared weight metadata length overflow"))
}

/// Bound the rank metadata with its catalog reconstructed in the inline representation.
/// The original signed length retains unknown fields and noncanonical wire overhead.
pub fn reconstructed_rank_nbytes(body: &wire::RankBody, signed_nbytes: usize) -> Result<u64> {
    validate_wrapper(body)?;
    let mut nbytes = signed_nbytes as u64;
    if let Some(compressed) = &body.compressed_prepared_weights {
        let stored_field = field_nbytes(17, compressed.encoded_len() as u64)?;
        let decoded_field = field_nbytes(16, compressed.decoded_nbytes)?;
        nbytes = nbytes
            .checked_sub(stored_field)
            .and_then(|value| value.checked_add(decoded_field))
            .ok_or_else(|| invalid("prepared weight metadata length overflow"))?;
    }
    if nbytes > schema::MAX_BODY as u64 {
        return Err(invalid("reconstructed rank body exceeds its limit"));
    }
    Ok(nbytes)
}

/// Read a catalog from authenticated rank metadata or trusted preparation input.
/// Compressed catalogs are temporary owners; inline catalogs remain borrowed.
pub fn catalog(body: &wire::RankBody) -> Result<Option<Cow<'_, wire::PreparedWeightCatalog>>> {
    reconstructed_rank_nbytes(body, body.encoded_len())?;
    if let Some(catalog) = &body.prepared_weights {
        return Ok(Some(Cow::Borrowed(catalog)));
    }
    let Some(compressed) = &body.compressed_prepared_weights else {
        return Ok(None);
    };
    let mut decoder =
        zstd::stream::read::Decoder::with_buffer(Cursor::new(compressed.zstd.as_slice()))
            .map_err(|error| invalid(format!("create prepared weight decompressor: {error}")))?
            .single_frame();
    decoder
        .window_log_max(WINDOW_LOG)
        .map_err(|error| invalid(format!("set prepared weight decoder window: {error}")))?;
    let mut decoded = vec![0; compressed.decoded_nbytes as usize];
    decoder
        .read_exact(&mut decoded)
        .map_err(|error| invalid(format!("decode prepared weight catalog: {error}")))?;
    let mut extra = [0u8; 1];
    if decoder
        .read(&mut extra)
        .map_err(|error| invalid(format!("finish prepared weight catalog: {error}")))?
        != 0
    {
        return Err(invalid(
            "prepared weight catalog decoded length differs from its declaration",
        ));
    }
    if decoder.finish().position() != compressed.zstd.len() as u64 {
        return Err(invalid(
            "compressed prepared weight catalog has trailing bytes",
        ));
    }
    let catalog = wire::PreparedWeightCatalog::decode(decoded.as_slice())
        .map_err(|error| invalid(format!("invalid prepared weight catalog: {error}")))?;
    Ok(Some(Cow::Owned(catalog)))
}

/// Replace an inline catalog with its zstd encoding when that reduces signed metadata.
pub fn compact(body: &mut wire::RankBody) -> Result<()> {
    reconstructed_rank_nbytes(body, body.encoded_len())?;
    let Some(catalog) = &body.prepared_weights else {
        return Ok(());
    };
    let decoded = catalog.encode_to_vec();
    let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 3)
        .map_err(|error| invalid(format!("create prepared weight compressor: {error}")))?;
    encoder
        .window_log(WINDOW_LOG)
        .map_err(|error| invalid(format!("set prepared weight compressor window: {error}")))?;
    encoder
        .set_pledged_src_size(Some(decoded.len() as u64))
        .map_err(|error| invalid(format!("set prepared weight catalog length: {error}")))?;
    encoder
        .include_checksum(true)
        .map_err(|error| invalid(format!("set prepared weight checksum: {error}")))?;
    encoder
        .include_contentsize(true)
        .map_err(|error| invalid(format!("set prepared weight content size: {error}")))?;
    encoder
        .write_all(&decoded)
        .map_err(|error| invalid(format!("compress prepared weight catalog: {error}")))?;
    let compressed = wire::CompressedPreparedWeightCatalog {
        zstd: encoder
            .finish()
            .map_err(|error| invalid(format!("finish prepared weight compression: {error}")))?,
        decoded_nbytes: decoded.len() as u64,
    };
    if field_nbytes(17, compressed.encoded_len() as u64)? < field_nbytes(16, decoded.len() as u64)?
    {
        body.prepared_weights = None;
        body.compressed_prepared_weights = Some(compressed);
    }
    Ok(())
}

#[cfg(test)]
#[path = "prepared_weights_tests.rs"]
mod tests;
