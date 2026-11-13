/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Token bytes before UTF-8 replacement and stream-boundary whitespace cleanup.

use crate::Result;
use tokenizers::{Decoder, Tokenizer, decoders::DecoderWrapper};

pub(super) struct VocabularyBytes(Vec<Vec<u8>>);

impl VocabularyBytes {
    pub fn new(tokenizer: &Tokenizer) -> Result<Self> {
        let mut mapping = std::collections::HashMap::new();
        let mut extension = u32::from(u8::MAX) + 1;
        for byte in 0..=u32::from(u8::MAX) {
            // GPT-2 byte-level encoding preserves these visible Latin-1 ranges.
            let visible = (33..=126).contains(&byte)
                || (161..=172).contains(&byte)
                || (174..=255).contains(&byte);
            let code = if visible {
                byte
            } else {
                let n = extension;
                extension += 1;
                n
            };
            mapping.insert(char::from_u32(code).unwrap(), byte as u8);
        }
        let mut result = Vec::with_capacity(tokenizer.get_vocab_size(true));
        for id in 0..tokenizer.get_vocab_size(true) as u32 {
            let token = tokenizer.id_to_token(id).unwrap_or_default().into_bytes();
            result.push(match tokenizer.get_decoder() {
                Some(decoder) => decode_piece(decoder, token, &mapping)?,
                None => token,
            });
        }
        Ok(Self(result))
    }
    pub fn get(&self, token: u32) -> Result<&[u8]> {
        self.0.get(token as usize).map(Vec::as_slice)
            .ok_or_else(|| "token ID exceeds tokenizer vocabulary".into())
    }
}

fn decode_piece(
    decoder: &DecoderWrapper,
    mut bytes: Vec<u8>,
    mapping: &std::collections::HashMap<char, u8>,
) -> Result<Vec<u8>> {
    match decoder {
        DecoderWrapper::Sequence(sequence) => {
            for decoder in sequence.get_decoders() {
                bytes = decode_piece(decoder, bytes, mapping)?;
            }
            Ok(bytes)
        }
        // These act on the stream; they do not change a token's own bytes.
        DecoderWrapper::Fuse(_) | DecoderWrapper::Strip(_) => Ok(bytes),
        DecoderWrapper::ByteFallback(_) => {
            if let Ok(token) = std::str::from_utf8(&bytes) {
                // Byte fallback tokens use the six-byte spelling "<0xHH>".
                if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                    if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                        return Ok(vec![byte]);
                    }
                }
            }
            Ok(bytes)
        }
        DecoderWrapper::ByteLevel(_) => {
            let text = String::from_utf8(bytes)
                .map_err(|_| "byte-level decoder requires encoded Unicode pieces")?;
            let mut output = Vec::new();
            for character in text.chars() {
                if let Some(byte) = mapping.get(&character) {
                    output.push(*byte);
                } else {
                    output.extend(character.to_string().bytes());
                }
            }
            Ok(output)
        }
        DecoderWrapper::Metaspace(space) => {
            let text = String::from_utf8(bytes)
                .map_err(|_| "metaspace precedes byte decoding")?;
            Ok(text.replace(space.get_replacement(), " ").into_bytes())
        }
        DecoderWrapper::WordPiece(piece) => {
            let text = String::from_utf8(bytes)
                .map_err(|_| "wordpiece requires Unicode pieces")?;
            Ok(text.strip_prefix(&piece.prefix).unwrap_or(&text)
                .as_bytes().to_vec())
        }
        DecoderWrapper::BPE(piece) => {
            let text = String::from_utf8(bytes)
                .map_err(|_| "BPE suffix decoding requires Unicode pieces")?;
            Ok(text.replace(&piece.suffix, " ").into_bytes())
        }
        _ => {
            let text = String::from_utf8(bytes)
                .map_err(|_| "text decoder follows a partial UTF-8 byte token")?;
            Ok(decoder.decode_chain(vec![text]).map_err(|e| e.to_string())?
                .concat().into_bytes())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::{BPE, Vocab};
    #[test]
    fn byte_level_and_byte_fallback_preserve_partial_utf8_and_sentencepiece_spaces() {
        for (pieces, decoder, expected) in [
            (
                vec!["ä", "¸", "Ġword"],
                serde_json::json!({
                    "type": "ByteLevel", "add_prefix_space": false,
                    "trim_offsets": false, "use_regex": false,
                }),
                vec![vec![0xe4], vec![0xb8], b" word".to_vec()],
            ),
            (
                vec!["<0xE4>", "<0xB8>", "▁word"],
                serde_json::json!({"type": "Sequence", "decoders": [
                    {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                    {"type": "ByteFallback"}, {"type": "Fuse"},
                    {"type": "Strip", "content": " ", "start": 1, "stop": 0},
                ]}),
                vec![vec![0xe4], vec![0xb8], b" word".to_vec()],
            ),
        ] {
            let vocab: Vocab = pieces.iter().enumerate()
                .map(|(i, p)| ((*p).to_owned(), i as u32)).collect();
            let mut tokenizer = Tokenizer::new(
                BPE::builder().vocab_and_merges(vocab, vec![]).build().unwrap(),
            );
            tokenizer.with_decoder(Some(serde_json::from_value::<DecoderWrapper>(decoder).unwrap()));
            let bytes = VocabularyBytes::new(&tokenizer).unwrap();
            for (id, expected) in expected.iter().enumerate() {
                assert_eq!(bytes.get(id as u32).unwrap(), expected);
            }
        }
    }
}
