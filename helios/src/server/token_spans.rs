/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Attribute delayed decoder text to its tokens without retaining prefix copies.

use crate::{Result, execution::TokenLogprobs, frontend::{Frontend, IncrementalDecoder}};

#[derive(Clone, Debug, PartialEq)]
pub struct ReturnedLogprob {
    pub value: TokenLogprobs,
    pub start: usize,
    pub text_offset: usize,
    pub silent: bool,
}

pub(super) struct SpanToken {
    pub token: u32,
    pub start: usize,
    pub logprob: Option<ReturnedLogprob>,
}
pub(super) struct Span {
    pub text: String,
    pub tokens: Vec<SpanToken>,
}
struct Pending<'a> {
    token: u32,
    value: Option<TokenLogprobs>,
    start: usize,
    bytes: &'a [u8],
    silent: bool,
}
pub(super) struct TokenSpans<'a> {
    frontend: &'a Frontend,
    decoder: IncrementalDecoder<'a>,
    pending: Vec<Pending<'a>>,
    provisional: String,
    bytes: usize,
    chars: usize,
}
impl<'a> TokenSpans<'a> {
    pub fn new(frontend: &'a Frontend) -> Self {
        Self {
            frontend,
            decoder: frontend.incremental_decoder(false),
            pending: Vec::new(),
            provisional: String::new(),
            bytes: 0,
            chars: 0,
        }
    }
    pub fn push(&mut self, token: u32, value: Option<TokenLogprobs>) -> Result<Option<Span>> {
        self.pending.push(Pending {
            token, value, start: self.provisional.len(),
            bytes: self.frontend.token_bytes(token)?, silent: false,
        });
        match self.decoder.push(&[token])? {
            Some(text) => {
                self.revise(&text);
                Ok(Some(self.span(text)))
            }
            None => {
                let text = self.decoder.pending_text()?;
                self.revise(&text);
                self.provisional = text;
                Ok(None)
            }
        }
    }
    pub fn silent(&mut self, token: u32, value: Option<TokenLogprobs>) -> Option<Span> {
        self.pending.push(Pending {
            token, value, start: self.provisional.len(), bytes: &[], silent: true,
        });
        (self.pending.len() == 1).then(|| self.span(String::new()))
    }
    pub fn finish(&mut self) -> Result<Span> {
        let text = self.decoder.finish()?;
        self.revise(&text);
        Ok(self.span(text))
    }
    fn revise(&mut self, text: &str) {
        let common = self.provisional.chars().zip(text.chars()).take_while(|(a,b)| a == b)
            .map(|(c,_)| c.len_utf8()).sum::<usize>();
        if common < self.provisional.len() {
            for pending in &mut self.pending {
                pending.start = pending.start.min(common);
            }
        }
    }
    fn span(&mut self, text: String) -> Span {
        // Exact byte ranges distinguish a literal U+FFFD from provisional UTF-8
        // replacement, and map every byte token of one character to its start.
        let mapped = byte_starts(&self.pending, &text);
        let mut previous = 0;
        let mut characters = 0;
        let tokens = self.pending.drain(..).enumerate().map(|(i, pending)| {
            let (start, silent) = mapped.as_ref()
                .map_or((pending.start, pending.silent), |m| m[i]);
            characters += text[previous..start].chars().count();
            previous = start;
            SpanToken {
                token: pending.token,
                start,
                logprob: pending.value.map(|value| ReturnedLogprob {
                    value, start: self.bytes + start,
                    text_offset: self.chars + characters, silent,
                }),
            }
        }).collect();
        self.provisional.clear();
        self.bytes += text.len();
        self.chars += text.chars().count();
        Span { text, tokens }
    }
}

fn byte_starts(pending: &[Pending<'_>], text: &str) -> Option<Vec<(usize, bool)>> {
    let raw: Vec<u8> = pending.iter().flat_map(|p| p.bytes.iter().copied()).collect();
    let decoded = String::from_utf8_lossy(&raw);
    let removed = decoded.len().checked_sub(text.len())?;
    if decoded.get(removed..)? != text || !decoded[..removed].chars().all(char::is_whitespace) {
        return None;
    }
    // Each range is (raw end, decoded start, decoded end). Invalid byte groups
    // occupy one replacement character, exactly as the matched decoded text.
    let mut ranges = Vec::new();
    let (mut source, mut output) = (0, 0);
    while source < raw.len() {
        let error = std::str::from_utf8(&raw[source..]).err();
        let valid = error.map_or(raw.len() - source, |e| e.valid_up_to());
        for character in std::str::from_utf8(&raw[source..source + valid]).ok()?.chars() {
            source += character.len_utf8();
            ranges.push((source, output, output + character.len_utf8()));
            output += character.len_utf8();
        }
        if let Some(error) = error {
            source += error.error_len().unwrap_or(raw.len() - source);
            ranges.push((source, output, output + char::REPLACEMENT_CHARACTER.len_utf8()));
            output += char::REPLACEMENT_CHARACTER.len_utf8();
        }
    }
    let stripped = ranges.iter().take_while(|(_, _, end)| *end <= removed)
        .last().map_or(0, |r| r.0);
    let (mut source, mut range) = (0, 0);
    Some(pending.iter().map(|pending| {
        while range < ranges.len() && ranges[range].0 <= source {
            range += 1;
        }
        let start = ranges.get(range).map_or(decoded.len(), |r| r.1)
            .saturating_sub(removed);
        source += pending.bytes.len();
        (start, pending.silent || pending.bytes.is_empty() || source <= stripped)
    }).collect())
}
