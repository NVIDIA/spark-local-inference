/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

const TOKEN_BOUND: usize = 16;

#[kani::proof]
#[kani::unwind(20)]
fn completion_vector_validation_is_safe_and_sound() {
    let len = usize::from(kani::any::<u8>());
    kani::assume(len <= TOKEN_BOUND);
    let raw_tokens: [i64; TOKEN_BOUND] = kani::any();
    let mut tokens = Vec::from(raw_tokens);
    tokens.truncate(len);
    let vector = TokenVector {
        emitted_count: kani::any(),
        offered_drafts: kani::any(),
        accepted_drafts: kani::any(),
        copy_offered_mask: kani::any(),
        model_reserved: kani::any(),
        draft_cache_consumed: kani::any(),
        verification_width: kani::any(),
        tokens,
    };
    let mut record = CompletionRecord {
        status: kani::any(),
        token_id: kani::any(),
        vector: Some(vector),
        ..Default::default()
    };
    let _detached_vector = if kani::any::<bool>() {
        record.vector.take()
    } else {
        None
    };
    let accepted = record.validate_vector().is_ok();
    if accepted && record.status != 0 {
        if let Some(vector) = &record.vector {
            assert!((1..=len as i64).contains(&vector.verification_width));
            assert!((0..=vector.verification_width).contains(&vector.emitted_count));
            assert!((0..vector.verification_width).contains(&vector.offered_drafts));
            assert!(vector.accepted_drafts >= vector.emitted_count.saturating_sub(1).max(0));
            assert!(vector.accepted_drafts <= vector.emitted_count);
            assert!(vector.accepted_drafts <= vector.offered_drafts);
            assert!(vector.copy_offered_mask >= 0);
            let provenance_bits = vector.offered_drafts.min(63) as u32;
            assert_eq!((vector.copy_offered_mask as u64) >> provenance_bits, 0);
            assert!(vector.draft_cache_consumed >= -1);
            let emitted = vector.emitted_count as usize;
            let index = usize::from(kani::any::<u8>());
            if index < len {
                if index < emitted {
                    assert!((0..=u32::MAX as i64).contains(&vector.tokens[index]));
                } else {
                    assert_eq!(vector.tokens[index], -1);
                }
            }
            assert_eq!(
                record.token_id,
                if emitted == 0 {
                    -1
                } else {
                    vector.tokens[emitted - 1]
                }
            );
            assert!(record.status != 3 || emitted == 0);
        }
    }
}

#[kani::proof]
#[kani::unwind(20)]
fn well_formed_completion_vectors_are_accepted() {
    let len = usize::from(kani::any::<u8>());
    let width: u8 = kani::any();
    let emitted: u8 = kani::any();
    let offered: u8 = kani::any();
    let accepted: u8 = kani::any();
    kani::assume((1..=16).contains(&len));
    kani::assume(width > 0 && usize::from(width) <= len);
    kani::assume(emitted <= width && offered < width);
    kani::assume(accepted <= emitted && accepted <= offered);
    kani::assume(accepted >= emitted.saturating_sub(1));
    let copy_mask: i64 = kani::any();
    kani::assume(copy_mask >= 0);
    kani::assume((copy_mask as u64) >> offered.min(63) == 0);
    let raw_tokens: [u32; 16] = kani::any();
    let mut token_storage = [-1; 16];
    for index in 0..16 {
        if index < usize::from(emitted) {
            token_storage[index] = i64::from(raw_tokens[index]);
        }
    }
    let mut tokens = Vec::from(token_storage);
    tokens.truncate(len);
    let record = CompletionRecord {
        status: 1,
        token_id: if emitted == 0 {
            -1
        } else {
            tokens[usize::from(emitted) - 1]
        },
        vector: Some(TokenVector {
            emitted_count: i64::from(emitted),
            offered_drafts: i64::from(offered),
            accepted_drafts: i64::from(accepted),
            copy_offered_mask: copy_mask,
            verification_width: i64::from(width),
            tokens,
            model_reserved: -1,
            draft_cache_consumed: 0,
        }),
        ..Default::default()
    };
    assert!(record.validate_vector().is_ok());
    kani::cover!(
        len == 8 && width == 8 && emitted == 8,
        "eight-token completion"
    );
    kani::cover!(
        len == 16 && width == 16 && emitted == 16,
        "sixteen-token completion"
    );
}

#[kani::proof]
#[kani::unwind(66)]
fn completion_copy_mask_boundary_matches_signed_mask_domain() {
    let offered = if kani::any::<bool>() { 63 } else { 64 };
    let copy_mask: i64 = kani::any();
    let token = i64::from(kani::any::<u32>());
    let accepted: u8 = kani::any();
    kani::assume(accepted <= 1);
    let mut tokens = vec![-1; 65];
    tokens[0] = token;
    let record = CompletionRecord {
        status: 1,
        token_id: token,
        vector: Some(TokenVector {
            emitted_count: 1,
            offered_drafts: offered,
            accepted_drafts: i64::from(accepted),
            copy_offered_mask: copy_mask,
            model_reserved: -1,
            draft_cache_consumed: 0,
            verification_width: offered + 1,
            tokens,
        }),
        ..Default::default()
    };
    assert_eq!(record.validate_vector().is_ok(), copy_mask >= 0);
    kani::cover!(
        offered == 63 && copy_mask == i64::MAX,
        "63 offered drafts with all provenance bits set"
    );
    kani::cover!(
        offered == 64 && copy_mask == i64::MAX,
        "64 offered drafts with all provenance bits set"
    );
}
