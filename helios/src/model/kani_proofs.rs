/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::contract::*;
use super::*;
use crate::execution::TokenVector;

struct CompletionFixture {
    config: ModelConfig,
    graph: GraphContract,
    kind: GraphKind,
    record: CompletionRecord,
    _detached_vector: Option<TokenVector>,
}

fn completion_fixture(tokens_below_vocabulary: bool) -> CompletionFixture {
    let vocabulary: u32 = kani::any();
    kani::assume(vocabulary > 0);
    let lag: u32 = kani::any();
    let has_cursor: bool = kani::any();
    let speculative: bool = kani::any();
    let graph = GraphContract {
        graph: String::new(),
        kind: GraphOperation::Speculative,
        inputs: Vec::new(),
        state: Vec::new(),
        state_rows_per_request: 1,
        draft_cursor: has_cursor.then_some(CompletionCursor {
            state: StateView {
                buffer: String::new(),
                offset: 0,
                stride: 8,
                nbytes: 8,
            },
            lag_tokens: lag as usize,
        }),
        embedding_tables: Vec::new(),
        pending_token: None,
        tuned_bindings: Vec::new(),
    };
    let len = usize::from(kani::any::<u8>());
    kani::assume(len <= 16);
    let raw_tokens: [i64; 16] = kani::any();
    if tokens_below_vocabulary {
        for token in raw_tokens {
            kani::assume(token < i64::from(vocabulary));
        }
    }
    let mut tokens = Vec::from(raw_tokens);
    tokens.truncate(len);
    let config = ModelConfig {
        completion_token_capacity: 16,
        max_top_logprobs: 0,
        logprob_token_id_capacity: 0,
        prepared_sequence_tokens: 1,
        contract: PreparedContract {
            version: 3,
            vocabulary: vocabulary as usize,
            input_vocabulary: vocabulary as usize,
            vision: VisionCapabilities::disabled(),
            cache: CacheContract {
                page_tokens: 64,
                regions: Vec::new(),
                page_tables: Vec::new(),
            },
            graphs: Vec::new(),
            frontend: FrontendContract {
                tools: None,
                reasoning: None,
                image_preparation: None,
                image_layout: None,
                rejected_tokens: Vec::new(),
                position_dimensions: 0,
                assistant_marker: String::new(),
                message_end_markers: Vec::new(),
                prefix_boundary: None,
                image_tokens: None,
            },
        },
        identity: [0; 32],
        model_id: String::new(),
        limits: SchedulerLimits {
            request_slots: 1,
            max_active: 1,
            max_sequence_tokens: 1,
            prefill_chunk_size: 1,
            pages: 1,
            snapshots: 1,
        },
        eos_token_ids: Vec::new(),
        sampling_defaults: Default::default(),
        sampling_ranges: Default::default(),
        max_new_tokens: 1,
        speculative_widths: Vec::new(),
        speculative_fixed_width: None,
        speculative_token_capacity: 1,
        decode_weight_bytes: 0,
        history_drafts: false,
        speculative_proposal_source: Default::default(),
    };
    let vector = TokenVector {
        tokens,
        model_reserved: kani::any(),
        draft_cache_consumed: kani::any(),
        emitted_count: 0,
        offered_drafts: 0,
        accepted_drafts: 0,
        copy_offered_mask: 0,
        verification_width: 1,
    };
    let mut record = CompletionRecord {
        status: kani::any(),
        token_id: kani::any(),
        consumed_tokens: kani::any(),
        vector: Some(vector),
        ..Default::default()
    };
    let detached_vector = if kani::any::<bool>() {
        record.vector.take()
    } else {
        None
    };
    let kind = if speculative {
        GraphKind::Speculative
    } else {
        GraphKind::Decode
    };
    CompletionFixture {
        config,
        graph,
        kind,
        record,
        _detached_vector: detached_vector,
    }
}

fn expected_cursor(fixture: &CompletionFixture) -> i64 {
    fixture.graph.draft_cursor.as_ref().map_or(-1, |cursor| {
        (i128::from(fixture.record.consumed_tokens) - cursor.lag_tokens as i128).max(0) as i64
    })
}

#[kani::proof]
#[kani::unwind(20)]
fn accepted_completions_meet_token_and_cursor_requirements() {
    let fixture = completion_fixture(false);
    let record = &fixture.record;
    if validate_completion(&fixture.config, &fixture.graph, fixture.kind, record).is_ok() {
        if record.status == 1 || record.status == 2 {
            assert!(record.token_id < fixture.config.contract.vocabulary as i64);
        }
        if let Some(vector) = &record.vector {
            assert_eq!(vector.model_reserved, -1);
            if matches!(fixture.kind, GraphKind::Speculative) && record.status != 3 {
                assert_eq!(vector.draft_cache_consumed, expected_cursor(&fixture));
            }
            let index = usize::from(kani::any::<u8>());
            if index < vector.tokens.len() {
                assert!(vector.tokens[index] < fixture.config.contract.vocabulary as i64);
            }
        }
    }
}

#[kani::proof]
#[kani::unwind(20)]
fn completions_meeting_token_and_cursor_requirements_are_accepted() {
    let fixture = completion_fixture(true);
    let record = &fixture.record;
    if record.status == 1 || record.status == 2 {
        kani::assume(record.token_id < fixture.config.contract.vocabulary as i64);
    }
    if let Some(vector) = &record.vector {
        kani::assume(vector.model_reserved == -1);
        if matches!(fixture.kind, GraphKind::Speculative) && record.status != 3 {
            kani::assume(vector.draft_cache_consumed == expected_cursor(&fixture));
        }
    }
    assert!(validate_completion(&fixture.config, &fixture.graph, fixture.kind, record).is_ok());
}
