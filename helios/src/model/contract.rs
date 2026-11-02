/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Prepared storage, value sources, and frontend syntax independent of model identity.

use super::VisionCapabilities;
use crate::{
    Result,
    execution::{ExecutionManifest, GraphKind, TunedBinding},
};

macro_rules! contract_enum {
    ($name:ident { $($variant:ident = $value:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(i32)]
        pub enum $name { $($variant = $value),+ }
        impl TryFrom<i32> for $name {
            type Error = crate::Error;
            fn try_from(value: i32) -> Result<Self> {
                match value {
                    $($value => Ok(Self::$variant),)+
                    _ => Err(concat!("unsupported ", stringify!($name)).into()),
                }
            }
        }
        #[cfg(any(feature = "frontend", feature = "runtime-package"))]
        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
                serializer.serialize_i32(*self as i32)
            }
        }
        #[cfg(any(feature = "frontend", feature = "runtime-package"))]
        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
                let value = <i32 as serde::Deserialize>::deserialize(deserializer)?;
                Self::try_from(value).map_err(serde::de::Error::custom)
            }
        }
    };
}

contract_enum!(HistoryKind {
    PagedKeyValue = 1, PagedLatent = 2, SlidingWindow = 3, RecurrentState = 4, ContinuationState = 5,
});
contract_enum!(ValueSource {
    Constant = 1, RequestSlot = 2, RequestGeneration = 3, LiveRows = 4,
    QueryLength = 5, SampleEnabled = 6, PromptToken = 7, PendingToken = 8,
    PromptAndPendingToken = 9, PromptPosition = 10, ImageFeatureIndex = 11,
    ImageLeftContext = 12, ImageRightContext = 13, PageTable = 14,
    LiveStateRows = 15, LiveStateRow = 16, PrefixStateRow = 17, SnapshotStateRow = 18,
    ConsumedTokens = 19, SnapshotConsumedTokens = 20, PrefillRow = 21,
    SamplingSeed = 22, SamplingParameters = 23, SamplingTopK = 24, MaxNewTokens = 25,
    EosTokenIds = 26, EosCount = 27, PromptHistogram = 28, GrammarMask = 29,
    GrammarEnabled = 30, CopyTokenIds = 31, CopyTokenCount = 32,
    CopyConsumedTokens = 33, CopyGeneration = 34, CopyGrammarMask = 35,
    CopyGrammarCount = 36, ImageTensor = 37, ImageFeatureSlot = 38, PromptTargetToken = 39, SpeculativePrefixLimit = 40,
});
contract_enum!(InputIndexing { Scalar = 1, SubmissionRow = 2, RequestRow = 3, PackedToken = 4, ActiveRow = 5 });
contract_enum!(StateAccessKind { Read = 1, Write = 2, ReadWrite = 3 });
contract_enum!(ToolParserKind { QwenCoder = 1, GlmArgumentTags = 2, Dsml = 3, Xtml = 4, MiniMaxXml = 5, GemmaCall = 6, Atem = 7 });
contract_enum!(ReasoningParserKind { ThinkTags = 1, XtmlThink = 2, MmThink = 3, ChannelThought = 4, RecipientHeaders = 5 });
contract_enum!(ImagePreparationKind { Qwen = 1, Mimo = 2, Deepseek = 3, Moonvit = 4, Step = 5, Radio = 6, DeepseekRowMajor = 7, Glm = 8, PatchMerge = 9 });
contract_enum!(ImageLayoutKind { Qwen = 1, Mimo = 2, Deepseek = 3, Moonvit = 4, Step = 5, Radio = 6, DeepseekRowMajor = 7, Glm = 8, Bracketed = 9 });
contract_enum!(GraphOperation { Init = 1, Restore = 2, Snapshot = 3, Prefill = 4, Decode = 5, Resume = 6, Vision = 7, Speculative = 8, Mixed = 9, ScoredPrefill = 12 });

impl TryFrom<GraphKind> for GraphOperation {
    type Error = crate::Error;

    fn try_from(kind: GraphKind) -> Result<Self> {
        Ok(match kind {
            GraphKind::Init => Self::Init,
            GraphKind::Restore => Self::Restore,
            GraphKind::Snapshot => Self::Snapshot,
            GraphKind::Prefill => Self::Prefill,
            GraphKind::ScoredPrefill => Self::ScoredPrefill,
            GraphKind::Decode => Self::Decode,
            GraphKind::Resume => Self::Resume,
            GraphKind::Vision => Self::Vision,
            GraphKind::Speculative => Self::Speculative,
            GraphKind::Mixed => Self::Mixed,
        })
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct PreparedContract {
    pub version: u32,
    pub vocabulary: usize,
    pub input_vocabulary: usize,
    pub vision: VisionCapabilities,
    pub cache: CacheContract,
    pub graphs: Vec<GraphContract>,
    pub frontend: FrontendContract,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct CacheContract {
    pub page_tokens: usize,
    pub regions: Vec<HistoryRegion>,
    pub page_tables: Vec<PageTableView>,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct HistoryRegion {
    pub kind: HistoryKind,
    pub storage: StateView,
    pub tokens_per_record: usize,
    pub window_tokens: usize,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct PageTableView {
    pub buffer: String,
    pub page_offset: usize,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct ValueBinding {
    pub buffer: String,
    pub source: ValueSource,
    pub indexing: InputIndexing,
    pub value_offset: i64,
    pub padding: i64,
    pub tensor: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct StateView {
    pub buffer: String,
    pub offset: usize,
    pub stride: usize,
    pub nbytes: usize,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct StateAccess {
    pub view: StateView,
    pub access: StateAccessKind,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct CompletionCursor {
    pub state: StateView,
    pub lag_tokens: usize,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct GraphContract {
    pub graph: String,
    pub kind: GraphOperation,
    pub inputs: Vec<ValueBinding>,
    pub state: Vec<StateAccess>,
    pub state_rows_per_request: usize,
    pub draft_cursor: Option<CompletionCursor>,
    pub embedding_tables: Vec<u32>,
    pub pending_token: Option<StateView>,
    pub tuned_bindings: Vec<TunedBinding>,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct ToolParser {
    pub kind: ToolParserKind,
    pub wrapping_newlines: u32,
    pub tag_separator: String,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct ReasoningParser {
    pub kind: ReasoningParserKind,
    pub open: String,
    pub close: String,
    pub wrapping_newlines: u32,
    pub terminal_markers: Vec<String>,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct PrefixBoundary {
    pub roles: Vec<String>,
    pub append_role: Option<String>,
    pub continue_final_message: bool,
    pub minimum_messages: usize,
    pub required_start: String,
    pub required_end: String,
    pub required_marker: String,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct FrontendContract {
    pub tools: Option<ToolParser>,
    pub reasoning: Option<ReasoningParser>,
    pub image_preparation: Option<ImagePreparationKind>,
    pub image_layout: Option<ImageLayoutKind>,
    pub rejected_tokens: Vec<u32>,
    pub position_dimensions: usize,
    pub assistant_marker: String,
    pub message_end_markers: Vec<String>,
    pub prefix_boundary: Option<PrefixBoundary>,
    pub image_tokens: Option<ImageTokenLayout>,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct ImageTokenLayout {
    pub feature_token: u32,
    pub span: Option<ImageSpan>,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct ImageSpan {
    pub start_token: u32,
    pub end_token: u32,
    pub max_tokens: usize,
    pub interior_tokens: Vec<u32>,
    pub unscoped_tokens: Vec<u32>,
}

impl PreparedContract {
    pub fn graph(&self, name: &str) -> Result<&GraphContract> {
        self.graphs
            .iter()
            .find(|g| g.graph == name)
            .ok_or_else(|| format!("graph {name} has no semantic contract").into())
    }

    pub fn validate(&self, manifest: &ExecutionManifest) -> Result<()> {
        if self.version != 3 {
            return Err("unsupported prepared model contract version; re-export the model package".into());
        }
        if self.vocabulary == 0
            || self.input_vocabulary < self.vocabulary
            || self.input_vocabulary > u32::MAX as usize
            || self.cache.page_tokens == 0
            || !matches!(self.frontend.position_dimensions, 0 | 3)
        {
            return Err("invalid prepared model contract vocabulary or geometry".into());
        }
        self.vision.validate()?;
        self.frontend.validate()?;
        if self.vision.enabled() != self.frontend.image_layout.is_some()
            || self.frontend.image_layout.is_some() != self.frontend.image_preparation.is_some()
        {
            return Err("image processing and prepared vision capacities must agree".into());
        }
        for region in &self.cache.regions {
            region.storage.validate(manifest)?;
            if region.tokens_per_record == 0
                || (matches!(
                    region.kind,
                    HistoryKind::PagedKeyValue | HistoryKind::PagedLatent
                ) && (!self
                    .cache
                    .page_tokens
                    .is_multiple_of(region.tokens_per_record)
                    || region.window_tokens != 0))
                || (region.kind == HistoryKind::SlidingWindow && region.window_tokens == 0)
            {
                return Err("invalid attention history storage geometry".into());
            }
        }
        if self.vision.enabled() != self.frontend.image_tokens.is_some()
            || self.frontend.image_tokens.as_ref().is_some_and(|layout| {
                layout.feature_token as usize >= self.input_vocabulary
                    || layout.span.as_ref().is_some_and(|span| {
                        span.max_tokens == 0
                            || span.start_token == span.end_token
                            || [span.start_token, span.end_token]
                                .iter()
                                .chain(&span.interior_tokens)
                                .chain(&span.unscoped_tokens)
                                .any(|id| *id as usize >= self.input_vocabulary)
                    })
            })
        {
            return Err("invalid prepared image-token layout".into());
        }
        let mut views = std::collections::HashSet::new();
        for view in &self.cache.page_tables {
            if !views.insert(&view.buffer) {
                return Err("duplicate page-table view".into());
            }
        }
        let protected: std::collections::HashSet<_> = manifest.tuning_graphs.iter()
            .flat_map(|graph| graph.outputs.iter().chain(std::iter::once(&graph.status_buffer)))
            .collect();
        for name in &protected {
            manifest.buffer(name)?;
            if self.cache.regions.iter().any(|region| &region.storage.buffer == *name)
                || self.cache.page_tables.iter().any(|view| &view.buffer == *name)
                || self.graphs.iter().any(|graph| {
                    graph.inputs.iter().any(|input| &input.buffer == *name)
                        || graph.state.iter().any(|state| &state.view.buffer == *name)
                        || graph.pending_token.as_ref().is_some_and(|view| &view.buffer == *name)
                        || graph.draft_cursor.as_ref().is_some_and(|cursor| &cursor.state.buffer == *name)
                }) {
                return Err("startup tuning storage overlaps request inputs or state".into());
            }
        }
        let mut names = std::collections::HashSet::new();
        for graph in &self.graphs {
            if !names.insert(&graph.graph) || !(1..=crate::execution::MAX_OUTPUT_TOKENS).contains(&graph.state_rows_per_request) {
                return Err(
                    "duplicate graph contract or invalid continuation-row requirement".into(),
                );
            }
        }
        for descriptor in &manifest.graphs {
            let graph = self.graph(&descriptor.name)?;
            if graph.kind != descriptor.kind.try_into()? {
                return Err("semantic graph operation differs from its execution catalog".into());
            }
            let semantic: std::collections::BTreeMap<_, _> = graph.tuned_bindings.iter()
                .map(|binding| (&binding.buffer, &binding.tuning_graph)).collect();
            let physical: std::collections::BTreeMap<_, _> = descriptor.tuned_bindings.iter()
                .map(|binding| (&binding.buffer, &binding.tuning_graph)).collect();
            if semantic.len() != graph.tuned_bindings.len()
                || physical.len() != descriptor.tuned_bindings.len() || semantic != physical {
                return Err("semantic tuned bindings differ from the execution catalog".into());
            }
            let targets: Vec<_> = graph.inputs.iter().filter(|b| b.source == ValueSource::PromptTargetToken).collect();
            if (descriptor.kind == GraphKind::ScoredPrefill && (targets.len() != 1
                || targets[0].indexing != InputIndexing::PackedToken || targets[0].value_offset != 0
                || targets[0].padding != -1 || targets[0].buffer != "prompt_target_tokens"))
                || (descriptor.kind != GraphKind::ScoredPrefill && !targets.is_empty()) {
                return Err("prompt targets belong only to scored prefill graphs".into());
            }
            let mut destinations = std::collections::HashSet::new();
            for input in &graph.inputs {
                if !destinations.insert(&input.buffer)
                    || !descriptor
                        .input_bindings
                        .iter()
                        .any(|b| b.buffer_name == input.buffer)
                {
                    return Err("semantic input must have one graph transport binding".into());
                }
                let buffer = manifest.buffer(&input.buffer)?;
                let mut stride = 1;
                if buffer
                    .shape
                    .iter()
                    .zip(&buffer.strides)
                    .rev()
                    .any(|(size, actual)| {
                        let mismatch = *actual != stride;
                        stride = stride.saturating_mul(*size);
                        mismatch
                    })
                {
                    return Err("host input bindings require contiguous prepared storage".into());
                }
            }
            let prefix: Vec<_> = graph.inputs.iter().filter(|input| input.source == ValueSource::SpeculativePrefixLimit).collect();
            let query: Vec<_> = graph.inputs.iter().filter(|input| input.source == ValueSource::QueryLength).collect();
            if descriptor.kind == GraphKind::Speculative {
                let binding = match (prefix.as_slice(), query.as_slice()) {
                    ([binding], []) | ([], [binding]) => binding,
                    _ => return Err("speculative operation requires exact query lengths or prefix ceilings exclusively".into()),
                };
                if binding.indexing != InputIndexing::SubmissionRow
                    || binding.value_offset != 0 || binding.padding != 0
                    || manifest.buffer(&binding.buffer)?.dtype != crate::execution::Dtype::Int32 {
                    return Err("speculative lengths require one int32 submission-row binding".into());
                }
                if !prefix.is_empty() && descriptor.draft_tokens > 0
                    && descriptor.confidence_width != descriptor.draft_tokens {
                    return Err("device prefix allocation requires declared proposal confidence output".into());
                }
            } else if !prefix.is_empty() {
                return Err("speculative prefix ceilings require a speculative operation".into());
            }
            for access in &graph.state {
                access.view.validate(manifest)?;
            }
            if graph.inputs.iter().any(|b| {
                matches!(
                    b.source,
                    ValueSource::CopyTokenIds
                        | ValueSource::CopyTokenCount
                        | ValueSource::CopyConsumedTokens
                        | ValueSource::CopyGeneration
                        | ValueSource::CopyGrammarMask
                        | ValueSource::CopyGrammarCount
                ) && !matches!(descriptor.kind, GraphKind::Speculative | GraphKind::Mixed)
            }) {
                return Err("copy proposals require a verification operation".into());
            }
            if let Some(cursor) = &graph.draft_cursor {
                cursor.state.validate(manifest)?;
                if !graph.state.iter().any(|access| {
                    access.access != StateAccessKind::Read && access.view.contains(&cursor.state)
                }) {
                    return Err("history cursor requires a declared graph state update".into());
                }
            }
            if let Some(pending) = &graph.pending_token {
                pending.validate(manifest)?;
                if !graph
                    .state
                    .iter()
                    .any(|s| s.view.contains(pending) && s.access != StateAccessKind::Write)
                {
                    return Err("pending token requires a declared device-state read".into());
                }
            }
            if let Some(pending) = &graph.pending_token {
                for operation in [
                    GraphOperation::Init,
                    GraphOperation::Restore,
                    GraphOperation::Prefill,
                    GraphOperation::ScoredPrefill,
                    GraphOperation::Resume,
                    GraphOperation::Decode,
                    GraphOperation::Speculative,
                    GraphOperation::Mixed,
                ] {
                    let mut producers = self
                        .graphs
                        .iter()
                        .filter(|producer| producer.kind == operation)
                        .peekable();
                    let required = matches!(operation, GraphOperation::Init | GraphOperation::Restore | GraphOperation::Prefill);
                    if (required && producers.peek().is_none()) || producers.any(|producer| {
                            !producer.state.iter().any(|s| {
                                s.access != StateAccessKind::Read && s.view.contains(pending)
                            })
                        })
                    {
                        return Err(
                            "device pending-token input lacks an admission or sampling producer"
                                .into(),
                        );
                    }
                }
            }
            if matches!(
                descriptor.kind,
                GraphKind::Decode
                    | GraphKind::Speculative
                    | GraphKind::Mixed
            ) && !graph.inputs.iter().any(|b| {
                matches!(
                    b.source,
                    ValueSource::PendingToken | ValueSource::PromptAndPendingToken
                )
            }) && graph.pending_token.is_none()
            {
                return Err("decode graph must declare its pending token source".into());
            }
        }
        Ok(())
    }
}

impl StateView {
    fn contains(&self, other: &Self) -> bool {
        self.buffer == other.buffer
            && self.stride == other.stride
            && self.offset <= other.offset
            && self
                .offset
                .checked_add(self.nbytes)
                .zip(other.offset.checked_add(other.nbytes))
                .is_some_and(|(outer, inner)| inner <= outer)
    }

    fn validate(&self, manifest: &ExecutionManifest) -> Result<()> {
        let buffer = manifest.buffer(&self.buffer)?;
        if self.nbytes == 0
            || buffer.shape.is_empty()
            || buffer
                .strides
                .first()
                .and_then(|stride| stride.checked_mul(buffer.dtype.size()))
                != Some(self.stride)
            || self
                .offset
                .checked_add(self.nbytes)
                .is_none_or(|end| end > self.stride)
            || buffer.shape[0]
                .checked_mul(self.stride)
                .is_none_or(|end| end > buffer.nbytes)
        {
            return Err("persistent state view exceeds its prepared storage".into());
        }
        Ok(())
    }
}

impl FrontendContract {
    fn validate(&self) -> Result<()> {
        if self.assistant_marker.is_empty() || self.message_end_markers.iter().any(String::is_empty)
        {
            return Err("assistant envelope requires nonempty delimiters".into());
        }
        if let Some(tool) = &self.tools {
            if tool.wrapping_newlines > 1
                || (tool.kind != ToolParserKind::QwenCoder && tool.wrapping_newlines != 0)
                || (tool.kind != ToolParserKind::Dsml && !tool.tag_separator.is_empty())
                || !matches!(tool.tag_separator.as_str(), "" | " ")
            {
                return Err("unsupported tool parser syntax parameters".into());
            }
        }
        if self.tools.as_ref().is_some_and(|t| t.kind == ToolParserKind::Atem)
            != self.reasoning.as_ref().is_some_and(|r| r.kind == ReasoningParserKind::RecipientHeaders)
        {
            return Err("ATEM recipient framing requires recipient-header reasoning syntax".into());
        }
        if let Some(reasoning) = &self.reasoning {
            let (open, close) = match reasoning.kind {
                ReasoningParserKind::ThinkTags => ("<think>", "</think>"),
                ReasoningParserKind::XtmlThink => ("<|open|>think<|sep|>", "<|close|>think<|sep|>"),
                ReasoningParserKind::MmThink => ("<mm:think>", "</mm:think>"),
                ReasoningParserKind::ChannelThought => ("<|channel>thought", "<channel|>"),
                ReasoningParserKind::RecipientHeaders => ("<|start|>assistant to=self<|message|>", "<|eom|>"),
            };
            let tool_open = self.tools.as_ref().and_then(|tool| match tool.kind {
                ToolParserKind::QwenCoder | ToolParserKind::GlmArgumentTags => Some("<tool_call>"),
                ToolParserKind::GemmaCall => Some("<|tool_call>"),
                _ => None,
            });
            if reasoning.open != open
                || reasoning.close != close
                || (reasoning.kind == ReasoningParserKind::XtmlThink
                    && (reasoning.wrapping_newlines != 0
                        || !reasoning.terminal_markers.is_empty()
                        || self
                            .tools
                            .as_ref()
                            .is_none_or(|tool| tool.kind != ToolParserKind::Xtml)))
                || (reasoning.kind == ReasoningParserKind::RecipientHeaders
                    && (reasoning.wrapping_newlines != 0 || !reasoning.terminal_markers.is_empty()))
                || reasoning.wrapping_newlines > 1
                || reasoning
                    .terminal_markers
                    .iter()
                    .any(|marker| Some(marker.as_str()) != tool_open)
            {
                return Err("unsupported think-tag reasoning parameters".into());
            }
        }
        if self.prefix_boundary.as_ref().is_some_and(|p| {
            p.roles.is_empty()
                || p.roles.iter().any(String::is_empty)
                || p.append_role.as_ref().is_some_and(String::is_empty)
        }) {
            return Err("prefix boundary requires nonempty message roles".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipient_syntax_requires_matching_typed_parsers_and_known_identifiers() {
        assert_eq!(ToolParserKind::try_from(7).unwrap(), ToolParserKind::Atem);
        assert_eq!(ReasoningParserKind::try_from(5).unwrap(), ReasoningParserKind::RecipientHeaders);
        for value in [0, 8, 999] { assert!(ToolParserKind::try_from(value).is_err()); }
        for value in [0, 6, 999] { assert!(ReasoningParserKind::try_from(value).is_err()); }
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.tools = Some(ToolParser {kind: ToolParserKind::Atem, wrapping_newlines: 0, tag_separator: String::new()});
        assert!(syntax.validate().is_err());
        syntax.reasoning = Some(ReasoningParser {kind: ReasoningParserKind::ThinkTags, open: "<think>".into(), close: "</think>".into(), wrapping_newlines: 0, terminal_markers: vec![]});
        let reasoning = syntax.reasoning.as_mut().unwrap();
        reasoning.kind = ReasoningParserKind::RecipientHeaders;
        reasoning.open = "<|start|>assistant to=self<|message|>".into();
        reasoning.close = "<|eom|>".into();
        reasoning.wrapping_newlines = 0;
        reasoning.terminal_markers.clear();
        syntax.tools.as_mut().unwrap().wrapping_newlines = 0;
        assert!(syntax.validate().is_ok());
    }

    #[test]
    fn startup_tuning_bindings_match_transport_and_exclude_request_state() {
        use crate::execution::{BufferDescriptor, Dtype, TuningGraphDescriptor};
        let config = crate::model::tests::config();
        let mut contract = config.contract.clone();
        let mut manifest = crate::model::testing::manifest(&config);
        let base = manifest.buffers.iter().map(|buffer| buffer.address + buffer.nbytes).max().unwrap() + 256;
        for (index, name) in ["tuning_winner", "tuning_status"].into_iter().enumerate() {
            manifest.buffers.push(BufferDescriptor {
                name: name.into(), address: (base + 255) / 256 * 256 + index * 256,
                nbytes: 4, dtype: Dtype::Int32, shape: vec![1], strides: vec![1],
            });
        }
        manifest.tuning_graphs.push(TuningGraphDescriptor {
            name: "startup_tune".into(), handle: 99999, outputs: vec!["tuning_winner".into()],
            status_buffer: "tuning_status".into(),
        });
        let binding = TunedBinding { buffer: "tuning_winner".into(), tuning_graph: "startup_tune".into() };
        manifest.graphs[0].tuned_bindings.push(binding.clone());
        let name = manifest.graphs[0].name.clone();
        contract.graphs.iter_mut().find(|graph| graph.graph == name).unwrap().tuned_bindings.push(binding);
        contract.validate(&manifest).unwrap();
        let mut wrong = contract.clone();
        wrong.graphs.iter_mut().find(|graph| graph.graph == name).unwrap().tuned_bindings.clear();
        assert!(wrong.validate(&manifest).unwrap_err().to_string().contains("semantic tuned bindings"));
        let mut wrong = contract.clone();
        let graph = wrong.graphs.iter_mut().find(|graph| graph.graph == name).unwrap();
        graph.tuned_bindings.push(graph.tuned_bindings[0].clone());
        assert!(wrong.validate(&manifest).unwrap_err().to_string().contains("semantic tuned bindings"));
        for protected in [contract.graphs[0].inputs[0].buffer.clone(),
            contract.graphs[0].state[0].view.buffer.clone()] {
            let mut wrong_manifest = manifest.clone();
            wrong_manifest.tuning_graphs[0].status_buffer = protected;
            assert!(contract.validate(&wrong_manifest).unwrap_err().to_string().contains("overlaps request"));
        }
    }

    #[test]
    fn startup_tuning_outputs_cannot_alias_cache_storage() {
        let config = crate::model::tests::config();
        let mut contract = config.contract.clone();
        let mut manifest = crate::model::testing::manifest(&config);
        let storage = contract.graphs[0].state[0].view.clone();
        contract.cache.regions.push(HistoryRegion {
            kind: HistoryKind::RecurrentState, storage: storage.clone(), tokens_per_record: 1, window_tokens: 0,
        });
        manifest.tuning_graphs.push(crate::execution::TuningGraphDescriptor {
            name: "tune".into(), handle: 99999, outputs: vec![storage.buffer], status_buffer: "frame_index".into(),
        });
        assert!(contract.validate(&manifest).unwrap_err().to_string().contains("overlaps request"));
    }

    #[test]
    fn pending_token_admission_writes_are_distinct_from_generation_reads() {
        let mut config = crate::model::tests::config();
        let manifest = crate::model::testing::manifest(&config);
        let pending = config.contract.graphs.iter()
            .find_map(|graph| graph.pending_token.clone()).unwrap();
        for graph in &mut config.contract.graphs {
            if matches!(graph.kind, GraphOperation::Init | GraphOperation::Restore) {
                graph.pending_token = None;
                for state in &mut graph.state {
                    if state.view.contains(&pending) {
                        state.access = StateAccessKind::Write;
                    }
                }
            }
        }
        config.contract.validate(&manifest).unwrap();
        for operation in [GraphOperation::Init, GraphOperation::Restore] {
            let mut invalid = config.contract.clone();
            invalid.graphs.iter_mut().find(|graph| graph.kind == operation)
                .unwrap().pending_token = Some(pending.clone());
            assert!(invalid.validate(&manifest).unwrap_err().to_string()
                .contains("pending token requires a declared device-state read"));

            let mut missing_producer = config.contract.clone();
            missing_producer.graphs.iter_mut().find(|graph| graph.kind == operation)
                .unwrap().state.retain(|state| !state.view.contains(&pending));
            assert!(missing_producer.validate(&manifest).unwrap_err().to_string()
                .contains("device pending-token input lacks an admission or sampling producer"));
        }
        let mut invalid = config.contract.clone();
        for state in &mut invalid.graphs.iter_mut()
            .find(|graph| graph.kind == GraphOperation::Decode).unwrap().state
        {
            if state.view.contains(&pending) {
                state.access = StateAccessKind::Write;
            }
        }
        assert!(invalid.validate(&manifest).unwrap_err().to_string()
            .contains("pending token requires a declared device-state read"));
    }

    #[test]
    fn reasoning_terminal_markers_follow_the_declared_tool_opener() {
        let think = (ReasoningParserKind::ThinkTags, "<think>", "</think>");
        let channel = (ReasoningParserKind::ChannelThought, "<|channel>thought", "<channel|>");
        let bare = (ReasoningParserKind::ChannelThought, "<|channel>", "<channel|>");
        for (tool, (kind, open, close), marker, valid) in [
            (ToolParserKind::QwenCoder, think, "<tool_call>", true),
            (ToolParserKind::GlmArgumentTags, think, "<tool_call>", true),
            (ToolParserKind::GemmaCall, channel, "<|tool_call>", true),
            (ToolParserKind::GemmaCall, think, "<|tool_call>", true),
            (ToolParserKind::GemmaCall, channel, "<tool_call>", false),
            (ToolParserKind::QwenCoder, think, "<|tool_call>", false),
            (ToolParserKind::MiniMaxXml, think, "<tool_call>", false),
            (ToolParserKind::GemmaCall, bare, "<|tool_call>", false),
        ] {
            let mut syntax = crate::model::testing::metadata().frontend;
            syntax.tools = Some(ToolParser {
                kind: tool,
                wrapping_newlines: 0,
                tag_separator: String::new(),
            });
            syntax.reasoning = Some(ReasoningParser {
                kind,
                open: open.into(),
                close: close.into(),
                wrapping_newlines: 1,
                terminal_markers: vec![marker.into()],
            });
            let result = syntax.validate();
            assert_eq!(result.is_ok(), valid, "{tool:?} {open} {marker} {result:?}");
        }
    }
}
