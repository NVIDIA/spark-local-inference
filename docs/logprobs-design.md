# Generation logprob contract

Generation logprobs are a captured-graph operation contract. Ignition
prepares the reduction in shared sampling and commit components; Helios manages
its declared buffers, transfers, and protocol presentation. Recipes allocate a
completion bundle and supply target logits to the shared commit. Native
scheduling, cache policy, and model computation retain their existing boundaries.
Wire formats are documented in [api.md](api.md#generation-log-probabilities).

## Numerical meaning

For every returned sampled token `t`, report `logits[t] - logsumexp(logits)` over
the full vocabulary received by the sampler, accumulating in FP32. This precedes every
sampling transformation: penalties, grammar, temperature, top-k, top-p, and min-p.
Speculative positions always use target logits, including correction tokens,
width-one rounds, complete speculative rounds with packed targets, and history proposals.
Draft probabilities never enter the reporting path.

Forced tokens, such as the `thinking_token_budget` close, have no logprob entry.

Exports provide K=20 alternatives. Requests select zero through K. Alternatives
are ordered by decreasing logprob and increasing token ID for equal values. The
chosen token is stored separately, so it need not belong to the alternatives.
The reduction does not change sampling or Philox state.

## Persistent I/O

`CompletionOutputs` owns records, one packed output, and request flags. `capacity`
is the padded physical request capacity. Eight positions cover every supported
speculative completion vector. All tensors have contiguous row-major storage.

| Buffer | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `completion_records` | int64 | `[2, capacity, 24]` | ABI5 completion vectors |
| `completion_logprobs` | int32 | `[2, capacity, 8, 2+2*K]` | Packed token statistics |
| `logprob_requested` | int32 | `[capacity]` | Nonzero for requesting lanes in this submission |

Each packed position has chosen token ID, chosen FP32 logprob bits, then K pairs
of alternative ID and FP32 logprob bits. Bit storage preserves FP32 values without
requiring separate transfer allocations. If the vocabulary has fewer than K
entries, unused alternative IDs are -1. The flags are ordinary graph inputs;
the output buffer cannot be uploaded. Flags and output must both be declared.

The native worker uploads the full flag vector on each submission, clearing
unused slots. Only lanes in that submission can set flags. Graphs without
logprob outputs receive neither flags nor logprob transfers. The prepared
semantic contract contains the requested output layout and its value bindings.

## Writers and committed positions

The reduction reads the active completion frame after its token writer finishes.
For a sampling row, the row's slot and epoch must match the record and the record
must emit a token. ABI5 records identify the emitted prefix; ordinary sampling
emits position zero. The raw logit row corresponds to that slot's sampled token
position.

The speculative writer runs after verification's acceptance broadcast and after
commit writes the physical-slot records. It therefore consumes the same emitted
count and physical ownership decision as the commit itself. Duplicate, inactive,
stale, rejected, exhausted, and unoffered positions do no reduction. A packed
request's columns beyond its committed prefix are never read, since they may
refer to another request's packed rows. Token limits, budget, and EOS clipping
remain the shared commit's responsibility.

Raw logits remain separate from filtering scratch. Target verification may
reuse filtering scratch; reporting never reads it. Logprob writes complete before subsequent
draft context updates or model work can reuse the target logits.

Admission, maintenance, vision, and invalid rounds write records only.
They do not create an emitted token requiring a payload. Readers use record
emission, never stale payload contents, to decide which positions are meaningful.

Mixed prompt/decode graphs retain disjoint private role outputs. The shared merge
copies only the active frame and chooses each slot's role for both records and
payload. The inactive frame may still belong to a native transfer and is never
rewritten by the merge.

## Preparation and scratch

Four prepared Triton launches reduce requested positions. Vocabulary tiles first
compute FP32 maxima and exponential sums, then a small reduction writes each
position's normalization. Tiles retain their best 32 integer sort keys; a final
merge stores K pairs and the chosen token. Keys rank the reported FP32 logprobs
and inverse token ID, including ties introduced by subtraction. Equal values at
the cutoff therefore select the smallest IDs. Signed zero is normalized for tie
ordering. Padded vocabulary lanes cannot enter the result.

FlashInfer's deterministic graph-safe selection requires more shared memory than
SM120 provides for this contract. The shared tiled implementation needs no full
`[rows, vocabulary]` FP32 copy, no runtime compilation, and no serving allocation.
Triton kernels are prepared before complete-forward compilation and capture.
Opaque launch inputs and outputs expose the record, scratch, and logit ordering
to the compiler.

A bundle's `GraphBufferPool` shares reduction scratch across every serial graph
variant and writer. Mixed role outputs use a second shared pool; roles within one
graph remain disjoint until merging. Persistent two-frame outputs never alias
scratch. Owners retain tensors and kernel handles through export and drained
native replay. Export zero ranges include these pools and outputs. Memory reports
include their physical bytes separately from allocated/reserved device totals.

A zero request flag bypasses vocabulary reads, sorting, and payload writes.
Requests without logprobs incur only the gated launches. Performance claims must
still be established with complete replay and service measurements.

## Transfer lifetime and TP

The executor records the producing graph's event and uses its existing copy
stream and frame transfer event. A requesting frame copies only its half of the
packed output into preallocated pinned storage, or reads the corresponding mapped
host allocation. Frames without a requesting lane enqueue no payload copy.
A pending frame retains a physical-slot request mask, so unrelated lanes' stale
payload is never decoded.

Decoding finishes before the frame is released. Chosen IDs must agree with emitted
token IDs; missing data fails the requesting request rather than silently dropping
an entry. The scheduler's generated-token order selects each frame position.
Errors and host stops suppress later tokens of that request in the same frame.
Only the coordinating TP rank copies values. Peers retain identical I/O layouts,
upload flags, and participate in the existing acceptance protocol; peer agreement
does not compare or transport logprob payloads.

## Text attribution and bounded retention

Token byte lookup follows the tokenizer decoder while preserving byte-level and
byte-fallback pieces before UTF-8 replacement and stream-boundary cleanup. Text
presentation uses the existing incremental decoder. Buffered spans track provisional
starts and refine them with matching byte ranges, including literal replacement
characters and split UTF-8, without retaining copies of decoder prefixes. Completion
offsets count characters at those starts, not when a whole span is released.

String-stop filtering retains entries only for contributing tokens before the
stop. Partially returned tokens keep full bytes. Silent ignored EOS entries keep
their generation order; terminating EOS and token-ID stops have no entry.
Token-stop queues, decoder queues, and native output channels stay bounded.
Streaming Chat and Completions drain entries without retaining a response history.
Responses retains output-part entries needed for its completed response, bounded
by the request's prepared output/context limit, while parser attribution queues
release entries as soon as their source bytes are consumed.

Each output parser reports source ranges for content events. Responses selects a
token only when its first text byte belongs to such a range. Reasoning, tool data,
stripped whitespace, and delimiters cannot receive output-text entries. Synthetic
content without a source span receives no token entries.

## Compatibility

No Protobuf field or completion-record ABI changes. Absent buffers mean K=0:
old packages load and serve unchanged, advertise `allow_logprobs: false`, and
reject logprob requests with HTTP 400 before admission. Present declarations
require both buffers, their exact dtypes and frame/slot geometry, and 1<=K<=20.
Sampling graphs must bind the flag vector. Malformed declarations fail loading.
All newly prepared recipes allocate the complete bundle; enabling the feature on
an old package requires a new export.

## Prompt scoring

`scored_prefill` consumes the ordinary causal prefill inputs plus
`prompt_target_tokens: int32[C]`. Target row `j` is the prompt token at position
`chunk_start+j+1`, or -1 at the prompt end and inside image spans. The operation
advances exactly the same cache and continuation storage as prefill, preserves
draft context, and may sample the final chunk's next token through the existing
completion contract. A zero output budget performs prefill only.

`--prompt-logprobs` defaults off, allocating no scoring storage or graphs.
When enabled, scored capacities grow by four from 128 through the configured chunk
limit. Ordinary graph catalogs and inputs retain their shapes and order.

One optional packed output, `prompt_logprobs: int32[2,C,43]`, uses the ordinary
submission frame index. Each row contains actual token ID, raw FP32 logprob bits,
actual rank, then twenty (token ID, FP32 logprob bits) alternatives. Invalid target
rows write ID -1, NaN, rank zero and alternative IDs -1. Only the producing frame
and graph capacity are written. Native readback uses the producing event and
preallocated pinned frame storage, and retires it only after decoding. Rank zero
copies each scored chunk; no full prompt is retained on the GPU. The server emits
prompt chunks in order before the same frame's generated tokens. Position zero
and image positions become null entries.

The shared component projects at most 64 hidden rows at a time with the same
checkpoint head and b12x preparation mechanisms as generation. The only logit
matrix is `[64, local_vocabulary]`; each row block reuses pooled reduction storage.
Each vocabulary tile computes FP32 maximum and exponential mass. A compact
bit-preserving exchange combines per-shard statistics and finds the actual token's
value. Tiles count strictly greater reported values and select the best twenty
by the generation ordering, including subtraction-induced ties. A second compact
exchange merges shard candidates and adds one to the summed counts. Each exchange
carries 64 int32 words (256 bytes) per row per rank; it never transports vocabulary
matrices. Reductions and collectives are prepared before capture.

Every recipe uses this component with its own final normalization. DeepSeek V4.1
ordinary prefill uses the [technical report's](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)
bounded SWA replay (§3.2.2) for chunks over 128 rows. Scoring runs all layers on all
rows, with full-row candidate storage. A long prompt's last-row score can differ
from the first generated token's logprob: measured mean +0.29 nats, up to about 1.1 nats.
Uncached prefill also varies between runs: b12x routed-expert reduction order
changes layer-0 results by one BF16 ULP, amplified by subsequent expert routing.
Accuracy comparisons must use fresh-run spread, rather than exact repeatability.

On four RTX PRO 6000 GPUs, TP4 at concurrency 32, flag-on/off ABBA serving
measurements found ordinary prefill and unrequested decode unchanged within noise.
KV capacity fell 5.4%, from 21,714,176 to 20,539,648 tokens (Qwen TP1: 0.9%).
The packages add about 1.0 GB of fixed device allocations per rank, including
about 408 MiB of graph scratch, 185 MiB of other tensors and additional capture
storage; eight scored graphs add about 51,000 nodes. Scored-prefill service
throughput was about 0.50× ordinary at K=0 and 0.37× at K=20 (Qwen estimate:
0.77×). Both counts execute the same top-20 GPU reduction; the K-dependent cost
includes CPU formatting and response bytes. These measurements include legacy
echo formatting for non-echo requests.

Helios selects the operation from the request and prepared catalog, independently
of model identity. Scored requests bypass GPU and host prefix lookup, but use the
shared reservation, snapshot and publication machinery. They cannot enter the
prompt role of a mixed graph. Readmission preserves the original prompt end and
published score position, finishes any remaining scores, and suppresses duplicates.
Requests without scoring never select this operation.

Protobuf declares operation 12 and value source 39 without new message fields.
Absent prompt buffers and graphs mean no scoring; partial or malformed layouts
fail validation. Old packages retain their operations, default values, loading
and serving behavior.

Wire formats follow vLLM's [completion serving](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/completion/serving.py),
and [chat serving](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/chat_completion/serving.py).
