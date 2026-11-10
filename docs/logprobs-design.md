# Generation logprob contract

Generation logprobs are a captured-graph operation contract. Ignition
prepares the reduction in shared sampling and commit components; Helios manages
its declared buffers, transfers, and protocol presentation. Recipes allocate a
completion bundle and supply target logits to the shared commit. The contract
adds no native scheduling, cache-policy, or model-computation behavior.
Wire formats are documented in [api.md](api.md#generation-log-probabilities).

## Numerical meaning

For every returned sampled token `t`, report `logits[t] - logsumexp(logits)` over
the full vocabulary received by the sampler, accumulating in FP32. This precedes every
sampling transformation: penalties, grammar, temperature, top-k, top-p, and min-p.
Speculative positions always use target logits, including correction tokens,
width-one rounds, complete speculative rounds with packed targets, and history proposals.
Draft probabilities never enter the reporting path.

Forced tokens, such as the `thinking_token_budget` close, have no logprob entry.

Exports provide K=20 natural alternatives. Requests select zero through K.
Natural alternatives are ordered by decreasing logprob and increasing token ID
for equal values. A request may instead supply up to 128 distinct token IDs
(`logprob_token_ids`); every position of that request then reports those IDs in
caller order with the same full-vocabulary normalization, and no natural
alternatives. The chosen token is stored separately, so it need not belong to
the alternatives. The reduction does not change sampling or Philox state.

## Persistent I/O

`CompletionOutputs` (`ignition/components/completion.py`) owns records, one
packed output, request flags, and the explicit-ID inputs. `capacity`
is the padded physical request capacity. T is the resolved core
`completion_token_capacity`: zero or omission means eight, and positive values
are literal. All tensors have contiguous row-major storage.

| Buffer | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `completion_records` | int64 | `[2, capacity, 16+T]` | ABI6 completion vectors |
| `completion_logprobs` | int32 | `[2, capacity, T, 258]` | Packed token statistics: the chosen pair and up to 128 alternative pairs |
| `logprob_requested` | int32 | `[capacity]` | Nonzero for requesting lanes in this submission |
| `logprob_token_ids` | int32 | `[capacity, 128]` | Caller-selected token IDs per slot |
| `logprob_token_id_counts` | int32 | `[capacity]` | Number of selected IDs per slot; zero selects the K natural alternatives |

Each packed position has chosen token ID, chosen FP32 logprob bits, then pairs
of alternative ID and FP32 logprob bits: the K natural alternatives when the
slot's count is zero, otherwise that many selected IDs in their uploaded order.
Bit storage preserves FP32 values without requiring separate transfer
allocations. If the vocabulary has fewer than K entries, unused alternative IDs
are -1. The flags, IDs, and counts are ordinary graph inputs; the output buffer
cannot be uploaded. A core that declares the ID inputs must declare all five
buffers with these shapes.

The native worker uploads the full flag and count vectors on each submission,
clearing unused slots, and uploads each selecting slot's ID prefix. Only lanes
in that submission can set flags, and a nonzero count requires that slot's
flag. Graphs without logprob outputs receive neither flags nor logprob
transfers. The prepared semantic contract contains the requested output layout
and its value bindings.

## Writers and committed positions

The reduction reads the active completion frame after its token writer finishes.
For a sampling row, the row's slot and epoch must match the record and the record
must emit a token. ABI6 records identify the emitted prefix; ordinary sampling
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
payload. The inactive frame may belong to a native transfer and is never
rewritten by the merge.

## Preparation and scratch

Four prepared Triton launches reduce requested positions. Vocabulary tiles first
compute FP32 maxima and exponential sums, then a small reduction writes each
position's normalization. Tiles retain their best 32 integer sort keys; a final
merge stores K pairs and the chosen token. Keys rank the reported FP32 logprobs
and inverse token ID, including ties introduced by subtraction. Equal values at
the cutoff therefore select the smallest IDs. Signed zero is normalized for tie
ordering. Padded vocabulary lanes cannot enter the result. For a slot with
selected IDs, the final launch instead reads each selected ID's raw logit and
subtracts the same normalization.

Vocabulary-sharded speculative verification (`ignition/components/parallel_draft.py`)
keeps each rank's target logits as a vocabulary shard and uses
`PreparedVocabularyParallelLogprobs` (`ignition/components/vocabulary_logprobs.py`):
six prepared Triton launches per rank and two all-gather exchanges of 256 FP32
words per position per rank. The first exchange carries each shard's maximum
and exponential mass for the global normalization. The second carries the
chosen token's raw logit from its owning rank and either each shard's best 32
sort keys or the selected IDs' raw logits. The final launch merges the gathered
candidates into the same packed output.

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
Requests without logprobs incur only the gated launches. Performance claims
about the reduction require complete replay and service measurements.

## Transfer lifetime and TP

The executor records the producing graph's event and uses its frame copy
stream and frame transfer event. A requesting frame copies only its half of the
packed output into preallocated pinned storage, or reads the corresponding mapped
host allocation. The copy covers each position's chosen pair and its first W
alternative pairs, where W is the larger of K and the frame's largest selected
ID count. Frames without a requesting lane enqueue no payload copy.
A pending frame retains a physical-slot request mask, so unrelated lanes' stale
payload is never decoded.

Decoding finishes before the frame is released. Chosen IDs must agree with emitted
token IDs; missing data fails the requesting request rather than silently dropping
an entry. The scheduler's generated-token order selects each frame position.
Errors and host stops suppress later tokens of that request in the same frame.
Only the coordinating TP rank copies values. Peers retain identical I/O layouts,
upload flags, and participate in the acceptance protocol; peer agreement
does not compare or transport logprob payloads.

## Text attribution and bounded retention

Token byte lookup follows the tokenizer decoder while preserving byte-level and
byte-fallback pieces before UTF-8 replacement and stream-boundary cleanup. Text
presentation uses the server's incremental decoder. Buffered spans track provisional
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

The contract uses named buffers only: it has no dedicated Protobuf field and
adds no completion-record fields. A core without the buffers has K=0:
it loads and serves, advertises `allow_logprobs: false`, and
rejects logprob requests with HTTP 400 before admission. A core that declares
`completion_logprobs` and `logprob_requested` without the ID inputs has a last
output extent of `2+2*K` with 1<=K<=20; it serves natural alternatives and
rejects `logprob_token_ids`. Declarations must have the exact dtypes and
frame/slot geometry above. Sampling graphs must bind the flag vector and, when
the ID inputs are declared, both of them. Malformed declarations fail loading.
`CompletionOutputs.allocate` always allocates the complete bundle, so every
recipe exports it; a core exported without it provides these features only
after re-export.

## Prompt scoring

`scored_prefill` consumes the ordinary causal prefill inputs plus
`prompt_target_tokens: int32[C]`. Target row `j` is the prompt token at position
`chunk_start+j+1`, or -1 at the prompt end and inside image spans. The operation
advances exactly the same cache and continuation storage as prefill, preserves
draft context, and may sample the final chunk's next token through the
completion contract. A zero output budget performs prefill only.

`--prompt-logprobs` defaults off, allocating no scoring storage or graphs.
When enabled, scored capacities start at 128 and grow by factors of four below
the configured chunk limit, which is always the largest capacity. The flag does
not change the shapes or order of ordinary graph catalogs and inputs.

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
rows, with full-row candidate storage. A long DeepSeek V4.1 prompt's last-row
score therefore differs from the first generated token's logprob, by +0.29 nats
on average and by up to about 1.1 nats. Uncached prefill is also not
bit-repeatable: b12x routed-expert reduction order changes layer-0 results by
one BF16 ULP, which subsequent expert routing amplifies. Accuracy comparisons
must use the spread across fresh runs, not exact repeatability.

`--prompt-logprobs` costs KV capacity and fixed device memory, and scored prefill
is slower than ordinary prefill. For DeepSeek V4.1 Flash at TP4 and concurrency
32 on four RTX PRO 6000 GPUs, the flag adds about 1.0 GB of fixed device
allocations per rank (about 408 MiB of graph scratch, 185 MiB of other tensors,
and capture storage for eight scored graphs of about 51,000 nodes) and lowers
KV capacity by 5.4%; for Qwen at TP1 it lowers KV capacity by 0.9%. Ordinary
prefill and decode without requested logprobs cost the same with and without
the flag. Scored-prefill service throughput is about 0.50× ordinary prefill at
K=0 and 0.37× at K=20. Both run the same top-20 GPU reduction; the K-dependent
cost is CPU formatting and response bytes. The ratios include formatting the
Completions `logprobs` arrays, which the server performs only for `echo`
requests, so scoring requests without `echo` are at least this fast.

Helios selects the operation from the request and prepared catalog, independently
of model identity. Scored requests bypass GPU and host prefix lookup, but use the
shared reservation, snapshot and publication machinery. They cannot enter the
prompt role of a mixed graph. Readmission preserves the original prompt end and
published score position, finishes any remaining scores, and suppresses duplicates.
Requests without scoring never select this operation.

Protobuf declares `SCORED_PREFILL` as operation 12 and `PROMPT_TARGET_TOKEN` as
value source 39; scoring has no dedicated message fields.
Absent prompt buffers and graphs mean no scoring; partial or malformed layouts
fail validation. Prompt scoring also requires K=20 generation logprob outputs.
A core without scoring buffers and graphs loads and serves, and rejects prompt
logprob requests with HTTP 400.

Wire formats follow vLLM's [completion serving](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/completion/serving.py),
and [chat serving](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/chat_completion/serving.py).
