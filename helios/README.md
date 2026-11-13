# Helios

Helios (`flare-helios`) is Flare's Rust runtime library. It supplies a native
frontend, prefix cache, live-request scheduler, CUDA core loader, and graph
execution through direct driver calls. The `flare/` shared application library
links Helios; `flare-cli/` provides the terminal client and `flare-winui/` the
native Windows client. Prepared-core serving runs in each client's process.
These application and runtime libraries do not depend on Python or the internal
preparation tools.

The internal `ignition` command imports the `flare_heliopause` CPython extension
to prepare and export a model in its own process. Heliopause depends on Helios;
its crate owns the Python bindings and export allocator and builds no executable.
`ignition export` writes the core and closes the prepared model.
`flare run CORE` loads it in a separate serving process. See
[runtime cores](../docs/runtime-core-design.md) for the resource contract.

The `server` feature supplies HTTP/SSE and model scheduling; the `runtime-core`
feature supplies core loading. `flare/` and `flare-cli/` enable both. Native core
serving uses the typed prepared contract for single-rank and tensor-parallel
execution. Raw replay and two-frame graph execution are available through
Rust APIs and heliopause's diagnostic Python bindings. In-process HTTP
bindings require heliopause's `qualification` feature; normal installations
serve only through a core.

During export, Ctrl-C and SIGTERM terminate Ignition directly, bypassing
Python finalizers and CUDA graph destruction. While serving, `flare run`
stops admission, gives active responses five seconds to finish, drains native
execution, and exits 0. Unexpected runtime failure exits 1. The library's
terminal dashboard reads native metrics without contacting the GPU.
`flare run` starts it after loading when stderr is a usable terminal; redirected
output remains plain startup logs.

Left/right arrows or Tab switch between the throughput overview and scheduler
details. The scheduler pane totals graph completions over two seconds for each
captured graph. A row's label gives the graph kind, batch capacity (`b`), and
token capacity (`t`). In place of `t`, a speculative graph shows its target
width (`w`) and packed target rows (`rows`), and a mixed graph shows its prompt
capacity (`p`) and decode width (`w`).
Each row shows the call count, mean GPU duration, and the graph's share of
observed graph time. Wider
terminals also show the maximum GPU duration and actual requests per call, which
can be below the captured batch capacity. GPU duration is the maximum participating rank's CUDA event
interval; these shares are not device utilization. Missing timings are counted
as untimed and excluded from duration averages.

Request rows identify the native slot and generation, keeping reused slots
separate. Their ten-second history marks completion observations, not GPU start
or end timestamps. Each cell shows the latest prefill, decode, speculative, or
vision completion in its interval; a completion of another graph kind is shown
only when the cell contains none of those. Up/down arrows scroll requests.
Every row shows `Request` (`slot.generation`), `Phase` (the scheduler's request
phase), and `Gap ms`, the time since the request last committed prompt or output
tokens, or since its arrival. A request pane at least 100 columns wide adds
`Prompt` (consumed prompt tokens over prompt length), `Output` (generated
tokens), `Out ms` (time since output was last published, `—` before the first
publication), and `Bursts` (completed submissions whose output was published; a
multi-token speculative round counts once). Retired requests remain visible
while their history remains.
Both gaps are measured at the request-state sample, whose age is displayed.
The last submitted graph displays its frozen estimate; `?` means indeterminate.

Dashboard telemetry is enabled only while the dashboard runs. The execution
thread writes to a preallocated 1024-completion buffer with a nonblocking lock;
the dashboard retains at most 16384 completions and 128 request rows. Overflow or
lock contention increments the displayed dropped-update count. The display
marks history truncated by its retention limit. Request state is sampled on
demand every 500 ms, and its age is displayed. Terminal input preserves signal
handling, so Ctrl-C follows normal server shutdown.

Prefill throughput counts consumed prompt tokens, excluding padding and prefix
reuse. Generation throughput counts committed output tokens, including EOS and
tokens withheld by text stops. Drafter steps sum the actual learned-proposal depth D of completed speculative
programs; verification steps count completed
speculative graphs. Copy-only proposals contribute no drafter steps. Mean draft
acceptance length is accepted draft tokens per progressing request round with
at least one offered draft, excluding the bonus token and width-one rounds.
DEPTH is allocated draft positions per progressing generation request
round, including zero-draft rounds. Packed rounds contribute their allocated
per-request prefixes; the graph's maximum width and padded batch size do not
determine this average. Rejected drafts count toward depth.
The display also shows active/queued requests, completed/failed requests, reused
tokens, and cached/free KV pages. Token counters are available through `/metrics`.

Native APIs return `flare_helios::Error`, with separate `AdmissionError`,
`ExecutionError`, and `FrontendError` variants. Validation, capacity exhaustion,
unsupported features, grammar failures, image decoding, manifest violations,
worker failure, poison, and drain failures remain distinct through worker channels.
Errors retain their underlying source and CUDA operation/status code. Cloned
errors share their underlying cause.

HTTP request validation and unsupported features return 400. A prompt and
output limit beyond the prepared context return 400 with both token counts and
the context. Exhausted HTTP ingress capacity returns 429; exhausted model
admission capacity and unavailable workers return 503. Context and capacity
rejections append the configured `context_hint` or `capacity_hint`, which names
the setting that raises the limit. A missing or wrong bearer token returns 401
when the server configuration sets `api_key`; `/health` and `/metrics` need no
token.
Execution and presentation failures return 500. SSE terminal failures and diagnostic Python exceptions serialize the same display
messages. Failed CUDA drains retain prepared resources. Worker recovery is unsupported.

The server exposes `/health` and `/metrics`; model discovery at `/v1/models`,
`/v1/models/{model}`, `/api/tags`, and `/api/show`; and generation at
`/v1/chat/completions`, `/v1/completions`, `/v1/messages`,
`/v1/messages/count_tokens`, `/v1/responses`, `/v1/systemone`, and
`/v1/decisions`.
Embedded Rust applications can move a loaded `NativeModel` into
`server::LoadedSession`, adjust its `ServerConfig`, and start a `ServingSession`.
The session retains every local rank's native resources through confirmed GPU
drain and releases them when the closed session is dropped. Failed startup or
shutdown drain retains those resources until process exit. Multi-host rank
coordination uses the explicit `Server::start` interface.
`StartupProgress::observed` exposes a bounded latest-stage snapshot without
terminal output, and owned sessions disable stderr diagnostics by default.
`Server::snapshot` reads typed runtime and HTTP/frontend gauges with a session
identifier and monotonic sample time. These independently read counters do not
form a transactional sample. `pause_admission` rejects subsequent generation
requests with 503 while admitted responses finish; `wait_for_requests` waits up
to the caller's timeout. `resume_admission` reopens a healthy listener after a
cancelled model switch. Closing a server allows the five-second HTTP grace
period before stopping and draining native execution.

`/v1/models` lists one card, using the first nonempty `served_names` entry as its
`id`, or the checkpoint identifier when none is set. The card includes the
effective context limit in `max_model_len` and prepared capabilities, with the
checkpoint identifier in `root`. `/v1/models/{model}` returns the same card for
the checkpoint identifier or any alias, including repository IDs with slashes.
Unknown names return a JSON 404.
See [the API reference](../docs/api.md) for discovery fields and generation routes.
HTTP admission reserves at most `queue_capacity + max_active` requests before
reading JSON bodies. Admitted requests await a bounded CPU preparation worker;
each of the `frontend_workers` permits, two by default, prepares one request at
a time.
`ignition` cores set the queue to four times the concurrency limit, one worker per
concurrent request, and a body limit of the recipe's per-image byte allowance
times its image limit plus 32 bytes per context token.
An admission permit survives CPU preparation and the streamed response lifetime.
Disconnects cancel model scheduling. A paused reader keeps its response: each
event queue holds the request's complete output budget, allocated as events
arrive. Finished, unread responses release their model slot and live KV cache
but still count toward the admission limit. HTTP capacity and physical GPU slots
are independent.
Tool-call history accepts OpenAI JSON-string arguments and object arguments;
strings must parse as JSON objects before checkpoint-template rendering.

`/metrics` emits Prometheus text without request labels, prompt text or token
IDs. Counters cover model admission/completion/cancellation/failure, prefix
lookups/hits/misses/reused tokens, successful launches by graph kind, ready
checkpoint publications and logical evictions. Gauges expose live/free/retiring
request slots, reusable pages/snapshots, cached checkpoints/pages, model pending
requests, HTTP/frontend capacities, seconds since the listener started, and tokens
generated during the most recent minute, sampled every five seconds. Prefix lookup counters include admission
attempts later rejected for resource pressure. Publication counts unique
checkpoints, excluding duplicate proposals. Eviction precedes GPU retirement;
free-resource gauges count only IDs whose fences have completed. Scrapes read
independent atomics and may span concurrent worker updates. Per-request deltas
require terminal responses followed by model quiescence, with no other traffic.

```sh
cargo test --manifest-path helios/Cargo.toml --offline
cargo build --manifest-path heliopause/Cargo.toml --features extension-module
```

Some tests built with the `frontend` feature, which `images`, `grammar`, and
`server` enable, read chat templates from the directory named by
`FLARE_TEST_TEMPLATE_DIR`; they panic when it is unset. Create that directory
with `python tests/frontend/prepare_templates.py --output DIR`, which downloads
the pinned upstream templates through `huggingface_hub`. The `runtime-core`
feature requires `protoc`. The `Native serving` workflow in
`.github/workflows/native-serving.yml` raises the locked-memory limit to 64 MiB
on Linux and runs
`cargo test --locked --release --manifest-path helios/Cargo.toml --features runtime-core,server`.

The `images` and `server` features decode JPEG through the `turbojpeg` crate,
pinned to 1.5.1 with only its `cmake` feature. The build compiles the crate's
bundled libjpeg-turbo source with CMake and a C compiler and links it
statically. It performs no `pkg-config` discovery and uses no system
`libturbojpeg` at build time or run time. Without NASM on `PATH`, libjpeg-turbo
builds without SIMD extensions.

Image admission accepts base64 `image/png`, `image/jpeg`, and `image/webp` data
URLs. JPEG uses
accurate IDCT, smooth chroma upsampling, and at most 100 progressive scans.
Baseline/progressive Huffman JPEG is supported; lossless and arithmetic JPEG
are rejected. Encoded byte and decoded pixel limits apply before pixel allocation.
PNG, JPEG, and WebP EXIF orientation is applied before resize, except by the
MoonViT, DeepSeek row-major, and padded-grid profiles, which keep the stored
pixel order; orientations 1–8 include mirrored and axis-swapped images. Invalid
orientation metadata is rejected. RGB conversion discards alpha except where the
profile consumes it: MoonViT resizes it, and the padded grid composites it onto
a checkerboard. Embedded color profiles are not applied.
Pillow is used only to generate test fixtures, never by native image admission.

Heliopause builds the internal Python extension named `flare_heliopause`. It requires the NVIDIA driver at runtime
and dynamically loads `libcuda.so.1` (`nvcuda.dll` on Windows); building and running native unit tests does
not require CUDA libraries. Tests exercise queueing and teardown using explicit
Rust test functions. Production always calls the CUDA driver.

The `GraphReplayWorker` and `ReplayCompletion` Python bindings require the
`qualification` Cargo feature. Production core builds omit them. The server
uses its native execution transport directly.

Raw replay qualification entrypoints:

```text
current_context() -> (context_handle, device_ordinal)
GraphReplayWorker(owner, context, device, graph_execs, buffers, queue_capacity=8)
worker.submit(graph_index, repetitions=1) -> ticket
worker.poll(timeout_ms=0) -> ReplayCompletion | None
worker.outstanding -> accepted commands whose completion has not been consumed
worker.close()
```

`buffers` is a list of `(address, nbytes)` spans. Registration checks span
arithmetic and executable-handle uniqueness; it cannot prove that raw addresses
refer to live allocations. Preparation must supply valid handles and an `owner`
that strongly retains every graph, captured tensor, module, mapped allocation,
and CUDA context. The caller must finish warmup and synchronize preparation
before registration. Re-instantiation, tensor resizing, and concurrent external
mutation of captured storage are forbidden until `close()` completes.

One submission repeats a selected executable entirely in Rust, synchronizes the
worker stream, and emits a completion ticket. All submissions use the same
nonblocking stream, so aliased graph scratch is serialized. The completion queue
and pending command queue share a capacity bound: an unconsumed completion still
occupies admission capacity. `poll()` releases the GIL while waiting. This is a
replay primitive, not the model-serving loop's two-frame transfer pipeline.

Serialization alone does not preserve tensors allocated inside a shared graph
memory pool. Preparation must verify that the captured allocation lifetimes
permit the submitted variant order. Outputs or state that survive another
variant's replay require independent persistent storage; pooled intermediates
cannot escape their graph unless the preparation contract protects their
lifetimes.

Raw replay bypasses PyTorch RNG bookkeeping. Prepared graphs must use explicit
device RNG state and cannot contain generator-backed PyTorch random operations.

`close()` rejects future submissions, executes queued submissions, synchronizes
the stream, joins the thread, and then releases the Python owner. It is
idempotent. Launch failure prevents later launches and is reported on subsequent
completion tickets and on close. If stream drain cannot be confirmed, the owner
is retained and ultimately leaked rather than freeing potentially referenced
device storage. An unclosed object's destructor attempts the same drain; callers
must close explicitly before interpreter finalization. No device context is
created or destroyed by this crate.

Qualify real PyTorch-to-Rust graph replay on an explicitly assigned idle GPU:

```sh
PYO3_PYTHON=/path/to/python cargo build --manifest-path heliopause/Cargo.toml --features extension-module,qualification
CUDA_VISIBLE_DEVICES=GPU-UUID /path/to/python helios/tests/test_cuda_graph_replay.py \
  --extension heliopause/target/debug/libflare_heliopause.so
```

The integration cases capture real PyTorch graphs, change device inputs between
completed submissions, replay different graph capacities, check native repetition
counts, and verify graph-owner lifetimes during close and destructor draining.
They qualify interoperability and resource handling, not model correctness or
serving throughput. The selected Python environment is read without installation
or modification.

## Two-frame execution

[manifest.md](manifest.md) specifies the prepared execution ABI.
`execution::ExecutionManifest::validate` checks graph capacities, required
completion geometry, byte extents, contiguous strides, alignment, unique names
and graph handles, and nonoverlapping host/device I/O spans before startup.
The owner supplies live pointers on the declared context; metadata validation
cannot establish allocation lifetime or graph numerical behavior.

`execution::GraphExecutor` owns two pinned input frames, two pinned completion
frames, a model stream and a copy stream. `submit` copies selected input
subranges into a free host frame, enqueues asynchronous uploads and graph
execution, records a production event, and queues the completion transfer on
the copy stream behind that event. Omitted input ranges preserve device state.
`poll` returns ordered `FrameCompletion` records after transfer completion and
then permits frame reuse. Only ordinary decode and complete speculative
generation can share the two-frame window; other operations drain it. A third
submission receives backpressure; a grammar submission requires an empty window and blocks further submissions until its
completion is consumed. The direct executor has no Python calls or per-step
CUDA allocation.

Tensor-parallel coordination counts preparing steps, enqueued steps, and
unconsumed all-rank completions against the same two-slot bound. Prepare and
ready messages carry the submission fence and graph identity; commit follows
validation by every rank. Rank workers process control messages while CUDA work
is outstanding. Completion publication and resource retirement remain ordered
across all ranks.

Speculative descriptors carry actual request/proposal/target geometry B/D/W/P,
including explicit P for rectangular programs. Exact `QueryLength` inputs select
host allocation; `SpeculativePrefixLimit` inputs select captured GPU allocation.
A produced confidence sidecar declares `confidence_width` and uses two FP32
frames shaped `[2, request_slots, draft_capacity]`. Host dependencies are derived
per selected program. Planning projects queued expected service and output age
at submission; observed delivery age changes only when tokens are published.

An executor belongs to one native OS thread. A Rust scheduler can call it
directly. `execution::ExecutionWorker` adds a synchronous native-thread control
channel for bootstrap and interoperability tests. The optional Python binding
uses that worker and releases the GIL around native operations:

```text
GraphExecutor(owner, manifest, *, peer_fds=None)
executor.submit(graph_index, [(buffer_name, byte_offset, bytes), ...],
                grammar_constrained=False) -> fence | None
executor.poll() -> ExecutionCompletion | None
executor.outstanding -> number of occupied frames
executor.drained -> whether close confirmed both streams drained
executor.close()
```

`peer_fds` supplies connected rank sockets for tensor-parallel execution; it is
supported only on Linux.
`ExecutionCompletion.records` contains one ABI6 record per captured physical
request slot, with 16 metadata fields followed by the core-defined number of
token fields. An omitted or zero `completion_token_capacity` resolves to eight.
`consumed_tokens` and `generated_tokens` are cumulative.
Submission failure poisons the executor. `close` drains both streams before
releasing pinned memory or the Python owner; failed drain retains both. Owner
release happens outside the ownership mutex, permitting a Python finalizer to
reenter `close`. Production request processing belongs in Rust; Python control
methods provide a qualification surface.

```bash
PYO3_PYTHON=/path/to/python cargo build --manifest-path heliopause/Cargo.toml \
  --features extension-module --offline
timeout 60s env CUDA_VISIBLE_DEVICES=GPU-UUID /path/to/python \
  helios/tests/test_execution.py --extension heliopause/target/debug/libflare_heliopause.so
```

The five CUDA integration cases cover physical-slot scatter, different graph
capacities, partial uploads, two occupied frames, grammar serialization, changed
resident inputs, rejected uploads, asynchronous owner retention, reentrant
finalizers and destructor draining. They use real PyTorch-captured executables.

## Live requests

`scheduler::RequestScheduler` owns up to 64 live request slots within the selected
concurrency and allocates private page tails and exact checkpoint snapshots from
prepared capacities.
Request handles contain a physical slot and a positive int64-safe generation.
Each physical request slot owns a fixed range of execution-state rows, leaving
state row zero for padding. Cached pages remain immutable and shared through pinned ancestors.
Resource pressure evicts unpinned cache leaves; retirement fences determine
when their page and snapshot IDs become reusable.

`RequestScheduler::plan` returns an immutable `ScheduledPlan` for initialize,
restore, snapshot, vision, prefill, scored-prefill, resume-sampling, decode,
speculative, or mixed prefill/generation work.
The plan retains its scheduler identity and each participating request's state
epoch. Admissions of unrelated requests and timing observations preserve its
validity; mutation or retirement of a selected request invalidates it.
Selection ranks every candidate graph and cohort by expected useful value per
estimated graph duration, less a duration-proportional waiting cost for requests
that are decoding or awaiting terminal retirement. Each loaded
prefill capacity up to `prefill_chunk_size` competes on the prompt tokens it can
actually consume, which the next selected message boundary, atomic image spans,
and the prompt end limit. No rule pads a logical length to the smallest covering
capacity. Among candidates of one graph kind and cohort that perform the same
logical work, selection first discards any shape whose capacities are all at
least as large as another candidate's and larger in at least one. An unmeasured
prefill graph is not explored when a smaller loaded capacity consumes the same
tokens. Checkpoint selection prioritizes the last eligible message boundary,
using the prompt end when no message boundary is eligible. Earlier messages and
a distinct prompt end compete within a work allowance of
`scheduler_checkpoint_extra_work_fraction` times estimated uncached prefill work,
defaulting to 0.05. Extra captures charge snapshot-copy cost and additional prefill
caused by splitting chunks. Productive completed decode earns the same fraction
of measured time for a distinct terminal capture. Unknown costs exclude extras
while preserving best-effort anchor retention. There is no capture-count cap or
per-request share of snapshot rows. The global allocator accounts for snapshot
aliases, guards a later anchor against earlier optional captures, and reclaims
unused guards for mandatory execution. Only obtainable capture destinations
introduce prefill cuts. See [checkpoint ownership](../docs/core-design.md#exact-endpoint-capture). `prefill_chunk_size` must be 64-aligned, at least 64, and within the
context capacity. Batched graphs use power-of-two capacities up to the selected
concurrency rounded up to a power of two.
Prefill, decode cohorts, speculative widths, mixed graphs, and prerequisite
stages compete in that one selection. Full prompt cache hits restore state and
resume sampling from saved hidden state without replaying prompt tokens. See
[useful service and selection](../docs/scheduling-runtime-design.md#useful-service-and-selection).

Before CUDA enqueue, callers check `GraphExecutor::can_submit`, obtain its
`next_fence`, and call `RequestScheduler::commit` with the plan to reserve pages
and record use of cache ancestors. Commit validates the plan's epochs and
resources without rerunning scheduling policy. Completion validation checks
generations, physical slots, monotonic counters and per-step logical bounds before
mutating request state.
Two ordinary decode or complete speculative steps can reference a request
concurrently; a repeated terminal
completion does not emit its sampled token twice. A CUDA submission failure
requires confirmed stream drain before `release_after_shutdown` releases
resources associated with the attempted submission.

Each request owns conditional draft-acceptance statistics by depth and proposal
source. Validated speculative completions update only that request. The emitted
accepted prefix supplies positive observations. When a correction token was
emitted and an offered draft remained, the next draft supplies a rejection.
Termination within an accepted prefix censors the remaining depths. Expected
yield is clipped by the remaining output limit, context capacity, and available
proposals. A fixed captured width may therefore contain fewer active tokens.
Acceptance learning and accepted-draft metrics exclude proposals beyond the
lane's logical token limit even when the completion reports physical padding.

`RequestScheduler::estimated_duration_ns` returns the estimate for a step's
captured graph shape. Every estimate starts at zero
(unmeasured). A valid completion sets its cost directly; further fresh observations
update it with the median of up to nine replay observations. After 30 seconds without a valid observation,
the cost becomes stale but remains available for selection. The next valid
observation replaces the stale estimate. Tensor-parallel observations use the
maximum positive rank interval, with measured embedding-readiness waits subtracted
when available. Host delays and queue wait do not enter graph estimates. An
unmeasured candidate ranks ahead of every measured candidate; unmeasured
candidates compete among themselves by expected useful value. Both
unmeasured and stale physical dispatches require an empty queue and completion
before another submission.

Ordinary valid execution earns refresh credit at `0.002 / 0.998` times its
measured duration, capped at 5 ms. Full credit permits a stale alternative to
displace the ordinary winner, ordered by the later of its last valid observation
and last attempted dispatch.
Refresh pays for one complete physical program, including its proposal and
verification work. Another refresh waits for full credit. Idle discards positive credit and preserves debt.
This bounds aggregate refresh expense, not individual stalls; see
[funded timing refresh](../docs/scheduling-runtime-design.md#funded-timing-refresh).

Speculative width selection combines this cost with request-local expected
prefix survival. Complete programs learn from terminal verification and keep
completed confidence forecasts separate from acceptance outcomes. Ordinary selection uses
historical means, while a separate 1% raw-time budget funds optimistic
acceptance probes. Observational weight ages without changing unobserved means;
width-one rounds still measure cost. See
[funded acceptance learning](../docs/scheduling-runtime-design.md#funded-acceptance-learning).

Fairness enters selection through per-request service prices. Each resident
runnable request carries a token-service price, and a request that can deliver
output also carries a delivery-service price. A price grows with elapsed
execution time while the request is unserved and decays as it receives service,
normalized by its individually feasible service rate. `commit` records the
participating requests with rate normalizers from measured graph costs;
unmeasured graphs contribute none. `complete` updates each
participant's prices from the frame's measured interval and the tokens that
request committed, so a participant the frame did not serve observes zero
service. A delivery is credited when `published_output` reports visible output
for that completion, once per completion regardless of token count. Selection
projects the expected service of queued submissions into copies of the prices
without changing observed history. A missing token price initializes from the
request's feasible token rate divided by the participant count; a missing
delivery price initializes from its reference delivery interval. See
[service accounting and publication](../docs/scheduling-runtime-design.md#service-accounting-and-publication).

`PolicyConfig` holds value coefficients, not time limits. Its defaults are an
output-token value of 64 prompt-token units, token fairness 16, delivery
fairness 4, decode-age price 0.1, terminal value 256, terminal-age price 1, a
100 ms service-price adaptation time, a 30-second timing freshness interval, and
adaptive-verification exponent 1. No millisecond cap applies to a frame or to
the queued frames together. When the oldest queued graph runs past its expected
finish, further queueing stops until its completion arrives. The coefficients
are preferences, not latency bounds. Mixed prefill/generation graphs are
scheduled when the core supplies them; Ignition captures them only with
`--mixed-prefill`. See
[the native scheduling design](../docs/scheduling-runtime-design.md#policy-preferences).

Admission reserves checkpoints for the selected message boundaries, the prompt
end, and the terminal endpoint beyond the reused prefix.
Prefill stops at each selected message boundary and at the prompt end. Snapshot graphs
copy live state and partial cache tails to immutable snapshot rows after the
selected requests' model submissions complete. Message and prompt publication retain the live physical
page sequence while transferring complete pages to the cache. Admission reuse
counts stay fixed. Duplicate publication never remaps the live request.

`finish` publishes exact terminal consumed state after its successful snapshot
copy, excluding the final pending sampled token. Terminal retirement competes by
its own intrinsic value and waiting-age cost under both scheduler policies.
A winning feedback operation drains queued work before submission; terminal
snapshots have no unconditional priority. With P the contract's `page_tokens`,
a child at T after parent S owns `floor(T/P)-floor(S/P)` complete pages. Partial payloads restore from the snapshot
into private pages. Failed copies prevent publication. `cancel` stops scheduling
immediately and retains slot/page/snapshot ownership through final GPU use.
Generation reuse waits for that retirement.

This scheduler operates on logical IDs and completion records. `model::PreparedContract`
declares typed graph value sources, storage views, dependencies, and frontend syntax.
Shared `model::Inputs` binds and packs admission, state, vision, and forward inputs.
`server::ModelRuntime`
owns the scheduler and executor on one native model thread. Its private `Worker`
owns ingress, pending and active requests, model admissions, and reusable input
storage. Admission precedes ordered completion processing, terminal publication,
copy-proposal planning, and submission. Completion-triggered retirement runs
before polling another completion; exact message and prompt publication remains
atomic inside `RequestScheduler::complete`.

The model thread retains the executor outside the `catch_unwind` closure that
runs `Worker`. Worker failure notifies active, pending, and queued consumers
before draining the executor;
scheduler resources release only after confirmed drain. Input storage remains
owned through that drain. Model packing initializes prompt penalty counts once
and uploads only newly assigned page-table entries during decode.
[model-contract.md](model-contract.md) specifies the concrete handoff.

## Native frontend

The `frontend` Cargo feature provides `Frontend::load` over explicit tokenizer and
chat-template assets with expected SHA256 fingerprints, plus typed token settings.
Each asset is hashed and parsed from the same bytes. No model lookup, downloading,
Python import, or Python callback occurs in this API.

`Frontend::render_chat` preserves JSON message content and tool argument order,
applies the prepared recipe's template, and returns prompt bytes, token IDs,
and complete-message token endpoints. Template defaults fill missing options;
explicit request options win. `frontend::jinja` compiles Jinja statements and
expressions to compact bytecode at asset load. Constants, names, call arguments,
and jump targets belong to the immutable compiled template; operand stacks,
lexical scopes, namespaces, and output buffers belong to each render. Rendering
borrows JSON input values, emits concatenations directly into the output, and
appends namespace lists without copying their accumulated contents. Python
methods and the JSON filter preserve Hugging Face spacing, Unicode, insertion
order, and float formatting. MiniJinja 2.24.0 is a development-only differential
and benchmark reference.

Recipe templates annotate complete preambles and messages in their actual
rendered order with `{% mark %}`, which records the UTF-8 output byte position
without emitting text. Segmented templates use `{% mark expression %}` with a
nonnegative segment count, translated through the segment text lengths after
rendering. Marks are forbidden inside macros, captured assignments, and filter
blocks; a root call site marks output after a macro returns. Generation headers
remain outside completed messages. Annotations follow Jinja whitespace controls.
Hugging Face `{% generation %}` blocks emit their bodies without adding prompt bytes; training token masks do
not participate in native rendering. Boolean `sameas` tests and `strftime_now`
follow the checkpoint template's identity tests and local-calendar formatting.
Literal replacement filters carry advisory block endpoints through mapped
substitutions and omit endpoints inside replaced spans.

The exact deployed Qwen3.8 and inference-transformed Step templates are upgraded
in memory using bundled insertion offsets, guarded by source and result SHA-256
hashes. Their upstream text is supplied by the core and is not embedded in
Helios. The segmented Kimi K3 template uses its custom project renderer. Kimi K3
uses segment-count marks at the preamble and complete-message endpoints.
Other unannotated templates fail to load. Checkpoint-derived annotations must
reproduce the source bytes when removed. Gemma 4 selects one of two annotated
Google canonical templates by model size across all supported quantization
publishers; see
[recipe frontend metadata](../docs/model-development.md#sampling-and-frontend-metadata).
In-memory template upgrades modify neither core files nor metadata formats.

DeepSeek V4 and V4.1 use the canonical native encoder, patched to record byte
positions in its normalized-message append loop. Adjacent users share a merged
message endpoint, and reordered tool results receive endpoints in output order.
Continuation truncates the rendered text before tokenization and clips later
positions. Recording positions does not change the encoder's division into text,
ordinary-token, and image-placeholder segments. The single encoding supplies original byte offsets; a cut is
retained only if every preceding token ends at or before it and every following
token starts at or after it. Overlapping Unicode spans and BPE tokens crossing a mark cannot
create partial-token cuts. Token endpoints are sorted, unique, and strictly
inside the prompt. Image expansion adjusts them with one ordered pass.

The TMLv0 encoder renders Thinking Machines' message frames: a role header, an
optional author or tool name, one content-kind token, an ordinary-text body, and
`<|end_message|>`. Control tokens come from token IDs and every body is encoded
without the added vocabulary, so control-token spellings in conversation text
stay ordinary text. The leading system messages come first, so a shared system
prompt remains a reusable prefix when the tools differ, then the tool
declaration, a system frame stating the thinking effort, and the conversation.
A tool declared without parameters, or with null parameters, declares `{}`;
chat templates and the DeepSeek encoders receive an empty object schema. A
failed tool result, marked by `is_error` or, in Chat Completions, by
`{"type":"tool_error","text":...}` content parts, renders its text in
`<|content_tool_error|>` frames; other prompt formats receive its content after
a "Tool execution failed:" text part. Each
assistant message ends with `<|content_model_end_sampling|>`; there is no
generation header. Tool results take the name of the most recent preceding call
with their ID. Endpoints follow the declaration, the effort frame, and every
completed message; a content-block hint ends at its frame. Continuation leaves
the final assistant text or reasoning frame open. The encoder matches
tml-renderers 0.4.1 token for token on the cases in
`tests/frontend/inkling2_tml.json`.

Advisory content-block hints travel separately from prompt text. Text hints keep
their positions through captured strings and JSON serialization; segmented
templates map JSON string positions back to decoded segment bytes before token
mapping. Image hints retain indices into the rendered image-source list and
resolve to the ends of prepared image spans after expansion. Hints do not change
prompt bytes, token IDs, image conditioning, or ordinary/special-token selection.

`Frontend::encode`, `decode`, and `incremental_decoder` use Hugging Face's Rust
tokenizers library pinned to the fixture environment's version. Incremental
decoding preserves incomplete UTF-8 between chunks and flushes remaining bytes
at completion. It does not parse reasoning or tool-call events.

Run exact-checkpoint CPU parity qualification:

```sh
cargo run --manifest-path helios/Cargo.toml --features frontend,test-tools --bin test_frontend -- \
  /absolute/path/to/model/snapshot tests/frontend/qwen38.json
```

This compares rendered bytes, token IDs, template exceptions, full decoding,
incremental decoding, and configured-default precedence with golden outputs
generated by the pinned Transformers environment. Image contents render the
template's image placeholders; image decoding, patch expansion, and preprocessing
are separate model-processor work. The frontend does not expose HTTP sockets,
run a model, support arbitrary Python tool callables, or claim compatibility
with every Transformers template extension. JSON inputs use UTF-8 strings,
64-bit integer ranges, and finite double-precision numbers.

The Jinja environment and continuation semantics follow Transformers
`utils/chat_template_utils.py` at revision
`4cc2aa84301c9aa210b5513dab9fefae03981f6e` (Apache-2.0).

`cargo bench --manifest-path helios/Cargo.toml --features frontend --bench template --
NAME=/absolute/path/to/chat_template.jinja` compares annotated bytecode rendering
against the equivalent unannotated MiniJinja template. It measures compilation,
warm median and p95 latency, allocations, peak live bytes, compiled storage, and
four-thread throughput on short prompts, long histories, large messages, tools,
and images. Tokenization is excluded. `--check` runs output comparisons without
timing; `--owned-only` measures the owned renderer while retaining output checks
and concurrent comparisons against MiniJinja.

## Embedding row cache

Disk-backed embedding tables use a bounded host-RAM row cache, defaulting to
4 GiB per host on discrete GPUs or 1 GiB when any selected GPU shares host memory.
`flare run --embedding-table-cache-size SIZE` overrides the budget; `0` disables
caching. RAM table storage allocates no separate row cache. The loader accounts
for its host storage before sizing KV pages and continuation snapshots. Entries hold immutable
table rows shared across requests; they copy into the pinned row staging, and
the cache does not alter that staging's consumer fences. Approximate usage counts protect frequently
requested rows; recency breaks ties, and periodic count decay adapts to workload
changes. Each server start creates an empty row cache; request gathers populate
it without persistent cache files. `/metrics` exposes `flare_engram_` gather
time, disk traffic, cache hits, misses, evictions, occupancy, capacity and bytes.

## Prefix checkpoint policy

`cache::PrefixHistory` incrementally hashes 64-token blocks with BLAKE2b-128.
Exact partial endpoints hash the aligned parent plus the remaining tokens and
conditioning; complete-block keys do not depend on them.
The root and each block include a 32-byte model/recipe identity. Conditioning
events carry absolute token positions and 32-byte identities, retaining their
order even at the same position. An event at position N first affects prefixes
that consume that position; appending a later image preserves earlier keys.
Exact endpoint keys remain stable when later tokens or conditioning are appended.

`cache::PrefixCheckpointCache` owns metadata for immutable exact checkpoints.
A checkpoint privately owns its tail pages and one complete state snapshot;
ancestor pages are shared through parent links. Snapshot consumed length must
equal the endpoint exactly. The parent must match the supplied history,
and pages and snapshot IDs must fit prepared capacities without overlapping
existing or retired ownership.

Publication has two outcomes. `Created` transfers the proposed resources to the
cache. `Existing` keeps the proposal's unused resources with its caller; any
outstanding work on those resources must complete before the caller reuses them.
A created checkpoint whose readiness fence is incomplete reserves resources but
does not participate in lookup. `advance_completed` publishes it after the
reported fence completes. Pending-publication cancellation also observes that
fence before releasing resources.

Pinning increments the selected checkpoint and every ancestor. Direct pin
accounting prevents an ancestor unpin from consuming a descendant's reference.
The scheduler records GPU use through `mark_used` before submission; releasing
logical pins does not imply completion of that use. Leaf LRU eviction accepts
separate page and snapshot pressure targets and produces `RetiredResources`,
which explicitly carry their final-use fence. The cache continues reserving
those IDs. Only `advance_completed(...).reusable` grants permission to reuse
the corresponding storage.

Fences describe a single monotonic completion timeline covering model execution
and associated transfers. The caller supplies truthful completion information;
this metadata library never queries CUDA or performs state copies. Checkpoint
IDs are monotonic within one cache and are not reused after eviction. Resource
IDs are zero-based logical ordinals within the prepared page/snapshot capacities;
the prepared graph contract maps them to backing storage and padding sentinels.

## Native image preparation

The `images` feature adds `frontend::images::ImageProcessor` and PNG, JPEG,
and static WebP decoding; it also enables `frontend`. Typed core metadata
selects the preparation profile and token layout independently. Profiles cover
Qwen, MiMo, patch-merge, DeepSeek, DeepSeek row-major, Moonvit, Step, RADIO,
GLM, and padded-grid inputs.
The Qwen processor uses an RGB8, bicubic-antialias profile with
16×16 patches, temporal duplication by two, spatial merge by two, a 48×48
learned position table, and fused normalization `(pixel - 127.5) / 127.5`.

`process` produces packed `pixel_values[N,1536]` float32, image grids `[1,H,W]`,
vision positions `[N,2]` int64, interpolation indices `[N,4]` int64 and weights
`[N,4]` float32, and per-image cumulative patch offsets int32. Patch features
are ordered by channel, temporal copy, pixel row, and pixel column; patches
are ordered by 2×2 spatial merge blocks. `expand_prompt` consumes the rendered
chat text, expands each image placeholder, tokenizes the result, and returns
modality IDs and four unpadded position rows `[text,T,H,W]` plus the mRoPE delta.
The model scheduler can transpose the last three rows into `[tokens,3]`.

Prompt construction describes each image without building its tensors. The
description includes geometry, semantic identity, token dependency span, and
canonical feature range. Admission validates each image against its captured
patch capacity and feature-window capacity. Total conversation image tokens
are bounded by context, with encoded-body limits enforced separately.
Feature-slot concurrency limits requests needing vision computation.

After prefix lookup, only images whose dependency spans extend past the
consumed endpoint require tensors or vision graphs. A fixed CPU worker prepares
those inputs. Host retention can reuse source descriptions and prepared
tensors through the configured memory and disk tiers. Whole images are grouped
into windows that fit the prepared feature arena. Each window's features are
packed contiguously from zero without changing canonical prompt positions or
indices. Prefill stops before the next window's first image dependency; atomic
image spans remain indivisible.

A request spanning multiple windows reserves one private snapshot row. After
prefill consumes a window, a snapshot preserves complete continuation state,
including partial cache pages and recurrent histories. Restore installs that
state under a fresh execution generation before vision writes reuse the arena.
The snapshot stays private, does not require host retention, and does not add
cached-token usage. GPU fences protect the snapshot row, feature slot, and
inputs through cancellation and retirement. Preemption and failed host restores
retain encoded sources and replan from the restored endpoint. See
[host retention](../docs/prefix-cache-offload.md).

`RgbImage` holds the decoded representation required by its profile, including
alpha and CMYK data where preprocessing consumes them. Semantic identities
include these channels and resize mode as applicable. Encoded-source keys
include the declared media type, source bytes, and processor settings.
Animation, video, and remote URL fetching are unsupported. Decoding and
preprocessing execute in Rust.

The resize coefficient generator and integer convolution follow PyTorch
[`UpSampleKernel.cpp`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/UpSampleKernel.cpp)
and its uint8 antialias implementation at `cf30153c4c131c8164ee7798e5022d810682e2cb`;
the applicable license is [licenses/PyTorch.txt](licenses/PyTorch.txt).
The scalar Rust implementation preserves PyTorch's adaptive coefficient
precision and intermediate uint8 rounding. The `llm-multimodal` implementation
at `f0985ef65967615db2c79279aa07818499301bfd` uses Pillow's different coefficient
precision and serves as an interface reference, not a pixel reference.

Image fixtures in `tests/frontend/qwen38_images.json` pin Transformers
`4cc2aa84301c9aa210b5513dab9fefae03981f6e`, PyTorch `2.13.0+cu130`, Torchvision
`0.28.0+cu130`, Pillow `12.3.0`, and all checkpoint asset hashes. They use the
actual `AutoProcessor` Torchvision backend and canonical Qwen4Exp position
helpers. SHA256 checks compare every resized RGB byte and every little-endian
float/integer tensor byte, with no numerical tolerance. The fixture set covers
19 complete pipelines and six geometry cases, including the 16,384-patch
boundary, half-even rounding, independent-axis resize, PNG color modes, and
mixed/repeated image batches.

```bash
cargo test --manifest-path helios/Cargo.toml --features images --offline
cargo run --manifest-path helios/Cargo.toml --features images,test-tools --offline \
  --bin test_images -- "$SNAPSHOT" tests/frontend/qwen38_images.json
.venv/bin/python tests/frontend/generate_image_fixtures.py \
  "$SNAPSHOT" tests/frontend/qwen38_images.json \
  --binary-dir /tmp/flare-image-reference
```

The qualifier accepts an optional third argument for writing mismatching
native tensor bytes. Fixtures contain fingerprints rather than the full
100 MiB largest patch tensor; the generator's `--binary-dir` records full
reference tensors for diagnosis. These checks qualify CPU preparation; vision
graph execution and decoder consumption require separate model integration.

Padded-grid fixtures in `tests/frontend/inkling2_images.json` pin the Inkling 2
image processor and vision tower of Transformers pull request 50682 at commit
`1b2f625956be1416816447eba32447a1d268d095`, PyTorch `2.13.0+cpu`, Torchvision
`0.28.0+cpu`, Pillow `12.3.0`, and the checkpoint's `processor_config.json` and
`config.json` hashes. They record `Inkling2ImageProcessor` outputs and the
tensors that `Inkling2VisionModel` passes to its patch embedding, rotary
embedding, and first attention layer. Images open with `PIL.Image.open` without
EXIF transposition. The 26 cases cover sizes from 1×1 to 4096×4096, extreme
aspect ratios, mixed batches, PNG RGB, L, P, RGBA, LA, and tRNS inputs, EXIF
orientation 6 in PNG and JPEG, CMYK JPEG, and lossless WebP with and without
alpha. Library tests decode PNG and WebP natively, compare JPEG decoding with
Pillow within two levels per channel, and then compare every native-patch,
position, interpolation, and sequence-offset byte of preparation and
class-token packing. Regenerate them in an environment that holds only those
pinned packages:

```bash
python -I tests/frontend/generate_inkling2_image_fixtures.py \
  "$SNAPSHOT" tests/frontend/inkling2_images.json --binary-dir /tmp/inkling2-image-reference
```
