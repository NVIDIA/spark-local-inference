# GLM-5.3 Flash: batched speculative decoding corrupts answers on DGX Spark

Status: 22887c1 restored the selection width inside each graph and was validated
on Spark at TP2 and TP4. GLM now uses b12x's page-table extent directly, without
a width tensor in scratch (section 7). A separate, rarer device error at TP4
with 16 concurrent requests remains open. Before the fix, export zeroed the
selection width read by the tiled indexer, corrupting verification above 16
query rows.

## 1. Summary

On DGX Spark (GB10, SM121), with one rank per Spark over RoCEnante, GLM-5.3
Flash produces corrupted answers when MTP speculative decoding verifies several
requests in one round. It reproduces at tensor parallel 2 and 4. With eight or
more concurrent requests most replies are empty, invent content, or name the
wrong record. The same cores answer correctly with speculation disabled
(`--mtp-draft-tokens 0`) and with speculation limited to one or two concurrent
requests. On a GB300 (SM103, one GPU) the identical test is correct at 16
concurrent requests.

Two other symptoms were attributed to it:

- **Long prompts without a cached prefix answer wrongly.** In the failing runs,
  post-prefill execution included speculative rounds at larger batch capacities
  (16, 8, 4, then 2 rows, with the request in row zero and the rest inactive)
  before settling at batch one. Cached-prefix requests in those runs started
  at batch one. Cache reuse does not itself determine the graph capacity;
  scheduler state and adaptive verification determine the selected graph.
- **Concurrent streams occasionally fail with device error 6
  (`INVALID_SPECULATION`).** An instrumented build that reports the failing
  check shows the verifier rejecting the target distribution (non-finite or
  empty probabilities), often in several rows of the same round. This one
  survives the fix: TP4 cores at concurrency 16 still fail about one
  request in sixteen this way, as they did before it, while TP2 at
  concurrency 8 and the GB300 have not shown it. It is being investigated as a
  separate defect (section 7).

The cause is in core export, not in a kernel. Each GLM indexer passes b12x a
one-element `active_width` tensor that preparation fills once with the page
table's width in index pools. The tensor lives in the decoder's scratch, and
export treats all decoder scratch as disposable and zeroes it in the core.
b12x's tiled indexer route, which GLM decode graphs use above 16 query rows,
caps every row's candidate length at `active_width`, so in an exported core
it scores no candidates and returns indices left in its scratch from earlier
work. Sparse MLA then attends to unrelated records, and the model loses facts
from the prompt while staying fluent. The fused route used at 16 rows or fewer,
prefill and the GB300's SM103 indexer never read the tensor, and a model
prepared and served in one process is never exported, which is why each of
those is correct.

## 2. Environment

| Item | Value |
| --- | --- |
| Reproduced on | A development branch of Flare `main` carrying unmerged multi-host changes, with FlashInfer 64916e85 plus the RoCEnante variable-length all-gather from [local-inference-lab/flashinfer#12](https://github.com/local-inference-lab/flashinfer/pull/12) |
| Flare `main` | Reproduced at 7ba743d with its pinned FlashInfer (64916e85, `pyproject.toml`) at TP2, with the multi-host export fix: 2 correct and 6 corrupt of 8. Fixed by 22887c1; to reproduce, build 41c9ff6, which has the export fix and not the indexer fix |
| Checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` (revision dec48abd) |
| Cores | TP2: `--ranks 2 --max-concurrency 8`; TP4: `--ranks 4 --max-concurrency 16`; both `--gpus 0 --prefill-chunk-size 4096 --no-context-parallel` with the default adaptive MTP |
| Hardware | DGX Spark, ConnectX-7 at 200G through one switch |
| Capacity per rank | TP2: 1,107,968 KV tokens, 54 state rows (1,159,680 tokens and 56 rows with speculation off). TP4: 6,716,416 KV tokens, 176 state rows. FP32 recurrent and BF16 convolution state |
| Sampling | The model card's T=1, top-p 0.95, with a seed per request |

## 3. Results

[`scripts/repro_glm_batched_speculation.py`](../scripts/repro_glm_batched_speculation.py)
produced every row. The 10K prompt is the script's default `--size 8000`
(about 10.5K tokens; answer `2U2GYCZF, 198`). "Cached" publishes the prompt
with a one-token request, then sends N copies at once with different seeds, so
every row of the speculative batch is a live request restored from the same
checkpoint. "Uncached" sends prompts that begin with distinct nonces, one at a
time. The script classifies each reply as `correct`, `neighbour` (the code
with record 197 or 199), `misplaced` (the code with another record, usually
99) or `corrupt` (empty, or a wrong, garbled or invented code). Neighbour and
misplaced answers are model-level misses that also occur on unaffected
servers; corrupt answers are the defect. It also reports the speculative
rounds the test ran by batch × width, read from the server's `/metrics`, and
how many of them were in graphs above 16 query rows.

| Server | Test | Correct |
| --- | --- | --- |
| GB300, adaptive MTP, concurrency 16 | cached 10K ×16, two runs | 15/16, 13/16 |
| GB300, adaptive MTP | cached 96K ×16 | 16/16 |
| GB300, adaptive MTP | uncached 133K–260K | 14/14 |
| Spark TP2, adaptive MTP, concurrency 8 | cached 10K ×8, two seeds | 1/8, 0/8 |
| Spark TP2, Flare `main` 7ba743d with the section 4 export fix, adaptive MTP, concurrency 8 | cached 10K ×8 | 2/8 (6 corrupt) |
| Spark TP2, adaptive MTP | cached 10K ×2 | 2/2 |
| Spark TP2, speculation off, concurrency 8 | cached 10K ×8 | 8/8 |
| Spark TP2, MTP draft removed from decode rounds (diagnostic, see section 6), fixed width 4, concurrency 8 | cached 10K ×8, two seeds | 0/8, 0/8 |
| Spark TP2, same diagnostic | cached 10K ×2 | 2/2 |
| Spark TP2, core exported with `--no-compile`, adaptive MTP, concurrency 8 | cached 10K ×8, two seeds | 1/8, 0/8 |
| Spark TP2, same core | cached 10K ×2 | 2/2 |
| Spark TP2, `--no-compile`, DSA indexer forced to its fused route at every row count (diagnostic, see below), concurrency 8 | cached 10K ×8, three seeds | 7/8 (+1 neighbour), 8/8, 8/8; no corrupt replies |
| Spark TP2, same diagnostic | cached 10K ×2 | 2/2 |
| Spark TP2, prepared and served in one process (`ignition debug-serve --no-compile`), concurrency 8 | cached 10K ×8, three seeds | 8/8, 6/8 (+2 neighbour), 6/8 (+1 neighbour, 1 misplaced); no corrupt replies |
| Spark TP2, same server | cached 10K ×2 | 2/2 |
| Spark TP2, exported with the fix in section 7, compiled, adaptive MTP, concurrency 8 | cached 10K ×8, three seeds | 4/8 (+4 neighbour), 8/8, 7/8 (+1 neighbour); no corrupt replies |
| Spark TP2, same core | cached 10K ×2; uncached 96K ×3 | 2/2; 3/3 |
| Spark TP2, Flare `main` 7ba743d with the export fix and the fix in section 7, concurrency 8 | cached 10K ×8, three seeds; uncached 96K ×2 | 8/8, 8/8, 4/8 (+4 neighbour); 2/2 |
| Spark TP4, exported with the fix in section 7, compiled, concurrency 16 | cached 10K ×16, two seeds | 15/16 (+1 device error 6), 15/16 (+1 neighbour); no corrupt replies |
| Spark TP4, same core | cached 96K ×16; uncached 96K ×3 | 14/16 (+2 device error 6); 3/3 |
| Spark TP4, adaptive MTP, concurrency 16 | cached 10K ×16 | 1/16 |
| Spark TP4, adaptive MTP, concurrency 16 | cached 96K ×16 | 0/16 (13 reached the 16,384-token limit) |
| Spark TP4, adaptive MTP | cached 10K ×2, ×4, ×8 | 2/2, 3/4, 1/8 |
| Spark TP4, fixed width 2 (`--mtp-draft-tokens 1 --no-mtp-adaptive-verification`), concurrency 8 | cached 10K ×8 | 3/8, all five misses the neighbouring record (no corruption) |
| Spark TP4, fixed width 4 (`--mtp-draft-tokens 3`), concurrency 8 | cached 10K ×8 | 1/8 |
| Spark TP4, speculation off | cached 10K ×16 | 10/16 |
| Spark TP4, speculation off | uncached 48K–133K | 5/5 |
| Spark TP4, adaptive MTP, default concurrency | uncached 48K–133K | about 2/15 |
| Spark TP4, adaptive MTP, `--max-concurrency 1` or `2` | uncached 96K–133K | 5/5, 4/4 |

With speculation on, the failures are empty replies that reason until the
token limit, invented or truncated codes (`2U2GYZF`, `2Q7XV81M`), claims that
the prompt holds no code, and records far from the needle. The misses with
speculation off all name the neighbouring record (199 for 198), a model-level
miss the GB300 also makes at a similar rate. Those are a separate question.

The trigger is the number of query rows in the speculative graph, batch
capacity times verification width: corruption appears above 16 rows and not
at or below it. Plain decode at 16 requests, eight requests at width 2, two at
width 6 and the draftless diagnostic at two requests and width 4 are all
correct apart from the neighbouring-record misses; eight requests at width 4
or 6, and sixteen at any width above one, corrupt. A single live request also
fails when it runs in a larger graph (the post-prefill rounds above), so the
graph's planned capacity matters, not the number of live requests. With the
MTP draft removed, every round commits exactly one token per request, as plain
decode does, and still fails, so the fault is in the target verification
graph rather than in the draft model.

The b12x DSA indexer, which selects each query row's 512 index pools for sparse
MLA, plans its fused route for GLM decode graphs of up to 16 query rows and its
tiled route above (`_C4_B12X_FUSED_MAX_ROWS` in
`flashinfer/experimental/b12x/attention/dsa_indexer/fused_indexer.py`). A
diagnostic FlashInfer build that raises that limit to 128, so every graph in a
concurrency-8 core uses the fused route, removes the corruption: 26 of 26
replies correct or neighbouring, with 95 of the 303 speculative rounds at
concurrency 8 in graphs of 24 to 48 rows. The same core without the change
corrupts six to eight replies of eight. The corruption therefore follows the
tiled route. The tiled route is nevertheless correct when called directly
(section 6), and the same TP2 configuration prepared and served in one process
with `ignition debug-serve` is also correct, so the fault lies in what an exported
core carries rather than in the kernel's arithmetic.

GLM's `IndexScratch` holds the `active_width` tensor that every indexer layer
passes to b12x, initialised once to the page table's width times 64 index pools
(`ignition/recipes/glm53/indexer.py`). No graph writes it. The recipe's
`disposable_ranges` (`ignition/recipes/glm53/export.py`) walks every field of the
decoder and prefill scratch, so the tensor is among the ranges that
`export_model` zeroes and omits from the core. b12x's tiled decode scorer
reads `active_width[0]` as the live width and caps each row's candidate count
by it; at zero it computes no logits, and the tiled top-k then selects from
whatever its reused scratch holds. A single-GPU harness that drives Flare's own
`PreparedSparseIndexer` lifecycle (tuning, finalisation, a cache rebind and
graph capture on a shared scratch arena) selects the exact top-k as prepared,
returns wrong selections for every live row of 32- and 48-row graphs once the
tensor is zeroed as export does, and is exact again when the graph rewrites it.
The DeepSeek recipes write their own `active_width` inside the graph, which is
why they are unaffected.

The fused kernel is not a drop-in replacement above 48 rows on GB10: its pack
scratch holds `num_sms × topk` entries, and a 64-row graph stops with `fused
indexer pack scratch too small`. Concurrency-16 cores, whose largest
verification graph has 128 rows, cannot use it as a workaround unchanged.

## 4. Reproduce on DGX Spark

Build Flare at 41c9ff6, the last `main` commit before the fix, with its
pinned FlashInfer (`git -C ../flashinfer checkout 64916e85`), and set up
multi-host serving as in [multi-host.md](multi-host.md): `flare agent` on every
peer with the same token, and RoCE between the Sparks. Two Sparks suffice. Host
names and addresses below are examples. Commits before 41c9ff6 cannot export a
multi-host core at all: each peer stops with `ValueError: runtime assets
must contain only tokenizer and chat_template`. To confirm the fix, repeat the
steps at 22887c1 or later.

1. Export a TP2 core. On rank zero:

   Replace `CACHE_ENTRY` in conversion commands with the directory path printed by export.

   ```sh
   .venv/bin/ignition export glm-5.3-flash --ranks 2 --rendezvous 10.200.0.12:29630 \
     --gpus 0 --max-concurrency 8 --no-context-parallel --verbose
   .venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/glm53-tp2.safetensors
   ```

   On the peer:

   ```sh
   .venv/bin/ignition join 10.200.0.12:29630 --rank 1 --ranks 2
   ```

   For four Sparks, use `--ranks 4 --max-concurrency 16`, run one `join` per
   peer with ranks 1–3, and pass `--concurrency 16` in step 3.

2. Serve from rank zero with the core defaults (adaptive MTP):

   ```sh
   ./flare-cli/target/release/flare run /data/flare-runtime/glm53-tp2.safetensors \
     --hosts spark2 --gpus 0 --host 0.0.0.0 --port 8000
   ```

   Each peer caches a complete copy of the core under
   `~/.cache/flare/cores` (about 181 GB at TP2), so leave that much free on
   every peer; copies of older cores are not removed automatically.

   Accept the development publisher on first use, or pass `--publisher-keys`,
   `--dangerously-approve-key` and `--dangerously-accept-publisher` for an
   unattended start.

3. From any machine that can reach the server:

   ```sh
   python3 scripts/repro_glm_batched_speculation.py cached --url http://spark1:8000 --concurrency 8
   ```

   A healthy server returns no `corrupt` replies (the GB300 gives 7 correct
   and 1 neighbour). The defect gives six to eight `corrupt` of eight, and the
   round report shows most rounds in graphs above 16 query rows.

4. Controls. Restart the server with one option at a time and repeat step 3:
   - `--mtp-draft-tokens 0`: correct (8/8 at TP2).
   - `--max-concurrency 2`, with `--concurrency 2` in the script: 2/2.

5. Long prompts without a cached prefix, sent one at a time (about 96K tokens
   each). They fail on the default server and pass under either control:

   ```sh
   python3 scripts/repro_glm_batched_speculation.py cold --url http://spark1:8000 --size 70000 --count 3
   ```

## 5. Reproduce on RTX PRO 6000 (SM120)

Flare's GLM recipe selects SM103 kernels on SM103 and b12x kernels everywhere
else, so an SM120 host runs the same path as Spark: the b12x DSA indexer, sparse
MLA, KDA verification recording and commit, NVFP4 MoE, blockscaled GEMMs and
short convolution. Three things differ: b12x compiles for `sm_120a`; autotuning
measures tactics on the local GPU (188 SMs against GB10's 48); and
tensor-parallel collectives use b12x's PCIe transport instead of RoCEnante.
None of these differences matter to the cause, which is in core export, so
an SM120 core exported without the fix in section 7 should corrupt answers
the same way and one exported with it should not. GLM has not yet been run
this way.

The checkpoint holds about 166 GB of weights. Four 96 GB cards at tensor
parallel 4 leave about 44 GB of weights per rank. TP2 also fits on two 96 GB
RTX PRO 6000 cards, with less KV capacity; the available capacity depends on
concurrency, graph workspace, and continuation-state backing. The defect appears
at both TP2 and TP4 on Spark.

1. Export, one rank per GPU:

   ```sh
   .venv/bin/ignition export glm-5.3-flash --gpus 0-3 --max-concurrency 16 \
     --no-context-parallel --verbose
   .venv/bin/ignition convert CACHE_ENTRY --form packed --output glm53-tp4-sm120.safetensors
   ```

2. Serve:

   ```sh
   ./flare-cli/target/release/flare run glm53-tp4-sm120.safetensors --gpus 0,1,2,3 --host 0.0.0.0 --port 8000
   ```

3. Run section 4's steps 3 and 4 against this server with `--concurrency 16`
   (and 2 for the control). Report the counts and the GPU model (Workstation or
   Max-Q).

A result that differs from Spark, either corruption with the fix or none
without it, would point to a second defect and is worth reporting.

## 6. What has been ruled out

Each check passed at the shapes the failing rounds use. Kernel checks ran
standalone against b12x or PyTorch references.

| Component | Check | Result |
| --- | --- | --- |
| RoCEnante all-reduce and vocabulary all-gather | Four Sparks, 2–96 rows (up to 7.4 MB per rank), CUDA-graph replays with a new pattern each replay | No mismatched bytes |
| b12x KDA verification recording and commit | TP4 geometry (16 heads), 1–16 sequences of mixed length, inactive rows, strided state indices as Flare passes them | Exact |
| b12x DSA indexer (32 heads, top-512 pools) | Fused route (up to 16 query rows) and tiled route (32–128 rows) on SM120 and GB10, graph replays with changing inputs, inactive rows, scratch overwritten between replays, 1 MiB guard bands around scratch and output | True top-k on every row; no writes outside the declared buffers; inputs unchanged |
| b12x sparse MLA decode | 16 heads, 2051 selected slots, 528-byte FP8 records, 16–96 rows with inactive rows (SM120) | Cosine ≥ 0.99998 |
| b12x NVFP4 MoE | Inactive tokens routed to expert -1 with NaN inputs, which route top-k produces for NaN logits; 16–96 tokens (SM120) | Live tokens bit-identical |
| b12x short convolution, speculative mode | 6144 and 24576 channels, 1–16 sequences, 7 draft columns (SM120) | Exact |
| Concurrent branches | Core with the MLA query chain and the MoE shared expert (`ConcurrentSharedExpert`) run serially | Still fails (0/16) |
| KDA verification implementation | Core that records and replays KDA verification through the portable Triton path (`glm53/kda_recovery.py`) on SM121 | Still fails (0/16) |
| Autotuned b12x tactics | Core prepared with b12x heuristic defaults (autotuning off) | Still fails (0/16) |
| Compiled forwards | Core exported with `--no-compile`: each rank captures its own uncompiled forwards, so no compiled code or baked handles are shared between ranks, and no Inductor or Triton code is generated for the speculative shapes | Still fails at TP2 (1/8, 0/8; 2/2 at two requests) |
| MTP draft model and draft-cache writes | Core whose decode rounds propose the pending token with certainty, with no draft forward or draft-cache writes (`glm53/speculative.py`), fixed width 4 | Still fails (0/8) |
| Per-step inputs to peer ranks | Each upload is sent separately and copied before the graph launches | Correct by inspection |
| Ranks disagreeing on tokens | The step validator compares every rank's pending token each round (error 32 on mismatch) | No mismatches |

The standalone kernel checks used b12x's default configurations; cores
normally serve autotuned tactics. A vLLM deployment of GLM-5.3 Flash on two
Sparks that we inspected uses the same b12x kernel families, but with b12x
tuning off, PyNCCL collectives and at most two concurrent requests in its
logs, so it does not cover this case.

## 7. Fix and remaining device error

22887c1 rewrites the width inside every graph, where the first indexer layer
already rewrites the page table and pool lengths (`PreparedGlmIndexer.__call__`
in `ignition/recipes/glm53/indexer.py`):

```python
        if self.prepares_tables:
            # Cores zero this scratch, so every graph restores the selection width b12x reads.
            s.active_width.fill_(s.page_table.shape[1] * 64)
```

With this change and an unmodified FlashInfer, no exported core has
returned a corrupt reply: a compiled TP2 core at concurrency 8 on the
development branch and on `main`, and a compiled TP4 core at concurrency 16,
the configuration in which the defect was first seen, including 96K-token
cached and uncached prompts (section 3).

The b12x FP8 DSA indexer accepts `active_width=None` to use the page-table
extent as its scoring bound. Its kernels derive that extent from tensor shape
metadata and still mask each row by its visible cache length. Callers with a
device-computed width, including DeepSeek, retain the explicit tensor input.
Neither binding nor scratch initialization manufactures a constant width tensor.

GLM passes `None` and has no `IndexScratch.active_width` or width-fill graph
node. The native GLM regression covers 2- and 32-row decode and 65- and
4096-row prefill after zeroing all index scratch. b12x graph tests cover
static and device widths, scratch poisoning, inactive rows, multiple page
spans, logical and physical selection outputs, and allocation stability.
These checks ran on SM120; the page-table-extent implementation has not been
rerun on Spark. Cores prepared before the fix require re-export.

1. Merged: 41c9ff6 (multi-host export) and 22887c1 (the fix). The indexer's
   native-replay test (`tests/recipes/test_glm53_indexer_cuda.py`) now replays
   a 32-row tiled decode graph after zeroing all index scratch, as export does;
   it fails with 95.8% of selections wrong without the fix. A broader check
   that zeroes every export-disposable range before replaying each captured
   graph would catch this class of defect in any recipe.
2. Find the source of the remaining device error 6 at TP4, concurrency 16.
   It concentrates in long prompts: a fixed core failed 3 of 48 cached 96K
   requests and 1 of 128 cached 10K requests, and none of 48 at concurrency 8.
   A TP4 core carrying the fix and a diagnostic that replaces error 6 with
   the failing check (`ignition/components/speculation.py`) failed 6 of 160
   cached 96K requests, every one with code 115: NaN raw target logits in all
   four vocabulary quarters, each the slice one TP4 rank projects. NaNs in all
   four slices do not establish that the pre-projection hidden state is NaN
   or identical across ranks. Measurements before and after projection and
   gathering are required to locate the first non-finite values. Failures
   arrive together (4 and 2 replies in two
   of ten batches of 16), with no batch × width mix distinguishing the
   failing batches.

   A core that also records, for each token row, the first decoder point
   whose output holds NaN (each layer's attention input and output before and
   after the all-reduce, and its MLP input and output before and after the
   all-reduce) placed all 3 failures in 160 such requests inside KDA layers
   29, 40 and 44: in each failing row, every earlier point and that layer's
   input were finite, and the layer's own output on the reporting rank, before
   its all-reduce, was NaN. Each failing request turned NaN in a different
   KDA layer.

   The failure is timing-sensitive. A core that additionally checks each
   KDA layer's recurrent and convolution state on entry and its projections,
   short convolution and recurrent output failed no requests in 480 (30
   batches), against 13 failures in 5 of 30 batches for the cores without
   those checks; the checks change no data, only the work around each KDA
   layer. The KDA output-gate projections run on an auxiliary stream beside
   the input projections (`ConcurrentProjectionChain` in
   `ignition/recipes/glm53/kda.py`). Their GEMMs stay within their declared
   workspace and output at every GLM decode shape from 8 to 96 rows at TP2
   and TP4 shards (guard-band test, SM120), so an overrun of the shared arena
   by those plans is not the cause.

   b12x's fused indexer, which serves every GLM decode graph of 16 rows or
   fewer on SM12x, lets thread zero announce its CTA's arrival at the
   cooperative merge's grid barrier (`_fused_group_barrier` in
   `fused_indexer.py`) without first synchronizing the CTA, so peers can read
   a histogram before every warp has published its bins. A test that delays
   one warp per CTA shows peers passing the barrier early in 8 of 8 runs, and
   none with a `cute.arch.sync_threads()` before the arrival. A core with
   that synchronization still failed 4 of 160 requests with the same code, so
   the race is real but is not the source of error 6.
3. An audit of the other recipes (DeepSeek V4 and V4.1, Qwen 3.8, MiMo V2.6)
   and of the b12x and FlashInfer bindings they serve found no other
   preparation-time value that export zeroes and a graph reads. b12x's
   compressed sparse MLA split-control scalars written at binding were also
   identified as initialized scratch but were unread by the inspected kernels.
   The indexer's internal constant width cap has been removed; omitting
   `active_width` uses page-table extent without initialized scratch.
