# DeepSeek V4.1 BRRR fusion measurements

These historical measurements motivate the TP4/SM120 small-verification
integration. They do **not** establish performance or numerical acceptance for
Flare base `e774a468cee4ade64393ba9d856788aaef974ed7`.

## Same-depth serving comparison

Both configurations used Flare `18b16e170bb9eab03e2b1800107c72966c56594f`, FlashInfer
`e977747a8354833835501e5e43adf38da38f84c5`, and the native DeepSeek checkpoint revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Four RTX PRO 6000 Blackwell Workstation GPUs
ran at 395 W each. The comparison retained D4, FP8 Markov head, NVFP4 draft
head, native expert scales, BF16 MLA, 32 request slots, a 128-row verification
budget, alpha 0.6, balanced scheduling, RAM Engrams and no host prefix-cache tier.

LLM Inference Bench 0.6.2 measured 30-second sustained-decode windows with a
three-second warmup and an 8,192-token per-request cap. Default reasoning and
model-default sampling were retained. EOS was suppressed only for speed tests.
Nominal 0K included 91 input tokens; nominal 32K included approximately 32.3K.
Each cell below is one window, not a mean. Fresh context nonces and stochastic
completions differed across runs.

Aggregate decode tokens/second:

| Context | Prepared graph path | C1 | C2 | C4 | C8 | C16 | C32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0K | Unfused D4 | 318.8 | 558.1 | 822.1 | 1139.7 | 1494.5 | 2280.9 |
| 32K | Unfused D4 | 339.6 | 573.0 | 793.5 | 1147.7 | 1470.7 | 2230.7 |
| 0K | Fused D4, first sweep | 359.1 | 585.0 | 814.0 | 1140.1 | 1479.9 | 2307.6 |
| 32K | Fused D4, first sweep | 338.7 | 577.9 | 820.2 | 1148.7 | 1478.0 | 2218.3 |
| 0K | Fused D4, final sweep | 350.7 | 559.3 | 793.2 | 1155.5 | 1470.1 | 2267.6 |
| 32K | Fused D4, final sweep | 359.4 | 583.5 | 816.1 | 1119.6 | 1469.8 | 2220.2 |

[Raw numeric observations](measurements.json) include both fused full sweeps
and the intervening C1/C2 repeat. The repeat measured 355.3/560.5 tok/s at 0K
and 352.0/544.4 tok/s at 32K. These results show variation and some regressions;
they are not a statistically established gain at every concurrency.

The final sweep's C1 point estimate increased 10.0% at 0K and 5.8% at 32K.
C2 increased only 0.2% and 1.8%. The checkpoint bytes were unchanged, but the
activation arithmetic was not identical. Treat this as a comparison of the two
execution paths, not an equal-arithmetic microoptimization.

## Combined deployment at 395 W and 450 W

The `972e28e44781d8d0c84d70ae434db92c36076a43` fused deployment additionally
used the masked-attention dependency update and expert-reuse tuning workload.
These are combined-stack results, not isolated effects of this PR:

| Power per GPU | Context | C1 mean +/- sample SD | C2 | C4 | C8 | C16 | C32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 395 W | 0K | 357.33 +/- 17.03 | 604.46 | 823.84 | 1119.47 | 1635.63 | 2222.72 |
| 395 W | 32K | 354.65 +/- 13.79 | 594.95 | 786.61 | 1110.62 | 1637.72 | 2194.78 |
| 450 W | 0K | 364.45 +/- 21.30 | 632.32 | 842.44 | 1145.03 | 1680.14 | 2266.83 |
| 450 W | 32K | 354.68 +/- 14.35 | 569.32 | 815.44 | 1118.26 | 1572.19 | 2239.50 |

The client was LLM Inference Bench 0.8.0 at
`52216d70865a015b34689626d04a80b89c9dc45f`. C1 used ten valid windows per
context; C2-C32 used one. Full 0K-256K matrices, individual C1 samples, rejected
windows and methodology are published in the
[395 W results](https://github.com/shihanqu/RTX-Pro-6000-Recipes/blob/c7718579cb24d4d9137e5fbb1095acd07f150615/deepseek-v41-flare-brrr-d4-tp4/results/2026-11-09.json)
and [450 W results](https://github.com/shihanqu/RTX-Pro-6000-Recipes/blob/c7718579cb24d4d9137e5fbb1095acd07f150615/deepseek-v41-flare-brrr-d4-tp4/results/2026-11-10-450w.json).
Do not pool these samples with the 0.6.2 comparison above.

## Numerical evidence and limits

The retained component diagnostic used real checkpoint weights with synthetic
activations at five and eight rows across four ranks. The port reproduced the
original BRRR suffix within its measured 17-run repeatability plus rounding bound.
That reference result establishes faithful porting, not Flare equivalence.

Against the decomposed Flare path, the attention output relative RMS differences
were 3.62% and 3.57%, and the downstream MLP differences were 23.94% and 15.83%.
Feeding **identical attention output** to both suffixes reduced MLP relative RMS
differences to 2.74% and 2.59%; the residual then matched exactly. Both comparisons
remain in the numeric evidence. The aligned-input result must not replace the
full-path discrepancy or be described as a whole-model error bound.

Historical generated-answer checks passed Estonia 10/10 at C1, C2 and C10,
four concurrent image checks, JSON output, tool round trips and request
transitions. Those checks do not establish broad quality equivalence, acceptable
operator tolerances or correctness on this PR's upstream base.

## Before merge

A dedicated four-GPU qualification must compare base and candidate on the same
commit, checkpoint, dependency pins, power, sampling and cache settings. Repeat
C1-C32 at 0K and 32K; report every retained measurement and regression. Re-export
both cores and measure them through `flare run`, not diagnostic Python serving.

Numerical acceptance must cover nonzero real checkpoint projections, routing,
partial/inactive rows, alternating graph capacities, speculative rollback,
cache boundaries and native replay allocation/lifetime checks. Preserve
operator-specific tolerances and the checkpoint calibration. Repeat vision,
tools and Estonia, while treating them as separate functional checks.
