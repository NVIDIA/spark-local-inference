# BRRR small-verification fusion

This preparation component adapts Fraser Price's BRRR decoder prologue and
suffix from revision `0bc4dbd0a15458e0957b899d72c7a5089e3335a5` of
[DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR/tree/0bc4dbd0a15458e0957b899d72c7a5089e3335a5).
The adjacent LICENSE and unchanged BRRR NOTICE retain its attribution to vLLM,
Local Inference Lab, b12x and FlashInfer. The Python graph integration and
Flare-specific caller-owned output and teardown bindings are adaptations, not
claims of authorship of the BRRR kernels.

## Execution boundary

The suffix combines grouped attention-output projections, TP reductions,
hyperconnections, routing, shared experts and routed experts. A compatible
carried-state layer without Engram lookup also uses the prologue for its input
hyperconnection and Q/KV projections. Flare still performs rotary transforms,
sparse attention, cache writes, rollback and sampling. No native scheduler,
graph-family contract or serialized executable extension is added.

Selection is confined to single-host TP4 on SM120, BF16 source tensors with
shape `(physical_rows, 4, 5120)`, native expert scales and five through eight
physical target-verification rows. Physical rows are not concurrent requests.
The host predicate checks the kernel's fixed normalization, hyperconnection,
routing and activation constants. Other modes, geometries, devices and arithmetic
retain the existing path. Explicit native-dense-activation preparation and joint
projection tuning retain the decomposed operators.

Weights and scales are rearranged without recalibration. Dense block scales
must be identical across their 32-row block; otherwise preparation rejects the
layout. BRRR activation quantization and reduction arithmetic differ from the
decomposed path. This branch does not claim numerical equivalence or approval
of those differences for an upstream default.

The extension is built during preparation. Native operators expose output
mutations to forward compilation; captured graphs contain the resulting CUDA
work. A preparation owner retains repacked tensors, the engine and peer buffers
through capture and export. Teardown drains GPU work before freeing the engine,
including an engine that has not registered a layer. Serving an exported core
does not import the extension or call Python.

## Validation

Host tests cover eligibility, calibration rejection, byte-exact weight packing,
and Python teardown behavior. The CUDA regression isolates residual mixing,
normalization and repeated graph replay with zero projection weights. It does
not qualify nonzero projection arithmetic, the distributed suffix or full-model
outputs. Run it only with an explicitly assigned, otherwise idle SM120 device:

```bash
pytest tests/recipes/test_deepseek41_brrr.py
FLARE_RUN_CUDA_TESTS=1 pytest tests/recipes/test_deepseek41_brrr_cuda.py
```

The [benchmark evidence](../../../../benchmarks/deepseek41_brrr/README.md)
distinguishes historical serving results from the qualification still required
on the proposed upstream base. Numerical and four-rank lifetime acceptance are
merge requirements, not implied by passing the host tests.
