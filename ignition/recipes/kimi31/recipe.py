# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 Pro MXFP4 preparation and serving defaults."""

from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults, Speculation
from ignition.recipes.kimi3.recipe import RECIPE as KIMI3, prepare as prepare_kimi, usage
from .dspark_config import DSPARK_REPOSITORY, DSPARK_REVISION


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    if not plan.context_parallel:
        raise ValueError("Kimi K3.1 preparation requires context parallelism")
    return prepare_kimi(snapshot, plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache,
        progress=progress, retain_graphs=retain_graphs, package_storage=package_storage, ranks=ranks,
        recipe=RECIPE, draft_source=(DSPARK_REPOSITORY, DSPARK_REVISION))


RECIPE = Recipe(
    name="kimi31", served_model_name="Kimi-K3.1-Pro", prepare=prepare, usage=usage,
    # https://huggingface.co/moonshotai/Kimi-K3.1-Pro#3-evaluation-results
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1),
    checkpoints=(Checkpoint(alias="kimi-k3.1-pro", repository="moonshotai/Kimi-K3.1-Pro",
        revision="d9e99b0792a5e93b78ea9bea62a68167e36ddd9e", architecture="KimiK31ForConditionalGeneration",
        stored_bytes=1_561_217_377_280,
        # K3's matching target layout plus 357,052,416 BF16 indexer bytes replicated on sixteen ranks.
        device_bytes=1_491_026_420_236 + 16 * 357_052_416 + 8_558_149_632,
        draft_bytes=8_558_149_632),),
    context=1048576, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True, graph_budget=64),
    max_draft_tokens=7, verification_rows=0, max_images=4, image_bytes=32 << 20,
    online_mxfp8=True, online_mxfp8_scope="kda-input", online_mxfp8_scopes=("all", "kda-input"),
    stream_overlap=True, language_model_only=True,
    b12x_environment=MappingProxyType({**KIMI3.b12x_environment, "B12X_PCIE_DCP_THREADS": "0"}),
    # Graph storage is a planning allowance; actual graph memory is reported after preparation.
    ranks=MappingProxyType({16: RankCosts(state_row_bytes=33_238_272, page_bytes=2_912_256,
        graph_bytes=9 << 29, page_tokens=128 * 16)}),
    plain_ranks=MappingProxyType({16: RankCosts(state_row_bytes=30_275_328, page_bytes=2_174_976,
        graph_bytes=9 << 29, page_tokens=128 * 16)}),
)
