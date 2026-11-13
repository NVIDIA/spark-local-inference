# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-backed windowed MLA arithmetic, accepted writes and captured replay."""

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned GPU")


def bf16_weight(weight):
    if isinstance(weight, torch.Tensor):
        return weight
    from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch, materialize_mxfp8_scale_rows
    return dequantize_mxfp8_rows_torch(weight.weight.values,
        materialize_mxfp8_scale_rows(weight.weight))[:, :weight.in_features].bfloat16()


@pytest.mark.parametrize("quantized", [False, True])
@torch.no_grad()
def test_windowed_mla_checkpoint_and_replay(quantized):
    from ignition.components.paged_batch import Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.config import DSparkConfig, ModelConfig
    from ignition.recipes.kimi31.dflash import DraftAttention, DraftScratch, PreparedContext
    from ignition.recipes.kimi31.dflash_weights import DraftWeights

    checkpoint = os.getenv("FLARE_KIMI31_DFLASH_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("set FLARE_KIMI31_DFLASH_CHECKPOINT to the RadixArk draft")
    target = os.getenv("FLARE_KIMI31_CHECKPOINT")
    if target is None:
        pytest.skip("set FLARE_KIMI31_CHECKPOINT to the target checkpoint")
    config = ModelConfig.load(target)
    config = replace(config, draft=DSparkConfig.load(checkpoint, config))
    device, shard = torch.device("cuda:0"), TensorParallelShard(0,1)
    draft = DraftWeights.load(checkpoint, config, device, shard, target_head=None, online_mxfp8=quantized)
    weights = SimpleNamespace(config=config, shard=shard, draft=draft)
    backing = torch.zeros((3,6*1024*576), dtype=torch.float8_e4m3fn, device=device)
    rings = {f"draft{i}.latent": backing[:,i*1024*576:(i+1)*1024*576].view(3,1024,576) for i in range(5)}
    pool = SimpleNamespace(reserved_region=rings.__getitem__)
    rows, block = 1088,8
    context_metadata = Metadata.allocate(rows,1,1,device)
    context_metadata.state_rows.fill_(1)
    context_metadata.lengths.fill_(rows)
    context_metadata.query_start_loc[-1] = rows
    m = Metadata.allocate(block,1,1,device)
    m.state_rows.fill_(1)
    m.lengths.fill_(block)
    m.query_start_loc[-1] = block
    torch.manual_seed(621)
    auxiliary = (torch.randn((rows,6*7168),device=device)*.2).bfloat16()
    source = (torch.randn((block,7168),device=device)*.2).bfloat16()
    scratch = DraftScratch.allocate(config,shard,block,block,device)
    output = torch.empty_like(source)
    eps = config.draft.text["rms_norm_eps"]
    def norm(x,w):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)).bfloat16()*w
    with RecordedPreparationSession(device=device,checkpoint=str(checkpoint),workload_schema=f"mla-dflash-{quantized}",
            numerical_policy="a16-fp8-kv",compile_workers=16) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        context = PreparedContext(weights,auxiliary,context_metadata,pool,session=session,name="context",workspace=arena,
            runtime=None,coordinator=None,latent_runtime=None)
        attention = DraftAttention(0,draft.layers[0],source,config.draft,shard,scratch,m,pool,runtime=None,
            coordinator=None,session=session,name="attention",workspace=arena,latent_runtime=None)
        session.tune(show_progress=False)
        def set_position(start):
            context_metadata.starts.fill_(start)
            context_metadata.positions.copy_(torch.arange(start,start+rows,device=device))
            m.starts.fill_(start+rows)
            m.positions.copy_(torch.arange(start+rows,start+rows+block,device=device))
        set_position(0)
        context(auxiliary,context_metadata.lengths,context_metadata)
        output.copy_(attention(source,m))
        compiled_context = compile_forward(lambda: context(auxiliary,context_metadata.lengths,context_metadata))
        compiled_attention = compile_forward(lambda: output.copy_(attention(source,m)))
        compiled_context()
        compiled_attention()
        graphs = []
        for fn in (compiled_context,compiled_attention):
            graph = torch.cuda.CUDAGraph()
            with session.capture(),torch.cuda.graph(graph):
                fn()
            graphs.append(graph)
        session.freeze()
        w = draft.layers[0].attention
        def rotate(values,positions):
            angle = positions.float()[:,None]*attention.rotary.inverse[None]
            shape = (positions.numel(),)+(1,)*(values.ndim-2)+(32,)
            cos, sin = angle.cos().view(shape)*attention.rotary.scale, angle.sin().view(shape)*attention.rotary.scale
            pairs = values[..., -64:].float().reshape(*values.shape[:-1],32,2)
            tail = torch.stack((pairs[...,0]*cos-pairs[...,1]*sin,pairs[...,1]*cos+pairs[...,0]*sin),-1).flatten(-2).bfloat16()
            return torch.cat((values[...,:-64],tail),-1)
        for start in (0,1023,500001):
            set_position(start)
            source.add_(.00390625)
            graphs[0].replay()
            before = rings["draft0.latent"].clone()
            feedback = norm(torch.nn.functional.linear(auxiliary,bf16_weight(draft.projection)),draft.hidden_norm)
            projected_feedback = torch.nn.functional.linear(feedback,bf16_weight(draft.context_kv)).view(rows,5,576)[:,0]
            expected_feedback = rotate(torch.cat((norm(projected_feedback[:,:512],w["kv_norm"]),
                projected_feedback[:,512:]),-1),context_metadata.positions).float().clamp(-448,448).to(torch.float8_e4m3fn).float()
            indices = context_metadata.positions[-1024:].remainder(1024)
            error_feedback = (before[1,indices].float()-expected_feedback[-1024:]).norm()/expected_feedback[-1024:].norm()
            assert float(error_feedback)<.015
            graphs[1].replay()
            actual = output.clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(before.view(torch.uint8),rings["draft0.latent"].view(torch.uint8),rtol=0,atol=0)
            projected = torch.nn.functional.linear(source,bf16_weight(w["latent_gate"]))
            q = torch.nn.functional.linear(norm(projected[:,:1536],w["query_norm"]),bf16_weight(w["q_b"])).view(block,96,192)
            q = rotate(q,m.positions)
            kv = rotate(torch.cat((norm(projected[:,1536:2048],w["kv_norm"]),projected[:,2048:2112]),-1),m.positions)
            absorbed = torch.cat((torch.bmm(q[:,:,:128].transpose(0,1),w["key"]).transpose(0,1),q[:,:,128:]),-1)
            prior = before[1,torch.arange(start+rows-1024,start+rows,device=device).remainder(1024)].float()
            records = torch.cat((prior,kv.float().clamp(-448,448).to(torch.float8_e4m3fn).float()))
            query = absorbed.float().clamp(-448,448).to(torch.float8_e4m3fn).float()
            scores = torch.einsum("rhd,kd->rhk",query,records)*config.draft.attention_scale
            positions = torch.arange(1024+block,device=device)[None]
            end = 1024+torch.arange(block,device=device)[:,None]
            scores.masked_fill_(~((positions<=end)&(positions>end-1024))[:,None],-torch.inf)
            attended = torch.einsum("rhk,kd->rhd",scores.softmax(-1),records[:,:512]).bfloat16()
            expanded = torch.bmm(attended.transpose(0,1),w["value"]).transpose(0,1).flatten(1)
            expected = torch.nn.functional.linear(expanded,bf16_weight(w["output"]))
            error = ((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-10)).item()
            print(dict(quantized=quantized,start=start,relative_l2=error),flush=True)
            assert error < .015
        # No accepted rows means no persistent history write.
        before = [ring.clone() for ring in rings.values()]
        context_metadata.lengths.zero_()
        graphs[0].replay()
        torch.cuda.synchronize()
        for a,b in zip(before,rings.values(),strict=True):
            torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0)
        for graph in graphs:
            graph.reset()
