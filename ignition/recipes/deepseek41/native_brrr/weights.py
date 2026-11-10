# SPDX-FileCopyrightText: Copyright (c) 2026 Fraser Price
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation-only weight and peer-buffer adaptation of BRRR's TP4 decoder.

Weight packing follows Fraser Price's BRRR decoder.py. Checkpoint values and
scales are rearranged, not requantized. The original license/notice is adjacent.
"""
import torch
from b12x.moe.fused_moe._impl import _PreparedW4A8Weights
from ignition.preparation.tensor_metadata import record_numeric_tensor

from .extension import extension


def stages(weight):
    weight = weight.contiguous().view(torch.uint8)
    weight = torch.nn.functional.pad(weight, (0, -weight.shape[1] % 128))
    return weight.reshape(weight.shape[0] // 64, 4, 2, 8, -1, 4, 4, 8).permute(0, 4, 5, 1, 3, 6, 2, 7).contiguous().view(-1)


def interleave(weight, block):
    return weight.reshape(2, -1, block, *weight.shape[1:]).transpose(0, 1).reshape(weight.shape).contiguous()


def block_scales(packed, rows, columns):
    scales = packed.scale_rows.view(torch.uint8).reshape(rows, columns // 32)
    blocks = scales[::32].contiguous()
    if not torch.equal(scales, blocks.repeat_interleave(32, dim=0)):
        raise ValueError('Dense checkpoint scales are not shared across 32 rows')
    return blocks


class DecoderAdapter:
    def __init__(self, runtime, rank, ext=None):
        self.runtime = runtime
        channel = runtime.for_stream()
        assert runtime.world_size == 4 and channel._ipc is not None
        self.ext = extension() if ext is None else ext
        self.anchor = torch.empty(0, dtype=torch.uint8, device=channel.device)
        shared = channel._allocate_shared_buffer(channel.exchange_group,
            self.ext.flag_bytes() + self.ext.inbox_bytes(), zero_fill=True, ipc=channel._ipc)
        channel._owned_buffers.append(shared)
        self.shared = shared
        self.handle = self.ext.create(list(shared.peer_ptrs), rank)
        self.layers = {}
        self.owners = []

    def register(self, index, layer):
        if index in self.layers:
            return self.layers[index]
        output = layer.attention['output']
        assert (output.groups, output.group_width, output.rank, output.hidden) == (2, 4096, 1024, 5120)
        assert output.wo_a.values.shape == (1024, 4096, 2)
        rep = layer.experts.experts._impl.representation_for('w4a8_mx')
        assert isinstance(rep, _PreparedW4A8Weights) and rep.n64_repack and rep.intermediate_size == 576
        assert layer.experts.experts.num_experts == 384
        shared = layer.experts.shared
        up, down = shared.gate_up.weight, shared.down.weight
        residual = layer.residual
        planes = dict(
            woa=stages(output.wo_a.values.permute(2, 0, 1).reshape(2048, 4096)),
            woa_s=block_scales(output.wo_a, 2048, 4096),
            wob=stages(output.wo_b.values.reshape(5120, 2048)),
            wob_s=block_scales(output.wo_b, 5120, 2048),
            gate=stages(layer.experts.router), gate_b=layer.experts.correction.float().contiguous(),
            bias_vl=None,
            sh13=stages(interleave(up.values.reshape(1152, 5120), 32)),
            sh13_s=interleave(block_scales(up, 1152, 5120), 1),
            sh2=stages(down.values.reshape(5120, 640)),
            sh2_s=block_scales(down, 5120, 640),
            w13=rep.w13_rp, s13=rep.w13_sfb, w2=rep.w2_rp, s2=rep.w2_sfb,
            fn=residual['hc_ffn_fn'], hc_scale=residual['hc_ffn_scale'],
            hc_base=residual['hc_ffn_base'], norm=residual['ffn_norm.weight'], image_lo=0)
        for tensor in planes.values():
            if isinstance(tensor, torch.Tensor):
                record_numeric_tensor(tensor)
                self.owners.append(tensor)
        handle = self.ext.add_layer(self.handle, **planes)
        self.layers[index] = handle
        return handle

    def close(self):
        if self.handle is not None:
            self.ext.destroy_engine(self.handle, self.anchor)
            self.handle = None
            self.owners.clear()
            self.layers.clear()
