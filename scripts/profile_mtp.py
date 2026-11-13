# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure batch-one MTP graph replay and annotate one round for Nsight Compute.

CUDA events exclude input uploads and completion readback. Each trial starts
with the same 512-token prompt and advances real target/draft state until EOS,
the output budget, or the round limit. Width one retains MTP absorption.
"""

import argparse
from contextlib import nullcontext
from dataclasses import fields
import json
from pathlib import Path
import statistics
import sys
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from ignition.components.sampling import AdmissionBuffers
from ignition.recipes.qwen38.bootstrap import PreparationConfig, prepare_model
from ignition.recipes.qwen38.sampling import RequestAdmission


class Replay:
    def __init__(self, model, tokens):
        self.model, self.tokens = model, tokens
        self.host = {name: tensor.detach().cpu().clone() for name, tensor in model.buffers.items()}
        self.admission = AdmissionBuffers(**{field.name: self.host[field.name]
                                             for field in fields(AdmissionBuffers)})
        self.stream = torch.cuda.Stream(device=model.device)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.generation = self.sequence = 0

    def rows(self):
        h = self.host
        h['row_slot_ids'].fill_(-1)
        h['row_generations'].zero_()
        h['row_slot_ids'][0] = 0
        h['row_generations'][0] = self.generation
        h['live_rows'].fill_(1)
        h['query_lengths'].zero_()
        h['sample_enabled'].fill_(1)
        h['vision_token_indices'].fill_(-1)

    def execute(self, name, *, timed=False, profile=False):
        if self.model.signatures[name].kind == "speculative":
            self.host['query_lengths'][0] = self.model.signatures[name].token_capacity
        self.sequence += 1
        for field in self.model.signatures[name].inputs:
            self.model.buffers[field].copy_(self.host[field])
        self.model.buffers['frame_index'].zero_()
        self.model.buffers['submission_sequence'].fill_(self.sequence)
        self.stream.wait_stream(torch.cuda.current_stream(self.model.device))
        with torch.cuda.stream(self.stream):
            scope = torch.cuda.nvtx.range('mtp.' + name) if profile else nullcontext()
            with scope:
                if timed:
                    self.start.record()
                self.model.graphs[name].replay()
                if timed:
                    self.end.record()
        self.stream.synchronize()
        record = self.model.buffers['completion_records'][0, 0].cpu().tolist()
        if record[:2] != [self.generation, self.sequence] or record[6:8] != [0, 0]:
            raise RuntimeError(f'{name}: invalid completion {record}')
        return (self.start.elapsed_time(self.end) if timed else None), record

    def fresh(self, seed, temperature):
        self.generation += 1
        h = self.host
        h['admission_count'].fill_(1)
        h['admission_slots'].fill_(-1)
        self.admission.stage(0, 0, RequestAdmission(self.generation, seed,
            max_new_tokens=512, temperature=temperature, top_k=20, top_p=.95), self.tokens)
        for key in ('page_tables', 'draft_page_tables'):
            h[key].fill_(-1)
            h[key][0, :17] = torch.arange(17, dtype=h[key].dtype)
        h['state_num_ops'].fill_(1)
        for key in ('source_slots', 'destination_slots', 'operation', 'accepted_tokens'):
            h['state_' + key].zero_()
            h['state_' + key][0] = 1
        h['state_qsa_slots'].fill_(-1)
        h['state_qsa_slots'][0] = 1
        h['state_expected_consumed'].fill_(-1)
        h['state_expected_consumed'][0] = 0
        self.rows()
        self.execute('init1')
        h['query_lengths'][0] = len(self.tokens)
        h['token_ids'].fill_(248044)
        h['token_ids'][:len(self.tokens)] = torch.tensor(self.tokens)
        h['prefill_positions'].zero_()
        h['prefill_positions'][:len(self.tokens)] = torch.arange(len(self.tokens))[:, None]
        _, record = self.execute('prefill512')
        if record[2:4] != [512, 1] or record[5] != 1:
            raise RuntimeError(f'prefill did not establish an active request: {record}')
        self.rows()
        return record


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--widths', nargs='+', type=int, choices=range(1, 9), default=[1, 2, 4, 8])
    parser.add_argument('--rounds', type=int, default=64)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=.7)
    parser.add_argument('--ncu-width', type=int, choices=range(1, 9))
    args = parser.parse_args()
    if args.rounds < 9 or args.trials < 1 or args.temperature <= 0:
        parser.error('require at least 9 rounds, one trial, and positive temperature')
    if args.ncu_width is not None and args.ncu_width not in args.widths:
        parser.error('--ncu-width must occur in --widths')
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    text = 'Explain how an operating system schedules processes and manages virtual memory. '
    tokens = tokenizer.encode(text * 800, add_special_tokens=False)[:512]
    if len(tokens) != 512:
        raise ValueError('profiling requires a 512-token prompt')
    started = monotonic()
    progress = lambda message: print(f'{monotonic() - started:.1f}s {message}', flush=True)
    with prepare_model(args.snapshot, PreparationConfig(mtp=True, speculative_widths=tuple(range(1, 9)), speculative_graph_budget=24),
                       progress=progress) as model:
        replay = Replay(model, tokens)
        selected = False
        for trial in range(args.trials):
            widths = args.widths if trial % 2 == 0 else args.widths[::-1]
            for width in widths:
                record = replay.fresh(17 + trial, args.temperature)
                milliseconds, emitted, offered, accepted, positions = [], [], [], [], []
                output_tokens = [record[4]]
                name = f'speculative1x{width}'
                for step in range(args.rounds):
                    position = record[2]
                    profile = args.ncu_width == width and trial == 0 and step == 8
                    elapsed, record = replay.execute(name, timed=True, profile=profile)
                    selected |= profile
                    if record[5] not in (1, 2) or not 1 <= record[8] <= width:
                        raise RuntimeError(f'invalid speculative progress: {record}')
                    milliseconds.append(elapsed)
                    emitted.append(record[8])
                    offered.append(record[9])
                    accepted.append(record[10])
                    positions.append(position)
                    output_tokens.extend(record[16:16 + record[8]])
                    if record[5] == 2:
                        break
                print(json.dumps(dict(width=width, trial=trial, seed=17 + trial,
                    temperature=args.temperature, top_k=20, top_p=.95,
                    rounds=len(milliseconds), gpu_ms=milliseconds, positions=positions,
                    emitted=emitted, offered=offered, accepted=accepted,
                    median_round_ms=statistics.median(milliseconds),
                    gpu_ms_per_token=sum(milliseconds) / sum(emitted),
                    tokens_per_round=statistics.mean(emitted),
                    acceptance=sum(accepted) / sum(offered) if sum(offered) else None,
                    output_tokens=output_tokens, output_text=tokenizer.decode(output_tokens),
                    final_status=record[5])), flush=True)
        if args.ncu_width is not None and not selected:
            raise RuntimeError('request finished before the selected profiling round')


if __name__ == '__main__':
    main()
