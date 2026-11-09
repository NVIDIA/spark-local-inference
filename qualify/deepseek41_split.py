# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare combined and split speculative decoding for DeepSeek V4.1 Flash.

Research-only: the Python driver chooses diagnostic steps. GPU timing and driver
wall timing are recorded separately; neither is production HTTP throughput.
Paired decode checks restore the same saved prefix before each execution so
floating-point differences from separate prefill runs do not change the inputs.

Build Heliopause with its qualification feature before running:
python -m qualify.deepseek41_split --gpus 0,1,2,3 --split-speculation --output /tmp/split.json
"""

import argparse
import importlib
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import struct
import signal
import time
from types import SimpleNamespace

if __name__ == '__main__':
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply

    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument('--gpus', required=True)
    selected, _ = selection.parse_known_args()
    apply(Configuration.load(None), [SimpleNamespace(uuid=gpu) for gpu in selected.gpus.split(',')])

import numpy as np
import torch

from ignition.cli.config import Configuration
from ignition.components.split_draft import causal_prefix, lagged_budget
from ignition.direct.scheduler import Lane, Layout, Request, SamplingParameters, Scheduler, Step
from ignition.preparation.core_storage import CoreStorage
from ignition.preparation.rank_group import prepare_group
from ignition.recipes.deepseek41.bootstrap import PreparationConfig, prepare_rank
from ignition.recipes.deepseek41.recipe import RECIPE


PROMPTS = {
    'essay': 'Explain how a transformer language model generates text, including attention, tokenization, and sampling. Give concrete examples and discuss limitations.',
    'code': 'Write a Python implementation of a thread-safe bounded LRU cache with get, put, delete and clear. Include type annotations, tests, and an explanation of the design.',
    'math': 'A train leaves city A at 9:00 at 80 km/h and another leaves city B, 500 km away, at 10:00 at 120 km/h toward A. When and where do they meet? Explain step by step, then verify the answer.',
}


class Driver:
    def __init__(self, group):
        self.group, self.model = group, group.model
        self.layout = Layout.of(SimpleNamespace(server_config=self.model.server_config,
            buffers=self.model.buffers, max_active=self.model.max_active,
            signatures={n: s for n, s in self.model.signatures.items() if not n.startswith('combined')}))
        self.indices = {g['name']: i for i, g in enumerate(self.model.manifest()['graphs'])}
        self.executor = group.start_executor()
        self.generation = 0
        self.last_fence = 0

    def execute(self, graph, uploads):
        start = time.perf_counter_ns()
        fence = self.executor.submit(self.indices[graph], uploads)
        assert fence is not None
        while (result := self.executor.poll()) is None:
            time.sleep(.00001)
        assert result.fence == fence and result.graph_index == self.indices[graph]
        self.last_fence = fence
        ranks = list(result.gpu_ns)
        active = [value for value in ranks if value is not None]
        if len(active) != 4 or any(value <= 0 for value in active):
            raise AssertionError(f'expected four positive GPU intervals, got {ranks}')
        completion = SimpleNamespace(fence=result.fence, graph_index=result.graph_index,
            records=result.records, gpu_ns=max(active), rank_gpu_ns=ranks,
            wall_ns=result.wall_ns, queued=result.queued)
        return completion, time.perf_counter_ns() - start

    def snapshot(self, pages):
        o = self.model.proposal_state.owner
        values = {f'request.{name}': value.detach().cpu().clone()
                  for name, value in vars(o.requests).items() if isinstance(value, torch.Tensor)
                  and name != 'generation'}
        values['continuation'] = o.pool.backing[self.layout.state_row(0)].cpu().clone()
        for name, value in o.pool.pages.items():
            values['page.' + name] = value[:pages].cpu().clone()
        values['draft_rng'] = o.offset.cpu().clone()
        values['target_rng'] = o.rng.target_offset.cpu().clone()
        return values

    def initialize(self, tokens, seed, output_tokens, *, reuse_prefix=False):
        if reuse_prefix:
            if not hasattr(self, 'reuse_scheduler'):
                self.reuse_scheduler = Scheduler(self.layout, prefix_cache=True)
            scheduler = self.reuse_scheduler
            assert scheduler.idle
        else:
            scheduler = Scheduler(self.layout, prefix_cache=False)
        scheduler.advance_to(time.monotonic_ns())
        scheduler.sequence = self.last_fence
        scheduler.generations[0] = self.generation
        request = Request(1, tokens, output_tokens,
            SamplingParameters(seed=seed, temperature=1., top_p=.95, top_k=0), ignore_eos=True)
        scheduler.submit(request)
        while request.phase != 'decode':
            scheduler.advance_to(time.monotonic_ns())
            step = scheduler.plan()
            assert step is not None
            completion, _ = self.execute(step.graph, scheduler.pack(step))
            step.sequence = scheduler.sequence = completion.fence
            scheduler.advance_to(time.monotonic_ns())
            scheduler.complete(step, completion.records, completion.gpu_ns)
            if request.error is not None:
                raise AssertionError(f'initialization failed: {request.error}')
        self.generation = request.generation
        return scheduler, request

    def step(self, scheduler, request, width):
        graph = self.layout.graphs['speculative', 1, width]
        return Step('speculative', graph, 1, width,
            [Lane(request, request.consumed, scheduler._round_tokens(request, width))],
            self.last_fence + 1)

    def generate(self, tokens, seed, count, *, mode, width=8, threshold=.25, costs=None, capture_state=False,
                 debug=False, full_uploads=False, reuse_prefix=False, widths=None, poison=False):
        if mode not in ('combined', 'split', 'lagged', 'causal'):
            raise ValueError('unknown split-speculation diagnostic mode')
        if type(count) is not int or count < 2 or type(width) is not int or not 1 <= width <= 8:
            raise ValueError('diagnostics require at least two output tokens and verification width 1 through 8')
        if widths is not None and (not widths or mode not in ('combined', 'split')
                or any(type(w) is not int or not 1 <= w <= 8 for w in widths)):
            raise ValueError('fixed width sequences require combined or split execution and widths 1 through 8')
        if poison and mode != 'split':
            raise ValueError('scratch poisoning requires a completed proposal before split verification')
        scheduler, request = self.initialize(tokens, seed, count, reuse_prefix=reuse_prefix)
        rows, stale, elapsed, gpu_total = [], [1.] * 7, 0, 0
        debug_states = []
        if debug:
            debug_states.append(self.debug_state(1, include_cache=True))
        allocated = torch.cuda.memory_stats()['allocation.all.allocated']
        decode_utc_start_ns, decode_started = time.time_ns(), time.perf_counter_ns()
        while not request.done:
            started = time.perf_counter_ns()
            scheduler.advance_to(time.monotonic_ns())
            if widths is not None:
                width = widths[len(rows) % len(widths)]
            proposal_ns = proposal_wall = 0
            confidence = None
            if mode != 'combined':
                selected = lagged_budget(stale, costs) if mode == 'lagged' else width
                proposal_step = self.step(scheduler, request, 8)
                proposal, proposal_wall = self.execute('proposal1x8', scheduler.pack(proposal_step))
                proposal_ns = proposal.gpu_ns
                assert proposal.records[0][8] == 0 and proposal.records[0][5] == 1
                confidence = self.model.proposal_confidence[0].tolist()
                if poison:
                    self.model.proposal_state.owner.workspace.tensor.fill_(255)
                    forwards = next(r for r in self.model.resources
                                    if isinstance(r, dict) and 'verification1x8' in r)
                    for workspace in forwards['verification1x8'].target.sampling_scratch.values():
                        for tensor in workspace.values():
                            tensor.fill_(True if tensor.dtype == torch.bool else 255)
                    torch.cuda.synchronize()
                if mode == 'causal':
                    selected = causal_prefix(confidence, threshold) + 1
                width_used = selected
            else:
                width_used = width
            step = self.step(scheduler, request, width_used)
            if step is None or step.kind != 'speculative':
                raise AssertionError('a live decode request requires a speculative step')
            width_used = step.width
            name = f'combined1x{width_used}' if mode == 'combined' else f'verification1x{width_used}'
            uploads = (scheduler.pack(step) if full_uploads or mode == 'combined' else
                       [('query_lengths', 0, struct.pack('<i', step.lanes[0].tokens))])
            completion, verification_wall = self.execute(name, uploads)
            step.sequence = scheduler.sequence = completion.fence
            gpu = proposal_ns + completion.gpu_ns
            scheduler.advance_to(time.monotonic_ns())
            scheduler.complete(step, completion.records, gpu)
            if confidence is not None and request.error is None:
                stale = confidence
            duration = time.perf_counter_ns() - started
            elapsed += duration
            gpu_total += gpu
            record = completion.records[0]
            rows.append(dict(width=width_used, limit=step.lanes[0].tokens, gpu_ns=gpu,
                proposal_gpu_ns=proposal_ns, verify_gpu_ns=completion.gpu_ns, wall_ns=duration,
                proposal_wall_ns=proposal_wall, verify_wall_ns=verification_wall,
                verify_rank_gpu_ns=completion.rank_gpu_ns, verify_native_wall_ns=completion.wall_ns,
                proposal_rank_gpu_ns=None if mode == 'combined' else proposal.rank_gpu_ns,
                proposal_native_wall_ns=0 if mode == 'combined' else proposal.wall_ns,
                emitted=record[8], offered=record[9], accepted=record[10], confidence=confidence))
            if request.error is not None:
                raise AssertionError(f'{mode} width {width_used}: device error {request.error}')
            if debug:
                debug_states.append(self.debug_state(width_used))
        decode_elapsed_ns = time.perf_counter_ns() - decode_started
        assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
        snapshot = self.snapshot((len(tokens) + count + 127) // 128) if capture_state else None
        if snapshot is not None:
            snapshot['confidence'] = self.model.proposal_confidence.clone()
        self.debug_states = debug_states
        return dict(mode=mode, width=width, threshold=threshold, seed=seed, tokens=request.generated,
            prompt_token_ids=list(tokens),
            ignore_eos=True, policy_clock='monotonic wall time', snapshot_scope='rank-zero committed state; request generation excluded',
            debug=debug, full_uploads=full_uploads, reuse_prefix=reuse_prefix, widths=widths, poison=poison,
            reused_prompt_tokens=request.reused,
            decode_tokens=sum(r['emitted'] for r in rows), gpu_ns=gpu_total, wall_ns=elapsed,
            decode_utc_start_ns=decode_utc_start_ns, decode_elapsed_ns=decode_elapsed_ns,
            decode_elapsed_tokens_s=sum(r['emitted'] for r in rows) / (decode_elapsed_ns / 1e9),
            gpu_tokens_s=sum(r['emitted'] for r in rows) / (gpu_total / 1e9),
            driver_tokens_s=sum(r['emitted'] for r in rows) / (elapsed / 1e9), rounds=rows), snapshot

    def debug_state(self, width, *, include_cache=False):
        forwards = next(r for r in self.model.resources if isinstance(r, dict) and 'speculative1x1' in r)
        target = forwards[f'verification1x{width}'].target
        o = target.owner
        probability, _, verification, _ = target.operations
        values = dict(pending=o.requests.pending_token, consumed=o.requests.consumed_tokens,
            seed=o.requests.seed, counter=o.requests.rng_counter, target_offset=o.rng.target_offset,
            logits=target.decoder.scratch.logits, probabilities=probability.probabilities,
            verifier_seed=verification.seed_bits, verifier_offset=verification.offset,
            source_base=self.model.proposal_state.base_offset,
            candidate=target.candidates, target_hidden=target.decoder.scratch.hidden,
            proposal_confidence=self.model.proposal_confidence)
        if include_cache:
            values['initial_continuation'] = o.pool.backing[self.layout.state_row(0)]
            for name, value in o.pool.pages.items():
                values['initial_page.' + name] = value[:1]
        return {name: value.detach().cpu().clone() for name, value in values.items()}

    def rejection_checks(self, tokens):
        scheduler, request = self.initialize(tokens, 415991, 32)
        step = self.step(scheduler, request, 8)
        pages = (len(tokens) + 32 + 127) // 128
        before = self.snapshot(pages)
        self.execute('proposal1x8', scheduler.pack(step))
        proposed = self.snapshot(pages)
        assert proposed['draft_rng'][0] == 28
        before['draft_rng'] = proposed['draft_rng']
        assert_same(before, proposed, 'proposal only advances draft randomness')
        uploads = scheduler.pack(step)
        uploads = [(name, offset, np.array([0], dtype=np.int32).tobytes() if name == 'live_rows' else data)
                   for name, offset, data in uploads]
        completion, _ = self.execute('verification1x8', uploads)
        assert completion.records[0][5] == 0
        assert_same(before, self.snapshot(pages), 'inactive verification')

        self.execute('proposal1x8', scheduler.pack(step))
        replacement, fresh = self.initialize(tokens, 415992, 32)
        before = self.snapshot(pages)
        invalid = self.step(replacement, fresh, 8)
        completion, _ = self.execute('verification1x8', replacement.pack(invalid))
        assert completion.records[0][5:7] == [3, 256]
        assert_same(before, self.snapshot(pages), 'generation reuse')

        # A retained proposal from another generation remains invalid on repeated verification.
        completion, _ = self.execute('verification1x8', replacement.pack(invalid))
        assert completion.records[0][5:7] == [3, 256]
        assert_same(before, self.snapshot(pages), 'duplicate verification')

        # A terminal request must not consume a previously retained proposal.
        terminal, fresh = self.initialize(tokens, 415993, 2)
        step = self.step(terminal, fresh, 8)
        self.execute('proposal1x8', terminal.pack(step))
        completion, _ = self.execute('speculative1x1', terminal.pack(self.step(terminal, fresh, 1)))
        assert completion.records[0][5] == 2
        before = self.snapshot(pages)
        completion, _ = self.execute('verification1x8', terminal.pack(step))
        assert completion.records[0][5:7] == [3, 256]
        assert_same(before, self.snapshot(pages), 'terminal request')
        return ['proposal only advances draft randomness', 'inactive verification leaves state unchanged',
                'generation reuse rejects without changing replacement', 'duplicate verification rejects',
                'terminal request rejects without state or RNG mutation']


def assert_same(left, right, label):
    different = [name for name in left if not torch.equal(left[name], right[name])]
    if different:
        raise AssertionError(f'{label}: state differs in {different}')


def read_io(pids):
    return {str(pid): {k: int(v) for k, v in (line.split(':', 1)
            for line in Path(f'/proc/{pid}/io').read_text().splitlines())} for pid in pids}


def control_loop(driver, prompts, commands, output):
    """Run JSON-line diagnostic commands while retaining weights, tables and graphs.

    Only driver methods reload between commands; prepared GPU programs remain
    immutable. A stop command or SIGTERM drains native workers before cleanup.
    """
    jobs, cursor = {}, 0
    output.mkdir(parents=True, exist_ok=True)
    print('Native diagnostic control ready', flush=True)
    while True:
        lines = commands.read_text().splitlines()
        if cursor >= len(lines):
            time.sleep(.1)
            continue
        command = json.loads(lines[cursor])
        cursor += 1
        action, name = command['action'], command.get('name', f'command-{cursor}')
        if action == 'stop':
            return
        module = importlib.reload(importlib.import_module('qualify.deepseek41_split'))
        driver.__class__ = module.Driver
        try:
            if action == 'generate':
                arguments = {k: v for k, v in command.items() if k not in ('action', 'name', 'workload')}
                case, state = driver.generate(prompts[command.get('workload', 'math')],
                    capture_state=True, **arguments)
                jobs[name] = (case, state, driver.debug_states)
                report = case
                torch.save(dict(state=state, steps=driver.debug_states), output / (name + '.pt'))
            elif action == 'compare':
                left, right = (jobs[command[k]] for k in ('left', 'right'))
                report = dict(first_token_difference=next((i for i, (a, b) in enumerate(
                    zip(left[0]['tokens'], right[0]['tokens'])) if a != b), None),
                    state_differences=[k for k in left[1] if not torch.equal(left[1][k], right[1][k])],
                    step_differences=[dict(step=i, fields=[k for k in a if not torch.equal(a[k], b[k])])
                                      for i, (a, b) in enumerate(zip(left[2], right[2]))])
            elif action == 'rejection':
                report = driver.rejection_checks(prompts['math'])
            else:
                raise ValueError(f'unknown diagnostic command {action!r}')
        except Exception as error:
            import traceback
            traceback.print_exc()
            report = dict(error=f'{type(error).__name__}: {error}')
        (output / (name + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(command=name, finished=True)), flush=True)


@torch.no_grad()
def main(args):
    configuration = Configuration.load(None)
    checkpoint = RECIPE.checkpoints[0]
    snapshot = configuration.directory('hub_cache') / ('models--' + checkpoint.repository.replace('/', '--')) / 'snapshots' / checkpoint.revision
    config = PreparationConfig(max_active=1, speculative_widths=tuple(range(1, 9)),
        split_speculation=args.split_speculation, speculation_reference_graphs=True,
        kernel_cache=configuration.directory('kernel_cache'),
        snapshot_budget_bytes=8 * RECIPE.ranks[4].state_row_bytes, headroom_bytes=1 << 30,
        retain_graphs=True, compile_decoder=True, prefill_chunk_size=4096)
    print(json.dumps(dict(preparation=asdict(config), sampling=dict(temperature=1., top_p=.95, top_k=0),
                          embedding_table_storage='ram'), default=str), flush=True)
    from tokenizers import Tokenizer
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    tokenizer = Tokenizer.from_file(str(inspect_indexed_checkpoint(snapshot).metadata_path / 'tokenizer.json'))
    from ignition.recipes.deepseek41 import bootstrap
    template = Path(bootstrap.__file__).with_name('assets').joinpath('chat_template.jinja').read_text()
    from transformers.utils.chat_template_utils import _compile_jinja_template
    render = _compile_jinja_template(template)
    prompts = {}
    requests = {name: dict(messages=[dict(role='user', content=prompt)],
                chat_template_kwargs=dict(thinking=True, enable_thinking=True,
                                          thinking_mode='thinking', reasoning_effort='high'))
                for name, prompt in PROMPTS.items()}
    if args.requests is not None:
        requests = json.loads(args.requests.read_text())
    for workload, request in requests.items():
        text = render.render(messages=request['messages'], add_generation_prompt=True,
                             **request.get('chat_template_kwargs', {}))
        prompts[workload] = tokenizer.encode(text, add_special_tokens=False).ids
    torch.cuda.set_device(0)
    storage = CoreStorage(world=4)
    with storage:
        group = prepare_group(snapshot, config, prepare_rank, core_storage=storage,
                              progress=lambda message: print(message, flush=True))
    result = dict(preparation=asdict(config), gpus=args.gpus, requests=requests, qualification=[], cases=[],
        scope='research-only diagnostic native GraphExecutor; not HTTP serving')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, default=str, indent=2) + '\n')
    try:
        with group:
            driver = Driver(group)
            if args.control is not None:
                control_loop(driver, prompts, args.control, args.output)
                return
            result['qualification'] = driver.rejection_checks(prompts['math'])
            save()
            # A shared saved prefix gives both executions the same initial floating-point state.
            driver.generate(prompts['math'], 1085999, 16, mode='combined', width=1, reuse_prefix=True)
            costs = [None] * 8
            for width in range(1, 9):
                mono, mono_state = driver.generate(prompts['math'], 1086000, args.qualification_tokens,
                    mode='combined', width=width, capture_state=True, reuse_prefix=True)
                split, split_state = driver.generate(prompts['math'], 1086000, args.qualification_tokens,
                    mode='split', width=width, capture_state=True, reuse_prefix=True)
                result['cases'].extend([dict(workload='math', **mono), dict(workload='math', **split)])
                save()
                assert mono['tokens'] == split['tokens'], f'fixed width {width}: generated tokens differ'
                assert_same(mono_state, split_state, f'fixed width {width}')
                costs[width - 1] = statistics.median(r['gpu_ns'] for r in split['rounds'])
                result['qualification'].append(f'width {width}: exact tokens, committed state and RNG')
                save()
                print(json.dumps(dict(width=width, combined=mono['gpu_tokens_s'], split=split['gpu_tokens_s'], exact=True)), flush=True)
            widths = [1, 8, 2, 7, 4, 1, 5, 3, 6, 8]
            reference, reference_state = driver.generate(prompts['math'], 1087000, 256,
                mode='combined', widths=widths, capture_state=True, reuse_prefix=True)
            for poison in (False, True):
                candidate, candidate_state = driver.generate(prompts['math'], 1087000, 256,
                    mode='split', widths=widths, poison=poison, capture_state=True, reuse_prefix=True)
                assert reference['tokens'] == candidate['tokens'], 'alternating verification widths changed tokens'
                assert_same(reference_state, candidate_state, f'alternating widths, poison={poison}')
                result['cases'].append(dict(workload='math', **candidate))
                result['qualification'].append(f'alternating widths with scratch poisoning={poison}: exact tokens, state and RNG')
                save()
            result['qualification'].extend(driver.rejection_checks(prompts['math']))
            save()
            if args.qualify_only:
                return
            result['costs_ns'] = costs
            result['combined_warmup'] = {}
            for workload, tokens in prompts.items():
                warmup, _ = driver.generate(tokens, 1085999, 128, mode='combined', reuse_prefix=True)
                result['combined_warmup'][workload] = warmup
            import os
            pids = [os.getpid(), *(p.pid for p in group.processes)]
            before_io = read_io(pids)
            for seed in args.seeds:
                for workload, tokens in prompts.items():
                    for mode, threshold in [('combined', .25), ('lagged', .25), ('causal', .15), ('causal', .25), ('causal', .4)]:
                        case, _ = driver.generate(tokens, seed, args.tokens, mode=mode, width=8,
                            threshold=threshold, costs=costs, reuse_prefix=True)
                        result['cases'].append(dict(workload=workload, **case))
                        print(json.dumps(dict(workload=workload, mode=mode, threshold=threshold,
                            gpu_tokens_s=case['gpu_tokens_s'], driver_tokens_s=case['driver_tokens_s'])), flush=True)
            after_io = read_io(pids)
            result['serving_io_delta'] = {pid: {k: after_io[pid][k] - before_io[pid][k] for k in after_io[pid]}
                                          for pid in before_io}
            save()
    finally:
        for region in storage.host_regions:
            region.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-speculation', action='store_true', required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--requests', type=Path, help='JSON object mapping workload names to chat request bodies')
    parser.add_argument('--control', type=Path, help='JSON-line commands; retain the native model between diagnostic runs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualification-tokens', type=int, default=128)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1086000, 1091000])
    parser.add_argument('--qualify-only', action='store_true')
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    main(parser.parse_args())
