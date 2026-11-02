# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Direct execution chooses decode rounds exactly as Helios does, on Helios's own recorded choices.

`helios_decode_rounds.txt` holds the rounds Helios's `RequestScheduler` chose for
each scenario below: `select_plan` and `candidates` in helios/src/scheduler/policy.rs
over the service prices of service.rs, the value rates and cohorts of utility.rs,
the expiring graph costs of cost.rs and the acceptance estimate of speculation.rs.
Every request decodes from the start, each round takes the scenario's graph time
(`duration`) and accepts the scenario's drafts (`accepted`), and the clock advances
by each graph time. The direct scheduler, given the same requests, times and
acceptance, must choose the same batch capacity, verification width and requests
in every round, including Helios's exploration of unmeasured graphs and of graphs
whose estimates expired.
"""

from pathlib import Path

import numpy as np
import pytest

from ignition.direct.scheduler import Layout, Request, SamplingParameters, Scheduler

FIXTURES = Path(__file__).with_name("helios_decode_rounds.txt")
WORD = (1 << 64) - 1

# cost: graph time base = fixed + per_row * batch * width + per_width * width + per_batch * batch
# (+ growth * round * batch * width / 16), varied by up to 2% each round; acceptance: per-mille
# probability that each successive draft of a slot's request is accepted; gap: 31 s idle before that round.
SCENARIOS = {
    "sixteen_cold": dict(active=16, widths=(1, 2, 3, 4, 5, 6), fixed=None, rounds=1000, seed=11,
                         acceptance=[450 + 30 * slot for slot in range(16)],
                         cost=(30_000_000, 150_000, 500_000, 100_000)),
    "eight_idle_gap": dict(active=8, widths=(1, 2, 3, 4, 5, 6), fixed=None, rounds=600, seed=23, gap=300,
                           acceptance=[900 if slot % 2 == 0 else 300 for slot in range(8)],
                           cost=(5_000_000, 400_000, 300_000, 50_000)),
    "five_fixed_width": dict(active=5, widths=(1, 2, 3, 4, 5, 6), fixed=4, rounds=400, seed=37,
                             acceptance=[800, 700, 600, 500, 400], cost=(6_000_000, 250_000, 400_000, 200_000)),
    "three_decode": dict(active=3, widths=(), fixed=None, rounds=200, seed=41, acceptance=[0, 0, 0],
                         cost=(4_000_000, 300_000, 0, 100_000)),
    "sixteen_row_bound": dict(active=16, widths=(1, 2, 3, 4, 5, 6), fixed=None, rounds=800, seed=53,
                              acceptance=[950 if slot < 6 else 150 for slot in range(16)],
                              cost=(1_000_000, 900_000, 100_000, 20_000)),
    "sixteen_growing_context": dict(active=16, widths=(1, 2, 3, 4, 5, 6), fixed=None, rounds=1300, seed=67,
                                    growth=300, acceptance=[600 + 20 * (slot % 8) for slot in range(16)],
                                    cost=(20_000_000, 120_000, 300_000, 80_000)),
}


def splitmix(value):
    value = (value + 0x9E3779B97F4A7C15) & WORD
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & WORD
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & WORD
    return value ^ (value >> 31)


def duration(scenario, batch, width, round_):
    fixed, per_row, per_width, per_batch = scenario["cost"]
    base = (fixed + per_row * batch * width + per_width * width + per_batch * batch
            + scenario.get("growth", 0) * round_ * batch * width // 16)
    noise = splitmix(scenario["seed"] ^ ((round_ << 20) & WORD)) % 41
    return base + base * noise // 1000 - base * 20 // 1000


def accepted(scenario, round_, slot, offered):
    count = 0
    while count < offered:
        draw = splitmix(scenario["seed"] ^ ((round_ << 20) & WORD) ^ (slot << 8) ^ (count + 1)) % 1000
        if draw >= scenario["acceptance"][slot]:
            break
        count += 1
    return count


def layout(scenario):
    active = scenario["active"]
    capacity = 1 << (active - 1).bit_length()
    batches = [batch for batch in (1, 2, 4, 8, 16) if batch <= capacity]
    graphs = {("prefill", 1, tokens): f"prefill{tokens}" for tokens in (64, 128, 256, 512, 1024, 2048, 4096)}
    for kind in ("init", "restore", "snapshot", "resume", "decode"):
        graphs.update({(kind, batch, 1): f"{kind}{batch}" for batch in batches})
    graphs.update({("speculative", batch, width): f"speculative{batch}x{width}"
                   for batch in batches for width in scenario["widths"]})
    return Layout(bindings={}, buffer_specs={}, page_views={}, capacity=capacity, max_active=active, context=131072, page_tokens=64, table_width=2048,
                  pages=1 << 16, snapshots=64, prefill_chunk=4096, vocabulary=1024, eos=(7,), graphs=graphs,
                  widths=scenario["widths"], fixed_width=scenario["fixed"], state_rows_per_request=1,
                  draft_pages=0, record_fields=32)


def rounds(scenario):
    """Run the scenario's rounds through the direct scheduler; returns (batch, width, slot mask) per round."""
    scheduler = Scheduler(layout(scenario), prefix_cache=False)
    requests = [Request(ident, list(range(64)), 100_000, SamplingParameters()) for ident in range(scenario["active"])]
    for current in requests:
        scheduler.submit(current)
    scheduler._admit()
    for current in requests:
        current.phase, current.consumed, current.generated = "decode", 64, [99]
    chosen = []
    for round_ in range(1, scenario["rounds"] + 1):
        if scenario.get("gap") == round_:
            scheduler.advance_to(scheduler.now_ns + 31_000_000_000)
        step = scheduler.plan()
        assert step.sequence == round_ and step.kind == ("speculative" if scenario["widths"] else "decode")
        nanos = duration(scenario, step.batch, step.width, round_)
        scheduler.advance_to(scheduler.now_ns + nanos)
        records = np.zeros((scheduler.layout.capacity, 32), dtype=np.int64)
        records[:, 4] = -1
        records[:, 16:] = -1
        mask = 0
        for lane in step.lanes:
            current = lane.request
            mask |= 1 << current.slot
            count = 1 + (accepted(scenario, round_, current.slot, lane.tokens - 1) if step.kind == "speculative" else 0)
            row = records[current.slot]
            row[:8] = (current.generation, step.sequence, current.consumed + count, len(current.generated) + count,
                       99, 1, 0, 0)
            row[8:11] = (count, lane.tokens - 1, count - 1)
            row[16:16 + count] = 99
        scheduler.complete(step, records, nanos)
        chosen.append((step.batch, step.width, mask))
    return chosen


def encoded(chosen):
    runs = []
    for shape in chosen:
        if runs and runs[-1][0] == shape:
            runs[-1][1] += 1
        else:
            runs.append([shape, 1])
    return " ".join(f"{batch}x{width}:{mask:x}*{count}" for (batch, width, mask), count in runs)


def helios():
    lines = [line for line in FIXTURES.read_text().splitlines() if line and not line.startswith("#")]
    return dict(line.split(" ", 1) for line in lines)


@pytest.mark.parametrize("name", SCENARIOS)
def test_decode_rounds_match_helios(name):
    expected = helios()[name].split()
    actual = encoded(rounds(SCENARIOS[name])).split()
    difference = next((index for index, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
    assert actual == expected, f"first differing run {difference}: direct {actual[difference:difference + 3]}" \
                               f" Helios {expected[difference:difference + 3]}"
