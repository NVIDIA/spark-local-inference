"""Vocabulary-shard summaries preserve every token and survival position."""

import os
from pathlib import Path
import subprocess
import sys

import torch


def test_vocabulary_completion_kernels_with_cpu_interpreter():
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
        env=dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES=""),
        text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def kernel_cases():
    from ignition.components.speculation import SpeculativeAcceptance, TARGET_RNG_DOMAIN
    from ignition.components import vocabulary_sampling as v

    # Triton's CPU interpreter cannot combine tensor wrappers with global constexpr wrappers.
    v._TARGET_DOMAIN = TARGET_RNG_DOMAIN
    for width in (1, 8, 13, 16, 32):
        batch, vocabulary, words = 2, 64, max(32, 4 * (1 << (width - 1).bit_length()))
        p = torch.zeros((batch, width, vocabulary))
        q = torch.zeros((batch, width - 1, vocabulary))
        for column in range(width):
            p[:, column, column + 1] = 1
            if column < width - 1:
                q[:, column, column + 1] = 1
        drafts = torch.arange(1, width).repeat(batch, 1)
        offset = torch.zeros(1, dtype=torch.int32)
        partials = torch.full((batch, width, 3, 1), -77.)
        selected = torch.full((batch, words), -88.)
        v._distribution_masses[(1, batch * width)](p, q, partials,
            LOCAL=vocabulary, WIDTH=width, BLOCKS=1, BLOCK=64)
        v._selected_records[(batch,)](offset, p, q, drafts, partials, selected,
            LOCAL=vocabulary, WIDTH=width, BLOCKS=1, PAD=1, WORDS=words)
        plane = words // 4
        assert selected[:, 2 * plane:2 * plane + width].eq(1).all()
        assert selected[:, :width - 1].eq(1).all()
        assert selected[:, plane:plane + width - 1].eq(1).all()
        assert selected[:, 3 * plane:3 * plane + width - 1].eq(1).all()

        residual = torch.zeros((batch, width, words))
        residual[:, :, 1] = 0
        residual[:, :, 2] = torch.arange(1, width + 1)
        slots = torch.tensor([3, 1], dtype=torch.int32)
        epochs = torch.full((batch,), 7, dtype=torch.int64)
        live = torch.tensor([1], dtype=torch.int32)
        generation = torch.full((4,), 7, dtype=torch.int64)
        status = torch.ones(4, dtype=torch.int32)
        seeds = torch.arange(4, dtype=torch.int64)
        rng_offset = torch.zeros(4, dtype=torch.int64)
        draft_valid = torch.ones((batch, width - 1), dtype=torch.bool)
        target_valid = torch.ones((batch, width), dtype=torch.bool)
        for limit in sorted({1, min(9, width), width}):
            limits = torch.full((batch,), limit, dtype=torch.int32)
            output = SpeculativeAcceptance.allocate(batch, width, "cpu")
            output.survival.fill_(-88.)
            v._complete_block[(batch,)](selected, residual, drafts, draft_valid, target_valid,
                limits, slots, epochs, live, generation, status, seeds, generation, rng_offset,
                output.token_ids, output.proposed_count, output.accepted_drafts,
                output.offered_drafts, output.valid, output.rng_increment, output.survival,
                WIDTH=width, WORLD=1, RANK_PAD=1, CAPACITY=4, BATCH=batch,
                VOCAB=vocabulary, RUNNING=1, RESERVATION=20, HAS_LIMITS=True, WORDS=words, SURVIVAL=width - 1,
                POSITIONS=1 << (width - 1).bit_length())
            assert output.valid.tolist() == [True, False]
            assert output.proposed_count.tolist() == [limit, 0]
            assert output.accepted_drafts.tolist() == [limit - 1, 0]
            assert output.token_ids[0, :limit].tolist() == list(range(1, limit + 1))
            assert output.token_ids[0, limit:].eq(-1).all()
            assert output.token_ids[1].eq(-1).all()
            assert output.rng_increment.tolist() == [20, 0]
            assert output.survival[0, :limit - 1].eq(1).all()
            assert output.survival[0, limit - 1:].eq(-1).all()
            assert output.survival[1].eq(-1).all()


if __name__ == "__main__":
    kernel_cases()
