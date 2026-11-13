# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact prompt endpoints and generated terminal-prefix HTTP qualification."""

from __future__ import annotations

import copy

from qualify.server import (
    common_prefix,
    metric_delta,
    prefill_launches,
    system_prefix_ids,
    message_prefixes,
    template_input,
)


PROMPT_LENGTHS = (128, 129, 130, 131, 191)


def prompt_ids(tokenizer, request):
    messages, options = template_input(request)
    return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options)


def endpoint_request(tokenizer, model_id, nonce, label, length, *, max_tokens=1):
    for repeats in range(length + 64):
        content = (f"Exact checkpoint {nonce} {label}. Return one exclamation mark.\nContext:"
                   + " apple" * repeats + "\nAnswer:")
        request = {
            "model": model_id, "messages": [{"role": "user", "content": content}],
            "temperature": 1, "top_k": 20, "top_p": .95, "min_p": 0,
            "seed": 41873, "max_tokens": max_tokens, "enable_thinking": False,
            "add_generation_prompt": True, "return_token_ids": True, "grammar": 'start: "!"',
        }
        ids = prompt_ids(tokenizer, request)
        if len(ids) == length:
            if system_prefix_ids(tokenizer, request, ids):
                raise ValueError("endpoint fixture must not create an instruction checkpoint")
            return request, ids
    raise ValueError(f"cannot render the exact {length}-token checkpoint fixture")


def observed_tokens(record, expected_prompt, expected_generated):
    body = record["body"]
    choice, = body["choices"]
    ids = choice.get("token_ids")
    if (body.get("prompt_token_ids") != expected_prompt or not isinstance(ids, list)
            or len(ids) != expected_generated or any(type(value) is not int or value < 0 for value in ids)
            or body.get("usage") != {"prompt_tokens": len(expected_prompt),
                "completion_tokens": len(ids), "total_tokens": len(expected_prompt) + len(ids)}):
        raise ValueError("native observed prompt/generated token IDs or exact usage differ")
    return ids


def terminal_branch(tokenizer, seed_request, seed_prompt, generated_ids, message, suffix):
    if len(generated_ids) != 2 or message.get("content") != "!" or message.get("tool_calls") or message.get("reasoning_content"):
        raise ValueError("terminal fixture requires one literal token followed by EOS")
    request = copy.deepcopy(seed_request)
    request["max_tokens"] = 1
    request["messages"].extend([
        {"role": "assistant", "content": message["content"]},
        {"role": "user", "content": f"Continuation {suffix}: return one exclamation mark."},
    ])
    ids = prompt_ids(tokenizer, request)
    consumed = seed_prompt + generated_ids[:-1]
    if len(ids) <= len(consumed) or ids[:len(consumed)] != consumed:
        raise ValueError("appended conversation does not preserve the actual consumed token prefix")
    return request, ids, consumed


def counter_checks(delta, *, prompt, reused, chunk_size, snapshots, publications, prompts, message_boundaries=()):
    return {
        "one_completed_request": delta["flare_requests_admitted_total"] == 1
            and delta["flare_requests_completed_total"] == 1
            and delta["flare_requests_failed_total"] == delta["flare_requests_cancelled_total"] == 0,
        "exact_admission": delta["flare_cache_hits_total"] == int(reused > 0)
            and delta["flare_cache_misses_total"] == int(reused == 0)
            and delta["flare_cache_reused_tokens_total"] == reused
            and delta['flare_graph_launches_total{kind="restore"}'] == int(reused > 0)
            and delta['flare_graph_launches_total{kind="init"}'] == int(reused == 0),
        "exact_prompt_execution": delta['flare_graph_launches_total{kind="prefill"}']
            == prefill_launches(prompt, reused, chunk_size, message_boundaries)
            and delta['flare_graph_launches_total{kind="resume"}'] == int(reused == prompt),
        "exact_publication": delta['flare_graph_launches_total{kind="snapshot"}'] == snapshots
            and delta["flare_snapshot_publications_total"] == publications
            and delta["flare_prompt_checkpoints_total"] == prompts
            and delta["flare_message_checkpoints_total"] == sum(reused < n < prompt for n in message_boundaries),
        "no_eviction": delta["flare_snapshot_evictions_total"] == 0,
    }


def run_endpoint_cases(run, tokenizer, model_id, nonce, chunk_size, eos_ids):
    literal, = tokenizer.encode("!", add_special_tokens=False)
    if literal in eos_ids:
        raise ValueError("literal fixture token must not be EOS")

    def measure(name, request, ids, *, reused, snapshots, publications, prompts=0):
        boundaries = [len(prefix) for prefix in message_prefixes(tokenizer, request, ids)]
        messages = sum(n > reused for n in boundaries)
        snapshots += messages
        publications += messages
        before = run.metrics(name + "-before")
        response = run.request(name, request)
        choice = run.complete(response, len(ids))
        after = run.metrics(name + "-after")
        delta = metric_delta(before, after)
        generated = observed_tokens(response, ids, request["max_tokens"])
        run.check(name + ".literal_token_sequence", generated[0] == literal
            and (len(generated) == 1 or generated[1] in eos_ids)
            and choice["message"].get("content") == "!"
            and not choice["message"].get("tool_calls") and not choice["message"].get("reasoning_content"),
            generated_token_ids=generated, eos_token_ids=sorted(eos_ids))
        run.check(name + ".reported_exact_reuse", response["body"].get("reused_prompt_tokens") == reused,
                  expected=reused, observed=response["body"].get("reused_prompt_tokens"))
        for check, passed in counter_checks(delta, prompt=len(ids), reused=reused,
                chunk_size=chunk_size, snapshots=snapshots,
                publications=publications, prompts=prompts, message_boundaries=boundaries).items():
            run.check(name + "." + check, passed, delta=delta)
        run.check(name + ".retained_checkpoint_count",
                  after["flare_cached_checkpoints"] - before["flare_cached_checkpoints"] == publications,
                  before=before["flare_cached_checkpoints"], after=after["flare_cached_checkpoints"])
        run.reclaimed(name, before, after)
        run.results.setdefault("model_metric_deltas", []).append({"case": name, "delta": delta})
        run.save()
        return choice, generated

    def require_capacity(name, snapshots):
        before = run.metrics(name + "-capacity")
        if before["flare_free_snapshots"] < snapshots:
            raise ValueError(f"{name} requires {snapshots} free snapshots to exclude eviction")

    def residues():
        require_capacity("endpoint-residues", 2 * len(PROMPT_LENGTHS))
        seen = []
        for length in PROMPT_LENGTHS:
            name = f"endpoint-{length}"
            request, ids = endpoint_request(tokenizer, model_id, nonce, name, length)
            if any(ids[:len(previous)] == previous for previous in seen):
                raise ValueError("residue fixtures must not reuse another fixture's prompt")
            first, first_ids = measure(name + "-cold", request, ids,
                                      reused=0, snapshots=1, publications=1)
            repeated, repeated_ids = measure(name + "-repeat", request, ids,
                                            reused=length, snapshots=0, publications=0)
            run.check(name + ".exact_residue_replay", ids == prompt_ids(tokenizer, request)
                and first_ids == repeated_ids and first["message"] == repeated["message"],
                prompt_tokens=length, residue=length % 64, prompt_token_ids=ids)
            seen.append(ids)

    def terminal():
        require_capacity("generated-terminal", 9)
        request, ids = endpoint_request(tokenizer, model_id, nonce, "generated-terminal", 194, max_tokens=2)
        first, generated = measure("terminal-seed", request, ids,
                                   reused=0, snapshots=2, publications=2, prompts=1)
        repeated, repeated_ids = measure("terminal-duplicate", request, ids,
                                        reused=len(ids), snapshots=1, publications=0)
        consumed = ids + generated[:-1]
        run.check("terminal-duplicate.same_consumed_generated_prefix",
                  generated[:-1] == repeated_ids[:-1] and first["message"] == repeated["message"],
                  prompt_tokens=len(ids), consumed_tokens=len(consumed), pending_token_id=generated[-1],
                  consumed_token_ids=consumed)
        retained = [consumed]
        for suffix in ("A", "B"):
            branch, branch_ids, prefix = terminal_branch(tokenizer, request, ids, generated, first["message"], suffix)
            run.check(f"terminal-branch-{suffix}.actual_consumed_prefix",
                      prefix == consumed and common_prefix(consumed, branch_ids) == len(consumed),
                      consumed_token_ids=consumed, branch_prompt_token_ids=branch_ids)
            reused = max(len(prefix) for prefix in retained if branch_ids[:len(prefix)] == prefix)
            measure(f"terminal-branch-{suffix}", branch, branch_ids,
                    reused=reused, snapshots=1, publications=1)
            retained.extend(message_prefixes(tokenizer, branch, branch_ids))
            retained.append(branch_ids)

    run.case("prompt_endpoint_residues", residues)
    run.case("generated_terminal_prefix", terminal)
