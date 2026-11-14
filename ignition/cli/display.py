# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation error and speculation formatting."""

import sys


def _console(stream, **options):
    from rich.console import Console

    return Console(file=stream, highlight=False, markup=False, emoji=False, **options)



def error(message: str) -> None:
    """Print a failure on stderr with its continuation lines aligned under the first."""
    from rich.text import Text

    _console(sys.stderr, soft_wrap=True).print(
        Text.assemble(("error:", "bold red"), " " + message.replace("\n", "\n       ")))



def speculation(settings, *, brief: bool = False) -> str:
    """Describe speculation settings: the drafts verified per round and where they come from.

    The brief form fits a startup line.
    """
    if settings is None:
        return "speculation off" if brief else "off"
    drafts = settings.draft_tokens
    depth = f"{'up to' if settings.adaptive else 'exactly'} {drafts} draft token{'' if drafts == 1 else 's'}"
    parallel = " · vocabulary-parallel block sampling" if settings.vocabulary_parallel_sampling else ""
    selection = " · GPU prefix selection" if settings.draft_prefix_selection == "gpu" else ""
    heads = [f"{head} head {getattr(settings, head + '_head_quantization').upper()}"
             for head in ("draft", "markov") if getattr(settings, head + '_head_quantization') is not None]
    if getattr(settings, "draft_body_quantization", None) is not None:
        heads.insert(0, f"draft body {settings.draft_body_quantization.upper()}")
    precision = " · " + ", ".join(heads) if heads else ""
    catalog = " · MTP-only graphs" if settings.mtp_only else ""
    if brief:
        return depth + selection + precision + parallel + catalog
    source = "DFlash2" if settings.method == "dflash2" else "MTP module"
    return f"{depth} per round · {source}" + selection + precision + parallel + catalog
