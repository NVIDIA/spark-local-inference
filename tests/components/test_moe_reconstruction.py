# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Routed experts survive the pickling that carries forward metadata to compiler workers."""

import pickle

from ignition.components.moe import PreparedRoutedExperts


def test_prepared_routed_experts_unpickle_without_constructor_arguments():
    prepared = PreparedRoutedExperts.__new__(PreparedRoutedExperts)
    prepared.name = "layer0.experts"
    restored = pickle.loads(pickle.dumps(prepared))
    assert type(restored) is PreparedRoutedExperts and restored.name == "layer0.experts"
