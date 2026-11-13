# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Every peer must acknowledge resident table setup before native graph handoff."""

from types import SimpleNamespace

from ignition.preparation.rank_group import PreparedRankGroup


def test_rank_setup_waits_for_all_table_mappings_before_starting_execution(monkeypatch):
    import multiprocessing.reduction

    actions = []
    catalog = {"vocabulary": 32, "tables": [object()]}
    resident = SimpleNamespace(descriptors=[31, 32], nbytes=2048)

    class Control:
        def __init__(self, rank):
            self.rank = rank

        def send(self, message):
            actions.append((self.rank, message))

        def recv(self):
            actions.append((self.rank, "mapped"))
            return "embedding_tables_ready"

        def close(self):
            pass

    class Model:
        memory_report = {}
        embedding_catalog = catalog

        def prepare_embedding_tables(self):
            actions.append((0, "loaded"))
            return resident

        def start_server(self, **kwargs):
            actions.append((0, "executing"))
            return self

    def send_handle(control, descriptor, pid):
        actions.append((control.rank, (descriptor, pid)))

    def unexpected_abort(message):
        raise AssertionError(message)

    monkeypatch.setattr(multiprocessing.reduction, "send_handle", send_handle)
    controls = [Control(1), Control(2)]
    group = PreparedRankGroup(Model(), [SimpleNamespace(pid=101), SimpleNamespace(pid=102)],
        [], controls, SimpleNamespace(abort=unexpected_abort), None, [{}, {}])
    assert group.start_server() is group.model
    mapped = [actions.index((rank, "mapped")) for rank in (1, 2)]
    for rank in (1, 2):
        assert actions.index((rank, "serve")) > max(mapped)
    assert actions.index((0, "executing")) > max(mapped)
    assert actions[0] == (0, "loaded")
