# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve scheduler dependencies without copying sets for in-place reuse."""

from . import _replace


def replacements():
    from torch._inductor.scheduler import BaseSchedulerNode, Scheduler

    decide_inplace_update = _replace(BaseSchedulerNode.decide_inplace_update, (
        (
            """    inconsequential_nodes = (
        self.ancestors
        | V.graph.removed_operations
        | self.scheduler.completed_operations
    )
""",
            """    ancestors = self.ancestors
    removed_operations = V.graph.removed_operations
    completed_operations = self.scheduler.completed_operations
""",
        ),
        (
            "if x.node.get_name() not in inconsequential_nodes",
            """if (name := x.node.get_name()) not in ancestors
                    and name not in removed_operations
                    and name not in completed_operations""",
        ),
    ))
    replace_node = _replace(Scheduler._replace_node, ((
        """    def rename_deps(deps: OrderedSet[Dep]) -> OrderedSet[Dep]:
        return OrderedSet(dep.rename(mutation_renames) for dep in deps)

    new_scheduler_node.unmet_dependencies = rename_deps(
        new_scheduler_node.unmet_dependencies
    )
    new_scheduler_node.read_writes.reads = rename_deps(
        new_scheduler_node.read_writes.reads
    )
""",
        """    # Rebuild unmet dependencies after renaming inputs to their mutation outputs.
    # Initial pruning treats the original graph inputs as already available.
    new_scheduler_node.update_mutated_names(mutation_renames)
""",
    ),), allow_applied=True)
    return [(BaseSchedulerNode, "decide_inplace_update", decide_inplace_update),
            (Scheduler, "_replace_node", replace_node)]
