# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Metadata-only compatibility entry point for the native serving encoder."""

import json


def serving(index, server):
    import flare_heliopause
    index.ParseFromString(flare_heliopause.export_serving(index.SerializeToString(deterministic=True),
                                                         json.dumps(server)))
