# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark's recipient-header prompt syntax plus bracketed <|patch|> images for Muse Glimmer."""

from pathlib import Path

from ignition.preparation import runtime_package_pb2 as p
from ignition.preparation.frontend import prepare_frontend as _prepare_frontend
from ignition.recipes.muse_spark.frontend import FRONTEND as _SPARK_FRONTEND
from ignition.recipes.muse_spark.frontend import prepare_frontend as _prepare_spark_frontend
from ignition.recipes.muse_spark.frontend import runtime_assets as _runtime_assets

TEMPLATE_PATCH = Path(__file__).with_name("assets") / "chat_template.patch.json"

FRONTEND = dict(
    _SPARK_FRONTEND,
    image_preparation=p.MUSE_GLIMMER_IMAGE_PREPARATION,
    image_layout=p.BRACKETED_IMAGE_LAYOUT,
    image_tokens=dict(feature_token=200092, span=None),
    rejected_tokens=[200091],
)


def prepare_frontend(snapshot, *, eos_token_ids):
    snapshot = Path(snapshot)
    settings = _prepare_spark_frontend(snapshot, eos_token_ids=eos_token_ids)
    return _prepare_frontend(snapshot, eos_token_ids=settings["eos_token_ids"], image_profile="muse_glimmer",
                             preprocessor=snapshot / "processor_config.json")


def runtime_assets(snapshot):
    return _runtime_assets(snapshot, patch=TEMPLATE_PATCH)
