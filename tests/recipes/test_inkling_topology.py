# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inkling NVFP4 placements and memory planning through the shared planner."""

import pytest

from ignition import planning
from ignition.errors import CapacityError
from ignition.recipes import resolve
from ignition.recipes.inkling.recipe import RECIPE


@pytest.mark.parametrize('devices,hosts', [
    ((3,), None), ((1, 3), None), ((0, 0), ('spark-a', 'spark-b')),
])
def test_one_or_two_distinct_local_or_remote_placements(devices, hosts):
    from ignition.recipes.inkling.bootstrap import PreparationConfig

    config = PreparationConfig(devices=devices, hosts=hosts)
    assert config.devices == devices and config.hosts == hosts


@pytest.mark.parametrize('devices,hosts', [
    ((), None), ((0, 1, 2, 3), None), ((0, 0), None), ((0, -1), None),
    ((0, 0), ('spark-a', 'spark-a')), ((0, 1), ('spark-a',)),
])
def test_unsupported_placements_fail_before_loading(devices, hosts):
    from ignition.recipes.inkling.bootstrap import PreparationConfig

    with pytest.raises(ValueError, match='one or two distinct host/device placements'):
        PreparationConfig(devices=devices, hosts=hosts)


def test_roce_ring_requires_remote_placements():
    from ignition.recipes.inkling.bootstrap import PreparationConfig

    with pytest.raises(ValueError, match='ranks on separate hosts'):
        PreparationConfig(roce_ring=True)


def test_unsupported_gpu_is_rejected_before_checkpoint_loading(monkeypatch, tmp_path):
    from ignition.recipes.inkling import bootstrap

    monkeypatch.setattr(bootstrap.torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setattr(bootstrap.torch.cuda, 'get_device_capability', lambda device: (9, 0))
    with pytest.raises(ValueError, match='Inkling NVFP4 preparation requires SM'):
        bootstrap.prepare_rank(tmp_path / 'not-downloaded', bootstrap.PreparationConfig(),
                               rank=0, group=None, package_storage=None)


@pytest.mark.parametrize('devices,group', [((0,), object()), ((0, 1), None)])
def test_rank_group_matches_tensor_parallel_placement(tmp_path, devices, group):
    from ignition.recipes.inkling import bootstrap

    with pytest.raises(ValueError, match='rank group exists exactly'):
        bootstrap.prepare_rank(tmp_path, bootstrap.PreparationConfig(devices=devices),
                               rank=0, group=group, package_storage=None)


def test_nvfp4_plan_fits_one_station():
    station = planning.Gpu(0, 'NVIDIA GB300', 'GPU-0', 256 << 30, 240 << 30, (10, 3))
    plan = planning.plan(RECIPE, RECIPE.checkpoints[0], (station,), selected=None,
                         concurrency=4, reserve_bytes=4 << 30, speculation=None)
    assert plan.world_size == 1 and plan.fits
    assert plan.context == plan.guaranteed_tokens == 262144
    assert plan.costs.state_row_bytes == 82_837_504
    assert plan.costs.page_bytes == 3_670_016
    assert plan.costs.page_bytes * (plan.context // 128) == 7 << 30
    assert plan.weights_bytes >= 166_130_594_960


def test_nvfp4_plan_fits_two_sparks_with_host_reserve():
    model = resolve('inkling-small-nvfp4')
    assert model.recipe is RECIPE and model.checkpoint.repository == 'thinkingmachines/Inkling-Small-NVFP4'
    gpus = tuple(planning.Gpu(index, 'NVIDIA GB10', f'GPU-{index}', 121 << 30, 118 << 30, (12, 1))
                 for index in range(2))
    plan = planning.plan(RECIPE, model.checkpoint, gpus, selected=(0, 1),
                         concurrency=4, reserve_bytes=4 << 30, speculation=None)
    assert plan.world_size == 2 and plan.fits
    assert plan.context == plan.guaranteed_tokens == 262144
    assert plan.costs.state_row_bytes == 43_515_904
    assert plan.costs.page_bytes == 1_835_008
    assert plan.weights_bytes >= 166_130_594_960 // 2
    assert plan.reserve_bytes == 4 << 30
    with pytest.raises(CapacityError, match='weights'):
        planning.plan(RECIPE, model.checkpoint, gpus[:1], selected=(0,),
                      concurrency=4, reserve_bytes=4 << 30, speculation=None)
    small = tuple(planning.Gpu(index, 'NVIDIA RTX', f'GPU-{index}', 64 << 30, 63 << 30, (12, 0))
                  for index in range(2))
    with pytest.raises(CapacityError):
        planning.plan(RECIPE, model.checkpoint, small, selected=(0, 1),
                      concurrency=4, reserve_bytes=0, speculation=None)
