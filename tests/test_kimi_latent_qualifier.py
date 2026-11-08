"""The latent qualifier installs recipe dependency policy before importing b12x."""

import json
from pathlib import Path
import subprocess
import sys


def test_recipe_environment_is_applied_before_library_imports(tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text(json.dumps({'architectures': ['KimiK31ForConditionalGeneration']}))
    configuration = tmp_path / 'configuration.toml'
    configuration.write_text('[paths]\nkernel_cache = "' + str(tmp_path / 'kernels') + '"\n')
    code = '''
import os
import sys
from types import ModuleType, SimpleNamespace
import torch

def no_cuda(*args, **kwargs):
    raise AssertionError('environment configuration must not initialize CUDA')
torch._C._cuda_init = no_cuda
torch.cuda._lazy_init = no_cuda
from ignition import runtime_patches
from ignition.recipes.kimi31.recipe import RECIPE
from qualify.kimi_latent_prefill import configure_environment

def check_before_patches():
    assert all(os.environ[name] == value for name, value in RECIPE.b12x_environment.items())
    assert not any(name == 'b12x' or name.startswith('b12x.') for name in sys.modules)
runtime_patches.apply = check_before_patches
policy = configure_environment(sys.argv[1], sys.argv[2], [SimpleNamespace(uuid='GPU-selected')])
assert policy['recipe'] == 'kimi31'
assert policy['recipe_b12x'] == dict(RECIPE.b12x_environment)
assert policy['collective_tuning'] == 'static' and policy['autotune_effort'] == 'default'
assert policy['effective_b12x']['B12X_PCIE_DCP_BLOCK_LIMIT'] == '8'
assert policy['effective_b12x']['B12X_COMPILE_CACHE_DIR'].endswith('/kernels/b12x')
assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-selected'
assert not torch.cuda.is_initialized()
sys.modules['b12x'] = ModuleType('b12x')
try:
    configure_environment(sys.argv[1], sys.argv[2], [SimpleNamespace(uuid='GPU-selected')])
except RuntimeError as error:
    assert 'before importing b12x' in str(error)
else:
    raise AssertionError('late dependency policy application was accepted')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(checkpoint), str(configuration)],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
