# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level checks for native rank failure containment and parent lifetime."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("flare_heliopause")

DRIVER = r'''
import json, os, subprocess, sys, time
import flare_heliopause
child = "import flare_heliopause,sys; flare_heliopause.bind_parent_lifetime(int(sys.argv[1])); print('ready',flush=True); sys.stdin.read(1)"
peers = [subprocess.Popen([sys.executable, '-c', child, str(os.getpid())], stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(2)]
for peer in peers:
    assert peer.stdout.readline() == b'ready\n'
supervisor = flare_heliopause.RankSupervisor([peer.pid for peer in peers])
print(json.dumps([peer.pid for peer in peers]), flush=True)
mode = sys.stdin.readline().strip()
if mode == 'normal':
    supervisor.begin_shutdown(5000)
    for peer in peers:
        peer.stdin.write(b'q'); peer.stdin.flush()
    supervisor.finish_shutdown()
    assert all(peer.wait() == 0 for peer in peers)
elif mode == 'crash':
    peers[0].kill()
    time.sleep(60)
elif mode == 'deadline':
    supervisor.begin_shutdown(100)
    supervisor.finish_shutdown()
elif mode == 'orphan':
    os._exit(0)
'''


def terminated(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].startswith("Z ")
    except (FileNotFoundError, ProcessLookupError):
        return True


@pytest.mark.parametrize("mode", ["normal", "crash", "deadline", "orphan"])
def test_rank_group_shutdown_and_failure_do_not_leave_live_peers(mode):
    driver = subprocess.Popen([sys.executable, "-u", "-c", DRIVER], text=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    peers = []
    try:
        peers = json.loads(driver.stdout.readline())
        _, error = driver.communicate(mode + "\n", timeout=10)
        assert driver.returncode == (0 if mode in ("normal", "orphan") else 1), error
        if mode == "crash":
            assert "exited before coordinated shutdown" in error
        if mode == "deadline":
            assert "shutdown deadline expired" in error
        deadline = time.monotonic() + 3
        while not all(terminated(pid) for pid in peers) and time.monotonic() < deadline:
            time.sleep(.01)
        assert all(terminated(pid) for pid in peers)
    finally:
        if driver.poll() is None or not all(terminated(pid) for pid in peers):
            os.killpg(driver.pid, signal.SIGKILL)
        driver.wait(timeout=5)


def test_supervisor_rejects_unowned_process():
    import flare_heliopause

    with pytest.raises(ValueError):
        flare_heliopause.RankSupervisor([os.getpid()])


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_cli_signals_exit_during_native_gil_hold_without_python_finalization(signum):
    code = r'''
import atexit, ctypes, time
import flare_heliopause
atexit.register(lambda: print("finalizer ran", flush=True))
with flare_heliopause.CliSignals():
    print("ready", flush=True)
    ctypes.PyDLL(None).sleep(60)
'''
    process = subprocess.Popen([sys.executable, "-u", "-c", code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        assert process.stdout.readline() == b"ready\n"
        started = time.monotonic()
        process.send_signal(signum)
        output, error = process.communicate(timeout=2)
        assert process.returncode == 128 + signum
        assert time.monotonic() - started < 1
        assert output == error == b""
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_cli_signal_scope_restores_embedded_python_handlers():
    import flare_heliopause

    observed = []
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: observed.append(signum))
    try:
        with flare_heliopause.CliSignals():
            with pytest.raises(RuntimeError, match="already installed"):
                flare_heliopause.CliSignals()
        os.kill(os.getpid(), signal.SIGTERM)
        assert observed == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_cli_interrupt_leaves_no_compiler_processes():
    code = r'''
import json, os, time
from pathlib import Path
import flare_heliopause
from torch._inductor.compile_worker.subproc_pool import SubprocPool
with flare_heliopause.CliSignals():
    pool = SubprocPool(1)
    pending = pool.submit(time.sleep, 60)
    coordinator = pool.process.pid
    children = Path(f"/proc/{coordinator}/task/{coordinator}/children")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        workers = [int(pid) for pid in children.read_text().split()]
        if workers:
            break
        time.sleep(.01)
    else:
        raise RuntimeError("compiler worker did not start")
    print(json.dumps([coordinator, *workers]), flush=True)
    pending.result()
'''
    process = subprocess.Popen([sys.executable, "-u", "-c", code], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    peers = []
    try:
        line = process.stdout.readline()
        assert line, process.stderr.read()
        peers = json.loads(line)
        process.send_signal(signal.SIGINT)
        process.communicate(timeout=3)
        assert process.returncode == 130
        deadline = time.monotonic() + 5
        while not all(terminated(pid) for pid in peers) and time.monotonic() < deadline:
            time.sleep(.01)
        assert all(terminated(pid) for pid in peers)
    finally:
        if process.poll() is None or not all(terminated(pid) for pid in peers):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


PEER_CLEANUP = r'''
import atexit, os, sys, time
from contextlib import nullcontext
from types import SimpleNamespace
import torch
from ignition.preparation import rank_group

# Controlled owners isolate process lifecycle; this test does not execute CUDA.
mode, failure = sys.argv[1:]
def cleanup(name):
    print(name, flush=True)
    if name == failure:
        raise RuntimeError(f"cleanup failed: {name}")

class Model:
    memory_report = {}
    def manifest(self):
        return {}
    def close(self):
        cleanup("model-close")
    def __del__(self):
        print("terminal-destructor", flush=True)
        time.sleep(60)

class Region:
    def __init__(self, name):
        self.name = name
    def close(self):
        cleanup(self.name)

class Storage:
    def __init__(self, **kwargs):
        self.host_regions = [Region("host-0-close"), Region("host-1-close")]
    def __enter__(self):
        return self
    def __exit__(self, *error):
        return False

class Control:
    def send(self, value):
        assert value == ("tuning_complete" if mode == "tuning" else {})
    def recv(self):
        return "close"
    def close(self):
        cleanup("control-close")

rank_group.dist = SimpleNamespace(
    init_process_group=lambda *args, **kwargs: None,
    group=SimpleNamespace(WORLD=object()),
    destroy_process_group=lambda: cleanup("distributed-close"),
)
torch.cuda.set_device = lambda device: None
sys.modules["ignition.preparation.module_images"] = SimpleNamespace(ModuleImages=nullcontext)
sys.modules["ignition.preparation.core_storage"] = SimpleNamespace(CoreStorage=Storage)
connection = SimpleNamespace(close=lambda: cleanup("connection-close"))
def prepare(*args, **kwargs):
    if mode == "tuning":
        def terminal_destructor():
            print("terminal-destructor", flush=True)
            time.sleep(60)
        atexit.register(terminal_destructor)
        raise rank_group.TuningComplete()
    return Model()
rank_group._peer(os.getppid(), None, SimpleNamespace(devices=(0, 1)), 1,
    "unused", connection, Control(), prepare, True, True)
print("peer-returned", flush=True)
'''


@pytest.mark.parametrize("mode,failure", [
    ("model", "success"), ("model", "model-close"), ("model", "host-0-close"),
    ("model", "host-1-close"), ("model", "distributed-close"), ("tuning", "success"),
])
def test_preparation_peer_exits_only_after_explicit_cleanup(mode, failure):
    process = subprocess.Popen([sys.executable, "-u", "-c", PEER_CLEANUP, mode, failure], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        output, error = process.communicate(timeout=10)
        expected = ["control-close", "connection-close", "model-close", "host-0-close",
                    "host-1-close", "distributed-close"]
        if mode == "tuning":
            expected = ["control-close", "connection-close", "distributed-close"]
        if failure != "success":
            expected = expected[:expected.index(failure) + 1]
            assert process.returncode == 1, (output, error)
            assert f"cleanup failed: {failure}" in error
        else:
            assert process.returncode == 0, (output, error)
            assert error == ""
        assert output.splitlines() == expected
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
