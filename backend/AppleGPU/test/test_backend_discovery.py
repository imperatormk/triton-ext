"""Compiles a kernel down to MSL, which is every stage before the toolchain"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent / "discovery"


def _run(script: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        capture_output=True,
        text=True)


def _skip_if_no_plugin(proc):
    if "plugin pass missing" in proc.stderr or "add_emit_msl" in proc.stderr:
        pytest.skip("AppleGPU plugin not loaded (build the backend)")


def test_triton_discovery_imports_the_backend_in_order():
    proc = _run("import_order.py")
    _skip_if_no_plugin(proc)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


def test_the_compiler_lowers_a_kernel_to_msl():
    proc = _run("lower_to_msl.py")
    _skip_if_no_plugin(proc)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
