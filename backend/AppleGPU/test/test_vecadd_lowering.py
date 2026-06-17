"""AppleGPU backend: a Triton vecadd lowers to the expected AIR IR.

Compiles a @triton.jit vecadd through the backend with an explicit GPUTarget,
so it runs with no Apple GPU present (no device dispatch) and the emitted IR can
be checked on any CI runner. Skips when the backend .so is not built.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_DIR = Path(os.environ.get("BUILD_DIR", REPO_ROOT / "build"))
COMPILE_SCRIPT = Path(__file__).resolve().parent / "compile_vecadd.py"

_LIB_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
PLUGIN = BUILD_DIR / "lib" / f"libapplegpu_backend{_LIB_SUFFIX}"

pytestmark = pytest.mark.skipif(
    not PLUGIN.is_file(), reason=f"AppleGPU backend not built at {PLUGIN}")


def _compile_stage(stage: str) -> str:
    env = dict(os.environ)
    env["TRITON_PLUGIN_PATHS"] = str(PLUGIN)
    triton_python = Path(os.environ["TRITON_INSTALL_DIR"]) / "python"
    backend_python = REPO_ROOT / "backend" / "AppleGPU" / "python"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(triton_python),
         str(backend_python),
         env.get("PYTHONPATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join([
        str(Path(os.environ["LLVM_INSTALL_DIR"]) / "lib"),
        env.get("LD_LIBRARY_PATH", "")
    ])
    metal_llc = BUILD_DIR / "bin" / "metal-llc"
    if metal_llc.is_file():
        env["METAL_LLC_PATH"] = str(metal_llc)
    result = subprocess.run(
        [sys.executable, str(COMPILE_SCRIPT), stage],
        env=env,
        capture_output=True,
        text=True)
    assert result.returncode == 0, (
        f"compile_vecadd.py {stage} failed\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    return result.stdout


def test_vecadd_ttgir_layout_and_add() -> None:
    ttgir = _compile_stage("ttgir")
    assert "tt.func" in ttgir
    assert "#ttg.blocked" in ttgir
    assert "arith.addf" in ttgir


def test_vecadd_llir_air_intrinsics_and_add() -> None:
    llir = _compile_stage("llir")
    assert "air.thread_position_in_threadgroup" in llir
    assert "fadd" in llir
