"""Compiles a kernel down to MSL, which is every stage before the toolchain"""

from __future__ import annotations

import os
import sys

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import passes
from triton.backends.compiler import GPUTarget

from triton_apple_backend.compiler import MetalBackend
from triton_apple_backend.hw_constants import TARGET, WARP_SIZE


def _require_plugin():
    if hasattr(passes.plugin, "add_emit_msl"):
        return
    if os.environ.get("TRITON_EXT_REQUIRE_APPLEGPU"):
        pytest.fail("the plugin registered no add_emit_msl pass")
    pytest.skip("AppleGPU plugin not loaded (build the backend)")


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, B: tl.constexpr):
    pid = tl.program_id(axis=0)
    off = pid * B + tl.arange(0, B)
    m = off < n
    tl.store(o_ptr + off,
             tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m),
             mask=m)


def test_triton_discovery_imports_the_backend():
    _require_plugin()
    # torch is this backend's only dispatch path, so discovery pulls it in too.
    for mod in ("triton_apple_backend", "triton_apple_backend.driver",
                "triton_apple_backend.compiler", "torch"):
        assert mod in sys.modules, f"{mod} was not imported by discovery"

    from triton_apple_backend import driver
    assert hasattr(driver, "MetalDriver")


def test_the_compiler_lowers_a_kernel_to_msl():
    _require_plugin()
    backend = MetalBackend(GPUTarget(TARGET, "apple_m", WARP_SIZE))
    options = backend.parse_options({"num_warps": 4})
    stages: dict = {}
    backend.add_stages(stages, options, "msl")
    assert list(stages) == ["ttir", "ttgir", "msl", "metallib"], list(stages)

    src = triton.compiler.ASTSource(fn=add_kernel,
                                    signature={
                                        "x_ptr": "*fp32",
                                        "y_ptr": "*fp32",
                                        "o_ptr": "*fp32",
                                        "n": "i32",
                                        "B": "constexpr"
                                    },
                                    constexprs={"B": 1024})

    ctx = triton._C.libtriton.ir.context()
    triton._C.libtriton.ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    mod = src.make_ir(backend.target, options,
                      backend.get_codegen_implementation(options),
                      backend.get_module_map(), ctx)

    meta: dict = {}
    for stage in ("ttir", "ttgir", "msl"):
        mod = stages[stage](mod, meta)

    assert "kernel void" in mod, mod[:200]
    assert "arg0 [[buffer(0)]]" in mod, mod[:200]
