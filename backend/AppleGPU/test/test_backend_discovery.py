"""Drives the backend as Triton does, as far as a machine without Metal can go.

Triton's discovery imports this package during `import triton`, so the import
order is itself load-bearing. After that the compiler runs ttir -> ttgir -> msl
on a real kernel; only the metallib stage needs the Metal toolchain.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

IMPORT_ORDER = textwrap.dedent("""
    import sys

    # Nothing of ours is loaded before Triton asks for it.
    assert "triton_apple_backend" not in sys.modules
    import triton
    from triton._C.libtriton import passes

    # Discovery imported the package and both Triton-facing modules.
    for mod in ("triton_apple_backend", "triton_apple_backend.driver",
                "triton_apple_backend.compiler"):
        assert mod in sys.modules, f"{mod} was not imported by discovery"

    # And the plugin registered its pass.
    assert hasattr(passes.plugin, "add_emit_msl"), "plugin pass missing"

    import triton_apple_backend.driver as driver
    assert hasattr(driver, "MetalDriver")
    print("OK")
""")

KERNEL = textwrap.dedent("""
    import torch
    import triton
    import triton.language as tl
    from triton_apple_backend.compiler import MetalBackend, MetalOptions
    from triton.backends.compiler import GPUTarget
    from triton_apple_backend.hw_constants import TARGET, WARP_SIZE

    @triton.jit
    def add_kernel(x_ptr, y_ptr, o_ptr, n, B: tl.constexpr):
        pid = tl.program_id(axis=0)
        off = pid * B + tl.arange(0, B)
        m = off < n
        tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) +
                 tl.load(y_ptr + off, mask=m), mask=m)

    backend = MetalBackend(GPUTarget(TARGET, "apple_m", WARP_SIZE))
    options = backend.parse_options({"num_warps": 4})
    stages = {}
    backend.add_stages(stages, options, "msl")
    assert list(stages) == ["ttir", "ttgir", "msl", "metallib"], list(stages)

    src = triton.compiler.ASTSource(
        fn=add_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32",
                   "n": "i32", "B": "constexpr"},
        constexprs={"B": 1024})

    meta = {}
    ctx = triton._C.libtriton.ir.context()
    triton._C.libtriton.ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    mod = src.make_ir(backend.target, options,
                      backend.get_codegen_implementation(options),
                      backend.get_module_map(), ctx)
    for stage in ("ttir", "ttgir", "msl"):
        mod = stages[stage](mod, meta)

    assert "kernel void" in mod, mod[:200]
    assert "arg0 [[buffer(0)]]" in mod, mod[:200]
    print("OK")
""")


def _run(script):
    # Through a file: `@triton.jit` reads its function's source.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        return subprocess.run([sys.executable, path],
                              capture_output=True,
                              text=True)
    finally:
        os.unlink(path)


def _skip_if_no_plugin(proc):
    if "plugin pass missing" in proc.stderr or "add_emit_msl" in proc.stderr:
        pytest.skip("AppleGPU plugin not loaded (build the backend)")


def test_triton_discovery_imports_the_backend_in_order():
    proc = _run(IMPORT_ORDER)
    _skip_if_no_plugin(proc)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


def test_the_compiler_lowers_a_kernel_to_msl():
    proc = _run(KERNEL)
    _skip_if_no_plugin(proc)
    if proc.returncode != 0 and "No module named 'torch'" in proc.stderr:
        pytest.skip("torch is needed to build a kernel's signature")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
