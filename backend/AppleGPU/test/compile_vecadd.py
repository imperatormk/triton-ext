#!/usr/bin/env python3
"""Lower a Triton vecadd through the AppleGPU backend and print one IR stage.

Compiles with an explicit GPUTarget so it runs without an Apple GPU present
(no device dispatch), making the emitted IR FileCheck-able on any CI runner.

Usage:
    TRITON_PLUGIN_PATHS=.../libapplegpu_backend.so \\
    PYTHONPATH=.../triton-*/python:.../backend/AppleGPU/python \\
    METAL_LLC_PATH=.../bin/metal-llc \\
        python compile_vecadd.py {ttir|ttgir|llir}
"""

import sys

import triton
import triton.backends
import triton.language as tl
from triton.backends import Backend
from triton.backends.compiler import GPUTarget


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def _register_backend() -> None:
    from triton.backends.driver import DriverBase
    from triton_apple_backend.compiler import MPSBackend
    triton.backends.backends["apple"] = Backend(MPSBackend, DriverBase)


def main() -> int:
    _register_backend()
    stage = sys.argv[1] if len(sys.argv) > 1 else "llir"
    src = triton.compiler.ASTSource(
        fn=add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n": "i32",
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": 1024},
    )
    compiled = triton.compile(src, target=GPUTarget("mps", "apple_m", 32))
    sys.stdout.write(compiled.asm[stage])
    return 0


if __name__ == "__main__":
    sys.exit(main())
