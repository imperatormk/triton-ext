#!/usr/bin/env python3
"""Lower a basic kernel through the AppleGPU (MPS) backend, headless.

Drives make_ttir -> make_ttgir -> make_llir in-process and prints the
resulting LLVM IR. Unlike `triton-opt`, this runs entirely inside the
interpreter where `libtriton` provides a single shared MLIR dialect identity,
so the conversion patterns operate on ops they actually recognise. No GPU,
driver, or Metal toolchain is required -- only the MLIR->LLVM lowering runs.

Run by hand to debug the Apple lowering:

    TRITON_PLUGIN_PATHS=build/lib/libapplegpu_backend.so \
    TRITON_PASS_PLUGIN_PATH=build/lib/libapplegpu_backend.so \
        python testing/scripts/lower_apple_kernel.py

Exits 0 and prints the lowered LLVM IR to stdout on success.
"""

import sys

import triton
import triton.language as tl
import triton._C.libtriton as libtriton
from triton.backends.compiler import GPUTarget

from triton_apple_backend.compiler import MPSBackend


@triton.jit
def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)  # -> tt.make_range -> ttg.warp_id lowering
    tl.store(out_ptr + offs, tl.load(in_ptr + offs))


def lower() -> str:
    target = GPUTarget("mps", "apple_m", 32)
    backend = MPSBackend(target)
    options = backend.parse_options({
        "num_warps": 4,
        "num_ctas": 1,
        "num_stages": 1,
    })

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={
            "in_ptr": "*fp32",
            "out_ptr": "*fp32"
        },
        constexprs={"BLOCK": 128},
    )

    ctx = libtriton.ir.context()
    backend.load_dialects(ctx)
    libtriton.ir.load_dialects(ctx)

    metadata: dict = {}
    mod = src.make_ir(target, options,
                      backend.get_codegen_implementation(options),
                      backend.get_module_map(), ctx)
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)
    llvm_mod = backend.make_llir(mod, metadata, options)
    return metadata.get("_llvm_ir") or str(llvm_mod)


def main() -> int:
    llir = lower()
    print(llir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
