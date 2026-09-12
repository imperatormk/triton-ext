import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

from triton_apple_backend.compiler import MetalBackend
from triton_apple_backend.hw_constants import TARGET, WARP_SIZE


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, B: tl.constexpr):
    pid = tl.program_id(axis=0)
    off = pid * B + tl.arange(0, B)
    m = off < n
    tl.store(o_ptr + off,
             tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m),
             mask=m)


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

meta: dict = {}
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
