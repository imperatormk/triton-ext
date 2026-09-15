"""Round-trips a tensor through threadgroup memory on the GPU.

The shared encoding decides where each element is stored: a swizzle permutes
the offsets inside a row, padding splices extra elements into them. Reading
back what was written is what proves the writer and the reader agree.

Gluon is used because it names the memdesc ops directly; a `tl` kernel reaches
them only through ops this backend does not lower yet.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton.experimental.gluon as gluon  # noqa: E402
import triton.experimental.gluon.language as ttgl  # noqa: E402

N = 128


@gluon.jit
def roundtrip(x_ptr, o_ptr, n: ttgl.constexpr, layout: ttgl.constexpr,
              shared: ttgl.constexpr):
    off = ttgl.arange(0, n, layout=layout)
    x = ttgl.load(x_ptr + off)
    smem = ttgl.allocate_shared_memory(x.dtype, (n, ), shared)
    smem.store(x)
    ttgl.store(o_ptr + off, smem.load(layout))


def swizzled(vec, per_phase, max_phase):
    return ttgl.SwizzledSharedLayout(vec=vec,
                                     per_phase=per_phase,
                                     max_phase=max_phase,
                                     order=[0])


SHARED_LAYOUTS = {
    "unswizzled": swizzled(1, 1, 1),
    "vec4": swizzled(4, 2, 4),
    "vec8": swizzled(8, 1, 8),
    "padded": ttgl.PaddedSharedLayout.with_identity_for([[16, 4]], [N], [0]),
}


@pytest.mark.parametrize("shared",
                         SHARED_LAYOUTS.values(),
                         ids=SHARED_LAYOUTS.keys())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_a_tensor_survives_threadgroup_memory(shared, dtype):
    layout = ttgl.BlockedLayout([1], [32], [4], [0])
    x = torch.randn(N, device="mps").to(dtype)
    out = torch.empty_like(x)

    roundtrip[(1, )](x, out, N, layout, shared, num_warps=4)
    torch.mps.synchronize()

    assert torch.equal(out.cpu(), x.cpu())


@pytest.mark.xfail(raises=RuntimeError,
                   strict=True,
                   reason="nvmma_shared tiles rows against columns, so it "
                   "says nothing at rank 1; a rank-2 kernel needs "
                   "tt.expand_dims")
def test_nvmma_needs_two_dimensions():
    layout = ttgl.BlockedLayout([1], [32], [4], [0])
    shared = ttgl.NVMMASharedLayout(swizzle_byte_width=0,
                                    element_bitwidth=16,
                                    rank=1)
    x = torch.randn(N, device="mps").to(torch.float16)
    out = torch.empty_like(x)

    roundtrip[(1, )](x, out, N, layout, shared, num_warps=4)
