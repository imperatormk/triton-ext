"""K loops pipelined by num_stages: operand loads run ahead into registers
the loop carries, the first ones before the loop and the last dot after it."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def mm(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
       BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        in_k = k0 + rk < K
        a = tl.load(a_ptrs, mask=in_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=in_k[:, None], other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BK
        b_ptrs += BK * N
    tl.store(c_ptr + rm[:, None] * N + rn[None, :],
             acc.to(c_ptr.dtype.element_ty))


@pytest.mark.parametrize("num_stages", [1, 2, 3])
@pytest.mark.parametrize("K", [0, 16, 32, 48, 200, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mm(num_stages, K, dtype):
    M, N, BM, BN, BK = 64, 64, 32, 32, 16
    a = torch.randn(M, K, dtype=dtype)
    b = torch.randn(K, N, dtype=dtype)
    c = torch.empty(M, N, device="mps", dtype=dtype)
    mm[(M // BM, N // BN)](a.to("mps"),
                           b.to("mps"),
                           c,
                           M,
                           N,
                           K,
                           BM=BM,
                           BN=BN,
                           BK=BK,
                           num_warps=4,
                           num_stages=num_stages)
    tol = 1e-4 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(c.cpu().float(),
                               a.float() @ b.float(),
                               rtol=tol,
                               atol=tol * max(K, 1)**0.5)
