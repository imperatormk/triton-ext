"""Shape ops that move data between threads.

`tl.trans` stages the tile in threadgroup memory so the read and the write are
each contiguous. A layout change that keeps every element in its own warp is a
lane exchange, and one whose permutation is the identity emits nothing.
"""

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

DEVICE = torch.device("mps")


@triton.jit
def transpose_tile(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    tile = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    tl.store(o_ptr + offs_n[:, None] * M + offs_m[None, :], tl.trans(tile))


@triton.jit
def transpose_blocked(x_ptr, o_ptr, M, N, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr):
    offs_m = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(axis=1) * BLOCK_N + tl.arange(0, BLOCK_N)
    tile = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    tl.store(o_ptr + offs_n[:, None] * M + offs_m[None, :], tl.trans(tile))


@pytest.mark.parametrize("shape", [(32, 32), (16, 64), (64, 16), (8, 128)])
def test_transpose_one_tile(shape):
    M, N = shape
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE)
    out = torch.zeros((N, M), device=DEVICE)

    transpose_tile[(1, )](x, out, M=M, N=N)
    torch.testing.assert_close(out, x.T.contiguous())


@pytest.mark.parametrize("size", [64, 128, 512])
def test_transpose_grid(size):
    torch.manual_seed(0)
    x = torch.randn((size, size), device=DEVICE)
    out = torch.zeros((size, size), device=DEVICE)

    grid = (triton.cdiv(size, 32), triton.cdiv(size, 32))
    transpose_blocked[grid](x, out, size, size, BLOCK_M=32, BLOCK_N=32)
    torch.testing.assert_close(out, x.T.contiguous())


def test_transpose_stages_one_buffer():
    """The convert the frontend pairs with `trans` is a rename, not a trip."""
    M = N = 32
    x = torch.randn((M, N), device=DEVICE)
    out = torch.zeros((N, M), device=DEVICE)

    compiled = transpose_tile[(1, )](x, out, M=M, N=N)
    msl = compiled.asm["msl"]
    assert msl.count("threadgroup float rd") == 1
