"""Work only one arm of a select reads runs under that arm's condition."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the dispatch path is torch's")

if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip("needs an Apple GPU", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def alternate_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, tl.where(i % 2 == 0, tl.sin(x), tl.cos(x)))


@triton.jit
def nested_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i)
    inner = tl.where(i % 64 < 48, tl.cos(x), tl.exp(x))
    tl.store(o_ptr + i, tl.where(i % 64 < 32, tl.sin(x), inner))


@triton.jit
def halves_2d_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    x = tl.load(x_ptr + r * C + c)
    tl.store(o_ptr + r * C + c, tl.where(c < C // 2, tl.sin(x), tl.exp(x)))


@triton.jit
def cheap_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, tl.where(i % 64 < 32, x * 2.0 + 1.0, x - 3.0))


def _body(k):
    return k.asm["msl"].split("kernel void")[-1]


@pytest.mark.parametrize("num_warps", [1, 4])
def test_a_lane_varying_condition_still_picks_per_element(num_warps):
    x = torch.randn(4096, device="mps")
    out = torch.empty_like(x)
    k = alternate_kernel[(4, )](x, out, BLOCK=1024, num_warps=num_warps)
    even = torch.arange(4096, device="mps") % 2 == 0
    torch.testing.assert_close(out, torch.where(even, x.sin(), x.cos()))
    assert "if (" in _body(k)


def test_nested_selects_predicate_the_outer_arms():
    x = torch.randn(4096, device="mps")
    out = torch.empty_like(x)
    k = nested_kernel[(4, )](x, out, BLOCK=1024)
    j = torch.arange(4096, device="mps") % 64
    ref = torch.where(j < 32, x.sin(), torch.where(j < 48, x.cos(), x.exp()))
    torch.testing.assert_close(out, ref)
    assert "if (" in _body(k)


def test_a_2d_select_predicates_per_thread():
    x = torch.randn(32, 64, device="mps")
    out = torch.empty_like(x)
    halves_2d_kernel[(1, )](x, out, R=32, C=64)
    ref = torch.cat([x[:, :32].sin(), x[:, 32:].exp()], dim=1)
    torch.testing.assert_close(out, ref)


def test_cheap_arms_stay_branch_free():
    x = torch.randn(4096, device="mps")
    out = torch.empty_like(x)
    k = cheap_kernel[(4, )](x, out, BLOCK=1024)
    j = torch.arange(4096, device="mps") % 64
    torch.testing.assert_close(out, torch.where(j < 32, x * 2 + 1, x - 3))
    assert "if (" not in _body(k)
