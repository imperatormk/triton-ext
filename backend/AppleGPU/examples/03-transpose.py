"""
Transpose
=========

In this tutorial, you will write a tiled matrix transpose and see where
threadgroup memory comes from.

A transpose reads along rows and writes along columns. One thread cannot do
both efficiently: whichever side it walks contiguously, the other side strides.
Triton solves this by staging the tile in threadgroup memory, so the read and
the write are each contiguous and the strided step happens where it is cheap.

You do not ask for that. `tl.load` and `tl.store` with swapped indices is the
whole kernel; the compiler decides the layouts need converting and puts the
tile in threadgroup memory on the way.

In doing so, you will learn about:

* Indexing a 2D tile with `tl.arange` and broadcasting.

* How a layout conversion becomes threadgroup memory without appearing in
  the source.

"""

# %%
# Compute Kernel
# --------------

import torch

import triton
import triton.language as tl

DEVICE = torch.device("mps")


@triton.jit
def transpose_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # The tile as it sits in `x`: rows of `N`.
    src = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    tile = tl.load(src)

    # The same tile in `out`, which is N by M: the row and column swap.
    dst = out_ptr + offs_n[:, None] * M + offs_m[None, :]
    tl.store(dst, tl.trans(tile))


def transpose(x: torch.Tensor):
    M, N = x.shape
    out = torch.empty((N, M), device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out


# %%
# Let's check it against torch.

torch.manual_seed(0)
x = torch.randn((512, 512), device=DEVICE, dtype=torch.float32)
out_triton = transpose(x)
out_torch = x.T.contiguous()
print(out_torch)
print(out_triton)
print(f"The maximum difference between torch and triton is "
      f"{torch.max(torch.abs(out_torch - out_triton))}")

# %%
# Benchmark
# ---------
#
# A transpose moves each element once and computes nothing, so bandwidth is
# the whole story. Torch's `.T.contiguous()` is the reference.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(7, 13, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='transpose-performance',
        args={},
    ))
def benchmark(size, provider):
    x = torch.randn((size, size), device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x.T.contiguous(),
                                                     quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose(x),
                                                     quantiles=quantiles)

    def gbps(ms):
        return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# Pass `print_data=True` to see the numbers, `show_plots=True` to plot them,
# and/or `save_path='/path/to/results/'` to save them along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
