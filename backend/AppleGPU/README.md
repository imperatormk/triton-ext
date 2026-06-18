# Apple GPU Backend for Triton

Out-of-tree Apple GPU (Metal) backend for the Triton compiler, built as a
triton-ext plugin.

## Architecture

```text
triton-ext/backend/AppleGPU/
  ├── ExportAppleGPU.cpp         Plugin registration (tritonGetPluginInfo API)
  ├── CMakeLists.txt + lib/      C++ MLIR passes
  ├── python/                    Python backend (pip installable)
  │     └── triton_apple_backend/
  │           ├── compiler.py    TTIR → TTGIR → LLVM IR → metallib
  │           ├── driver.py      MPS GPU dispatch, buffer binding, scalar packing
  │           └── metal_utils.m  ObjC++ Metal bridge (compiled at install time)
  └── llvm-metal-target/         LLVM IR → Metal AIR → metallib compiler
```

## Prerequisites

- macOS 26+ with Xcode (Metal framework + clang)
- Python 3.10+
- CMake + Ninja

## Setup

### 1. Build Triton from source

```bash
git clone https://github.com/triton-lang/triton.git
cd triton

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install pybind11 numpy pytest

# Build LLVM (first time only, ~40 min)
cd llvm-project
git checkout $(cat ../cmake/llvm-hash.txt)
cmake -B build -G Ninja llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
cd ..

# Build & install Triton
LLVM_SYSPATH=$(pwd)/llvm-project/build TRITON_BUILD_WITH_CCACHE=true \
  TRITON_EXT_ENABLED=1 pip install -e . --no-build-isolation
```

### 2. Clone triton-ext with submodules

```bash
git clone --recurse-submodules https://github.com/imperatormk/triton-ext.git
cd triton-ext
git checkout apple-gpu
```

### 3. Build the plugin

Configure with cmake, pointing at your Triton + LLVM trees, then build this
extension's targets:

```bash
cmake -S . -B build -G Ninja \
  -DTRITON_SOURCE_DIR=<triton repo>  \
  -DTRITON_BUILD_DIR=<triton repo>/build/cmake.<...>  \
  -DTRITON_LIB=<triton repo>/python/triton/_C/libtriton.so \
  -DLLVM_TABLEGEN_EXE=<llvm-project>/build/bin/llvm-tblgen
# (export LLVM_INSTALL_DIR=<pinned llvm dir> so MLIR/LLVM cmake packages resolve)

ninja -C build metal-llc libapplegpu_backend.dylib
```

This builds `libapplegpu_backend.dylib` (or `.so`) under `build/lib/` and the
`metal-llc` binary under `build/bin/`. cmake configures every extension
(`dialect pass backend language extensions`); naming the ninja targets keeps the
build to AppleGPU only.

### 4. Run

Set the environment (run from the repo root, so `$PWD` resolves the build). The
dylib in `TRITON_PLUGIN_PATHS` and the `triton_apple_backend` on `PYTHONPATH`
must come from the same build.

```bash
export METAL_LLC_PATH=$PWD/build/bin/metal-llc
export TRITON_PLUGIN_PATHS=$PWD/build/lib/libapplegpu_backend.dylib
export TRITON_PASS_PLUGIN_PATH=$TRITON_PLUGIN_PATHS
export PYTHONPATH=$PWD/backend/AppleGPU/python
```

Then run:

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 4096
x = torch.randn(n, device="mps")
y = torch.randn(n, device="mps")
out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
torch.mps.synchronize()
assert (out - (x + y)).abs().max().item() == 0.0
print("vecadd ok")
```

## What's included

### C++ MLIR Passes (loaded via TRITON_PLUGIN_PATHS)

| Pass                             | Purpose                                                     |
| -------------------------------- | ----------------------------------------------------------- |
| `add_accelerate_matmul`          | Rewrite tt.dot → AppleMmaEncoding (simdgroup MMA)           |
| `add_simplify_gather`            | Strip efficient_layout from large gathers (Metal JIT limit) |
| `add_to_llvmir`                  | Lower AppleMmaEncoding → LLVM IR with simdgroup intrinsics  |
| `add_lower_gpu_to_air`           | Lower gpu.thread_id/block_dim → AIR intrinsics              |
| `add_reconcile_unrealized_casts` | Clean up leftover conversion casts                          |

### TritonAppleGPU Dialect

- `AppleMmaEncodingAttr` — 8x8 simdgroup matrix multiply encoding

## Test Status

- 71/72 backend-specific tests passing (1 xfail for shared memory limit)
- Upstream test_core.py: remaining failures are float64, FP8, int64 atomics,
  NVIDIA-specific tests
- 3 backend bugs tracked: phi(undef,ptr) crash, LICM metadata, multi-return
  noinline

## Known Limitations

- `float64` — MPS doesn't support double precision
- `float8` (e4m3, e5m2) — NVIDIA-specific hardware types
- `int64` atomics — Metal doesn't support 64-bit atomic operations
- `num_warps >= 16` — exceeds Apple GPU max threads per threadgroup (384)
- Cross-threadgroup spinlocks — Apple GPU has no forward progress guarantee
