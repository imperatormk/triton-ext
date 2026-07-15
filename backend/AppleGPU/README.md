# Apple GPU Backend for Triton

Out-of-tree Apple GPU backend for the Triton compiler, built as a triton-ext
plugin. Codegen lowers TTGIR straight to Metal Shading Language (MSL) text,
which is compiled to a `.metallib` in-process via the Metal framework.

## Architecture

```text
triton-ext/backend/AppleGPU/
  ├── ExportAppleGPU.cpp         Plugin registration (tritonGetPluginInfo API)
  ├── CMakeLists.txt + lib/      C++ MLIR passes (incl. the EmitMSL emitter)
  └── python/                    Python backend (pip installable)
        └── triton_apple_backend/
              ├── compiler.py    TTIR → TTGIR → MSL → metallib
              ├── driver.py      MPS GPU dispatch, buffer binding, scalar packing
              └── metal_utils.m  ObjC++ Metal bridge (compiled at install time)
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

ninja -C build libapplegpu_backend.dylib
```

This builds `libapplegpu_backend.dylib` (or `.so`) under `build/lib/`. cmake
configures every extension (`dialect pass backend language extensions`); naming
the ninja target keeps the build to AppleGPU only.

### 4. Run

Set the environment (run from the repo root, so `$PWD` resolves the build). The
dylib in `TRITON_PLUGIN_PATHS` and the `triton_apple_backend` on `PYTHONPATH`
must come from the same build.

```bash
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

| Pass                       | Purpose                                                     |
| -------------------------- | ----------------------------------------------------------- |
| `add_accelerate_matmul`    | Rewrite tt.dot → AppleMmaEncoding (simdgroup MMA)           |
| `add_simplify_gather`      | Strip efficient_layout from large gathers (Metal JIT limit) |
| `add_store_shuffle_layout` | Re-lay MMA epilogue stores as within-simdgroup shuffles     |
| `add_emit_msl`             | Emit MSL text directly from TTGIR (terminal codegen)        |

### TritonAppleGPU Dialect

- `AppleMmaEncodingAttr` — 8x8 simdgroup matrix multiply encoding

## Debug env vars

- `TRITON_MSL_DUMP=<path>` — write the emitted MSL for each kernel to `<path>`.
- `TRITON_MPS_DEBUG=1` — print the emitted MSL to stdout.
- `TRITON_MSL_OUT` — internal handoff for the MSL output path between the C++
  emitter and the Python compiler; set automatically, not a user knob.

## Test Status

`test_core.py` passes on the MPS device; the residual failures are the known
limitations below (float64, fp8, 64-bit atomics, tf32/bf16xN dot precision,
acquire/release atomics). Tests that assert LLVM-IR markers (`llvm.assume`,
`llvm.licm.disable`, poison) also fail: the MSL path emits no LLVM IR, so those
inspections are not applicable.

## Known Limitations

- `float64` — MPS has no double precision.
- `float8` (e4m3, e5m2) — the torch MPS backend has no fp8 dtype, and the
  emitter has no fp8 conversion path yet. A capability gap, not a hardware wall.
- `int64`/`uint64` atomics — Metal has no 64-bit device atomics.
- acquire/release atomics — Metal device atomics are relaxed-only; ordered
  variants lower to relaxed.
- `tf32` / `bf16xN` dot input precision — split-precision emulation modes with
  no Metal equivalent (rejected at compile).
- `num_warps >= 16` — exceeds the Apple GPU max threads per threadgroup (384).
- Cross-threadgroup spinlocks — Apple GPU has no forward-progress guarantee.
