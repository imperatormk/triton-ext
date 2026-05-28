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
git clone https://github.com/triton-lang/triton.git ~/projects/oss/triton-main
cd ~/projects/oss/triton-main

# Create venv
python3.12 -m venv pytorch25-venv
source pytorch25-venv/bin/activate
pip install pybind11 numpy pytest

# macOS patches needed before building:
# - third_party/nvidia/CMakeLists.txt: skip GSan CUDA on Apple (stub gsan.ll)
# - CMakeLists.txt: skip examples/plugins on Apple (visibility link errors)

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

### 3. Build metal-ir-pipeline

```bash
cd backend/AppleGPU/metal-ir-pipeline
cmake -B build -DLLVM_DIR=~/projects/oss/triton-main/llvm-project/build/lib/cmake/llvm
cmake --build build -j$(sysctl -n hw.ncpu)
cd ../../..
```

### 4. Build the plugin

```bash
mkdir build && cd build
TRITON_INSTALL_DIR=~/projects/oss/triton-main/build/install \
cmake .. \
  -DLLVM_DIR=~/projects/oss/triton-main/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=~/projects/oss/triton-main/llvm-project/build/lib/cmake/mlir \
  -DFILECHECK_PATH=~/projects/oss/triton-main/llvm-project/build/bin/FileCheck
make -j$(sysctl -n hw.ncpu)
```

### 5. Install the Python backend

```bash
cd backend/AppleGPU/python
pip install -e . --no-build-isolation
```

### 6. Run

```bash
export TRITON_PLUGIN_PATHS=~/projects/oss/triton-ext/build/lib/libTritonAppleGPUBackend.dylib

python your_triton_script.py  # kernels run on MPS
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
