# Apple GPU Backend for Triton

Apple GPU (Metal) backend for Triton, built as a triton-ext plugin.

## Structure

```text
backend/AppleGPU/
  ├── ExportAppleGPU.cpp         Plugin registration (tritonGetPluginInfo API)
  ├── CMakeLists.txt             C++ MLIR plugin build
  ├── include/                   Dialect + pass headers (.td, .h) + metal-ir/
  ├── lib/
  │     ├── TritonAppleGPUToLLVM/  TritonGPU → LLVM IR lowering (Apple-specific)
  │     └── MetalIR/               LLVM IR → Metal AIR → metallib compiler
  │           ├── Transforms/        AIR-conformance passes
  │           ├── Serialization/     typed-pointer bitcode + metallib writer
  │           └── Bridge/            libMetalIRBridge C entrypoint
  └── python/                    Python backend (pip installable)
        └── triton_apple_backend/
              ├── compiler.py    TTIR → TTGIR → LLVM IR → metallib
              ├── driver.py      MPS GPU dispatch + buffer binding
              └── metal_utils.m  ObjC++ Metal bridge (compiled at install time)
```

## Prerequisites

- macOS 14+ with Xcode (Metal framework + clang)
- Python 3.10+
- CMake

## Build

### 1. Build metal-ir-pipeline

```bash
cd backend/AppleGPU/metal-ir-pipeline
cmake -B build -DLLVM_DIR=~/projects/oss/triton-main/llvm-project/build/lib/cmake/llvm
cmake --build build -j$(sysctl -n hw.ncpu)
cd ../../..
```

### 2. Build the plugin

```bash
mkdir build && cd build
TRITON_INSTALL_DIR=~/projects/oss/triton-main/build/install \
cmake .. \
  -DLLVM_DIR=~/projects/oss/triton-main/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=~/projects/oss/triton-main/llvm-project/build/lib/cmake/mlir \
  -DFILECHECK_PATH=~/projects/oss/triton-main/llvm-project/build/bin/FileCheck
make -j$(sysctl -n hw.ncpu) TritonAppleGPUBackend
```

### 3. Install the Python backend

```bash
cd backend/AppleGPU/python
pip install -e . --no-build-isolation
```

### 4. Run vector_add

```bash
export TRITON_PLUGIN_PATHS=$PWD/../../../build/lib/libTritonAppleGPUBackend.dylib
python3 -c '
import torch, triton, triton.language as tl
@triton.jit
def add(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))
a = torch.randn(64, device="mps")
b = torch.randn(64, device="mps")
c = torch.empty(64, device="mps")
add[(1,)](a, b, c, 64, num_warps=1)
torch.mps.synchronize()
print("add:", torch.allclose(c, a+b))
'
```

## MLIR passes

| Pass                             | Purpose                                 |
| -------------------------------- | --------------------------------------- |
| `add_to_llvmir`                  | Lower TritonGPU ops to LLVM IR          |
| `add_lower_gpu_to_air`           | Lower `gpu.thread_id` / `gpu.block_dim` |
| `add_reconcile_unrealized_casts` | Clean up leftover conversion casts      |

## metal-ir-pipeline passes

Transforms LLVM IR into Metal-compatible AIR bitcode:

- `InlineNonKernel` — inline all non-kernel functions (Metal has no call stack)
- `DecomposeStructPhis` — split struct phi nodes to scalar phis
- `PtrPhiToI64` / `PtrSelectToI64` — convert pointer phis/selects through i64
- `LLVMToAIRIntrinsics` — rename LLVM intrinsics to AIR equivalents
- `InferTypedPointers` — reconstruct typed pointers (Metal requires typed ptrs)
- `NormalizeAllocas` / `NormalizeI1Pointers` — pre-serialization normalization
- `ScalarBufferPacking` — pack scalar kernel args into a constant buffer
- `AIRSystemValues` — inject AIR kernel metadata
- `DeviceLoadsVolatile` — mark device loads volatile

The final output is serialized directly to a `.metallib` container.
