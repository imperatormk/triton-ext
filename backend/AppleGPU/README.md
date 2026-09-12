# Apple GPU Backend for Triton

Out-of-tree Apple GPU backend for the Triton compiler, built as a triton-ext
plugin. Codegen lowers TTGIR straight to Metal Shading Language (MSL) text,
which the Metal toolchain compiles to a `.metallib`.

## Architecture

```text
triton-ext/backend/AppleGPU/
  ├── ExportAppleGPU.cpp         Plugin registration (tritonGetPluginInfo API)
  ├── lib/TritonAppleGPUTransforms/   The emit-msl pass
  ├── lib/TritonAppleGPUToMSL/   TTGIR in, agpu facts out
  ├── agpu/                      The emitter; see agpu/README.md
  └── python/
        └── triton_apple_backend/
              ├── compiler.py    TTIR → TTGIR → MSL → metallib
              ├── driver.py      MPS dispatch, buffer binding, scalar packing
              └── metal_torch.mm ObjC++ Metal bridge over torch's MPS stream
```

## Scope

`tt.get_program_id`, `tt.make_range`, `tt.splat`, `tt.addptr`, masked `tt.load`
and `tt.store`, `arith.constant` and the integer and float elementwise and
comparison operators. An op with no handler declines by name.

## Prerequisites

macOS 14+ with Xcode. Everything else (LLVM, Triton, cmake, ninja) is the
repo-wide setup in the [top-level README](../../README.md).

The build defaults `CMAKE_OSX_DEPLOYMENT_TARGET` to 14.0, unless
`-DCMAKE_OSX_DEPLOYMENT_TARGET=` or the `MACOSX_DEPLOYMENT_TARGET` environment
variable says otherwise. On a machine whose LLVM and libtriton were built for a
newer macOS the linker warns about the mismatch; either override silences it.

The emitter targets Metal 3. Per-device limits such as max threads per
threadgroup are always read back from the compiled pipeline.

## Building Triton

This backend is a plugin: it compiles against an installed Triton and links
`libtriton`, so Triton has to exist first and has to expose its symbols.

Two requirements are non-negotiable:

- **`TRITON_EXT_ENABLED=1`.** Off by default (`CMakeLists.txt` in the Triton
  tree), it is what drops `-fvisibility=hidden` and the export-list restriction
  so a plugin can resolve MLIR/LLVM symbols from the loaded `libtriton`.
  Official PyPI wheels are built with it; a source build is not.
- **An asserts build of LLVM.** `configure_triton_extension()` checks
  `LLVM_ENABLE_ABI_BREAKING_CHECKS=1`, because that flag changes the mangled
  type of every `ilist_iterator` and a mismatch shows up as undefined symbols at
  `dlopen` time, not at link time.

A worked source build, macOS, Triton `v3.8.0`:

```bash
cd /path/to/triton
export TRITON_EXT_ENABLED=1

# LLVM. Clones llvm-project/ at the hash cmake/llvm-info.json pins and builds
# it into .llvm-project/build -- note the two different directories. Roughly
# 4 GB of clone and an hour of build.
make dev-install-llvm
```

`make dev-install-llvm` goes on to run `make dev-install`, which needs a Python
with `pip`. Two things commonly stop it: `python3` resolving to an interpreter
without pip, and PEP 668 (`externally-managed-environment`). A venv settles
both:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r python/requirements.txt torch

# Point the build at the LLVM just built. Without these, setup.py silently
# downloads its own prebuilt LLVM to ~/.triton/llvm/ and links that instead --
# same pinned commit, so it works, but the local build goes unused.
export LLVM_INCLUDE_DIRS=$PWD/.llvm-project/build/include \
       LLVM_LIBRARY_DIR=$PWD/.llvm-project/build/lib \
       LLVM_SYSPATH=$PWD/.llvm-project/build

pip install -e . --no-build-isolation
```

Confirm the two requirements landed, in the generated cmake cache:

```text
TRITON_EXT_ENABLED:BOOL=1
CMAKE_BUILD_TYPE:STRING=TritonRelBuildWithAsserts
```

### Staging Triton's C++ headers (releases before the `wheel_headers` change)

`configure_triton_extension()` needs Triton's headers *inside* the installed
package, at `<site-packages>/triton/include/triton`. Triton `main` stages them
there from a `wheel_headers` install component that `setup.py` runs; the 3.8
release branch predates it, so with a 3.8.x source install the configure step
fails with a misleading *"Triton wheel is missing C++ headers ... must be built
with `TRITON_EXT_ENABLED=1`"* even though it was. Stage them by hand:

```bash
cd /path/to/triton
B=$(ls -d build/cmake.*)
cmake --install $B --component headers --prefix /tmp/triton-hdrs
# drop the build-system leftovers, and python/triton -- the install rule for
# python/ recurses into its own destination if the prefix is the package dir
rm -rf /tmp/triton-hdrs/include/CMakeFiles /tmp/triton-hdrs/include/python/triton
cp -R /tmp/triton-hdrs/include python/triton/include
```

Redo this after any Triton rebuild that changes tablegen output: the staged
`.h.inc` files go stale, and the plugin compiles against them.

## Building the backend

From the repo root, `make build` builds every extension through pip. To build
this one alone, configure it directly. Triton is located by importing it with
`Python_EXECUTABLE`; `LLVM_INSTALL_DIR` names the LLVM install or build tree,
and may come from the environment or `-D`:

```bash
cd backend/AppleGPU
cmake -S . -B build -G Ninja \
  -DPython_EXECUTABLE=/path/to/triton/.venv/bin/python \
  -DLLVM_INSTALL_DIR=/path/to/triton/.llvm-project/build
ninja -C build libapplegpu_backend.dylib
```

A Triton *build* tree works as `LLVM_INSTALL_DIR` as well as an install tree -
it has `lib/cmake/mlir`, `bin/mlir-tblgen` and
`include/llvm/Config/abi-breaking.h`, which is all the configure step reads. The
prebuilt LLVM that Triton downloads
(`~/.triton/llvm/llvm-<hash>-<platform>-<build>`) also works, and is the same
pinned commit.

This builds `libapplegpu_backend.dylib` under `build/lib/`; `__init__.py` looks
for exactly that name.

The plugin target also builds the `metal_torch` ObjC++ bridge, which links the
torch cmake found and dispatches MPS tensors zero-copy on torch's own stream;
point `-DTorch_DIR` at a different torch and it links that one. Torch is
required: kernel pointer arguments are MPS tensors and there is no other
runtime, so cmake fails rather than emit a backend that deactivates itself at
import.

### If the backend is already installed

Once `triton_apple_backend` is pip-installed, `cmake -S . -B build` can fail at
*"Could not import Triton"*. The real error is above it: the configure step runs
`ci/probe_triton_wheel.py`, which imports Triton, which walks the
`triton.backends` entry points, which imports this package. Any import error
here - a stale dylib against a rebuilt `libtriton`, say - takes the probe down
with it. Skip entry-point discovery for the probe:

```bash
TRITON_BACKENDS_IN_TREE=1 cmake -S . -B build -G Ninja ...
```

It affects only that subprocess; the resulting build is identical. `ninja` needs
no such flag.

### After a rebuild

`pip install -e .` **copies** `libapplegpu_backend.dylib` into
`<site-packages>/triton_apple_backend/`; it is not a symlink. Every `ninja`
rebuild therefore leaves that copy stale, and anything resolving the plugin
through the installed package keeps loading the old binary. Either point at the
build output,

```bash
TRITON_PLUGIN_PATHS=$PWD/build/lib/libapplegpu_backend.dylib python your_kernel.py
```

or reinstall to refresh it.

## Run

Install the package. Triton discovers the backend through its `triton.backends`
entry point, and `__init__.py` hands the bundled plugin library to
`libtriton.passes.plugin.extend_with`, so no `TRITON_PLUGIN_PATHS` is needed:

```bash
export LLVM_INSTALL_DIR=/path/to/triton/.llvm-project/build
pip install -e backend/AppleGPU --no-build-isolation --no-deps
```

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

## Debug env vars

- `MSL_ENABLE_DUMP=1` - dump the emitted MSL, the MSL counterpart of Triton's
  `MLIR_ENABLE_DUMP=1`. Emitted from the pass itself, on the same stream, and
  `MLIR_DUMP_PATH` redirects both alike.
- `TRITON_MSL_DUMP=<path>` - write the emitted MSL and TTGIR for each kernel to
  `<path>`.
- `TRITON_MSL_TRACE=1` - print the launcher's threads/group_size/args per
  dispatch.

The dump knobs are silent for a kernel already in Triton's cache, exactly as
`MLIR_ENABLE_DUMP` is: a cache hit runs no passes. Prefix with
`TRITON_ALWAYS_COMPILE=1` to force the pipeline to run.

## Tests

`pytest backend/AppleGPU/test` covers discovery, compilation and, on an Apple
GPU, a kernel end to end.

`metal_compiles` feeds `agpu/test/emit_probe.cpp`'s output to `xcrun metal` and
fails if the toolchain rejects it. Configure `agpu/` on its own, or pass
`-DAGPU_BUILD_TESTS=ON` here, then `ctest` in the build directory.

## Known limitations

- `float64` - Metal has no double, so an f64 kernel silently computes in f32.
  See `narrowsSilently` in `agpu/include/agpu/plan/ElemType.h`.
- large `num_warps` - capped per kernel by the pipeline state's
  `maxTotalThreadsPerThreadgroup`, which the backend queries and reports so
  Triton drops over-large configs. Register pressure lowers it, so the ceiling
  is kernel-dependent.
