# llvm-metal-target

An **out-of-tree fork** of an LLVM AIR (Apple GPU) codegen target, kept in sync
with its canonical twin (the `AIR` target on the `metal-target-poc` llvm-project
branch).

## Why this exists

Triton pins its LLVM to a specific commit and ships prebuilt LLVM tarballs under
`~/.triton/llvm/`. The triton-ext AppleGPU backend cannot patch that pinned
LLVM, but it still needs an AIR codegen target to translate LLVM IR into a
`.metallib`.

This directory builds that target as a **plugin shared library**
(`libLLVMMetalTarget.dylib`) plus a thin driver (`metal-llc`, in/out: LLVM
bitcode to metallib) that `triton_apple_backend/compiler.py` invokes via
`subprocess`.

## Layout

```text
llvm-metal-target/
├── CMakeLists.txt              top-level: find_package(LLVM CONFIG)
├── lib/Target/Metal/           the AIR codegen target
│   ├── *.cpp *.h *.td *.def    TargetMachine + IR-legalization passes
│   ├── AIRWriter/              LLVM IR to AIR bitcode writer (typed-pointer
│   │                           reconstruction, metadata, metallib container)
│   ├── MCTargetDesc/
│   └── TargetInfo/
└── test/                       lit tests (FileCheck against emitted IR)
```

The writer (`AIRWriter/`) is the substantial part: it reconstructs the typed
POINTER records and `!air.*` kernel metadata the AGX driver requires, and
serializes the metallib. Its file structure is kept symmetric with the twin so
changes transfer between the two.

## Build

Built as part of the parent triton-ext build
(`ninja -C build metal-llc libapplegpu_backend.dylib` from the repo root; see
`backend/AppleGPU/README.md`). To build this target standalone against any LLVM
install:

```sh
cmake -B build -S . -DLLVM_DIR=<llvm install>/lib/cmake/llvm
cmake --build build -j
```

`LLVM_DIR` defaults to Triton's prebuilt LLVM under `~/.triton/llvm/`.

## How the AppleGPU backend consumes it

`triton_apple_backend.compiler._find_llc()` honours the `METAL_LLC_PATH` env
var, falling back to `build/bin/metal-llc`:

```sh
export METAL_LLC_PATH=$PWD/build/bin/metal-llc
```

## Upstream twin

The canonical source is the `AIR` codegen target on the `metal-target-poc`
llvm-project branch (`llvm/lib/Target/AIR/`, files prefixed `AIR*`). This
`Metal*`-prefixed directory stays symmetric with it: writer/transform fixes land
in both. It is a fork, not a mirror; expect deliberate divergence where we work
around the gaps in Triton's pinned LLVM.
