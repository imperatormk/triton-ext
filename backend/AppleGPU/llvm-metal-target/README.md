# llvm-metal-target

An out-of-tree LLVM Metal/AIR codegen target, built against the prebuilt LLVM
that triton-ext already provides. It produces `libLLVMMetalTarget` plus a thin
`metal-llc` driver that lowers LLVM IR to a `.metallib`.

## Why this exists

triton-ext pins LLVM and ships prebuilt LLVM tarballs that an extension cannot
patch in place. This directory builds the Metal target as a separate library
plus driver instead, so the backend can lower LLVM IR to a `.metallib` without
modifying the pinned LLVM.

## Layout

```text
llvm-metal-target/
├── CMakeLists.txt              top-level: find_package(LLVM CONFIG)
├── README.md                   this file
├── lib/Target/Metal/           the Metal/AIR target
│   ├── *.cpp *.h *.td *.def
│   ├── AIRWriter/
│   ├── MCTargetDesc/
│   └── TargetInfo/
├── tools/metal-llc/            llc-style driver (LLVM IR in, metallib out)
└── test/                       lit tests driving metal-llc
```

## Build

Built automatically as a subdirectory when triton-ext is built top-level; it
inherits LLVM discovery from the parent project.

To iterate on the target alone, build it standalone:

```sh
cmake -B build -S . \
  -DLLVM_DIR=<path-to>/lib/cmake/llvm
cmake --build build -j
```

`LLVM_DIR` defaults to triton-ext's prebuilt LLVM. Override it to build against
another LLVM install.

## How the backend consumes it

`metal-llc` lands at `build/bin/metal-llc`. The backend reads `METAL_LLC_PATH`;
point it at that binary:

```sh
export METAL_LLC_PATH=<build>/bin/metal-llc
```

## Tests

The lit tests under `test/` drive `metal-llc` directly and check its output with
`FileCheck`. They run as part of the triton-ext lit suite (`make test`), and
skip themselves if `metal-llc` has not been built.
