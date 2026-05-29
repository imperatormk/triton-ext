# llvm-metal-target

An **out-of-tree fork** of `llvm/lib/Target/Metal` from the
[`imperatormk/llvm-project`](https://github.com/imperatormk/llvm-project) branch
`metal-target-poc`.

## Why this exists

Triton-main pins its LLVM at SHA `87717bf9` and ships prebuilt LLVM tarballs to
`~/.triton/llvm/llvm-87717bf9-macos-arm64/`. The triton-ext AppleGPU backend
cannot patch that pinned LLVM, but it still needs the Metal/AIR codegen target
to translate LLVM IR into a `.metallib`.

This directory builds the Metal target as a **plugin shared library**
(`libLLVMMetalTarget.dylib`) plus a thin driver (`metal-llc`) that
`triton_apple_backend/compiler.py` can invoke in place of the in-tree
`llc -mtriple=air -filetype=obj`.

## Layout

```text
llvm-metal-target/
├── CMakeLists.txt              top-level: builds against the parent's LLVM
├── README.md                   this file
├── STAGE_2_NOTES.md            known build gaps and the plan to close them
├── MIRROR_MANIFEST.txt         source files mirrored + their in-tree origin
├── lib/Target/Metal/           mirror of llvm/lib/Target/Metal
│   ├── *.cpp *.h *.td *.def
│   ├── AIRWriter/
│   ├── MCTargetDesc/
│   └── TargetInfo/
├── tools/metal-llc/            minimal llc shim (bitcode in, metallib out)
└── test/                       placeholder for lit tests
```

## Build

Built as part of triton-ext: `backend/AppleGPU/CMakeLists.txt` pulls it in via
`add_subdirectory(llvm-metal-target)`, so a normal top-level build produces the
driver at `<triton-ext>/build/bin/metal-llc`. It inherits LLVM discovery from
the parent project (the prebuilt LLVM that triton-ext already locates).

## How the AppleGPU backend consumes it

`triton_apple_backend.compiler._find_llc()` honours the `METAL_LLC_PATH` env
var; point it at the built driver to override the default lookup:

```sh
export METAL_LLC_PATH=$PWD/build/bin/metal-llc
```

A later refinement may swap the subprocess for an in-process `dlopen` of
`libLLVMMetalTarget.dylib` driven by `ctypes` (env: `METAL_TARGET_DYLIB_PATH`)
to avoid the fork/exec.

## Upstream reference

The authoritative source remains
`/Users/zimski/projects/oss/triton-main/llvm-project/llvm/lib/Target/Metal/` on
the `metal-target-poc` branch. This directory is a **fork**, not a mirror:
expect divergence as we work around the gaps in Triton's pinned LLVM (see
`STAGE_2_NOTES.md`).

## Sync workflow

When the in-tree branch moves forward:

1. `diff -ru` the in-tree `Metal/` against `lib/Target/Metal/` here using
   `MIRROR_MANIFEST.txt` as the file list.
1. Cherry-pick changes that don't touch the Triple::air / MetalLib hooks.
1. Re-apply the out-of-tree shims listed in `STAGE_2_NOTES.md` if the in-tree
   files now reference newly-added APIs.
1. Bump the timestamps + sizes in `MIRROR_MANIFEST.txt`.
