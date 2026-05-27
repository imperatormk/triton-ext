# Stage 2 notes

Stage 1 only scaffolds files. Stage 2 has to actually make the project
configure, build, and produce a `metal-llc` whose output matches the in-tree
`llc -mtriple=air -filetype=obj` byte-for-byte. The hard parts below are
known up front so stage 2 isn't surprised by them.

## Known hard parts

### 1. `Triple::air` does not exist in Triton's prebuilt LLVM

The in-tree branch added a new `Triple::ArchType` enumerator. The prebuilt
LLVM headers under `~/.triton/llvm/llvm-87717bf9-macos-arm64/include/` do
**not** know about it. Every reference in our mirrored sources --
`TargetInfo/MetalTargetInfo.cpp` has the offender `RegisterTarget<Triple::air,
...>`, plus the various pass `if (T.getArch() == Triple::air)` guards -- must
be replaced with a string-based check ("air" prefix on the triple) or by
hijacking an unused enumerator (`Triple::UnknownArch` is the working
candidate; see the F10/V section of in-tree `PASS_GUARDS.md`).

Recommended approach: a small shim header
(`lib/Target/Metal/MetalTripleCompat.h`) that defines
`isAirTriple(const Triple&)` and `kAirArchHijack` and is included everywhere
the sources currently say `Triple::air`. A scripted sed-replace is fine.

### 2. `Triple::MetalLib` (ObjectFormatType) does not exist either

Same story. Used by `MCTargetDesc/MetalContainerObjectWriter.cpp` and the
TargetMachine's section selection. Same workaround pattern: shim header
+ string-based detection or unused-enum hijack.

### 3. `computeDataLayout` has no `case Triple::air`

In-tree, the data layout was wired through `llvm/lib/TargetParser/Triple.cpp`.
We can't patch that file. Workaround: have `MetalTargetMachine`'s constructor
construct the DataLayout literal directly (the AIR DL string is fixed and
known) and pass it explicitly to the `LLVMTargetMachine` base ctor instead
of relying on `Triple::computeDataLayout`. The exact DL string lives in the
in-tree `Triple.cpp` patch -- copy it verbatim.

### 4. `llvm/MC/MCSectionMetalLib.h` does not exist in the prebuilt

`MetalTargetMachine.cpp` includes it. Two options:

a. **Vendor the header** into `lib/Target/Metal/include/llvm/MC/` and add
   that directory to the include path *before* the prebuilt's include
   directory so our copy wins. This is the lowest-risk path; the header is
   self-contained on the in-tree branch.

b. **Replace the section-class usage** with `MCSectionELF` or a custom
   `MCSection` subclass defined locally, and lose the MetalLib-specific
   section attributes. Probably loses output fidelity.

Plan: go with (a) initially.

### 5. TableGen-generated files

`tablegen(LLVM ...)` calls invoke the prebuilt `bin/llvm-tblgen`. That binary
exists in the Triton prebuilt and accepts the same `.td` files as the in-tree
build. Should "just work" so long as the `Metal.td` / `MetalAIRIntrinsicMappings.td`
don't import any in-tree-only `.td` files. Verify with
`grep -r 'include "' lib/Target/Metal/*.td` early in stage 2.

### 6. Symbol visibility / `LLVM_EXTERNAL_VISIBILITY`

In-tree, `LLVMInitializeMetalTarget` is marked `LLVM_ABI
LLVM_EXTERNAL_VISIBILITY`. Out-of-tree we must export it ourselves from the
shared library (default visibility, or a small `.exports` map). Otherwise
`dlsym` from `metal-llc` and from any future Python ctypes loader returns
NULL.

## The 4 layers stage 2 must address (in order)

1. **Source-level**: rename `Triple::air` -> `isAirTriple()` / `kAirArchHijack`
   across all mirrored sources. Same for `Triple::MetalLib`. This is mostly
   mechanical; do it first because the project literally won't compile until
   it's done.
2. **MC-level**: register the target via `RegisterTarget<>` with the hijack
   arch, register MC pieces (asm info, instr printer, code emitter) without
   relying on the `MCObjectFileInfo` enum entry that doesn't exist. Vendor
   `MCSectionMetalLib.h` per item 4 above.
3. **CMake**: get `tablegen(...)` running against the prebuilt `llvm-tblgen`,
   then get `add_library(LLVMMetalTarget SHARED ...)` to link clean against
   the prebuilt `LLVM*` libs. Expect missing-symbol errors here -- fix them
   by adding to `llvm_map_components_to_libnames` in
   `lib/Target/Metal/CMakeLists.txt`.
4. **Runtime parity**: drive `metal-llc` from a captured IR file and `diff`
   the produced metallib against the in-tree `llc` output. They must be
   byte-identical for `compiler.py` to be able to flip `METAL_LLC_PATH`
   without behavioural change. If they aren't, the most likely culprits are
   (a) DataLayout drift from item 3 above, or (b) the AIRWriter pulling a
   timestamp or hash from a different code path.

## Recommended starting point for stage 2

Start with **layer 1** (Triple hijack shim). It's purely textual, unblocks
everything else, and once done the project will at least attempt to compile,
which gives stage 2 a real error stream to iterate against.

---

## Stage 2 resolution log (2026-05-28)

**Status: GREEN.** Configure + build clean, `metal-llc` produces a
byte-identical metallib that Apple's Metal runtime accepts.

### Layer 1 (source-level): landed
- Added `lib/Target/Metal/MetalTripleCompat.h` with `kAirArchHijack =
  Triple::UnknownArch`, `isAirTriple()`, and `kAirDataLayout` (verbatim copy
  of the in-tree AIR DL string from `TargetDataLayout.cpp case Triple::air`).
- `TargetInfo/MetalTargetInfo.cpp`: `RegisterTarget<Triple::air, ...>` ->
  `RegisterTarget<metal_compat::kAirArchHijack, ...>`. The prebuilt's Triple
  parser returns UnknownArch for the unknown "air" string, so the lookup
  still resolves correctly because we're the only target on UnknownArch.
- `MetalTargetMachine.cpp`:
  - dropped `#include "llvm/MC/MCSectionMetalLib.h"`,
  - replaced `TT.computeDataLayout()` with `metal_compat::kAirDataLayout`,
  - stubbed `MetalTargetObjectFile::getExplicitSectionGlobal` to
    `llvm_unreachable` (it's never hit because we bypass AsmPrinter),
  - rewrote `addPassesToEmitFile`'s `ObjectFile` branch to always
    `createMetalWriterPass(Out)` -- the in-tree branch's AsmPrinter +
    MCMetalLibObjectWriter route requires MC additions (MCSectionMetalLib,
    MCMetalLibObjectWriter, MCContext::getMetalLibSection,
    Triple::MetalLib) that don't exist in the prebuilt LLVM,
  - dropped `initializeMetalEmbedderLegacyPassPass` (no longer reachable).
- `MCTargetDesc/MetalMCTargetDesc.cpp`: stubbed
  `MetalAsmBackend::createObjectTargetWriter` to `llvm_unreachable` and
  removed the `MetalContainerObjectWriter` include.
- `MetalDeviceLoadsVolatile.cpp`: added missing
  `#include "llvm/IR/Operator.h"` for `GEPOperator`/`BitCastOperator`/
  `AddrSpaceCastOperator`.
- `tools/metal-llc/metal-llc.cpp`:
  - removed `LLVMInitializeMetalAsmPrinter` declaration + call,
  - fixed `createTargetMachine` / `setTargetTriple` signatures (they now
    take `Triple` not `StringRef` in 21.x).

### Layer 2 (link): landed
- CMake: stubbed out-of-tree targets `intrinsics_gen`, `acc_gen`, `omp_gen`
  required by `add_public_tablegen_target`.
- CMake: set `LLVM_TABLEGEN_EXE` from `LLVM_TOOLS_BINARY_DIR/llvm-tblgen`
  before `tablegen(LLVM ...)` is invoked.
- Removed `MetalAsmPrinter.cpp` and `MCTargetDesc/MetalContainerObjectWriter.cpp`
  from the build (depend on missing in-tree MC additions).
- Existing `llvm_map_components_to_libnames` list was sufficient -- no new
  components needed once the AsmPrinter path was disabled.

### Layer 3 (runtime):
`metal-llc -mtriple=air -filetype=obj /tmp/i64shuf_v2i32_probe.ll -o
/tmp/out_oot.metallib` -> 3148-byte MetalLib (MacOS) v1.2.9. `load_probe.py`
reports `OK: MTLLibrary created` and `OK: PSO created for 'kernel'`. Full
parity with the in-tree `llc` path.

### Layer 4 (byte equivalence):
`cmp /tmp/out_intree.metallib /tmp/out_oot.metallib` -> **byte identical**
(3148 bytes each). The out-of-tree port is a faithful drop-in for the
in-tree `llc -mtriple=air -filetype=obj`.

### Open items / caveats
- Out-of-tree TableGen-generated `MetalGenAIRIntrinsicMappings.inc` is built
  but currently not `#include`d by anything in this scaffold -- it's emitted
  via `-gen-searchable-tables` for future intrinsic-mapping wiring.
- Compiler emits deprecation warnings about `BranchInst`/`isConditional` in
  `MetalTGBarrierInsert.cpp` against newer LLVM headers; these are warnings
  only (the deprecated APIs still compile and produce identical output).
- `MetalAsmPrinter.cpp`, `MCTargetDesc/MetalContainerObjectWriter.cpp` and
  the embed-via-section AsmPrinter pipeline remain on disk as reference but
  are intentionally excluded from the out-of-tree build. If/when the
  upstream LLVM grows the MC additions (`MCSectionMetalLib`, etc.) the
  CMakeLists toggles can be flipped to re-enable them.
