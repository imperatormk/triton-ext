# `tt.func` TypeID mismatch under `triton-opt` (minimal repro)

This is a minimal reproduction of why an out-of-tree extension *pass* cannot
operate on Triton ops under `triton-opt`: the op `triton-opt` parses carries a
different MLIR `TypeID` than the one the plugin computes, so
`isa`/`cast<triton::FuncOp>` fails.

It complements the dialect/type-loading work already landed (#9783): a plugin
can now *load* and register dialects under `triton-opt`, but a pass that does
`cast<triton::FuncOp>` (as every real conversion pattern does) still aborts.

## Build

It builds as a normal `triton-ext` pass plugin (auto-discovered from
`pass/TypeIdReproPass/triton-ext.toml`):

```
make build LLVM_INSTALL_DIR=<llvm-artifact> TRITON_INSTALL_DIR=<triton-artifact>
# produces build/lib/libtypeid_repro.so
```

## Run

`triton-opt` is not a Python process, so `libtriton.so`'s undefined CPython
symbols have to be satisfied first; preloading the matching `libpython` does
that (this is the load-side problem #9783 addresses). Then:

```
LD_PRELOAD=<libpython3.x.so> \
LD_LIBRARY_PATH=<triton-artifact>/lib:<llvm-artifact>/lib \
TRITON_PLUGIN_PATHS=$PWD/build/lib/libtypeid_repro.so \
  <triton-artifact>/bin/triton-opt pass/TypeIdReproPass/func.mlir --typeid-repro
```

Output:

```
tt.func registered TypeID (triton-opt's copy): 0x62c18fd49df0
TypeID::get<triton::FuncOp>() (this plugin's copy): 0x7e27962c3180
isa<triton::FuncOp>(op): 0
MISMATCH: the two copies are different symbols; any pattern doing cast<triton::FuncOp> will abort.
```

(The pass prints rather than aborts; uncomment the `cast<triton::FuncOp>` line
in `TypeIdReproPass.cpp` to see the assertion fire.)

## Why this happens

`triton-opt` statically links the Triton dialect (the `${triton_libs}` object
archives) and registers its ops with self-owning `TypeID` symbols baked into
the executable. The plugin links `libtriton.so`, which carries its *own* copy
of those `TypeID` symbols. MLIR identity is symbol-address equality, so the two
copies are never equal — hence the mismatch above.

## Why it can't be fixed from the extension side

Each of these was tried against the shipped artifacts and fails for a concrete,
structural reason:

1. **Preload `libtriton.so`** so `triton-opt` binds to it.
   `triton-opt`'s `triton::FuncOp::id` is a *defined, non-preemptible* symbol in
   the executable. The dynamic loader resolves the executable's own copy first;
   a preload cannot override it.

2. **Link the plugin against the same static archive `triton-opt` used.**
   The Triton artifact ships only `libtriton.so` — there are no `.a` archives.
   The objects `triton-opt` statically linked are not distributed, so the
   plugin cannot share them.

3. **Drop `libtriton.so` from the plugin and resolve Triton symbols from
   `triton-opt` instead.** `triton-opt` exports essentially none of the C++ API
   a plugin needs (it exports a registration surface, not the dialect/IR API):
   of this plugin's undefined Triton/MLIR symbols, `triton-opt` exports 0. A
   real conversion pass needs far more of that API, none of which is available,
   so the plugin would not link or load against `triton-opt` alone.

4. **Use string-fallback TypeIDs (`MLIR_*_FALLBACK_TYPE_ID`) in the plugin** so
   identity is by type name via the shared `libMLIRSupport` registry.
   `triton-opt` registered its ops with *self-owning* TypeIDs
   (`MLIR_DEFINE_EXPLICIT_TYPE_ID`), which never enter that string registry, so
   a fallback resolver in the plugin still computes a different value.

The common root is that `triton-opt` owns a private copy of the Triton dialect
that it neither shares (no distributed archive, no dynamic `libtriton` link) nor
exports (registration surface only). Matching its op identity requires changing
how `triton-opt` is built — exporting its symbols and/or linking the same
`libtriton` the plugin does. That is the half of the original approach (the
`triton-opt`-side of PR #9550) that was not part of #9783.

## Files

- `TypeIdReproPass.cpp` — the pass (prints the two TypeIDs + `isa` result).
- `func.mlir` — a one-function `tt.func` module.
- `triton-ext.toml`, `CMakeLists.txt` — standard plugin build wiring.
