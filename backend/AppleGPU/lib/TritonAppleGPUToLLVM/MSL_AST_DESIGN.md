# MSL AST rewrite — architecture spec

Rewrite of `EmitMSL.cpp`'s text-emission engine from raw `os << ind() << "..."`
string-slinging into a typed MSL AST + a printer that owns all scope, brace, and
indentation emission. Branch: `msl-ast-rewrite` (big-bang; merges to `msl-mlir`
only when the full corpus is green).

## Why (not EmitC, not an RAII wrapper)

The load-bearing ~60% of what EmitMSL emits — address-space-qualified pointers
(`device`/`threadgroup`/`constant`), `simdgroup_*8x8` matrix types + intrinsics,
`atomic_*` types with `memory_order_*`/`mem_flags::*`, `[[buffer(N)]]` /
`[[thread_position_in_threadgroup]]` param attributes, `as_type<T>` template
casts, `coherent(device)`, `uint3` — has no faithful EmitC representation
(EmitC's `emitc.ptr` has no address-space channel and `printFunctionArgs` drops
`arg_attrs`). In EmitC these all degrade to `emitc.opaque`/`verbatim` strings:
the same untyped text as today, wrapped in MLIR ops. Forking EmitC's printer
would work but permanently taxes the twin-repo LLVM-fork rebase seam. So: a
standalone typed `msl::` AST that models these first-class, plus a printer that
owns indent + the barrier-flush invariant, so no lowering method ever writes a
`{`, touches an indent counter, or forgets a barrier flush again.

## Files

- `MSLAst.h` / `MSLAst.cpp` — the node set (types, exprs, stmts, decls) + arena.
- `MSLPrinter.h` / `MSLPrinter.cpp` — walks the AST, owns indent + scope braces +
  barrier peephole, produces the final MSL text. THE ONLY place `{`/`}`/indent
  is emitted.
- `EmitMSL.cpp` — becomes the TTGIR->AST lowering (`emitXxx` build nodes, never
  strings). Keeps all IR-analysis state (valMap, LinearLayout helpers, pool
  accounting, fused-dot matchers, memdescMap).

## Node set (fully typed — "model everything")

Nodes are arena-allocated (bump allocator on the printer/emitter context; AST is
build-once, print-once, never mutated after build). Pointers are stable; no
ownership juggling. All nodes carry a kind enum for the printer's switch.

### Types (`msl::Type`)
- `ScalarType{ enum Kind: F32,F16,BF16,I1,I8,I16,I32,I64,U8,U16,U32,U64,Void,SizeT }`
- `VectorType{ ScalarType elem; unsigned n; }`            // uint3, uint2
- `MatrixType{ enum Elem: Half,Bfloat,Float; }`           // simdgroup_<e>8x8
- `AtomicType{ ScalarType elem; }`                        // atomic_int/uint/float/long
- `PointerType{ Type* pointee; AddrSpace as; bool coherent; bool volatile_; }`
  - `AddrSpace: None, Device, Threadgroup, Constant, Thread`
- `NamedType{ StringRef name; }`                          // return structs, opaque
- Printing lives entirely in `MSLPrinter::printType` — AS qualifier prefix,
  `*`, `coherent(device)`, `volatile` all decided here.

### Expressions (`msl::Expr`) — value-producing, printed inline
- `VarRef{ StringRef name; }`                             // an SSA var name from valMap
- `Literal{ StringRef text; }`                            // %.17g, INFINITY, NAN, nullptr, ints
- `Binary{ enum Op; Expr* lhs; Expr* rhs; }`              // + - * / % & | ^ << >> < <= ... == !=
- `Unary{ enum Op; Expr* x; }`                            // -x, !x, (bool)((x)&1)
- `Cast{ enum Style: CStyle,Static,AsType; Type* to; Expr* x; }` // (T)x / static_cast<T>(x) / as_type<T>(x)
- `Call{ StringRef callee; SmallVector<Type*> templateArgs; SmallVector<Expr*> args; }`
  - covers metal::precise::*, simd_shuffle*, atomic_*_explicit, min/max/fma,
    simdgroup_load/multiply_accumulate/store, tt_erf, ...
- `Ternary{ Expr* c; Expr* a; Expr* b; }`                 // c ? a : b
- `Subscript{ Expr* base; Expr* idx; }`                   // p[i]
- `Member{ Expr* base; StringRef field; }`                // s.f0
- `Deref{ Expr* x; }` / `AddrOf{ Expr* x; }`
- `Paren{ Expr* x; }`                                     // explicit grouping when needed
- `Raw{ StringRef text; }`                                // ESCAPE HATCH: the fp-narrowing
  bit-twiddling blocks + tt_erf preamble body. Typed-core + verbatim per the
  design call, but confined to genuinely-bespoke text; NOT a general crutch.

### Statements (`msl::Stmt`) — the printer owns their layout
- `DeclStmt{ Type* ty; StringRef name; Expr* init /*nullable*/; }`  // T v = e;  or  T v;
- `ArrayDeclStmt{ Type* elem; StringRef name; unsigned n; SmallVector<Expr*> init; }`
- `AssignStmt{ Expr* lhs; Expr* rhs; }`                   // lhs = rhs;
- `ExprStmt{ Expr* e; }`                                  // call();
- `ReturnStmt{ Expr* val /*nullable*/; SmallVector<Expr*> structFields; }`
- `BreakStmt{}` / `ContinueStmt{}`
- `BarrierStmt{ bool device; }`                           // goes through the peephole, NOT printed directly
- `RawStmt{ StringRef text; }`                            // banded compound one-liners, escape hatch

### Scopes (`msl::Scope` — a Stmt subtype that OWNS a `{ ... }` block)
Every brace in the output comes from a Scope node. The printer emits the header,
`{`, `++indent`, flushes+prints children, `--indent`, `}` — uniformly. No
lowering method ever writes a brace.
- `KernelFn{ attrs (maxThreads?), name, params[], Block body }`
- `DeviceFn{ retType, name, params[], Block body }`  (+ prototype emission)
- `ForScope{ init-decl, cond, step, Block body }`        // normal scf.for
- `TripCountForScope{ counter-decl, ivDecl, breakCond, Block body }` // wide-IV i64 form
- `IfScope{ cond, Block then, Block? else }`
- `WhileScope{ cond-or-true, Block body }`                // scf.while, spin loops, CFG state machine
- `StateMachineScope{ stateVar, cases: [ (label, Block) ] }` // the while(true){if(state==N)} CFG
- `PlainScope{ Block body }`                              // bare `{ ... }` (banded blocks)

`Block` = `SmallVector<Stmt*>`. A Scope with an empty Block still prints `{}`
correctly — the empty-for bug becomes structurally impossible (the header and
the body are one node; you cannot emit a header without its block).

### Params & attributes (`msl::Param`)
- `Param{ Type* ty; StringRef name; Attr* attr /*nullable*/ }`
- `Attr`: `Buffer{n}`, `ThreadgroupPosInGrid`, `ThreadPosInThreadgroup`,
  `ThreadgroupsPerGrid`, `MaxTotalThreads{n}` — printed as `[[...]]` by the
  printer's `printAttr`, the one place attribute syntax lives.

## Printer contract (MSLPrinter)

- Owns `raw_ostream &os` + `unsigned indent` + the barrier peephole
  (`barrierPending`, `barrierPendingDevice`). These fields move OUT of the
  emitter entirely.
- `printBlock(Block)` is the single choke point: for each Stmt, `flushBarrier()`
  first (preserving the invariant "flush before any non-barrier output and at
  every scope boundary"), then dispatch. A `BarrierStmt` sets pending instead of
  printing. Scope entry/exit also flush. This makes the barrier invariant a
  property of the printer, not 30 hand-placed `flushBarrier()` calls.
- `printType`/`printExpr`/`printStmt`/`printScope`/`printParam`/`printAttr` —
  pure structural recursion. Precedence: `printExpr` inserts parens by operator
  precedence (or the emitter inserts explicit `Paren` nodes; decide once — lean
  on explicit `Paren` from the builder to keep the printer dumb).
- The preamble (`#include`s, `using namespace metal;`, `tt_erf`) is a fixed
  header the printer emits before walking functions.

## State that STAYS in EmitMSL (the lowering), unchanged in behavior

These are IR-analysis, not text, and keep their exact semantics:
- `valMap : DenseMap<Value, SmallVector<std::string>>` — the per-register symbol
  table. Builders read it to make `VarRef` nodes. Rehash-safe snapshotting rule
  unchanged.
- `fresh()` name generator; fixed names `__pool`, `__tg_buf_n`, `__tgpos`, etc.
- LinearLayout helpers: `regCount`, `registerCoords`, `layoutOffsetExpr`,
  `flatTileOffset`, ... (these now return `Expr*` where they returned strings).
- Pool accounting: `poolBytes`, `globalPoolBytes`, `liveTgBytes`, `poolBudget()`,
  `scanPool` — unchanged.
- `memdescMap` — but `baseOffset` becomes `Expr*` not `std::string`.
- Fused-dot state: `FusedDotCtx`, `DirectStore`, `InPlaceOperand`, the matchers
  `matchGemmDotLoop`/`matchDirectStore`/`matchRowMajorOffset`/`matchBoundaryMask`
  — unchanged IR analysis; they now steer which Scope/Stmt nodes get built.
- CFG: `blockLabel`, `cfgState` — feed `StateMachineScope`.
- `scalarSpinlock` — flips `PointerType.coherent`.

## Migration (big-bang on branch)

1. Land `MSLAst.h/.cpp` + `MSLPrinter.h/.cpp` with the full node set + a
   round-trip test (build a small AST by hand, print it, diff against expected
   MSL). Wire into CMake. This is the foundation; nothing else starts until the
   printer round-trips.
2. Convert EmitMSL bottom-up: types first (all `mslType`-family helpers return
   `Type*`), then expressions (binary/cmp/cast/call/select), then the leaf
   statements (decl/assign/store/load), then each scope family (func, for, if,
   while, state-machine, fused-dot). Each converted family compiles but the file
   is RED until the whole walk is nodes.
3. Flip `emit()` to: build one AST per function, hand to `MSLPrinter`. Delete the
   raw `os`/`indent`/`ind()`/inline `flushBarrier` from the emitter.
4. Bring the corpus green: core (dot/scan/atomics/reduce/cast/precise_math),
   test_kernels, fla, gluon, inductor GPUTests + stress. Byte-diff emitted MSL
   against the pre-rewrite output for a sample of kernels where practical
   (semantics-preserving refactor; diffs should be whitespace-only or explained).
5. Merge to `msl-mlir` only when green.

## Invariants the rewrite must not break

- Barrier peephole semantics (adjacent-barrier collapse, stronger-scope wins,
  flush at every boundary) — now printer-owned but behaviorally identical.
- Per-register unrolling via valMap — one MSL var per register, dataless =
  empty vector.
- The wide-IV trip-counter loop form (iv declared INSIDE body) — dodges the
  AGX i65 Gauss-sum fold. Encoded as `TripCountForScope`.
- Safe-math / precise transcendentals — unchanged (that's the metallib compile
  flag + `metal::precise::` call names).
- No new comments beyond genuinely non-obvious invariants; match repo style.
