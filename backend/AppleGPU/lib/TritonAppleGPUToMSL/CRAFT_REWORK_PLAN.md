# MSL codegen craft-rework plan

STATUS: proposal. READ-ONLY diagnostic produced this; no code was changed.

## Purpose & hard constraint

The MSL AST codegen (`EmitMSL*.cpp` + `MSLEmitter.h` + `MSLAst.h`/`MSLPrinter.cpp`)
is correct and fully test-green, but it was built by ~15 sequential agents and
reads "synthetic": ternary ladders where a table belongs, copy-adapted twin arms,
string/node double-call migration scars, and ~260 lines of imperative raw-leaf
method bodies inlined in a header. This plan de-"synthetics" it as a pure
**code restructuring**.

**THE OVERRIDING CONSTRAINT — byte-identical MSL output.** There is an external
byte-diff oracle over 14 golden kernels that must stay 14/14 after every step.
Every step below is a semantics-preserving refactor: it changes how the C++
*builds* the AST, never which AST nodes get built or in what order, so the
emitted MSL text is byte-for-byte unchanged. Steps are ordered and
**independently executable**; each is one oracle-gated commit. If a step's diff
ever perturbs the golden output, it has a bug — revert, don't "bless" the diff.

Files (all under `backend/AppleGPU/lib/TritonAppleGPUToLLVM/`):
- Primary offenders: `EmitMSLOps.cpp` (2447 lines; the 1907-line `astEmitOp`
  dispatch, L538-2445), `MSLEmitter.h` (2124 lines; 96 `os <<`, ~260 lines of
  inline raw-leaf bodies).
- Secondary: `EmitMSLExpr.cpp`, `EmitMSLDot.cpp`, `EmitMSLAtomic.cpp`,
  `EmitMSLReduce.cpp`, `EmitMSLControlFlow.cpp`, `EmitMSLFunc.cpp`, `EmitMSL.cpp`.
- Support: `MSLAst.h`, `MSLPrinter.cpp`, `MSLTypes.cpp`, `MSLConstants.h`.

**Twin-repo note.** Per `CLAUDE.local.md`, the AIR/MSL backend exists in two
repos that stay in sync (triton-ext `Metal*`/`EmitMSL*` and llvm-project
`AIR*`). These files are triton-ext-only (MSL AST path); confirm whether the
`msl-only` branch's `EmitMSL*` has a twin before mirroring. Where a twin exists,
mirror each committed step. Match surrounding no-comment style (per the global
CLAUDE.md: comments only for genuinely non-obvious invariants).

---

## Ordering rationale

Do boilerplate removal **before** splitting `astEmitOp`, so the split lands on
already-dense code and the sub-dispatchers come out small. Trivial comment
cleanup (Step 0) can land anytime. Recommended sequence:

0. Stale-comment sweep (trivial, anytime).
1. cmp-predicate table (kills the enum→string→enum round-trip).
2. float/int/bitwise/shift binop table (shared op→BinOp lookup).
3. program-id / num-programs twin merge.
4. floatLit/astFloatLit node migration (DELETES real double-call code).
5. Secondary-file twin-merges & helper hoists (Dot, Expr, Atomic, Reduce, CFG, Func).
6. Move raw-leaf method bodies out of `MSLEmitter.h` into a `.cpp`.
7. Split `astEmitOp` into op-family sub-dispatchers (LAST — cosmetic; only pays
   off after 1-5 shrink the arms).

---

## Step 0 — Stale-comment sweep (trivial, anytime)

**What.** Delete/repair every comment that references a deleted string-route
`emit*` function. The S2 sweep missed ~11. These are dangling "Mirrors emitXxx"
tombstones pointing at symbols that no longer exist. (Kills a documentation
smell; zero code effect.)

**Where.** Confirmed dangling references (target has 0 live definition):

| File:line | Comment text | Deleted target |
|---|---|---|
| `EmitMSLOps.cpp:291` | "Mirrors emitFusedGemm with AST nodes." | emitFusedGemm |
| `EmitMSLOps.cpp:1724` | "Mirrors emitScan with AST nodes." | emitScan |
| `EmitMSLOps.cpp:1922` | "Mirrors emitReduce with AST nodes." | emitReduce |
| `EmitMSLOps.cpp:422` | "(mirrors emitLoad/emitStoreBody)." | emitLoad/emitStoreBody |
| `EmitMSLDot.cpp:290` | "Mirrors emitDot's fused branch." | emitDot |
| `EmitMSLDot.cpp:470` | "Mirrors emitDot's dotNeedsPanel branch." | emitDot |
| `EmitMSLDot.cpp:622` | "Mirrors emitDotScalar." | emitDotScalar |
| `EmitMSLAtomic.cpp:289` | "(mirrors emitPacked16CAS)" | emitPacked16CAS |
| `EmitMSLControlFlow.cpp:134` | "Mirrors emitBlockCFG." | emitBlockCFG |
| `EmitMSLOps.cpp:424` | "cast form used by emitLoad/StoreBody" | emitLoad/emitStoreBody |
| `EmitMSLOps.cpp:478` | "mirroring emitStoreBody's thread" | emitStoreBody |
| `EmitMSLOps.cpp:497` | "print identically to emitStoreBody" | emitStoreBody |
| `EmitMSLReduce.cpp:26,29,35,91,148,228` | "emitCombineN/emitCombine/emitReduce/emitScan/emitScanWarpCarry/emitMapElementwise/emitMapCFG" banners | all deleted |
| `EmitMSLAtomic.cpp:81,98,217,372,393` | "emitAtomicRMW/emitAtomicCAS/emitAtomicPoll/emitHistogram" banners | all deleted |
| `EmitMSLFunc.cpp:90` | "mirrors emitDeviceSignature's bindArgs" | emitDeviceSignature |

Full count (grep-verified): **26 comment lines** contain `emit[A-Z]`/`declRetStruct`
references; **22 reference ONLY deleted symbols** (19 distinct deleted names). The
`EmitMSLAtomic.cpp` banners (L81,98,217,372,393) name deleted functions — the
sections are now `astEmit*`; RENAME the banner to the `ast*` name (don't just
delete the banner).

DO NOT TOUCH (these targets STILL EXIST — leave the comments):
`EmitMSLFunc.cpp:3-4` reference `emitFunc` / `emitDeviceFunc` / `emitDeviceFuncProto`
/ `declRetStruct` — all live in `MSLEmitter.h`. The raw-leaf emitters
`emitRoundedHalfValue`/`emitTruncatedFloatValue`/`emitFloat32CASLoop`/
`emitPacked16CASLoop` are live code and appear in NO comments.

**How.** Either drop the "Mirrors emitXxx" clause (the sentence usually stands
without it) or replace with the true intent (e.g. `// Readback-phase dot`). Do
NOT add a tombstone explaining the removal.

**Risk.** Byte-identical-safe (comments only).

**Payoff.** Removes ~11 misleading references; a reader stops grepping for
functions that don't exist.

---

## Step 1 — Cmp-predicate table (enum→string→enum round-trip)

**What.** The `arith.cmpi`/`arith.cmpf` arms build a `const char *o` string from a
switch, then immediately call `cmpBinOp(o)` (`EmitMSLOps.cpp:18-25`) to convert
that string **back** into a `msl::BinOp`. That's a pointless enum→string→enum
round-trip. Replace with direct predicate→`{BinOp, isUnsigned}` lookup. Kills
anti-pattern #1 (ladder-that-should-be-a-table) at its densest site and lets the
`cmpBinOp` helper be deleted.

**Where.** `EmitMSLOps.cpp`:
- CmpI arm: L703-725 (the `switch` L706-717 + `cmpBinOp(o)` L718).
- CmpF arm: L726-748 (the `switch` L728-741 + `cmpBinOp(o)` L743).
- `cmpBinOp` helper: L18-25 (delete after both arms migrate).
- 2 call sites of `cmpBinOp` (both in the arms above).

**How.** For CmpI, a static `constexpr` map from `arith::CmpIPredicate` to
`{msl::BinOp, bool uns}`:

```cpp
// before (L706-718):
const char *o; bool uns = false;
switch (ci.getPredicate()) {
case arith::CmpIPredicate::ult: uns = true; [[fallthrough]];
case arith::CmpIPredicate::slt: o = "<"; break;
... 8 more ...
}
msl::BinOp bo = cmpBinOp(o);

// after:
struct CmpIRow { msl::BinOp op; bool uns; };
static const CmpIRow tbl[] = { /* indexed by CmpIPredicate enum order:
  eq,ne,slt,sle,sgt,sge,ult,ule,ugt,uge */ ... };
auto [bo, uns] = tbl[(int)ci.getPredicate()];
```
(Confirm the exact enum ordinal order in the MLIR arith header before indexing;
if fragile, use a small `switch` returning the struct instead of `const char*`.)
CmpF maps OLT/ULT→Lt etc. (predicate→BinOp only, no `uns`); same shape, table or
struct-returning switch. Both then feed the existing
`astElementwiseExpr(bo, opCast, ...)`.

**Risk.** Byte-identical-safe. Care: the CmpI table must preserve the exact
signed/unsigned `opCast` decision (the `uns` flag drives `astUnsignedType` at
L720). Verify the golden kernels that exercise unsigned compares still 14/14.

**Payoff.** Removes `cmpBinOp` (8 lines) + collapses two 12-line switches into
two ~3-line lookups. Kills the most egregious "build a string just to parse it
back" scar in the file.

---

## Step 2 — Binop op→BinOp table (float / int / bitwise / shift)

**What.** Several arms carry inline `isa<XOp>(op) ? B::A : isa<YOp>(op) ? B::B
: ...` ladders mapping an arith op to a `msl::BinOp`. Consolidate the ones that
are pure op→enum lookups into one shared table/helper. Kills anti-pattern #1.

**Where.** `EmitMSLOps.cpp`:
- Float binop ladder: L583-586 (`isa<AddFOp>?Add : SubFOp?Sub : MulFOp?Mul : Div`).
- Bitwise ladder: L607-609 (`AndIOp?And : OrIOp?Or : Xor`).
- Int binop: L599-602 routes through `astIntBinaryExpr` → `intBinOp`
  (`EmitMSLExpr.cpp:53-63`, a 5-way `isa` ladder).
- Shift: L617-621 routes through `astShiftExpr` → the `ShLIOp?Shl:Shr` ternary
  (`EmitMSLExpr.cpp:97`).
- The math.* dialect ladders (L872-885 unary, L892-894 binary) are ALREADY
  `llvm::StringMap` tables — leave them; they are the good pattern to copy.

**How.** Add a single `static msl::BinOp arithBinOp(Operation *op)` (near
`intBinOp` in `EmitMSLExpr.cpp`, or a shared anon-namespace helper) that covers
add/sub/mul/div/rem/and/or/xor/shl/shr for both float and int arith ops via one
`TypeSwitch<Operation*, msl::BinOp>` or a flat `if (isa<...>)` block returning
the enum — ONE place, not four. The float arm at L581-588 becomes:

```cpp
return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
  return astElementwiseExpr(arithBinOp(op), nullptr,
                            opnd(op->getOperand(0), r),
                            opnd(op->getOperand(1), r));
});
```
Fold `intBinOp` (EmitMSLExpr.cpp:53-63) into `arithBinOp` and have
`astIntBinaryExpr` call it. `astShiftExpr`'s `ShLIOp?Shl:Shr` (L97) is a 2-way
switch — genuinely tiny; either fold into `arithBinOp` or LEAVE (see DO NOT
TOUCH). The int-decl-type promotion logic (L595-598: i1 and unsigned div/rem)
stays exactly where it is — that's not a mapping, it's a type decision.

**Risk.** Byte-identical-safe. Care: do NOT merge in the min/max family
(L834-857) — that maps to a *call name* (`"max"`, `"metal::fmod"`, `"mulhi"`)
plus `opCast`/`propagateNan` flags, a different codomain; it's a distinct
table (see Step 2b).

**Payoff.** One op→BinOp lookup instead of 4 scattered ladders; new arith ops
get added in one place.

### Step 2b (optional) — min/max/mulhi/remf family descriptor table

**What.** The block at `EmitMSLOps.cpp:834-857` is a hand-rolled if-ladder
assigning `{fn, opCast, propagateNan}` per op (MaximumFOp→{"max",_,nan},
MaxUIOp→{"max",unsigned,_}, RemFOp→{"metal::fmod",...}, MulhiUIOp→{"mulhi",...}).
This IS already tabular in spirit. Consider a `static const` array of
`{TypeID, StringRef fn, bool unsigned, bool nan}` rows iterated once. HONEST
verdict: it's borderline — 8 rows with per-row flag combos. If the array reads
cleaner than the ladder, do it; if it just moves the same data behind an index,
**leave as-is**. Low priority; don't force it.

**Risk/Payoff.** Byte-identical-safe; marginal readability. Optional.

---

## Step 3 — GetProgramId / GetNumPrograms twin merge

**What.** `GetProgramIdOp` (L624-634) and `GetNumProgramsOp` (L635-645) are
byte-identical arms except for ONE token: the builtin var (`tgposId` vs
`numTgId`). Classic copy-adapted twin (anti-pattern #2). Merge into one helper.

**Where.** `EmitMSLOps.cpp:624-645` (two 10-line arms → one shared helper + two
2-line call sites).

**How.** Both compute `comp` = axis→"x"/"y"/"z" (itself a 3-way ternary — a
tiny shared axis helper `static const char *axisComp(ProgramIDDim)` cleans both
at once), then emit `(int)(builtin.comp)`. Shared helper:

```cpp
// emit  int id = (int)(builtinVar.comp);  and bind result.
void MSLEmitter::astProgramDim(Operation *op, StringRef builtinVar,
                               tt::ProgramIDDim axis, msl::Block &body) {
  const char *comp = axisComp(axis);
  msl::Expr *e = ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::I32),
                          ctx.paren(ctx.member(ctx.var(builtinVar), comp)));
  std::string id = fresh();
  body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), id, e));
  bindScalar(op->getResult(0), id);
}
```
Arms become:
```cpp
if (auto p = dyn_cast<tt::GetProgramIdOp>(op)) { astProgramDim(op, tgposId, p.getAxis(), body); return true; }
if (auto n = dyn_cast<tt::GetNumProgramsOp>(op)) { astProgramDim(op, numTgId, n.getAxis(), body); return true; }
```

**Risk.** Byte-identical-safe (the emitted `(int)(__tgpos.x)` etc. is unchanged;
the only difference between arms was always the var name, now a parameter).

**Payoff.** ~16 lines → ~10; the "one differing token" duplication is gone and
the axis→comp ternary is shared.

---

## Step 4 — floatLit/astFloatLit node migration (DELETES double-call code)

**What.** `astFloatLit(v, StringRef sc)` (`EmitMSLExpr.cpp:38-43`) and
`floatLit(v, StringRef sc)` (L31-36) take a **string** `sc` only to test
`sc == "bfloat" || sc == "half"` — i.e. "is this a narrow float type." At every
call site the arm computes a *second* string `scStr = mslScalarType(...)`
alongside the already-computed node `sc = astScalarType(...)`, purely to feed
this predicate. Change the signature to take the type node (or the MLIR `Type`)
and test the kind internally; the companion `mslScalarType` calls then DELETE.
This is anti-pattern #3 and it removes REAL code (highest-value scar kill).

**Where.** `EmitMSLOps.cpp`, the two `arith.constant` sub-arms:
- Splat/dense-table: L654-668 — `scStr = mslScalarType(...)` (L656) feeds
  `floatLit(..., scStr)` (L659) and `astFloatLit(v, scStr)` (L668).
- Scalar constant: L678-682 — `scStr = mslScalarType(...)` (L679) feeds
  `astFloatLit(fa.getValue(), scStr)` (L682).
- Signatures: `MSLEmitter.h:970-971` (`floatLit(v, StringRef)`,
  `astFloatLit(v, StringRef)`), defs `EmitMSLExpr.cpp:31-43`.
- Total call sites of the string-`sc` overloads: 3 (L659, L668, L682). The
  bare `floatLit(v)` overload (no sc) is unaffected.

**How.** The predicate is "narrow float type" — either pass the `msl::Type*`
node (test `isa<ScalarType>` with `Kind==F16||BF16`, matching how
`astFloatLit` builds `ctx.call(sc, {lit})`) OR pass the MLIR `Type` and test
`t.isF16() || t.isBF16()`. Passing the MLIR `Type` is cleanest (the caller
already has `rt.getElementType()` / `res.getType()` in hand):

```cpp
// after — signature:
msl::Expr *astFloatLit(const APFloat &v, Type ty);   // tests ty.isF16()||ty.isBF16()
std::string floatLit(const APFloat &v, Type ty);      // same predicate, string form

// after — dense arm (L654-668 shape):
msl::Type *sc = astScalarType(rt.getElementType());
Type elemTy = rt.getElementType();               // scStr line DELETED
...
? floatLit(dense.getSplatValue<APFloat>(), elemTy)
...
init.push_back(astFloatLit(v, elemTy));
```
For the narrow-float ctor name, `astFloatLit` currently does
`ctx.call(sc /* "bfloat"|"half" */, {lit})` — after migration it derives that
name from the Type (`t.isBF16() ? "bfloat" : "half"`), matching
`mslScalarType`'s exact spelling so output is unchanged.

**Risk.** Byte-identical-safe BUT this is the migration to run carefully: the
narrow-float ctor spelling MUST match `mslScalarType` exactly (`"bfloat"` /
`"half"`), and the non-narrow path must still emit the bare `%.17g` literal.
Gate on the golden kernels that contain bf16/f16 constants. This is the one step
that changes a public helper signature — do it as its own commit.

**Payoff.** Deletes 2 `scStr = mslScalarType(...)` lines + removes an entire
redundant type-stringification per constant arm; finishes the "floatLit takes a
node" migration the earlier agents left half-done. This is the plan's cleanest
"delete real code" win.

NOTE on the OTHER `mslScalarType`+`astScalarType` adjacencies (L944/945,
978/979, 1377/1378, 1402/1403, 1661/1662, 2087/2088, 2126/2127, 2276/2277,
EmitMSLDot.cpp:58/59, 639/640): these feed the string into *string-only*
consumers (`astPoolRegion`, `astInit0`, `ctx.named("threadgroup "+sc)`,
`astDerefPtr`, the atomic `sc`-driven CAS leaves). They are a SEPARATE, larger
migration (make those consumers take nodes). Do NOT bundle them into Step 4 —
they're higher-risk and lower-value. Consider a `astScalarTypePair(t) ->
{msl::Type*, std::string}` accessor as a *stopgap* only if a follow-up wants to
tackle them; for now, LEAVE (see DO NOT TOUCH).

---

## Step 5 — Secondary-file twin-merges & helper hoists

Independent per-file commits. Each collapses copy-adapted twins/triplets
(anti-pattern #2) behind one helper. Ordered by payoff.

### 5a — EmitMSLDot.cpp: A/B-stage store loops → `astStageOperand`

**Where.** Three twin pairs, each two blocks differing only by operand A vs B
(`{tgName, stageTy, names, inPlace}`):
- `astEmitDot` L213-220 (A) vs L221-228 (B)
- `astEmitDotFused` L321-325 (A) vs L326-330 (B)
- `astEmitDotScalar` L677-683 (A) vs L684-690 (B)

**How.** One helper
`void astStageOperand(msl::Block &body, StringRef tgName, RankedTensorType stageTy,
ArrayRef<std::string> names, int rank, bool skip, llvm::function_ref<msl::Expr*(int)> batchCond)`.
Each pair → two calls. **Payoff:** 6 blocks → 1 helper + 6 calls.

### 5b — EmitMSLDot.cpp: per-file lambda hoists

- `tgPtr` lambda defined 3× (L93-95, L495, L654; L495 & L654 byte-identical) →
  private method `msl::NamedType *tgPtr(StringRef sc)`.
- `barrier` lambda defined 3× byte-identical (L96, L305, L502) → reuse one
  `astBarrier(msl::Block&)` (or push the node directly).
- `batchCond` lambda 2× (L207-210, L662-665) → private method.
- `fragMMA` inner K-loop re-inlined in `astEmitDotPanel` L567-574 (vs the lambda
  at L176-186, differing only by base names + ld) → hoist `fragMMA` to a method
  and call from the panel path.
- Per-warp `if(warpId==w){sgStore...}` loops (L405-423, L431-441, L453-462) →
  hoist the existing `warpIf` wrapper (L138-142) into
  `astPerWarpFrags(dst, nFrag, nT, numWarps, emitFrag)`.

**Risk.** Byte-identical-safe; the lambdas already close over the same state.

### 5c — EmitMSLExpr.cpp: unify `astShuffle` / `astShuffleExpr` + `scalarOf` table

**Where.** `astShuffle` (L204-255, statement form) and `astShuffleExpr`
(L257-299, expr form) are the SAME four-way (`long/ulong` | `bool` |
`bfloat/half/short/char` | default) dispatch; the scalar-name→`msl::Scalar`
ladder is spelled 3× (L210, L237, L239-242, and the `scalarOf` lambda L259-273).

**How.** (1) One `static msl::Scalar scalarOf(StringRef)` table (+ a
`bitsScalarOf` for the packed round-trip). (2) A pure-expr helper
`msl::Expr *astShuffleBits(StringRef op, StringRef sc, StringRef val, StringRef arg)`
returning the shuffled value; `astShuffle` wraps it in one `declStmt`. Removes
the entire statement-form ladder (~50 lines).

Also: `recast` lambda is byte-identical in two functions (L67-70 in
`astElementwiseExpr`, L113-116 in `astMinMaxExpr`) → one private method
`msl::Expr *recast(msl::Type *opCast, StringRef n)`.

**Risk.** Byte-identical-safe. The stmt-vs-expr merge must preserve the exact
`declStmt` var-name minting order in the statement form — verify with the
shuffle-heavy golden kernels (scan/reduce).

### 5d — EmitMSLAtomic.cpp: packed16 split/merge + CAS-call helpers

**Where.**
- Packed16 hi/lo extract `(isHigh)?(word>>16):(word&0xffffu)` built 2× (L164-169,
  L318-321) → `astPacked16Extract(word, isHigh)`.
- Packed16 repack ternary built 2× (L181-200, L333-349), differing only by the
  new-bits expr (`as_type<ushort>(newLane)` vs plain `lane`) →
  `astPacked16Merge(word, isHigh, newBitsU32)`.
- CAS-weak call `ctx.call(CompareExchangeWeak, {ptr, addrOf(exp), newVal,
  Relaxed, Relaxed})` built 5× (L134-140, L202-208, L233-239, L265-271,
  L351-357) → `msl::Expr *astCasWeak(wordPtr, expVar, newVal)`.

**Risk.** Byte-identical-safe. Care: two of the five CAS sites are while-cond
form and three are break-in-while-true — the helper returns the *call expr*;
each site keeps its own control-flow wrapper.

### 5e — EmitMSLControlFlow.cpp: reg-broadcast-copy + cross-file stateMachine

**Where.**
- Broadcast-copy loop `assignStmt(var(dst[r]), var(src[size==1?0:r]))` in 3
  functions (L97-100, L123-127, L198-200) →
  `void astCopyRegs(msl::Block&, ArrayRef<std::string> dst, ArrayRef<std::string> src)`.
- `astBlockCFG` (L80-88) is byte-identical to `astMapCFGStateMachine`
  (EmitMSLReduce.cpp:249-256) — a **cross-file** duplicate building
  `StateMachineScope::Case`s. Hoist ONE shared helper (e.g. in EmitMSLOps.cpp or
  a shared spot) and call from both.

**Risk.** Byte-identical-safe.

### 5f — EmitMSLReduce.cpp: scratch-slot / warp-base / named-decl helpers

**Where.** `warpId*32+laneId` slot built 2× (L65-67, L171-173) →
`astScratchSlot()`. `(warpId & ~mask)*32 + tail` built 2× (L74-77, L184-187,
differ only by tail `laneId` vs `axisTopLane`) → `astWarpBase(mask, tail)`.
`astReduceAccInit`/`astScanAccInit`/`astScanLaneSeed` (L39-42, 95-98, 101-104)
are 3 identically-shaped `declStmt(named(sc), x, var(y))` → optional
`astNamedDecl(sc, dst, src)`.

**Risk.** Byte-identical-safe. The named-decl collapse is marginal; do only if
it reads cleaner.

### 5g — EmitMSLFunc.cpp: ret-struct field append

**Where.** `astRetStructDecl` L40-45 (tensor branch) vs L46-49 (multi-result
branch) both append `"  "+scalarType+" f"+i+";\n"` → tiny `appendField(body, sc,
i)`. Low value; optional. (`astDeviceRetType` L58-67 is a legitimate dispatch —
LEAVE.)

### 5h — `fresh()` / `LocalGen::fresh()` duplication (optional, low value)

**Where.** Two logically-identical fresh-name bodies: `MSLEmitter::fresh()`
(`MSLEmitter.h:437`, `"v"+std::to_string(nextId++)`) and `LocalGen::fresh()`
(`MSLEmitter.h:42`, `"v"+std::to_string(id++)`). `LocalGen` (L40-43) exists only
to give the free-function helpers (`astElemwiseDecls`, `astCombineN`, the 6
`EmitMSLAtomic.cpp` sites, `EmitMSLFunc.cpp:95`) a `fresh()` bound to a passed
`int&`. **HONEST VERDICT: LEAVE.** The two are the same string logic but mutate
different counters by design (a member vs a passed-by-ref seed); unifying would
mean threading the emitter or a counter object through the free helpers — pure
indirection. The `"v"` prefix could be a shared `constexpr` if you want ONE
source of the literal, but that's the only justified touch. Do not "fix" the
LocalGen split.

**Payoff (Step 5 total).** ~150+ lines of copy-adapted duplication collapse
into ~10 helpers; the biggest single win is 5a (6 blocks → 1).

---

## Step 6 — Move raw-leaf method bodies out of `MSLEmitter.h`

**What.** ~260 lines of imperative `os <<`/`indent`/`fresh()` raw-leaf method
bodies are inlined in the class declaration in `MSLEmitter.h` — 96 `os <<`
statements in a **header**. These are the fp-narrowing and fp-emulated-CAS
leaves. Move the bodies to a `.cpp`, leaving the header a clean set of
declarations. (Kills the "96 os<< in a header" smell; the header becomes a class
decl, not a source file in disguise.)

**Where.** `MSLEmitter.h` inline method bodies to relocate:
- `emitRoundedHalfValueFull` L1862-1930
- `emitTruncatedFloatValue` L1934-~1978
- `emitRoundedHalfValue` L1980-~2024
- `emitPacked16Base` L2025-~2043
- `emitPacked16CASLoop` L2044-~2072
- `emitFloat32CASLoop` L2073-~2094
- `floatRmwExpr` L2095-2110 (`static`, pure switch)
- `init0` L2117-2119 (`static`, tiny)

**How.** Move the bodies to a `.cpp`. Two options:
- (a) NEW `EmitMSLRawLeaves.cpp` — cleanest; these leaves are a cohesive family
  (the sanctioned `Raw` escape-hatch content per `MSL_AST_DESIGN.md` §Raw). Add
  it to the CMake source list next to the other `EmitMSL*.cpp`.
- (b) Fold into `EmitMSLExpr.cpp` (the narrowing leaves are expr-ish) and
  `EmitMSLAtomic.cpp` (the CAS leaves are atomic-ish). More natural homes but
  splits the family across two files.

RECOMMENDATION: **(a) new `EmitMSLRawLeaves.cpp`** — one file, one `#include
"MSLEmitter.h"`, the whole sanctioned-raw family in one place; matches the
existing per-family `EmitMSL*.cpp` convention. In the header, replace each body
with a declaration (`std::string emitRoundedHalfValueFull(const std::string &sc,
const std::string &v);` etc.). `floatRmwExpr` and `init0` are `static`/tiny — you
MAY leave them inline (they're one-liners / pure switches, legitimately header
inline); moving them is optional.

Also relocate any *other* non-trivial inline member body in the header (grep for
`os <<` and multi-line bodies inside the class). The small `inline` FREE
functions at the top (`mslScalarType` L45-66, `mslUnsignedType`, `mslStorageType`,
`mslKernelName`, `mslDeviceFuncName`, `barrierMemFlags`) are pure string maps
used across TUs — legitimately header-inline; **LEAVE them** (see DO NOT TOUCH).

**Risk.** Byte-identical-safe (pure relocation; identical statements execute in
identical order). Care: these bodies read mutable emitter state (`os`, `indent`,
`ind()`, `fresh()`) — they must remain non-`static` member functions in the
.cpp (the `#include "MSLEmitter.h"` + `MSLEmitter::` qualifier). Confirm the
build still links (add the new file to CMake). No output change.

**Payoff.** Header drops from 2124 → ~1860 lines and becomes a real class
declaration; the 96-`os<<`-in-a-header smell is gone; the raw leaves sit with
the rest of the lowering.

---

## Step 7 — Split `astEmitOp` into op-family sub-dispatchers (LAST)

**What.** `astEmitOp` (`EmitMSLOps.cpp:538-2445`, ~1907 lines) is one giant
waterfall of `if (auto x = dyn_cast<...>(op))` arms. AFTER Steps 1-5 shrink the
arms, split it into cohesive per-family sub-dispatchers so the spine reads as a
table of families. This is COSMETIC — do it last, on already-dense code, or it
just moves boilerplate around.

**Where.** `EmitMSLOps.cpp:538-2445`.

**How.** `astEmitOp` keeps the early structural no-ops (barriers L544-561,
no-ops L563-572, `resElem` L574-576) then delegates to sub-dispatchers, each
returning `optional<bool>`/a "handled" flag so the spine tries them in order.
Proposed split (names read by intent, NOT `astEmitOp2`):

| Sub-dispatcher | Ops (current line ranges) |
|---|---|
| `astEmitArith` | float/int/bitwise/shift binops, cmpi/cmpf, select, clamp, negf, min/max family, precise_sqrt (L578-864) |
| `astEmitMath` | the `math.*` dialect StringMap block (L866-915) |
| `astEmitConstAndRange` | arith.constant, make_range (L647-700) |
| `astEmitGridId` | program_id / num_programs (L623-645; post-Step-3 one helper) |
| `astEmitReshape` | splat/unsplat/expand/broadcast/join/split, trans, reshape (L917-1001) |
| `astEmitMemDesc` | memdesc_index/subslice, local_alloc/store/load, async_copy, convert_layout (L1003-1251) |
| `astEmitDotAndMap` | dot (→astEmitDot), map_elementwise (L1253-1311) |
| `astEmitReduceScan` | histogram, scan, reduce (L1313-.., L1723-2080) |
| `astEmitAtomic` | atomic_rmw, atomic_poll, atomic_cas (L1397-1721) |
| `astEmitTensorMove` | cat, gather (L2081-2173) |
| `astEmitCallReturn` | return, call, addptr, load, store (L2174-2318) |
| `astEmitControlFlow` | scf.if, scf.for, scf.while (L2320-2441) |

`astEmitOp` becomes: no-ops, then `return astEmitArith(op,body) || astEmitMath(...)
|| ...` (guarded so a matched-but-failed arm still returns false correctly — keep
the existing tri-state: matched+ok=true, matched+fail=false, unmatched=fall
through). Preserve arm ORDER within and across sub-dispatchers exactly (some arms
rely on earlier arms not matching).

**Risk.** Byte-identical-safe IF arm order is preserved and the
matched-vs-unmatched contract is kept (each sub-dispatcher must distinguish "I
didn't handle this op, try the next family" from "I handled it and it failed").
This is the highest-care split — do it as its own commit, run the FULL golden
oracle, and keep each sub-dispatcher's arms in their current relative order.

**Payoff.** The 1907-line method becomes a ~30-line spine + 12 focused
sub-methods; a reader finds "where is atomic_cas handled" instantly. But it adds
zero density — that's why it's LAST.

---

## Naming (low-value churn — flag, don't chase)

Renaming is low-value; only two names genuinely mislead. Fold into whichever
step touches the file, or skip.

- **`astWalkBlock2`** (decl `MSLEmitter.h:384`, def `EmitMSLControlFlow.cpp:181`,
  called `EmitMSLControlFlow.cpp:167` + `EmitMSLOps.cpp:273`) vs its sibling
  `astWalkBlock` (def `EmitMSLControlFlow.cpp:238`). The `2` is the
  generated tell — it's the block-walker variant that takes a hoist list instead
  of a depth. Rename to intent, e.g. `astWalkBlockHoisting` /
  `astWalkBlockWithSpills`. 4 references. Byte-identical-safe (rename only).
- **`astInit0` / `init0`** (`MSLEmitter.h:276`/`EmitMSLAtomic.cpp:29` and
  `MSLEmitter.h:2117`) — the `0` suffix + AST-vs-string twin. Minor. Rename to
  `astZeroLit` / `zeroLit` if a step touches them; otherwise LEAVE.

No `*Impl` names exist. Other abbreviations (`isF32`, `asU32`, etc.) are
conventional — leave.

## DO NOT TOUCH (looks synthetic, is actually correct-and-fine)

- **The `Raw`/`captureRaw` leaf CONTENTS** — the fp-narrowing bit-twiddling
  (`emitRoundedHalfValueFull` etc.) and the emulated-CAS loops. Step 6 MOVES
  these bodies; it must NOT rewrite the arithmetic. Every mask/shift/round-bias
  is IEEE-exact and oracle-load-bearing. Restructuring the C++ that *emits* them
  is fine; touching the emitted bit-twiddling is a correctness change → forbidden.
- **The explicit `ctx.paren(...)` discipline.** The builder inserts `Paren`
  nodes deliberately (per `MSL_AST_DESIGN.md`: "lean on explicit Paren from the
  builder to keep the printer dumb"). Do NOT "simplify" by removing parens or
  pushing precedence into the printer — that changes output.
- **The per-register `names(v)[size==1?0:r]` broadcast idiom.** Appears
  everywhere; it's the splat-vs-per-register selection. It's repetitive but
  correct and clear; a helper (`opnd`) already exists in `astEmitOp` — do NOT
  over-abstract it into every file.
- **The math.* `llvm::StringMap` tables** (EmitMSLOps.cpp:872-894) — already the
  GOOD pattern. Leave; copy their style for Steps 1-2.
- **`astDeclBind` and its 4-arg-lambda call sites** (25 in EmitMSLOps.cpp).
  HONEST VERDICT: the lambda-per-register dance is the least-bad C++ here. Each
  arm's expr-builder genuinely differs (different operands, casts, callees); a
  "table of {op-set → expr-builder}" would just relocate the same lambdas behind
  an indirection and HURT readability. LEAVE AS-IS. (After Step 2's binop table,
  the *bodies* of many lambdas collapse to a one-line `astElementwiseExpr(arithBinOp(op),
  ...)` — that's the real win; the scaffolding stays.)
- **The int-decl-type promotion at L595-598** (i1 → I1, unsigned div/rem →
  unsigned type). Not a mapping — a type decision. Leave inline.
- **`astDeviceRetType`** (EmitMSLFunc.cpp:58-67) and similar 3-4-way *dispatch*
  ladders that map to genuinely different code paths (not a flat enum lookup).
  Leave.
- **The `rev ? X : Y` 2-way polarity switches** in EmitMSLReduce.cpp (L110, 121,
  210). Two-way; a table is overkill. Leave.
- **The small `inline` free type-helpers at the top of MSLEmitter.h**
  (`mslScalarType` etc.). Legitimately header-inline (pure, cross-TU). Leave.
- **The remaining `mslScalarType`+`astScalarType` adjacencies** feeding
  string-only consumers (NOT the floatLit ones). A real migration, but
  higher-risk/lower-value than Step 4 — out of scope for this pass unless a
  follow-up explicitly takes the "make astPoolRegion/astInit0/astDerefPtr take
  nodes" migration. Leave for now.
- **`cmpBinOp` is deleted in Step 1, `intBinOp` folded in Step 2** — don't also
  "clean up" `astShiftExpr`'s 2-way `ShLIOp?Shl:Shr` into a table; it's two
  cases. Leave (or fold into `arithBinOp` only if trivial).

---

## Oracle-gating checklist (every step)

1. Build: `ninja -C build metal-llc` (backend) — zero new warnings.
2. Run the 14-kernel byte-diff golden oracle — must stay 14/14, byte-identical.
3. If ANY golden diff appears: the step has a bug (wrong var-mint order, changed
   paren, reordered arm). REVERT and re-derive — never re-bless the golden.
4. Commit per step (independent, revertable). Mirror to the twin repo if the
   `EmitMSL*` file has an `AIR*` twin on this branch.
