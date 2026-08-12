# DOT_REFACTOR_PLAN

Status: proposal, nothing implemented. Written 2026-08-11 against `1eef2f7`.

## Why now

`EmitMSLDot.cpp` is 4308 lines, the largest file in the emitter by 15%. It has
absorbed every GEMM feature of the last two months — panelisation, device-direct
A, fused-C overlay, async DMA staging, transpose-on-load, the ragged-drain
epilogue — and each arrived as a branch inside an existing function rather than
as a unit with a boundary.

The cost is not aesthetic. Four bugs in the last week were *the same bug*: two
code paths answering one question differently.

| bug       | the two disagreeing sides                                                  |
| --------- | -------------------------------------------------------------------------- |
| `739091a` | `scanPool`'s panel gate vs `planDot`'s                                     |
| `ad518bf` | `scanPool` sizing the pool vs `planDot` laying it out                      |
| `cb7173d` | `computeDmaStoreTransposed` asked during planning vs during emission       |
| `1eef2f7` | `dotStageRowPads` asked with `bInPlace` in the Decl phase vs the MMA phase |

That is a structural defect with a name: **the same predicate is evaluated more
than once, at points where its inputs are not yet equal.** Everything below is
aimed at that, not at line count.

## What the file actually contains

Four terminal emitters, dispatched from `emitDot` (line 1442):

| emitter         | lines | role                                         |
| --------------- | ----- | -------------------------------------------- |
| `emitDotDirect` | 293   | one whole-tile MMA block, no K-loop fusion   |
| `emitDotFused`  | 712   | the K-loop GEMM: staging, rotation, epilogue |
| `emitDotPanel`  | 162   | 64x128 split into two 64x64 N-panels         |
| `emitDotScalar` | 112   | fallback, no simdgroup matrices              |

Plus roughly 900 lines of DMA plumbing (`dmaCopyEligible` .. `dmaBeginInto`),
600 lines of IR pattern matching (`matchTileIndex` .. `matchDirectStore`), and
`planDot` (209 lines) which decides everything.

`DotPlan` carries **15 fields**. `DotEmitCtx` carries most of the emission
state. Between them they are the de-facto interface, but neither is documented
as such and both are mutated after construction.

### The duplication, measured

Each emitter independently orchestrates the same five-step sequence — stage the
operands, declare fragments, load them, multiply-accumulate, store — using the
same primitives:

```
                stageOperand  fragDecl  sgLoad  sgMAC  sgStore  hardBarrier
emitDotDirect        2           6        1       3       2         1
emitDotFused         2           2        1       1       2         5
emitDotPanel         -           2        2       1       1         2
emitDotScalar        2           -        -       -       -         3
```

63 call sites across four functions. None of them share a loop nest. A change to
how a fragment is addressed has to be made, and tested, four times.

## The proposal

Three steps, each independently landable and independently revertable. Do them
in order; step 1 is the one that pays for itself immediately.

### Step 1 — one question, one answer, one place — **LANDED (`d6f39c8`)**

Implemented, with one correction to the plan below. `scanPool` runs to
completion *before* the body walk reaches `planDot` — it is what computes
`poolBytes` — so the two cannot share anything downstream of the budget.
`DotFacts` therefore holds only the IR-derived half (shape, element widths, A/B
residency, in-place/no-stage, device-direct A, the phase-free B DMA candidate).
The budget-dependent fields the sketch below lists — `aPad`, `bPad`, `stagedA`,
`stagedB`, `cReserved` — stay with whichever pass decides them, because
`scanPool` necessarily works against a provisional budget.

That still covers all four bugs in the table: every one was an *IR-derived*
predicate asked twice. Verified byte-identical MSL across five GEMM configs.

Original sketch follows.

**Problem.** Predicates like "is B in place", "is A staged", "does this store go
direct", "does this copy store transposed" are each computed in two or three
places, and their inputs (`memdescMap`, `valMap`) are populated progressively,
so the answer depends on *when* you ask.

**Change.** Introduce a `DotFacts` struct, computed exactly once per `tt.DotOp`,
cached by `Operation*`, and containing every derived predicate the rest of the
pipeline consults:

```cpp
struct DotFacts {
  bool aInPlace, bInPlace, aNoStage, bNoStage;
  bool aDirect, bDma, bStoreTransposed;
  bool cDirect, cHasFallback;
  int64_t aPad, bPad, stagedA, stagedB, cReserved;
};
```

Rules that make it work:

- Every field is a pure function of the IR, never of emission order. Anything
  that needs a bound MSL name is *not* a fact — it is an emission detail.
- `scanPool` and `planDot` both read `DotFacts`. Neither recomputes.
- Computed on first request, asserted-consistent on later ones under a debug
  flag.

**This alone would have prevented all four bugs in the table.** It is also the
cheapest step: the predicates already exist, they just need one owner.

**Risk:** low. No codegen change intended; verify with a byte-identical MSL diff
across the corpus.

### Step 2 — extract the MMA block — **NOT DONE, and the premise is wrong**

**The claim below ("the five-step sequence is written out four times") does not
survive reading the emitters.** They are not four copies of one loop; they are
three different *addressing strategies*, and the difference is load-bearing:

| emitter | fragment index                                  | offsets                 | shared helper              |
| ------- | ----------------------------------------------- | ----------------------- | -------------------------- |
| Panel   | C++ `int64_t` (`mi`, `ni`)                      | literals (`mi * 8 * K`) | none                       |
| Direct  | runtime expr (`warpId * mB + r`)                | built AST nodes         | `fragMMABand` + frag cache |
| Fused   | runtime, accumulator persists across the K-loop | built AST nodes         | none                       |

Counted at `d6f39c8`: `fragMMABand`/`clearFragCache` appears **9 times in Direct
and 0 times in Fused and Panel**. So the shared abstraction already exists — it
covers exactly one emitter, because the other two do not have a shape it fits.

An `emitMmaBlock` spanning all three would have to be generic over "is this
index a compile-time constant or an AST expression". Templating it buys nothing;
pushing everything through the expression path would **change Panel's codegen**,
turning literal offsets into expression trees the Metal compiler has to re-fold.
That is a real regression risk taken on to satisfy a line count.

This is the same failure the "What NOT to do" section below warns about, one
level down: forcing one function over paths that differ in structure rather than
in detail produces exactly the boolean soup the plan exists to remove.

**Recommendation: do not do this as stated.** If the duplication becomes
painful, the honest smaller move is to widen `fragMMABand` to cover Fused (which
shares Direct's runtime-index style) and leave Panel alone — two paths, not
four. Measure first that Fused's fragment addressing really is congruent; that
was not verified here.

### Step 3 — split the file — **measured, and smaller than proposed**

Step 2 is withdrawn, so this is the only remaining step. Before doing it I
measured the actual cross-boundary edges at `7221bda`. The proposed four-way
split does not survive them either, though less badly than step 2.

**What the file actually looks like** (4354 lines):

| block                                                  | lines     | ~size |
| ------------------------------------------------------ | --------- | ----- |
| anonymous namespace (IR match helpers)                 | 25–423    | 400   |
| DMA plumbing (`dmaCopyEligible` .. `stageVectorWidth`) | 424–1232  | 810   |
| facts / plan / the four emitters                       | 1233–3030 | 1800  |
| fragment primitives + matchers + policy                | 3031–4354 | 1320  |

**Two problems with the proposed seams:**

1. **The anonymous namespace is shared across the DMA/match boundary.**
   `peelBroadcast` is used once by DMA and 5 times by the matchers;
   `matchDirectStage` twice by DMA and once by the emitters. Splitting forces
   them into a shared header, converting internal-linkage helpers into a
   published interface between files. `DotDma.cpp` would not be self-contained
   after all.

1. **`DotEmit` and `DotMatch` are not two units.** The emit block makes **51
   calls** into what the plan calls `DotMatch`, and 31 of them are the fragment
   primitives (`fragDecl` 10, `accFragDecl` 7, `sgStore` 5,
   `sgMultiplyAccumulate` 5, `sgLoad` 4) — the emitters' own vocabulary, not a
   separate concern. Another 6 are tiling policy (`dotNeedsPanel`,
   `dotColChunk`, `dotCBandRows`, `dotPanelDims`) the emitters must consult.

**What the real seams are**, if this is done at all:

- `DotFragments.cpp` — `sgFragType` .. `sgMultiplyAccumulate` (~65 lines).
  Genuinely a leaf: pure AST builders over `msl::`, no emitter state beyond
  `ctx`. This one is clean and could move today.
- `DotDma.cpp` — the 810-line DMA block, **but only after** `peelBroadcast` and
  `matchDirectStage` move to a shared internal header. That is the price, and it
  should be paid deliberately.
- Everything else stays. `planDot` + the four emitters + the policy helpers they
  call are one unit whether or not they live in one file.

**Recommendation:** move `DotFragments.cpp` if someone wants the win; leave the
rest. A 4354-line file that is *correctly grouped* — and it is, the four blocks
above are contiguous and in dependency order — costs less than a split that
publishes three new internal interfaces. Revisit if the file grows past ~6000
lines or if the DMA path is ever revived (it ships off, see `CLAUDE.local.md`).

**DONE (`2eb18fa`).** The fragment move landed as `EmitMSLDotFragments.cpp` (89
lines; `EmitMSLDot.cpp` 4354 → 4282). It was as clean as predicted: the
declarations were already contiguous in `MSLEmitter.h`, all 45 call sites are
inside `EmitMSLDot.cpp`, and sources are globbed so CMake needed no edit —
nothing new is published between files. Verified byte-identical emitted MSL
across 8 GEMM configs (fp32/fp16/bf16, even-K and ragged, four tilings).

The `DotDma.cpp` half was **not** done, for the reason given above: it requires
promoting `peelBroadcast` and `matchDirectStage` to a shared header. That price
has not been paid, so the DMA block stays put.

Two things worth knowing for whoever picks this up next:

- `EmitMSLDot.cpp` is **not clang-format clean at HEAD**.
  `pre-commit run --all-files` will reformat large regions of it that have
  nothing to do with your change. Format only the files you add.
- The CMake source list is a `file(GLOB ...)` resolved at configure time, so a
  newly added `.cpp` is silently **not compiled** until you re-run
  `cmake -B build`. `ninja` alone relinks the stale objects and looks like it
  succeeded; the resulting dylib then fails at runtime in a way that looks like
  a codegen bug. Re-run cmake when adding a file.

## What NOT to do

- **Do not unify the four emitters into one.** They differ in loop structure,
  not in detail. `emitDotScalar` has no fragments at all. Forcing one function
  produces exactly the boolean-soup this plan is trying to remove.
- **Do not touch the DMA path's behaviour.** It ships off and is a measured loss
  (see the async-copy notes in `CLAUDE.local.md`). Move it, do not tune it.
- **Do not delete the `panel` path** on the theory that panelisation is
  subsumed. It is reachable and tested; the 64x128 split exists because the
  whole tile does not fit the 32KB cap.

## Verification for every step

1. `test_ragged_gemm.py` — 294 cases, currently 0 failures, must stay 0.
1. Default-path MSL byte-identical where no behaviour change is intended.
   Compare emitted MSL hashes across a fixed kernel set before and after.
1. The corpus rebench (see `REBENCH_2026_08_11.md`) as the perf gate — no row
   may regress beyond its measured spread.

## Outcome

Step 1 said it was worth doing even if 2 and 3 never happened. That turned out
to be the whole plan: **step 1 landed (`d6f39c8`), step 2 was withdrawn as
false, step 3 shrank to one 65-line move — which landed in `2eb18fa`.** The plan
is now fully executed, and what it delivered is step 1 plus a file move.

The reason is the same in both withdrawals. Step 1 was derived from four
observed bugs — it described something that had actually gone wrong four times.
Steps 2 and 3 were derived from reading line counts and guessing at structure,
and both dissolved on contact with the call graph: the emitters do not share an
MMA shape, and the file's blocks do not separate without publishing new internal
interfaces.

Worth remembering when the file next "feels convoluted": measure the edges
(`fragMMABand` usage, cross-block call counts, anon-namespace sharing) before
proposing a shape. Those measurements took under an hour and refuted two of
three steps.
