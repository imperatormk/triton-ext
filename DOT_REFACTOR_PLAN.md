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

| bug | the two disagreeing sides |
|---|---|
| `739091a` | `scanPool`'s panel gate vs `planDot`'s |
| `ad518bf` | `scanPool` sizing the pool vs `planDot` laying it out |
| `cb7173d` | `computeDmaStoreTransposed` asked during planning vs during emission |
| `1eef2f7` | `dotStageRowPads` asked with `bInPlace` in the Decl phase vs the MMA phase |

That is a structural defect with a name: **the same predicate is evaluated more
than once, at points where its inputs are not yet equal.** Everything below is
aimed at that, not at line count.

## What the file actually contains

Four terminal emitters, dispatched from `emitDot` (line 1442):

| emitter | lines | role |
|---|---|---|
| `emitDotDirect` | 293 | one whole-tile MMA block, no K-loop fusion |
| `emitDotFused` | 712 | the K-loop GEMM: staging, rotation, epilogue |
| `emitDotPanel` | 162 | 64x128 split into two 64x64 N-panels |
| `emitDotScalar` | 112 | fallback, no simdgroup matrices |

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
`DotFacts` therefore holds only the IR-derived half (shape, element widths,
A/B residency, in-place/no-stage, device-direct A, the phase-free B DMA
candidate). The budget-dependent fields the sketch below lists — `aPad`,
`bPad`, `stagedA`, `stagedB`, `cReserved` — stay with whichever pass decides
them, because `scanPool` necessarily works against a provisional budget.

That still covers all four bugs in the table: every one was an *IR-derived*
predicate asked twice. Verified byte-identical MSL across five GEMM configs.

Original sketch follows.


**Problem.** Predicates like "is B in place", "is A staged", "does this store go
direct", "does this copy store transposed" are each computed in two or three
places, and their inputs (`memdescMap`, `valMap`) are populated progressively, so
the answer depends on *when* you ask.

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

| emitter | fragment index | offsets | shared helper |
|---|---|---|---|
| Panel | C++ `int64_t` (`mi`, `ni`) | literals (`mi * 8 * K`) | none |
| Direct | runtime expr (`warpId * mB + r`) | built AST nodes | `fragMMABand` + frag cache |
| Fused | runtime, accumulator persists across the K-loop | built AST nodes | none |

Counted at `d6f39c8`: `fragMMABand`/`clearFragCache` appears **9 times in
Direct and 0 times in Fused and Panel**. So the shared abstraction already
exists — it covers exactly one emitter, because the other two do not have a
shape it fits.

An `emitMmaBlock` spanning all three would have to be generic over "is this
index a compile-time constant or an AST expression". Templating it buys
nothing; pushing everything through the expression path would **change Panel's
codegen**, turning literal offsets into expression trees the Metal compiler has
to re-fold. That is a real regression risk taken on to satisfy a line count.

This is the same failure the "What NOT to do" section below warns about, one
level down: forcing one function over paths that differ in structure rather
than in detail produces exactly the boolean soup the plan exists to remove.

**Recommendation: do not do this as stated.** If the duplication becomes
painful, the honest smaller move is to widen `fragMMABand` to cover Fused
(which shares Direct's runtime-index style) and leave Panel alone — two paths,
not four. Measure first that Fused's fragment addressing really is congruent;
that was not verified here.

### Step 3 — split the file

Now the *only* remaining step, since step 2 is withdrawn. It no longer depends
on step 2: `DotMatch` and `DotDma` were already near-separable, and step 1
landing means the fact computation has an owner. The four emitters move as they
are.

- `DotFacts.cpp` — the fact computation from step 1
- `DotMatch.cpp` — `matchTileIndex` .. `matchDirectStore`, pure IR pattern
  matching with no emitter state
- `DotDma.cpp` — the async-copy staging path (~900 lines, already cohesive)
- `DotEmit.cpp` — the four emitters plus the shared MMA block

`DotMatch` and `DotDma` are already near-separable; the other two are not until
step 1 exists.

**Risk:** low, mechanical. Do it last so the moves are pure.

## What NOT to do

- **Do not unify the four emitters into one.** They differ in loop structure,
  not in detail. `emitDotScalar` has no fragments at all. Forcing one function
  produces exactly the boolean-soup this plan is trying to remove.
- **Do not touch the DMA path's behaviour.** It ships off and is a measured
  loss (see the async-copy notes in `CLAUDE.local.md`). Move it, do not tune it.
- **Do not delete the `panel` path** on the theory that panelisation is
  subsumed. It is reachable and tested; the 64x128 split exists because the
  whole tile does not fit the 32KB cap.

## Verification for every step

1. `test_ragged_gemm.py` — 294 cases, currently 0 failures, must stay 0.
2. Default-path MSL byte-identical where no behaviour change is intended.
   Compare emitted MSL hashes across a fixed kernel set before and after.
3. The corpus rebench (see `REBENCH_2026_08_11.md`) as the perf gate — no row
   may regress beyond its measured spread.

## Sequencing note

Step 1 is worth doing even if 2 and 3 never happen. It is the step that maps
directly onto the observed failure mode, and it is the one that makes the other
two safe.
