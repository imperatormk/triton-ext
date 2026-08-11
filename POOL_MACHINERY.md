# The threadgroup pool: how it is sized, laid out, and how it goes wrong

Status: description of current behaviour at `1eef2f7`, written 2026-08-11.
No changes proposed here beyond the ones already in `DOT_REFACTOR_PLAN.md`;
this exists because the pool is where three of last week's four bugs lived and
there was no single place that explained it.

## The one-paragraph version

There is exactly one threadgroup allocation per kernel, `threadgroup char
__pool[N]`, emitted in the function prologue. `N` is decided by a **sizing**
pass that walks every op and takes a maximum. Every consumer then **lays itself
out** inside that block by computing byte offsets independently. Sizing and
layout are separate code, run at different times, over different information —
and nothing checks that the second agrees with the first except a late
assertion.

## Sizing: `scanPool`

`EmitMSLFunc.cpp:819` runs `func.walk([&](Operation *op) { scanPool(op); })`,
then declares the pool if `poolBytes > 0` (`EmitMSLFunc.cpp:820-824`).

`scanPool` (`EmitMSLMemory.cpp:173`) is a per-op switch that accumulates:

```
poolBytes = std::max(poolBytes, <this op's requirement>)
```

Contributors, each with its own sizing rule:

| op | rule |
|---|---|
| `ttg.ConvertLayoutOp` | `tgScratchBytes(srcTy, band2D=true)`, after four early-outs |
| `tt.TransOp` | `tgScratchBytes(resTy, band2D=false)` |
| `tt.CatOp` | full tile, no banding |
| `tt.ReshapeOp` | `tgScratchBytes(resTy, band2D=false)` |
| `tt.DotOp` | the complicated one — see below |
| `tt.ReduceOp` | `numWarps * 32 * elemBytes` per operand, cross-warp stage only |
| `tt.ScanOp` | full tile |

Because it is a max, **the largest single consumer sets the pool for the whole
kernel**, and a kernel with one oversized `convert_layout` pays that everywhere.

### `tgScratchBytes` and banding

`tgScratchBytes` (`EmitMSLMemory.cpp:155`) is the escape valve: if the whole
tile fits in `poolBudget()` it returns the tile; otherwise it **bands**, either
by rows (`band2D`) or by reshaping into equal chunks (`reshapeBandElems`). This
is what produces the `if (__f >= 8192 && __f < 16384)` guards seen in task #251
— banding is correct but its emission is per-element.

### `poolBudget()` — the policy knob

```cpp
int64_t MSLEmitter::poolBudget() const {
  int64_t b = kTGResidentBudgetBytes - liveTgBytes;   // 32768 - live
  return b < 0 ? 0 : b;
}
```

`kTGResidentBudgetBytes` is **32768, the hard per-threadgroup cap**. So the
emitter sizes to *one* resident threadgroup by construction and never trades
footprint for occupancy.

Measured 2026-08-11: **this policy never binds on the default path.** Base
GEMM pools across six configs are 8704 / 8704 / 16384 / 16384 / 4352 / 4352 —
4 to 15 threadgroups per core, nowhere near the cap. The occupancy-aware
machinery (`kTGCoreBudgetBytes`, `tgResidency()`, `tgPoolForResidency()`,
`kTGResidencyFloor`) already exists in `MSLConstants.h` and is consulted by
`dotStageRowPads`. So lowering the constant is *not* the free win it looks
like; the kernels that blow the cap are the pipeliner's `ttg.local_alloc`
buffers, which bypass `poolBudget()` entirely because they are sized upstream.

## Layout: `planDot` and friends

For a dot, `planDot` (`EmitMSLDot.cpp:1233`) decides where A, B and C live
*inside* the pool: `stagedA`, `stagedB`, `stagedAB`, `bandRows`, `disjointC`,
plus the pads `aPad`/`bPad`. `dotPoolPtrs` then emits the three pointers via
`poolRegion(byteOffset, scalarType)`.

`poolRegion` (`EmitMSLMemory.cpp:51`) is a pure cast — `((threadgroup T*)(__pool
+ off))`. It does no bounds checking and cannot: it does not know `poolBytes`.

## The failure mode

The sizing pass and the layout pass answer the same questions **separately**:

- `scanPool` asks "how big must the pool be for this dot?"
- `planDot` asks "where inside the pool does each operand go?"

Both need to know whether A is staged, whether B is in place, whether C stores
direct, whether the panel path applies. Both compute those predicates from
scratch. When their inputs differ — because `memdescMap` is populated
progressively, or because a phase runs before the body walk — they disagree,
and the layout overruns the size.

Four bugs, all this shape:

| commit | disagreement |
|---|---|
| `739091a` | panel gate: `scanPool` said panel, `planDot` said whole-tile |
| `ad518bf` | both called `poolBudget()`, double-counting the same budget |
| `cb7173d` | a transposed-store query answered differently while planning vs emitting |
| `1eef2f7` | A's pad sized with `bInPlace`, which is false during the Decl phase |

### The safety net, and the one thing wrong with it

`checkPoolRegions` (`EmitMSLDot.cpp:1519`) is better than I first assumed when
writing this file. It checks **both** failure modes:

- **overrun** — any region extending past `poolBytes` (lines 1520-1528)
- **overlap** — any two simultaneously live regions intersecting (lines
  1529-1544), which is the silent-miscompile case no count-based census can
  see, since MMA and accumulator counts stay correct while one region's stores
  land inside another's tile

Both produce a precise message naming the region, its byte extent, and what it
collided with. `checkDotPoolRegions` feeds it the three live extents and
correctly excludes the panel path (which carries its own check) and an overlaid
C (not live alongside A and B).

**The defect is that the message is thrown away by default.** `mslReject`
(`MSLReject.cpp:131`) opens with:

```cpp
if (!mslLogReject())
  return;
```

so unless `MSL_LOG_REJECT` is set, the diagnosis is computed and discarded,
while `emitFailed = true` still unwinds to `emitOp` returning false — which
surfaces as **`EmitMSL: unhandled op 'scf.for'`**, naming the enclosing loop
rather than the dot whose budget arithmetic was wrong.

That is exactly what happened with `1eef2f7`: the emitter knew precisely what
was wrong (`A[0,18432) past pool 16384`) and said nothing useful. Hours went
into "why can't the emitter handle this loop" when the answer was one env var
away.

**Suggested fix:** a check that sets `emitFailed` is not a routing rejection,
it is a bug in our own arithmetic — it should print unconditionally. Either
call `llvm::errs()` directly in `checkPoolRegions`, or give `mslReject` a
severity so hard failures bypass the `mslLogReject()` gate while ordinary
"this gate declined" messages stay behind it.

## What would actually fix it

`DOT_REFACTOR_PLAN.md` step 1: compute the predicates **once** per dot into a
`DotFacts` struct, cached by `Operation*`, and have both `scanPool` and
`planDot` read it rather than recompute. Every field must be a pure function of
the IR, never of emission order — anything needing a bound MSL name is an
emission detail, not a fact.

Two smaller improvements that stand alone:

- Make `checkPoolRegions` name the **dot**, not the enclosing op, and say which
  region overran by how much. Pure diagnostics, no behaviour change.
- Extend it to detect **overlap** between live regions, not just overrun. That
  is the silent-miscompile case the current check cannot see.

## Reading order for someone new to this

1. `EmitMSLMemory.cpp:173` `scanPool` — how the size is decided
2. `EmitMSLMemory.cpp:155` `tgScratchBytes` — the banding escape valve
3. `EmitMSLMemory.cpp:134` `poolBudget` — the policy, and its 32768 pin
4. `EmitMSLDot.cpp:1233` `planDot` — how a dot lays itself out
5. `EmitMSLDot.cpp:1519` `checkPoolRegions` — the net, and its blind spot
