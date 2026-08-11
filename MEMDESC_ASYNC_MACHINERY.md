# Memdesc and async-copy lowering in `EmitMSLOps.cpp`

Status: description of current behaviour at `590343f`, written 2026-08-11.
Companion to `POOL_MACHINERY.md` (which covers how the pool is *sized*) and
`DOT_REFACTOR_PLAN.md` (which covers the dot emitters). This one covers the
other half of `emitMemDesc` — the seven `ttg` ops that move tiles between
registers, threadgroup memory and device memory.

Everything here lives in one function, `MSLEmitter::emitMemDesc`
(`EmitMSLOps.cpp:1682-2180`, ~500 lines), plus the async-wait handler at
`EmitMSLOps.cpp:1067-1125`.

## The one-paragraph version

Triton's memory model has an explicit shared-memory tier: `local_alloc` makes a
buffer, `local_store`/`local_load` move registers in and out of it,
`async_copy_global_to_local` fills it straight from device, and `async_wait`
makes the fill visible. MSL has threadgroup arrays and one barrier primitive.
The mapping is mostly direct — the difficulty is entirely in **when barriers go
in** and **which token a wait waits on**, because we lower an asynchronous
operation onto a synchronous one and the pipeliner assumes the asynchrony.

## The op table

| op | lowering | emits text? |
|---|---|---|
| `memdesc_index` | rebind in `memdescMap` | no |
| `memdesc_subslice` | rebind in `memdescMap` | no |
| `local_alloc` | `threadgroup T buf[N]` + optional init scatter | yes |
| `local_store` | per-register scatter + `barrier` | yes |
| `local_load` | per-register gather | yes |
| `async_copy_global_to_local` | DMA shim call, **or** masked per-thread stage | yes |
| `async_commit_group` | nothing | no |
| `async_wait` | `dmaWait` per resolved token + `barrier` | yes |
| `convert_layout` | rebind, shuffle, or threadgroup round-trip | depends |

The first two are the reason `memdescMap` is a map rather than a naming
convention: an index or subslice produces a *new SSA value* denoting a region of
an existing buffer, and the only record of that relationship is the map entry
(`{buf, baseOffset, strides}`).

## Where the subtlety actually is

### 1. `local_alloc` is pre-declared, and that is load-bearing

`emitFunc` pre-registers every allocation before the body walk
(`EmitMSLFunc.cpp`), so `planDot` can see allocations that the walk has not
reached yet. `emitMemDesc` therefore only mints a buffer if the prescan did not
run (device functions):

```cpp
if (!memdescMap.count(la.getResult())) { ... }
```

This is the same phase-ordering hazard `DOT_REFACTOR_PLAN.md` catalogues: a
predicate answered during Decl differs from the same predicate answered during
the body walk. Here it is handled by making the earlier phase authoritative.

### 2. The async copy is lowered *synchronously*, so it needs a fence the IR
does not ask for

On hardware `async_copy_global_to_local` lands at the matching `async_wait`, so
the pipeliner freely issues a copy into a slot the current trip is still
reading — with `numBuffers == 1` it always does. We lower it to a plain store,
which would clobber that slot immediately. Hence the leading fence at
`EmitMSLOps.cpp:1767`:

```cpp
if (!asyncCopyFenced && !barrierCoversTail(body))
  body.push_back(ctx.hardBarrier(false));
asyncCopyFenced = true;
```

`asyncCopyFenced` is a **batch flag**, not a per-copy flag: Apple's own M1 GEMM
issues its whole batch of copies back to back and waits once, so the requests
overlap in the queue. Fencing between members of a batch would separate two
staging writes that nothing reads in between. The flag is cleared by
`async_wait` (`EmitMSLOps.cpp:1120`), which is what closes a batch.

There is no barrier *after* a copy, deliberately (`EmitMSLOps.cpp:1990-1994`) —
the visibility point is the wait.

**This fence has been removed once and it was a race** (`cb7173d`, restored in
`7c9d880`: six launches, six different answers). Do not re-derive that.

### 3. `async_wait` resolves tokens through the IR, not through a pending list

The natural implementation — push each issued token onto a list, drain it at the
wait — is wrong here, because **the loop body is walked more than once** (Decl
phase then MMA phase). The first walk drains the list and the second sees it
empty. So the wait re-derives its tokens from the IR every time, via
`dmaHandleFor` keyed by copy site (`EmitMSLOps.cpp:1085-1112`).

The resolution itself is a small fixpoint over block arguments:

```cpp
if (auto arg = dyn_cast<BlockArgument>(tok)) {
  collect(yield.getOperand(arg.getArgNumber() - 1));   // later trips
  collect(forOp.getInitArgs()[arg.getArgNumber() - 1]); // trip 0
  return;
}
```

Both edges matter. A token reaches the wait as a loop iter-arg; on trip 0 that
iter-arg still holds the peeled prologue's commit, and a rotating pipeline
yields a *different* stage's token into it. Following only the yield leaves
trip 0 reading a tile whose copy was never waited on.

### 4. Token identity is keyed by (allocation, pipeline slot)

`EmitMSLFunc.cpp:849-865`. The obvious key — the allocation — is wrong: a
depth-2 pipeline aims two prologue copies and one in-loop copy at one allocation
but at **two different iter-args**. Keying on the allocation collapses them onto
one token, the last issue overwrites the others, and every trip's wait blocks on
the copy it was supposed to be overlapping. Keying on `dmaStageSlot(ac)` as well
is the fix (`1eef2f7`).

Conversely the token must be declared in the **function prologue**, not at the
issue site, because the wait may sit in an outer scope than the issue; and there
is one token per copy *site*, reused every trip, because minting a fresh name
per emission leaves the loop waiting on the prologue's stale tokens.

That is three distinct scoping rules for one `ulong`, and all three were bugs
first.

### 5. The non-DMA staging path mirrors `tt.load` — including its two vector
gates

When the copy does not qualify for the shim it becomes a masked per-thread
stage. Two independent widths have to agree before a wide store is legal:

- `accessVectorWidth(srcTy, ac.getSrc())` — the **device** addresses of a
  register run are contiguous.
- `slotStride == 1` — the **threadgroup slots** are contiguous too
  (`EmitMSLOps.cpp:1874-1892`).

The second is not implied by the first. A column-major fp32 operand has
`sizePerThread [4,1]`, so its four consecutive registers land 64 slots apart:
the device load is still one wide load, but the scatter must go per lane. This
was bug 4 of the four listed in `CLAUDE.local.md`'s pipeliner section.

Masked copies must also honour `other` — leaving a masked-off slot untouched
keeps the previous trip's tile, which with rotating buffers is *another tile's
data*. Observed as NaN on a boundary-masked pipelined GEMM.

The `ms.size() == 1` special case (`EmitMSLOps.cpp:1948`) exists because when
every register sits under one predicate, the hot guard already *is* the whole
condition — emitting the else arm produces the unreachable
`if (m) {...} else { if (m) {...} }`, which doubled the device load count.

## `convert_layout`: three banding modes and the one that hurts

`EmitMSLOps.cpp:2000-2177`. In order:

1. **Identical linear layouts** → rebind, no text (`2012`).
2. **Intra-warp permutation** → `emitIntraWarpShuffleConvert`, no threadgroup
   traffic, no barrier (`2019`).
3. **Threadgroup round-trip** — everything else, and it is `mslReject`-logged as
   `"threadgroup-roundtrip"` because reaching it is a missed opportunity.

The round-trip has three sub-modes chosen by whether the tile fits `poolBytes`:

| condition | mode | lines |
|---|---|---|
| `tileBytes > cap && rank >= 2` | row-banded | 2048-2113 |
| `tileBytes > cap` (rank 1) | reshape-banded | 2115-2159 |
| fits | whole tile | 2161-2175 |

**The banded modes are task #251.** Both emit, per band and per register, a
guarded scalar store:

```cpp
body.push_back(ctx.compactIf(cond, ctx.assignStmt(
    ctx.subscript(ctx.var(buf), bandOffset(srcTy, r, r0)), scatterVal(...))));
```

so a tile needing `B` bands emits `B × regCount` predicated scalar accesses on
each side, against `regCount` unpredicated ones for the fitting case. The
measured symptom is 18 → 1026 threadgroup stores with the shuffle count
unchanged. The whole-tile mode two blocks below does exactly the same work with
a plain unguarded subscript — the banding is the only difference.

Three things stand out as fixable without changing the algorithm:

- **The band predicate is loop-invariant in `r0` but recomputed per register.**
  `layoutCoordExpr(srcTy, r, srcRowDim)` is emitted twice per register per band
  (once for `Ge`, once for `Lt`, lines 2084-2088) and the Metal compiler has to
  re-CSE all of it.
- **No vectorisation at all.** The fitting path can and does use wide accesses;
  the banded path is scalar by construction even when a run of registers lands
  in one band contiguously.
- **The rank-1 reshape mode re-declares `__f` inside a fresh `plainScope` per
  register** (2127-2139), i.e. one scope per register per band.

None of this is a correctness issue and none of it is on the GEMM path — it is
reached by oversized-tile conversions, which is why it took a large-last-dim
reduction to surface it.

## Shared mutable state, and why it is a hazard

Five members of `MSLEmitter` are read and written across these paths:

| member | written by | read by |
|---|---|---|
| `memdescMap` | `emitFunc` prescan, `local_alloc`, index/subslice | every path + `planDot` |
| `asyncCopyFenced` | copy (set), wait (clear) | copy |
| `dmaHandleFor` | `emitFunc` prescan | copy, wait |
| `tgScratchId` | `local_alloc` | `local_alloc` |

There was a fourth, `pendingDmaHandles`, **removed in this commit**: it was
pushed at the copy and cleared at the wait, and no path ever read it. It was the
residue of the pending-list design that the double-walk defeated (point 3
above), and its name actively misled — it reads as the wait's source of truth,
which is exactly what it is not.

## Suggested work, in order of value

1. **Hoist the band predicate** in the row-banded convert (#251). The
   `layoutCoordExpr` call is emitted twice per register per band; computing the
   row coordinate once into a named local is a mechanical change.
2. **Vectorise the banded scatter/gather** where a register run lands
   contiguously within one band — reuse the `slotStride` test the async-copy
   path already has (point 5 above). This is the actual fix for #251 and is the
   larger job.
3. **Consider whether the round-trip is needed at all** for the shapes that hit
   it. `mslReject(cl, "convertLayout", "threadgroup-roundtrip")` already logs
   every occurrence under `MSL_LOG_REJECT`; a census over the corpus would say
   whether mode 3 is reached by anything but oversized reductions.

## Reading order for someone new to this

1. `EmitMSLOps.cpp:1682` — `emitMemDesc`, the dispatch
2. `EmitMSLOps.cpp:1754` — the async copy, both arms
3. `EmitMSLFunc.cpp:838` — where its tokens are minted, and the slot key
4. `EmitMSLOps.cpp:1067` — the wait, and the iter-arg fixpoint
5. `EmitMSLOps.cpp:2000` — `convert_layout` and its three banding modes
