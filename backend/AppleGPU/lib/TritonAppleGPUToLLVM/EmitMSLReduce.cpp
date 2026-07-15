// EmitMSLReduce.cpp - reduce / scan / map_elementwise lowering.
//
// AST builders for the reduce/scan/map families, dispatched from astEmitOp. The
// cross-lane warp-carry sequences use the simd_shuffle family via builtin::simd;
// the map_elementwise unstructured region is modelled as a real
// StateMachineScope node (MSL forbids goto), the scan sweep loops as guarded
// IfScope updates.
//
// INVARIANT: the printer inserts no grouping parens; a builder inserts an
// explicit ctx.paren(...) wherever a subexpression needs precedence grouping.
// A builder must never touch nextId/indent, so builders minting fresh names run
// over a private LocalGen id counter seeded from the emitter's current nextId
// without mutating it.

#include "MSLConstants.h"
#include "MSLEmitter.h"

using namespace mlir;

namespace mlir::triton::applegpu {

namespace bsimd = msl::builtin::simd;
using B = msl::BinOp;

//===----------------------------------------------------------------------===//
// astCombineAssign - bind the combined value into a target
//===----------------------------------------------------------------------===//

// dst = res;  (the single-result combine tail).
msl::Stmt *MSLEmitter::astCombineAssign(StringRef dst, StringRef res) {
  return ctx.assignStmt(ctx.var(dst), ctx.var(res));
}

//===----------------------------------------------------------------------===//
// astReduce* - per-group intra-register fold, cross-lane, cross-warp
//===----------------------------------------------------------------------===//

// sc acc = src;  (the group accumulator seed, and the wacc/grand seeds).
msl::Stmt *MSLEmitter::astReduceAccInit(StringRef sc, StringRef acc,
                                        StringRef src) {
  return ctx.declStmt(ctx.named(sc), acc, ctx.var(src));
}

// acc = out;  (write-back after a combine step).
msl::Stmt *MSLEmitter::astReduceAccStep(StringRef acc, StringRef out) {
  return ctx.assignStmt(ctx.var(acc), ctx.var(out));
}

// The cross-lane butterfly operand: simd_shuffle_xor(acc, <m>u).
msl::Expr *MSLEmitter::astReduceLaneShuffle(StringRef sc, StringRef acc,
                                            unsigned m) {
  return astShuffleExpr(bsimd::ShuffleXor, sc, acc,
                        std::to_string(m) + "u");
}

// threadgroup sc* scratch = <poolRegion>;
msl::Stmt *MSLEmitter::astReduceScratchDecl(StringRef sc, StringRef scratch,
                                            int64_t byteOff) {
  msl::Type *tgptr = ctx.ptr(ctx.named(sc), msl::AddrSpace::Threadgroup);
  return ctx.declStmt(tgptr, scratch, astPoolRegion(byteOff, sc));
}

// scratch[warpId * 32 + laneId] = acc;
msl::Stmt *MSLEmitter::astReduceScratchStore(StringRef scratch, StringRef acc) {
  msl::Expr *slot = ctx.binary(
      B::Add, ctx.binary(B::Mul, ctx.var(warpId), ctx.lit("32")),
      ctx.var(laneId));
  return ctx.assignStmt(ctx.subscript(ctx.var(scratch), slot), ctx.var(acc));
}

// The cross-warp base index ((warpId & <~warpMask>) * 32 + laneId), spelled as
// a paren so scratch[base + K] keeps the base parenthesised.
msl::Expr *MSLEmitter::astReduceWarpBase(unsigned warpMask) {
  msl::Expr *masked = ctx.paren(ctx.binary(
      B::And, ctx.var(warpId), ctx.lit(std::to_string(~warpMask))));
  return ctx.paren(ctx.binary(
      B::Add, ctx.binary(B::Mul, masked, ctx.lit("32")), ctx.var(laneId)));
}

// sc wacc = scratch[base];  /  sc wv = scratch[base + <off>];
msl::Stmt *MSLEmitter::astReduceWarpLoad(StringRef sc, StringRef dst,
                                         StringRef scratch, msl::Expr *base,
                                         int off) {
  msl::Expr *idx = off == 0
                       ? base
                       : ctx.binary(B::Add, base, ctx.lit(std::to_string(off)));
  return ctx.declStmt(ctx.named(sc), dst, ctx.subscript(ctx.var(scratch), idx));
}

//===----------------------------------------------------------------------===//
// astScan* - lane sweep, guarded prefix, cross-run carry
//===----------------------------------------------------------------------===//

// sc acc = src;  (per-register scan accumulator seed).
msl::Stmt *MSLEmitter::astScanAccInit(StringRef sc, StringRef acc,
                                      StringRef src) {
  return ctx.declStmt(ctx.named(sc), acc, ctx.var(src));
}

// sc laneScan = acc;  (the lane-scan running value seed).
msl::Stmt *MSLEmitter::astScanLaneSeed(StringRef sc, StringRef laneScan,
                                       StringRef acc) {
  return ctx.declStmt(ctx.named(sc), laneScan, ctx.var(acc));
}

// The lane sweep shuffle: simd_shuffle_up / simd_shuffle_down of laneScan by
// <delta>u. `rev` picks _down.
msl::Expr *MSLEmitter::astScanLaneShuffle(bool rev, StringRef sc,
                                          StringRef val, unsigned delta) {
  return astShuffleExpr(rev ? bsimd::ShuffleDown : bsimd::ShuffleUp, sc, val,
                        std::to_string(delta) + "u");
}

// The axis-local lane predicate: `(laneId & <axisLaneMask>) >= <delta>` for a
// forward sweep, `<= <axisLaneMask - delta>` for reverse. Returned parens keep
// the local subexpression grouping.
msl::Expr *MSLEmitter::astScanLaneGuard(bool rev, unsigned axisLaneMask,
                                        unsigned delta) {
  msl::Expr *local = ctx.paren(ctx.binary(
      B::And, ctx.var(laneId), ctx.lit(std::to_string(axisLaneMask))));
  return rev ? ctx.binary(B::Le, local,
                          ctx.lit(std::to_string(axisLaneMask - delta)))
             : ctx.binary(B::Ge, local, ctx.lit(std::to_string(delta)));
}

// dst = (guard ? out : dst);  the guarded lane-sweep / prefix write-back.
msl::Stmt *MSLEmitter::astScanGuardedUpdate(StringRef dst, msl::Expr *guard,
                                            StringRef out) {
  return ctx.assignStmt(
      ctx.var(dst),
      ctx.paren(ctx.ternary(guard, ctx.var(out), ctx.var(dst))));
}

// The top lane of a scan partition:
//   axisLaneMask == 0 -> laneId
//   forward           -> ((laneId & ~axisLaneMask) | axisLaneMask)
//   reverse           -> ((laneId & ~axisLaneMask) | 0)
msl::Expr *MSLEmitter::astScanAxisTopLane(bool rev, unsigned axisLaneMask) {
  if (axisLaneMask == 0)
    return ctx.var(laneId);
  msl::Expr *masked = ctx.paren(ctx.binary(
      B::And, ctx.var(laneId), ctx.lit(std::to_string(~axisLaneMask))));
  return ctx.paren(ctx.binary(
      B::Or, masked, ctx.lit(rev ? "0" : std::to_string(axisLaneMask))));
}

//===----------------------------------------------------------------------===//
// astWarpCarry* - cross-warp partition scan
//===----------------------------------------------------------------------===//

// runTotal = simd_shuffle(laneScan, axisTopLane);  (the no-warp-bits carry).
msl::Stmt *MSLEmitter::astWarpCarryLaneOnly(StringRef sc, StringRef runTotal,
                                            StringRef laneScan,
                                            StringRef axisTopLane) {
  msl::Expr *sh = astShuffleExpr(bsimd::Shuffle, sc, laneScan, axisTopLane);
  return ctx.assignStmt(ctx.var(runTotal), sh);
}

// The top-lane store guard: `true` when axisTopLane is laneId itself, else
// `laneId == axisTopLane`.
msl::Expr *MSLEmitter::astWarpCarryTopGuard(StringRef axisTopLane) {
  if (axisTopLane == laneId)
    return ctx.lit("true");
  return ctx.binary(B::Eq, ctx.var(laneId), ctx.var(axisTopLane));
}

// if (topGuard) scratch[warpId * 32 + laneId] = laneScan;
msl::Stmt *MSLEmitter::astWarpCarryTopStore(msl::Expr *topGuard,
                                            StringRef scratch,
                                            StringRef laneScan) {
  msl::Expr *slot = ctx.binary(
      B::Add, ctx.binary(B::Mul, ctx.var(warpId), ctx.lit("32")),
      ctx.var(laneId));
  msl::Block then;
  then.push_back(
      ctx.assignStmt(ctx.subscript(ctx.var(scratch), slot), ctx.var(laneScan)));
  return ctx.ifScope(topGuard, std::move(then));
}

// The partition base ((warpId & <~axisWarpMask>) * 32 + axisTopLane), parenned
// to mirror scratch[base + K].
msl::Expr *MSLEmitter::astWarpCarryBase(unsigned axisWarpMask,
                                        StringRef axisTopLane) {
  msl::Expr *masked = ctx.paren(ctx.binary(
      B::And, ctx.var(warpId), ctx.lit(std::to_string(~axisWarpMask))));
  return ctx.paren(ctx.binary(
      B::Add, ctx.binary(B::Mul, masked, ctx.lit("32")), ctx.var(axisTopLane)));
}

// int myPart = <warpPos>;  where warpPos ORs the axis-warp mask bits down into
// contiguous partition-index bits: ((((warpId >> b) & 1) << r) | ...).
msl::Stmt *MSLEmitter::astWarpCarryMyPart(StringRef myPart,
                                          ArrayRef<int> maskBits) {
  auto term = [&](int b, int r) -> msl::Expr * {
    msl::Expr *shifted = ctx.paren(
        ctx.binary(B::Shr, ctx.var(warpId), ctx.lit(std::to_string(b))));
    msl::Expr *bit = ctx.paren(ctx.binary(B::And, shifted, ctx.lit("1")));
    return ctx.paren(ctx.paren(
        ctx.binary(B::Shl, bit, ctx.lit(std::to_string(r)))));
  };
  msl::Expr *pos = term(maskBits[0], 0);
  for (size_t r = 1; r < maskBits.size(); ++r)
    pos = ctx.paren(ctx.binary(B::Or, pos, term(maskBits[r], (int)r)));
  return ctx.declStmt(ctx.scalar(msl::Scalar::I32), myPart, pos);
}

// The per-warp exclusive-prefix partition predicate: `myPart < <p>` (reverse)
// or `myPart > <p>` (forward).
msl::Expr *MSLEmitter::astWarpCarryPartCond(bool rev, StringRef myPart, int p) {
  return ctx.binary(rev ? B::Lt : B::Gt, ctx.var(myPart),
                    ctx.lit(std::to_string(p)));
}

// bool init = false;  (the exclusive-prefix seen-first flag).
msl::Stmt *MSLEmitter::astWarpCarryInitFlag(StringRef init) {
  return ctx.declStmt(ctx.scalar(msl::Scalar::I1), init, ctx.lit("false"));
}

// acc = (init ? out : acc);  the guarded per-register carry application.
msl::Stmt *MSLEmitter::astWarpCarryApply(StringRef acc, StringRef init,
                                         StringRef out) {
  return ctx.assignStmt(
      ctx.var(acc),
      ctx.paren(ctx.ternary(ctx.var(init), ctx.var(out), ctx.var(acc))));
}

//===----------------------------------------------------------------------===//
// astMap* - state-machine dispatch of a CFG region
//===----------------------------------------------------------------------===//

// sc capture;  (predeclared multi-block result slot).
msl::Stmt *MSLEmitter::astMapCaptureDecl(StringRef sc, StringRef name) {
  return ctx.declStmt(ctx.named(sc), name);
}

// capture = operand;  the map_elementwise.return spill into a caller slot.
msl::Stmt *MSLEmitter::astMapReturnSpill(StringRef capture, StringRef operand) {
  return ctx.assignStmt(ctx.var(capture), ctx.var(operand));
}

// dst = src;  a hoisted cross-block value copy inside a state case.
msl::Stmt *MSLEmitter::astMapHoistCopy(StringRef dst, StringRef src) {
  return ctx.assignStmt(ctx.var(dst), ctx.var(src));
}

// Assemble the full `int state = 0; while (true) { if (state==L){..} ... }`
// dispatch as one StateMachineScope. `cases` supplies each block's label and its
// already-built body block (op emission + spill/break tails live in the body).
msl::Stmt *MSLEmitter::astMapCFGStateMachine(
    StringRef state, ArrayRef<std::pair<std::string, msl::Block>> cases) {
  llvm::SmallVector<msl::StateMachineScope::Case, 4> smCases;
  for (auto &c : cases)
    smCases.push_back({ctx.save(c.first), c.second});
  return ctx.stateMachineScope(ctx.scalar(msl::Scalar::I32), state,
                               std::move(smCases));
}

} // namespace mlir::triton::applegpu
