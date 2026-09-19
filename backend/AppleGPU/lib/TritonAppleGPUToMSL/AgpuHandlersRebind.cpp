// Shape op handlers: expand_dims, broadcast, convert_layout, trans, reshape.
// A shape change that leaves every element in the register holding it is a
// rename; one that moves elements between threads goes through threadgroup
// memory.
#include "AgpuEmitter.h"
#include "AgpuOpTables.h"
#include "AgpuShape.h"

#include "agpu/emit/EmitMemDesc.h"
#include "agpu/emit/EmitPoison.h"
#include "agpu/emit/EmitShuffle.h"
#include "agpu/plan/RebindPlan.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

namespace {

std::optional<int64_t> elemThroughShapeOp(Value res, RankedTensorType srcTy,
                                          RankedTensorType resTy, int reg) {
  if (auto tr = res.getDefiningOp<triton::TransOp>())
    return elemThroughTranspose(srcTy, resTy, tr.getOrder(), reg);
  if (res.getDefiningOp<triton::ReshapeOp>())
    return elemThroughReshape(srcTy, resTy, reg);
  return elemThroughRebind(srcTy, resTy, reg);
}

llvm::ArrayRef<int32_t> transposeOrderOf(Value res) {
  if (auto tr = res.getDefiningOp<triton::TransOp>())
    return tr.getOrder();
  return {};
}

} // namespace

// Whether a layout change can stay inside the warp.
agpu::ShufflePlan AgpuEmitter::shuffleFor(RankedTensorType srcTy,
                                          RankedTensorType resTy,
                                          llvm::ArrayRef<int32_t> order) {
  if (tileElemCount(srcTy) != tileElemCount(resTy))
    return agpu::ShufflePlan();

  // A shuffle only moves values within a warp and the element maps below are
  // read at warp 0 only.
  if (!warpsAgree(srcTy, resTy, order))
    return agpu::ShufflePlan();

  std::vector<std::vector<int64_t>> src = elemsPerLaneOf(srcTy);
  const std::vector<std::vector<int64_t>> &dst = elemsPerLaneOf(resTy);
  if (src.empty() || dst.empty())
    return agpu::ShufflePlan();

  // Result element (i,j) is source element (j,i); renumber into the result's
  // order before comparing.
  if (!order.empty()) {
    const std::optional<std::vector<int64_t>> renumber =
        transposedElemMap(srcTy, resTy, order);
    if (!renumber)
      return agpu::ShufflePlan();
    for (std::vector<int64_t> &perReg : src)
      for (int64_t &e : perReg) {
        if (e < 0 || e >= (int64_t)renumber->size())
          return agpu::ShufflePlan();
        e = (*renumber)[(std::size_t)e];
      }
  }
  return agpu::planShuffleFromElems(src, dst);
}

// Stage the whole tile, then read it back by the result's coordinates. The
// buffer is declared at the op's site and lives only across these barriers.
agpu::Decision AgpuEmitter::emitRedistribute(const agpu::OpView &o,
                                             RankedTensorType srcTy,
                                             RankedTensorType resTy,
                                             llvm::ArrayRef<int32_t> order) {
  if (isTensorOfPointers(srcTy))
    return declined(o.name, "a pointer tensor cannot be redistributed");

  if (order.empty()) {
    if (tileElemCount(srcTy) != tileElemCount(resTy))
      return declined(o.name,
                      "a redistribution cannot change the element count");
  } else if (!isPermutationOf(srcTy.getShape(), resTy.getShape(), order)) {
    return declined(o.name, "the result is not that permutation of the source");
  }

  const agpu::ElemType *elem = elemOf(o.results[0]);
  if (!elem)
    return declined(o.name, "result type was never recorded");

  if (const agpu::ShufflePlan sp = shuffleFor(srcTy, resTy, order); sp.usable())
    return emitShuffled(o, srcTy, resTy, sp, *elem);

  am::Context &mc = agpu_.context();
  const agpu::TileView srcView = rowMajorViewOf(srcTy);
  const agpu::TileView dstView =
      order.empty() ? rowMajorViewOf(resTy) : permutedView(srcView, order);
  const agpu::MemDesc md{"rd" + std::to_string(o.results[0]), srcView};

  const int64_t bytes = md.cosizeElems() * agpu::byteWidthOf(*elem);
  if (body_.threadgroupBytes + bytes > agpu::kTGResidentBudgetBytes)
    return declined(o.name, "threadgroup memory past the hardware limit");
  body_.threadgroupBytes += bytes;

  cur_->push_back(agpu::memDescDecl(mc, md, agpu::mslTypeOf(*elem)));
  cur_->push_back(mc.hardBarrier());
  if (const agpu::Decision d = stageWholeTensor(
          o.operands[0], srcTy, md.buffer, md.view, *elem, o.name, "a source");
      !d.ok())
    return d;
  cur_->push_back(mc.hardBarrier());

  am::SmallVec<agpu::StageAction, 8> actions;
  if (const agpu::Decision d =
          planTileActions(o.results[0], resTy, wholeWindowsOf(resTy), dstView,
                          elem->bits, actions, o.name);
      !d.ok())
    return d;

  const int64_t regs = registerCount(resTy);
  int64_t covered = 0;
  for (const agpu::StageAction &a : actions)
    covered += a.width;
  if (covered < regs)
    return declined(o.name, "a result register never lands");

  am::SmallVec<am::Str, 8> dst;
  agpu::ValueNames names;
  for (int64_t r = 0; r < regs; ++r) {
    const am::Str n = nameFor('t', o.results[0], r);
    cur_->push_back(agpu::poisonDecl(mc, n, *elem));
    dst.push_back(n);
    names.push_back(n);
  }
  agpu::emitReadback(mc, *cur_, dstView, md.buffer, actions, dst, {},
                     coordSourceOf(resTy), *elem, *elem);

  // The next op-site buffer reuses these bytes, so no thread may still be
  // reading when it scatters.
  cur_->push_back(mc.hardBarrier());
  body_.sym.bindRegs(o.results[0], std::move(names));
  return agpu::Decision::emitted();
}

// Every value stays in its own warp, so the move is a lane exchange. An
// identity permutation emits nothing and hands back the source names.
agpu::Decision AgpuEmitter::emitShuffled(const agpu::OpView &o,
                                         RankedTensorType srcTy,
                                         RankedTensorType resTy,
                                         const agpu::ShufflePlan &sp,
                                         agpu::ElemType elem) {
  const Ready ready = readyForCounted(o, 0, 1, registerCount(srcTy),
                                      "a source register was never bound");
  if (!ready.ok())
    return ready.why;

  am::SmallVec<am::Str, 8> src;
  for (int64_t r = 0; r < ready.regs; ++r)
    src.push_back(ready.ops[0].at(r));

  am::SmallVec<am::Str, 8> dstNames;
  for (int64_t r = 0, n = registerCount(resTy); r < n; ++r)
    dstNames.push_back(nameFor('w', o.results[0], r));

  // Suffixed per shuffle: both names are declared and a second shuffle in the
  // same kernel scope would redeclare them.
  agpu::ShuffleNames snm;
  const std::string sseq = std::to_string(body_.shuffleSeq++);
  snm.srcLane += sseq;
  snm.table += sseq;

  const am::SmallVec<am::Str, 8> held =
      agpu::emitShuffle(agpu_.context(), *cur_, sp, src, dstNames, elem, snm);
  if (held.empty())
    return declined(o.name, "the shuffle produced no registers");

  agpu::ValueNames names(held.begin(), held.end());
  body_.sym.bindRegs(o.results[0], std::move(names));
  return agpu::Decision::emitted();
}

agpu::Decision AgpuEmitter::emitRebindOp(const agpu::OpView &o) {
  if (o.operands.size() != 1 || o.results.size() != 1)
    return declined(o.name, "unexpected operand or result count");

  const Value src = mlirValueOf(o.operands[0]);
  if (!src)
    return declined(o.name, "operand value was never recorded");
  const Value res = mlirValueOf(o.results[0]);
  if (!res)
    return declined(o.name, "result value was never recorded");
  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  auto resTy = dyn_cast<RankedTensorType>(res.getType());
  if (!srcTy || !resTy)
    return declined(o.name, "an operand is not a ranked tensor");

  // A transpose whose two layouts are interchangeable is still data movement,
  // which comparing the register maps cannot see.
  const llvm::ArrayRef<int32_t> order = transposeOrderOf(res);
  if (!order.empty())
    return emitRedistribute(o, srcTy, resTy, order);

  // plan/RebindPlan.h decides which source register feeds each result
  // register; this layer supplies the coordinate sets.
  const int64_t srcRegs = registerCount(srcTy);
  std::vector<agpu::RegCoord> srcCoords;
  srcCoords.reserve((std::size_t)srcRegs);
  for (int64_t r = 0; r < srcRegs; ++r) {
    const std::optional<int64_t> e = flatElemAt(srcTy, (int)r);
    srcCoords.push_back(e ? agpu::RegCoord{(int32_t)*e}
                          : agpu::RegCoord{-1 - (int32_t)r});
  }

  const int64_t resRegs = registerCount(resTy);
  std::vector<agpu::RegCoord> resCoords;
  resCoords.reserve((std::size_t)resRegs);
  for (int64_t r = 0; r < resRegs; ++r) {
    const std::optional<int64_t> want =
        elemThroughShapeOp(res, srcTy, resTy, (int)r);
    if (!want)
      return declined(o.name, "cannot map a result register to a "
                              "source element");
    resCoords.push_back({(int32_t)*want});
  }

  const agpu::Rebind plan =
      agpu::rebind(resCoords, agpu::indexByCoord(srcCoords),
                   [](const agpu::RegCoord &rc, agpu::RegCoord &want) {
                     want = rc;
                     return true;
                   });

  if (!plan.complete())
    return emitRedistribute(o, srcTy, resTy, order);

  const Ready ready =
      readyForCounted(o, 0, 1, srcRegs, "a source register was never bound");
  if (!ready.ok())
    return ready.why;

  agpu::ValueNames names;
  for (int64_t r = 0; r < resRegs; ++r) {
    const int from = plan.from[(std::size_t)r];
    names.push_back(ready.ops[0].at(from));

    // A pointer register is a name and an offset.
    inheritOffset(o.operands[0], (int64_t)from, o.results[0], r);
  }
  body_.sym.bindRegs(o.results[0], std::move(names));
  return agpu::Decision::emitted();
}

void AgpuEmitter::registerRebindHandler() {
  table_.add(
      "rebind",
      agpu::forOps({kExpandDims, kBroadcast, kConvertLayout, kTrans, kReshape},
                   [this](const agpu::OpView &o) { return emitRebindOp(o); }));
}

} // namespace mlir::triton::applegpu::bridge
