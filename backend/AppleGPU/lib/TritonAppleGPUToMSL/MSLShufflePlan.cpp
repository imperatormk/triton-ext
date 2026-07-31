#include "MSLShufflePlan.h"

#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::applegpu {

namespace {

// A tensor coordinate packed into one integer so it can key a DenseMap. Every
// out dim is < 2^16 for the tile sizes the emitter builds.
std::optional<uint64_t> packCoords(ArrayRef<std::pair<StringAttr, int32_t>> cs) {
  uint64_t key = 0;
  for (auto &[dim, v] : cs) {
    if (v < 0 || v >= (1 << 16))
      return std::nullopt;
    key = (key << 16) | (uint32_t)v;
  }
  return key;
}

struct Ids {
  StringAttr reg, lane, warp, block;
};

// Applies `ll` at (reg, lane, warp), with the block dim pinned to 0 (Apple has
// a single CTA).
std::optional<uint64_t> at(const LinearLayout &ll, const Ids &ids, int reg,
                           int lane, int warp) {
  SmallVector<std::pair<StringAttr, int32_t>> in;
  in.push_back({ids.reg, reg});
  if (ll.hasInDim(ids.lane))
    in.push_back({ids.lane, lane});
  if (ll.hasInDim(ids.warp))
    in.push_back({ids.warp, warp});
  if (ll.hasInDim(ids.block))
    in.push_back({ids.block, 0});
  return packCoords(ll.apply(in));
}

} // namespace

std::optional<ShufflePlan> planIntraWarpShuffle(RankedTensorType srcTy,
                                                RankedTensorType dstTy) {
  if (srcTy.getShape() != dstTy.getShape())
    return std::nullopt;

  MLIRContext *mctx = srcTy.getContext();
  Ids ids{StringAttr::get(mctx, "register"), StringAttr::get(mctx, "lane"),
          StringAttr::get(mctx, "warp"), StringAttr::get(mctx, "block")};

  LinearLayout sl = ttg::toLinearLayout(srcTy);
  LinearLayout dl = ttg::toLinearLayout(dstTy);

  // Both sides must agree on the thread grid, or "same warp" is not even a
  // well-posed question.
  if (!sl.hasInDim(ids.lane) || !dl.hasInDim(ids.lane))
    return std::nullopt;
  if (sl.getInDimSize(ids.lane) != dl.getInDimSize(ids.lane))
    return std::nullopt;
  int32_t nLanes = sl.getInDimSize(ids.lane);
  auto warpCount = [&](const LinearLayout &ll) {
    return ll.hasInDim(ids.warp) ? ll.getInDimSize(ids.warp) : 1;
  };
  if (warpCount(sl) != warpCount(dl))
    return std::nullopt;
  int32_t nWarps = warpCount(sl);
  int32_t nSrcRegs = sl.getInDimSize(ids.reg);
  int32_t nDstRegs = dl.getInDimSize(ids.reg);

  // A replicated source has several (reg, lane) writing one coordinate; picking
  // one is still correct, but a replicated *destination* would need the same
  // value in several places, which the per-register plan below already covers.
  // What it cannot cover is a source that fails to produce some destination
  // coordinate, so the lookup below bails on a miss.
  ShufflePlan plan;
  plan.steps.resize(nDstRegs);

  for (int32_t warp = 0; warp < nWarps; ++warp) {
    // Coordinates this warp owns on the source side, and who owns them.
    llvm::DenseMap<uint64_t, std::pair<int32_t, int32_t>> owner; // -> (lane,reg)
    for (int32_t reg = 0; reg < nSrcRegs; ++reg)
      for (int32_t lane = 0; lane < nLanes; ++lane) {
        auto k = at(sl, ids, reg, lane, warp);
        if (!k)
          return std::nullopt;
        owner.try_emplace(*k, std::make_pair(lane, reg));
      }

    for (int32_t dreg = 0; dreg < nDstRegs; ++dreg) {
      ShuffleStep &step = plan.steps[dreg];
      if (warp == 0) {
        step.srcReg = -1;
        step.srcLane.assign(nLanes, -1);
      }
      for (int32_t lane = 0; lane < nLanes; ++lane) {
        auto k = at(dl, ids, dreg, lane, warp);
        if (!k)
          return std::nullopt;
        auto it = owner.find(*k);
        if (it == owner.end())
          return std::nullopt; // the value lives in another warp
        auto [srcLane, srcReg] = it->second;
        // The same shuffle sequence is emitted once for all warps, so every
        // warp has to agree on both the source register and the lane mapping.
        if (step.srcReg == -1)
          step.srcReg = srcReg;
        else if (step.srcReg != srcReg)
          return std::nullopt;
        if (step.srcLane[lane] == -1)
          step.srcLane[lane] = srcLane;
        else if (step.srcLane[lane] != srcLane)
          return std::nullopt;
      }
    }
  }

  for (auto &step : plan.steps) {
    if (step.srcReg < 0)
      return std::nullopt;
    step.identity = true;
    for (int32_t l = 0; l < nLanes; ++l)
      if (step.srcLane[l] != l) {
        step.identity = false;
        break;
      }
  }

  plan.uniformLanePerm = true;
  for (auto &step : plan.steps)
    if (step.srcLane != plan.steps.front().srcLane) {
      plan.uniformLanePerm = false;
      break;
    }
  if (plan.uniformLanePerm) {
    ArrayRef<int32_t> perm = plan.steps.front().srcLane;
    plan.lanePermLinear = perm[0] == 0;
    if (plan.lanePermLinear) {
      for (int32_t b = 1; b < nLanes; b <<= 1)
        plan.laneBasis.push_back(perm[b]);
      for (int32_t l = 0; l < nLanes && plan.lanePermLinear; ++l) {
        int32_t acc = 0;
        for (size_t b = 0; b < plan.laneBasis.size(); ++b)
          if (l & (1 << b))
            acc ^= plan.laneBasis[b];
        if (acc != perm[l])
          plan.lanePermLinear = false;
      }
    }
    if (!plan.lanePermLinear)
      plan.laneBasis.clear();
  }
  return plan;
}

} // namespace mlir::triton::applegpu
