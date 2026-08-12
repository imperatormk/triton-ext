// MSLShufflePlan.h - intra-warp convert_layout plan.
//
// A convert_layout whose every destination element already lives in the
// destination lane's own simdgroup is a lane permutation, not a cross-thread
// exchange: simd_shuffle moves it with no threadgroup traffic and no barrier.
// The GEMM epilogue's #apple_mma -> #blocked convert is exactly this shape.
#ifndef MSL_SHUFFLE_PLAN_H
#define MSL_SHUFFLE_PLAN_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::triton::applegpu {

namespace ttg = mlir::triton::gpu;

// One destination register: its value is srcReg of lane srcLane[dstLane], all
// within the destination thread's own warp.
struct ShuffleStep {
  int srcReg;
  // Indexed by destination lane; entry is the lane to read from.
  SmallVector<int32_t> srcLane;
  // True when srcLane[l] == l for every l, i.e. a plain register rebind.
  bool identity;
};

struct ShufflePlan {
  SmallVector<ShuffleStep> steps; // one per destination register
  // Every step shares one lane permutation, and that permutation is linear over
  // GF(2): srcLane = XOR of laneBasis[b] over the set bits b of the dst lane.
  // This lets the lane index be one XOR chain hoisted ahead of the shuffles
  // instead of a 32-entry lookup per register.
  bool uniformLanePerm = false;
  bool lanePermLinear = false;
  SmallVector<int32_t> laneBasis;
};

// Returns a plan when the convert is a pure intra-warp permutation, and
// std::nullopt when any destination element comes from another warp (or the
// layouts are not shuffle-shaped: differing warp counts, replication, or a
// register count that does not match the layout).
std::optional<ShufflePlan> planIntraWarpShuffle(RankedTensorType srcTy,
                                                RankedTensorType dstTy);

// Same, for a shape-changing op (tt.trans / tt.reshape). `perm` maps a result
// axis to the source axis it came from; pass an empty `perm` for a plain
// reshape. Both sides are compared as row-major flat element indices, which is
// the coordinate the threadgroup round-trip these replace keys on.
std::optional<ShufflePlan> planIntraWarpShufflePermuted(RankedTensorType srcTy,
                                                        RankedTensorType dstTy,
                                                        ArrayRef<int32_t> perm);

} // namespace mlir::triton::applegpu

#endif // MSL_SHUFFLE_PLAN_H
