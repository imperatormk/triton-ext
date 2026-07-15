// MSLLayoutExpr.h - linear-layout coordinate/offset expression builder.
//
// Builds the per-register out-dim coordinate and row-major flat-offset AST
// exprs for a distributed tensor value, sharing the emitter's MSLContext and
// lane/warp/threadgroup id names. Pure over (layout, reg): no emitter state
// beyond ctx + the id names bound by reference at construction.
#ifndef MSL_LAYOUT_EXPR_H
#define MSL_LAYOUT_EXPR_H

#include "MSLAst.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::applegpu {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

class LayoutExprBuilder {
public:
  LayoutExprBuilder(msl::MSLContext &ctx, const std::string &laneId,
                    const std::string &warpId, const std::string &tgposId)
      : ctx(ctx), laneId(laneId), warpId(warpId), tgposId(tgposId) {}

  SmallVector<int32_t> registerCoords(RankedTensorType rt, int reg);
  msl::Expr *layoutCoordExpr(RankedTensorType rt, int reg, StringAttr outDim);
  msl::Expr *layoutOffsetExpr(RankedTensorType rt, int reg);
  msl::Expr *flatTileOffset(RankedTensorType rt, int reg);
  msl::Expr *sliceFlatOffset(RankedTensorType rt, int reg);
  msl::Expr *batchCoordExpr(RankedTensorType rt, int reg);
  msl::Expr *transFlatOffset(RankedTensorType srcTy, ArrayRef<int32_t> perm,
                             ArrayRef<int64_t> resShape, int reg);

  static uint64_t coordKey(ArrayRef<int32_t> c);

private:
  msl::MSLContext &ctx;
  const std::string &laneId;
  const std::string &warpId;
  const std::string &tgposId;
};

} // namespace mlir::triton::applegpu

#endif // MSL_LAYOUT_EXPR_H
