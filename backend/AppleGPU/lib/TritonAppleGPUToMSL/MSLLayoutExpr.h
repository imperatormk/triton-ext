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
#include "llvm/ADT/StringMap.h"
#include <string>

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

  // Coordinate trees read only lane/warp/threadgroup ids, so one decl at the
  // top of the function dominates every use. Between beginFunc and takeDecls
  // each distinct tree is emitted once and referenced by name thereafter.
  void beginFunc(int *idCounter) {
    hoistId = idCounter;
    hoisted.clear();
    decls.clear();
  }
  msl::Block takeDecls() {
    hoistId = nullptr;
    msl::Block out = std::move(decls);
    decls.clear();
    return out;
  }
  msl::Expr *layoutOffsetExpr(RankedTensorType rt, int reg);
  msl::Expr *flatTileOffset(RankedTensorType rt, int reg);
  // `rowPad` widens the innermost (row) stride past the tensor's own width,
  // spreading successive rows across threadgroup banks.
  msl::Expr *sliceFlatOffset(RankedTensorType rt, int reg, int64_t rowPad = 0);
  msl::Expr *batchCoordExpr(RankedTensorType rt, int reg);
  // Every value `layoutCoordExpr(rt, reg, outDim)` can take across all lanes
  // and warps. The bases are disjoint powers of two, so the register's constant
  // and the runtime bits compose by or: the set is the constant offset by every
  // subset of the runtime mask.
  void coordRange(RankedTensorType rt, int reg, StringAttr outDim, int32_t &lo,
                  int32_t &hi);
  msl::Expr *transFlatOffset(RankedTensorType srcTy, ArrayRef<int32_t> perm,
                             ArrayRef<int64_t> resShape, int reg);

  static uint64_t coordKey(ArrayRef<int32_t> c);

private:
  msl::Expr *buildCoordExpr(const tt::LinearLayout &ll, MLIRContext *mctx,
                            int reg, StringAttr outDim);

  msl::MSLContext &ctx;
  const std::string &laneId;
  const std::string &warpId;
  const std::string &tgposId;

  int *hoistId = nullptr;
  llvm::StringMap<llvm::StringRef> hoisted;
  msl::Block decls;
};

} // namespace mlir::triton::applegpu

#endif // MSL_LAYOUT_EXPR_H
