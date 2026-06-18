// AccelerateAppleMatmul: rewrite tt.dot to AppleMmaEncoding so the output gets
// simdgroup_multiply_accumulate codegen. A/B stay BlockedEncoding; the result
// is converted back to the user's layout. Mirrors AMD's BlockedToMFMA.

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DEF_ACCELERATEAPPLEMATMUL
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;
using namespace mlir::triton::applegpu;

namespace {
// warpsPerCTA for a dot shape and warp count. Apple simdgroup tile = 8x8.
SmallVector<unsigned> warpsPerTileApple(int64_t M, int64_t N, int numWarps) {
  unsigned tilesM = std::max<int64_t>(1, M / 8);
  unsigned tilesN = std::max<int64_t>(1, N / 8);

  unsigned bestM = 1, bestN = 1;
  unsigned bestProduct = 1;
  unsigned bestOperandFrags = tilesM + tilesN;
  unsigned bestBalance = std::max(tilesM, tilesN) - std::min(tilesM, tilesN);
  unsigned bestWarpBalance = 1;

  // Use as many warps as legally tile the CTA, then pick the split with the
  // smallest per-warp A/B working set (the split changes how many A/B simdgroup
  // tiles are live around the MMA loop on rectangular GEMMs).
  for (unsigned wm = 1; wm <= tilesM; ++wm) {
    if (tilesM % wm != 0)
      continue;
    for (unsigned wn = 1; wn <= tilesN; ++wn) {
      if (tilesN % wn != 0)
        continue;
      unsigned product = wm * wn;
      if (product > (unsigned)numWarps)
        continue;

      unsigned ownM = tilesM / wm;
      unsigned ownN = tilesN / wn;
      unsigned operandFrags = ownM + ownN;
      unsigned balance = std::max(ownM, ownN) - std::min(ownM, ownN);
      unsigned warpBalance = std::min(wm, wn);

      bool better = product > bestProduct;
      if (!better && product == bestProduct)
        better = operandFrags < bestOperandFrags;
      if (!better && product == bestProduct && operandFrags == bestOperandFrags)
        better = balance < bestBalance;
      if (!better && product == bestProduct &&
          operandFrags == bestOperandFrags && balance == bestBalance)
        better = warpBalance > bestWarpBalance;
      if (!better)
        continue;

      bestM = wm;
      bestN = wn;
      bestProduct = product;
      bestOperandFrags = operandFrags;
      bestBalance = balance;
      bestWarpBalance = warpBalance;
    }
  }

  return {bestM, bestN};
}

bool isSupportedDotType(mlir::Type elemTy) {
  return elemTy.isF16() || elemTy.isBF16() || elemTy.isF32();
}

// Pattern: BlockedEncoding tt.dot → AppleMmaEncoding tt.dot
struct BlockedToAppleMma : public OpRewritePattern<tt::DotOp> {
  int numWarps;

  BlockedToAppleMma(MLIRContext *ctx, int numWarps, PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), numWarps(numWarps) {}

  LogicalResult matchAndRewrite(tt::DotOp dot,
                                PatternRewriter &rewriter) const override {
    auto ctx = dot.getContext();
    auto cType = cast<RankedTensorType>(dot.getC().getType());
    auto aType = cast<RankedTensorType>(dot.getA().getType());

    if (isa<AppleMmaEncodingAttr>(cType.getEncoding()))
      return failure();

    if (!isSupportedDotType(aType.getElementType()))
      return failure();

    auto shape = cType.getShape();
    if (shape.size() != 2)
      return failure();

    int64_t M = shape[0], N = shape[1];

    // Tile must divide into 8x8 simdgroup tiles and be large enough to
    // warp-tile.
    if (M % 8 != 0 || N % 8 != 0)
      return failure();
    if (M < 16 || N < 16)
      return failure();

    int64_t K = aType.getShape()[1];

    // K must be a multiple of the simdgroup MMA K-tile (8); a non-aligned K
    // would silently corrupt the accumulation. Always true for a valid tl.dot.
    if (K % 8 != 0)
      return failure();

    // MMA encoding on the result only; A/B stay blocked (DotOpToLLVM scatters
    // them through TG), so only one result->blocked ConvertLayoutOp is needed
    // downstream. Oversized tiles fail cleanly via the shared-memory budget.
    auto wpc = warpsPerTileApple(M, N, numWarps);
    auto mmaEnc = AppleMmaEncodingAttr::get(ctx, wpc);

    auto newCType =
        RankedTensorType::get(shape, cType.getElementType(), mmaEnc);

    auto loc = dot.getLoc();

    // Strip any DotOperandEncoding on A/B back to plain blocked: its parent
    // must match the result encoding, but we keep A/B blocked while C becomes
    // MMA.
    auto stripDotOpEnc = [&](Value operand) -> Value {
      auto ty = cast<RankedTensorType>(operand.getType());
      if (auto dotEnc =
              dyn_cast<ttg::DotOperandEncodingAttr>(ty.getEncoding())) {
        auto parentTy = RankedTensorType::get(
            ty.getShape(), ty.getElementType(), dotEnc.getParent());
        if (auto cvt = operand.getDefiningOp<ttg::ConvertLayoutOp>()) {
          if (cvt.getSrc().getType() == parentTy)
            return cvt.getSrc();
        }
        return ttg::ConvertLayoutOp::create(rewriter, loc, parentTy, operand);
      }
      return operand;
    };
    Value newA = stripDotOpEnc(dot.getA());
    Value newB = stripDotOpEnc(dot.getB());

    Value newC = dot.getC();
    if (auto cvt = newC.getDefiningOp<ttg::ConvertLayoutOp>()) {
      if (cvt.getSrc().getType() == newCType)
        newC = cvt.getSrc();
      else
        newC =
            ttg::ConvertLayoutOp::create(rewriter, loc, newCType, dot.getC());
    } else {
      newC = ttg::ConvertLayoutOp::create(rewriter, loc, newCType, dot.getC());
    }

    auto newDot = tt::DotOp::create(rewriter, loc, newCType, newA, newB, newC,
                                    dot.getInputPrecisionAttr(),
                                    dot.getMaxNumImpreciseAccAttr());

    SmallVector<OpOperand *> mmaUses;
    SmallVector<OpOperand *> blockedUses;
    SmallVector<ttg::ConvertLayoutOp> passthroughCasts;
    for (OpOperand &use : dot->getUses()) {
      auto *owner = use.getOwner();
      if (isa<tt::DotOp>(owner) && use.getOperandNumber() == 2) {
        mmaUses.push_back(&use);
        continue;
      }
      if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(owner)) {
        if (cvt.getType() == newCType) {
          for (OpOperand &cvtUse : cvt->getUses())
            mmaUses.push_back(&cvtUse);
          passthroughCasts.push_back(cvt);
          continue;
        }
      }
      blockedUses.push_back(&use);
    }

    for (OpOperand *use : mmaUses)
      use->set(newDot.getResult());
    for (ttg::ConvertLayoutOp cvt : passthroughCasts)
      if (cvt->use_empty())
        rewriter.eraseOp(cvt);

    if (blockedUses.empty()) {
      rewriter.eraseOp(dot);
      return success();
    }

    auto result =
        ttg::ConvertLayoutOp::create(rewriter, loc, cType, newDot.getResult());
    for (OpOperand *use : blockedUses)
      use->set(result.getResult());

    rewriter.eraseOp(dot);
    return success();
  }
};

struct AccelerateAppleMatmul
    : public ::impl::AccelerateAppleMatmulBase<AccelerateAppleMatmul> {

  void runOnOperation() override {
    auto mod = getOperation();

    int numWarps = ttg::lookupNumWarps(mod);

    RewritePatternSet patterns(&getContext());
    patterns.add<BlockedToAppleMma>(&getContext(), numWarps);

    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir::triton::applegpu {
std::unique_ptr<mlir::Pass> createAccelerateAppleMatmulPass() {
  return ::createAccelerateAppleMatmul();
}
} // namespace mlir::triton::applegpu
