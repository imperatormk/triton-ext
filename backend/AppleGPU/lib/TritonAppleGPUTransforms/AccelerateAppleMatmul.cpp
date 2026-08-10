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

#include <tuple>

#define GEN_PASS_DEF_ACCELERATEAPPLEMATMUL
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;
using namespace mlir::triton::applegpu;

namespace {
// A row reduction is warp-local only when one warp owns whole rows.
bool resultFeedsRowReduction(tt::DotOp dot) {
  SmallVector<Value, 4> work{dot.getResult()};
  SmallPtrSet<Operation *, 8> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    for (Operation *u : v.getUsers()) {
      if (!seen.insert(u).second)
        continue;
      if (auto red = dyn_cast<tt::ReduceOp>(u)) {
        auto srcTy = dyn_cast<RankedTensorType>(red.getOperands()[0].getType());
        if (srcTy && red.getAxis() == srcTy.getRank() - 1)
          return true;
        continue;
      }
      if (u->hasTrait<OpTrait::Elementwise>() ||
          isa<ttg::ConvertLayoutOp, tt::BroadcastOp, tt::ExpandDimsOp>(u))
        for (Value r : u->getResults())
          work.push_back(r);
    }
  }
  return false;
}

// A dot consuming another dot's result as its A operand has to agree with the
// producer's warp tiling, or the result crosses between them through
// threadgroup memory instead of staying in registers.
bool operandComesFromDot(tt::DotOp dot) {
  SmallVector<Value, 4> work{dot.getA()};
  SmallPtrSet<Operation *, 8> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || !seen.insert(def).second)
      continue;
    if (isa<tt::DotOp>(def))
      return true;
    if (def->hasTrait<OpTrait::Elementwise>() ||
        isa<ttg::ConvertLayoutOp, tt::BroadcastOp, tt::ExpandDimsOp,
            tt::FpToFpOp>(def))
      for (Value o : def->getOperands())
        work.push_back(o);
  }
  return false;
}

// warpsPerCTA for a dot shape and warp count. Apple simdgroup tile = 8x8.
//
// The wm|tilesM && wn|tilesN divisibility filter can in principle leave warps
// idle, but not for any shape Triton can actually produce: tensor dims are
// powers of two, so tilesM/tilesN/nFrag are too, and the search then always
// reaches product == min(nFrag, numWarps) (verified exhaustively over all
// power-of-two M,N up to 2048 and numWarps up to 64). The residual idleness is
// nFrag < numWarps - fewer 8x8 fragments than warps - which no choice of split
// can fix; planDot clamps its warp count to nFrag for exactly this reason. The
// returned warp dims still account for every warp, though: see below.
SmallVector<unsigned> warpsPerTileApple(int64_t M, int64_t N, int numWarps,
                                        bool preferRowOwnership = false) {
  unsigned tilesM = std::max<int64_t>(1, M / 8);
  unsigned tilesN = std::max<int64_t>(1, N / 8);

  // Ranking key, compared lexicographically and maximized: prefer the most
  // warps used, then the smallest per-warp A/B working set (negated so bigger
  // wins), then the most balanced ownership, then the squarest warp grid.
  auto rank = [](unsigned product, unsigned operandFrags, unsigned balance,
                 unsigned warpBalance, unsigned ownsRows = 0) {
    return std::tuple<unsigned, unsigned, int, int, unsigned>(
        product, ownsRows, -static_cast<int>(operandFrags),
        -static_cast<int>(balance), warpBalance);
  };

  unsigned bestM = 1, bestN = 1;
  auto best = rank(1, tilesM + tilesN,
                   std::max(tilesM, tilesN) - std::min(tilesM, tilesN), 1,
                   preferRowOwnership ? 1 : 0);

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
      auto key =
          rank(product, ownM + ownN,
               std::max(ownM, ownN) - std::min(ownM, ownN), std::min(wm, wn),
               (preferRowOwnership && wn == 1) ? 1 : 0);
      if (key <= best)
        continue;

      best = key;
      bestM = wm;
      bestN = wn;
    }
  }

  // A tile with fewer 8x8 fragments than warps leaves the surplus warps out of
  // the encoding entirely, and a slice of that layout has no valid encoding --
  // tl.max(qk, 1) over a 16x16 dot at numWarps=8 is rejected outright. Grow the
  // row axis past the fragment count so the warp dim covers every warp; the
  // extra warps own no rows, which is what they already did implicitly.
  while (bestM * bestN < (unsigned)numWarps)
    bestM *= 2;

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
    auto wpc = warpsPerTileApple(
        M, N, numWarps,
        resultFeedsRowReduction(dot) || operandComesFromDot(dot));
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

    // A chained dot's A is the previous dot's accumulator, which is
    // bit-identical to an A-operand fragment: retyping it to this #mma lets the
    // emitter read those registers instead of restaging the tile.
    if (auto aTy = dyn_cast<RankedTensorType>(newA.getType())) {
      if (aTy.getEncoding() != mmaEnc && operandComesFromDot(dot)) {
        auto aMmaTy =
            RankedTensorType::get(aTy.getShape(), aTy.getElementType(), mmaEnc);
        newA = ttg::ConvertLayoutOp::create(rewriter, loc, aMmaTy, newA);
      }
    }

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
      if (isa<tt::DotOp>(owner) &&
          (use.getOperandNumber() == 2 || use.getOperandNumber() == 0)) {
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

// A chained dot only learns its A is #mma after the producing dot converts, and
// BlockedToAppleMma will not revisit a dot whose C is already #mma. Drop the
// convert here instead.
struct DropMmaAConvert : public OpRewritePattern<tt::DotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::DotOp dot,
                                PatternRewriter &rewriter) const override {
    auto cTy = dyn_cast<RankedTensorType>(dot.getC().getType());
    if (!cTy || !isa<AppleMmaEncodingAttr>(cTy.getEncoding()))
      return failure();

    auto cvt = dot.getA().getDefiningOp<ttg::ConvertLayoutOp>();
    if (!cvt)
      return failure();
    auto srcTy = dyn_cast<RankedTensorType>(cvt.getSrc().getType());
    // Both dots must agree on warp tiling, or warp w would need fragments it
    // does not own.
    if (!srcTy || !isa<AppleMmaEncodingAttr>(srcTy.getEncoding()) ||
        cast<AppleMmaEncodingAttr>(srcTy.getEncoding()).getWarpsPerCTA() !=
            cast<AppleMmaEncodingAttr>(cTy.getEncoding()).getWarpsPerCTA())
      return failure();

    rewriter.modifyOpInPlace(dot,
                             [&] { dot.getAMutable().assign(cvt.getSrc()); });
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
    patterns.add<DropMmaAConvert>(&getContext());

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
