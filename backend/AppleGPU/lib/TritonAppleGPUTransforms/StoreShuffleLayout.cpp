// StoreShuffleLayout: re-lay an MMA epilogue store into a simdgroup-shuffle
// blocked layout (threadsPerWarp=[8,4], sizePerThread=[1,2], order=[1,0]) that
// keeps every C element in the same simdgroup as the #mma result, so the #mma
// -> #blocked convert becomes a within-warp shuffle instead of a TG round-trip.
// Runs after the final remove_layout_conversions, so it re-lays the store's
// pointer/mask address math too; only an allowlist of layout-parametric ops is
// handled, any other op aborts the rewrite for that store.

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "TritonAppleGPUTransforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;
using namespace mlir::triton::applegpu;

namespace mlir::triton::applegpu {

#define GEN_PASS_DEF_STORESHUFFLELAYOUT
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace {

// A layout-parametric op whose semantics are independent of the distributed
// encoding, so cloning it with a different result encoding is value-preserving.
static bool isRelayable(Operation *op) {
  return isa<tt::MakeRangeOp, tt::SplatOp, tt::ExpandDimsOp, tt::BroadcastOp,
             tt::AddPtrOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
             arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp,
             arith::CmpIOp, arith::AndIOp, arith::OrIOp, arith::SelectOp,
             arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::ConstantOp,
             ttg::ConvertLayoutOp>(op);
}

class StoreShuffleLayoutPass
    : public impl::StoreShuffleLayoutBase<StoreShuffleLayoutPass> {

  // Recursively rebuild `v` so its tensor result carries encoding `wantEnc`.
  // Returns nullptr on failure (unhandled op / non-tensor / shape mismatch).
  // Memoized per (value, encoding) so shared subexpressions are cloned once.
  Value relay(Value v, Attribute wantEnc, OpBuilder &b,
              DenseMap<std::pair<Value, Attribute>, Value> &memo) {
    auto tt2 = dyn_cast<RankedTensorType>(v.getType());
    if (!tt2)
      return v; // scalars are layout-free, pass through
    if (tt2.getEncoding() == wantEnc)
      return v; // already in the target layout
    auto key = std::make_pair(v, wantEnc);
    if (auto it = memo.find(key); it != memo.end())
      return it->second;

    Operation *def = v.getDefiningOp();
    if (!def || !isRelayable(def))
      return nullptr;

    // A convert_layout in the slice is just a layout reshuffle of its source;
    // re-lay its source directly into the wanted layout, dropping the convert.
    if (auto cv = dyn_cast<ttg::ConvertLayoutOp>(def)) {
      Value re = relay(cv.getSrc(), wantEnc, b, memo);
      if (re)
        memo[key] = re;
      return re;
    }

    // A splat constant carries its layout in the value attribute's type;
    // rebuild it directly in the wanted layout.
    if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
      auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!dense || !dense.isSplat())
        return nullptr;
      auto newResTy =
          RankedTensorType::get(tt2.getShape(), tt2.getElementType(), wantEnc);
      auto newAttr =
          DenseElementsAttr::get(newResTy, dense.getSplatValue<Attribute>());
      Value res = arith::ConstantOp::create(b, cst.getLoc(), newResTy, newAttr);
      memo[key] = res;
      return res;
    }

    // Encoding the tensor operands must carry to yield wantEnc: equals wantEnc
    // for elementwise ops, the slice/parent encoding for expand_dims/broadcast.
    bool hasTensorOperand = llvm::any_of(def->getOperands(), [](Value o) {
      return isa<RankedTensorType>(o.getType());
    });
    Attribute srcEnc;
    if (hasTensorOperand) {
      srcEnc = inferSrcEncoding(def, wantEnc);
      if (!srcEnc)
        return nullptr;
    }

    SmallVector<Value> newOperands;
    for (Value operand : def->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType())) {
        newOperands.push_back(operand);
        continue;
      }
      Value re = relay(operand, srcEnc, b, memo);
      if (!re)
        return nullptr;
      newOperands.push_back(re);
    }

    auto newResTy =
        RankedTensorType::get(tt2.getShape(), tt2.getElementType(), wantEnc);
    OperationState state(def->getLoc(), def->getName().getStringRef());
    state.addOperands(newOperands);
    state.addTypes({newResTy});
    state.addAttributes(def->getAttrs());
    Operation *cloned = b.create(state);
    Value res = cloned->getResult(0);
    memo[key] = res;
    return res;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    SmallVector<tt::StoreOp> targets;
    mod.walk([&](tt::StoreOp store) { targets.push_back(store); });

    for (tt::StoreOp store : targets) {
      // The stored value must be a #mma -> #blocked convert_layout result.
      auto cvt = store.getValue().getDefiningOp<ttg::ConvertLayoutOp>();
      if (!cvt)
        continue;
      auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
      auto mmaEnc = dyn_cast<AppleMmaEncodingAttr>(srcTy.getEncoding());
      auto dstTy = cast<RankedTensorType>(cvt.getResult().getType());
      auto blkEnc = dyn_cast<ttg::BlockedEncodingAttr>(dstTy.getEncoding());
      if (!mmaEnc || !blkEnc)
        continue;
      if (srcTy.getRank() != 2)
        continue;
      auto shape = srcTy.getShape();
      if (shape[0] % 8 != 0 || shape[1] % 8 != 0)
        continue;

      // The shuffle-friendly blocked layout (one 8x8 simdgroup tile per warp
      // slot, register dim = col bit 0).
      auto wpc = mmaEnc.getWarpsPerCTA();
      SmallVector<unsigned> spt{1, 2};
      SmallVector<unsigned> tpw{8, 4};
      SmallVector<unsigned> ord{1, 0};
      SmallVector<unsigned> warps(wpc.begin(), wpc.end());

      int numWarps = ttg::lookupNumWarps(mod);
      int curWarps = 1;
      for (unsigned w : warps)
        curWarps *= w;
      if (numWarps % curWarps == 0) {
        unsigned factor = numWarps / curWarps;
        if (factor > 1)
          warps[ord[0]] *= factor;
      }

      auto wEnc = ttg::BlockedEncodingAttr::get(ctx, spt, tpw, warps, ord,
                                                blkEnc.getCGALayout());
      auto wType = RankedTensorType::get(shape, dstTy.getElementType(), wEnc);

      // Only proceed when the mma -> W convert is genuinely within-simdgroup
      // (no shared memory). Otherwise this rewrite buys nothing.
      if (cvtNeedsSharedMemory(srcTy, wType))
        continue;

      // Re-lay the pointer and mask address math into W. If any op in their
      // backward slice is not relayable, abort this store (keep the original
      // coalesced layout + TG convert path).
      OpBuilder b(store);
      DenseMap<std::pair<Value, Attribute>, Value> memo;

      Value newPtr = relay(store.getPtr(), wEnc, b, memo);
      if (!newPtr)
        continue;
      Value newMask;
      if (store.getMask()) {
        newMask = relay(store.getMask(), wEnc, b, memo);
        if (!newMask)
          continue;
      }

      auto newCvt =
          ttg::ConvertLayoutOp::create(b, cvt.getLoc(), wType, cvt.getSrc());
      store.getValueMutable().assign(newCvt.getResult());
      store.getPtrMutable().assign(newPtr);
      if (newMask)
        store.getMaskMutable().assign(newMask);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createStoreShuffleLayoutPass() {
  return std::make_unique<StoreShuffleLayoutPass>();
}

} // namespace mlir::triton::applegpu
