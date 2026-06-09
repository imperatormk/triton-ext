// StoreShuffleLayout: re-lay an MMA epilogue store into the simdgroup-shuffle
// blocked layout so the #mma -> #blocked C-output convert_layout becomes a
// within-simdgroup register+shuffle move instead of a threadgroup round-trip.
//
// The AppleMma C result and the natural coalesced store layout use DIFFERENT
// warp tilings (mma warps own scattered 8x8 simdgroup tiles; the store layout
// owns contiguous row bands), so the convert is cross-warp and must spill
// through threadgroup memory + barriers. For large tiles that round-trip
// dominates the whole GEMM epilogue.
//
// There is exactly one blocked layout that maps every C element to a lane in
// the SAME simdgroup it already occupies in the #mma result:
//   warpsPerCTA = mma.warpsPerCTA, threadsPerWarp = [8, 4], sizePerThread =
//   [1, 2], order = [1, 0]
// (8 lane-rows x 4 lane-cols x 2 register-cols == one 8x8 simdgroup tile, with
// the register dim carrying col bit 0, matching the hardware per-lane storage).
// With this layout the convert is within-warp, so the shared upstream
// ConvertLayout pattern lowers it with simd_shuffle and no shared memory.
//
// This pass runs after make_ttgir's final remove_layout_conversions, so it must
// re-lay the store's pointer/mask address math into the shuffle layout itself
// (a later layout pass would otherwise revert it to the coalesced layout). It
// does this by cloning the backward slice feeding the pointer/mask with the
// shuffle-projected encodings. Only a fixed allowlist of layout-parametric ops
// is handled; any other op in the slice aborts the rewrite for that store, in
// which case the original coalesced layout + threadgroup convert path is kept
// (always correct).

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

// Returns true if op is a layout-parametric op we can safely re-lay: its
// semantics are independent of the chosen distributed encoding (only the
// per-thread mapping changes), so cloning it with a different result encoding
// is value-preserving.
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

  // Recursively rebuild `v` so its tensor result carries encoding `wantEnc`
  // (a 2D blocked or its rank-reduced slice). Returns the rebuilt value or
  // nullptr on failure (unhandled op / non-tensor / shape mismatch). Rebuilt
  // values are memoized per (value, encoding) so shared subexpressions are
  // cloned once.
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

    // The encoding each tensor operand must carry so the op result is wantEnc.
    // For elementwise ops this equals wantEnc; for the rank-changing ops
    // (expand_dims/broadcast) it is the matching slice/parent encoding. Ops
    // with no tensor operands (make_range) need no source encoding; clone them
    // with the wanted result encoding directly.
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

    // Clone the op with the new operands and the wanted result encoding.
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
      auto wEnc = ttg::BlockedEncodingAttr::get(
          ctx, spt, tpw, SmallVector<unsigned>(wpc.begin(), wpc.end()), ord,
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

      // Build the within-simdgroup mma -> W convert and rewrite the store.
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
