// ConvertTritonAppleGPUToLLVM pass
//
// Lowers TritonGPU IR → LLVM IR for Apple MPS using shared Triton patterns
// and an Apple-specific TargetInfo.

#include "Dialect/TritonAppleGPU/IR/AppleMmaFragment.h"
#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "TritonAppleGPUToLLVM/Passes.h"
#include "TritonAppleGPUToLLVM/TargetInfo.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::applegpu {

namespace {

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
namespace ttg = mlir::triton::gpu;

// A use of an AppleMma-encoded value is "dot-chain" if it keeps the value on
// the simdgroup register path: as the dot accumulator (C), carried through the
// scf.for loop, or unpacked to #blocked by a convert_layout. Any other consumer
// (elementwise arith/math, broadcast/expand from a #mma slice, integer #mma)
// needs the generic flat per-thread layout, so its type stays out of the
// fragment ABI. The decision is per-type and conservative: a type is
// fragment-eligible only if EVERY #mma value of that exact type in the module
// is consumed solely by dot-chain ops.
static bool isAppleMmaTensor(Type t) {
  auto rt = dyn_cast<RankedTensorType>(t);
  // The fragment ABI carries <64 x f32> simdgroup_matrix fragments, so only f32
  // #mma accumulators qualify; bf16/i32 #mma tensors keep the flat layout.
  return rt && isa<AppleMmaEncodingAttr>(rt.getEncoding()) &&
         rt.getElementType().isF32();
}

// Fragment-ABI candidate predicate for the kkt elementwise/mask chain. SEPARATE
// from isAppleMmaTensor (which is load-bearing on the f32 GATE-A/B dot path):
// admits the f32 accumulator AND the rank-2 i32/i1 #mma temporaries that the
// kkt op-web builds (cmpi index compares, andi/select masks) so they can ride
// the same per-lane simdgroup slot map. bf16/i64 #mma and slice<#mma> stay
// flat.
static bool isFragmentCandidateTensor(Type t) {
  auto rt = dyn_cast<RankedTensorType>(t);
  if (!rt || !isa<AppleMmaEncodingAttr>(rt.getEncoding()) || rt.getRank() != 2)
    return false;
  Type elt = rt.getElementType();
  return elt.isF32() || elt.isInteger(32) || elt.isInteger(1);
}

// A rank-2 f16/bf16 #mma tensor. This is NOT a general fragment candidate (it
// must NOT enter the kkt elementwise web, which would reintroduce the
// bf16-#mma elementwise scalarization leak) — it is admitted to the fragment
// ABI ONLY as the narrow dot-accumulator epilogue: the f16/bf16 GEMM
// accumulates in <64 x f32>, then `arith.truncf : f32#mma -> f16/bf16#mma`
// narrows the result before the convert_layout to #blocked. Carrying that
// truncf result as a <64 x half/bf16> fragment (instead of poisoning the f32
// accumulator's type, which scalarized the whole loop) is what puts f16/bf16
// GEMM on the same fragment baseline as f32.
static bool isHalfMmaTensor(Type t) {
  auto rt = dyn_cast<RankedTensorType>(t);
  if (!rt || !isa<AppleMmaEncodingAttr>(rt.getEncoding()) || rt.getRank() != 2)
    return false;
  Type elt = rt.getElementType();
  return elt.isF16() || elt.isBF16();
}

// True iff `op` is the f16/bf16 accumulator-epilogue truncf: a TruncFOp from an
// f32 #mma fragment candidate to an f16/bf16 #mma, whose result feeds only a
// convert_layout to a non-#mma layout (the store epilogue), AND whose f32 #mma
// input is produced DIRECTLY by a dot (or a loop-carried dot accumulator).
//
// The direct-dot requirement is load-bearing: it admits the GEMM epilogue
// (dot -> truncf -> store) but REJECTS solve_tril's merge kernel
// (dot -> negf -> truncf -> store). With an intervening elementwise op the
// mid-end sinks it through the truncf (negf(truncf) == truncf(negf)), leaving a
// `fsub <64 x bfloat>` on the simdgroup bf16 fragment — which, combined with
// the bf16 round-trip bitcast, crashes the AGX PSO materializer
// (agx-crash-trunk/solve_tril_bf16_merge_pso_crash). The pure dot accumulator
// has no such elementwise op on the fragment, so it stays safe.
static bool isAccumTruncEpilogue(Operation *op) {
  auto tf = dyn_cast<arith::TruncFOp>(op);
  if (!tf)
    return false;
  if (!isFragmentCandidateTensor(tf.getIn().getType()) ||
      !isHalfMmaTensor(tf.getType()))
    return false;
  // The narrowed result must feed EXACTLY ONE convert_layout to a non-#mma
  // layout — the single store epilogue of a GEMM. fla's chunk_delta_h
  // recurrence truncates an intermediate h and fans it out to several
  // convert_layouts (one of which re-feeds a dot), and the multiple distinct
  // #blocked targets don't share one fragment slot map → a flat/fragment slot
  // mismatch at pack time. Requiring a single blocked consumer keeps this on
  // the pure GEMM store narrow.
  ttg::ConvertLayoutOp theCvt;
  for (Operation *user : tf->getUsers()) {
    auto cvt = dyn_cast<ttg::ConvertLayoutOp>(user);
    if (!cvt)
      return false;
    auto resTy = dyn_cast<RankedTensorType>(cvt.getType());
    if (!resTy || isa<AppleMmaEncodingAttr>(resTy.getEncoding()))
      return false;
    if (theCvt)
      return false; // more than one consumer
    theCvt = cvt;
  }
  if (!theCvt)
    return false;
  // The f32 #mma input must come straight from a dot accumulator: either a
  // tt.dot result, or a block argument (the scf.for / cf loop carry of the
  // accumulator). Any other producer (negf/add/... on the #mma, as in
  // solve_tril's merge kernel) is rejected — the mid-end sinks the elementwise
  // op through the truncf onto a bf16 simdgroup fragment, which crashes the AGX
  // PSO materializer.
  Value in = tf.getIn();
  if (isa<BlockArgument>(in))
    return true;
  Operation *def = in.getDefiningOp();
  return def && isa<triton::DotOp>(def);
}

static llvm::DenseSet<Type> computeFragmentEligibleTypes(ModuleOp mod) {
  llvm::DenseSet<Type> eligible;
  llvm::DenseSet<Type> blocked;

  // The accumulator reaches this pass after SCF→ControlFlow, so the loop carry
  // is a cf.br/cf.cond_br block-argument forward, not an scf.yield. The kkt
  // op-web additionally keeps the fragment on the register path through
  // expand_dims (slice<#mma>→#mma), broadcast (col/row replicate), and the
  // per-slot elementwise mask ops (cmpi/andi/select + the f32 binary/unary
  // ops). Each of those has a fragment lowering, so recognizing them here is
  // what flips the atomic gate that admits kkt's accumulator type.
  auto isDotChainUse = [](Operation *user, Value v) -> bool {
    if (isa<triton::DotOp>(user))
      return true; // dot consumes #mma only as accumulator C
    if (isa<scf::YieldOp>(user) || isa<scf::ForOp>(user) ||
        isa<cf::BranchOp>(user) || isa<cf::CondBranchOp>(user))
      return true;
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(user)) {
      auto resTy = dyn_cast<RankedTensorType>(cvt.getType());
      return resTy && !isa<AppleMmaEncodingAttr>(resTy.getEncoding());
    }
    // f16/bf16 accumulator epilogue: the f32 fragment narrowed to f16/bf16
    // before the store convert. Keeps the f32 accumulator on the fragment path.
    if (isAccumTruncEpilogue(user))
      return true;
    // Fragment elementwise / view consumers (kkt chain). A use is clean ONLY if
    // the consumer is fully fragment-lowerable: its #mma result is a fragment
    // candidate AND every ranked-tensor operand is itself a same-encoding
    // fragment candidate (or, for expand_dims, a slice of the #mma parent).
    // This rejects mixed-layout elementwise (e.g. chunk_delta_h's
    // `load(#blocked) - dot(#mma)`) which has no per-slot fragment lowering and
    // would otherwise be wrongly admitted, then fall back to a flat pack and
    // crash. broadcast/expand_dims take a single operand.
    auto sameMmaFragmentOperands = [](Operation *u) {
      for (Value o : u->getOperands()) {
        auto rt = dyn_cast<RankedTensorType>(o.getType());
        if (!rt)
          continue; // scalar operand (none expected here)
        if (!isFragmentCandidateTensor(rt))
          return false;
      }
      return true;
    };
    if (isa<arith::CmpIOp, arith::AndIOp, arith::SelectOp, arith::AddFOp,
            arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::NegFOp,
            math::ExpOp>(user)) {
      bool mmaResult = false;
      for (Value r : user->getResults())
        if (isFragmentCandidateTensor(r.getType()))
          mmaResult = true;
      return mmaResult && sameMmaFragmentOperands(user);
    }
    if (isa<triton::BroadcastOp>(user)) {
      auto resTy = dyn_cast<RankedTensorType>(user->getResult(0).getType());
      auto srcTy = dyn_cast<RankedTensorType>(user->getOperand(0).getType());
      return resTy && srcTy && isFragmentCandidateTensor(resTy) &&
             isFragmentCandidateTensor(srcTy);
    }
    if (auto ed = dyn_cast<triton::ExpandDimsOp>(user)) {
      auto resTy = dyn_cast<RankedTensorType>(ed.getType());
      return resTy && isFragmentCandidateTensor(resTy);
    }
    return false;
  };

  // A fragment-candidate value can only live on the fragment path if its
  // PRODUCER emits a fragment struct. The dot, the fragment elementwise/view
  // patterns, a splat constant, and a loop-carried block argument all do. A
  // convert_layout INTO #mma (e.g. chunk_delta_h materializes b_v as #blocked
  // then converts to #mma for `load - dot`) has no fragment lowering and yields
  // a flat struct, so any type that some value reaches via such a producer must
  // stay flat — otherwise the flat producer's element count collides with the
  // fragment struct slot count at pack time.
  auto isFragmentProducer = [](Value v) -> bool {
    if (isa<BlockArgument>(v))
      return true; // loop carry / entry forward
    Operation *def = v.getDefiningOp();
    if (!def)
      return true;
    if (isa<triton::DotOp, triton::ExpandDimsOp, triton::BroadcastOp,
            arith::CmpIOp, arith::AndIOp, arith::SelectOp, arith::AddFOp,
            arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::NegFOp,
            math::ExpOp>(def))
      return true;
    if (isAccumTruncEpilogue(def))
      return true; // the f16/bf16 truncf emits a fragment struct
    if (auto c = dyn_cast<arith::ConstantOp>(def)) {
      auto sp = dyn_cast<DenseElementsAttr>(c.getValue());
      return sp && sp.isSplat();
    }
    return false;
  };

  // The f16/bf16 truncf-epilogue result rides the fragment ABI alongside the
  // f32 candidates, but is NOT an isFragmentCandidateTensor (so it can't be
  // dragged into the kkt elementwise web). It is collected/poisoned separately.
  auto isCollectible = [](Value v) -> bool {
    if (isFragmentCandidateTensor(v.getType()))
      return true;
    if (!isHalfMmaTensor(v.getType()))
      return false;
    Operation *def = v.getDefiningOp();
    return def && isAccumTruncEpilogue(def);
  };

  // Fixpoint over candidate VALUES, then collapse to TYPES. A fragment chain is
  // admitted atomically: a value is "bad" if its producer can't emit a fragment
  // struct or any consumer isn't a recognized fragment op; badness then
  // propagates BOTH ways across the fragment elementwise/view web (a bad result
  // poisons the op's #mma operands and vice-versa) so a chain is never half
  // admitted (which previously left a flat producer feeding a fragment consumer
  // and crashed at pack time). A type is eligible only if EVERY value of that
  // type is good.
  llvm::SmallVector<Value> candidates;
  llvm::DenseSet<Value> bad;
  auto collect = [&](Value v) {
    if (isCollectible(v))
      candidates.push_back(v);
  };
  mod.walk([&](Operation *op) {
    for (Value res : op->getResults())
      collect(res);
    for (Region &reg : op->getRegions())
      for (Block &blk : reg)
        for (BlockArgument arg : blk.getArguments())
          collect(arg);
  });

  llvm::SmallVector<Value> worklist;
  auto poison = [&](Value v) {
    if (isCollectible(v) && bad.insert(v).second)
      worklist.push_back(v);
  };
  for (Value v : candidates) {
    if (!isFragmentProducer(v)) {
      poison(v);
      continue;
    }
    for (OpOperand &use : v.getUses())
      if (!isDotChainUse(use.getOwner(), v)) {
        poison(v);
        break;
      }
  }
  // Type→values index: the type converter decides the ABI per TYPE, so badness
  // is a per-type property — if ANY value of a type is bad, EVERY value of it
  // must be (otherwise a flat producer of that type would feed a fragment
  // consumer, or vice-versa, and crash). So poison propagates three ways:
  // through a fragment op's operands, through its results, and across all
  // same-type siblings.
  llvm::DenseMap<Type, SmallVector<Value>> byType;
  for (Value v : candidates)
    byType[v.getType()].push_back(v);

  auto isFragmentOp = [](Operation *o) {
    return isa<triton::ExpandDimsOp, triton::BroadcastOp, arith::CmpIOp,
               arith::AndIOp, arith::SelectOp, arith::AddFOp, arith::SubFOp,
               arith::MulFOp, arith::DivFOp, arith::NegFOp, math::ExpOp>(o);
  };
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Value sib : byType[v.getType()])
      poison(sib);
    if (Operation *def = v.getDefiningOp()) {
      if (isFragmentOp(def))
        for (Value o : def->getOperands())
          poison(o);
    }
    for (OpOperand &use : v.getUses())
      if (isFragmentOp(use.getOwner()))
        for (Value r : use.getOwner()->getResults())
          poison(r);
  }

  llvm::DenseSet<Type> present;
  for (Value v : candidates) {
    present.insert(v.getType());
    if (bad.count(v))
      blocked.insert(v.getType());
  }
  for (Type t : present)
    if (!blocked.count(t))
      eligible.insert(t);
  return eligible;
}

static Value emitFragShuffle(ConversionPatternRewriter &rewriter, Location loc,
                             ModuleOp mod, Value val, Value srcLaneI16);
static Value emitLaneId(ConversionPatternRewriter &rewriter, Location loc,
                        ModuleOp mod);

// ConvertLayoutOp for DotOperandEncoding or blocked→blocked:
//
// - DotOperandEncoding target: identity pass-through (elements same per thread)
// - blocked→blocked: TG scatter/gather redistribution
struct ConvertLayoutOpAppleConversion
    : public mlir::ConvertOpToLLVMPattern<ttg::ConvertLayoutOp> {
  using mlir::ConvertOpToLLVMPattern<
      ttg::ConvertLayoutOp>::ConvertOpToLLVMPattern;

  // Per-context counter for unique TG global names
  static unsigned &getCounter(MLIRContext *ctx) {
    static llvm::DenseMap<MLIRContext *, unsigned> counters;
    return counters[ctx];
  }

  // Stable pool key for a shared convert TG global. Conversions with the same
  // TG element type alias one buffer (their live ranges are barrier-disjoint),
  // so the key only needs to distinguish element widths and pointer storage.
  static std::string getCvtPoolKey(Type tgElemTy) {
    if (isa<LLVMPointerType>(tgElemTy))
      return "p";
    return ("i" + llvm::Twine(tgElemTy.getIntOrFloatBitWidth())).str();
  }

  LogicalResult
  matchAndRewrite(ttg::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = cast<RankedTensorType>(op.getResult().getType());

    // DotOperandEncoding target — identity pass-through when source is
    // blocked encoding.  Our dot lowering looks through convert_layout to
    // get source blocked values and uses blocked encoding for scatter/gather.
    // When the source is MMA encoding, the element count per thread differs
    // from the DotOperandEncoding count, so we must NOT pass through
    // (the shared LinearLayout-based convert_layout handles that case).
    if (isa<ttg::DotOperandEncodingAttr>(dstTy.getEncoding()) &&
        isa<ttg::BlockedEncodingAttr>(srcTy.getEncoding())) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }

    // A/B operands of an AppleMma dot never need lowering: the dot pattern
    // reads operand registers from the convert SOURCE (resolveOperand peels
    // the cvt) or straight from SMEM/device, so the converted struct is dead
    // in every dot path. Lowering it anyway burns a __tg_cvt strip plus
    // stores+barriers DCE cannot strip, which pushes pipelined kernels at
    // the 32KB staging cap over the TG budget.
    if (!op->use_empty() &&
        isa<ttg::BlockedEncodingAttr>(srcTy.getEncoding())) {
      bool onlyMmaABUses = true;
      for (OpOperand &use : op->getUses()) {
        auto dot = dyn_cast<triton::DotOp>(use.getOwner());
        if (!dot || use.getOperandNumber() >= 2 ||
            !isa<AppleMmaEncodingAttr>(
                cast<RankedTensorType>(dot.getType()).getEncoding())) {
          onlyMmaABUses = false;
          break;
        }
      }
      if (onlyMmaABUses) {
        rewriter.replaceOp(op, adaptor.getSrc());
        return success();
      }
    }

    // blocked→blocked redistribution via TG scatter/gather, plus the
    // #mma (AppleMma) → #blocked C-output conversion (see mma branch below).
    auto srcEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    auto srcMmaEnc = dyn_cast<AppleMmaEncodingAttr>(srcTy.getEncoding());
    auto dstEnc = dyn_cast<ttg::BlockedEncodingAttr>(dstTy.getEncoding());
    // Destination must be blocked; source must be blocked OR AppleMma.
    if (!dstEnc || (!srcEnc && !srcMmaEnc))
      return failure();

    // When the conversion is within a single simdgroup (or a pure register
    // shuffle) it needs no shared memory: hand it to the shared upstream
    // ConvertLayout pattern, which moves the values with simd_shuffle (Apple
    // TargetInfo implements shuffleIdx/shuffleXor/permute). This is what turns
    // the #mma -> #blocked C-output convert into a register+shuffle epilogue
    // (no __tg_cvt buffer, no barriers) once StoreShuffleLayout has re-laid the
    // store into the simdgroup-shuffle layout. The threadgroup scatter/gather
    // below stays as the fallback for the cross-warp converts that genuinely
    // need shared memory (the upstream smem path miscompiles some replicated
    // fp16/bf16 cases, which is exactly why those route through here instead).
    // Pointer-element tensors stay on the TG path: the upstream shuffle
    // pattern cannot move !tt.ptr elements.
    // #mma sources stay on the in-tree TG-scatter epilogue: the upstream
    // simd-shuffle pattern builds a ColumnAction from the source LinearLayout
    // and asserts on the fragment struct's element count (LinearLayout.cpp
    // ColumnAction::apply). The TG-scatter path reads the fragment via the
    // (fragIdx, vecIdx) slot map and is correct (just slower).
    if (!srcMmaEnc && !isa<triton::PointerType>(srcTy.getElementType()) &&
        !cvtNeedsSharedMemory(srcTy, dstTy))
      return failure();

    // Same encoding — identity (blocked→blocked only)
    if (srcEnc && srcEnc == dstEnc) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();
    auto shape = srcTy.getShape();

    // 1D blocked→blocked: simple TG scatter/gather with flat indices
    if (shape.size() == 1) {
      return convertLayout1D(op, adaptor, rewriter, srcEnc, dstEnc, srcTy,
                             dstTy, loc, ctx, mod);
    }

    // #mma (fragment) -> W-blocked store epilogue: when StoreShuffleLayout has
    // re-laid the store into the within-simdgroup "W" layout, the convert moves
    // every C element to a lane in the SAME simdgroup it already occupies, so
    // it needs no shared memory. Lower it as a register-only air.simd_shuffle
    // restripe (oracle-validated in tools/fragment-oracle) instead of the TG
    // scatter/gather below. On any unexpected shape we fall through to the
    // (always-correct) TG path.
    if (srcMmaEnc && !cvtNeedsSharedMemory(srcTy, dstTy)) {
      if (succeeded(convertMmaToWShuffle(op, adaptor, rewriter, srcMmaEnc,
                                         dstEnc, srcTy, dstTy, loc, ctx, mod)))
        return success();
    }

    // ND blocked->blocked: handle rank>=2 by operating on the trailing two
    // dimensions (rows, cols). For rank>2 we require every leading dim to be
    // size 1 so the whole tensor is a single (rows x cols) tile; this is the
    // case the upstream transferWithinBlockSwizzling miscompiles for fp16/bf16
    // replicated layouts (it vectorizes the replicated registers into a
    // <2 x half> store at a 2-byte stride but reads them back at a 4-byte
    // stride, corrupting the row+16 slot). Our scatter/gather is fully scalar
    // and addresses TG by (row,col) coordinate, so it is correct regardless of
    // replication. Routing the convert here keeps the fix entirely in-tree.
    unsigned rank = shape.size();
    if (rank < 2)
      return failure();
    unsigned rd = rank - 2; // row dim index
    unsigned cd = rank - 1; // col dim index
    for (unsigned d = 0; d < rd; ++d)
      if (shape[d] != 1)
        return failure();

    int64_t rows = shape[rd], cols = shape[cd];
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto tgPtrTy = LLVMPointerType::get(ctx, 3);

    // For pointer elements, use i64 in TG (Metal can't store ptrs in TG)
    bool isPointerElem = isa<LLVMPointerType>(elemTy);
    Type tgElemTy = isPointerElem ? i64Ty : elemTy;

    // Get lane/warp IDs (same helpers as DotOp)
    auto laneIdFnTy = LLVMFunctionType::get(i32Ty, {}, false);
    LLVMFuncOp laneIdFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn =
              mod.lookupSymbol<LLVMFuncOp>("air.thread_index_in_simdgroup"))
        laneIdFn = fn;
      else
        laneIdFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                      "air.thread_index_in_simdgroup",
                                      laneIdFnTy, Linkage::External);
    }

    auto arrI32x3Ty = LLVMArrayType::get(i32Ty, 3);
    auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
    LLVMFuncOp tidFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn = mod.lookupSymbol<LLVMFuncOp>(
              "air.thread_position_in_threadgroup"))
        tidFn = fn;
      else
        tidFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                   "air.thread_position_in_threadgroup",
                                   tidFnTy, Linkage::External);
    }

    auto barrFnTy =
        LLVMFunctionType::get(LLVMVoidType::get(ctx), {i32Ty, i32Ty}, false);
    LLVMFuncOp tgBarrFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn = mod.lookupSymbol<LLVMFuncOp>("air.threadgroup.barrier"))
        tgBarrFn = fn;
      else
        tgBarrFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                      "air.threadgroup.barrier", barrFnTy,
                                      Linkage::External);
    }

    Value laneId =
        LLVM::CallOp::create(rewriter, loc, laneIdFn, ValueRange{}).getResult();
    Value tidStruct =
        LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
    Value tid32 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                               ArrayRef<int64_t>{0});
    Value c32 = arith::ConstantIntOp::create(rewriter, loc, 32, 32);
    Value warpId = arith::DivUIOp::create(rewriter, loc, tid32, c32);

    // Create TG global for tiled scatter/gather.
    // Use the largest strip height (multiple of 8) that fits in 32KB TG.
    // Fewer strips = fewer barriers = better performance.
    // Floor to rows when full tensor already fits.
    // Account for global_smem (from allocate-shared-memory pass) and
    // MMA dot TG buffers (from tt.dot pre-scan) in the 32KB TG budget.
    constexpr int64_t tgBudgetBytes = 32 * 1024;
    int64_t smemBytes = 0;
    bool smemLive = true;
    if (auto attr = mod->getAttrOfType<BoolAttr>("applegpu.smem_live"))
      smemLive = attr.getValue();
    if (smemLive)
      if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.shared"))
        smemBytes = attr.getValue().getZExtValue();
    int64_t mmaBytes = 0;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.mma_shared"))
      mmaBytes = attr.getValue().getZExtValue();
    int64_t availBytes = tgBudgetBytes - smemBytes - mmaBytes;
    int64_t elemBytes = isPointerElem ? 8 : elemTy.getIntOrFloatBitWidth() / 8;
    // Fit as many rows as possible within the remaining TG budget.
    int64_t capRows = (availBytes / elemBytes) / cols;
    int64_t maxStripRows;
    if (capRows >= 8)
      maxStripRows = capRows - (capRows % 8); // round down to 8
    else
      maxStripRows = std::max<int64_t>(capRows, 1); // budget below 8 rows
    int64_t stripRows = std::min(maxStripRows, rows);
    int64_t tgStripSize = stripRows * cols;
    int64_t tgSize = tgStripSize;
    // Pool every 2D convert scatter/gather into ONE shared TG global per
    // (element-type) key, sized to the running max, instead of a fresh
    // counter-numbered buffer per conversion. Each convert round-trips
    // through this buffer fully fenced (scatter, barrier, gather, barrier),
    // so distinct conversions never have overlapping live TG ranges and can
    // safely alias the same storage. Sharing one buffer keeps the addrspace(3)
    // footprint at a single tile instead of N tiles, which is what kept the
    // fused fp16 epilogue (3 separate 64x64 half buffers) under the 32KB cap.
    std::string tgName =
        ("__tg_cvt_" + llvm::Twine(getCvtPoolKey(tgElemTy))).str();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      auto existing = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
      if (!existing) {
        auto arrTy = LLVMArrayType::get(tgElemTy, tgSize);
        LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy, false,
                               Linkage::Internal, tgName, Attribute(),
                               isPointerElem ? 8 : 4, 3u);
      } else if (auto exAT = dyn_cast<LLVMArrayType>(existing.getGlobalType());
                 exAT && (int64_t)exAT.getNumElements() < tgSize) {
        existing.setGlobalTypeAttr(
            TypeAttr::get(LLVMArrayType::get(tgElemTy, tgSize)));
      }
    }
    auto tgGlobal = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
    Value tgPtr =
        LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgGlobal.getName());

    // Helper: compute base (row, col) with wrap + in-bounds predicate
    // Returns {bR, bC, pred} where pred is true if this thread owns valid data
    auto makeBase =
        [&](ttg::BlockedEncodingAttr enc) -> std::tuple<Value, Value, Value> {
      auto spt = enc.getSizePerThread();
      auto tpw = enc.getThreadsPerWarp();
      auto wpc = enc.getWarpsPerCTA();
      int64_t sM = spt[rd], sN = spt[cd];
      int64_t tM = tpw[rd], tN = tpw[cd];
      int64_t wM = wpc[rd], wN = wpc[cd];
      int64_t tileM = wM * tM * sM;
      int64_t tileN = wN * tN * sN;

      Value wN_v = arith::ConstantIntOp::create(rewriter, loc, wN, 32);
      Value tN_v = arith::ConstantIntOp::create(rewriter, loc, tN, 32);
      Value tMsM = arith::ConstantIntOp::create(rewriter, loc, tM * sM, 32);
      Value sM_v = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
      Value tNsN = arith::ConstantIntOp::create(rewriter, loc, tN * sN, 32);
      Value sN_v = arith::ConstantIntOp::create(rewriter, loc, sN, 32);

      // Respect layout order: order[0] is the fastest-changing dimension
      auto order = enc.getOrder();
      bool colFastest =
          (order[0] == cd); // order=[..,cd,rd] => col fastest (default)

      // Warp decomposition: faster dim uses mod, slower uses div
      Value wR, wC;
      if (colFastest) {
        wR = arith::DivUIOp::create(rewriter, loc, warpId, wN_v);
        wC = arith::RemUIOp::create(rewriter, loc, warpId, wN_v);
      } else {
        Value wM_v = arith::ConstantIntOp::create(rewriter, loc, wM, 32);
        wR = arith::RemUIOp::create(rewriter, loc, warpId, wM_v);
        wC = arith::DivUIOp::create(rewriter, loc, warpId, wM_v);
      }
      // Lane decomposition: faster dim uses mod, slower uses div
      Value lR, lC;
      if (colFastest) {
        lR = arith::DivUIOp::create(rewriter, loc, laneId, tN_v);
        lC = arith::RemUIOp::create(rewriter, loc, laneId, tN_v);
      } else {
        Value tM_v = arith::ConstantIntOp::create(rewriter, loc, tM, 32);
        lR = arith::RemUIOp::create(rewriter, loc, laneId, tM_v);
        lC = arith::DivUIOp::create(rewriter, loc, laneId, tM_v);
      }

      Value bR = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, wR, tMsM),
          arith::MulIOp::create(rewriter, loc, lR, sM_v));
      Value bC = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, wC, tNsN),
          arith::MulIOp::create(rewriter, loc, lC, sN_v));

      // Compute in-bounds predicate before wrapping
      Value pred;
      auto i1Ty = IntegerType::get(ctx, 1);
      if (tileM > rows || tileN > cols) {
        Value trueVal = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
        pred = trueVal;
        if (tileM > rows) {
          // Check: bR + max_sM_offset < rows (i.e. bR < rows since sM offsets
          // are 0-based)
          Value rowsV = arith::ConstantIntOp::create(rewriter, loc, rows, 32);
          Value inR = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::ult, bR, rowsV);
          pred = arith::AndIOp::create(rewriter, loc, pred, inR);
          bR = arith::RemUIOp::create(rewriter, loc, bR, rowsV);
        }
        if (tileN > cols) {
          Value colsV = arith::ConstantIntOp::create(rewriter, loc, cols, 32);
          Value inC = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::ult, bC, colsV);
          pred = arith::AndIOp::create(rewriter, loc, pred, inC);
          bC = arith::RemUIOp::create(rewriter, loc, bC, colsV);
        }
      } else {
        pred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
      }
      return {bR, bC, pred};
    };

    // Per-lane base (row,col) for an AppleMma source under the PHYSICAL
    // toLinearLayout (AppleMmaLayoutConversions.cpp). emitOffsetForLayout fixes
    // lane=0, warp=0 and enumerates only the register in-dim, so the lane+warp
    // contribution to the absolute (row,col) must be added here. The physical
    // per-lane storage is:
    //   phys_row = L1 | (L2<<1) | (L4<<2)
    //   phys_col = (L0<<1) | (L3<<2)        (register supplies col bit 0)
    // with column-major warp tiling (warpRow = warpId/wN, warpCol = warpId%wN,
    // warpOrder={1,0}), each warp owning an 8-row/8-col simdgroup tile step.
    // The owned offsets enumerated by emitOffsetForLayout already cover exactly
    // the in-bounds MxN tensor positions, so the predicate is always true.
    auto makeBaseMma =
        [&](AppleMmaEncodingAttr enc) -> std::tuple<Value, Value, Value> {
      auto wpc = enc.getWarpsPerCTA();
      int64_t wN = wpc[cd];
      auto bit = [&](Value v, int64_t shift) -> Value {
        Value s = arith::ShRUIOp::create(
            rewriter, loc, v,
            arith::ConstantIntOp::create(rewriter, loc, shift, 32));
        return arith::AndIOp::create(
            rewriter, loc, s,
            arith::ConstantIntOp::create(rewriter, loc, 1, 32));
      };
      auto shl = [&](Value v, int64_t shift) -> Value {
        return arith::ShLIOp::create(
            rewriter, loc, v,
            arith::ConstantIntOp::create(rewriter, loc, shift, 32));
      };
      // phys_row = L1 | (L2<<1) | (L4<<2)
      Value physRow = arith::OrIOp::create(
          rewriter, loc,
          arith::OrIOp::create(rewriter, loc, bit(laneId, 1),
                               shl(bit(laneId, 2), 1)),
          shl(bit(laneId, 4), 2));
      // phys_col = (L0<<1) | (L3<<2)
      Value physCol = arith::OrIOp::create(
          rewriter, loc, shl(bit(laneId, 0), 1), shl(bit(laneId, 3), 2));
      Value c8 = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
      Value wN_v = arith::ConstantIntOp::create(rewriter, loc, wN, 32);
      Value warpRow = arith::DivUIOp::create(rewriter, loc, warpId, wN_v);
      Value warpCol = arith::RemUIOp::create(rewriter, loc, warpId, wN_v);
      Value bR = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, warpRow, c8),
          physRow);
      Value bC = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, warpCol, c8),
          physCol);
      Value truePred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
      return {bR, bC, truePred};
    };

    // Use LinearLayout-based offsets (matches upstream element ordering).
    // The source may be blocked or AppleMma; both expose toLinearLayout, so
    // emitOffsetForLayout enumerates this lane's owned per-register offsets.
    Attribute srcEncAttr = srcMmaEnc ? Attribute(srcMmaEnc) : Attribute(srcEnc);
    auto srcOffsets = emitOffsetForLayout(srcEncAttr, srcTy);
    auto dstOffsets = emitOffsetForLayout(dstEnc, dstTy);

    // Convert to (row, col) pairs
    SmallVector<std::pair<int64_t, int64_t>> srcCoords, dstCoords;
    for (auto &off : srcOffsets)
      srcCoords.push_back({off[rd], off[cd]});
    for (auto &off : dstOffsets)
      dstCoords.push_back({off[rd], off[cd]});

    // Unpack source elements
    Value src = adaptor.getSrc();
    SmallVector<Value> srcElems;
    auto sStructTy = dyn_cast<LLVMStructType>(src.getType());
    bool srcFragment = srcMmaEnc && sStructTy && !sStructTy.getBody().empty() &&
                       isa<VectorType>(sStructTy.getBody()[0]);
    if (srcFragment) {
      // #mma fragment struct → per-element scalars via the (fragIdx, vecIdx)
      // slot map, so the downstream TG-scatter epilogue sees flat scalars.
      auto info = applegpu::getAppleMmaFragmentInfo(srcTy, srcMmaEnc);
      auto fragElemTy =
          cast<VectorType>(sStructTy.getBody()[0]).getElementType();
      // The fragment carries f32 even for an f16/bf16 #mma accumulator (the
      // narrowing is deferred here). Narrow each EXTRACTED SCALAR to the dst
      // element type — a scalar fptrunc the AGX JIT compiles correctly, unlike
      // a vector bf16 round-trip on the simdgroup register.
      Type dstElemTy = getTypeConverter()->convertType(dstTy.getElementType());
      bool narrowScalar = fragElemTy != dstElemTy &&
                          isa<FloatType>(fragElemTy) &&
                          isa<FloatType>(dstElemTy) &&
                          dstElemTy.getIntOrFloatBitWidth() <
                              fragElemTy.getIntOrFloatBitWidth();
      SmallVector<Value> frags;
      for (unsigned i = 0; i < sStructTy.getBody().size(); ++i)
        frags.push_back(ExtractValueOp::create(rewriter, loc,
                                               sStructTy.getBody()[i], src,
                                               ArrayRef<int64_t>{(int64_t)i}));
      srcElems.resize(srcCoords.size());
      for (size_t i = 0; i < srcCoords.size(); ++i) {
        int64_t fragIdx, vecIdx;
        applegpu::appleMmaFragmentSlot(srcCoords[i].first, srcCoords[i].second,
                                       info, fragIdx, vecIdx);
        Value frag = (fragIdx < (int64_t)frags.size()) ? frags[fragIdx]
                     : frags.empty()                   ? Value()
                                                       : frags[0];
        Value vIdx = arith::ConstantIntOp::create(rewriter, loc, vecIdx, 32);
        Value sc = LLVM::ExtractElementOp::create(rewriter, loc, fragElemTy,
                                                  frag, vIdx);
        if (narrowScalar)
          sc = arith::TruncFOp::create(rewriter, loc, dstElemTy, sc);
        srcElems[i] = sc;
      }
    } else if (sStructTy) {
      for (unsigned i = 0; i < sStructTy.getBody().size(); ++i)
        srcElems.push_back(
            ExtractValueOp::create(rewriter, loc, sStructTy.getBody()[i], src,
                                   ArrayRef<int64_t>{(int64_t)i}));
    } else {
      srcElems = {src};
    }

    if (srcElems.size() != srcCoords.size())
      return failure();

    auto [srcBaseRow, srcBaseCol, srcPred] =
        srcMmaEnc ? makeBaseMma(srcMmaEnc) : makeBase(srcEnc);
    auto [dstBaseRow, dstBaseCol, dstPred] = makeBase(dstEnc);

    // Strip flat index: row offset relative to strip start
    auto stripFlatIdx = [&](Value bR, Value bC, int64_t rOff, int64_t cOff,
                            int64_t stripRowStart) -> Value {
      Value r =
          arith::AddIOp::create(rewriter, loc, bR,
                                arith::ConstantIntOp::create(
                                    rewriter, loc, rOff - stripRowStart, 32));
      Value c = arith::AddIOp::create(
          rewriter, loc, bC,
          arith::ConstantIntOp::create(rewriter, loc, cOff, 32));
      Value f = arith::AddIOp::create(
          rewriter, loc,
          arith::MulIOp::create(
              rewriter, loc, r,
              arith::ConstantIntOp::create(rewriter, loc, cols, 32)),
          c);
      return arith::ExtUIOp::create(rewriter, loc, i64Ty, f);
    };

    Value fenceTG = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    Value execMod = arith::ConstantIntOp::create(rewriter, loc, 4, 32);

    // Initialize destination elements with undef (will be filled strip by
    // strip)
    SmallVector<Value> dstElems(dstCoords.size());
    Value zeroElem;
    if (isPointerElem) {
      // The masked-out fallback for a pointer-typed lane must NOT be a null
      // (address 0) device pointer. Metal's AIR materializer refuses to build a
      // PSO for a kernel that can store through a constant-null device pointer
      // ("Failed to materializeAll"), even when that store is predicated off.
      // Reuse a real input pointer (srcElems[0]) as the neutral element: it is
      // a valid pointer of the right type and address space, and is only ever
      // selected into masked-out lanes whose stores are disabled, so it is
      // never actually dereferenced. This keeps the materializer happy without
      // changing observable behaviour.
      if (!srcElems.empty()) {
        zeroElem = srcElems[0];
      } else {
        Value zeroInt = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
        zeroElem = LLVM::IntToPtrOp::create(rewriter, loc, elemTy, zeroInt);
      }
    } else {
      zeroElem = arith::ConstantOp::create(rewriter, loc, elemTy,
                                           rewriter.getZeroAttr(elemTy));
    }
    for (size_t i = 0; i < dstElems.size(); ++i)
      dstElems[i] = zeroElem;

    // Tiled scatter/gather: process stripRows rows at a time.
    // When rows*cols fits in 32KB, stripRows==rows → single pass (no overhead).
    // Otherwise tiles down to fit, adding 2 barriers per extra strip.
    int64_t numStrips = (rows + stripRows - 1) / stripRows;
    for (int64_t strip = 0; strip < numStrips; ++strip) {
      int64_t rowStart = strip * stripRows;
      int64_t rowEnd = std::min(rowStart + stripRows, rows);

      // Scatter source elements for this strip (skip out-of-strip via a
      // predicated store). pred = srcPred && inStrip(rOff); srcPred is a single
      // shared in-bounds value and inStrip depends only on rOff, so all
      // elements in a row share one predicate. Group by rOff and emit one
      // conditional block per distinct row instead of per element, keeping the
      // block count proportional to strip rows rather than the full tile.
      std::map<int64_t, SmallVector<size_t>> srcByRow;
      SmallVector<int64_t> srcRowOrder;
      for (size_t i = 0; i < srcElems.size(); ++i) {
        int64_t rOff = srcCoords[i].first;
        if (srcByRow.find(rOff) == srcByRow.end())
          srcRowOrder.push_back(rOff);
        srcByRow[rOff].push_back(i);
      }
      bool singleStripSrc = (numStrips == 1);
      for (int64_t rOff : srcRowOrder) {
        // Single strip => every row is in-strip, so pred collapses to srcPred
        // and the row-range compare is dropped.
        Value pred;
        if (singleStripSrc) {
          pred = srcPred;
        } else {
          Value actualRow = arith::AddIOp::create(
              rewriter, loc, srcBaseRow,
              arith::ConstantIntOp::create(rewriter, loc, rOff, 32));
          Value inStrip = arith::AndIOp::create(
              rewriter, loc,
              arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::uge, actualRow,
                  arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
              arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::ult, actualRow,
                  arith::ConstantIntOp::create(rewriter, loc, rowEnd, 32)));
          pred = arith::AndIOp::create(rewriter, loc, srcPred, inStrip);
        }
        auto *curBlock = rewriter.getInsertionBlock();
        auto curPoint = rewriter.getInsertionPoint();
        auto *endBlock = curBlock->splitBlock(curPoint);
        auto *thenBlock = rewriter.createBlock(endBlock);
        rewriter.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(rewriter, loc, pred, thenBlock, endBlock);
        rewriter.setInsertionPointToEnd(thenBlock);
        for (size_t i : srcByRow[rOff]) {
          int64_t cOff = srcCoords[i].second;
          Value idx =
              stripFlatIdx(srcBaseRow, srcBaseCol, rOff, cOff, rowStart);
          Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, tgElemTy,
                                          tgPtr, ArrayRef<LLVM::GEPArg>{idx});
          Value toStore = srcElems[i];
          if (isPointerElem)
            toStore = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, toStore);
          LLVM::StoreOp::create(rewriter, loc, toStore, gep);
        }
        LLVM::BrOp::create(rewriter, loc, endBlock);
        rewriter.setInsertionPointToStart(endBlock);
      }

      // Barrier: all threads done scattering this strip
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});

      // Gather destination elements for this strip.
      // Use wrapped dstBaseRow (already < rows) for strip check — do NOT
      // gate by dstPred. When tileM > rows, multiple threads wrap to the
      // same row; all need the correct TG value regardless of dstPred.
      // inStrip depends only on rOff, so compute it once per distinct row and
      // reuse it across that row's elements instead of recomputing the strip
      // compare per element.
      std::map<int64_t, SmallVector<size_t>> dstByRow;
      SmallVector<int64_t> dstRowOrder;
      for (size_t i = 0; i < dstCoords.size(); ++i) {
        int64_t rOff = dstCoords[i].first;
        if (dstByRow.find(rOff) == dstByRow.end())
          dstRowOrder.push_back(rOff);
        dstByRow[rOff].push_back(i);
      }
      // Single-strip fast path: the whole tile fits one TG pass, so every
      // destination row is unconditionally in-strip (rowStart=0, rowEnd=rows,
      // dstBaseRow already wrapped < rows). The per-element safeIdx select and
      // the merge select then both reduce to the gathered value, so emit a
      // plain GEP+load+assign and drop the two selects and the per-row strip
      // compare. This is the dominant cost of the #mma->#blocked output convert
      // (its select/icmp chain is ~99% of a 128x128x64 dot's LLVM IR), so the
      // single-strip elision shrinks that IR by roughly half.
      bool singleStrip = (numStrips == 1);
      Value zeroIdx = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
      for (int64_t rOff : dstRowOrder) {
        Value inStrip;
        if (!singleStrip) {
          Value actualRow = arith::AddIOp::create(
              rewriter, loc, dstBaseRow,
              arith::ConstantIntOp::create(rewriter, loc, rOff, 32));
          inStrip = arith::AndIOp::create(
              rewriter, loc,
              arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::uge, actualRow,
                  arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
              arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::ult, actualRow,
                  arith::ConstantIntOp::create(rewriter, loc, rowEnd, 32)));
        }
        for (size_t i : dstByRow[rOff]) {
          int64_t cOff = dstCoords[i].second;
          Value idx =
              stripFlatIdx(dstBaseRow, dstBaseCol, rOff, cOff, rowStart);
          Value safeIdx = singleStrip
                              ? idx
                              : arith::SelectOp::create(rewriter, loc, inStrip,
                                                        idx, zeroIdx)
                                    .getResult();
          Value gep =
              LLVM::GEPOp::create(rewriter, loc, tgPtrTy, tgElemTy, tgPtr,
                                  ArrayRef<LLVM::GEPArg>{safeIdx});
          Value gathered =
              LLVM::LoadOp::create(rewriter, loc, tgElemTy, gep).getResult();
          if (isPointerElem)
            gathered =
                LLVM::IntToPtrOp::create(rewriter, loc, elemTy, gathered);
          // Use gathered value if in strip, keep previous otherwise
          dstElems[i] = singleStrip
                            ? gathered
                            : arith::SelectOp::create(rewriter, loc, inStrip,
                                                      gathered, dstElems[i])
                                  .getResult();
        }
      }

      // Barrier: all threads done gathering before next strip's scatter
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});
    }

    // Pack result
    auto outTy = getTypeConverter()->convertType(dstTy);
    if (!outTy)
      return failure();

    if (auto outSt = dyn_cast<LLVMStructType>(outTy)) {
      if (outSt.getBody().size() != dstElems.size())
        return failure();
      Value result = UndefOp::create(rewriter, loc, outSt);
      for (size_t i = 0; i < dstElems.size(); ++i)
        result =
            InsertValueOp::create(rewriter, loc, outSt, result, dstElems[i],
                                  ArrayRef<int64_t>{(int64_t)i});
      rewriter.replaceOp(op, result);
    } else {
      rewriter.replaceOp(op, dstElems[0]);
    }
    return success();
  }

  // #mma fragment -> W-blocked register shuffle restripe (store epilogue).
  // Each W destination element (row,col) is gathered from the #mma physical
  // lane that owns tile-local (row%8, col%8) via air.simd_shuffle on the
  // col-parity register. Pure registers, no shared memory, no barriers.
  // Validated bit-exact in tools/fragment-oracle (store-restripe case).
  LogicalResult convertMmaToWShuffle(ttg::ConvertLayoutOp op, OpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter,
                                     AppleMmaEncodingAttr srcMmaEnc,
                                     ttg::BlockedEncodingAttr dstEnc,
                                     RankedTensorType srcTy,
                                     RankedTensorType dstTy, Location loc,
                                     MLIRContext *ctx, ModuleOp mod) const {
    unsigned rank = srcTy.getShape().size();
    if (rank < 2)
      return failure();
    unsigned rd = rank - 2, cd = rank - 1;
    for (unsigned d = 0; d < rd; ++d)
      if (srcTy.getShape()[d] != 1)
        return failure();

    // The fragment source must be the vectorized <64 x f32> struct.
    Value src = adaptor.getSrc();
    auto sStructTy = dyn_cast<LLVMStructType>(src.getType());
    if (!sStructTy || sStructTy.getBody().empty() ||
        !isa<VectorType>(sStructTy.getBody()[0]))
      return failure();

    auto outTy = getTypeConverter()->convertType(dstTy);
    auto outSt = dyn_cast_or_null<LLVMStructType>(outTy);
    if (!outSt)
      return failure();
    Type fragElemTy = cast<VectorType>(sStructTy.getBody()[0]).getElementType();
    if (!fragElemTy.isF32())
      return failure();

    auto i16Ty = IntegerType::get(ctx, 16);

    // Per-lane W base (row,col). The W layout is tile-aligned within each warp,
    // so dstEnc's per-register offsets plus this lane's base give the absolute
    // (row,col) every output element lands on.
    Value laneId = emitLaneId(rewriter, loc, mod);
    auto tpw = dstEnc.getThreadsPerWarp();
    auto spt = dstEnc.getSizePerThread();
    auto order = dstEnc.getOrder();
    int64_t tM = tpw[rd], tN = tpw[cd];
    int64_t sM = spt[rd], sN = spt[cd];
    bool colFastest = (order[0] == cd);
    Value tN_v = arith::ConstantIntOp::create(rewriter, loc, tN, 32);
    Value tM_v = arith::ConstantIntOp::create(rewriter, loc, tM, 32);
    Value lR, lC;
    if (colFastest) {
      lR = arith::DivUIOp::create(rewriter, loc, laneId, tN_v);
      lC = arith::RemUIOp::create(rewriter, loc, laneId, tN_v);
    } else {
      lR = arith::RemUIOp::create(rewriter, loc, laneId, tM_v);
      lC = arith::DivUIOp::create(rewriter, loc, laneId, tM_v);
    }
    Value sM_v = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
    Value sN_v = arith::ConstantIntOp::create(rewriter, loc, sN, 32);
    Value laneRow = arith::MulIOp::create(rewriter, loc, lR, sM_v);
    Value laneCol = arith::MulIOp::create(rewriter, loc, lC, sN_v);

    auto dstOffsets = emitOffsetForLayout(dstEnc, dstTy);
    if (dstOffsets.size() != outSt.getBody().size())
      return failure();

    auto srcInfo = applegpu::getAppleMmaFragmentInfo(srcTy, srcMmaEnc);
    SmallVector<Value> frags;
    for (unsigned i = 0; i < sStructTy.getBody().size(); ++i)
      frags.push_back(ExtractValueOp::create(rewriter, loc,
                                             sStructTy.getBody()[i], src,
                                             ArrayRef<int64_t>{(int64_t)i}));

    auto bit = [&](Value v, int64_t shift) -> Value {
      Value s = arith::ShRUIOp::create(
          rewriter, loc, v,
          arith::ConstantIntOp::create(rewriter, loc, shift, 32));
      return arith::AndIOp::create(
          rewriter, loc, s, arith::ConstantIntOp::create(rewriter, loc, 1, 32));
    };
    auto shl = [&](Value v, int64_t shift) -> Value {
      return arith::ShLIOp::create(
          rewriter, loc, v,
          arith::ConstantIntOp::create(rewriter, loc, shift, 32));
    };

    Value result = UndefOp::create(rewriter, loc, outSt);
    for (size_t i = 0; i < dstOffsets.size(); ++i) {
      int64_t rOff = dstOffsets[i][rd], cOff = dstOffsets[i][cd];
      Value absRow = arith::AddIOp::create(
          rewriter, loc, laneRow,
          arith::ConstantIntOp::create(rewriter, loc, rOff, 32));
      Value absCol = arith::AddIOp::create(
          rewriter, loc, laneCol,
          arith::ConstantIntOp::create(rewriter, loc, cOff, 32));
      // tile-local (row%8, col%8) drive the #mma physical source lane:
      //   srcLane = L0:(col>>1) L1:row L2:(row>>1) L3:(col>>2) L4:(row>>2)
      Value tRow = arith::AndIOp::create(
          rewriter, loc, absRow,
          arith::ConstantIntOp::create(rewriter, loc, 7, 32));
      Value tCol = arith::AndIOp::create(
          rewriter, loc, absCol,
          arith::ConstantIntOp::create(rewriter, loc, 7, 32));
      Value srcLane = arith::OrIOp::create(
          rewriter, loc,
          arith::OrIOp::create(rewriter, loc,
                               arith::OrIOp::create(rewriter, loc, bit(tCol, 1),
                                                    shl(bit(tRow, 0), 1)),
                               shl(bit(tRow, 1), 2)),
          arith::OrIOp::create(rewriter, loc, shl(bit(tCol, 2), 3),
                               shl(bit(tRow, 2), 4)));
      Value srcLane16 = arith::TruncIOp::create(rewriter, loc, i16Ty, srcLane);

      // Which fragment register holds (absRow,absCol). fragIdx is constant per
      // owned tile; for the W store one frag per 8x8 tile, vecIdx = col parity.
      int64_t fragIdx, vecIdx;
      applegpu::appleMmaFragmentSlot(rOff, cOff, srcInfo, fragIdx, vecIdx);
      if (fragIdx >= (int64_t)frags.size())
        return failure();
      Value vIdx = arith::ConstantIntOp::create(rewriter, loc, vecIdx, 32);
      Value scalar =
          LLVM::ExtractElementOp::create(rewriter, loc, frags[fragIdx], vIdx);
      Value shuffled = emitFragShuffle(rewriter, loc, mod, scalar, srcLane16);
      // f16/bf16 accumulator: the fragment is f32; narrow the shuffled scalar
      // to the dst element type (a scalar fptrunc, off the simdgroup register).
      Type outElemTy = outSt.getBody()[i];
      if (shuffled.getType() != outElemTy &&
          isa<FloatType>(shuffled.getType()) && isa<FloatType>(outElemTy) &&
          outElemTy.getIntOrFloatBitWidth() <
              shuffled.getType().getIntOrFloatBitWidth())
        shuffled = arith::TruncFOp::create(rewriter, loc, outElemTy, shuffled);
      if (shuffled.getType() != outElemTy)
        return failure();
      result = InsertValueOp::create(rewriter, loc, outSt, result, shuffled,
                                     ArrayRef<int64_t>{(int64_t)i});
    }
    rewriter.replaceOp(op, result);
    return success();
  }

  // 1D blocked→blocked conversion via TG scatter/gather
  LogicalResult convertLayout1D(ttg::ConvertLayoutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter,
                                ttg::BlockedEncodingAttr srcEnc,
                                ttg::BlockedEncodingAttr dstEnc,
                                RankedTensorType srcTy, RankedTensorType dstTy,
                                Location loc, MLIRContext *ctx,
                                ModuleOp mod) const {

    int64_t numElems = srcTy.getShape()[0];
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto tgPtrTy = LLVMPointerType::get(ctx, 3);

    // For pointer elements, use i64 in TG (Metal can't store ptrs in TG)
    bool isPointerElem = isa<LLVMPointerType>(elemTy);
    Type tgElemTy = isPointerElem ? i64Ty : elemTy;

    // Get thread/warp IDs
    auto laneIdFnTy = LLVMFunctionType::get(i32Ty, {}, false);
    LLVMFuncOp laneIdFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn =
              mod.lookupSymbol<LLVMFuncOp>("air.thread_index_in_simdgroup"))
        laneIdFn = fn;
      else
        laneIdFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                      "air.thread_index_in_simdgroup",
                                      laneIdFnTy, Linkage::External);
    }

    auto arrI32x3Ty = LLVMArrayType::get(i32Ty, 3);
    auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
    LLVMFuncOp tidFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn = mod.lookupSymbol<LLVMFuncOp>(
              "air.thread_position_in_threadgroup"))
        tidFn = fn;
      else
        tidFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                   "air.thread_position_in_threadgroup",
                                   tidFnTy, Linkage::External);
    }

    auto barrFnTy =
        LLVMFunctionType::get(LLVMVoidType::get(ctx), {i32Ty, i32Ty}, false);
    LLVMFuncOp tgBarrFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn = mod.lookupSymbol<LLVMFuncOp>("air.threadgroup.barrier"))
        tgBarrFn = fn;
      else
        tgBarrFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                      "air.threadgroup.barrier", barrFnTy,
                                      Linkage::External);
    }

    Value laneId =
        LLVM::CallOp::create(rewriter, loc, laneIdFn, ValueRange{}).getResult();
    Value tidStruct =
        LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
    Value tid32 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                               ArrayRef<int64_t>{0});
    Value c32 = arith::ConstantIntOp::create(rewriter, loc, 32, 32);
    Value warpId = arith::DivUIOp::create(rewriter, loc, tid32, c32);

    // Create TG global (pooled per element type, sized to running max). The
    // 1D scatter/gather is fully fenced like the 2D path, so distinct
    // conversions of the same element type share one barrier-disjoint buffer.
    std::string tgName =
        ("__tg_cvt_" + llvm::Twine(getCvtPoolKey(tgElemTy))).str();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      auto existing = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
      if (!existing) {
        auto arrTy = LLVMArrayType::get(tgElemTy, numElems);
        LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy, false,
                               Linkage::Internal, tgName, Attribute(),
                               isPointerElem ? 8 : 4, 3u);
      } else if (auto exAT = dyn_cast<LLVMArrayType>(existing.getGlobalType());
                 exAT && (int64_t)exAT.getNumElements() < numElems) {
        existing.setGlobalTypeAttr(
            TypeAttr::get(LLVMArrayType::get(tgElemTy, numElems)));
      }
    }
    auto tgGlobal = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
    Value tgPtr =
        LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgGlobal.getName());

    // Compute per-thread base index for 1D blocked layout
    auto computeBase1D = [&](ttg::BlockedEncodingAttr enc) -> Value {
      auto spt = enc.getSizePerThread()[0];
      auto tpw = enc.getThreadsPerWarp()[0];
      auto wpc = enc.getWarpsPerCTA()[0];
      // base = warpId * (tpw * spt) + laneId * spt
      Value tpwSpt = arith::ConstantIntOp::create(rewriter, loc, tpw * spt, 32);
      Value sptV = arith::ConstantIntOp::create(rewriter, loc, spt, 32);
      Value base = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, warpId, tpwSpt),
          arith::MulIOp::create(rewriter, loc, laneId, sptV));
      return base;
    };

    // Use emitOffsetForLayout to get canonical element ordering
    auto srcOffsets = emitOffsetForLayout(srcEnc, srcTy);
    auto dstOffsets = emitOffsetForLayout(dstEnc, dstTy);

    // Unpack source elements
    Value src = adaptor.getSrc();
    SmallVector<Value> srcElems;
    if (auto sTy = dyn_cast<LLVMStructType>(src.getType())) {
      for (unsigned i = 0; i < sTy.getBody().size(); ++i)
        srcElems.push_back(
            ExtractValueOp::create(rewriter, loc, sTy.getBody()[i], src,
                                   ArrayRef<int64_t>{(int64_t)i}));
    } else {
      srcElems = {src};
    }

    Value srcBase = computeBase1D(srcEnc);
    Value dstBase = computeBase1D(dstEnc);

    // Scatter source elements to TG
    for (size_t i = 0; i < srcElems.size(); ++i) {
      int64_t elemOff = srcOffsets[i][0];
      Value idx = arith::AddIOp::create(
          rewriter, loc, srcBase,
          arith::ConstantIntOp::create(rewriter, loc, elemOff, 32));
      Value idx64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, idx);
      Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, tgElemTy, tgPtr,
                                      ArrayRef<LLVM::GEPArg>{idx64});
      Value toStore = srcElems[i];
      if (isPointerElem)
        toStore = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, toStore);
      LLVM::StoreOp::create(rewriter, loc, toStore, gep);
    }

    // Barrier
    Value fenceTG = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    Value execMod = arith::ConstantIntOp::create(rewriter, loc, 4, 32);
    LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

    // Gather destination elements from TG
    SmallVector<Value> dstElems;
    for (size_t i = 0; i < dstOffsets.size(); ++i) {
      int64_t elemOff = dstOffsets[i][0];
      Value idx = arith::AddIOp::create(
          rewriter, loc, dstBase,
          arith::ConstantIntOp::create(rewriter, loc, elemOff, 32));
      Value idx64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, idx);
      Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, tgElemTy, tgPtr,
                                      ArrayRef<LLVM::GEPArg>{idx64});
      Value loaded =
          LLVM::LoadOp::create(rewriter, loc, tgElemTy, gep).getResult();
      if (isPointerElem)
        loaded = LLVM::IntToPtrOp::create(rewriter, loc, elemTy, loaded);
      dstElems.push_back(loaded);
    }

    // Pack result
    auto outTy = getTypeConverter()->convertType(dstTy);
    if (!outTy)
      return failure();

    if (auto outSt = dyn_cast<LLVMStructType>(outTy)) {
      if (outSt.getBody().size() != dstElems.size())
        return failure();
      Value result = UndefOp::create(rewriter, loc, outSt);
      for (size_t i = 0; i < dstElems.size(); ++i)
        result =
            InsertValueOp::create(rewriter, loc, outSt, result, dstElems[i],
                                  ArrayRef<int64_t>{(int64_t)i});
      rewriter.replaceOp(op, result);
    } else {
      rewriter.replaceOp(op, dstElems[0]);
    }
    return success();
  }
};

// Lower triton::AtomicRMWOp → air.atomic.global.{op}.{type}
//
// Metal uses explicit AIR intrinsics for atomics:
//   float @air.atomic.global.add.f32(float addrspace(1)*, float, i32 order, i32
//   scope, i1 volatile) i32   @air.atomic.global.add.s.i32(i32 addrspace(1)*,
//   i32, i32 order, i32 scope, i1 volatile) i32
//   @air.atomic.global.max.s.i32(...) i32   @air.atomic.global.min.s.i32(...)
//   i32   @air.atomic.global.xchg.s.i32(...)
//
// For unsupported native atomics (f32 max/min, f16/bf16 add), we emit a CAS
// loop:
//   air.atomic.global.cmpxchg.weak.i32(ptr, expected_ptr, desired, succ_order,
//   fail_order, scope, vol) returns old i32. Expected is passed by pointer and
//   updated on failure.

// Emit a predicate that is true only for threads that own unique data.
// This prevents redundant threads from executing atomics (e.g. 128 threads
// for a 4-element tensor → only 4 should execute).
// Returns null Value if no predication is needed.
static Value emitAppleRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc, ModuleOp mod) {
  auto *ctx = rewriter.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);

  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");

  // Get thread_position_in_threadgroup (tid within threadgroup)
  auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
  auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
      LLVMFuncOp::create(rewriter, mod.getLoc(),
                         "air.thread_position_in_threadgroup", tidFnTy,
                         Linkage::External);
  }
  auto tidFn =
      mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");
  Value tidStruct =
      LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
  Value tid = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                           ArrayRef<int64_t>{0});

  // laneId = tid % 32, warpId = tid / 32
  int tpw = ttg::lookupThreadsPerWarp(rewriter);
  Value warpSize = arith::ConstantIntOp::create(rewriter, loc, tpw, 32);
  Value laneId = arith::RemUIOp::create(rewriter, loc, tid, warpSize);
  Value warpId = arith::DivUIOp::create(rewriter, loc, tid, warpSize);

  Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  Value pred;

  // Check each dimension: (dimId & mask) == 0 means this thread is canonical
  auto dimNames = {kLane, kWarp};
  auto dimIds = {laneId, warpId};
  for (auto [dimName, dimId] : llvm::zip(dimNames, dimIds)) {
    auto it = freeVarMasks.find(dimName);
    int32_t mask = (it != freeVarMasks.end()) ? it->second : 0;
    if (mask != 0) {
      Value maskVal = arith::ConstantIntOp::create(rewriter, loc, mask, 32);
      Value masked = LLVM::AndOp::create(rewriter, loc, dimId, maskVal);
      Value dimPred = LLVM::ICmpOp::create(
          rewriter, loc, LLVM::ICmpPredicate::eq, masked, zero);
      if (pred)
        pred = LLVM::AndOp::create(rewriter, loc, pred, dimPred);
      else
        pred = dimPred;
    }
  }
  return pred;
}

// Combine two predicates with AND, handling null values.
static Value extractFirstElemScalar(Value tensor, bool peelModulo);
static bool extractFirstElemConst(Value tensor, int64_t &out, bool peelModulo);

static Value maybeAnd(ConversionPatternRewriter &rewriter, Location loc,
                      Value a, Value b) {
  if (!a)
    return b;
  if (!b)
    return a;
  return LLVM::AndOp::create(rewriter, loc, a, b);
}

struct AtomicRMWOpAppleConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Declare an AIR intrinsic function if not already declared
  LLVMFuncOp declareAIR(ConversionPatternRewriter &rewriter, ModuleOp mod,
                        StringRef name, Type retTy,
                        ArrayRef<Type> argTys) const {
    if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
      return fn;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto fnTy = LLVMFunctionType::get(retTy, argTys, false);
    return LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                              Linkage::External);
  }

  // Emit a direct AIR atomic intrinsic call (no CAS loop)
  Value emitDirectAtomic(ConversionPatternRewriter &rewriter, Location loc,
                         ModuleOp mod, StringRef airName, Type valueTy,
                         Value ptr) const {
    auto *ctx = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto fn = declareAIR(rewriter, mod, airName, valueTy,
                         {ptrTy, valueTy, i32Ty, i32Ty, i1Ty});
    // Unused return needed: the call still needs a value operand.
    // Actually this helper is called from emitDirectAtomicCall below.
    (void)fn;
    return {};
  }

  // Emit CAS loop for f32 max/min:
  //   alloca expected
  //   load old from *ptr (via xchg 0 trick or just initial load)
  //   loop:
  //     store old → expected
  //     new_f = max/min(old_f, val_f)
  //     new_i = bitcast new_f → i32
  //     old_i = bitcast old_f → i32
  //     store old_i → expected
  //     old_ret = cmpxchg(ptr_i32, &expected, new_i, ...)
  //     expected_after = load expected
  //     cmp = icmp eq old_ret, old_i (success if unchanged)
  //     br cmp → done, loop
  //   done:
  //     result = bitcast old_ret → float
  Value emitF32CASLoop(ConversionPatternRewriter &rewriter, Location loc,
                       ModuleOp mod, Value ptr, Value val, RMWOp rmwOp) const {
    auto *ctx = rewriter.getContext();
    auto f32Ty = Float32Type::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);  // device
    auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0); // private (alloca)

    // Declare cmpxchg intrinsic
    auto cmpxchgFn =
        declareAIR(rewriter, mod, "air.atomic.global.cmpxchg.weak.i32", i32Ty,
                   {ptrTy, ptrTy0, i32Ty, i32Ty, i32Ty, i32Ty, i1Ty});

    // Alloca for expected value (i32)
    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    Value expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, i32Ty,
                                                  one, /*alignment=*/4);

    // Initial load: use xchg to atomically read current value
    // Actually, a simple non-atomic load is fine for the initial guess —
    // the CAS loop will retry if it's stale.
    Value oldI32 = LLVM::LoadOp::create(rewriter, loc, i32Ty, ptr);
    Value oldF32 = LLVM::BitcastOp::create(rewriter, loc, f32Ty, oldI32);

    // Create loop and exit blocks
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *loopBlock = rewriter.createBlock(afterBlock);

    // Branch from current block to loop
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{oldF32, oldI32}, loopBlock);

    // Loop block: phi for old_f32, old_i32
    loopBlock->addArgument(f32Ty, loc);
    loopBlock->addArgument(i32Ty, loc);
    Value phiOldF32 = loopBlock->getArgument(0);
    Value phiOldI32 = loopBlock->getArgument(1);

    rewriter.setInsertionPointToStart(loopBlock);

    // Compute new value
    Value newF32;
    if (rmwOp == RMWOp::MAX)
      newF32 = LLVM::MaximumOp::create(rewriter, loc, phiOldF32, val);
    else
      newF32 = LLVM::MinimumOp::create(rewriter, loc, phiOldF32, val);

    Value newI32 = LLVM::BitcastOp::create(rewriter, loc, i32Ty, newF32);

    // Store expected (old) into alloca
    LLVM::StoreOp::create(rewriter, loc, phiOldI32, expectedAlloca);

    // CAS: cmpxchg(ptr, &expected, desired, succ_order=0, fail_order=0,
    // scope=2, vol=true)
    Value order0 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope2 = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value volT = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    Value oldRet =
        LLVM::CallOp::create(rewriter, loc, cmpxchgFn,
                             ValueRange{ptr, expectedAlloca, newI32, order0,
                                        order0, scope2, volT})
            .getResult();

    // Check success: old_ret == old_i32 means no other thread changed it
    Value success = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                         oldRet, phiOldI32);

    // On failure, the expected alloca now contains the current value
    Value failedOldF32 = LLVM::BitcastOp::create(rewriter, loc, f32Ty, oldRet);

    // Branch: success → afterBlock, failure → loopBlock with new old values
    LLVM::CondBrOp::create(rewriter, loc, success, afterBlock, ValueRange{},
                           loopBlock, ValueRange{failedOldF32, oldRet});

    // After block: result is the last successful old value (bitcast old_ret to
    // f32) But we need the result in afterBlock. Add a block arg.
    afterBlock->addArgument(f32Ty, loc);
    // Fix: afterBlock needs args from both paths. Actually we always arrive
    // from the success path of the CondBr above. Let me restructure.

    // Actually, CondBrOp success path goes to afterBlock — we need to pass the
    // result. Let me redo: erase the CondBr and rebuild with the right args.
    rewriter.eraseOp(success.getDefiningOp()->getBlock()->getTerminator());
    LLVM::CondBrOp::create(rewriter, loc, success, afterBlock,
                           ValueRange{phiOldF32}, loopBlock,
                           ValueRange{failedOldF32, oldRet});

    rewriter.setInsertionPointToStart(afterBlock);
    return afterBlock->getArgument(0);
  }

  // Emit CAS loop for f16/bf16 atomic add.
  // Strategy: bitcast ptr to i32*, load i32, extract the target half, compute,
  // pack back, cmpxchg i32.
  // Since Triton scalar atomics always target a single element, and the pointer
  // is already to the specific f16/bf16 element, we need to:
  //   1. Align ptr down to i32 boundary
  //   2. Determine which half (low/high) within the i32
  //   3. CAS loop on the i32
  // But actually, Triton's atomic_rmw on f16 gives us a ptr to a single f16.
  // We need to widen to i32 for the CAS. The element could be at an odd offset.
  //
  // Simpler approach: just use i16 CAS if Metal supports it.
  // Metal does NOT have i16 cmpxchg. So we must use i32.
  //
  // For the i32 widening approach:
  //   - ptr_i32 = ptr & ~3  (align down)
  //   - byte_offset = ptr & 3  → 0 or 2
  //   - shift = byte_offset * 8  → 0 or 16
  //   - mask = 0xFFFF << shift
  Value emitF16BF16CASLoop(ConversionPatternRewriter &rewriter, Location loc,
                           ModuleOp mod, Value ptr, Value val, Type elemTy,
                           RMWOp rmwOp) const {
    auto *ctx = rewriter.getContext();
    auto i16Ty = IntegerType::get(ctx, 16);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0);

    auto cmpxchgFn =
        declareAIR(rewriter, mod, "air.atomic.global.cmpxchg.weak.i32", i32Ty,
                   {ptrTy, ptrTy0, i32Ty, i32Ty, i32Ty, i32Ty, i1Ty});

    // Alloca for expected
    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    Value expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, i32Ty,
                                                  one, /*alignment=*/4);

    // Compute aligned i32 pointer and shift amount
    // ptr_as_int = ptrtoint ptr
    Value ptrInt = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptr);
    // byte_offset = ptr_as_int & 3
    Value three64 = arith::ConstantIntOp::create(rewriter, loc, 3, 64);
    Value byteOff64 = LLVM::AndOp::create(rewriter, loc, ptrInt, three64);
    // aligned_ptr_int = ptr_as_int & ~3
    Value notThree64 = arith::ConstantIntOp::create(rewriter, loc, ~3LL, 64);
    Value alignedInt = LLVM::AndOp::create(rewriter, loc, ptrInt, notThree64);
    Value alignedPtr =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, alignedInt);
    // shift = byte_offset * 8 (in i32)
    Value byteOff32 = LLVM::TruncOp::create(rewriter, loc, i32Ty, byteOff64);
    Value eight = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
    Value shift = LLVM::MulOp::create(rewriter, loc, byteOff32, eight);
    // mask = 0xFFFF << shift
    Value mask16 = arith::ConstantIntOp::create(rewriter, loc, 0xFFFF, 32);
    Value mask = LLVM::ShlOp::create(rewriter, loc, mask16, shift);
    Value notMask = LLVM::XOrOp::create(
        rewriter, loc, mask,
        arith::ConstantIntOp::create(rewriter, loc, -1, 32));

    // Convert val to f32 for computation, then back
    // Actually: val is already f16 or bf16. We do the add in f32 for
    // simplicity.
    Value valF32;
    if (elemTy.isF16())
      valF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, val);
    else // bf16
      valF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, val);

    // Initial load
    Value oldI32 = LLVM::LoadOp::create(rewriter, loc, i32Ty, alignedPtr);

    // Create loop and exit blocks
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *loopBlock = rewriter.createBlock(afterBlock);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{oldI32}, loopBlock);

    loopBlock->addArgument(i32Ty, loc);
    Value phiOldI32 = loopBlock->getArgument(0);

    rewriter.setInsertionPointToStart(loopBlock);

    // Extract the target i16 from the i32 word
    Value shifted = LLVM::LShrOp::create(rewriter, loc, phiOldI32, shift);
    Value oldI16 = LLVM::TruncOp::create(rewriter, loc, i16Ty, shifted);

    // Convert old i16 to f32
    Value oldF32;
    if (elemTy.isF16()) {
      Value oldF16 =
          LLVM::BitcastOp::create(rewriter, loc, Float16Type::get(ctx), oldI16);
      oldF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, oldF16);
    } else {
      Value oldBF16 = LLVM::BitcastOp::create(rewriter, loc,
                                              BFloat16Type::get(ctx), oldI16);
      oldF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, oldBF16);
    }

    // Compute: add in f32
    Value newF32 = arith::AddFOp::create(rewriter, loc, oldF32, valF32);

    // Convert back to i16
    Value newI16;
    if (elemTy.isF16()) {
      Value newF16 =
          arith::TruncFOp::create(rewriter, loc, Float16Type::get(ctx), newF32);
      newI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, newF16);
    } else {
      Value newBF16 = arith::TruncFOp::create(rewriter, loc,
                                              BFloat16Type::get(ctx), newF32);
      newI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, newBF16);
    }

    // Pack back into i32: (old & ~mask) | (new_i16_zext << shift)
    Value newI32Ext = LLVM::ZExtOp::create(rewriter, loc, i32Ty, newI16);
    Value newShifted = LLVM::ShlOp::create(rewriter, loc, newI32Ext, shift);
    Value cleared = LLVM::AndOp::create(rewriter, loc, phiOldI32, notMask);
    Value newI32 = LLVM::OrOp::create(rewriter, loc, cleared, newShifted);

    // Store expected, call cmpxchg
    LLVM::StoreOp::create(rewriter, loc, phiOldI32, expectedAlloca);
    Value order0 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope2 = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value volT = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    Value oldRet =
        LLVM::CallOp::create(rewriter, loc, cmpxchgFn,
                             ValueRange{alignedPtr, expectedAlloca, newI32,
                                        order0, order0, scope2, volT})
            .getResult();

    Value success = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                         oldRet, phiOldI32);

    LLVM::CondBrOp::create(rewriter, loc, success, afterBlock, ValueRange{},
                           loopBlock, ValueRange{oldRet});

    // After block: return the old element value (before our update)
    rewriter.setInsertionPointToStart(afterBlock);

    // Extract old element from the last successful i32
    // We need the pre-update value. The phiOldI32 at success is the value
    // that matched. Extract the element from it.
    // Actually, we need to pass the extracted old value out. Let me add a block
    // arg.

    // Reconstruct: on success, phiOldI32 was the matched expected.
    // The element we care about is oldI16 (extracted above). But that's in the
    // loop block. Simpler: add afterBlock arg with the old f16/bf16 value.

    // Redo: erase terminator and rebuild
    // The loop block terminator is the CondBrOp we just created.
    // We need to pass oldI16 to afterBlock on success.
    afterBlock->addArgument(elemTy, loc);

    auto *term = loopBlock->getTerminator();
    rewriter.setInsertionPoint(term);

    // Convert oldI16 to the element type
    Value oldElem;
    if (elemTy.isF16())
      oldElem =
          LLVM::BitcastOp::create(rewriter, loc, Float16Type::get(ctx), oldI16);
    else
      oldElem = LLVM::BitcastOp::create(rewriter, loc, BFloat16Type::get(ctx),
                                        oldI16);

    rewriter.eraseOp(term);
    LLVM::CondBrOp::create(rewriter, loc, success, afterBlock,
                           ValueRange{oldElem}, loopBlock, ValueRange{oldRet});

    rewriter.setInsertionPointToStart(afterBlock);
    return afterBlock->getArgument(0);
  }

  // Emit a single scalar atomic: either direct AIR intrinsic or CAS loop.
  // Returns the old value.
  Value emitOneAtomic(ConversionPatternRewriter &rewriter, Location loc,
                      ModuleOp mod, Value ptr, Value val, Value mask,
                      Type valueElemTy, RMWOp rmwOp, const std::string &airName,
                      bool needsCAS) const {
    auto *ctx = rewriter.getContext();

    // CAS loop path
    if (needsCAS) {
      auto emitCAS = [&]() -> Value {
        if (valueElemTy.isF32())
          return emitF32CASLoop(rewriter, loc, mod, ptr, val, rmwOp);
        else
          return emitF16BF16CASLoop(rewriter, loc, mod, ptr, val, valueElemTy,
                                    rmwOp);
      };

      if (mask) {
        // Wrap CAS in conditional to skip non-canonical threads
        auto *currentBlock = rewriter.getInsertionBlock();
        auto *afterBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        auto *casBlock = rewriter.createBlock(afterBlock);

        afterBlock->addArgument(valueElemTy, loc);

        rewriter.setInsertionPointToEnd(currentBlock);
        Value zeroVal = LLVM::ConstantOp::create(
            rewriter, loc, valueElemTy, rewriter.getZeroAttr(valueElemTy));
        LLVM::CondBrOp::create(rewriter, loc, mask, casBlock, ValueRange{},
                               afterBlock, ValueRange{zeroVal});

        rewriter.setInsertionPointToStart(casBlock);
        Value casResult = emitCAS();
        LLVM::BrOp::create(rewriter, loc, ValueRange{casResult}, afterBlock);

        rewriter.setInsertionPointToStart(afterBlock);
        return afterBlock->getArgument(0);
      }
      return emitCAS();
    }

    // Direct AIR intrinsic path
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto fnTy = LLVMFunctionType::get(
        valueElemTy, {ptrTy, valueElemTy, i32Ty, i32Ty, i1Ty}, false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (!mod.lookupSymbol<LLVMFuncOp>(airName))
        LLVMFuncOp::create(rewriter, mod.getLoc(), airName, fnTy,
                           Linkage::External);
    }
    auto atomicFn = mod.lookupSymbol<LLVMFuncOp>(airName);

    Value order = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value vol = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    if (mask) {
      auto *currentBlock = rewriter.getInsertionBlock();
      auto *afterBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      auto *atomicBlock = rewriter.createBlock(afterBlock);

      afterBlock->addArgument(valueElemTy, loc);

      rewriter.setInsertionPointToEnd(currentBlock);
      Value zeroVal = LLVM::ConstantOp::create(
          rewriter, loc, valueElemTy, rewriter.getZeroAttr(valueElemTy));
      LLVM::CondBrOp::create(rewriter, loc, mask, atomicBlock, ValueRange{},
                             afterBlock, ValueRange{zeroVal});

      rewriter.setInsertionPointToStart(atomicBlock);
      Value atomicResult =
          LLVM::CallOp::create(rewriter, loc, atomicFn,
                               ValueRange{ptr, val, order, scope, vol})
              .getResult();
      LLVM::BrOp::create(rewriter, loc, ValueRange{atomicResult}, afterBlock);

      rewriter.setInsertionPointToStart(afterBlock);
      return afterBlock->getArgument(0);
    } else {
      return LLVM::CallOp::create(rewriter, loc, atomicFn,
                                  ValueRange{ptr, val, order, scope, vol})
          .getResult();
    }
  }

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto rmwOp = op.getAtomicRmwOp();
    auto tensorTy = dyn_cast<RankedTensorType>(op.getType());

    // Determine element type
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : getTypeConverter()->convertType(op.getType());

    // Determine AIR intrinsic name or CAS loop
    std::string airName;
    bool needsCAS = false;
    if (valueElemTy.isF32()) {
      switch (rmwOp) {
      case RMWOp::FADD:
        airName = "air.atomic.global.add.f32";
        break;
      case RMWOp::XCHG:
        airName = "air.atomic.global.xchg.f32";
        break;
      default:
        return failure();
      }
    } else if (valueElemTy.isF16() || valueElemTy.isBF16()) {
      switch (rmwOp) {
      case RMWOp::FADD:
        needsCAS = true;
        break;
      default:
        return failure();
      }
    } else if (valueElemTy.isInteger(32)) {
      switch (rmwOp) {
      case RMWOp::ADD:
        airName = "air.atomic.global.add.s.i32";
        break;
      case RMWOp::MAX:
        airName = "air.atomic.global.max.s.i32";
        break;
      case RMWOp::MIN:
        airName = "air.atomic.global.min.s.i32";
        break;
      case RMWOp::UMAX:
        airName = "air.atomic.global.max.u.i32";
        break;
      case RMWOp::UMIN:
        airName = "air.atomic.global.min.u.i32";
        break;
      case RMWOp::AND:
        airName = "air.atomic.global.and.s.i32";
        break;
      case RMWOp::OR:
        airName = "air.atomic.global.or.s.i32";
        break;
      case RMWOp::XOR:
        airName = "air.atomic.global.xor.s.i32";
        break;
      case RMWOp::XCHG:
        airName = "air.atomic.global.xchg.i32";
        break;
      default:
        return failure();
      }
    } else {
      return failure();
    }

    if (!tensorTy) {
      // Scalar atomic: only thread 0 executes, broadcast result via TG.
      // Without this, all threads execute the atomic independently,
      // which corrupts spin-lock patterns (e.g. all threads doing
      // xchg(Lock, 0) allows another group to acquire between threads).
      auto i32Ty = IntegerType::get(ctx, 32);

      // Get thread_position_in_threadgroup[0]
      auto arrI32x3Ty = LLVMArrayType::get(i32Ty, 3);
      auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
          LLVMFuncOp::create(rewriter, mod.getLoc(),
                             "air.thread_position_in_threadgroup", tidFnTy,
                             Linkage::External);
      }
      auto tidFn =
          mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");

      Value tidStruct =
          LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
      Value tid0 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                                ArrayRef<int64_t>{0});
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      Value isThread0 = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, tid0, zero);

      // Combine thread-0 predicate with the op's own mask (e.g. sign-based
      // masks from float atomic_max/min decomposition).
      Value combinedMask = isThread0;
      if (llMask) {
        combinedMask = LLVM::AndOp::create(rewriter, loc, isThread0, llMask);
      }

      // Thread 0 (with mask) executes the atomic; others get a default value
      Value result =
          emitOneAtomic(rewriter, loc, mod, llPtr, llVal, combinedMask,
                        valueElemTy, rmwOp, airName, needsCAS);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Tensor atomic: unpack → per-element atomic → pack
    // Compute redundant thread predicate to mask out threads that
    // don't own unique elements (e.g. 128 threads, 4 elements → 124 idle)
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitAppleRedundantThreadPredicate(freeVarMasks, rewriter, loc, mod);
    uint32_t regMask = freeVarMasks[StringAttr::get(ctx, "reg")];

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    SmallVector<Value> resultVals(ptrElements.size());
    for (size_t i = 0; i < ptrElements.size(); ++i) {
      // Skip redundant register elements — reuse canonical result
      if (!isCanonicalIndex(i, regMask)) {
        resultVals[i] = resultVals[i & ~regMask];
        continue;
      }

      // Combine thread predicate with per-element mask
      Value mask = llMask ? maybeAnd(rewriter, loc, threadPred, maskElements[i])
                          : threadPred;
      Value res =
          emitOneAtomic(rewriter, loc, mod, ptrElements[i], valElements[i],
                        mask, valueElemTy, rmwOp, airName, needsCAS);
      resultVals[i] = res;
    }

    Value packed = packLLElements(loc, getTypeConverter(), resultVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, packed);
    return success();
  }
};

// Lower triton::AtomicCASOp → air.atomic.global.cmpxchg.weak.{i32,i64}
//
// Metal CAS signature:
//   i32 @air.atomic.global.cmpxchg.weak.i32(
//       ptr addrspace(1) ptr, ptr addrspace(0) expected,
//       i32 desired, i32 succ_order, i32 fail_order, i32 scope, i1 volatile)
// expected is passed by pointer and updated on failure.
// Returns old value.
struct AtomicCASOpAppleConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Emit f16/bf16 CAS via i32 CAS on aligned word.
  // Strategy: align ptr to i32 boundary, read i32 word, replace the target
  // half with cmp/val, do i32 CAS. No loop needed — CAS semantics guarantee
  // atomicity. If the CAS fails (other half changed), return the old f16 value.
  Value emitF16BF16CAS(ConversionPatternRewriter &rewriter, Location loc,
                       ModuleOp mod, Value ptr, Value cmp, Value val,
                       Type elemTy) const {
    auto *ctx = rewriter.getContext();
    auto i16Ty = IntegerType::get(ctx, 16);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0);

    // Declare i32 CAS
    auto airName = StringRef("air.atomic.global.cmpxchg.weak.i32");
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (!mod.lookupSymbol<LLVMFuncOp>(airName)) {
        auto fnTy = LLVMFunctionType::get(
            i32Ty, {ptrTy, ptrTy0, i32Ty, i32Ty, i32Ty, i32Ty, i1Ty}, false);
        LLVMFuncOp::create(rewriter, mod.getLoc(), airName, fnTy,
                           Linkage::External);
      }
    }
    auto casFn = mod.lookupSymbol<LLVMFuncOp>(airName);

    // Align pointer to i32 boundary
    Value ptrInt = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptr);
    Value one64 = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    Value offset = LLVM::AndOp::create(
        rewriter, loc, ptrInt,
        one64); // 0 or 1 (byte offset within i16 pair → 0 or 2 bytes)
    // Actually: ptr points to f16 (2 bytes). Aligned i32 = ptr & ~3.
    Value three64 = arith::ConstantIntOp::create(rewriter, loc, 3, 64);
    Value notThree64 = arith::ConstantIntOp::create(rewriter, loc, ~3LL, 64);
    Value alignedInt = LLVM::AndOp::create(rewriter, loc, ptrInt, notThree64);
    Value alignedPtr =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, alignedInt);

    // Byte offset within i32 word (0 or 2)
    Value byteOffset = LLVM::AndOp::create(rewriter, loc, ptrInt, three64);
    Value byteOffset32 =
        LLVM::TruncOp::create(rewriter, loc, i32Ty, byteOffset);
    // Shift in bits
    Value shift = LLVM::ShlOp::create(
        rewriter, loc, byteOffset32,
        arith::ConstantIntOp::create(rewriter, loc, 3, 32)); // bytes → bits

    // Mask for the target half: 0xFFFF shifted to position
    Value mask16 = arith::ConstantIntOp::create(rewriter, loc, 0xFFFF, 32);
    Value mask = LLVM::ShlOp::create(rewriter, loc, mask16, shift);
    Value notMask = LLVM::XOrOp::create(
        rewriter, loc, mask,
        arith::ConstantIntOp::create(rewriter, loc, -1, 32));

    // Truncate f32 cmp/val to f16/bf16 if needed (Triton CAS passes f32 cmp/val
    // for f16 ptrs)
    Value cmpElem = cmp, valElem = val;
    if (cmp.getType().isF32()) {
      cmpElem = arith::TruncFOp::create(rewriter, loc, elemTy, cmp);
      valElem = arith::TruncFOp::create(rewriter, loc, elemTy, val);
    }
    // Bitcast to i16 then zext to i32
    Value cmpI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, cmpElem);
    Value valI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, valElem);
    Value cmpI32 = LLVM::ZExtOp::create(rewriter, loc, i32Ty, cmpI16);
    Value valI32 = LLVM::ZExtOp::create(rewriter, loc, i32Ty, valI16);

    // Shift cmp/val to position within i32 word
    Value cmpShifted = LLVM::ShlOp::create(rewriter, loc, cmpI32, shift);
    Value valShifted = LLVM::ShlOp::create(rewriter, loc, valI32, shift);

    // Read current i32 word (non-atomic, as initial guess)
    Value curI32 = LLVM::LoadOp::create(rewriter, loc, i32Ty, alignedPtr);

    // Build expected i32 = (curI32 & ~mask) | cmpShifted
    Value otherBits = LLVM::AndOp::create(rewriter, loc, curI32, notMask);
    Value expectedI32 =
        LLVM::OrOp::create(rewriter, loc, otherBits, cmpShifted);
    // Build desired i32 = (curI32 & ~mask) | valShifted
    Value desiredI32 = LLVM::OrOp::create(rewriter, loc, otherBits, valShifted);

    // Alloca for expected — must be in entry block
    Value expectedAlloca;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto *funcOp = rewriter.getInsertionBlock()->getParent()->getParentOp();
      auto &entryBlock = cast<LLVM::LLVMFuncOp>(funcOp).getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      Value oneI64 = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, i32Ty,
                                              oneI64, /*alignment=*/4);
    }

    LLVM::StoreOp::create(rewriter, loc, expectedI32, expectedAlloca);

    Value order = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value vol = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    Value oldI32 =
        LLVM::CallOp::create(rewriter, loc, casFn,
                             ValueRange{alignedPtr, expectedAlloca, desiredI32,
                                        order, order, scope, vol})
            .getResult();

    // Extract old f16/bf16 from the returned i32 word
    Value oldShifted = LLVM::LShrOp::create(rewriter, loc, oldI32, shift);
    Value oldI16 = LLVM::TruncOp::create(rewriter, loc, i16Ty, oldShifted);
    return LLVM::BitcastOp::create(rewriter, loc, elemTy, oldI16);
  }

  // Emit a single scalar CAS operation. Returns the old value.
  Value emitOneCAS(ConversionPatternRewriter &rewriter, Location loc,
                   ModuleOp mod, Value ptr, Value cmp, Value val,
                   Type valueTy) const {
    auto *ctx = rewriter.getContext();

    // f16/bf16 CAS via i32 word CAS
    if (valueTy.isF16() || valueTy.isBF16())
      return emitF16BF16CAS(rewriter, loc, mod, ptr, cmp, val, valueTy);

    // Only 32-bit CAS reaches here — i64/f64 are rejected at the pattern entry
    // (no 64-bit atomics on Apple GPU). f16/bf16 take the emitF16BF16CAS path
    // above.
    std::string airName;
    Type casTy;
    if (valueTy.isInteger(32) || valueTy.isF32()) {
      airName = "air.atomic.global.cmpxchg.weak.i32";
      casTy = IntegerType::get(ctx, 32);
    } else {
      llvm_unreachable("unsupported CAS type");
    }

    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (!mod.lookupSymbol<LLVMFuncOp>(airName)) {
        auto fnTy = LLVMFunctionType::get(
            casTy, {ptrTy, ptrTy0, casTy, i32Ty, i32Ty, i32Ty, i1Ty}, false);
        LLVMFuncOp::create(rewriter, mod.getLoc(), airName, fnTy,
                           Linkage::External);
      }
    }
    auto casFn = mod.lookupSymbol<LLVMFuncOp>(airName);

    Value cmpI = cmp, valI = val;
    bool needBitcast = (valueTy != casTy);
    if (needBitcast) {
      cmpI = LLVM::BitcastOp::create(rewriter, loc, casTy, cmp);
      valI = LLVM::BitcastOp::create(rewriter, loc, casTy, val);
    }

    // Alloca for expected — entry block
    Value one, expectedAlloca;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto *funcOp = rewriter.getInsertionBlock()->getParent()->getParentOp();
      auto &entryBlock = cast<LLVM::LLVMFuncOp>(funcOp).getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, casTy, one,
                                              /*alignment=*/4);
    }

    LLVM::StoreOp::create(rewriter, loc, cmpI, expectedAlloca);

    Value order = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value vol = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    Value oldI = LLVM::CallOp::create(rewriter, loc, casFn,
                                      ValueRange{ptr, expectedAlloca, valI,
                                                 order, order, scope, vol})
                     .getResult();

    if (needBitcast)
      return LLVM::BitcastOp::create(rewriter, loc, valueTy, oldI);
    return oldI;
  }

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    auto tensorTy = dyn_cast<RankedTensorType>(op.getType());
    Type valueTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : getTypeConverter()->convertType(op.getType());

    // Check supported types. Apple GPUs have NO 64-bit atomics — there is no
    // `air.atomic.global.cmpxchg.weak.i64` intrinsic, and emitting one crashes
    // the Metal compiler service (XPC_ERROR_CONNECTION_INTERRUPTED). Fail the
    // lowering cleanly (the op stays illegal -> compile error) rather than
    // producing AIR that brings down the GPU compiler.
    if (!(valueTy.isInteger(32) || valueTy.isF32() || valueTy.isF16() ||
          valueTy.isBF16()))
      return rewriter.notifyMatchFailure(
          op, "Apple GPU has no 64-bit atomics (i64/f64 CAS unsupported)");

    if (!tensorTy) {
      // Scalar CAS: only thread 0 executes, broadcast result to all threads.
      // Without this, a spin-lock pattern (while CAS == 1) deadlocks:
      // all threads spin independently but only one succeeds, and subsequent
      // barriers can never be reached by the blocked threads.
      auto *ctx = rewriter.getContext();
      auto i32Ty = IntegerType::get(ctx, 32);

      // Get thread_position_in_threadgroup[0] to identify thread 0
      auto arrI32x3Ty = LLVMArrayType::get(i32Ty, 3);
      auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
          LLVMFuncOp::create(rewriter, mod.getLoc(),
                             "air.thread_position_in_threadgroup", tidFnTy,
                             Linkage::External);
      }
      auto tidFn =
          mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");

      // Declare barrier
      auto voidTy = LLVMVoidType::get(ctx);
      auto barrFnTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        if (!mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier"))
          LLVMFuncOp::create(rewriter, mod.getLoc(), "air.wg.barrier", barrFnTy,
                             Linkage::External);
      }
      auto barrFn = mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier");

      // Create TG global to broadcast the CAS result
      Type tgElemTy = valueTy.isF32()   ? (Type)i32Ty
                      : valueTy.isF64() ? (Type)IntegerType::get(ctx, 64)
                                        : valueTy;
      std::string tgName = "__tg_cas_bcast";
      auto tgPtrTy = LLVMPointerType::get(ctx, 3);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        if (!mod.lookupSymbol<LLVM::GlobalOp>(tgName)) {
          auto arrTy = LLVMArrayType::get(tgElemTy, 1);
          LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy, false,
                                 Linkage::Internal, tgName, Attribute(), 4, 3u);
        }
      }
      auto tgGlobal = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
      Value tgPtr =
          LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgGlobal.getName());

      // Get tid and check if tid == 0
      Value tidStruct =
          LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
      Value tid0 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                                ArrayRef<int64_t>{0});
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      Value isThread0 = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, tid0, zero);

      // Create blocks: thread0 does CAS and stores to TG; others skip to
      // barrier
      auto *currentBlock = rewriter.getInsertionBlock();
      auto *afterBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      auto *casBlock = rewriter.createBlock(afterBlock);
      auto *mergeBlock = rewriter.createBlock(afterBlock);

      // Branch: thread 0 → casBlock, others → mergeBlock
      rewriter.setInsertionPointToEnd(currentBlock);
      LLVM::CondBrOp::create(rewriter, loc, isThread0, casBlock, mergeBlock);

      // casBlock: execute CAS, store result to TG
      rewriter.setInsertionPointToStart(casBlock);
      Value casResult =
          emitOneCAS(rewriter, loc, mod, llPtr, llCmp, llVal, valueTy);
      Value resultToStore = casResult;
      if (valueTy.isF32())
        resultToStore =
            LLVM::BitcastOp::create(rewriter, loc, i32Ty, casResult);
      else if (valueTy.isF64())
        resultToStore = LLVM::BitcastOp::create(
            rewriter, loc, IntegerType::get(ctx, 64), casResult);
      LLVM::StoreOp::create(rewriter, loc, resultToStore, tgPtr);
      LLVM::BrOp::create(rewriter, loc, mergeBlock);

      // mergeBlock: barrier then load from TG
      rewriter.setInsertionPointToStart(mergeBlock);
      // device memory fence (flag=1) to ensure the CAS side-effects are visible
      Value flagDev = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
      Value scope1 = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
      LLVM::CallOp::create(rewriter, loc, barrFn, ValueRange{flagDev, scope1});
      // TG memory fence (flag=2) to ensure the TG store is visible
      Value flagTG = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
      LLVM::CallOp::create(rewriter, loc, barrFn, ValueRange{flagTG, scope1});
      Value loaded =
          LLVM::LoadOp::create(rewriter, loc, tgElemTy, tgPtr).getResult();
      if (valueTy.isF32())
        loaded = LLVM::BitcastOp::create(rewriter, loc, valueTy, loaded);
      else if (valueTy.isF64())
        loaded = LLVM::BitcastOp::create(rewriter, loc, valueTy, loaded);

      // Move remaining ops after the load
      rewriter.setInsertionPointAfter(loaded.getDefiningOp());
      // Splice afterBlock's contents into mergeBlock
      mergeBlock->getOperations().splice(mergeBlock->end(),
                                         afterBlock->getOperations());
      afterBlock->erase();

      rewriter.replaceOp(op, loaded);
      return success();
    }

    // Tensor CAS: unpack → per-element CAS → pack
    // Compute redundant thread predicate
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitAppleRedundantThreadPredicate(freeVarMasks, rewriter, loc, mod);
    uint32_t regMask =
        freeVarMasks[StringAttr::get(rewriter.getContext(), "reg")];

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    SmallVector<Value> resultVals(ptrElements.size());
    for (size_t i = 0; i < ptrElements.size(); ++i) {
      // Skip redundant register elements
      if (!isCanonicalIndex(i, regMask)) {
        resultVals[i] = resultVals[i & ~regMask];
        continue;
      }

      // For CAS, wrap with thread predicate: skip non-canonical threads
      if (threadPred) {
        auto *currentBlock = rewriter.getInsertionBlock();
        auto *afterBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        auto *casBlock = rewriter.createBlock(afterBlock);
        afterBlock->addArgument(valueTy, loc);

        rewriter.setInsertionPointToEnd(currentBlock);
        Value zeroVal = LLVM::ConstantOp::create(rewriter, loc, valueTy,
                                                 rewriter.getZeroAttr(valueTy));
        LLVM::CondBrOp::create(rewriter, loc, threadPred, casBlock,
                               ValueRange{}, afterBlock, ValueRange{zeroVal});

        rewriter.setInsertionPointToStart(casBlock);
        Value casResult = emitOneCAS(rewriter, loc, mod, ptrElements[i],
                                     cmpElements[i], valElements[i], valueTy);
        LLVM::BrOp::create(rewriter, loc, ValueRange{casResult}, afterBlock);

        rewriter.setInsertionPointToStart(afterBlock);
        resultVals[i] = afterBlock->getArgument(0);
      } else {
        resultVals[i] = emitOneCAS(rewriter, loc, mod, ptrElements[i],
                                   cmpElements[i], valElements[i], valueTy);
      }
    }

    Value packed = packLLElements(loc, getTypeConverter(), resultVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, packed);
    return success();
  }
};

// Safe tt.store lowering: use conditional branch instead of read-modify-write.
//
// The LoadStoreToLLVM.cpp StoreOpConversion uses a read-modify-write pattern
// for masked stores: load(ptr); select(mask, val, loaded); store(ptr).
// This is broken when masked-out pointers alias with other threads' valid
// addresses (e.g., when M < RBLOCK and row strides cause overlap), creating
// race conditions and data corruption.
//
// This pattern uses a conditional branch: if (mask) store(val, ptr), which
// is safe regardless of the pointer value when the mask is false.
// Interior-tile fast path for masked stores.
//
// A GEMM C-store carries a rectangular boundary mask
// `(om[:,None] < M) & (on[None,:] < N)` so edge tiles don't write past the
// matrix. For PROVABLY-FULL (interior) tiles every per-element predicate is
// all-true, yet the lowering still emits a 64-way per-element predicated store
// chain that burns ALU. This derives a single simdgroup-uniform guard
// `tile_fully_in_bounds` from the mask so the caller can branch to an unmasked
// store for interior tiles while keeping the predicated path for genuine edges.
//
// The guard is `(rowFirst + BM) <= rowBound && (colFirst + BN) <= colBound`
// (AND every uniform i1 leaf), where rowFirst/colFirst are the smallest tile
// indices and BM/BN the tile extents (the non-1 cmp result dims). It is
// CONSERVATIVE: any leaf we cannot recognize as a uniform splat or a
// `cmp slt(idx, splat(bound))` row/col bound makes this return null, so the
// store keeps the exact existing masked path. Returns null when the mask is not
// a recognizable full rectangular boundary mask.
static Value computeRectStoreFullGuard(triton::StoreOp op,
                                       ConversionPatternRewriter &rewriter,
                                       Location loc) {
  Value mask = op.getMask();
  if (!mask)
    return nullptr;

  Value rowBound, rowFirst, colBound, colFirst;
  int64_t rowExtent = 0, colExtent = 0;
  int64_t rowFirstConst = 0, colFirstConst = 0;
  bool hasRow = false, hasCol = false;
  SmallVector<Value, 2> uniforms;

  // Walk the AND-tree of the mask; every leaf must be either a uniform i1 splat
  // or a single-axis `cmp slt(idx, splat(bound))`.
  SmallVector<Value, 8> work{mask};
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    auto *def = cur.getDefiningOp();
    if (!def)
      return nullptr;
    if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
      work.push_back(andOp.getLhs());
      work.push_back(andOp.getRhs());
      continue;
    }
    if (auto bc = dyn_cast<triton::BroadcastOp>(def)) {
      work.push_back(bc.getSrc());
      continue;
    }
    if (auto splat = dyn_cast<triton::SplatOp>(def)) {
      Value s = splat.getSrc();
      if (!s.getType().isInteger(1))
        return nullptr;
      uniforms.push_back(s);
      continue;
    }
    auto cmp = dyn_cast<arith::CmpIOp>(def);
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
      return nullptr;
    auto resTy = dyn_cast<RankedTensorType>(cmp.getType());
    if (!resTy || resTy.getRank() != 2)
      return nullptr;
    bool boundsCols;
    if (resTy.getDimSize(0) == 1 && resTy.getDimSize(1) > 1)
      boundsCols = true;
    else if (resTy.getDimSize(1) == 1 && resTy.getDimSize(0) > 1)
      boundsCols = false;
    else
      return nullptr;
    int64_t extent = boundsCols ? resTy.getDimSize(1) : resTy.getDimSize(0);
    Value bound = extractFirstElemScalar(cmp.getRhs(), false);
    if (!bound)
      return nullptr;
    Value first = extractFirstElemScalar(cmp.getLhs(), false);
    int64_t firstConst = 0;
    if (!first && !extractFirstElemConst(cmp.getLhs(), firstConst, false))
      return nullptr;
    if (boundsCols) {
      if (hasCol)
        return nullptr;
      hasCol = true;
      colBound = bound;
      colFirst = first;
      colFirstConst = firstConst;
      colExtent = extent;
    } else {
      if (hasRow)
        return nullptr;
      hasRow = true;
      rowBound = bound;
      rowFirst = first;
      rowFirstConst = firstConst;
      rowExtent = extent;
    }
  }

  if (!hasRow && !hasCol)
    return nullptr;

  auto i64Ty = rewriter.getI64Type();
  auto toI64 = [&](Value v) -> Value {
    Value r = rewriter.getRemappedValue(v);
    if (!r)
      return nullptr;
    if (r.getType() != i64Ty)
      r = LLVM::SExtOp::create(rewriter, loc, i64Ty, r);
    return r;
  };

  // Per-axis guard: firstScalar + extent <= bound  (both sides as i64).
  auto axisGuard = [&](Value boundV, Value firstV, int64_t firstConst,
                       int64_t extent) -> Value {
    Value b = toI64(boundV);
    if (!b)
      return nullptr;
    Value f;
    if (firstV) {
      f = toI64(firstV);
      if (!f)
        return nullptr;
    } else {
      f = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                   rewriter.getI64IntegerAttr(firstConst));
    }
    Value ext = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                         rewriter.getI64IntegerAttr(extent));
    Value hi = LLVM::AddOp::create(rewriter, loc, i64Ty, f, ext);
    return LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sle, hi, b);
  };

  Value guard;
  if (hasRow) {
    guard = axisGuard(rowBound, rowFirst, rowFirstConst, rowExtent);
    if (!guard)
      return nullptr;
  }
  if (hasCol) {
    Value g = axisGuard(colBound, colFirst, colFirstConst, colExtent);
    if (!g)
      return nullptr;
    guard = guard ? LLVM::AndOp::create(rewriter, loc, guard, g) : g;
  }
  for (Value u : uniforms) {
    Value ru = rewriter.getRemappedValue(u);
    if (!ru)
      return nullptr;
    guard = LLVM::AndOp::create(rewriter, loc, guard, ru);
  }
  return guard;
}

struct SafeStoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static SmallVector<Value> unpackElems(Value v, OpBuilder &b, Location loc) {
    if (!v)
      return {};
    if (auto sTy = dyn_cast<LLVMStructType>(v.getType())) {
      SmallVector<Value> elems(sTy.getBody().size());
      for (size_t i = 0; i < elems.size(); ++i)
        elems[i] = ExtractValueOp::create(b, loc, sTy.getBody()[i], v,
                                          ArrayRef<int64_t>{(int64_t)i});
      return elems;
    }
    return {v};
  }

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value ptr = adaptor.getPtr();
    Value val = adaptor.getValue();

    auto ptrs = unpackElems(ptr, rewriter, loc);
    auto vals = unpackElems(val, rewriter, loc);

    if (ptrs.size() != vals.size())
      return failure();

    Value maskOperand = adaptor.getMask();
    auto masks = maskOperand ? unpackElems(maskOperand, rewriter, loc)
                             : SmallVector<Value>{};

    // When the stored tensor is replicated across lanes/warps (the threadgroup
    // has more threads than the tensor has elements — e.g. a 64-thread group
    // writing a 1-element reduction result), every redundant thread computes
    // the same destination pointer. Without predication they all race-store to
    // that address, and a thread holding a stale/zeroed replica can win, so the
    // result is corrupted (observed as ~all-but-one programs storing 0). Emit a
    // redundant-thread predicate so only the canonical owner of each element
    // stores, and skip redundant register-replicated copies entirely.
    Value threadPred;
    uint32_t regMask = 0;
    if (isa<RankedTensorType>(op.getPtr().getType())) {
      auto mod = op->getParentOfType<ModuleOp>();
      auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
      threadPred =
          emitAppleRedundantThreadPredicate(freeVarMasks, rewriter, loc, mod);
      regMask = freeVarMasks[StringAttr::get(rewriter.getContext(), "reg")];
    }

    // Emit the per-element store sequence. When dropElemMask is set the
    // boundary store mask is dropped (the tile is provably full); the
    // redundant-thread predicate still applies since it guards replicated
    // writers, not bounds.
    auto emitStores = [&](bool dropElemMask) {
      for (size_t i = 0; i < ptrs.size(); ++i) {
        if (!isCanonicalIndex(i, regMask))
          continue;

        Value mask = threadPred;
        if (!dropElemMask && !masks.empty() && masks[i])
          mask = maybeAnd(rewriter, loc, threadPred, masks[i]);

        if (mask) {
          auto *curBlock = rewriter.getInsertionBlock();
          auto curPoint = rewriter.getInsertionPoint();
          auto *endBlock = curBlock->splitBlock(curPoint);
          auto *thenBlock = rewriter.createBlock(endBlock);
          rewriter.setInsertionPointToEnd(curBlock);
          LLVM::CondBrOp::create(rewriter, loc, mask, thenBlock, endBlock);
          rewriter.setInsertionPointToEnd(thenBlock);
          LLVM::StoreOp::create(rewriter, loc, vals[i], ptrs[i]);
          LLVM::BrOp::create(rewriter, loc, endBlock);
          rewriter.setInsertionPointToStart(endBlock);
        } else {
          LLVM::StoreOp::create(rewriter, loc, vals[i], ptrs[i]);
        }
      }
    };

    // Interior-tile fast path: if the boundary mask is a full rectangular mask
    // and the tile is provably (at runtime, simdgroup-uniformly) fully in
    // bounds, branch to a maskless store and skip the per-element predication.
    Value fullGuard =
        !masks.empty() ? computeRectStoreFullGuard(op, rewriter, loc) : nullptr;
    if (fullGuard) {
      auto *curBlock = rewriter.getInsertionBlock();
      auto curPoint = rewriter.getInsertionPoint();
      auto *contBlock = curBlock->splitBlock(curPoint);
      auto *fullBlock = rewriter.createBlock(contBlock);
      auto *edgeBlock = rewriter.createBlock(contBlock);
      rewriter.setInsertionPointToEnd(curBlock);
      LLVM::CondBrOp::create(rewriter, loc, fullGuard, fullBlock, edgeBlock);

      rewriter.setInsertionPointToEnd(fullBlock);
      emitStores(/*dropElemMask=*/true);
      LLVM::BrOp::create(rewriter, loc, contBlock);

      rewriter.setInsertionPointToEnd(edgeBlock);
      emitStores(/*dropElemMask=*/false);
      LLVM::BrOp::create(rewriter, loc, contBlock);

      rewriter.setInsertionPointToStart(contBlock);
    } else {
      emitStores(/*dropElemMask=*/false);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// Safe tt.load lowering: use conditional branch for masked loads.
//
// Similar to SafeStoreOpConversion, the LoadStoreToLLVM.cpp LoadOpConversion
// unconditionally loads from the pointer (even when masked out), then selects
// the result. Loading from out-of-bounds pointers is undefined behavior on
// Metal. This pattern uses a conditional branch to avoid the invalid load.
struct SafeLoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static SmallVector<Value> unpackElems(Value v, OpBuilder &b, Location loc) {
    if (!v)
      return {};
    if (auto sTy = dyn_cast<LLVMStructType>(v.getType())) {
      SmallVector<Value> elems(sTy.getBody().size());
      for (size_t i = 0; i < elems.size(); ++i)
        elems[i] = ExtractValueOp::create(b, loc, sTy.getBody()[i], v,
                                          ArrayRef<int64_t>{(int64_t)i});
      return elems;
    }
    return {v};
  }

  static Value packElems(ArrayRef<Value> elems, OpBuilder &b, Location loc) {
    SmallVector<Type> tys;
    for (auto v : elems)
      tys.push_back(v.getType());
    auto sTy = LLVMStructType::getLiteral(b.getContext(), tys);
    Value result = UndefOp::create(b, loc, sTy);
    for (size_t i = 0; i < elems.size(); ++i)
      result = InsertValueOp::create(b, loc, sTy, result, elems[i],
                                     ArrayRef<int64_t>{(int64_t)i});
    return result;
  }

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value ptr = adaptor.getPtr();
    Type resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    // Scalar load: bare pointer
    if (!isa<LLVMStructType>(ptr.getType())) {
      Value maskOperand = adaptor.getMask();
      Value otherOperand = adaptor.getOther();
      if (maskOperand) {
        Value other = otherOperand
                          ? otherOperand
                          : LLVM::ZeroOp::create(rewriter, loc, resultTy);
        // Load unconditionally, select result.
        // For scalar loads the pointer is always valid.
        Value val = LLVM::LoadOp::create(rewriter, loc, resultTy, ptr);
        val = LLVM::SelectOp::create(rewriter, loc, maskOperand, val, other);
        rewriter.replaceOp(op, val);
      } else {
        Value val = LLVM::LoadOp::create(rewriter, loc, resultTy, ptr);
        rewriter.replaceOp(op, val);
      }
      return success();
    }

    // Tensor load: struct of pointers
    auto ptrs = unpackElems(ptr, rewriter, loc);
    auto sTy = dyn_cast<LLVMStructType>(resultTy);
    if (!sTy || sTy.getBody().size() != ptrs.size())
      return failure();

    Value maskOperand = adaptor.getMask();
    Value otherOperand = adaptor.getOther();
    auto masks = maskOperand ? unpackElems(maskOperand, rewriter, loc)
                             : SmallVector<Value>{};
    auto others = otherOperand ? unpackElems(otherOperand, rewriter, loc)
                               : SmallVector<Value>{};

    SmallVector<Value> loaded;
    for (size_t i = 0; i < ptrs.size(); ++i) {
      if (!masks.empty()) {
        Value other = others.empty() ? LLVM::ZeroOp::create(rewriter, loc,
                                                            sTy.getBody()[i])
                                     : others[i];
        // Conditional load via branch to avoid accessing invalid pointers
        // when the mask is false (e.g., rindex >= M with M < RBLOCK).
        auto *curBlock = rewriter.getInsertionBlock();
        auto curPoint = rewriter.getInsertionPoint();
        auto *endBlock = curBlock->splitBlock(curPoint);
        auto *thenBlock = rewriter.createBlock(endBlock);
        endBlock->addArgument(sTy.getBody()[i], loc);
        rewriter.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(rewriter, loc, masks[i], thenBlock, ValueRange{},
                               endBlock, ValueRange{other});
        rewriter.setInsertionPointToEnd(thenBlock);
        Value val =
            LLVM::LoadOp::create(rewriter, loc, sTy.getBody()[i], ptrs[i]);
        LLVM::BrOp::create(rewriter, loc, ValueRange{val}, endBlock);
        rewriter.setInsertionPointToStart(endBlock);
        loaded.push_back(endBlock->getArgument(0));
      } else {
        loaded.push_back(
            LLVM::LoadOp::create(rewriter, loc, sTy.getBody()[i], ptrs[i]));
      }
    }
    rewriter.replaceOp(op, packElems(loaded, rewriter, loc));
    return success();
  }
};

// Lower ttg::WarpIdOp → air.dispatch_thread_id[0] / threadsPerWarp.
struct WarpIdOpConversion
    : public mlir::ConvertOpToLLVMPattern<triton::gpu::WarpIdOp> {
  using mlir::ConvertOpToLLVMPattern<
      triton::gpu::WarpIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::WarpIdOp op,
                  triton::gpu::WarpIdOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto mod = op->getParentOfType<ModuleOp>();

    // Use air.thread_position_in_threadgroup (returns [3 x i32]) + extractvalue
    // 0. _add_air_metadata() rewrites this call+extractvalue to a function arg.
    auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
    auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
        LLVMFuncOp::create(rewriter, mod.getLoc(),
                           "air.thread_position_in_threadgroup", tidFnTy,
                           Linkage::External);
    }
    auto tidFn =
        mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");
    Value tidStruct =
        LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
    Value tid = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                             ArrayRef<int64_t>{0});

    // warpId = tid / threadsPerWarp (32 on Apple simdgroup)
    int tpw = ttg::lookupThreadsPerWarp(rewriter);
    Value warpSize = arith::ConstantIntOp::create(rewriter, loc, tpw, 32);
    Value warpId = arith::DivUIOp::create(rewriter, loc, tid, warpSize);
    rewriter.replaceOp(op, warpId);
    return success();
  }
};

// Lower triton::GetNumProgramsOp → call @air.threadgroups_per_grid() +
// extractvalue Returns the grid dimension (number of threadgroups) for the
// given axis.
struct GetNumProgramsOpAppleConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto arrTy = LLVM::LLVMArrayType::get(i32Ty, 3);
    auto fnTy = LLVMFunctionType::get(arrTy, {}, false);

    auto fnName = StringRef("air.threadgroups_per_grid");
    auto mod = op->getParentOfType<ModuleOp>();
    if (!mod.lookupSymbol<LLVMFuncOp>(fnName)) {
      OpBuilder b(mod.getBodyRegion());
      b.setInsertionPointToStart(mod.getBody());
      LLVMFuncOp::create(b, mod.getLoc(), fnName, fnTy, Linkage::External);
    }
    auto fn = mod.lookupSymbol<LLVMFuncOp>(fnName);

    Value gridStruct =
        LLVM::CallOp::create(rewriter, loc, fn, ValueRange{}).getResult();
    int axis = static_cast<int>(op.getAxis());
    Value result = LLVM::ExtractValueOp::create(
        rewriter, loc, i32Ty, gridStruct, ArrayRef<int64_t>{(int64_t)axis});
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lower triton::FuncOp → LLVM::LLVMFuncOp for Apple Metal kernels.
//
// Metal passes scalar kernel args (i32, i64, etc.) via setBytes — a pointer
// to constant address space (addrspace 2). The LLVM IR must reflect this:
// scalar args become `i32 addrspace(2)*` pointers, and we insert explicit
// loads at function entry. This matches what `xcrun metal` emits for
// `constant T&` parameters, and eliminates the Python regex workaround.
//
// Pointer args (addrspace 1 = device) are passed through unchanged.
struct AppleFuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = funcOp.getContext();
    auto loc = funcOp.getLoc();
    bool isKernel = triton::isKernel(funcOp);

    // Build new LLVM arg types.
    // Kernel: scalar i32/i64/etc → addrspace(2)* pointer (Metal constant
    // buffer). Device function: convert types directly, no addrspace wrapping.
    SmallVector<Type> newArgTypes;
    SmallVector<bool> isScalar;
    for (auto argTy : funcOp.getFunctionType().getInputs()) {
      Type converted = getTypeConverter()->convertType(argTy);
      if (!converted)
        return failure();
      if (isKernel && isa<IntegerType>(converted)) {
        auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addrspace=*/2);
        newArgTypes.push_back(ptrTy);
        isScalar.push_back(true);
      } else {
        newArgTypes.push_back(converted);
        isScalar.push_back(false);
      }
    }

    // Build return type: void for kernels, converted type for device functions.
    Type retTy = LLVM::LLVMVoidType::get(ctx);
    if (!isKernel) {
      auto results = funcOp.getFunctionType().getResults();
      if (results.size() == 1) {
        retTy = getTypeConverter()->convertType(results[0]);
        if (!retTy)
          return failure();
      } else if (results.size() > 1) {
        // Pack multiple return values into a struct
        SmallVector<Type> memberTypes;
        for (auto resTy : results) {
          Type converted = getTypeConverter()->convertType(resTy);
          if (!converted)
            return failure();
          memberTypes.push_back(converted);
        }
        retTy = LLVM::LLVMStructType::getLiteral(ctx, memberTypes);
      }
    }

    auto llvmFuncTy = LLVM::LLVMFunctionType::get(retTy, newArgTypes);
    auto newFuncOp = LLVM::LLVMFuncOp::create(
        rewriter, loc, funcOp.getName(), llvmFuncTy, LLVM::Linkage::External);
    // Mark the launchable entry: downstream consumers (entry-name detection,
    // non-kernel inlining/pruning, !air.kernel emission) must not infer
    // kernel identity from the call graph — the optimizer may inline a
    // noinline helper's only call site, leaving two uncalled functions.
    if (isKernel)
      newFuncOp.setPassthroughAttr(
          rewriter.getArrayAttr({rewriter.getStringAttr("air-kernel")}));

    // Move function body into new func
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Fix up block argument types and insert loads for scalar kernel args
    Block &entryBlock = newFuncOp.getBody().front();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&entryBlock);

    for (unsigned i = 0; i < newArgTypes.size(); ++i) {
      BlockArgument oldArg = entryBlock.getArgument(i);
      if (isScalar[i]) {
        oldArg.setType(newArgTypes[i]);
        auto origTy = getTypeConverter()->convertType(
            funcOp.getFunctionType().getInput(i));
        // Mark the scalar-arg load volatile so it is never eliminated. Once the
        // arg becomes an opaque addrspace(2) pointer, its original width is not
        // recoverable from the type; the scalar-buffer-packing pass recovers it
        // from this load's result type. If the load were dead (e.g. it feeds
        // only a bounds-check assert that later gets elided), the pass would
        // have to guess the width, mis-size the packed slot, and corrupt every
        // following scalar's byte offset. Keeping the load alive guarantees an
        // exact, signature-matching layout. A constant-buffer load is cheap and
        // any genuinely unused result is dropped after packing rewrites it.
        Value loaded = LLVM::LoadOp::create(rewriter, loc, origTy, oldArg,
                                            /*alignment=*/0,
                                            /*isVolatile=*/true);
        oldArg.replaceAllUsesExcept(loaded, loaded.getDefiningOp());
      } else {
        oldArg.setType(newArgTypes[i]);
      }
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct AppleReturnOpConversion
    : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    if (operands.size() <= 1) {
      // Void or single return — direct lowering
      LLVM::ReturnOp::create(rewriter, op.getLoc(), operands);
    } else {
      // Multi-return: pack values into a struct (matches the struct return
      // type created by AppleFuncOpConversion for multi-result functions).
      auto loc = op.getLoc();
      auto *ctx = rewriter.getContext();
      SmallVector<Type> memberTypes;
      for (auto v : operands)
        memberTypes.push_back(v.getType());
      auto structTy = LLVM::LLVMStructType::getLiteral(ctx, memberTypes);

      Value packed = LLVM::UndefOp::create(rewriter, loc, structTy);
      for (unsigned i = 0; i < operands.size(); ++i) {
        packed = LLVM::InsertValueOp::create(
            rewriter, loc, packed, operands[i],
            ArrayRef<int64_t>{static_cast<int64_t>(i)});
      }
      LLVM::ReturnOp::create(rewriter, loc, ValueRange{packed});
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower triton::PrintOp → no-op (Metal has no printf).
// Erase the op so it doesn't block legalization.
struct ApplePrintOpConversion : public ConvertOpToLLVMPattern<triton::PrintOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower triton::AssertOp → no-op (Metal has no device-side assert).
struct AppleAssertOpConversion
    : public ConvertOpToLLVMPattern<triton::AssertOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower triton::CallOp → LLVM::CallOp for Apple device function calls.
// Unlike CUDA, we don't append shared memory stack pointers.
struct AppleCallOpConversion : public ConvertOpToLLVMPattern<triton::CallOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = callOp.getLoc();
    auto promotedOperands = getTypeConverter()->promoteOperands(
        loc, callOp->getOperands(), adaptor.getOperands(), rewriter);

    // Build result type
    SmallVector<Type> resultTypes;
    for (auto resTy : callOp.getResultTypes()) {
      Type converted = getTypeConverter()->convertType(resTy);
      if (!converted)
        return failure();
      resultTypes.push_back(converted);
    }

    if (resultTypes.size() <= 1) {
      // Single or void return — direct lowering
      auto newCallOp = LLVM::CallOp::create(
          rewriter, loc,
          resultTypes.empty() ? TypeRange() : TypeRange(resultTypes),
          promotedOperands, callOp->getAttrs());
      newCallOp.getProperties().setOpBundleSizes(
          rewriter.getDenseI32ArrayAttr({}));
      newCallOp.getProperties().setOperandSegmentSizes(
          {static_cast<int>(promotedOperands.size()), 0});
      rewriter.replaceOp(callOp, newCallOp.getResults());
    } else {
      // Multi-return: call returns a struct, extract each field
      auto *ctx = rewriter.getContext();
      auto structTy = LLVM::LLVMStructType::getLiteral(ctx, resultTypes);
      auto newCallOp =
          LLVM::CallOp::create(rewriter, loc, TypeRange(structTy),
                               promotedOperands, callOp->getAttrs());
      newCallOp.getProperties().setOpBundleSizes(
          rewriter.getDenseI32ArrayAttr({}));
      newCallOp.getProperties().setOperandSegmentSizes(
          {static_cast<int>(promotedOperands.size()), 0});

      SmallVector<Value> extracted;
      Value structResult = newCallOp.getResult();
      for (unsigned i = 0; i < resultTypes.size(); ++i) {
        extracted.push_back(LLVM::ExtractValueOp::create(
            rewriter, loc, resultTypes[i], structResult,
            ArrayRef<int64_t>{static_cast<int64_t>(i)}));
      }
      rewriter.replaceOp(callOp, extracted);
    }
    return success();
  }
};

// Lower ExternElementwiseOp (libdevice calls) to LLVM intrinsics.
// Maps __nv_exp → llvm.exp.f32, __nv_sin → llvm.sin.f32, etc.
struct ExternElementwiseOpAppleConversion
    : public mlir::triton::gpu::ElementwiseOpConversionBase<
          triton::ExternElementwiseOp, ExternElementwiseOpAppleConversion> {
  using Base = mlir::triton::gpu::ElementwiseOpConversionBase<
      triton::ExternElementwiseOp, ExternElementwiseOpAppleConversion>;
  using Base::Base;

  SmallVector<Value>
  createDestOps(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, Type elemTy,
                gpu::MultipleOperandsRange operands, Location loc) const {

    StringRef symbol = op.getSymbol();

    // Map libdevice symbols to LLVM intrinsic names
    // __nv_exp → llvm.exp, __nv_sin → llvm.sin, etc.
    static const llvm::StringMap<StringRef> unaryMap = {
        {"__nv_exp", "llvm.exp"},
        {"__nv_exp2", "llvm.exp2"},
        {"__nv_log", "llvm.log"},
        {"__nv_log2", "llvm.log2"},
        {"__nv_log10", "llvm.log10"},
        {"__nv_sin", "llvm.sin"},
        {"__nv_cos", "llvm.cos"},
        {"__nv_sqrt", "llvm.sqrt"},
        {"__nv_rsqrt", "llvm.sqrt"}, // will invert
        {"__nv_fabs", "llvm.fabs"},
        {"__nv_fabsf", "llvm.fabs"},
        {"__nv_floor", "llvm.floor"},
        {"__nv_floorf", "llvm.floor"},
        {"__nv_ceil", "llvm.ceil"},
        {"__nv_ceilf", "llvm.ceil"},
        {"__nv_trunc", "llvm.trunc"},
        {"__nv_truncf", "llvm.trunc"},
        {"__nv_nearbyint", "llvm.nearbyint"},
        {"__nv_rint", "llvm.rint"},
        {"__nv_llrint", "llvm.lrint"},
        {"__nv_expm1", "llvm.exp"}, // approx: will subtract 1
    };
    static const llvm::StringMap<StringRef> binaryMap = {
        {"__nv_copysign", "llvm.copysign"},
        {"__nv_copysignf", "llvm.copysign"},
        {"__nv_fmax", "llvm.maxnum"},
        {"__nv_fmaxf", "llvm.maxnum"},
        {"__nv_fmin", "llvm.minnum"},
        {"__nv_fminf", "llvm.minnum"},
        {"__nv_pow", "llvm.pow"},
        {"__nv_powf", "llvm.pow"},
        {"__nv_atan2", ""}, // no direct intrinsic
        {"__nv_atan2f", ""},
        {"__nv_fmod", ""},
        {"__nv_fmodf", ""},
    };

    // Unary intrinsics
    auto uit = unaryMap.find(symbol);
    if (uit != unaryMap.end() && !uit->second.empty()) {
      StringRef intrName = uit->second;
      // Build type-suffixed name: llvm.exp → llvm.exp.f32
      std::string fullName = (intrName + "." +
                              (elemTy.isF32()   ? "f32"
                               : elemTy.isF64() ? "f64"
                                                : "f16"))
                                 .str();

      auto funcTy = LLVM::LLVMFunctionType::get(elemTy, {elemTy});
      auto funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
          rewriter, op, fullName, funcTy);
      Value result = LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0])
                         .getResult();

      // rsqrt = 1.0 / sqrt
      if (symbol.contains("rsqrt")) {
        Value one = LLVM::ConstantOp::create(
            rewriter, loc, elemTy, rewriter.getFloatAttr(elemTy, 1.0));
        result = LLVM::FDivOp::create(rewriter, loc, one, result);
      }
      // expm1 = exp(x) - 1
      if (symbol.contains("expm1")) {
        Value one = LLVM::ConstantOp::create(
            rewriter, loc, elemTy, rewriter.getFloatAttr(elemTy, 1.0));
        result = LLVM::FSubOp::create(rewriter, loc, result, one);
      }
      return {result};
    }

    // Binary intrinsics
    auto bit = binaryMap.find(symbol);
    if (bit != binaryMap.end() && !bit->second.empty()) {
      StringRef intrName = bit->second;
      std::string fullName = (intrName + "." +
                              (elemTy.isF32()   ? "f32"
                               : elemTy.isF64() ? "f64"
                                                : "f16"))
                                 .str();
      auto funcTy = LLVM::LLVMFunctionType::get(elemTy, {elemTy, elemTy});
      auto funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
          rewriter, op, fullName, funcTy);
      SmallVector<Value> args = {operands[0][0], operands[0][1]};
      return {LLVM::createLLVMCallOp(rewriter, loc, funcOp, args).getResult()};
    }

    // fmod → LLVM::FRemOp
    if (symbol.contains("fmod")) {
      return {LLVM::FRemOp::create(rewriter, loc, elemTy, operands[0][0],
                                   operands[0][1])};
    }

    // Trig functions not in LLVM intrinsics — use math lib calls
    // tan, asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh, atanh
    // For now, emit as external function calls; metal-llc's
    // MetalLLVMToAIRIntrinsics pass maps them to air.* builtins.
    {
      auto funcTy = operands[0].size() == 1
                        ? LLVM::LLVMFunctionType::get(elemTy, {elemTy})
                        : LLVM::LLVMFunctionType::get(elemTy, {elemTy, elemTy});
      auto funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(rewriter, op,
                                                               symbol, funcTy);
      return {LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0])
                  .getResult()};
    }
  }
};

// Fragment ABI elementwise: when an f32 #mma value carries the fragment struct,
// apply the op per fragment vector (<64 x f32>). The struct layout is identical
// across operands of the same #mma type, so a binary op is a lane-wise vector
// op on matching slots and a unary op maps over the slots. Fires only when the
// converted operand/result types are the fragment struct; otherwise defers to
// the generic flat elementwise lowering.
template <typename SrcOp, typename LLVMOp>
struct AppleMmaFragmentBinaryConversion : public ConvertOpToLLVMPattern<SrcOp> {
  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SrcOp::Adaptor;

  static LLVMStructType fragStruct(Type t) {
    auto s = dyn_cast_or_null<LLVMStructType>(t);
    if (s && !s.getBody().empty() && isa<VectorType>(s.getBody()[0]))
      return s;
    return nullptr;
  }

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs(), rhs = adaptor.getRhs();
    auto sTy = fragStruct(lhs.getType());
    if (!sTy || fragStruct(rhs.getType()) != sTy)
      return failure();
    Value result = LLVM::UndefOp::create(rewriter, loc, sTy);
    for (size_t i = 0; i < sTy.getBody().size(); ++i) {
      Value a = LLVM::ExtractValueOp::create(
          rewriter, loc, sTy.getBody()[i], lhs, ArrayRef<int64_t>{(int64_t)i});
      Value b = LLVM::ExtractValueOp::create(
          rewriter, loc, sTy.getBody()[i], rhs, ArrayRef<int64_t>{(int64_t)i});
      Value v = LLVMOp::create(rewriter, loc, a, b);
      result = LLVM::InsertValueOp::create(rewriter, loc, sTy, result, v,
                                           ArrayRef<int64_t>{(int64_t)i});
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename SrcOp, typename LLVMOp>
struct AppleMmaFragmentUnaryConversion : public ConvertOpToLLVMPattern<SrcOp> {
  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SrcOp::Adaptor;

  static LLVMStructType fragStruct(Type t) {
    auto s = dyn_cast_or_null<LLVMStructType>(t);
    if (s && !s.getBody().empty() && isa<VectorType>(s.getBody()[0]))
      return s;
    return nullptr;
  }

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value in = adaptor.getOperands()[0];
    auto sTy = fragStruct(in.getType());
    if (!sTy)
      return failure();
    Value result = LLVM::UndefOp::create(rewriter, loc, sTy);
    for (size_t i = 0; i < sTy.getBody().size(); ++i) {
      Value a = LLVM::ExtractValueOp::create(rewriter, loc, sTy.getBody()[i],
                                             in, ArrayRef<int64_t>{(int64_t)i});
      Value v = LLVMOp::create(rewriter, loc, a);
      result = LLVM::InsertValueOp::create(rewriter, loc, sTy, result, v,
                                           ArrayRef<int64_t>{(int64_t)i});
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// f16/bf16 accumulator-epilogue truncf on a fragment struct: a FORWARD. The
// f16/bf16 #mma result rides the same <64 x f32> accumulator fragment as the
// f32 input (getAppleMmaFragmentElemType keeps it f32), so this truncf is a
// no-op at the fragment level — the actual f32 -> f16/bf16 narrowing is emitted
// per-element on the EXTRACTED SCALAR in the #mma->#blocked store convert
// (ConvertLayoutOpAppleConversion), keeping the accumulator vectorized and the
// narrowing off the simdgroup register. Matches only when in/out are the same
// f32 fragment struct; defers otherwise so the generic flat truncf still runs.
struct AppleMmaFragmentTruncFConversion
    : public ConvertOpToLLVMPattern<arith::TruncFOp> {
  using ConvertOpToLLVMPattern<arith::TruncFOp>::ConvertOpToLLVMPattern;

  static LLVMStructType fragStruct(Type t) {
    auto s = dyn_cast_or_null<LLVMStructType>(t);
    if (s && !s.getBody().empty() && isa<VectorType>(s.getBody()[0]))
      return s;
    return nullptr;
  }

  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value in = adaptor.getIn();
    auto inTy = fragStruct(in.getType());
    if (!inTy)
      return failure();
    auto outTy = fragStruct(getTypeConverter()->convertType(op.getType()));
    if (outTy != inTy)
      return failure();
    rewriter.replaceOp(op, in);
    return success();
  }
};

// ── Fragment-ABI integer/mask + view lowerings (kkt op-web) ──────────────
//
// These keep the i32/i1/f32 kkt temporaries on the simdgroup register path so
// the f32 accumulator never leaves the <64 x ELT> fragment struct. expand_dims
// is a pure local reg-pack (slice flat scalar → its (row,col) fragment slot);
// broadcast is the validated scalar simd_shuffle replicate (oracle-proven);
// cmpi/andi/select are per-slot lane-wise vector ops.

static LLVMStructType fragStructOf(Type t) {
  auto s = dyn_cast_or_null<LLVMStructType>(t);
  if (s && !s.getBody().empty() && isa<VectorType>(s.getBody()[0]))
    return s;
  return nullptr;
}

// air.simd_shuffle.<ty>(val, i16 srcLane): pull `val` from absolute lane
// `srcLane`. Scalar-only (matches the fragment-oracle recipe). f32 native;
// i32/i1 go through the s.i32 form (i1 zext/trunc around it).
static Value emitFragShuffle(ConversionPatternRewriter &rewriter, Location loc,
                             ModuleOp mod, Value val, Value srcLaneI16) {
  auto *ctx = mod.getContext();
  Type vt = val.getType();
  auto i16Ty = IntegerType::get(ctx, 16);
  auto declare = [&](StringRef name, Type ty) -> LLVMFuncOp {
    if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
      return fn;
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto fnTy = LLVMFunctionType::get(ty, {ty, i16Ty}, false);
    auto fn = LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                                 Linkage::External);
    SmallVector<Attribute> pass{
        StringAttr::get(ctx, "convergent"), StringAttr::get(ctx, "noduplicate"),
        StringAttr::get(ctx, "nounwind"), StringAttr::get(ctx, "willreturn")};
    fn.setPassthroughAttr(ArrayAttr::get(ctx, pass));
    return fn;
  };
  if (vt.isF32()) {
    auto fn = declare("air.simd_shuffle.f32", vt);
    return LLVM::CallOp::create(rewriter, loc, fn, ValueRange{val, srcLaneI16})
        .getResult();
  }
  auto i32Ty = IntegerType::get(ctx, 32);
  // f16/bf16 fragments shuffle through the i32 form by bitcasting to a 16-bit
  // integer first (ZExt on a float is invalid IR); the integer round-trips the
  // bit pattern exactly, then bitcast back to the float type.
  bool isFloat16 = vt.isF16() || vt.isBF16();
  Value asI32 = val;
  bool isBool = vt.isInteger(1);
  if (isFloat16) {
    Value asI16 =
        LLVM::BitcastOp::create(rewriter, loc, IntegerType::get(ctx, 16), val);
    asI32 = LLVM::ZExtOp::create(rewriter, loc, i32Ty, asI16);
  } else if (isBool || !vt.isInteger(32)) {
    asI32 = LLVM::ZExtOp::create(rewriter, loc, i32Ty, val);
  }
  auto fn = declare("air.simd_shuffle.s.i32", i32Ty);
  Value sh =
      LLVM::CallOp::create(rewriter, loc, fn, ValueRange{asI32, srcLaneI16})
          .getResult();
  if (isFloat16) {
    Value asI16 =
        LLVM::TruncOp::create(rewriter, loc, IntegerType::get(ctx, 16), sh);
    return LLVM::BitcastOp::create(rewriter, loc, vt, asI16);
  }
  if (isBool)
    return LLVM::TruncOp::create(rewriter, loc, vt, sh);
  if (!vt.isInteger(32))
    return LLVM::TruncOp::create(rewriter, loc, vt, sh);
  return sh;
}

// thread_index_in_simdgroup (lane id, i32).
static Value emitLaneId(ConversionPatternRewriter &rewriter, Location loc,
                        ModuleOp mod) {
  auto i32Ty = IntegerType::get(mod.getContext(), 32);
  LLVMFuncOp fn = mod.lookupSymbol<LLVMFuncOp>("air.thread_index_in_simdgroup");
  if (!fn) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    fn = LLVMFuncOp::create(
        rewriter, mod.getLoc(), "air.thread_index_in_simdgroup",
        LLVMFunctionType::get(i32Ty, {}, false), Linkage::External);
  }
  return LLVM::CallOp::create(rewriter, loc, fn, ValueRange{}).getResult();
}

// expand_dims: slice<#mma> (flat per-thread scalars) → #mma fragment struct.
// Layout-preserving: the slice flat element k corresponds to result offset
// (row_k,col_k); pack it into that lane's fragment slot. No cross-lane move.
struct AppleMmaExpandDimsConversion
    : public ConvertOpToLLVMPattern<triton::ExpandDimsOp> {
  using ConvertOpToLLVMPattern<triton::ExpandDimsOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto resTy = cast<RankedTensorType>(op.getType());
    auto enc = dyn_cast<AppleMmaEncodingAttr>(resTy.getEncoding());
    if (!enc)
      return failure();
    auto outTy = getTypeConverter()->convertType(resTy);
    auto sTy = fragStructOf(outTy);
    if (!sTy)
      return failure();

    // Flat slice source scalars (NOT a fragment struct).
    SmallVector<Value> srcElems;
    Value src = adaptor.getSrc();
    if (auto inSt = dyn_cast<LLVMStructType>(src.getType()))
      for (unsigned i = 0; i < inSt.getBody().size(); ++i)
        srcElems.push_back(
            LLVM::ExtractValueOp::create(rewriter, loc, inSt.getBody()[i], src,
                                         ArrayRef<int64_t>{(int64_t)i}));
    else
      srcElems.push_back(src);

    // expand_dims is layout-preserving: enumerate the SLICE source's per-thread
    // offsets (1D, kept dim) — these match the flat source struct element count
    // exactly — and insert the expand axis to recover each element's (row,col)
    // home in the result #mma fragment. (The result's own emitOffsetForLayout
    // counts register-broadcast copies and would over-count vs the flat slice.)
    auto sliceTy = cast<RankedTensorType>(op.getSrc().getType());
    auto sliceOffsets = emitOffsetForLayout(sliceTy.getEncoding(), sliceTy);
    if (sliceOffsets.size() != srcElems.size())
      return failure();
    unsigned axis = op.getAxis();
    auto info = applegpu::getAppleMmaFragmentInfo(resTy, enc);
    Type eltTy = applegpu::getAppleMmaFragmentElemType(ctx, resTy);

    SmallVector<Value> frags(sTy.getBody().size());
    for (size_t i = 0; i < frags.size(); ++i)
      frags[i] = LLVM::UndefOp::create(rewriter, loc, sTy.getBody()[i]);
    for (size_t k = 0; k < sliceOffsets.size(); ++k) {
      int64_t kept = sliceOffsets[k][0];
      int64_t row = (axis == 1) ? kept : 0; // expand axis 1 → Mx1 (kept=row)
      int64_t col = (axis == 1) ? 0 : kept; // expand axis 0 → 1xN (kept=col)
      int64_t fragIdx, vecIdx;
      applegpu::appleMmaFragmentSlot(row, col, info, fragIdx, vecIdx);
      if (fragIdx >= (int64_t)frags.size())
        continue;
      Value v = srcElems[k];
      if (v.getType() != eltTy) {
        if (eltTy.isInteger(1) && !v.getType().isInteger(1))
          v = LLVM::TruncOp::create(rewriter, loc, eltTy, v);
      }
      Value vIdx = arith::ConstantIntOp::create(rewriter, loc, vecIdx, 32);
      frags[fragIdx] = LLVM::InsertElementOp::create(
          rewriter, loc, sTy.getBody()[fragIdx], frags[fragIdx], v, vIdx);
    }
    Value res = LLVM::UndefOp::create(rewriter, loc, sTy);
    for (size_t i = 0; i < frags.size(); ++i)
      res = LLVM::InsertValueOp::create(rewriter, loc, sTy, res, frags[i],
                                        ArrayRef<int64_t>{(int64_t)i});
    rewriter.replaceOp(op, res);
    return success();
  }
};

// broadcast: Mx1→MxN (col-replicate) or 1xN→MxN (row-replicate) on a #mma
// fragment, via the oracle-validated scalar simd_shuffle network.
//   col-replicate out[r][c]=in[r][0]: srcLane = T & ~0b01001, srcReg 0 → regs.
//   row-replicate out[r][c]=in[0][c]: srcLane = T & ~0b10110, srcReg = R.
struct AppleMmaBroadcastConversion
    : public ConvertOpToLLVMPattern<triton::BroadcastOp> {
  using ConvertOpToLLVMPattern<triton::BroadcastOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto resTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto enc = dyn_cast<AppleMmaEncodingAttr>(resTy.getEncoding());
    if (!enc)
      return failure();
    auto outTy = getTypeConverter()->convertType(resTy);
    auto sTy = fragStructOf(outTy);
    auto inSt = fragStructOf(adaptor.getSrc().getType());
    if (!sTy || !inSt)
      return failure();

    int64_t srcRows = srcTy.getShape()[0], srcCols = srcTy.getShape()[1];
    bool colReplicate = (srcCols == 1); // Mx1 → MxN
    bool rowReplicate = (srcRows == 1); // 1xN → MxN
    if (!colReplicate && !rowReplicate)
      return failure();

    auto mod = op->getParentOfType<ModuleOp>();
    Value lane = emitLaneId(rewriter, loc, mod);
    auto i16Ty = IntegerType::get(ctx, 16);
    // clear masks: col-replicate clears L0,L3 (0b01001=9); row-replicate clears
    // L1,L2,L4 (0b10110=22).
    int64_t clr = colReplicate ? 0x9 : 0x16;
    Value clrV = arith::ConstantIntOp::create(rewriter, loc, ~clr & 0x1f, 32);
    Value srcLane32 = arith::AndIOp::create(rewriter, loc, lane, clrV);
    Value srcLane16 = arith::TruncIOp::create(rewriter, loc, i16Ty, srcLane32);

    auto resInfo = applegpu::getAppleMmaFragmentInfo(resTy, enc);
    auto srcInfo = applegpu::getAppleMmaFragmentInfo(srcTy, enc);
    Type vecTy = sTy.getBody()[0];

    SmallVector<Value> outFrags(sTy.getBody().size());
    for (size_t i = 0; i < outFrags.size(); ++i)
      outFrags[i] = LLVM::UndefOp::create(rewriter, loc, sTy.getBody()[i]);

    Value srcStruct = adaptor.getSrc();
    SmallVector<Value> srcFrags;
    for (unsigned i = 0; i < inSt.getBody().size(); ++i)
      srcFrags.push_back(LLVM::ExtractValueOp::create(
          rewriter, loc, inSt.getBody()[i], srcStruct,
          ArrayRef<int64_t>{(int64_t)i}));

    auto resOffsets = emitOffsetForLayout(enc, resTy);
    for (auto &off : resOffsets) {
      int64_t r = off[0], c = off[1];
      int64_t outFrag, outVec;
      applegpu::appleMmaFragmentSlot(r, c, resInfo, outFrag, outVec);
      if (outFrag >= (int64_t)outFrags.size())
        continue;
      // Source slot: col-replicate reads (r,0); row-replicate reads (0,c).
      int64_t sr = colReplicate ? r : 0;
      int64_t sc = colReplicate ? 0 : c;
      int64_t inFrag, inVec;
      applegpu::appleMmaFragmentSlot(sr, sc, srcInfo, inFrag, inVec);
      if (inFrag >= (int64_t)srcFrags.size())
        inFrag = 0;
      Value inIdx = arith::ConstantIntOp::create(rewriter, loc, inVec, 32);
      Value scalar = LLVM::ExtractElementOp::create(rewriter, loc,
                                                    srcFrags[inFrag], inIdx);
      Value shuffled = emitFragShuffle(rewriter, loc, mod, scalar, srcLane16);
      Value outIdx = arith::ConstantIntOp::create(rewriter, loc, outVec, 32);
      outFrags[outFrag] = LLVM::InsertElementOp::create(
          rewriter, loc, vecTy, outFrags[outFrag], shuffled, outIdx);
    }
    Value res = LLVM::UndefOp::create(rewriter, loc, sTy);
    for (size_t i = 0; i < outFrags.size(); ++i)
      res = LLVM::InsertValueOp::create(rewriter, loc, sTy, res, outFrags[i],
                                        ArrayRef<int64_t>{(int64_t)i});
    rewriter.replaceOp(op, res);
    return success();
  }
};

// Per-slot integer binary (cmpi/andi) on #mma fragments. Like the f32 binary
// pattern but the result element type may differ from operands (cmpi: i32→i1).
template <typename SrcOp>
struct AppleMmaFragmentIntBinaryConversion
    : public ConvertOpToLLVMPattern<SrcOp> {
  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SrcOp::Adaptor;
  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs(), rhs = adaptor.getRhs();
    auto sTy = fragStructOf(lhs.getType());
    if (!sTy || fragStructOf(rhs.getType()) != sTy)
      return failure();
    auto resTy = dyn_cast<RankedTensorType>(op.getType());
    if (!resTy)
      return failure();
    auto outTy = this->getTypeConverter()->convertType(resTy);
    auto outSt = fragStructOf(outTy);
    if (!outSt)
      return failure();
    Type outVecTy = outSt.getBody()[0];
    Value result = LLVM::UndefOp::create(rewriter, loc, outSt);
    for (size_t i = 0; i < sTy.getBody().size(); ++i) {
      Value a = LLVM::ExtractValueOp::create(
          rewriter, loc, sTy.getBody()[i], lhs, ArrayRef<int64_t>{(int64_t)i});
      Value b = LLVM::ExtractValueOp::create(
          rewriter, loc, sTy.getBody()[i], rhs, ArrayRef<int64_t>{(int64_t)i});
      Value v;
      if constexpr (std::is_same_v<SrcOp, arith::CmpIOp>)
        v = LLVM::ICmpOp::create(rewriter, loc, outVecTy,
                                 convertCmpIPredicate(op.getPredicate()), a, b);
      else // andi
        v = LLVM::AndOp::create(rewriter, loc, a, b);
      result = LLVM::InsertValueOp::create(rewriter, loc, outSt, result, v,
                                           ArrayRef<int64_t>{(int64_t)i});
    }
    rewriter.replaceOp(op, result);
    return success();
  }
  static LLVM::ICmpPredicate convertCmpIPredicate(arith::CmpIPredicate p) {
    using A = arith::CmpIPredicate;
    using L = LLVM::ICmpPredicate;
    switch (p) {
    case A::eq:
      return L::eq;
    case A::ne:
      return L::ne;
    case A::slt:
      return L::slt;
    case A::sle:
      return L::sle;
    case A::sgt:
      return L::sgt;
    case A::sge:
      return L::sge;
    case A::ult:
      return L::ult;
    case A::ule:
      return L::ule;
    case A::ugt:
      return L::ugt;
    case A::uge:
      return L::uge;
    }
    llvm_unreachable("bad CmpIPredicate");
  }
};

// select(i1-mask fragment, f32 fragment, f32 fragment) → per-slot vector
// select.
struct AppleMmaFragmentSelectConversion
    : public ConvertOpToLLVMPattern<arith::SelectOp> {
  using ConvertOpToLLVMPattern<arith::SelectOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value cond = adaptor.getCondition();
    Value tv = adaptor.getTrueValue(), fv = adaptor.getFalseValue();
    auto sTy = fragStructOf(tv.getType());
    auto cTy = fragStructOf(cond.getType());
    if (!sTy || !cTy || fragStructOf(fv.getType()) != sTy)
      return failure();
    Value result = LLVM::UndefOp::create(rewriter, loc, sTy);
    for (size_t i = 0; i < sTy.getBody().size(); ++i) {
      Value c = LLVM::ExtractValueOp::create(
          rewriter, loc, cTy.getBody()[i], cond, ArrayRef<int64_t>{(int64_t)i});
      Value a = LLVM::ExtractValueOp::create(rewriter, loc, sTy.getBody()[i],
                                             tv, ArrayRef<int64_t>{(int64_t)i});
      Value b = LLVM::ExtractValueOp::create(rewriter, loc, sTy.getBody()[i],
                                             fv, ArrayRef<int64_t>{(int64_t)i});
      Value v = LLVM::SelectOp::create(rewriter, loc, c, a, b);
      result = LLVM::InsertValueOp::create(rewriter, loc, sTy, result, v,
                                           ArrayRef<int64_t>{(int64_t)i});
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// ── Pipeliner async copy lowering ────────────────────────────────────────
//
// Lower ttg.async_copy_global_to_local → synchronous per-element copy
// Lower ttg.async_commit_group → no-op (token = 0)
// Lower ttg.async_wait → threadgroup barrier
//
// The Triton software pipeliner generates these ops for multi-buffered
// load-compute overlap. On NVIDIA, async_copy lowers to cp.async (hw DMA).
// On Apple GPU, we lower to per-element loads + shared memory stores
// using Triton's lowerLocalLdSt for correct layout mapping.
//
// The pipeliner's multi-buffering still provides benefit by structuring
// the code for compute/copy overlap across loop iterations.
//
// When possible, we emit true async DMA via air.simdgroup_async_copy_2d.
// This requires: (1) extractable row stride from the MLIR def chain,
// (2) no mask (unmasked copy), (3) 2D tile.
// Otherwise we fall back to sync per-element copy via lowerLocalLdSt.

// Helper: get or create an external function declaration in the module
static LLVMFuncOp getOrCreateFn(ModuleOp mod, RewriterBase &rewriter,
                                StringRef name, Type retTy,
                                ArrayRef<Type> argTys) {
  if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto fnTy = LLVMFunctionType::get(retTy, argTys, false);
  return LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                            Linkage::External);
}

// Async copy event storage notes:
//
// IMPORTANT: Metal's air.wait_simdgroup_events expects a thread-local
// (addrspace 0) pointer-to-pointer, NOT a threadgroup (addrspace 3) pointer.
// Using a TG global crashes the GPU compiler.
//
// The alloca type is `ptr addrspace(3)` (a single event pointer), matching
// the reference pattern: `%ev = alloca %event_t addrspace(3)*, align 8`.
// Metal v1 bitcode doesn't handle arrays of typed pointers well, so each copy
// uses its own scalar slot rather than one shared array.

// Create a FRESH single-event alloca in the function entry block, one per async
// copy. Each async copy owns its own scalar event slot; the slot pointer is
// threaded out as the op's !ttg.async.token result so the matching async_wait
// waits on exactly this copy (see the AsyncToken type conversion). A single
// shared slot made air.wait_simdgroup_events wait on only the LAST copy, and a
// scavenged "wait on every slot" set waited on the WRONG (loop-rotated) buffer
// and on slots not yet stored on the first iteration (UB). Per-copy slots stay
// scalar to avoid the Metal-v1 "array of typed pointers" bitcode limitation.
//
// The slot is zero-initialized in the entry block so that a token reaching a
// wait WITHOUT a preceding store (the masked-skip branch, the sync-copy
// fallback, or a wait that is loop-hoisted ahead of the store on iteration 0)
// holds a complete/empty event: air.wait_simdgroup_events on a zero event slot
// is a real no-op, never a read of an uninitialized pointer.
static Value createEventAlloca(Operation *op, RewriterBase &rewriter) {
  auto *ctx = op->getContext();
  auto ptrTy3 = LLVMPointerType::get(ctx, 3);
  auto ptrTy0 = LLVMPointerType::get(ctx, 0);
  auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
  OpBuilder::InsertionGuard guard(rewriter);
  if (funcOp)
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  auto i64Ty = IntegerType::get(ctx, 64);
  Value one = LLVM::ConstantOp::create(rewriter, op->getLoc(), i64Ty,
                                       rewriter.getI64IntegerAttr(1));
  auto alloca = LLVM::AllocaOp::create(rewriter, op->getLoc(), ptrTy0, ptrTy3,
                                       one, /*alignment=*/8);
  Value nullEv = LLVM::ZeroOp::create(rewriter, op->getLoc(), ptrTy3);
  LLVM::StoreOp::create(rewriter, op->getLoc(), nullEv, alloca.getResult());
  return alloca.getResult();
}

// A standalone zero-initialized event slot for an async-copy lowering path that
// produces no real event (the sync fallback). Waiting on it is a no-op.
static Value createCompletedEventSlot(Operation *op, RewriterBase &rewriter) {
  return createEventAlloca(op, rewriter);
}

// air.simdgroup_async_copy_2d is a SIMDGROUP-cooperative DMA: each warp issues
// its own copy with a single WARP-UNIFORM tile origin and waits its own event.
// When warpsPerCTA[outerDim] > 1 the staged pipeline buffer's outer (slowest)
// dim is split across warps, but the warp-uniform origin makes every warp
// redundantly DMA the WHOLE tile into the same shared region. Those concurrent
// copies write-write race: a per-simdgroup air.wait_simdgroup_events only
// drains the issuing warp's own copy, and the threadgroup barrier after the
// wait fences regular threadgroup stores, not the async DMA engine's writes, so
// a sibling warp can read the buffer before another warp's still-in-flight copy
// has finished (verified: both the partitioned A operand and the replicated B
// operand of the matmul_layer_norm 32x64x16 nw4 kernel corrupt nondeterminis-
// tically). The AIR JIT has no cooperative-copy lowering that merges/serializes
// them. Keep any such multi-warp-outer-dim copy on the layout-exact synchronous
// copy (real threadgroup stores, which the membar passes order correctly).
// Single-warp-per-outer-dim copies (warpsPerCTA[outerDim] == 1) are race-free
// and stay on the fast async DMA path.
//
// CAVEAT: this predicate is deliberately BROAD: it also routes correct nw>=2
// GEMM operand copies to the sync path, a PERF (not correctness) regression on
// those configs, because a tighter race-vs-safe discriminator could not be
// proven sound here. The proper fix is a cooperative multi-warp async copy (or
// a narrower predicate); tracked as the async-vs-sync path unification rework.
static bool asyncCopyOuterDimCrossWarp(ttg::AsyncCopyGlobalToLocalOp op) {
  auto srcTy = op.getSrc().getType();
  auto enc = srcTy.getEncoding();
  auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(enc);
  if (!blocked)
    return false;
  auto warpsPerCTA = blocked.getWarpsPerCTA();
  auto order = blocked.getOrder();
  if (warpsPerCTA.empty() || order.empty())
    return false;
  // Outer (slowest-varying) dim is the last entry of order.
  unsigned outerDim = order.back();
  if (outerDim >= warpsPerCTA.size())
    return false;
  return warpsPerCTA[outerDim] > 1;
}

// Partition factor for a multi-warp-outer async copy. Returns the number of
// warps along the tile's outer (row) dim when the row dim divides evenly across
// them (so the tile splits into that many disjoint per-warp row bands), else 0
// (not cleanly partitionable;
// caller keeps the sync bail). When >0 the copy is split into that many
// disjoint horizontal bands, one per outer-dim warp, so the warps no longer
// write-write race on a shared region.
static Value emitWarp0Pred(ttg::AsyncCopyGlobalToLocalOp op,
                           ConversionPatternRewriter &rewriter, Location loc) {
  auto *ctx = op.getContext();
  auto mod = op->getParentOfType<ModuleOp>();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
  auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
      LLVMFuncOp::create(rewriter, mod.getLoc(),
                         "air.thread_position_in_threadgroup", tidFnTy,
                         Linkage::External);
  }
  auto tidFn =
      mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");
  Value tidStruct =
      LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
  Value tid = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                           ArrayRef<int64_t>{0});
  Value c32 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                       rewriter.getI32IntegerAttr(32));
  return LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ult, tid,
                              c32);
}

// Emit the outer-dim warp index wR for the partitioned copy: warpId = tid/32
// decomposed against the layout's warp order so wR selects which row band this
// warp owns. Mirrors the wR decomposition at the MMA index emitters.

// Struct to hold all components extracted from the async_copy pointer chain.
struct AsyncCopyPtrInfo {
  Value stride;   // Row stride scalar (MLIR, in elements)
  Value basePtr;  // Scalar base pointer (MLIR, tt.ptr)
  Value rowStart; // Scalar first-row index (MLIR, i32/i64), or nullptr if 0
  Value colStart; // Scalar first-col index (MLIR, i32/i64), or nullptr if 0
  // Compile-time fallbacks used when the first-element offset is a FOLDED splat
  // constant (arith.constant dense<C>) rather than a tt.splat of an SSA scalar.
  // The software pipeliner emits the prefetched buffer's K-block offset this
  // way (make_range + dense<BLOCK_K>); without capturing it, every prefetched
  // slab read K-block 0, corrupting num_stages>=3. INT64_MIN means "no
  // constant".
  int64_t rowStartConst = 0;
  int64_t colStartConst = 0;
  // Row stride as a compile-time constant (inductor dense<C>); INT64_MIN means
  // use the `stride` SSA Value instead.
  int64_t strideConst = INT64_MIN;
};

// First element of a 1D index tensor as a compile-time constant, when the
// tensor is `addi(make_range, dense<C>)` / `dense<C>` / `make_range` (=0).
// Returns true
// + sets `out` on success; false when the offset is not a folded constant (the
// caller then falls back to the SSA-scalar extractFirstElemScalar path).
static bool extractFirstElemConst(Value tensor, int64_t &out,
                                  bool peelModulo = false) {
  auto *defOp = tensor.getDefiningOp();
  if (!defOp)
    return false;
  if (isa<triton::ExpandDimsOp>(defOp))
    return extractFirstElemConst(defOp->getOperand(0), out, peelModulo);
  // See extractFirstElemScalar: peel a proven-no-op boundary modulo so a
  // constant row origin survives the wrap instead of folding to 0.
  if (peelModulo && isa<arith::RemSIOp, arith::RemUIOp>(defOp))
    return extractFirstElemConst(defOp->getOperand(0), out, peelModulo);
  if (isa<triton::MakeRangeOp>(defOp)) {
    out = 0;
    return true;
  }
  if (auto cst = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue()))
      if (dense.isSplat()) {
        out = dense.getSplatValue<APInt>().getSExtValue();
        return true;
      }
    return false;
  }
  if (isa<arith::AddIOp>(defOp)) {
    int64_t a, b;
    if (extractFirstElemConst(defOp->getOperand(0), a, peelModulo) &&
        extractFirstElemConst(defOp->getOperand(1), b, peelModulo)) {
      out = a + b;
      return true;
    }
  }
  return false;
}

// Extract the first-element scalar from a 1D index tensor.
//
// Patterns:
//   addi(splat(scalar), make_range(0, N)) → scalar
//   splat(scalar) → scalar
//   make_range(0, N) → nullptr (first element is 0)
//   expand_dims(inner_1d, axis) → recurse on inner_1d
static Value extractFirstElemScalar(Value tensor, bool peelModulo = false) {
  auto *defOp = tensor.getDefiningOp();
  if (!defOp)
    return nullptr;

  // Peel through expand_dims
  if (isa<triton::ExpandDimsOp>(defOp))
    return extractFirstElemScalar(defOp->getOperand(0), peelModulo);

  // Peel through a boundary-wrap modulo to its dividend. The caller sets
  // peelModulo ONLY when the wrap was already proven block-aligned (a no-op
  // over the tile), so the first element of (dividend % M) equals the first
  // element of the dividend. Without this, a program-id-dependent row origin
  // such as (pid_m*BLOCK_M + arange) % M_total is dropped to 0 and the async
  // DMA reads every program's tile from row 0 (correct only for pid_m == 0).
  if (peelModulo && isa<arith::RemSIOp, arith::RemUIOp>(defOp))
    return extractFirstElemScalar(defOp->getOperand(0), peelModulo);

  // addi(splat(scalar), make_range(0, N)) → first elem = scalar + 0 = scalar
  if (isa<arith::AddIOp>(defOp)) {
    for (unsigned i = 0; i < 2; i++) {
      auto *op = defOp->getOperand(i).getDefiningOp();
      if (op && isa<triton::SplatOp>(op))
        return op->getOperand(0);
    }
    return nullptr;
  }

  // splat(scalar) → scalar (uniform tensor)
  if (isa<triton::SplatOp>(defOp))
    return defOp->getOperand(0);

  // make_range(0, N) → first element is 0 (return nullptr to signal zero)
  if (isa<triton::MakeRangeOp>(defOp))
    return nullptr; // caller treats nullptr as zero

  return nullptr;
}

// Detect whether a tensor value's def chain contains a modulo (arith.remui /
// arith.remsi). The async-DMA fast path below reconstructs each row's device
// address as `basePtr + rowStart*stride + ...` with a single constant stride,
// which is only valid when the index is an affine function of the program/
// thread coordinates. Inductor's mm template wraps row/col indices with
// `(pid*BLOCK + arange) % M` (the standard bounds-wrapping idiom); that modulo
// makes the per-row stride non-constant (it folds back to the tensor origin at
// the wrap), so the linear DMA reads the wrong rows for any program_id > 0.
// When detected we fall back to the synchronous per-element copy, which uses
// each element's own (already-correct) pointer and is modulo-safe.
static bool defChainContainsModulo(Value v, unsigned budget = 128) {
  llvm::SmallVector<Value, 16> worklist;
  llvm::SmallPtrSet<Operation *, 32> visited;
  worklist.push_back(v);
  while (!worklist.empty() && budget-- > 0) {
    Value cur = worklist.pop_back_val();
    Operation *def = cur.getDefiningOp();
    if (!def || !visited.insert(def).second)
      continue;
    if (isa<arith::RemUIOp, arith::RemSIOp>(def))
      return true;
    // Stop at splat/make_range leaves; they cannot hide a modulo upstream that
    // affects the per-element row pattern.
    if (isa<triton::SplatOp, triton::MakeRangeOp>(def))
      continue;
    for (Value in : def->getOperands())
      worklist.push_back(in);
  }
  return false;
}

// Read a dense-splat integer attribute (e.g. tt.contiguity = dense<16>) off an
// op, returning its splat value or 0 when absent / non-splat.
static int64_t denseSplatAttr(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<DenseIntElementsAttr>(name);
  if (!attr || !attr.isSplat())
    return 0;
  return attr.getSplatValue<APInt>().getSExtValue();
}

// Decide whether every boundary-wrap modulo (rm%M / rn%N) in the source def
// chain is a NO-OP over this tile, so the constant-stride / runtime-affine DMA
// form is exact. This is the IR-based replacement for the AxisInfo contiguity
// probe (which returns null here because the analysis is built on the original
// module before conversion). The inductor mm template annotates each
// remsi/remui it emits with tt.contiguity = dense<C> and tt.divisibility =
// dense<C>; when that C is >= the tile extent the index runs contiguously
// across the entire wrap period inside the tile, i.e. no element in the tile
// actually wraps, so the affine reconstruction reads the correct rows. For an
// UNALIGNED shape the annotated contiguity drops below the tile extent
// (gcd-based), and we refuse, keeping the modulo-safe sync copy. Returns true
// only if there is at least one modulo AND all of them are block-aligned to the
// tile.
static bool allModuloBlockAligned(Value v, ArrayRef<int64_t> tileShape,
                                  unsigned budget = 128) {
  (void)tileShape;
  llvm::SmallVector<Value, 16> worklist;
  llvm::SmallPtrSet<Operation *, 32> visited;
  worklist.push_back(v);
  bool sawModulo = false;
  while (!worklist.empty() && budget-- > 0) {
    Value cur = worklist.pop_back_val();
    Operation *def = cur.getDefiningOp();
    if (!def || !visited.insert(def).second)
      continue;
    if (isa<arith::RemUIOp, arith::RemSIOp>(def)) {
      sawModulo = true;
      int64_t contig = denseSplatAttr(def, "tt.contiguity");
      // The wrap is a no-op only if the result is contiguous across the full
      // extent of the dimension the modulo indexes. That dimension's extent is
      // exactly the number of elements in the (1D slice) modulo result, so the
      // self-consistent test is contig >= numElements(result). This correctly
      // accepts an aligned rm%M on a 16-row tile (contig 16 >= 16) regardless
      // of the other, larger tile dimension, and refuses an unaligned wrap
      // where the gcd-based contiguity drops below the extent.
      int64_t extent = 1;
      if (auto rt = dyn_cast<RankedTensorType>(def->getResult(0).getType()))
        extent = rt.getNumElements();
      if (contig < extent)
        return false;
      // Do not recurse past a proven-aligned modulo; its dividend's own
      // indexing is subsumed by the contiguity guarantee.
      continue;
    }
    if (isa<triton::SplatOp, triton::MakeRangeOp>(def))
      continue;
    for (Value in : def->getOperands())
      worklist.push_back(in);
  }
  return sawModulo;
}

// Function-level safety gate for the boundary-wrap (rm%M / rn%N) async path.
//
// The software pipeliner hoists the modulo into the loop PROLOGUE: the in-loop
// async-copy source is just an iter-arg pointer incremented by BK each step, so
// a def-chain walk from that op never reaches the remsi/remui. Per-op modulo
// detection (defChainContainsModulo / allModuloBlockAligned on op.getSrc())
// therefore cannot tell an ALIGNED kernel (wrap is a no-op, async is exact)
// from an UNALIGNED one (wrap is live, async silently corrupts). We
// disambiguate at FUNCTION scope: if the enclosing function contains ANY
// remsi/remui that is not proven block-aligned (its result is not annotated
// tt.contiguity >= its own extent), the kernel has a live wrap and the async
// modulo path is unsafe for EVERY copy in it. Aligned kernels carry the
// dense<extent> contiguity attr on every wrap (Triton's AxisInfo proves the
// divisibility); unaligned kernels emit the bare remsi with no such attr.
// Returns true when the function is safe.
static bool functionModuloIsSafe(Operation *op) {
  auto func = op->getParentOfType<FunctionOpInterface>();
  if (!func)
    return false;
  bool safe = true;
  func.walk([&](Operation *m) {
    if (!isa<arith::RemUIOp, arith::RemSIOp>(m))
      return;
    // Only per-element INDEX wraps threaten the affine form. A scalar modulo
    // (extent 1) is the GROUP-M program-id swizzle (pid % group_size): it picks
    // which tile a program computes, not the intra-tile element addresses, so
    // it never breaks the per-tile affine access. Skip non-tensor / 1-element
    // results.
    auto rt = dyn_cast<RankedTensorType>(m->getResult(0).getType());
    if (!rt || rt.getNumElements() <= 1)
      return;
    int64_t contig = 0;
    if (auto attr = m->getAttrOfType<DenseIntElementsAttr>("tt.contiguity"))
      if (attr.isSplat())
        contig = attr.getSplatValue<APInt>().getSExtValue();
    int64_t extent = rt.getNumElements();
    if (contig < extent)
      safe = false;
  });
  return safe;
}

// Extract all pointer components from a pointer tensor's MLIR def chain.
//
// Pattern: async_copy src = tt.addptr(broadcast(addptr(splat(base),
//                          muli(expand_dims(row_offs), splat(STRIDE)))),
//                          broadcast(col_offs))
//
// Returns true if extraction succeeded. Populates `info` with:
//   - stride: row stride scalar (in elements)
//   - basePtr: scalar base pointer
//   - rowStart: first-row index scalar (or nullptr if 0)
//   - colStart: first-col index scalar (or nullptr if 0)
// Flattened-index pattern (inductor / gemm_bench GEMM):
//   addptr(splat(base), addi(rowTerm, colTerm))
// where rowTerm = broadcast(muli(expand_dims(rowRange), splat(stride))) (row*K)
// and   colTerm = broadcast(expand_dims(colRange)) (col,1) This differs from
// the nested addptr(broadcast(addptr(splat,muli)),col) shape handled below:
// here a single addptr adds a COMBINED 2D index, so the row (strided) and col
// (unit-stride) terms are summed before the addptr.
static bool extractFlatAsyncCopyPtrInfo(triton::AddPtrOp addptrOp,
                                        AsyncCopyPtrInfo &info,
                                        bool allowModulo) {
  auto *baseSplatOp = addptrOp->getOperand(0).getDefiningOp();
  if (!baseSplatOp || !isa<triton::SplatOp>(baseSplatOp)) {
    return false;
  }
  Value combinedIdx = addptrOp->getOperand(1);
  // A modulo normally defeats constant-stride reconstruction. allowModulo is
  // set by the caller when AxisInfo proved the tile is contiguous in the inner
  // dim, i.e. the rm%M / rn%N boundary-wrap is a no-op over this access, so the
  // strided-DMA form is exact.
  if (!allowModulo && defChainContainsModulo(combinedIdx)) {
    return false;
  }
  auto *addiOp = combinedIdx.getDefiningOp();
  if (!addiOp || !isa<arith::AddIOp>(addiOp)) {
    return false;
  }

  // Identify which addi operand is the strided (row) term vs the unit (col)
  // term: the row term's def chain contains a muli(expand_dims, splat(stride)).
  auto peelBroadcast = [](Value v) -> Value {
    if (auto *bc = v.getDefiningOp())
      if (isa<triton::BroadcastOp>(bc))
        return bc->getOperand(0);
    return v;
  };
  // Recover the expand_dims axis of an index term, i.e. which logical tensor
  // dimension this index varies along. The DMA copies a row-major tile (rows
  // `stride` apart, each row `tileCols` CONTIGUOUS elements). That is only
  // valid when the strided term indexes the OUTER dim (axis-1 expand -> Nx1
  // column vector broadcast across columns) and the unit term indexes the INNER
  // dim (axis-0 expand -> 1xM broadcast across rows). A TRANSPOSED operand
  // swaps these (the strided term indexes the inner dim), which would make the
  // DMA read the tile transposed -> silent miscompile. Returns the varying
  // logical dim (0 = outer/row, 1 = inner/col) or -1 if it cannot be
  // determined.
  auto termVaryingDim = [&](Value term) -> int {
    Value inner = peelBroadcast(term);
    // Peel a muli(expand_dims, stride) to reach the expand_dims, or take the
    // term directly if it is already an expand_dims (unit-stride col term).
    if (auto *muli = inner.getDefiningOp())
      if (isa<arith::MulIOp>(muli))
        for (unsigned i = 0; i < 2; i++)
          if (auto *e = muli->getOperand(i).getDefiningOp())
            if (isa<triton::ExpandDimsOp>(e))
              inner = muli->getOperand(i);
    auto exp = inner.getDefiningOp<triton::ExpandDimsOp>();
    if (!exp)
      return -1;
    // expand_dims axis A inserts a size-1 dim at A; the original index then
    // varies along the OTHER dim. For 2D: axis 1 -> varies along dim 0 (row),
    // axis 0 -> varies along dim 1 (col).
    return exp.getAxis() == 1 ? 0 : 1;
  };
  auto matchRowTerm = [&](Value term, Value &strideOut,
                          Value &rangeOut) -> bool {
    Value inner = peelBroadcast(term);
    auto *muli = inner.getDefiningOp();
    if (!muli || !isa<arith::MulIOp>(muli))
      return false;
    for (unsigned i = 0; i < 2; i++) {
      Value opnd = muli->getOperand(i);
      auto *op = opnd.getDefiningOp();
      if (op && isa<triton::SplatOp>(op)) {
        // gemm_bench shape: stride = tt.splat(scalar).
        strideOut = op->getOperand(0);
      } else if (op && isa<triton::ExpandDimsOp>(op)) {
        rangeOut = opnd;
      } else if (auto cst = dyn_cast_or_null<arith::ConstantOp>(op)) {
        // inductor shape: stride = arith.constant dense<C>. Record the scalar
        // C; the LLVM emitter materializes it (creating an op here would leave
        // an unconvertible arith.constant post-legalization).
        if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue()))
          if (dense.isSplat())
            info.strideConst = dense.getSplatValue<APInt>().getSExtValue();
      }
    }
    return (strideOut || info.strideConst != INT64_MIN) && rangeOut;
  };

  Value lhs = addiOp->getOperand(0), rhs = addiOp->getOperand(1);
  Value strideVal, rowRange, colTerm;
  if (matchRowTerm(lhs, strideVal, rowRange))
    colTerm = rhs;
  else if (matchRowTerm(rhs, strideVal, rowRange))
    colTerm = lhs;
  else {
    return false;
  }

  // The strided term must index the OUTER dim and the unit-stride term the
  // INNER dim. If they are swapped (transposed operand: inner dim is strided),
  // the contiguous-row DMA would read the tile transposed. Bail so the caller
  // falls back to the layout-exact sync copy.
  {
    Value stridedTerm = (colTerm == rhs) ? lhs : rhs;
    int stridedDim = termVaryingDim(stridedTerm);
    int colDim = termVaryingDim(colTerm);
    if (stridedDim != 0 || colDim != 1)
      return false;
  }

  info.basePtr = baseSplatOp->getOperand(0);
  info.stride = strideVal;
  // The row range may pass through a boundary-wrap modulo. When the wrap was
  // proven block-aligned (allowModulo), peel it so the program-id-dependent row
  // origin is preserved; otherwise the DMA reads from row 0 for every program.
  info.rowStart = extractFirstElemScalar(rowRange, allowModulo);
  if (!info.rowStart)
    extractFirstElemConst(rowRange, info.rowStartConst, allowModulo);

  if (!allowModulo && defChainContainsModulo(colTerm))
    return false;
  Value colInner = peelBroadcast(colTerm);
  // The column index may pass through a proven-no-op boundary wrap (rn % N),
  // exactly like the row index above. Peel it under allowModulo so the
  // program-id-dependent column origin (pid_n * BLOCK_N) is preserved; without
  // this the wrap defeats scalar/const extraction and colStart silently drops
  // to 0, so every N-block's DMA reads B from column 0 -> only pid_n == 0 is
  // correct and the right output half is wrong.
  info.colStart = extractFirstElemScalar(colInner, allowModulo);
  if (!info.colStart)
    extractFirstElemConst(colInner, info.colStartConst, allowModulo);
  // With a contiguity-proven no-op modulo, the tile origin is the unwrapped
  // first index; the wrap contributes nothing, so default start = 0.
  return true;
}

static bool extractAsyncCopyPtrInfo(Value ptrTensor, AsyncCopyPtrInfo &info,
                                    bool allowModulo) {
  // Walk: ptrTensor → addptr → broadcast → addptr → muli → splat(stride)
  auto *addptrOp = ptrTensor.getDefiningOp();
  if (!addptrOp || !isa<triton::AddPtrOp>(addptrOp))
    return false;

  // The first operand of the outer addptr is broadcast(inner_addptr).
  // If it is instead a splat (flattened combined-index GEMM), use the flat
  // matcher.
  Value broadcastedBase = addptrOp->getOperand(0);
  auto *broadcastOp = broadcastedBase.getDefiningOp();
  if (broadcastOp && isa<triton::SplatOp>(broadcastOp))
    return extractFlatAsyncCopyPtrInfo(cast<triton::AddPtrOp>(addptrOp), info,
                                       allowModulo);
  if (!broadcastOp || !isa<triton::BroadcastOp>(broadcastOp))
    return false;

  Value innerAddptr = broadcastOp->getOperand(0);
  auto *innerOp = innerAddptr.getDefiningOp();
  if (!innerOp || !isa<triton::AddPtrOp>(innerOp))
    return false;

  // Extract scalar base pointer from splat(base) — first operand of inner
  // addptr
  auto *baseSplatOp = innerOp->getOperand(0).getDefiningOp();
  if (!baseSplatOp || !isa<triton::SplatOp>(baseSplatOp))
    return false;
  info.basePtr = baseSplatOp->getOperand(0);

  // The second operand of innerAddptr is the row offset:
  // muli(expand_dims(row_range), splat(stride))
  Value rowOffset = innerOp->getOperand(1);
  auto *muliOp = rowOffset.getDefiningOp();
  if (!muliOp || !isa<arith::MulIOp>(muliOp))
    return false;

  // Non-affine (modulo-indexed) row offset defeats the constant-stride DMA
  // reconstruction below. Bail so the caller uses the sync per-element copy.
  if (defChainContainsModulo(rowOffset))
    return false;

  // One operand of muli is expand_dims(range), the other is splat(stride)
  Value expandDimsVal;
  bool foundStride = false;
  for (unsigned i = 0; i < 2; i++) {
    auto *op = muliOp->getOperand(i).getDefiningOp();
    if (op && isa<triton::SplatOp>(op)) {
      info.stride = op->getOperand(0);
      foundStride = true;
    } else if (op && isa<triton::ExpandDimsOp>(op)) {
      expandDimsVal = muliOp->getOperand(i);
    }
  }
  if (!foundStride)
    return false;

  // The strided (row) offset must index the OUTER dim (expand_dims axis 1 ->
  // Nx1 column vector). A transposed operand instead puts the explicit stride
  // on the INNER dim (expand_dims axis 0), and the contiguous-row DMA would
  // then read the tile transposed -> silent miscompile. Bail to the sync copy.
  if (expandDimsVal)
    if (auto exp = expandDimsVal.getDefiningOp<triton::ExpandDimsOp>())
      if (exp.getAxis() != 1)
        return false;

  // Extract first-row scalar from expand_dims(row_offs_1d)
  if (expandDimsVal) {
    info.rowStart = extractFirstElemScalar(expandDimsVal, allowModulo);
    if (!info.rowStart)
      extractFirstElemConst(expandDimsVal, info.rowStartConst, allowModulo);
  }
  // nullptr (and rowStartConst 0) means first row = 0

  // Extract col offset from outer addptr's second operand:
  // broadcast(expand_dims(col_offs_1d)) or broadcast(col_offs)
  Value colOffset = addptrOp->getOperand(1);
  // A modulo on the column index is likewise non-affine; bail to sync copy.
  if (defChainContainsModulo(colOffset))
    return false;
  auto *colBroadcastOp = colOffset.getDefiningOp();
  if (colBroadcastOp && (isa<triton::BroadcastOp>(colBroadcastOp) ||
                         isa<triton::ExpandDimsOp>(colBroadcastOp))) {
    info.colStart = extractFirstElemScalar(colBroadcastOp->getOperand(0));
    if (!info.colStart)
      extractFirstElemConst(colBroadcastOp->getOperand(0), info.colStartConst);
  }
  // nullptr (and colStartConst 0) means first col = 0

  return true;
}

// Conservative compile-time proof that the INNER (dim-1) index of a 2D source
// pointer tensor is unit-stride, i.e. adjacent columns are adjacent in memory.
// The async 2D DMA copies each tile row as `tileCols` CONTIGUOUS elements, so a
// non-unit inner stride (transposed/strided-view operand) makes it read the
// tile transposed -> silent miscompile. This walks the addptr index chain and
// returns true ONLY when it positively proves the inner dim carries no stride
// multiply; any unrecognized shape returns false (forces the exact sync copy).
//
// The offset feeding the addptr is a sum of per-dim terms; a strided dim looks
// like broadcast?(muli(expand_dims(range, axis), splat/const stride)). The
// inner dim is the one whose expand_dims axis == 0 (1xN, broadcast over rows).
// If that inner term is multiplied by anything other than 1, columns are not
// contiguous.
static bool innerDimIsUnitStride(Value ptrTensor) {
  auto *addptrOp = ptrTensor.getDefiningOp();
  if (!addptrOp || !isa<triton::AddPtrOp>(addptrOp))
    return false;

  // Collect the additive index terms. Handle both the flat shape
  // (addptr(splat(base), addi(rowTerm, colTerm))) and the nested shape
  // (addptr(broadcast(addptr(splat(base), rowTerm)), colTerm)).
  SmallVector<Value> terms;
  SmallVector<Value> work;
  work.push_back(addptrOp->getOperand(1));
  if (auto *b = addptrOp->getOperand(0).getDefiningOp())
    if (isa<triton::BroadcastOp>(b))
      work.push_back(addptrOp->getOperand(0));
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (auto *bc = v.getDefiningOp()) {
      if (isa<triton::BroadcastOp>(bc)) {
        work.push_back(bc->getOperand(0));
        continue;
      }
      if (auto add = dyn_cast<arith::AddIOp>(bc)) {
        work.push_back(add.getLhs());
        work.push_back(add.getRhs());
        continue;
      }
      if (auto inAddptr = dyn_cast<triton::AddPtrOp>(bc)) {
        // nested: the base carries another index term
        work.push_back(inAddptr->getOperand(1));
        continue;
      }
    }
    terms.push_back(v);
  }

  auto peelBroadcast = [](Value v) -> Value {
    if (auto *bc = v.getDefiningOp())
      if (isa<triton::BroadcastOp>(bc))
        return bc->getOperand(0);
    return v;
  };

  // Find the inner-dim term (expand_dims axis 0). It must NOT be wrapped in a
  // muli by a non-unit stride. We require that exactly one term is the inner
  // dim and it is a bare expand_dims (unit stride).
  bool sawInner = false;
  for (Value t : terms) {
    Value inner = peelBroadcast(t);
    // A muli(expand_dims(.. axis 0 ..), stride) on the inner dim is non-unit.
    if (auto muli = inner.getDefiningOp<arith::MulIOp>()) {
      for (unsigned i = 0; i < 2; i++)
        if (auto exp =
                muli->getOperand(i).getDefiningOp<triton::ExpandDimsOp>())
          if (exp.getAxis() == 0)
            return false; // inner dim multiplied by a stride -> not unit
      continue;
    }
    if (auto exp = inner.getDefiningOp<triton::ExpandDimsOp>())
      if (exp.getAxis() == 0)
        sawInner = true;
  }
  return sawInner;
}

static bool rectStrideIs64B(ttg::AsyncCopyGlobalToLocalOp op) {
  return op->hasAttr("applegpu.rect_stride_64b");
}

struct AsyncCopyGlobalToLocalOpAppleConversion
    : public ConvertOpToLLVMPattern<ttg::AsyncCopyGlobalToLocalOp> {
  // AxisInfo lets us prove an access is affine (constant-stride, with no-op
  // modulo folded via divisibility) instead of brittle syntactic pattern walks.
  ModuleAxisInfoAnalysis *axisInfo = nullptr;

  AsyncCopyGlobalToLocalOpAppleConversion(LLVMTypeConverter &tc,
                                          ModuleAxisInfoAnalysis *ai,
                                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern(tc, benefit), axisInfo(ai) {}

  // Sync fallback: per-element load from device + store to shared memory
  LogicalResult lowerSyncCopy(ttg::AsyncCopyGlobalToLocalOp op,
                              OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());

    Value llSrc = adaptor.getSrc();
    Value llDst = adaptor.getResult();
    Value llMask = adaptor.getMask();

    auto srcElems = unpackLLElements(loc, llSrc, rewriter);
    if (srcElems.empty())
      return failure();

    SmallVector<Value> maskElems;
    if (llMask)
      maskElems = unpackLLElements(loc, llMask, rewriter);

    SmallVector<Value> loadedVals;
    unsigned numElems = srcElems.size();
    for (unsigned i = 0; i < numElems; i++) {
      Value loaded;
      if (maskElems.empty()) {
        loaded = LLVM::LoadOp::create(rewriter, loc, elemTy, srcElems[i]);
      } else {
        Value mask = maskElems[i];
        Value zero;
        if (elemTy.isIntOrIndex())
          zero = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                          rewriter.getIntegerAttr(elemTy, 0));
        else
          zero = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                          rewriter.getFloatAttr(elemTy, 0.0));
        Value val = LLVM::LoadOp::create(rewriter, loc, elemTy, srcElems[i]);
        loaded = LLVM::SelectOp::create(rewriter, loc, mask, val, zero);
      }
      loadedVals.push_back(loaded);
    }

    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, elemTy, rewriter);

    auto regLayout = ttg::toLinearLayout(srcTy);
    auto sharedLayout = ttg::toLinearLayout(dstTy);
    auto cvt = regLayout.invertAndCompose(sharedLayout);

    TargetInfo targetInfo;
    lowerLocalLdSt(loc, ctx, cvt, loadedVals, elemTy, dstTy, smemObj, rewriter,
                   targetInfo);

    // The data is already in shared memory synchronously, so the matching
    // async_wait must be a no-op for this token. Return a zero-initialized
    // event slot: waiting on it does nothing. (Token type is ptr addrspace(0).)
    Value evSlot = createCompletedEventSlot(op, rewriter);
    rewriter.replaceOp(op, evSlot);
    return success();
  }

  // Check if the MLIR mask is a tt.splat of a scalar i1.
  // The pipeliner generates: %mask_scalar = arith.cmpi ... ; %mask = tt.splat
  // %mask_scalar When this pattern holds, the mask is uniform (all-true or
  // all-false), so we can gate the async DMA on the scalar boolean. Returns the
  // scalar MLIR Value, or nullptr if not a splat.
  static Value extractScalarMask(Value mask) {
    if (!mask)
      return nullptr;
    auto *defOp = mask.getDefiningOp();
    if (!defOp || !isa<triton::SplatOp>(defOp))
      return nullptr;
    Value scalar = defOp->getOperand(0);
    if (!scalar.getType().isInteger(1))
      return nullptr;
    return scalar;
  }

  // Rectangular boundary mask: every leaf of the and-tree is either a uniform
  // splat(i1) or a row/col bound `cmp slt idx, splat(extent)`. The async DMA
  // honors such a mask exactly by clamping the source tile to
  // (extent - firstIdx) remaining rows/cols and zero-filling the rest
  // (clamp mode 0), so other=0 masked loads can stay on the DMA path.
  struct RectMaskInfo {
    SmallVector<Value, 2> uniforms; // scalar i1 leaves, all must be true
    Value rowBound, rowFirst;       // rows valid while row < rowBound
    Value colBound, colFirst;       // cols valid while col < colBound
    bool hasRow = false, hasCol = false;
    int64_t rowFirstConst = 0, colFirstConst = 0; // when scalar absent
  };

  static bool matchRectMask(Value mask, RectMaskInfo &info) {
    auto *def = mask.getDefiningOp();
    if (!def)
      return false;
    if (auto andOp = dyn_cast<arith::AndIOp>(def))
      return matchRectMask(andOp.getLhs(), info) &&
             matchRectMask(andOp.getRhs(), info);
    if (auto splat = dyn_cast<triton::SplatOp>(def)) {
      Value s = splat.getSrc();
      if (!s.getType().isInteger(1))
        return false;
      info.uniforms.push_back(s);
      return true;
    }
    if (auto bc = dyn_cast<triton::BroadcastOp>(def))
      return matchRectMask(bc.getSrc(), info);
    auto cmp = dyn_cast<arith::CmpIOp>(def);
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
      return false;
    auto resTy = dyn_cast<RankedTensorType>(cmp.getType());
    if (!resTy || resTy.getRank() != 2)
      return false;
    // Which logical dim does this bound? A (1 x N) row vector bounds the
    // inner (col) dim; an (N x 1) column vector bounds the outer (row) dim.
    bool boundsCols;
    if (resTy.getDimSize(0) == 1 && resTy.getDimSize(1) > 1)
      boundsCols = true;
    else if (resTy.getDimSize(1) == 1 && resTy.getDimSize(0) > 1)
      boundsCols = false;
    else
      return false;
    // Extent must be a uniform scalar.
    Value bound = extractFirstElemScalar(cmp.getRhs());
    if (!bound) {
      int64_t c;
      if (!extractFirstElemConst(cmp.getRhs(), c))
        return false;
      // Constant extents do not appear in the GEMM shapes this targets.
      return false;
    }
    // The 2D DMA mis-copies when the source row stride is not 64B-aligned
    // (measured: K=72/1000 corrupt, K=80/96/1024 exact). Strides are the
    // matrix extents here, so require every bound to be a kernel argument
    // Triton proved 16-element divisible.
    // First index of the bounded range (its smallest element: indices ascend).
    Value first = extractFirstElemScalar(cmp.getLhs());
    int64_t firstConst = 0;
    if (!first && !extractFirstElemConst(cmp.getLhs(), firstConst))
      return false;
    if (boundsCols) {
      if (info.hasCol)
        return false;
      info.hasCol = true;
      info.colBound = bound;
      info.colFirst = first;
      info.colFirstConst = firstConst;
    } else {
      if (info.hasRow)
        return false;
      info.hasRow = true;
      info.rowBound = bound;
      info.rowFirst = first;
      info.rowFirstConst = firstConst;
    }
    return true;
  }

  // The DMA writes only the src-tile intersection (no zero fill, measured on
  // M1), so the masked-out remainder of the dst slot must be zeroed
  // explicitly: every lane of the issuing warp stores 0 to the dst elements
  // with col >= srcCols or row >= srcRows. Runs after the fire (disjoint
  // bytes), skipped per element when the tile is full.
  static void emitResidualZero(ConversionPatternRewriter &rewriter,
                               Location loc, ModuleOp mod, Value dstBase,
                               int64_t rows, int64_t cols, unsigned elemBytes,
                               Value srcWBytes, Value srcHRows) {
    auto *ctx = rewriter.getContext();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i16Ty = IntegerType::get(ctx, 16);
    auto f32Ty = Float32Type::get(ctx);
    assert(llvm::isPowerOf2_64(cols) && llvm::isPowerOf2_32(elemBytes));
    int64_t colShift = llvm::Log2_64(cols);
    int64_t elemShift = llvm::Log2_32(elemBytes);

    Value elemShiftV = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elemShift));
    Value srcCols = LLVM::LShrOp::create(rewriter, loc, srcWBytes, elemShiftV);

    // Other lowerings declare this intrinsic with i32 result; match them so
    // a single declaration serves every call site.
    auto i32T = IntegerType::get(ctx, 32);
    auto laneFn =
        getOrCreateFn(mod, rewriter, "air.thread_index_in_simdgroup", i32T, {});
    Value lane32 =
        LLVM::CallOp::create(rewriter, loc, laneFn, ValueRange{}).getResult();
    Value lane = LLVM::ZExtOp::create(rewriter, loc, i64Ty, lane32);
    (void)i16Ty;

    Value zeroF = LLVM::ConstantOp::create(rewriter, loc, f32Ty,
                                           rewriter.getF32FloatAttr(0.0f));
    Value colShiftV = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(colShift));
    Value colMask = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(cols - 1));
    Value colsV = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(cols));
    Value rowsV = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(rows));
    Value colPart = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::slt, srcCols, colsV);
    Value rowPart = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::slt, srcHRows, rowsV);
    Value partial = LLVM::OrOp::create(rewriter, loc, colPart, rowPart);
    auto *entry = rewriter.getInsertionBlock();
    auto entryPt = rewriter.getInsertionPoint();
    auto *fillB = rewriter.createBlock(entry->getParent(),
                                       std::next(Region::iterator(entry)));
    auto *doneB = rewriter.createBlock(entry->getParent(),
                                       std::next(Region::iterator(fillB)));
    doneB->getOperations().splice(doneB->begin(), entry->getOperations(),
                                  entryPt, entry->end());
    rewriter.setInsertionPointToEnd(entry);
    LLVM::CondBrOp::create(rewriter, loc, partial, fillB, doneB);
    rewriter.setInsertionPointToStart(fillB);
    int64_t total = rows * cols;
    Value totalV = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                            rewriter.getI64IntegerAttr(total));
    Value c32 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                         rewriter.getI64IntegerAttr(32));
    auto *loopB = rewriter.createBlock(
        fillB->getParent(), std::next(Region::iterator(fillB)), {i64Ty}, {loc});
    rewriter.setInsertionPointToEnd(fillB);
    LLVM::BrOp::create(rewriter, loc, ValueRange{lane}, loopB);
    rewriter.setInsertionPointToStart(loopB);
    Value idx = loopB->getArgument(0);
    Value row = LLVM::LShrOp::create(rewriter, loc, idx, colShiftV);
    Value col = LLVM::AndOp::create(rewriter, loc, idx, colMask);
    Value colOOB = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sge,
                                        col, srcCols);
    Value rowOOB = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sge,
                                        row, srcHRows);
    Value oob = LLVM::OrOp::create(rewriter, loc, colOOB, rowOOB);
    auto *storeB = rewriter.createBlock(loopB->getParent(),
                                        std::next(Region::iterator(loopB)));
    auto *latchB = rewriter.createBlock(storeB->getParent(),
                                        std::next(Region::iterator(storeB)));
    rewriter.setInsertionPointToEnd(loopB);
    LLVM::CondBrOp::create(rewriter, loc, oob, storeB, latchB);
    rewriter.setInsertionPointToStart(storeB);
    Value p = LLVM::GEPOp::create(rewriter, loc, dstBase.getType(), f32Ty,
                                  dstBase, ArrayRef<LLVM::GEPArg>{idx});
    LLVM::StoreOp::create(rewriter, loc, zeroF, p);
    LLVM::BrOp::create(rewriter, loc, ValueRange{}, latchB);
    rewriter.setInsertionPointToStart(latchB);
    Value next = LLVM::AddOp::create(rewriter, loc, i64Ty, idx, c32);
    Value done = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sge,
                                      next, totalV);
    LLVM::CondBrOp::create(rewriter, loc, done, doneB, ValueRange{}, loopB,
                           ValueRange{next});
    rewriter.setInsertionPointToStart(doneB);
  }

  // remaining = clamp(bound - first, 0, cap) as i64, emitted from remapped
  // scalars. Returns null when the scalars cannot be remapped.
  static Value emitRemaining(ConversionPatternRewriter &rewriter, Location loc,
                             Value bound, Value first, int64_t firstConst,
                             int64_t cap) {
    auto i64Ty = rewriter.getI64Type();
    Value b = rewriter.getRemappedValue(bound);
    if (!b)
      return nullptr;
    if (b.getType() != i64Ty)
      b = LLVM::SExtOp::create(rewriter, loc, i64Ty, b);
    Value f;
    if (first) {
      f = rewriter.getRemappedValue(first);
      if (!f)
        return nullptr;
      if (f.getType() != i64Ty)
        f = LLVM::SExtOp::create(rewriter, loc, i64Ty, f);
    } else {
      f = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                   rewriter.getI64IntegerAttr(firstConst));
    }
    Value rem = LLVM::SubOp::create(rewriter, loc, i64Ty, b, f);
    Value zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(0));
    Value capV = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(cap));
    rem = LLVM::SMaxOp::create(rewriter, loc, rem, zero);
    return LLVM::SMinOp::create(rewriter, loc, rem, capV);
  }

  // Interior-tile fullness guard for the rect-clamped DMA load: a uniform i1
  // that is true exactly when every bounded axis is fully in range, so the
  // source clamp + residual zero-fill are no-ops and we can fire the maskless
  // full-tile DMA (byte-identical to the --even path).
  //
  // Per axis the guard is `first + extent <= bound`. The K axis is bounded too
  // (A's mask is `(om<M)&(kk<K)`, B's is `(kk<K)&(on<N)`), so the LAST partial
  // K-tile — where `kk + BK > K` on a ragged-K shape — makes the K-axis guard
  // false and correctly keeps the clamped path. Uniform leaves (pipeliner
  // K-guards) must also be true. Returns null if any scalar cannot be remapped,
  // in which case the caller keeps the unconditional clamped DMA.
  static Value emitRectFullGuard(ConversionPatternRewriter &rewriter,
                                 Location loc, const RectMaskInfo &rectMask,
                                 int64_t tileRows, int64_t tileCols) {
    auto i64Ty = rewriter.getI64Type();
    auto axisGuard = [&](Value boundV, Value firstV, int64_t firstConst,
                         int64_t extent) -> Value {
      Value b = rewriter.getRemappedValue(boundV);
      if (!b)
        return nullptr;
      if (b.getType() != i64Ty)
        b = LLVM::SExtOp::create(rewriter, loc, i64Ty, b);
      Value f;
      if (firstV) {
        f = rewriter.getRemappedValue(firstV);
        if (!f)
          return nullptr;
        if (f.getType() != i64Ty)
          f = LLVM::SExtOp::create(rewriter, loc, i64Ty, f);
      } else {
        f = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                     rewriter.getI64IntegerAttr(firstConst));
      }
      Value ext = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(extent));
      Value hi = LLVM::AddOp::create(rewriter, loc, i64Ty, f, ext);
      return LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sle, hi,
                                  b);
    };

    Value guard;
    if (rectMask.hasRow) {
      guard = axisGuard(rectMask.rowBound, rectMask.rowFirst,
                        rectMask.rowFirstConst, tileRows);
      if (!guard)
        return nullptr;
    }
    if (rectMask.hasCol) {
      Value g = axisGuard(rectMask.colBound, rectMask.colFirst,
                          rectMask.colFirstConst, tileCols);
      if (!g)
        return nullptr;
      guard = guard ? (Value)LLVM::AndOp::create(rewriter, loc, guard, g) : g;
    }
    if (!guard)
      return nullptr;
    for (Value u : rectMask.uniforms) {
      Value ru = rewriter.getRemappedValue(u);
      if (!ru)
        return nullptr;
      guard = LLVM::AndOp::create(rewriter, loc, guard, ru);
    }
    return guard;
  }

  // Runtime tile-origin + row-stride extraction fallback.
  //
  // When the syntactic IR walk (extractAsyncCopyPtrInfo) cannot reconstruct a
  // uniform affine tile description, we derive the SAME two scalars the async
  // 2D intrinsic needs - a uniform tile-origin device pointer and a uniform
  // src row stride (bytes) - directly from the MATERIALIZED per-element pointer
  // tensor, which always exists. This needs no IR pattern matching and is exact
  // for ANY affine access (every real GEMM/conv/inductor matmul operand).
  //
  // How it stays uniform across the simdgroup:
  //   The intrinsic is cooperative; all lanes must pass the same origin. For an
  //   affine access ptr(row,col) = base + row*rowStrideBytes + col*elemBytes,
  //   every lane computes
  //     origin = ptrtoint(srcElems[k]) - (row_k*rowStrideBytes +
  //     col_k*elemBytes)
  //   where (row_k,col_k) is the FULL tile logical coord of that lane's element
  //   k (obtained at runtime from emitIndices, which already folds in the
  //   lane/warp base). The subtraction cancels the per-lane part, so all lanes
  //   land on ptrtoint(base) - the value is uniform BY CONSTRUCTION.
  //
  // How the runtime row stride is derived:
  //   Pick two registers k0,k1 of THIS thread whose compile-time intra-thread
  //   logical offsets differ by exactly one row and zero cols. Then
  //     rowStrideBytes = ptrtoint(srcElems[k1]) - ptrtoint(srcElems[k0])
  //   is exactly one row step in bytes, computed with no pattern matching.
  //   If no such single-thread row-adjacent register pair exists (each thread
  //   owns a single tile row), we cannot derive the stride locally and bail to
  //   the sync copy. Real GEMM/conv operands always own multiple rows.
  LogicalResult lowerAsyncFromRuntimePtrs(
      ttg::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, bool useRectClamp = false,
      const RectMaskInfo *rectMask = nullptr, Value llvmGuard = nullptr,
      bool warp0Fire = false) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto shape = srcTy.getShape();
    if (shape.size() != 2)
      return failure();

    Value llSrc = adaptor.getSrc();
    auto srcElems = unpackLLElements(loc, llSrc, rewriter);
    if (srcElems.empty())
      return failure();

    // Compile-time per-register intra-thread logical offsets (register order
    // matches srcElems and emitIndices).
    auto regOffsets = emitOffsetForLayout(srcTy.getEncoding(), srcTy);
    if (regOffsets.size() != srcElems.size())
      return failure();

    // Find a register pair (k0,k1) within THIS thread sharing a column but on
    // different rows. The row gap dr need not be 1 (blocked layouts hand a
    // thread rows like 0 and 8); we divide the byte delta by dr to recover one
    // row step. Pick the SMALLEST positive dr to minimize rounding exposure
    // (the access is affine so any dr is exact, but a small dr is robust).
    int kRow0 = -1, kRow1 = -1;
    int64_t rowGap = 0;
    for (unsigned a = 0; a < regOffsets.size(); a++) {
      for (unsigned b = 0; b < regOffsets.size(); b++) {
        if (a == b || regOffsets[a][1] != regOffsets[b][1])
          continue;
        int64_t dr = regOffsets[b][0] - regOffsets[a][0];
        if (dr > 0 && (kRow0 < 0 || dr < rowGap)) {
          kRow0 = a;
          kRow1 = b;
          rowGap = dr;
        }
      }
    }
    if (kRow0 < 0)
      return failure();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy1 = LLVMPointerType::get(ctx, 1); // device
    auto ptrTy3 = LLVMPointerType::get(ctx, 3); // threadgroup
    auto vec2i64Ty = VectorType::get({2}, i64Ty);

    unsigned elemBits = elemTy.getIntOrFloatBitWidth();
    unsigned elemBytes = elemBits / 8;
    int64_t tileRows = shape[0];
    int64_t tileCols = shape[1];
    int64_t tileWidthBytes = tileCols * elemBytes;

    // Runtime row stride in bytes = (ptr(k1) - ptr(k0)) / rowGap, where k0,k1
    // share a column and are rowGap rows apart.
    Value pk0 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, srcElems[kRow0]);
    Value pk1 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, srcElems[kRow1]);
    Value gapDelta = LLVM::SubOp::create(rewriter, loc, i64Ty, pk1, pk0);
    Value srcStrideBytes = gapDelta;
    if (rowGap != 1) {
      Value gapVal = LLVM::ConstantOp::create(
          rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(rowGap));
      srcStrideBytes =
          LLVM::SDivOp::create(rewriter, loc, i64Ty, gapDelta, gapVal);
    }

    // Full per-element tile logical coords for THIS thread (runtime SSA).
    TargetInfo targetInfo;
    auto indices = emitIndices(loc, rewriter, targetInfo, srcTy.getEncoding(),
                               srcTy, /*withCTAOffset=*/true);
    if (indices.size() != srcElems.size() || indices[0].size() != 2) {
      return failure();
    }

    // origin = ptrtoint(srcElems[0]) - (row_0*rowStrideBytes + col_0*elemBytes)
    // Uniform across the whole simdgroup by the affine cancellation above.
    Value row0 = indices[0][0];
    Value col0 = indices[0][1];
    if (row0.getType() != i64Ty)
      row0 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, row0);
    if (col0.getType() != i64Ty)
      col0 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, col0);
    Value elemBytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elemBytes));
    Value rowByteOff =
        LLVM::MulOp::create(rewriter, loc, i64Ty, row0, srcStrideBytes);
    Value colByteOff =
        LLVM::MulOp::create(rewriter, loc, i64Ty, col0, elemBytesVal);
    Value intraOff =
        LLVM::AddOp::create(rewriter, loc, i64Ty, rowByteOff, colByteOff);
    Value p0 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, srcElems[0]);
    Value originInt = LLVM::SubOp::create(rewriter, loc, i64Ty, p0, intraOff);
    Value srcBase = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy1, originInt);

    // Destination base pointer: shared memory object base.
    Value llDst = adaptor.getResult();
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, elemTy, rewriter);
    Value dstBase = smemObj.getBase();

    Value dstStrideBytes = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileWidthBytes));

    // Cross-warp copies fire from one simdgroup only: tid<32 issues the whole
    // tile, the zero-init event slot makes other warps' waits no-ops and the
    // staging barrier publishes.
    int64_t bandRows = tileRows;
    if (warp0Fire) {
      Value w0 = emitWarp0Pred(op, rewriter, loc);
      llvmGuard = llvmGuard
                      ? (Value)LLVM::AndOp::create(rewriter, loc, llvmGuard, w0)
                      : w0;
    }

    Value widthVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileWidthBytes));
    Value heightVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(bandRows));

    Value idx0 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(0));
    Value idx1 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(1));
    Value tileVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
    tileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty, tileVec,
                                            widthVal, idx0);
    tileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty, tileVec,
                                            heightVal, idx1);

    Value zeroI64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                             rewriter.getI64IntegerAttr(0));
    Value offsetVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
    offsetVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                              offsetVec, zeroI64, idx0);
    offsetVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                              offsetVec, zeroI64, idx1);

    Value one64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(1));
    Value clamp = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(0));

    auto asyncCopyFn = getOrCreateFn(
        mod, rewriter, "air.simdgroup_async_copy_2d.p3i8.p1i8", ptrTy3,
        {i64Ty, i64Ty, ptrTy3, i64Ty, i64Ty, vec2i64Ty, ptrTy1, i64Ty, i64Ty,
         vec2i64Ty, vec2i64Ty, i32Ty});

    Value evAlloca = createEventAlloca(op, rewriter);

    // Build the (residual-zero + optionally-guarded) async copy for a given
    // source tile. `clampTile` runs the rect clamp + residual zero-fill; the
    // full-tile arm passes the compile-const tileVec and skips both, emitting
    // exactly what the maskless --even path does.
    auto emitCopy = [&](bool clampTile, Value baseGuard) {
      Value srcTileVec = tileVec;
      Value guard = baseGuard;
      if (clampTile && useRectClamp && rectMask) {
        Value srcW = widthVal, srcH = heightVal;
        if (rectMask->hasCol) {
          Value colRem = emitRemaining(rewriter, loc, rectMask->colBound,
                                       rectMask->colFirst,
                                       rectMask->colFirstConst, tileCols);
          srcW =
              LLVM::MulOp::create(rewriter, loc, i64Ty, colRem, elemBytesVal);
        }
        if (rectMask->hasRow) {
          srcH = emitRemaining(rewriter, loc, rectMask->rowBound,
                               rectMask->rowFirst, rectMask->rowFirstConst,
                               tileRows);
        }
        srcTileVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
        srcTileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                                   srcTileVec, srcW, idx0);
        srcTileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                                   srcTileVec, srcH, idx1);
        Value srcHForZero = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileRows));
        emitResidualZero(rewriter, loc, mod, dstBase, tileRows, tileCols,
                         elemBytes, srcW, srcHForZero);
        Value z = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(0));
        Value wPos = LLVM::ICmpOp::create(rewriter, loc,
                                          LLVM::ICmpPredicate::sgt, srcW, z);
        Value hPos = LLVM::ICmpOp::create(rewriter, loc,
                                          LLVM::ICmpPredicate::sgt, srcH, z);
        Value nonEmpty = LLVM::AndOp::create(rewriter, loc, wPos, hPos);
        guard = guard
                    ? (Value)LLVM::AndOp::create(rewriter, loc, guard, nonEmpty)
                    : nonEmpty;
      }
      if (guard) {
        auto *parentBlock = rewriter.getInsertionBlock();
        auto insertPt = rewriter.getInsertionPoint();
        auto *thenBlock = rewriter.createBlock(
            parentBlock->getParent(), std::next(Region::iterator(parentBlock)));
        auto *afterBlock = rewriter.createBlock(
            parentBlock->getParent(), std::next(Region::iterator(thenBlock)));
        afterBlock->getOperations().splice(afterBlock->begin(),
                                           parentBlock->getOperations(),
                                           insertPt, parentBlock->end());
        rewriter.setInsertionPointToEnd(parentBlock);
        LLVM::CondBrOp::create(rewriter, loc, guard, thenBlock, afterBlock);
        rewriter.setInsertionPointToStart(thenBlock);
        Value event = LLVM::CallOp::create(
                          rewriter, loc, asyncCopyFn,
                          ValueRange{one64, one64, dstBase, dstStrideBytes,
                                     one64, tileVec, srcBase, srcStrideBytes,
                                     one64, srcTileVec, offsetVec, clamp})
                          .getResult();
        LLVM::StoreOp::create(rewriter, loc, event, evAlloca);
        LLVM::BrOp::create(rewriter, loc, ValueRange{}, afterBlock);
        rewriter.setInsertionPointToStart(afterBlock);
      } else {
        Value event = LLVM::CallOp::create(
                          rewriter, loc, asyncCopyFn,
                          ValueRange{one64, one64, dstBase, dstStrideBytes,
                                     one64, tileVec, srcBase, srcStrideBytes,
                                     one64, srcTileVec, offsetVec, clamp})
                          .getResult();
        LLVM::StoreOp::create(rewriter, loc, event, evAlloca);
      }
    };

    // Interior-tile fast path: when the rect mask proves the tile is fully in
    // bounds (incl. a non-partial K-tile) the clamp + residual zero are no-ops,
    // so fire the maskless full-tile DMA; edge tiles keep the clamped DMA.
    Value fullGuard =
        (useRectClamp && rectMask)
            ? emitRectFullGuard(rewriter, loc, *rectMask, tileRows, tileCols)
            : nullptr;
    if (fullGuard) {
      auto *parentBlock = rewriter.getInsertionBlock();
      auto insertPt = rewriter.getInsertionPoint();
      auto *fullBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(parentBlock)));
      auto *edgeBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(fullBlock)));
      auto *contBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(edgeBlock)));
      contBlock->getOperations().splice(contBlock->begin(),
                                        parentBlock->getOperations(), insertPt,
                                        parentBlock->end());
      rewriter.setInsertionPointToEnd(parentBlock);
      LLVM::CondBrOp::create(rewriter, loc, fullGuard, fullBlock, edgeBlock);

      rewriter.setInsertionPointToEnd(fullBlock);
      emitCopy(/*clampTile=*/false, llvmGuard);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, contBlock);

      rewriter.setInsertionPointToEnd(edgeBlock);
      emitCopy(/*clampTile=*/true, llvmGuard);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, contBlock);

      rewriter.setInsertionPointToStart(contBlock);
    } else {
      emitCopy(/*clampTile=*/true, llvmGuard);
    }

    // Token IS this copy's event slot; the matching async_wait waits on it.
    rewriter.replaceOp(op, evAlloca);
    return success();
  }

  LogicalResult
  matchAndRewrite(ttg::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());

    Value llMask = adaptor.getMask();

    // Try async DMA path: requires 2D and extractable stride.
    // Mask must be absent or a uniform splat (scalar boolean).
    auto shape = srcTy.getShape();
    bool canAsyncDMA = (shape.size() == 2);
    // A multi-warp-outer staged copy (warpsPerCTA[outer]>1) cannot fire from
    // every warp: identical whole-tile copies write-write race on the same
    // bytes. Fire from one simdgroup only (tid<32), the staging barrier
    // publishes.
    bool outerCrossWarp = asyncCopyOuterDimCrossWarp(op);

    // A non-uniform per-element mask is a real boundary predicate that zeros
    // out-of-range rows/cols of a PARTIAL tile. Our async 2D copy reads the
    // full tile (clamp is not wired to the real M/N extent), so dropping such a
    // mask is only safe when the tile is PROVEN fully in-bounds (block-aligned
    // shape, where the mask is always-true). Compute that proof now; it also
    // gates the boundary-wrap modulo below.
    bool tileFullyInBounds =
        allModuloBlockAligned(op.getSrc(), shape) && functionModuloIsSafe(op);

    // Mask handling. The async DMA takes no mask or a UNIFORM scalar
    // (all-true/all-false) mask gating the whole copy. A non-uniform mask is
    // dropped ONLY when the tile is proven in-bounds; otherwise refuse async
    // and fall back to the per-element sync copy (which honors the mask
    // exactly), preventing out-of-matrix reads on unaligned shapes.
    Value mlirMaskScalar;
    Value llvmMaskScalar;
    RectMaskInfo rectMask;
    bool useRectClamp = false;
    if (canAsyncDMA && llMask) {
      mlirMaskScalar = extractScalarMask(op.getMask());
      if (mlirMaskScalar) {
        llvmMaskScalar = rewriter.getRemappedValue(mlirMaskScalar);
      } else if (!tileFullyInBounds) {
        // Rectangular row/col bound mask with other=0: the DMA honors it
        // exactly by clamping the source tile and zero-filling (clamp mode 0).
        RectMaskInfo probe;
        bool otherIsZero = true;
        if (op.getOther())
          if (auto cst = op.getOther().getDefiningOp<arith::ConstantOp>())
            if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
              otherIsZero =
                  dense.isSplat() && dense.getSplatValue<APFloat>().isZero();
        auto remappable = [&](Value v) {
          return !v || rewriter.getRemappedValue(v) != nullptr;
        };
        // Residual zeroing stores f32; other element widths keep sync copy.
        if (!elemTy.isF32())
          otherIsZero = false;
        // The DMA engine mis-copies when the source row stride is not 64B
        // aligned (measured: K=72/1000 corrupt, 80/96/1024 exact). The proof
        // is stamped pre-conversion (constant stride or a tt.divisibility>=16
        // kernel-arg stride); without it the sync copy is the only sound
        // fallback, so refuse rect.
        if (otherIsZero && !rectStrideIs64B(op))
          otherIsZero = false;
        if (otherIsZero && matchRectMask(op.getMask(), probe) &&
            remappable(probe.rowBound) && remappable(probe.rowFirst) &&
            remappable(probe.colBound) && remappable(probe.colFirst)) {
          rectMask = probe;
          useRectClamp = true;
          // Uniform leaves (pipeliner K-guards) gate the whole copy.
          for (Value u : rectMask.uniforms) {
            Value lu = rewriter.getRemappedValue(u);
            if (!lu) {
              useRectClamp = false;
              break;
            }
            llvmMaskScalar =
                llvmMaskScalar ? (Value)LLVM::AndOp::create(rewriter, loc,
                                                            llvmMaskScalar, lu)
                               : lu;
          }
          if (useRectClamp && !op->hasAttr("applegpu.rect_dma"))
            useRectClamp = false;
        }
        if (!useRectClamp) {
          // Non-uniform boundary mask on a possibly-partial tile: async cannot
          // honor per-element bounds, so use the sync copy.
          canAsyncDMA = false;
          llvmMaskScalar = nullptr;
        }
      }
    }

    // Use AxisInfo to decide whether a boundary-wrap modulo (rm%M / rn%N) in
    // the source index is a no-op for this tile: if the source pointer tensor
    // is fully contiguous in its inner dim, the wrap never crosses a tile edge,
    // so the constant-stride DMA form is exact. This lets inductor's triton_mm
    // template (which always emits the % wrap) take the async path.
    bool allowModulo = false;
    if (canAsyncDMA && axisInfo) {
      AxisInfo *ai = axisInfo->getAxisInfo(op.getSrc());
      if (ai) {
        int innerDim = shape.size() - 1;
        if (ai->getContiguity(innerDim) >= shape[innerDim])
          allowModulo = true;
      }
    }
    // AxisInfo is built on the pre-conversion module and frequently returns
    // null for the async-copy source here. Fall back to an IR-based proof:
    // accept the boundary-wrap modulo only when every remsi/remui in the def
    // chain is annotated contiguous across the full tile extent (no element
    // wraps inside the tile). This is exact for aligned shapes and refuses
    // unaligned ones.
    if (canAsyncDMA && !allowModulo &&
        allModuloBlockAligned(op.getSrc(), shape))
      allowModulo = true;

    // Function-level live-wrap guard. The pipeliner hides the modulo in the
    // prologue, so neither allowModulo nor the per-op def-chain walk can see an
    // UNALIGNED wrap on the in-loop copy. If the enclosing kernel contains any
    // non-block-aligned remsi/remui, the wrap is live and BOTH the
    // affine-modulo and the runtime affine paths would silently miscompile;
    // force every copy in such a kernel onto the modulo-safe sync path.
    bool funcModuloSafe = functionModuloIsSafe(op);
    if (!funcModuloSafe)
      allowModulo = false;

    AsyncCopyPtrInfo ptrInfo;
    Value llvmStride;
    if (canAsyncDMA && useRectClamp) {
      // Rect-clamped DMA needs a 64B-aligned source row stride (the engine
      // mis-copies otherwise); the proof is the pre-conversion stamp.
      if (!rectStrideIs64B(op))
        canAsyncDMA = false;
    }
    if (canAsyncDMA) {
      if (extractAsyncCopyPtrInfo(op.getSrc(), ptrInfo, allowModulo)) {
        if (ptrInfo.strideConst != INT64_MIN)
          llvmStride = LLVM::ConstantOp::create(
              rewriter, loc, IntegerType::get(ctx, 32),
              rewriter.getI32IntegerAttr((int32_t)ptrInfo.strideConst));
        else
          llvmStride = rewriter.getRemappedValue(ptrInfo.stride);
        if (!llvmStride) {
          canAsyncDMA = false;
        }
      } else {
        canAsyncDMA = false;
      }
    }

    // Also need the LLVM base pointer and tile-origin offsets
    Value llvmBasePtr;
    Value llvmRowStart;
    Value llvmColStart;
    if (canAsyncDMA) {
      llvmBasePtr = rewriter.getRemappedValue(ptrInfo.basePtr);
      if (!llvmBasePtr) {
        canAsyncDMA = false;
      }
      if (ptrInfo.rowStart) {
        llvmRowStart = rewriter.getRemappedValue(ptrInfo.rowStart);
        if (!llvmRowStart) {
          canAsyncDMA = false;
        }
      }
      if (ptrInfo.colStart) {
        llvmColStart = rewriter.getRemappedValue(ptrInfo.colStart);
        if (!llvmColStart) {
          canAsyncDMA = false;
        }
      }
    }
    if (!canAsyncDMA) {
      // Affine-from-IR reconstruction failed. Try the runtime tile-origin /
      // row-stride extraction from the materialized pointer tensor, which
      // covers ANY affine access (all real GEMM/conv/inductor matmul shapes).
      //
      // Gate on a 2D tile AND a function with no LIVE wrap (funcModuloSafe).
      // The runtime origin/stride derivation assumes the access is globally
      // affine over the tile. That holds when the kernel has no modulo at all,
      // or every modulo is block-aligned (no element wraps inside any tile).
      // When a boundary wrap is LIVE (unaligned shape) the materialized
      // per-element pointers fold back at the tile edge, breaking affinity, so
      // we keep the modulo-safe sync copy. Correctness over coverage. Note the
      // pipeliner hides the wrap in the prologue, so a per-op def-chain check
      // on op.getSrc() is insufficient; only the function-level guard is sound.
      // The runtime async path reads the FULL tile with no per-element mask, so
      // it is only correct when the tile is proven fully in-bounds. A live
      // boundary (non-uniform mask that we could not drop above, i.e.
      // canAsyncDMA was cleared) or an unaligned shape must NOT take it, else a
      // partial edge tile reads past the matrix and corrupts the result.
      // A rect-matched boundary mask is honored exactly by the clamped src
      // tile + zero fill, so it does not count as a live boundary.
      bool noLiveBoundary =
          !llMask || mlirMaskScalar || tileFullyInBounds || useRectClamp;
      // The runtime DMA assumes contiguous columns (inner unit stride). For a
      // transposed/strided-view operand the inner dim is strided, so the
      // contiguous-row copy reads the tile transposed. Require a positive
      // unit-inner-stride proof; otherwise use the layout-exact sync copy.
      bool innerUnit = innerDimIsUnitStride(op.getSrc());
      bool runtimeSafe =
          (shape.size() == 2) && funcModuloSafe && noLiveBoundary && innerUnit;
      if (runtimeSafe) {
        if (succeeded(
                lowerAsyncFromRuntimePtrs(op, adaptor, rewriter, useRectClamp,
                                          useRectClamp ? &rectMask : nullptr,
                                          llvmMaskScalar, outerCrossWarp)))
          return success();
      }
      return lowerSyncCopy(op, adaptor, rewriter);
    }

    // ── Async DMA path via air.simdgroup_async_copy_2d ──

    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy1 = LLVMPointerType::get(ctx, 1); // device
    auto ptrTy3 = LLVMPointerType::get(ctx, 3); // threadgroup
    auto vec2i64Ty = VectorType::get({2}, i64Ty);

    // Element size in bytes
    unsigned elemBits = elemTy.getIntOrFloatBitWidth();
    unsigned elemBytes = elemBits / 8;

    // Tile geometry
    int64_t tileRows = shape[0];
    int64_t tileCols = shape[1];
    int64_t tileWidthBytes = tileCols * elemBytes;

    // Source stride in bytes: llvmStride (elements) * elemBytes
    // llvmStride may be i32 — extend to i64
    Value strideI64 = llvmStride;
    if (llvmStride.getType() != i64Ty) {
      strideI64 = LLVM::SExtOp::create(rewriter, loc, i64Ty, llvmStride);
    }
    Value elemBytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elemBytes));
    Value srcStrideBytes =
        LLVM::MulOp::create(rewriter, loc, i64Ty, strideI64, elemBytesVal);

    // Destination stride in bytes (TG is packed, no padding)
    Value dstStrideBytes = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileWidthBytes));

    // Source base pointer: compute UNIFORM tile origin from extracted scalars.
    //
    // air.simdgroup_async_copy_2d is a simdgroup-cooperative operation — all
    // threads in the simdgroup must pass the SAME base pointer (the tile's
    // top-left corner). We cannot use per-thread pointers from srcElems[0].
    //
    // tile_origin = basePtr + rowStart * stride + colStart
    // where basePtr, rowStart, colStart, stride are all scalar (uniform).
    Value srcBase = llvmBasePtr;
    if (llvmRowStart) {
      // GEP by rowStart * stride (in elements)
      Value rowOff = LLVM::MulOp::create(rewriter, loc, llvmRowStart.getType(),
                                         llvmRowStart, llvmStride);
      srcBase = LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), elemTy,
                                    srcBase, ArrayRef<LLVM::GEPArg>{rowOff});
    } else if (ptrInfo.rowStartConst != 0) {
      // Folded-constant first row: GEP by (rowStartConst * stride) elements.
      Value rc = LLVM::ConstantOp::create(
          rewriter, loc, llvmStride.getType(),
          rewriter.getIntegerAttr(llvmStride.getType(), ptrInfo.rowStartConst));
      Value rowOff = LLVM::MulOp::create(rewriter, loc, llvmStride.getType(),
                                         rc, llvmStride);
      srcBase = LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), elemTy,
                                    srcBase, ArrayRef<LLVM::GEPArg>{rowOff});
    }
    if (llvmColStart) {
      // GEP by colStart (in elements)
      srcBase =
          LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), elemTy, srcBase,
                              ArrayRef<LLVM::GEPArg>{llvmColStart});
    } else if (ptrInfo.colStartConst != 0) {
      // Folded-constant first col (the prefetched K-block offset): GEP by
      // colStartConst elements. THIS is the num_stages>=3 correctness fix.
      Value cc = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)ptrInfo.colStartConst));
      srcBase = LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), elemTy,
                                    srcBase, ArrayRef<LLVM::GEPArg>{cc});
    }

    // Destination base pointer: shared memory object base
    Value llDst = adaptor.getResult();
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, elemTy, rewriter);
    Value dstBase = smemObj.getBase();

    // Cross-warp copies fire from one simdgroup only (tid<32); zero-init
    // event slots keep other warps' waits no-ops, the staging barrier
    // publishes.
    int64_t bandRows = tileRows;
    if (outerCrossWarp) {
      Value w0 = emitWarp0Pred(op, rewriter, loc);
      llvmMaskScalar =
          llvmMaskScalar
              ? (Value)LLVM::AndOp::create(rewriter, loc, llvmMaskScalar, w0)
              : w0;
    }

    // Build tile size vectors: <width_bytes, height_rows>. height = bandRows so
    // each warp copies only its own band (== tileRows when not partitioned).
    Value widthVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileWidthBytes));
    Value heightVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(bandRows));

    // Source tile = <tileWidthBytes, tileRows>
    Value tileVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
    Value idx0 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(0));
    Value idx1 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(1));
    tileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty, tileVec,
                                            widthVal, idx0);
    tileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty, tileVec,
                                            heightVal, idx1);

    // Offset = <0, 0>
    Value zeroI64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                             rewriter.getI64IntegerAttr(0));
    Value offsetVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
    offsetVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                              offsetVec, zeroI64, idx0);
    offsetVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                              offsetVec, zeroI64, idx1);

    // sizeof=1, alignof=1 (byte-granularity copy)
    Value one64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(1));

    // clamp = 0
    Value clamp = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(0));

    // Declare air.simdgroup_async_copy_2d.p3i8.p1i8
    // Returns ptr addrspace(3) (event pointer)
    auto asyncCopyFn = getOrCreateFn(
        mod, rewriter, "air.simdgroup_async_copy_2d.p3i8.p1i8", ptrTy3,
        {i64Ty, i64Ty, // sizeof, alignof
         ptrTy3, i64Ty, i64Ty,
         vec2i64Ty, // dst, dstStride, dstElemStride, dstTile
         ptrTy1, i64Ty, i64Ty,
         vec2i64Ty,          // src, srcStride, srcElemStride, srcTile
         vec2i64Ty, i32Ty}); // offset, clamp

    // One event alloca per copy so concurrent A/B copies do not clobber a
    // shared slot (see createEventAlloca).
    Value evAlloca = createEventAlloca(op, rewriter);

    // Build the (rect clamp + residual zero) optionally-guarded async copy for
    // a source tile. The full-tile arm passes the compile-const tileVec and
    // skips the clamp + residual zero, emitting exactly the maskless --even
    // DMA.
    auto emitCopy = [&](bool clampTile, Value baseGuard) {
      Value srcTileVec = tileVec;
      Value guard = baseGuard;
      if (clampTile && useRectClamp) {
        Value srcW = widthVal, srcH = heightVal;
        if (rectMask.hasCol) {
          Value colRem =
              emitRemaining(rewriter, loc, rectMask.colBound, rectMask.colFirst,
                            rectMask.colFirstConst, tileCols);
          Value eb = LLVM::ConstantOp::create(
              rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elemBytes));
          srcW = LLVM::MulOp::create(rewriter, loc, i64Ty, colRem, eb);
        }
        if (rectMask.hasRow) {
          srcH =
              emitRemaining(rewriter, loc, rectMask.rowBound, rectMask.rowFirst,
                            rectMask.rowFirstConst, tileRows);
        }
        srcTileVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
        srcTileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                                   srcTileVec, srcW, idx0);
        srcTileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                                   srcTileVec, srcH, idx1);
        Value srcHForZero = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileRows));
        emitResidualZero(rewriter, loc, mod, dstBase, tileRows, tileCols,
                         elemBytes, srcW, srcHForZero);
        Value z = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(0));
        Value wPos = LLVM::ICmpOp::create(rewriter, loc,
                                          LLVM::ICmpPredicate::sgt, srcW, z);
        Value hPos = LLVM::ICmpOp::create(rewriter, loc,
                                          LLVM::ICmpPredicate::sgt, srcH, z);
        Value nonEmpty = LLVM::AndOp::create(rewriter, loc, wPos, hPos);
        guard = guard
                    ? (Value)LLVM::AndOp::create(rewriter, loc, guard, nonEmpty)
                    : nonEmpty;
      }
      if (guard) {
        auto *parentBlock = rewriter.getInsertionBlock();
        auto insertPt = rewriter.getInsertionPoint();
        auto *thenBlock = rewriter.createBlock(
            parentBlock->getParent(), std::next(Region::iterator(parentBlock)));
        auto *afterBlock = rewriter.createBlock(
            parentBlock->getParent(), std::next(Region::iterator(thenBlock)));
        afterBlock->getOperations().splice(afterBlock->begin(),
                                           parentBlock->getOperations(),
                                           insertPt, parentBlock->end());
        rewriter.setInsertionPointToEnd(parentBlock);
        LLVM::CondBrOp::create(rewriter, loc, guard, thenBlock, afterBlock);
        rewriter.setInsertionPointToStart(thenBlock);
        Value event = LLVM::CallOp::create(
                          rewriter, loc, asyncCopyFn,
                          ValueRange{one64, one64, dstBase, dstStrideBytes,
                                     one64, tileVec, srcBase, srcStrideBytes,
                                     one64, srcTileVec, offsetVec, clamp})
                          .getResult();
        LLVM::StoreOp::create(rewriter, loc, event, evAlloca);
        LLVM::BrOp::create(rewriter, loc, ValueRange{}, afterBlock);
        rewriter.setInsertionPointToStart(afterBlock);
      } else {
        Value event = LLVM::CallOp::create(
                          rewriter, loc, asyncCopyFn,
                          ValueRange{one64, one64, dstBase, dstStrideBytes,
                                     one64, tileVec, srcBase, srcStrideBytes,
                                     one64, srcTileVec, offsetVec, clamp})
                          .getResult();
        LLVM::StoreOp::create(rewriter, loc, event, evAlloca);
      }
    };

    // Interior-tile fast path: when the rect mask proves the tile fully in
    // bounds (incl. a non-partial K-tile) the clamp + residual zero are no-ops,
    // so fire the maskless full-tile DMA; edge tiles keep the clamped DMA.
    Value fullGuard = useRectClamp ? emitRectFullGuard(rewriter, loc, rectMask,
                                                       tileRows, tileCols)
                                   : nullptr;
    if (fullGuard) {
      auto *parentBlock = rewriter.getInsertionBlock();
      auto insertPt = rewriter.getInsertionPoint();
      auto *fullBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(parentBlock)));
      auto *edgeBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(fullBlock)));
      auto *contBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(edgeBlock)));
      contBlock->getOperations().splice(contBlock->begin(),
                                        parentBlock->getOperations(), insertPt,
                                        parentBlock->end());
      rewriter.setInsertionPointToEnd(parentBlock);
      LLVM::CondBrOp::create(rewriter, loc, fullGuard, fullBlock, edgeBlock);

      rewriter.setInsertionPointToEnd(fullBlock);
      emitCopy(/*clampTile=*/false, llvmMaskScalar);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, contBlock);

      rewriter.setInsertionPointToEnd(edgeBlock);
      emitCopy(/*clampTile=*/true, llvmMaskScalar);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, contBlock);

      rewriter.setInsertionPointToStart(contBlock);
    } else {
      emitCopy(/*clampTile=*/true, llvmMaskScalar);
    }

    // The token IS this copy's event slot. async_wait waits on exactly the
    // slots carried by its operand tokens; because the token is a scf.for
    // iter_arg the loop-carried value selects the correct alternating buffer
    // each iteration. (Token type converts to ptr addrspace(0); see the
    // AsyncToken type conversion registered on the converter.)
    rewriter.replaceOp(op, evAlloca);
    return success();
  }
};

struct AsyncCommitGroupOpAppleConversion
    : public ConvertOpToLLVMPattern<ttg::AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttg::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Forward the copy's event slot through the group token. Metal events are
    // per-copy, so a "group" is just its member copies' slots threaded onward;
    // the pipeliner emits exactly one copy per commit_group here. With no input
    // token (empty group) return a zero-initialized completed slot so a later
    // wait on it is a no-op.
    auto inTokens = adaptor.getInputTokens();
    if (!inTokens.empty()) {
      rewriter.replaceOp(op, inTokens.front());
      return success();
    }
    Value evSlot = createCompletedEventSlot(op, rewriter);
    rewriter.replaceOp(op, evSlot);
    return success();
  }
};

struct AsyncWaitOpAppleConversion
    : public ConvertOpToLLVMPattern<ttg::AsyncWaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttg::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVMVoidType::get(ctx);
    auto ptrTy0 = LLVMPointerType::get(ctx, 0);

    // Wait on EXACTLY the copies whose tokens this wait consumes. Each token
    // (adaptor operand) is that copy's event slot (ptr addrspace(0)); see the
    // AsyncToken type conversion and the async_copy lowering. The token is a
    // scf.for iter_arg, so the loop-carried value selects the correct
    // alternating (double/triple-buffered) buffer each iteration: this is the
    // num_stages>=3 correctness fix. A separate wait_simdgroup_events(1,
    // slot) per token keeps the slot scalar (Metal v1 bitcode handles arrays
    // of typed pointers poorly). Each slot is zero-initialized in the entry
    // block, so a token from a sync/skip path holds a complete event and the
    // wait is a real no-op, never a read of an uninitialized pointer. The
    // waits are emitted UNCONDITIONALLY: gating them on the module already
    // containing the DMA declaration is conversion-order dependent (a loop
    // wait lowers before the loop's copies; sync prologue copies declare
    // nothing) and dropping the wait while any DMA is live reads in-flight
    // slots.
    auto waitFn = getOrCreateFn(mod, rewriter, "air.wait_simdgroup_events",
                                voidTy, {i32Ty, ptrTy0});
    Value oneI32 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                            rewriter.getI32IntegerAttr(1));
    for (Value evSlot : adaptor.getAsyncToken())
      LLVM::CallOp::create(rewriter, loc, waitFn, ValueRange{oneI32, evSlot});

    // Always emit TG barrier (needed for both sync and async paths
    // to ensure shared memory visibility across all threads).
    // NOTE: flag 2 (threadgroup fence) suffices here because the only async
    // copies that reach a consumer are single-simdgroup or per-warp-EXCLUSIVE
    // partitioned (warp w reads only its own band). A device fence (flag 3)
    // would be required for a CROSS-warp-consumed async band, but those are
    // routed to the sync copy instead (see runtimeSafe), since even flag 3 only
    // reduces (~7% residual) the cross-simdgroup DMA visibility race.
    auto barrFn =
        getOrCreateFn(mod, rewriter, "air.wg.barrier", voidTy, {i32Ty, i32Ty});
    Value barrFlag = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                              rewriter.getI32IntegerAttr(2));
    Value barrScope = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                               rewriter.getI32IntegerAttr(1));
    LLVM::CallOp::create(rewriter, loc, barrFn,
                         ValueRange{barrFlag, barrScope});

    // The wait's result token (retToken) is consumed; after the wait all its
    // input copies are complete, so return a zero-initialized completed event
    // slot. Waiting on it later is a no-op.
    Value zeroToken = createCompletedEventSlot(op, rewriter);
    rewriter.replaceOp(op, zeroToken);
    return success();
  }
};

// ttg.barrier → air.wg.barrier with proper addrSpace→flag mapping.
struct AppleBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op,
                  triton::gpu::BarrierOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVMVoidType::get(ctx);
    auto barrFnTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);

    LLVMFuncOp barrFn;
    if (auto existing = mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier"))
      barrFn = existing;
    else {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      barrFn = LLVMFuncOp::create(rewriter, mod.getLoc(), "air.wg.barrier",
                                  barrFnTy, Linkage::External);
    }

    // AIR flag 1 = device, flag 2 = TG, flag 3 = both
    auto addrSpace = op.getAddrSpace();
    bool needsDevice =
        static_cast<uint32_t>(addrSpace) &
        (static_cast<uint32_t>(triton::gpu::AddrSpace::GlobalRead) |
         static_cast<uint32_t>(triton::gpu::AddrSpace::GlobalWrite));
    bool needsTG = static_cast<uint32_t>(addrSpace) &
                   static_cast<uint32_t>(triton::gpu::AddrSpace::Local);
    int flag = 0;
    if (needsDevice)
      flag |= 1;
    if (needsTG)
      flag |= 2;
    if (flag == 0)
      flag = 2;

    Value flags = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(flag));
    Value scope = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(1));
    LLVM::CallOp::create(rewriter, loc, barrFn, ValueRange{flags, scope});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTritonAppleGPUToLLVMPass
    : public PassWrapper<ConvertTritonAppleGPUToLLVMPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTritonAppleGPUToLLVMPass)

  StringRef getArgument() const override {
    return "convert-triton-apple-gpu-to-llvm";
  }
  StringRef getDescription() const override {
    return "Lower TritonGPU ops (Apple MPS) to LLVM IR";
  }

  void runOnOperation() override {
    auto mod = getOperation();
    auto *ctx = &getContext();

    TargetInfo targetInfo;
    TritonGPUToLLVMTypeConverter typeConverter(ctx, targetInfo);

    // Thread the async-DMA completion event through the !ttg.async.token SSA
    // value instead of throwing it away as i32 0. The Triton software
    // pipeliner double/triple-buffers the K-loop and carries "which buffer's
    // copy to wait on" through a token that is a scf.for iter_arg, so it
    // alternates buffers each iteration. Mapping the token to the event-slot
    // pointer (ptr addrspace(0), the thread-local alloca that holds the
    // simdgroup event handle) lets air.wait_simdgroup_events wait on exactly
    // the copy whose token the wait consumes; the loop-carried iter_arg then
    // selects the correct alternating buffer. Registered last so it overrides
    // the upstream i32 mapping (TypeConverter tries conversions newest-first).
    typeConverter.addConversion(
        [ctx](triton::gpu::AsyncTokenType type) -> std::optional<Type> {
          return LLVM::LLVMPointerType::get(ctx, 0);
        });

    // Fragment ABI: pure dot-chain #mma tensors carry their simdgroup_matrix
    // fragments as !llvm.struct<(vector<64xf32> x F)> so O3 keeps the
    // accumulator vectorized (no SROA-to-scalar → no occupancy collapse).
    // Gated by consumer analysis: #mma tensors fed to elementwise/broadcast/
    // slice (fla) fall through to the generic flat per-thread struct.
    auto fragmentEligible = computeFragmentEligibleTypes(mod);
    typeConverter.addConversion(
        [ctx, fragmentEligible](RankedTensorType type) -> std::optional<Type> {
          auto enc = dyn_cast<AppleMmaEncodingAttr>(type.getEncoding());
          if (!enc || !fragmentEligible.count(type))
            return std::nullopt;
          return getAppleMmaFragmentType(ctx, type, enc);
        });

    // Membar analysis: insert barriers between conflicting TG memory accesses.
    {
      ModuleAllocation allocation(mod);
      ModuleMembarAnalysis membarAnalysis(&allocation);
      membarAnalysis.run();
    }

    {
      int64_t smemSize = 0;
      if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.shared"))
        smemSize = attr.getValue().getZExtValue();
      if (smemSize == 0)
        smemSize = 8;
      int64_t smemAligned = (smemSize + 15) & ~15;
      mod->setAttr("ttg.shared",
                   IntegerAttr::get(IntegerType::get(ctx, 64), smemAligned));
      {
        OpBuilder b(mod.getBodyRegion());
        auto loc = mod.getLoc();
        auto elemTy = typeConverter.convertType(b.getIntegerType(8));
        auto arrayTy = LLVM::LLVMArrayType::get(elemTy, smemSize);
        LLVM::GlobalOp::create(b, loc, arrayTy, /*isConstant=*/false,
                               LLVM::Linkage::Internal, "global_smem",
                               /*value=*/Attribute(), /*alignment=*/16,
                               /*addrSpace=*/3u);
      }
    }

    // Pre-compute MMA threadgroup memory usage from tt.dot ops.
    // Each dot creates a __tg_dot_ab TG buffer with potential bank-conflict
    // padding (TG_PAD extra elements per row). Must account for the padded
    // size so ConvertLayoutOp can correctly budget the 32KB TG limit.
    // The IR pipeline coalesces all dot TG globals into one (taking the max),
    // so total MMA TG cost = max over all dots.
    // Set as module attribute so ConvertLayoutOp can account for it in
    // the 32KB TG budget when sizing its own TG buffers.
    {
      int64_t maxMmaBytes = 0;
      mod.walk([&](mlir::triton::DotOp dot) {
        auto aType = cast<RankedTensorType>(dot.getA().getType());
        auto cType = cast<RankedTensorType>(dot.getC().getType());
        unsigned dotRank = cType.getRank();
        int64_t K = aType.getShape()[dotRank - 1];
        int64_t N = cType.getShape()[dotRank - 1];
        int64_t maxStride = std::max(K, N);
        // Match the padding logic in DotOpToLLVM: pad when stride % 8 == 0
        // and padded buffer fits in 16KB budget.
        int64_t paddedMaxStride = maxStride;
        if (maxStride % 8 == 0) {
          int64_t candidateStride = maxStride + 4; // TG_PAD = 4
          if ((8 * candidateStride + 1) * 4 <= 16384)
            paddedMaxStride = candidateStride;
        }
        int64_t tgFloats = 8 * paddedMaxStride + 1;
        int64_t tgBytes = tgFloats * 4; // f32
        maxMmaBytes = std::max(maxMmaBytes, tgBytes);
      });
      if (maxMmaBytes > 0)
        mod->setAttr("ttg.mma_shared",
                     IntegerAttr::get(IntegerType::get(ctx, 64), maxMmaBytes));
    }

    // ttg.shared reserves global_smem for whatever the kernel's standard shared
    // path needs (reductions, scans, gathers, histograms). The AppleMma
    // convert_layout lowering allocates its own __tg_cvt_ buffers instead of
    // using global_smem, so in a kernel with no such consumer the reservation
    // is dead (llc strips it via use_empty), yet it would otherwise be
    // subtracted from the convert's TG budget and force it into many tiny
    // strips. Detect the live consumers once here (the ops are lowered by
    // sibling patterns in the same conversion run, so checking after would
    // race) and record whether global_smem is actually needed.
    {
      bool smemLive = false;
      mod.walk([&](Operation *o) {
        if (isa<mlir::triton::ReduceOp, mlir::triton::ScanOp,
                mlir::triton::GatherOp, mlir::triton::HistogramOp>(o))
          smemLive = true;
        // A software-pipelined float dot stages its A/B operands through a
        // ttg.local_alloc backed by global_smem (the async-copy buffers the
        // K-loop reads). Those GEPs keep the reservation live (llc cannot strip
        // it), so the convert budgeter must subtract that global_smem from its
        // 32KB threadgroup budget. Otherwise the output convert grants itself a
        // single full-tile __tg_cvt strip and the two together overflow (e.g.
        // 128x64x16 fp32: 16KB global_smem + 28KB f32 convert = 45KB).
        if (isa<ttg::LocalAllocOp>(o))
          smemLive = true;
        // An integer (int8) dot aliases its A/B scatter buffer into global_smem
        // (DotOpToLLVM getOrGrowSharedArena), so its GEPs keep the reservation
        // live and llc cannot strip it. The convert budgeter must subtract that
        // global_smem from its 32KB threadgroup budget, otherwise the output
        // convert over-allocates its __tg_cvt strip and the two together
        // overflow (e.g. 64x128x128 int8: 16KB global_smem + 24KB i32 convert =
        // 40KB).
        if (auto dot = dyn_cast<mlir::triton::DotOp>(o)) {
          auto aTy = cast<RankedTensorType>(dot.getA().getType());
          if (isa<IntegerType>(aTy.getElementType()))
            smemLive = true;
        }
      });
      mod->setAttr("applegpu.smem_live", BoolAttr::get(ctx, smemLive));
    }

    mod.walk([&](ttg::AsyncCopyGlobalToLocalOp cp) {
      AsyncCopyPtrInfo sp;
      if (!extractAsyncCopyPtrInfo(cp.getSrc(), sp, true))
        return;
      int64_t divBytes = 0;
      if (sp.strideConst != INT64_MIN)
        divBytes = sp.strideConst * 4;
      else if (auto arg = dyn_cast_or_null<BlockArgument>(sp.stride))
        if (auto fn =
                dyn_cast<FunctionOpInterface>(arg.getOwner()->getParentOp()))
          if (auto attr = fn.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                                           "tt.divisibility"))
            divBytes = attr.getInt() * 4;
      if (divBytes > 0 && divBytes % 64 == 0)
        cp->setAttr("applegpu.rect_stride_64b", UnitAttr::get(ctx));
    });

    RewritePatternSet patterns(ctx);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Apple func lowering: kernel args → addrspace(2)* + load, device fns
    // direct. Higher priority than shared FuncOpConversion (which is
    // NVIDIA-specific).
    patterns.add<AppleFuncOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 20));
    // Apple call lowering: no shared memory stack pointer appending.
    patterns.add<AppleCallOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 20));
    patterns.add<AppleReturnOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 20));

    // Shared Triton → LLVM patterns (handles device functions, non-kernel)
    mlir::triton::populateSPMDOpToLLVMPattern(
        typeConverter, patterns, targetInfo, patternBenefitDefault);
    mlir::triton::populateFuncOpConversionPattern(
        typeConverter, patterns, targetInfo, patternBenefitDefault);
    mlir::triton::populateMemoryOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, patternBenefitDefault);
    mlir::triton::populateMakeRangeOpToLLVMPattern(
        typeConverter, targetInfo, patterns, patternBenefitDefault);
    mlir::triton::populateControlFlowOpToLLVMPattern(
        typeConverter, patterns, targetInfo, patternBenefitDefault);
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, patternBenefitDefault);
    mlir::triton::populateReduceOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, patternBenefitDefault);
    mlir::triton::populateScanOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, patternBenefitDefault);

    mlir::triton::populateGatherOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, patternBenefitDefault);

    mlir::triton::populateHistogramOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, patternBenefitDefault);

    populateDotOpToLLVMPatterns(typeConverter, patterns, patternBenefitDefault);
    populateLoadStoreToLLVMPatterns(typeConverter, patterns,
                                    patternBenefitDefault);

    patterns.add<AppleBarrierOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));

    patterns.add<SafeStoreOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));
    patterns.add<SafeLoadOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));

    patterns.add<WarpIdOpConversion>(typeConverter, patternBenefitDefault);
    patterns.add<ApplePrintOpConversion>(typeConverter,
                                         patternBenefitDefault + 10);
    patterns.add<AppleAssertOpConversion>(typeConverter,
                                          patternBenefitDefault + 10);
    patterns.add<GetNumProgramsOpAppleConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));
    patterns.add<AtomicRMWOpAppleConversion>(typeConverter,
                                             patternBenefitDefault + 10);
    patterns.add<AtomicCASOpAppleConversion>(typeConverter,
                                             patternBenefitDefault + 10);
    patterns.add<ConvertLayoutOpAppleConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));
    patterns.add<AsyncCopyGlobalToLocalOpAppleConversion>(
        typeConverter, &axisInfoAnalysis,
        PatternBenefit(patternBenefitDefault + 10));
    patterns.add<AsyncCommitGroupOpAppleConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));
    patterns.add<AsyncWaitOpAppleConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));

    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    // Triton elementwise + view patterns override arith scalar patterns
    // for tensor types (higher benefit wins for same op).
    mlir::triton::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo,
        patternBenefitDefault + 1);
    mlir::triton::populateClampFOpToLLVMPattern(typeConverter, patterns,
                                                axisInfoAnalysis, targetInfo,
                                                patternBenefitDefault + 1);
    mlir::triton::populateMinMaxFOpToLLVMPattern(
        typeConverter, patterns, axisInfoAnalysis,
        /*hwNanPropagationSupported=*/true, patternBenefitDefault + 1);
#define POPULATE_FLOAT_OP(SRC_OP, DST_OP)                                      \
  patterns.add<mlir::triton::gpu::ElementwiseOpConversion<SRC_OP, DST_OP>>(    \
      typeConverter, axisInfoAnalysis, patternBenefitDefault + 1)
    POPULATE_FLOAT_OP(arith::AddFOp, LLVM::FAddOp);
    POPULATE_FLOAT_OP(arith::SubFOp, LLVM::FSubOp);
    POPULATE_FLOAT_OP(arith::MulFOp, LLVM::FMulOp);
    POPULATE_FLOAT_OP(arith::DivFOp, LLVM::FDivOp);
    POPULATE_FLOAT_OP(triton::PreciseDivFOp, LLVM::FDivOp);
    POPULATE_FLOAT_OP(triton::PreciseSqrtOp, LLVM::SqrtOp);
    POPULATE_FLOAT_OP(arith::ExtFOp, LLVM::FPExtOp);
    POPULATE_FLOAT_OP(arith::TruncFOp, LLVM::FPTruncOp);
    POPULATE_FLOAT_OP(arith::SIToFPOp, LLVM::SIToFPOp);
    POPULATE_FLOAT_OP(arith::FPToSIOp, LLVM::FPToSIOp);
#undef POPULATE_FLOAT_OP

    // Fragment-ABI f32 elementwise (per <64 x f32> fragment vector). Higher
    // benefit so it wins over the generic flat lowering when the operands are
    // fragment structs; defers otherwise.
    patterns.add<AppleMmaFragmentBinaryConversion<arith::AddFOp, LLVM::FAddOp>>(
        typeConverter, patternBenefitDefault + 5);
    patterns.add<AppleMmaFragmentBinaryConversion<arith::SubFOp, LLVM::FSubOp>>(
        typeConverter, patternBenefitDefault + 5);
    patterns.add<AppleMmaFragmentBinaryConversion<arith::MulFOp, LLVM::FMulOp>>(
        typeConverter, patternBenefitDefault + 5);
    patterns.add<AppleMmaFragmentBinaryConversion<arith::DivFOp, LLVM::FDivOp>>(
        typeConverter, patternBenefitDefault + 5);
    patterns.add<AppleMmaFragmentUnaryConversion<arith::NegFOp, LLVM::FNegOp>>(
        typeConverter, patternBenefitDefault + 5);
    patterns.add<AppleMmaFragmentUnaryConversion<math::ExpOp, LLVM::ExpOp>>(
        typeConverter, patternBenefitDefault + 5);
    patterns.add<AppleMmaFragmentTruncFConversion>(typeConverter,
                                                   patternBenefitDefault + 5);

    // Fragment-ABI integer/mask + view lowerings (kkt op-web). Higher benefit
    // than the generic flat view/arith lowerings so they fire on fragment
    // structs and defer (return failure) otherwise.
    patterns.add<AppleMmaExpandDimsConversion>(typeConverter,
                                               patternBenefitDefault + 5);
    patterns.add<AppleMmaBroadcastConversion>(typeConverter,
                                              patternBenefitDefault + 5);
    patterns.add<AppleMmaFragmentIntBinaryConversion<arith::CmpIOp>>(
        typeConverter, patternBenefitDefault + 5);
    patterns.add<AppleMmaFragmentIntBinaryConversion<arith::AndIOp>>(
        typeConverter, patternBenefitDefault + 5);
    patterns.add<AppleMmaFragmentSelectConversion>(typeConverter,
                                                   patternBenefitDefault + 5);

    // ExternElementwiseOp: lower libdevice calls to LLVM intrinsics.
    // Inductor emits libdevice.exp, libdevice.sin, etc. which on CUDA
    // link to __nv_* functions. On MPS, map to llvm.* intrinsics.
    patterns.add<ExternElementwiseOpAppleConversion>(
        typeConverter, axisInfoAnalysis, patternBenefitDefault + 10);

    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               patternBenefitDefault + 1);
    // Expand math::ErfOp to polynomial approximation before MathToLLVM
    // (there is no llvm.erf intrinsic — NVIDIA uses libdevice, we expand
    // inline)
    mlir::populatePolynomialApproximateErfPattern(patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    // Conversion target: everything must lower to LLVM dialect
    ConversionTarget target(*ctx);
    target.addIllegalDialect<triton::TritonDialect>();
    target.addIllegalDialect<triton::gpu::TritonGPUDialect>();
    target.addIllegalDialect<applegpu::TritonAppleGPUDialect>();
    target.addIllegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    // gpu.thread_id is emitted by shared make_range/SPMD patterns;
    // it will be lowered to air intrinsics by a subsequent pass.
    target.addLegalOp<mlir::gpu::ThreadIdOp>();
    target.addLegalOp<mlir::gpu::BlockDimOp>();
    target.addLegalOp<mlir::gpu::BarrierOp>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();

    // Async-wait cleanup. AsyncWaitOp lowers a wait_simdgroup_events per
    // token unconditionally because at pattern time it cannot know whether
    // any copy in the kernel takes the DMA path (a loop wait converts before
    // the loop's copies). When the finished module has no DMA call at all,
    // the waits guard nothing and the AIR JIT refuses to materialize the
    // intrinsic, so strip them and the dead declaration here.
    bool hasDMACall = false;
    mod.walk([&](LLVM::CallOp call) {
      if (call.getCallee() == "air.simdgroup_async_copy_2d.p3i8.p1i8")
        hasDMACall = true;
    });
    if (!hasDMACall) {
      SmallVector<LLVM::CallOp> waitCalls;
      mod.walk([&](LLVM::CallOp call) {
        if (call.getCallee() == "air.wait_simdgroup_events")
          waitCalls.push_back(call);
      });
      for (auto call : waitCalls)
        call.erase();
      for (StringRef fn : {"air.wait_simdgroup_events",
                           "air.simdgroup_async_copy_2d.p3i8.p1i8"})
        if (auto decl = mod.lookupSymbol<LLVMFuncOp>(fn))
          if (decl.use_empty())
            decl.erase();
    }

    // Fix up llvm.loop_annotation on llvm.br / llvm.cond_br ops.
    //
    // The ControlFlowToLLVM BranchOpLowering copies cf.br attrs via
    // setAttrs(getAttrDictionary()), but the loop_annotation attr name
    // in the CF dict is "llvm.loop_annotation" (discardable, dialect-
    // prefixed), while the LLVM BrOp's inherent property is named
    // "loop_annotation" (no prefix). setAttrs doesn't match them, so
    // the attr stays discardable and getLoopAnnotationAttr() returns
    // null, causing translateModuleToLLVMIR to drop the !llvm.loop
    // metadata.
    //
    // Walk all branch ops and move the discardable attr to the proper
    // inherent property.
    mod.walk([](LLVM::BrOp brOp) {
      if (auto attr = brOp->getAttrOfType<LLVM::LoopAnnotationAttr>(
              "llvm.loop_annotation")) {
        brOp.setLoopAnnotationAttr(attr);
        brOp->removeDiscardableAttr("llvm.loop_annotation");
      }
    });
    mod.walk([](LLVM::CondBrOp brOp) {
      if (auto attr = brOp->getAttrOfType<LLVM::LoopAnnotationAttr>(
              "llvm.loop_annotation")) {
        brOp.setLoopAnnotationAttr(attr);
        brOp->removeDiscardableAttr("llvm.loop_annotation");
      }
    });

    // Cooperative air.* declarations must reach the LLVM mid-end marked
    // `convergent` (barriers and event waits also `noduplicate`), or its
    // CFG transforms may sink/duplicate them into divergent control flow
    // and desynchronize the threadgroup.
    mod.walk([](LLVMFuncOp fn) {
      if (!fn.isExternal())
        return;
      StringRef name = fn.getName();
      if (!name.starts_with("air."))
        return;
      bool isBarrierLike =
          name.contains("barrier") || name.contains("wait_simdgroup");
      if (!isBarrierLike && !name.contains("simdgroup"))
        return;
      SmallVector<Attribute> pass;
      if (auto existing = fn.getPassthroughAttr())
        pass.append(existing.begin(), existing.end());
      auto addAttr = [&](StringRef a) {
        for (Attribute e : pass)
          if (auto s = dyn_cast<StringAttr>(e))
            if (s.getValue() == a)
              return;
        pass.push_back(StringAttr::get(fn.getContext(), a));
      };
      addAttr("convergent");
      if (isBarrierLike)
        addAttr("noduplicate");
      fn.setPassthroughAttr(ArrayAttr::get(fn.getContext(), pass));
    });
  }
};

// ── LowerGPUToAirPass ─────────────────────────────────────────────────────
//
// Converts remaining gpu.thread_id / gpu.block_dim ops (emitted by shared
// Triton patterns like make_range / SPMD) to air intrinsics / constants so
// the MLIR module is pure LLVM dialect before llvm::toModule().
//
//   gpu.thread_id x  →  call @air.dispatch_thread_id[0]() : i32, index_cast
//   gpu.thread_id y/z → arith.constant 0 : index
//   gpu.block_dim x  →  arith.constant <numThreads> : index   (from module
//   attr) gpu.block_dim y/z → arith.constant 1 : index
//
struct LowerGPUToAirPass
    : public PassWrapper<LowerGPUToAirPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGPUToAirPass)

  StringRef getArgument() const override { return "lower-gpu-to-air"; }
  StringRef getDescription() const override {
    return "Lower gpu.thread_id / gpu.block_dim to air intrinsics / constants";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto *ctx = &getContext();
    auto i32Ty = IntegerType::get(ctx, 32);

    // Declare air.thread_position_in_threadgroup once at module start.
    // Returns [3 x i32]; we extractvalue index 0 for the flat thread ID.
    // _add_air_metadata() rewrites this call+extractvalue pattern to an arg.
    auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
    auto tidFnName = StringRef("air.thread_position_in_threadgroup");
    auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
    if (!mod.lookupSymbol<LLVMFuncOp>(tidFnName)) {
      OpBuilder b(mod.getBodyRegion());
      b.setInsertionPointToStart(mod.getBody());
      LLVMFuncOp::create(b, mod.getLoc(), tidFnName, tidFnTy,
                         Linkage::External);
    }
    auto tidFn = mod.lookupSymbol<LLVMFuncOp>(tidFnName);

    // Read total thread count from module attributes for gpu.block_dim.
    int64_t threadsPerWarp = 32;
    int64_t numWarps = 4;
    if (auto a = mod->getAttrOfType<IntegerAttr>("ttg.threads-per-warp"))
      threadsPerWarp = a.getInt();
    if (auto a = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
      numWarps = a.getInt();
    int64_t totalThreads = threadsPerWarp * numWarps;

    IRRewriter rewriter(ctx);

    // Walk and replace gpu.thread_id / gpu.block_dim
    // gpu.thread_id/block_dim return `index` type. Downstream users (e.g.
    // make_range) have already been lowered to LLVM i64/i32 ops by this
    // point. We need to produce a value of the same `index` type and let
    // the existing index-to-llvm lowering handle it — but that already
    // ran. So we emit LLVM ops directly:
    //   gpu.thread_id x → llvm.call @air.dispatch_thread_id[0]() → i32
    //                   → llvm.zext i32 → i64  (index = i64 in LLVM)
    //   gpu.thread_id y/z → llvm.mlir.constant(0 : i64)
    //   gpu.block_dim x  → llvm.mlir.constant(totalThreads : i64)
    //   gpu.block_dim y/z → llvm.mlir.constant(1 : i64)
    //
    // The `index` type maps to i64 in the LLVM type system (index-bitwidth=0
    // means native pointer width = 64-bit on Apple Silicon).
    auto i64Ty = IntegerType::get(ctx, 64);

    mod.walk([&](Operation *op) {
      rewriter.setInsertionPoint(op);
      auto loc = op->getLoc();

      if (auto tidOp = dyn_cast<mlir::gpu::ThreadIdOp>(op)) {
        Value replacement;
        if (tidOp.getDimension() == mlir::gpu::Dimension::x) {
          Value tidStruct =
              LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{})
                  .getResult();
          Value i32val = LLVM::ExtractValueOp::create(
              rewriter, loc, i32Ty, tidStruct, ArrayRef<int64_t>{0});
          // Extend i32 → i64 to match `index` type (pointer width on Apple
          // Silicon). Use SExt to produce a single instruction without an
          // intermediate SSA: wrap in a struct-free zext inline by using the
          // i64 directly. Actually: emit only ExtractValue (i32) then trunc or
          // zext as needed. The users of gpu.thread_id have already been
          // lowered to expect i64 (via index_to_llvm). Emit zext to match. The
          // extra SSA is OK since _add_air_metadata's renumbering now correctly
          // handles it.
          replacement = LLVM::ZExtOp::create(rewriter, loc, i64Ty, i32val);
        } else {
          replacement = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                                 rewriter.getI64IntegerAttr(0));
        }
        rewriter.replaceOp(op, replacement);
        return;
      }

      if (auto bdOp = dyn_cast<mlir::gpu::BlockDimOp>(op)) {
        int64_t val =
            (bdOp.getDimension() == mlir::gpu::Dimension::x) ? totalThreads : 1;
        Value replacement = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(val));
        rewriter.replaceOp(op, replacement);
        return;
      }

      if (isa<mlir::gpu::BarrierOp>(op)) {
        auto voidTy = LLVMVoidType::get(ctx);
        auto barrFnTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
        LLVMFuncOp barrFn;
        if (auto existing = mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier"))
          barrFn = existing;
        else {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(mod.getBody());
          barrFn = LLVMFuncOp::create(rewriter, mod.getLoc(), "air.wg.barrier",
                                      barrFnTy, Linkage::External);
        }
        Value flags = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                               rewriter.getI32IntegerAttr(2));
        Value scope = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                               rewriter.getI32IntegerAttr(1));
        LLVM::CallOp::create(rewriter, loc, barrFn, ValueRange{flags, scope});
        rewriter.eraseOp(op);
        return;
      }
    });
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createConvertTritonAppleGPUToLLVMPass() {
  return std::make_unique<ConvertTritonAppleGPUToLLVMPass>();
}

std::unique_ptr<mlir::Pass> createLowerGPUToAirPass() {
  return std::make_unique<LowerGPUToAirPass>();
}

void registerTritonAppleGPUToLLVMPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<ConvertTritonAppleGPUToLLVMPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<LowerGPUToAirPass>();
  });
}

} // namespace mlir::triton::applegpu
