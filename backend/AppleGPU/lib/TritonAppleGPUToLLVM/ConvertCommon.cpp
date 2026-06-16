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
#include "ConvertCommon.h"

namespace mlir::triton::applegpu {

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
namespace ttg = mlir::triton::gpu;

// ── Fragment-ABI eligibility analysis ─────────────────────────────────────
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

llvm::DenseSet<Type> computeFragmentEligibleTypes(ModuleOp mod) {
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
      if (isAccumTruncEpilogue(def))
        poison(cast<arith::TruncFOp>(def).getIn());
    }
    for (OpOperand &use : v.getUses()) {
      if (isFragmentOp(use.getOwner()))
        for (Value r : use.getOwner()->getResults())
          poison(r);
      if (isAccumTruncEpilogue(use.getOwner()))
        poison(use.getOwner()->getResult(0));
    }
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

// ── Cross-pattern lowering helpers ────────────────────────────────────────
Value emitAppleRedundantThreadPredicate(
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

Value maybeAnd(ConversionPatternRewriter &rewriter, Location loc,
                      Value a, Value b) {
  if (!a)
    return b;
  if (!b)
    return a;
  return LLVM::AndOp::create(rewriter, loc, a, b);
}

Value emitFragShuffle(ConversionPatternRewriter &rewriter, Location loc,
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
Value emitLaneId(ConversionPatternRewriter &rewriter, Location loc,
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

bool extractFirstElemConst(Value tensor, int64_t &out, bool peelModulo) {
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
Value extractFirstElemScalar(Value tensor, bool peelModulo) {
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

// Interior-tile fast path for masked stores. A GEMM C-store carries a
// rectangular boundary mask `(om[:,None] < M) & (on[None,:] < N)`. For
// provably-full interior tiles the guard `(rowFirst + BM) <= rowBound &&
// (colFirst + BN) <= colBound` lets the caller branch to an unmasked store
// instead of a 64-way predicated chain. CONSERVATIVE: any unrecognized leaf
// makes this return null and the store keeps the exact existing masked path.
Value computeRectStoreFullGuard(triton::StoreOp op,
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

} // namespace mlir::triton::applegpu
