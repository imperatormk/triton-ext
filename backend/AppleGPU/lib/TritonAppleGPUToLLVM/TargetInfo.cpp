// TargetInfo.cpp - Apple MPS TargetInfoBase implementation
//
// programId = air.threadgroup_position_in_grid (CTA/block index), NOT
// air.thread_position_in_grid (flat global thread ID).

#include "TritonAppleGPUToLLVM/TargetInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::triton;

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::applegpu {

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  if (mlir::isa<ttg::SharedMemorySpaceAttr>(addressSpace))
    return 3; // threadgroup
  return 0;
}

static LLVMFuncOp getOrInsertAirIntrinsic3xi32(RewriterBase &rewriter,
                                               ModuleOp mod, StringRef name) {
  if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto *ctx = mod.getContext();
  auto retTy = LLVMArrayType::get(IntegerType::get(ctx, 32), 3);
  auto fnTy = LLVMFunctionType::get(retTy, {}, false);
  return LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                            Linkage::External);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  auto *ctx = moduleOp.getContext();
  auto fn = getOrInsertAirIntrinsic3xi32(rewriter, moduleOp,
                                         "air.threadgroup_position_in_grid");
  auto retTy = LLVMArrayType::get(IntegerType::get(ctx, 32), 3);
  Value gridVec =
      LLVM::CallOp::create(rewriter, loc, fn, ValueRange{}).getResult();
  int idx = static_cast<int>(axis);
  return LLVM::ExtractValueOp::create(rewriter, loc, IntegerType::get(ctx, 32),
                                      gridVec, ArrayRef<int64_t>{idx});
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  // No correct AIR simd-vote/ballot lowering yet; a constant return computes
  // the wrong mask silently, so fail loudly instead. TODO: implement via Metal
  // simd_vote / simd_ballot when a real use case appears.
  mlir::emitError(loc) << "ballot is not implemented on the Apple GPU backend; "
                          "this kernel uses an unsupported warp-vote operation";
  return LLVM::UndefOp::create(rewriter, loc, type);
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace targets) const {
  // air.wg.barrier(i32 flags, i32 mem_scope); mem_scope=1 (threadgroup)
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
  auto voidTy = LLVMVoidType::get(rewriter.getContext());
  auto fnTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
  LLVMFuncOp fn;
  if (auto existing = mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier"))
    fn = existing;
  else {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    fn = LLVMFuncOp::create(rewriter, mod.getLoc(), "air.wg.barrier", fnTy,
                            Linkage::External);
  }
  // air.wg.barrier flags: 1 = device memory fence, 2 = threadgroup memory fence
  // Use flag 2 when the barrier needs to order threadgroup (shared) memory
  bool needsTGFence = static_cast<uint32_t>(targets) &
                      static_cast<uint32_t>(triton::gpu::AddrSpace::Local);
  int flag = needsTGFence ? 2 : 1;
  Value flags = arith::ConstantIntOp::create(rewriter, loc, flag, 32);
  Value scope = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
  LLVM::CallOp::create(rewriter, loc, fn, ValueRange{flags, scope});
}

void TargetInfo::clusterBarrier(Location loc, RewriterBase &rewriter,
                                Operation *sourceOp) const {
  // Apple has no cluster/CGA level; a cluster barrier degenerates to a CTA
  // (threadgroup) barrier. sourceOp is unused (upstream uses it only for
  // multi-CTA cluster lowering).
  barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  // Apple simdgroups do NOT auto-order threadgroup memory across lanes (unlike
  // NVIDIA lockstep warps), so emit a real TG-scoped fence; without it
  // single-warp MMA tiles read stale TG slots nondeterministically.
  barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
}

Value TargetInfo::getGlobalTimer(RewriterBase &rewriter, Location loc) const {
  // Apple AIR exposes no globaltimer special register; the timer intrinsic is
  // only reachable from profiling ops that never appear in an MPS kernel.
  auto i64Ty = rewriter.getIntegerType(64);
  return LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                  rewriter.getI64IntegerAttr(0));
}

StringRef TargetInfo::getAtomicSyncScope(MemSyncScope scope) const {
  switch (scope) {
  case MemSyncScope::CTA:
    return "threadgroup";
  case MemSyncScope::GPU:
    return "device";
  case MemSyncScope::SYSTEM:
    return {};
  }
  llvm_unreachable("unknown memory synchronization scope");
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              Value ctaId, Value val, Value pred) const {
  assert(!ctaId && "Apple does not support cross-CTA transfers");
  if (pred) {
    auto *curBlock = rewriter.getInsertionBlock();
    auto curPoint = rewriter.getInsertionPoint();
    auto *endBlock = curBlock->splitBlock(curPoint);
    auto *thenBlock = rewriter.createBlock(endBlock);
    rewriter.setInsertionPointToEnd(curBlock);
    LLVM::CondBrOp::create(rewriter, loc, pred, thenBlock, endBlock);
    rewriter.setInsertionPointToEnd(thenBlock);
    LLVM::StoreOp::create(rewriter, loc, val, ptr);
    LLVM::BrOp::create(rewriter, loc, endBlock);
    rewriter.setInsertionPointToStart(endBlock);
  } else {
    LLVM::StoreOp::create(rewriter, loc, val, ptr);
  }
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              Value ctaId, Type elemTy, Value pred,
                              Operation *localLoadOp) const {
  assert(!ctaId && "Apple does not support cross-CTA transfers");
  return LLVM::LoadOp::create(rewriter, loc, elemTy, ptr).getResult();
}

// Get or insert an AIR simd_shuffle intrinsic decl, e.g.
// air.simd_shuffle_xor.f32(float, i16) or .s.i32(i32, i16) for integers.
static LLVMFuncOp getOrInsertShuffleIntrinsic(RewriterBase &rewriter,
                                              ModuleOp mod, StringRef kind,
                                              Type valTy) {
  std::string base = "air.simd_shuffle";
  if (!kind.empty())
    base += "_" + kind.str();
  std::string name = base + ".";
  if (valTy.isF32())
    name += "f32";
  else if (valTy.isF16())
    name += "f16";
  else if (valTy.isBF16())
    name += "bf16";
  else if (valTy.isF64())
    name += "f64";
  else if (valTy.isInteger(32))
    name = base + ".s.i32";
  else if (valTy.isInteger(16))
    name = base + ".s.i16";
  else if (valTy.isInteger(64))
    name = base + ".s.i64";
  else
    llvm_unreachable("unsupported shuffle type");

  if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto i16Ty = IntegerType::get(mod.getContext(), 16);
  auto fnTy = LLVMFunctionType::get(valTy, {valTy, i16Ty}, false);
  auto fn =
      LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy, Linkage::External);
  // A SIMD shuffle is cross-lane (convergent): without `convergent` the mid-end
  // (SLP vectorizer at O1+) reorders/duplicates the calls, changing the
  // participating lane set and corrupting cross-lane reductions. `noduplicate`
  // forbids cloning. Mirrors what Apple's `metal` frontend emits.
  auto *ctx = mod.getContext();
  SmallVector<Attribute> pass{
      StringAttr::get(ctx, "convergent"), StringAttr::get(ctx, "noduplicate"),
      StringAttr::get(ctx, "nounwind"), StringAttr::get(ctx, "willreturn")};
  fn.setPassthroughAttr(ArrayAttr::get(ctx, pass));
  return fn;
}

// Emit a shuffle call, handling bitcast for types that need i32 shuffle
static Value emitShuffle(RewriterBase &rewriter, Location loc, Value val,
                         Value offset, StringRef kind) {
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Type valTy = val.getType();

  // Narrower than 32 bits: extend to i32, shuffle, truncate back.
  if (valTy.isF16() || valTy.isBF16() || valTy.isInteger(16) ||
      valTy.isInteger(8) || valTy.isInteger(1)) {
    auto i32Ty = IntegerType::get(mod.getContext(), 32);
    Value extended;
    if (valTy.isInteger())
      extended = LLVM::ZExtOp::create(rewriter, loc, i32Ty, val);
    else {
      auto i16Ty = IntegerType::get(mod.getContext(), 16);
      Value asInt = LLVM::BitcastOp::create(rewriter, loc, i16Ty, val);
      extended = LLVM::ZExtOp::create(rewriter, loc, i32Ty, asInt);
    }
    auto fn = getOrInsertShuffleIntrinsic(rewriter, mod, kind, i32Ty);
    Value result =
        LLVM::CallOp::create(rewriter, loc, fn, ValueRange{extended, offset})
            .getResult();
    Value truncated = LLVM::TruncOp::create(
        rewriter, loc, IntegerType::get(mod.getContext(), 16), result);
    if (valTy.isInteger())
      return LLVM::TruncOp::create(rewriter, loc, valTy, truncated);
    return LLVM::BitcastOp::create(rewriter, loc, valTy, truncated);
  }

  auto fn = getOrInsertShuffleIntrinsic(rewriter, mod, kind, valTy);
  return LLVM::CallOp::create(rewriter, loc, fn, ValueRange{val, offset})
      .getResult();
}

static Value makeI16Const(RewriterBase &rewriter, Location loc, int val) {
  auto i16Ty = IntegerType::get(rewriter.getContext(), 16);
  return LLVM::ConstantOp::create(rewriter, loc, i16Ty,
                                  rewriter.getI16IntegerAttr(val));
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return emitShuffle(rewriter, loc, val, makeI16Const(rewriter, loc, i), "xor");
}
Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return emitShuffle(rewriter, loc, val, makeI16Const(rewriter, loc, i), "up");
}
Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return emitShuffle(rewriter, loc, val, makeI16Const(rewriter, loc, i), "");
}
Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  auto i16Ty = IntegerType::get(rewriter.getContext(), 16);
  Value offset = LLVM::TruncOp::create(rewriter, loc, i16Ty, i);
  return emitShuffle(rewriter, loc, val, offset, "");
}
Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  // Byte permute matching NVIDIA prmt index mode. Source bytes concat {b,a}:
  // 0..3 = a low..high, 4..7 = b; result byte k picks source byte (nibble_k &
  // 7). AIR has no prmt intrinsic, so synthesize with portable 32-bit bit ops.
  auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
  auto i64Ty = IntegerType::get(rewriter.getContext(), 64);
  auto cst32 = [&](int v) {
    return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                    rewriter.getI32IntegerAttr(v));
  };
  auto cst64 = [&](int64_t v) {
    return LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                    rewriter.getI64IntegerAttr(v));
  };

  Value aw = LLVM::ZExtOp::create(rewriter, loc, i64Ty, a);
  Value bw = LLVM::ZExtOp::create(rewriter, loc, i64Ty, b);
  Value concat = LLVM::OrOp::create(
      rewriter, loc, aw, LLVM::ShlOp::create(rewriter, loc, bw, cst64(32)));

  Value result = cst32(0);
  for (int k = 0; k < 4; ++k) {
    Value shifted = LLVM::LShrOp::create(rewriter, loc, selector, cst32(4 * k));
    Value nib = LLVM::AndOp::create(rewriter, loc, shifted, cst32(0x7));
    Value byteShift = LLVM::ShlOp::create(rewriter, loc, nib, cst32(3));
    Value byteShift64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, byteShift);
    Value srcByte64 = LLVM::AndOp::create(
        rewriter, loc, LLVM::LShrOp::create(rewriter, loc, concat, byteShift64),
        cst64(0xFF));
    Value srcByte = LLVM::TruncOp::create(rewriter, loc, i32Ty, srcByte64);
    Value placed = LLVM::ShlOp::create(rewriter, loc, srcByte, cst32(8 * k));
    result = LLVM::OrOp::create(rewriter, loc, result, placed);
  }
  // Bitcast back to a's type to keep the contract (shuffle path passes i32).
  if (a.getType() != i32Ty)
    return LLVM::BitcastOp::create(rewriter, loc, a.getType(), result);
  return result;
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned reduceLaneIdMask) const {
  // false declines: the generic ReduceOpToLLVM does the reduction via
  // shuffleXor (correct above). Not a stub; there's no single-instruction
  // warp-reduce intrinsic to special-case.
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  if (resultElementTy.isInteger(32))
    return "__mulhi";
  if (resultElementTy.isInteger(64))
    return "__mul64hi";
  llvm_unreachable("unsupported mulhi type");
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int formatStrByteCount, ValueRange args,
                        ArrayRef<bool> isSigned) const {}
void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  LLVM::Trap::create(rewriter, loc);
}

} // namespace mlir::triton::applegpu
