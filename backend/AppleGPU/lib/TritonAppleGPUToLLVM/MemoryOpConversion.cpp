#include "ConvertCommon.h"
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

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
namespace ttg = mlir::triton::gpu;

namespace {

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
        // Branch-free masked load: the branch form's merge block let O3
        // SimplifyCFG drop the masked-store guards (div7 corruption). Load
        // unconditionally and discard masked-out lanes with a value select;
        // an OOB device load on AGX returns garbage but does not fault.
        Value val =
            LLVM::LoadOp::create(rewriter, loc, sTy.getBody()[i], ptrs[i]);
        val = LLVM::SelectOp::create(rewriter, loc, masks[i], val, other);
        loaded.push_back(val);
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

} // anonymous namespace

void populateMemoryOpPatterns(LLVMTypeConverter &typeConverter,
                              RewritePatternSet &patterns) {
  patterns.add<SafeStoreOpConversion>(
      typeConverter, PatternBenefit(patternBenefitDefault + 10));
  patterns.add<SafeLoadOpConversion>(
      typeConverter, PatternBenefit(patternBenefitDefault + 10));
}

} // namespace mlir::triton::applegpu
