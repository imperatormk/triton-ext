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

// Masked store via conditional branch (if (mask) store), not read-modify-write:
// masked-out pointers can alias other threads' valid addresses (M < RBLOCK,
// overlapping row strides), so the RMW select form races and corrupts data.
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

    // Replicated tensors (more threads than elements) make redundant threads
    // race on the same destination pointer; predicate so only the canonical
    // owner stores, and skip register-replicated copies.
    Value threadPred;
    uint32_t regMask = 0;
    if (isa<RankedTensorType>(op.getPtr().getType())) {
      auto mod = op->getParentOfType<ModuleOp>();
      auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
      threadPred =
          emitAppleRedundantThreadPredicate(freeVarMasks, rewriter, loc, mod);
      regMask = freeVarMasks[StringAttr::get(rewriter.getContext(), "reg")];
    }

    // dropElemMask drops the boundary mask (tile provably full); the
    // redundant-thread predicate still applies (it guards replicas, not
    // bounds).
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

    // Interior-tile fast path: when a full rectangular tile is provably
    // (runtime, simdgroup-uniform) in bounds, branch to a maskless store.
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

// Masked tt.load lowering.
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
        // Scalar pointer is always valid: load unconditionally, select result.
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

} // anonymous namespace

void populateMemoryOpPatterns(LLVMTypeConverter &typeConverter,
                              RewritePatternSet &patterns) {
  patterns.add<SafeStoreOpConversion>(
      typeConverter, PatternBenefit(patternBenefitDefault + 10));
  patterns.add<SafeLoadOpConversion>(
      typeConverter, PatternBenefit(patternBenefitDefault + 10));
}

} // namespace mlir::triton::applegpu
