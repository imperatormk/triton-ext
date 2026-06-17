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

// Grid dimension (number of threadgroups) for the given axis, via
// @air.threadgroups_per_grid().
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

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

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

    SmallVector<Type> resultTypes;
    for (auto resTy : callOp.getResultTypes()) {
      Type converted = getTypeConverter()->convertType(resTy);
      if (!converted)
        return failure();
      resultTypes.push_back(converted);
    }

    if (resultTypes.size() <= 1) {
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

} // anonymous namespace

void populateMiscOpPatterns(LLVMTypeConverter &typeConverter,
                            RewritePatternSet &patterns,
                            ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  patterns.add<AppleFuncOpConversion>(
      typeConverter, PatternBenefit(patternBenefitDefault + 20));
  patterns.add<AppleCallOpConversion>(
      typeConverter, PatternBenefit(patternBenefitDefault + 20));
  patterns.add<AppleReturnOpConversion>(
      typeConverter, PatternBenefit(patternBenefitDefault + 20));
  patterns.add<WarpIdOpConversion>(typeConverter, patternBenefitDefault);
  patterns.add<ApplePrintOpConversion>(typeConverter,
                                       patternBenefitDefault + 10);
  patterns.add<AppleAssertOpConversion>(typeConverter,
                                        patternBenefitDefault + 10);
  patterns.add<GetNumProgramsOpAppleConversion>(
      typeConverter, PatternBenefit(patternBenefitDefault + 10));
  patterns.add<ExternElementwiseOpAppleConversion>(
      typeConverter, axisInfoAnalysis, patternBenefitDefault + 10);
}

} // namespace mlir::triton::applegpu
