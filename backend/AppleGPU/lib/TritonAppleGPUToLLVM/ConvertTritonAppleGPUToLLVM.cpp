// ConvertTritonAppleGPUToLLVM pass.
//
// Lowers TritonGPU IR → LLVM IR for Apple MPS using shared Triton patterns
// and an Apple-specific TargetInfo. This file has been stripped down to the
// minimum set of patterns needed to compile a pure load/arith/store kernel
// (e.g. vector_add). No DotOp, MMA, ConvertLayout, atomics, async copy,
// reductions, barriers, print, assert, or libdevice rewrites.

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
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
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

// ── AppleFuncOpConversion ────────────────────────────────────────────────
//
// Lower triton::FuncOp → LLVM::LLVMFuncOp for Apple Metal kernels.
//
// Metal passes scalar kernel args (i32, i64, etc.) via setBytes — a pointer
// to constant address space (addrspace 2). The LLVM IR must reflect this:
// scalar args become `i32 addrspace(2)*` pointers, and we insert explicit
// loads at function entry. Pointer args (addrspace 1 = device) pass through.
struct AppleFuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = funcOp.getContext();
    auto loc = funcOp.getLoc();
    bool isKernel = triton::isKernel(funcOp);

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
        Value loaded = LLVM::LoadOp::create(rewriter, loc, origTy, oldArg);
        oldArg.replaceAllUsesExcept(loaded, loaded.getDefiningOp());
      } else {
        oldArg.setType(newArgTypes[i]);
      }
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

// ── AppleReturnOpConversion ───────────────────────────────────────────────
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

// ── AppleCallOpConversion ─────────────────────────────────────────────────
// Lower triton::CallOp → LLVM::CallOp (no shared memory stack pointer).
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

// ── GetNumProgramsOpAppleConversion ───────────────────────────────────────
// Lower triton::GetNumProgramsOp → call @air.threadgroups_per_grid() +
// extractvalue.  Not strictly needed for vector_add, but keeps the pass
// useful for any kernel that reads the grid shape.
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

// ── WarpIdOpConversion ────────────────────────────────────────────────────
// Lower ttg::WarpIdOp → air.thread_position_in_threadgroup[0] / threadsPerWarp.
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

    int tpw = ttg::lookupThreadsPerWarp(rewriter);
    Value warpSize = arith::ConstantIntOp::create(rewriter, loc, tpw, 32);
    Value warpId = arith::DivUIOp::create(rewriter, loc, tid, warpSize);
    rewriter.replaceOp(op, warpId);
    return success();
  }
};

// ── ConvertTritonAppleGPUToLLVMPass ───────────────────────────────────────
struct ConvertTritonAppleGPUToLLVMPass
    : public PassWrapper<ConvertTritonAppleGPUToLLVMPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTritonAppleGPUToLLVMPass)

  StringRef getArgument() const override {
    return "convert-triton-apple-gpu-to-llvm";
  }
  StringRef getDescription() const override {
    return "Lower TritonGPU ops (Apple MPS, nano) to LLVM IR";
  }

  // Constructing the TritonGPUToLLVMTypeConverter lazily loads the `llvm`
  // dialect, which is illegal once the PassManager is multi-threaded (as it is
  // under triton-opt). Pre-declare the dialects this pass *produces* so MLIR
  // loads them before threading begins:
  //   - LLVM: the conversion target; the type converter loads it eagerly.
  //   - gpu:  emitted by the shared make_range/SPMD patterns, consumed by the
  //           subsequent LowerGPUToAirPass.
  // The arith/math/index/cf/ub source dialects are only *consumed* (their ops
  // come in from the input IR, so they are already loaded) — registering them
  // is unnecessary, and `index`'s TypeID symbol is not exported by libtriton
  // on macOS, so listing it would break loading the plugin there.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto mod = getOperation();
    auto *ctx = &getContext();

    TargetInfo targetInfo;
    TritonGPUToLLVMTypeConverter typeConverter(ctx, targetInfo);

    // Membar analysis: the shared convert-layout lowering may insert TG
    // memory ops. For vector_add this is a no-op, but running it keeps
    // the pass robust for any kernel the nano backend might accept.
    {
      ModuleAllocation allocation(mod);
      ModuleMembarAnalysis membarAnalysis(&allocation);
      membarAnalysis.run();
    }

    // Minimal global_smem: populate with a 16-byte block and set
    // ttg.shared so the shared reduce/scan patterns have a home even if
    // they never run.
    {
      int64_t smemSize = 0;
      if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.shared"))
        smemSize = attr.getValue().getZExtValue();
      if (smemSize == 0)
        smemSize = 8;
      int64_t smemAligned = (smemSize + 15) & ~15;
      mod->setAttr("ttg.shared",
                   IntegerAttr::get(IntegerType::get(ctx, 64), smemAligned));
      OpBuilder b(mod.getBodyRegion());
      auto loc = mod.getLoc();
      auto elemTy = typeConverter.convertType(b.getIntegerType(8));
      auto arrayTy = LLVM::LLVMArrayType::get(elemTy, smemSize);
      LLVM::GlobalOp::create(b, loc, arrayTy, /*isConstant=*/false,
                             LLVM::Linkage::Internal, "global_smem",
                             /*value=*/Attribute(), /*alignment=*/16,
                             /*addrSpace=*/3u);
    }

    RewritePatternSet patterns(ctx);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Apple-specific func / call / return (higher priority).
    patterns.add<AppleFuncOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 20));
    patterns.add<AppleCallOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 20));
    patterns.add<AppleReturnOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 20));

    // Shared Triton → LLVM patterns.
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

    // Apple load/store/addptr (required — upstream ones are CUDA-specific).
    populateLoadStoreToLLVMPatterns(typeConverter, patterns,
                                    patternBenefitDefault);

    patterns.add<GetNumProgramsOpAppleConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));
    patterns.add<WarpIdOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));

    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::triton::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo,
        patternBenefitDefault + 1);
    mlir::triton::populateClampFOpToLLVMPattern(typeConverter, patterns,
                                                axisInfoAnalysis, targetInfo,
                                                patternBenefitDefault + 1);
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

    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               patternBenefitDefault + 1);
    mlir::populatePolynomialApproximateErfPattern(patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    // Conversion target: everything must lower to LLVM dialect.
    ConversionTarget target(*ctx);
    target.addIllegalDialect<triton::TritonDialect>();
    target.addIllegalDialect<triton::gpu::TritonGPUDialect>();
    target.addIllegalDialect<applegpu::TritonAppleGPUDialect>();
    target.addIllegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    // gpu.thread_id / gpu.block_dim are lowered by the subsequent
    // LowerGPUToAirPass.
    target.addLegalOp<mlir::gpu::ThreadIdOp>();
    target.addLegalOp<mlir::gpu::BlockDimOp>();
    target.addLegalOp<mlir::gpu::BarrierOp>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};

// ── LowerGPUToAirPass ─────────────────────────────────────────────────────
//
// Converts remaining gpu.thread_id / gpu.block_dim ops (emitted by shared
// Triton patterns like make_range / SPMD) to air intrinsics / constants so
// the MLIR module is pure LLVM dialect before llvm::toModule().
//
//   gpu.thread_id x  →  call @air.thread_position_in_threadgroup()[0], zext
//   gpu.thread_id y/z → constant 0 : i64
//   gpu.block_dim x  →  constant <totalThreads> : i64   (from module attrs)
//   gpu.block_dim y/z → constant 1 : i64
//   gpu.barrier → call @air.wg.barrier(2, 1)  (kept for robustness)
struct LowerGPUToAirPass
    : public PassWrapper<LowerGPUToAirPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGPUToAirPass)

  StringRef getArgument() const override { return "lower-gpu-to-air"; }
  StringRef getDescription() const override {
    return "Lower gpu.thread_id / gpu.block_dim to air intrinsics / constants";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto *ctx = &getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);

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

    int64_t threadsPerWarp = 32;
    int64_t numWarps = 4;
    if (auto a = mod->getAttrOfType<IntegerAttr>("ttg.threads-per-warp"))
      threadsPerWarp = a.getInt();
    if (auto a = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
      numWarps = a.getInt();
    int64_t totalThreads = threadsPerWarp * numWarps;

    IRRewriter rewriter(ctx);

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
