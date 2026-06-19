// ConvertTritonAppleGPUToLLVM pass
//
// Lowers TritonGPU IR → LLVM IR for Apple MPS using shared Triton patterns
// and an Apple-specific TargetInfo.

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

namespace {

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
namespace ttg = mlir::triton::gpu;

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

    // Map !ttg.async.token to the event-slot pointer (not i32 0) so the wait
    // targets the exact copy the token names. Registered last to override the
    // upstream i32 mapping (newest-first).
    typeConverter.addConversion(
        [ctx](triton::gpu::AsyncTokenType type) -> std::optional<Type> {
          return LLVM::LLVMPointerType::get(ctx, 0);
        });

    // Fragment ABI: pure dot-chain #mma tensors carry fragments as a vectorized
    // struct so O3 keeps the accumulator vectorized (no SROA-to-scalar ->
    // occupancy collapse). Gated by consumer analysis; non-dot consumers fall
    // through to the generic flat per-thread struct.
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

    // Pre-compute MMA threadgroup usage (max over all dots; the pipeline
    // coalesces __tg_dot_ab globals into one) as a module attr so
    // ConvertLayoutOp can subtract it from the 32KB TG budget. Accounts for
    // bank-conflict padding (TG_PAD).
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

    // global_smem is only live with a shared-path consumer (reduce/scan/gather/
    // histogram, or a dot staging through it). When dead, llc strips it, so it
    // must NOT be subtracted from the convert's TG budget. Detect now; checking
    // after sibling lowerings run would race.
    {
      bool smemLive = false;
      mod.walk([&](Operation *o) {
        if (isa<mlir::triton::ReduceOp, mlir::triton::ScanOp,
                mlir::triton::GatherOp, mlir::triton::HistogramOp>(o))
          smemLive = true;
        // A pipelined float dot stages A/B through a ttg.local_alloc backed by
        // global_smem, keeping the reservation live.
        if (isa<ttg::LocalAllocOp>(o))
          smemLive = true;
        // An int8 dot aliases its A/B scatter buffer into global_smem
        // (getOrGrowSharedArena), keeping the reservation live.
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

    // Apple func/call/return (addrspace(2)* kernel args), warp-id/print/assert/
    // num-programs, and libdevice ExternElementwise. Func/call/return outrank
    // the shared FuncOp patterns (benefit +20).
    populateMiscOpPatterns(typeConverter, patterns, axisInfoAnalysis);

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

    populateMemoryOpPatterns(typeConverter, patterns);
    populateAtomicOpPatterns(typeConverter, patterns);
    populateConvertLayoutOpPattern(typeConverter, patterns);
    populateAsyncCopyPatterns(typeConverter, patterns, axisInfoAnalysis);

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

    // Fragment-ABI elementwise/mask/view lowerings (kkt op-web). Benefit +5 so
    // they win over the generic flat lowering when operands are fragment
    // structs and defer otherwise.
    populateFragmentElementwisePatterns(typeConverter, patterns);

    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               patternBenefitDefault + 1);
    // Expand math::ErfOp to polynomial approximation before MathToLLVM
    // (there is no llvm.erf intrinsic - NVIDIA uses libdevice, we expand
    // inline)
    mlir::populatePolynomialApproximateErfPattern(patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

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

    // All smem consumers share one arena via AddressOfOp(@global_smem);
    // rebasing every such ref onto a single trailing param preserves that
    // offset model.
    if (auto smemGlobal = mod.lookupSymbol<LLVM::GlobalOp>("global_smem")) {
      int64_t arenaBytes = 0;
      if (auto arrTy = dyn_cast<LLVMArrayType>(smemGlobal.getGlobalType()))
        arenaBytes = (int64_t)arrTy.getNumElements() *
                     (arrTy.getElementType().getIntOrFloatBitWidth() / 8);

      auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
      bool anyRebased = false;
      for (auto fn : llvm::to_vector(mod.getOps<LLVMFuncOp>())) {
        if (fn.isExternal())
          continue;
        SmallVector<LLVM::AddressOfOp> refs;
        fn.walk([&](LLVM::AddressOfOp ao) {
          if (ao.getGlobalName() == "global_smem")
            refs.push_back(ao);
        });
        if (refs.empty())
          continue;

        auto oldTy = fn.getFunctionType();
        SmallVector<Type> argTys(oldTy.getParams().begin(),
                                 oldTy.getParams().end());
        argTys.push_back(tgPtrTy);
        fn.setType(LLVM::LLVMFunctionType::get(oldTy.getReturnType(), argTys,
                                               oldTy.isVarArg()));
        Block &entry = fn.getBody().front();
        BlockArgument tgArg = entry.addArgument(tgPtrTy, fn.getLoc());
        for (auto ao : refs) {
          ao.getResult().replaceAllUsesWith(tgArg);
          ao.erase();
        }
        SmallVector<Attribute> passthrough;
        if (auto existing = fn.getPassthroughAttr())
          passthrough.append(existing.begin(), existing.end());
        passthrough.push_back(StringAttr::get(ctx, "air.thread_group_bound"));
        fn.setPassthroughAttr(ArrayAttr::get(ctx, passthrough));
        anyRebased = true;
      }

      if (anyRebased && smemGlobal.use_empty())
        smemGlobal.erase();
      if (anyRebased)
        mod->setAttr("applegpu.dynamic_smem_bytes",
                     IntegerAttr::get(IntegerType::get(ctx, 64), arenaBytes));
    }

    // Async-wait cleanup: AsyncWaitOp emits waits unconditionally. With no DMA
    // call the waits guard nothing and the AIR JIT refuses to materialize them,
    // so strip the waits and dead decl here.
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

    // Move loop_annotation from the discardable "llvm.loop_annotation" attr to
    // the BrOp's inherent "loop_annotation" property; otherwise
    // getLoopAnnotationAttr() is null and the !llvm.loop metadata is dropped.
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

    // Cooperative air.* decls must be marked `convergent` (barriers/event waits
    // also `noduplicate`), or LLVM CFG transforms may sink/duplicate them into
    // divergent control flow and desynchronize the threadgroup.
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

// LowerGPUToAirPass: convert remaining gpu.thread_id / gpu.block_dim ops
// (from shared make_range / SPMD patterns) to air intrinsics / constants so
// the module is pure LLVM dialect before llvm::toModule().
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

    // Declare air.thread_position_in_threadgroup once; returns [3 x i32], we
    // extractvalue index 0 for the flat thread ID. _add_air_metadata() rewrites
    // this call+extractvalue pattern to an arg.
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

    // index-to-llvm already ran, so emit LLVM i64 ops directly (index = i64,
    // pointer width on Apple Silicon):
    //   gpu.thread_id x   -> air.thread_position[0] -> zext i32->i64
    //   gpu.thread_id y/z -> constant 0
    //   gpu.block_dim x   -> constant totalThreads;  y/z -> constant 1
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
