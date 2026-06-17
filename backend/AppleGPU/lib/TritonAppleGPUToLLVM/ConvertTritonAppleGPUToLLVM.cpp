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

    // Apple func/call/return (addrspace(2)* kernel args, no SMEM stack ptr),
    // warp-id/print/assert/num-programs, and the libdevice ExternElementwise
    // mapping. Func/call/return outrank the NVIDIA-specific shared FuncOp
    // patterns (benefit +20).
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

    populateLoadStoreToLLVMPatterns(typeConverter, patterns,
                                    patternBenefitDefault);

    populateMemoryOpPatterns(typeConverter, patterns);
    populateConvertLayoutOpPattern(typeConverter, patterns);

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

// LowerGPUToAirPass
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
          // Silicon). Users of gpu.thread_id were already lowered to expect i64
          // via index_to_llvm; the extra SSA is fine since _add_air_metadata's
          // renumbering handles it.
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
