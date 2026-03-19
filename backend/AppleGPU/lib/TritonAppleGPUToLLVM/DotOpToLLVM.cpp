// DotOpToLLVM: lower tt.dot to air simdgroup matrix intrinsics via TG memory.
//
// Strategy (tiled TG scatter, register-resident C):
//   Scatter/load 8 rows at a time to minimize TG memory usage.
//   TG buffer = 8 * max(K, N) floats -- phases alias the same memory.
//
//   1. For each 8-row strip tm: scatter A[8xK] -> TG, barrier, load A[tm][*], barrier
//   2. For each 8-row strip tm: scatter C[8xN] -> TG, barrier, load C[tm][*], barrier
//   3. For each 8-row strip tk: scatter B[8xN] -> TG, barrier, load B[tk][*], barrier
//      then MMA: C[tm][tn] += A[tm][tk] * B[tk][tn] for all tm,tn
//   4. For each 8-row strip tm: store C[tm][*] -> TG, barrier, gather, barrier
//
// For 64x64x64: TG = 8*64 = 512 floats = 2 KB (vs 16 KB untiled).
// Fits within Apple's 32 KB TG limit even with large tiles.
//
// Supports arbitrary M*K x K*N where M,N,K are multiples of 8.
// Handles any blocked encoding (reads sizePerThread/threadsPerWarp/warpsPerCTA).
// Supports batched (3D+) dot: each batch gets its own TG region.
// Batch routing: dot_op operands (A/B) with mixed batch offsets use compile-time
// batch indices; blocked operands (C) with uniform offsets use runtime warpId.
//
// Two patterns:
//   DotOpBlockedConversion  — blocked encoding on C, any rank (batch-aware)
//   DotOpAppleMmaConversion — AppleMmaEncoding on C, rank-2 only

#include "TritonAppleGPUToLLVM/Passes.h"
#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallVector.h"

namespace tt  = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
using namespace mlir::triton::applegpu;

namespace {

static Type getSimdgroupMatrixType(MLIRContext *ctx) {
    return LLVM::getVectorType(Float32Type::get(ctx), 64);
}

static Value makeI64Vec2(OpBuilder &b, Location loc, int64_t a, int64_t b_val) {
    auto ty  = LLVM::getVectorType(IntegerType::get(b.getContext(), 64), 2);
    Value vec = UndefOp::create(b, loc, ty);
    Value va  = arith::ConstantIntOp::create(b, loc, a,     64);
    Value vb  = arith::ConstantIntOp::create(b, loc, b_val, 64);
    Value i0  = arith::ConstantIntOp::create(b, loc, 0, 32);
    Value i1  = arith::ConstantIntOp::create(b, loc, 1, 32);
    vec = InsertElementOp::create(b, loc, ty, vec, va, i0);
    vec = InsertElementOp::create(b, loc, ty, vec, vb, i1);
    return vec;
}

static LLVMFuncOp getOrInsertIntrinsic(ConversionPatternRewriter &rewriter,
                                        ModuleOp mod,
                                        StringRef name, LLVMFunctionType fnTy) {
    if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
        return fn;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    return LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                               Linkage::External);
}

static LLVM::GlobalOp getOrCreateTGGlobal(ConversionPatternRewriter &rewriter,
                                            ModuleOp mod,
                                            StringRef name, int64_t size) {
    if (auto g = mod.lookupSymbol<LLVM::GlobalOp>(name))
        return g;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto f32Ty = Float32Type::get(mod.getContext());
    auto arrTy = LLVMArrayType::get(f32Ty, size);
    return LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy,
                                   /*isConstant=*/false,
                                   LLVM::Linkage::Internal,
                                   name,
                                   /*value=*/Attribute(),
                                   /*alignment=*/4,
                                   /*addrspace=*/3u);
}

// Convert a value to f32. Handles both float and integer element types.
static Value toF32(OpBuilder &rewriter, Location loc, Value val, Type f32Ty) {
    auto valTy = val.getType();
    if (valTy == f32Ty)
        return val;
    if (isa<FloatType>(valTy))
        return arith::ExtFOp::create(rewriter, loc, f32Ty, val);
    // Integer type (e.g. i8, i16, i32) -- use signed conversion
    return arith::SIToFPOp::create(rewriter, loc, f32Ty, val);
}

// Convert f32 to the target element type. Handles both float and integer types.
static Value fromF32(OpBuilder &rewriter, Location loc, Value val, Type targetTy) {
    if (val.getType() == targetTy)
        return val;
    if (isa<FloatType>(targetTy))
        return arith::TruncFOp::create(rewriter, loc, targetTy, val);
    // Integer type -- use signed conversion
    return arith::FPToSIOp::create(rewriter, loc, targetTy, val);
}

static unsigned &getDotCounter(MLIRContext *ctx) {
    static llvm::DenseMap<MLIRContext *, unsigned> counters;
    return counters[ctx];
}

// ============================================================================
// DotOpBlockedConversion: blocked encoding on C, any rank (batch-aware).
// This is the original batch-aware lowering that handles 2D and 3D+ dots.
// ============================================================================
struct DotOpBlockedConversion : public ConvertOpToLLVMPattern<tt::DotOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        tt::DotOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {

        auto loc = op.getLoc();
        auto ctx = op.getContext();
        auto mod = op->getParentOfType<ModuleOp>();

        auto cType = cast<RankedTensorType>(op.getC().getType());
        auto cEnc = dyn_cast<ttg::BlockedEncodingAttr>(cType.getEncoding());
        // Only handle blocked encoding on C. MMA encoding handled separately.
        if (!cEnc)
            return failure();

        auto aType = cast<RankedTensorType>(op.getA().getType());
        auto bType = cast<RankedTensorType>(op.getB().getType());

        // ── Extract M, N, K from last two dimensions ────────────────────
        unsigned rank = cType.getRank();
        unsigned rowDim = rank - 2;
        unsigned colDim = rank - 1;

        int64_t M = cType.getShape()[rowDim];
        int64_t N = cType.getShape()[colDim];
        int64_t K = aType.getShape()[colDim]; // A is [..., M, K]

        // Compute batch size (product of all dims except last 2)
        int64_t batchSize = 1;
        for (unsigned d = 0; d < rowDim; ++d)
            batchSize *= cType.getShape()[d];


        auto f32Ty     = Float32Type::get(ctx);
        auto tgPtrTy   = LLVMPointerType::get(ctx, 3);
        auto matTy     = getSimdgroupMatrixType(ctx);
        auto i32Ty     = IntegerType::get(ctx, 32);
        auto i64Ty     = IntegerType::get(ctx, 64);

        // ── Declare air intrinsics ────────────────────────────────────────

        auto laneIdFn = getOrInsertIntrinsic(rewriter, mod,
            "air.thread_index_in_simdgroup",
            LLVMFunctionType::get(i32Ty, {}, false));

        auto voidTy = LLVMVoidType::get(ctx);
        auto barrTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
        auto tgBarrFn = getOrInsertIntrinsic(rewriter, mod,
            "air.threadgroup.barrier", barrTy);
        (void)getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup.barrier", barrTy);

        auto vec2i64Ty = LLVM::getVectorType(IntegerType::get(ctx, 64), 2);
        auto loadFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_load.v64f32.p3f32",
            LLVMFunctionType::get(matTy, {tgPtrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty}, false));
        auto mmaFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
            LLVMFunctionType::get(matTy, {matTy, matTy, matTy}, false));
        auto storeFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_store.v64f32.p3f32",
            LLVMFunctionType::get(voidTy, {matTy, tgPtrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty}, false));

        // ── Constants ────────────────────────────────────────────────────

        Value fenceTG  = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
        Value execMod  = arith::ConstantIntOp::create(rewriter, loc, 4, 32);

        // ── Thread identification ─────────────────────────────────────────

        Value laneId = LLVM::CallOp::create(rewriter, loc, laneIdFn,
                                             ValueRange{}).getResult();

        auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
        auto tidFn = getOrInsertIntrinsic(rewriter, mod,
            "air.thread_position_in_threadgroup",
            LLVMFunctionType::get(arrI32x3Ty, {}, false));
        Value tidStruct = LLVM::CallOp::create(rewriter, loc, tidFn,
                                                ValueRange{}).getResult();
        Value tid32 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty,
                          tidStruct, ArrayRef<int64_t>{0});
        Value c32    = arith::ConstantIntOp::create(rewriter, loc, 32, 32);
        Value warpId = arith::DivUIOp::create(rewriter, loc, tid32, c32);

        // ── Get blocked encoding params for A, B, C ───────────────────────

        // Unpack struct elements
        auto unpack = [&](Value v) -> SmallVector<Value> {
            SmallVector<Value> elems;
            if (auto sTy = dyn_cast<LLVMStructType>(v.getType())) {
                for (unsigned i = 0; i < sTy.getBody().size(); ++i)
                    elems.push_back(ExtractValueOp::create(rewriter, loc,
                        sTy.getBody()[i], v, ArrayRef<int64_t>{(int64_t)i}));
            } else {
                elems = {v};
            }
            return elems;
        };

        // resolveOperand returns: (elements, offsets, encoding, dotOpIdx)
        // dotOpIdx: -1 if not from DotOperandEncoding, 0 for A-matrix, 1 for B-matrix
        auto resolveOperand = [&](Value tritonVal, Value adaptorVal,
                                  RankedTensorType opTy)
            -> std::tuple<SmallVector<Value>,
                          SmallVector<SmallVector<unsigned>>,
                          ttg::BlockedEncodingAttr,
                          int> {
            // Path 1: convert_layout -- look through to source blocked values
            if (auto cvt = tritonVal.getDefiningOp<ttg::ConvertLayoutOp>()) {
                Value mapped = rewriter.getRemappedValue(cvt.getSrc());
                if (mapped) {
                    auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
                    auto srcEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
                    if (srcEnc) {
                        auto offsets = emitOffsetForLayout(srcEnc, srcTy);
                        return {unpack(mapped), offsets, srcEnc, -1};
                    }
                }
            }
            // Path 2: DotOperandEncoding (e.g. local_load after optimize_dot_operands)
            auto enc = opTy.getEncoding();
            if (auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(enc)) {
                auto parentEnc = dyn_cast<ttg::BlockedEncodingAttr>(dotEnc.getParent());
                if (parentEnc) {
                    auto offsets = emitOffsetForLayout(enc, opTy);
                    return {unpack(adaptorVal), offsets, parentEnc,
                            (int)dotEnc.getOpIdx()};
                }
            }
            // Path 3: direct blocked encoding
            if (auto blk = dyn_cast<ttg::BlockedEncodingAttr>(enc)) {
                auto offsets = emitOffsetForLayout(blk, opTy);
                return {unpack(adaptorVal), offsets, blk, -1};
            }
            return {{}, {}, nullptr, -1};
        };

        auto [elemsA, aOffsets, aSrcEnc, aDotOpIdx] = resolveOperand(op.getA(), adaptor.getA(), aType);
        auto [elemsB, bOffsets, bSrcEnc, bDotOpIdx] = resolveOperand(op.getB(), adaptor.getB(), bType);
        auto elemsC = unpack(adaptor.getC());
        auto cOffsets = emitOffsetForLayout(cEnc, cType);

        if (!aSrcEnc || !bSrcEnc)
            return failure();

        // Verify element counts match
        if ((int64_t)elemsA.size() != (int64_t)aOffsets.size() ||
            (int64_t)elemsB.size() != (int64_t)bOffsets.size() ||
            (int64_t)elemsC.size() != (int64_t)cOffsets.size())
            return failure();

        // ── Compute runtime thread base position ──────────────────────────
        // For 3D+ tensors, use only the last two dims of the encoding for
        // the MMA row/col base. The batch dims are handled via compile-time
        // offset matching.
        auto makeBase = [&](ttg::BlockedEncodingAttr enc, int64_t rows, int64_t cols)
            -> std::pair<Value, Value> {
            auto spt = enc.getSizePerThread();
            auto tpw = enc.getThreadsPerWarp();
            auto wpc = enc.getWarpsPerCTA();
            auto order = enc.getOrder();

            unsigned encRank = spt.size();
            unsigned encRowDim = encRank - 2;
            unsigned encColDim = encRank - 1;

            int64_t sM = spt[encRowDim], sN = spt[encColDim];
            int64_t tM = tpw[encRowDim], tN = tpw[encColDim];
            int64_t wM = wpc[encRowDim], wN = wpc[encColDim];
            int64_t tileM = wM * tM * sM;
            int64_t tileN = wN * tN * sN;

            // For 3D+ encodings, strip batch warp component from warpId.
            // Batch dims use warps for batch distribution; the 2D row/col
            // decomposition should only use the row/col warps.
            int64_t batchWarps = 1;
            for (unsigned d = 0; d < encRowDim; ++d)
                batchWarps *= wpc[d];
            Value matWarpId = warpId;
            if (batchWarps > 1) {
                Value bw = arith::ConstantIntOp::create(rewriter, loc, batchWarps, 32);
                matWarpId = arith::RemUIOp::create(rewriter, loc, warpId, bw);
                // Actually we need warpId within the 2D tile, not within batch.
                // The warp decomposition maps warpId -> (batch_warp, mat_warp).
                // mat_warp = warpId % (wM * wN), batch_warp = warpId / (wM * wN)
                int64_t matWarps = wM * wN;
                Value mw = arith::ConstantIntOp::create(rewriter, loc, matWarps, 32);
                matWarpId = arith::RemUIOp::create(rewriter, loc, warpId, mw);
            }

            // Similarly strip batch lanes from laneId
            int64_t batchLanes = 1;
            for (unsigned d = 0; d < encRowDim; ++d)
                batchLanes *= tpw[d];
            Value matLaneId = laneId;
            if (batchLanes > 1) {
                int64_t matLanes = tM * tN;
                Value ml = arith::ConstantIntOp::create(rewriter, loc, matLanes, 32);
                matLaneId = arith::RemUIOp::create(rewriter, loc, laneId, ml);
            }

            // For the last two dims, check if col is the fastest-varying dim.
            // order[0] is the fastest dim index. For 2D: order[0]==1 means col-fast.
            // For 3D with order=[2,1,0]: order[0]==2 means colDim is fastest.
            bool colFastest = (order[0] == (unsigned)encColDim);

            Value wN_val  = arith::ConstantIntOp::create(rewriter, loc, wN, 32);
            Value tN_val  = arith::ConstantIntOp::create(rewriter, loc, tN, 32);
            Value tMsM    = arith::ConstantIntOp::create(rewriter, loc, tM * sM, 32);
            Value sM_val  = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
            Value tNsN    = arith::ConstantIntOp::create(rewriter, loc, tN * sN, 32);
            Value sN_val  = arith::ConstantIntOp::create(rewriter, loc, sN, 32);

            // Warp decomposition: faster dim uses mod, slower uses div
            Value wR, wC;
            if (colFastest) {
                wR = arith::DivUIOp::create(rewriter, loc, matWarpId, wN_val);
                wC = arith::RemUIOp::create(rewriter, loc, matWarpId, wN_val);
            } else {
                Value wM_val = arith::ConstantIntOp::create(rewriter, loc, wM, 32);
                wR = arith::RemUIOp::create(rewriter, loc, matWarpId, wM_val);
                wC = arith::DivUIOp::create(rewriter, loc, matWarpId, wM_val);
            }
            // Lane decomposition: faster dim uses mod, slower uses div
            Value lR, lC;
            if (colFastest) {
                lR = arith::DivUIOp::create(rewriter, loc, matLaneId, tN_val);
                lC = arith::RemUIOp::create(rewriter, loc, matLaneId, tN_val);
            } else {
                Value tM_val = arith::ConstantIntOp::create(rewriter, loc, tM, 32);
                lR = arith::RemUIOp::create(rewriter, loc, matLaneId, tM_val);
                lC = arith::DivUIOp::create(rewriter, loc, matLaneId, tM_val);
            }

            Value baseRow = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, wR, tMsM),
                arith::MulIOp::create(rewriter, loc, lR, sM_val));
            Value baseCol = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, wC, tNsN),
                arith::MulIOp::create(rewriter, loc, lC, sN_val));

            // Wrap to handle redundant threads (tileM > rows)
            if (tileM > rows) {
                Value modRow = arith::ConstantIntOp::create(rewriter, loc, rows, 32);
                baseRow = arith::RemUIOp::create(rewriter, loc, baseRow, modRow);
            }
            if (tileN > cols) {
                Value modCol = arith::ConstantIntOp::create(rewriter, loc, cols, 32);
                baseCol = arith::RemUIOp::create(rewriter, loc, baseCol, modCol);
            }

            return {baseRow, baseCol};
        };

        auto [aBaseRow, aBaseCol] = makeBase(aSrcEnc, M, K);
        auto [bBaseRow, bBaseCol] = makeBase(bSrcEnc, K, N);
        auto [cBaseRow, cBaseCol] = makeBase(cEnc, M, N);

        // For DotOperandEncoding (Path 2), the contracting dimension (K)
        // is fully replicated per thread. The offsets from emitOffsetForLayout
        // already span the full K range, so the base for the K dim must be 0.
        // Otherwise base + offset overshoots the K dimension.
        Value zero32 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
        if (aDotOpIdx == 0) {
            // A is [M, K]: col dim is K (contracting) -> zero base
            aBaseCol = zero32;
        }
        if (bDotOpIdx == 1) {
            // B is [K, N]: row dim is K (contracting) -> zero base
            bBaseRow = zero32;
        }

        // ── Create threadgroup global ─────────────────────────────────────
        unsigned id = getDotCounter(ctx)++;
        int64_t tgStripSize = 8 * std::max(K, N);
        // Each batch slice needs its own TG region so MMA ops don't
        // cross-contaminate between warps assigned to different batches.
        int64_t tgSize = tgStripSize * batchSize + 1;  // +1 garbage slot
        auto tgBuf = getOrCreateTGGlobal(rewriter, mod,
            ("__tg_dot_ab_" + llvm::Twine(id)).str(), tgSize);

        Value ptrTG = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgBuf.getName());

        // Compute runtime batch-offset pointer for SIMD matrix load/store.
        // Each batch slice gets its own tgStripSize region in TG memory so
        // MMA ops don't cross-contaminate between warps assigned to different batches.
        //
        // numBatchWarps = numTotalWarps / matWarpsC.
        // When batchSize > numBatchWarps, we must process multiple batches
        // per warp via an unrolled batch loop (see below).
        auto cWpc = cEnc.getWarpsPerCTA();
        int64_t matWarpsC = cWpc[rowDim] * cWpc[colDim];
        int64_t numTotalWarps = 1;
        for (auto w : cWpc) numTotalWarps *= w;
        int64_t numBatchWarps = numTotalWarps / matWarpsC;

        // Compute per-operand batch warp index (runtime).
        // batchWarpIdx = warpId / matWarps, where matWarps is the product
        // of warpsPerCTA for the row and col dims of each operand's encoding.
        // This determines which batch slice each warp handles.
        auto makeBatchWarpIdx = [&](ttg::BlockedEncodingAttr enc) -> Value {
            auto wpc = enc.getWarpsPerCTA();
            unsigned encRank = wpc.size();
            unsigned encRowDim = encRank - 2;
            unsigned encColDim = encRank - 1;
            int64_t matW = wpc[encRowDim] * wpc[encColDim];
            int64_t batchW = 1;
            for (unsigned d = 0; d < encRowDim; ++d) batchW *= wpc[d];
            if (batchW <= 1)
                return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
            Value mw = arith::ConstantIntOp::create(rewriter, loc, matW, 32);
            return arith::DivUIOp::create(rewriter, loc, warpId, mw);
        };

        Value cBatchWarpIdx = makeBatchWarpIdx(cEnc);
        Value aBatchWarpIdx = makeBatchWarpIdx(aSrcEnc);
        Value bBatchWarpIdx = makeBatchWarpIdx(bSrcEnc);

        Value ptrTGBatch = ptrTG;
        Value batchTGOffset64 = arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64);
        if (batchSize > 1 && numBatchWarps >= batchSize) {
            // Warp-distributed: each warp handles one batch via C's batchWarpIdx.
            Value batchOff32 = arith::MulIOp::create(rewriter, loc, cBatchWarpIdx,
                arith::ConstantIntOp::create(rewriter, loc, tgStripSize, 32));
            batchTGOffset64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, batchOff32);
            ptrTGBatch = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32Ty,
                ptrTG, ArrayRef<LLVM::GEPArg>{batchTGOffset64});
        }

        // ── GEP helpers ───────────────────────────────────────────────────

        auto gather1 = [&](Value ptr, Value flatIdx64) -> Value {
            Value gep = LLVM::GEPOp::create(rewriter, loc,
                tgPtrTy, f32Ty, ptr, ArrayRef<LLVM::GEPArg>{flatIdx64});
            return LLVM::LoadOp::create(rewriter, loc, f32Ty, gep).getResult();
        };

        // stripFlatIdx: (baseRow + rowOff - stripRowStart) * stride + (baseCol + colOff)
        auto stripFlatIdx = [&](Value baseRow, Value baseCol,
                                int64_t rowOff, int64_t colOff,
                                int64_t stride, int64_t stripRowStart) -> Value {
            Value row32 = arith::AddIOp::create(rewriter, loc, baseRow,
                arith::ConstantIntOp::create(rewriter, loc, rowOff - stripRowStart, 32));
            Value col32 = arith::AddIOp::create(rewriter, loc, baseCol,
                arith::ConstantIntOp::create(rewriter, loc, colOff, 32));
            Value flat32 = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, row32,
                    arith::ConstantIntOp::create(rewriter, loc, stride, 32)),
                col32);
            return arith::ExtUIOp::create(rewriter, loc, i64Ty, flat32);
        };

        int64_t tilesM = M / 8;
        int64_t tilesN = N / 8;
        int64_t tilesK = K / 8;

        // Garbage bin index -- last slot in TG, used for out-of-strip stores.
        // Points to the very last slot in the full (batch-expanded) TG buffer.
        Value garbageIdx = arith::ConstantIntOp::create(rewriter, loc,
            tgStripSize * batchSize, 64);

        // Determine if an operand has batch warps (runtime batch component).
        // If wpc[0] == 1, all batches are in compile-time offsets (no runtime batch).
        // If wpc[0] > 1, batch is partially runtime (batchWarpIdx contributes).
        auto hasBatchWarps = [&](ttg::BlockedEncodingAttr enc) -> bool {
            if (rowDim == 0) return false;
            auto wpc = enc.getWarpsPerCTA();
            int64_t bw = 1;
            for (unsigned d = 0; d < rowDim; ++d) bw *= wpc[d];
            return bw > 1;
        };

        // "hasBatchWarps" means the operand needs runtime batch filtering
        // in sequential mode. If false, compile-time elemBatchIndex covers
        // all batches and compile-time filtering suffices.
        bool aHasBatchWarps = hasBatchWarps(aSrcEnc);
        bool bHasBatchWarps = hasBatchWarps(bSrcEnc);
        bool cHasBatchWarps = hasBatchWarps(cEnc);

        // For warp-distributed mode (TG offset routing), we still need to know
        // if offsets have mixed batch values for the TG offset computation.
        auto hasMixedBatches = [&](const SmallVector<SmallVector<unsigned>> &offsets) -> bool {
            if (rowDim == 0 || offsets.empty()) return false;
            for (size_t i = 1; i < offsets.size(); ++i) {
                for (unsigned d = 0; d < rowDim; ++d) {
                    if (offsets[i][d] != offsets[0][d])
                        return true;
                }
            }
            return false;
        };

        bool aMixed = hasMixedBatches(aOffsets);
        bool bMixed = hasMixedBatches(bOffsets);
        bool cMixed = hasMixedBatches(cOffsets);

        // Helper: compute the TG batch offset for an element.
        // For operands with mixed batch offsets (e.g. dot_op A/B), use the
        // compile-time offset as the global batch index.
        // For operands with uniform batch offsets (e.g. C blocked), use the
        // runtime batchTGOffset64 derived from batchWarpIdx.
        auto elemBatchTGOffset = [&](const SmallVector<SmallVector<unsigned>> &offsets,
                                     size_t i, bool mixed) -> Value {
            if (rowDim == 0)
                return arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64);
            if (mixed) {
                // Compute flat batch index from compile-time offset dims
                int64_t batchOff = 0;
                int64_t stride = 1;
                for (int d = (int)rowDim - 1; d >= 0; --d) {
                    batchOff += offsets[i][d] * stride;
                    stride *= cType.getShape()[d];
                }
                return arith::ConstantIntOp::create(rewriter, loc,
                    batchOff * tgStripSize, 64);
            }
            // Uniform batch -- use runtime warp-based offset
            return batchTGOffset64;
        };

        // Helper: compute flat batch index for element i from compile-time offsets.
        auto elemBatchIndex = [&](const SmallVector<SmallVector<unsigned>> &offsets,
                                  size_t i) -> int64_t {
            int64_t batchIdx = 0;
            int64_t stride = 1;
            for (int d = (int)rowDim - 1; d >= 0; --d) {
                batchIdx += offsets[i][d] * stride;
                stride *= cType.getShape()[d];
            }
            return batchIdx;
        };

        // Helper: scatter elements into TG for an 8-row strip.
        // In sequential batch mode (curBatchRound >= 0), only scatter elements
        // matching the current batch. Data goes to TG base (no per-batch regions).
        // For operands without batch warps: compile-time filter by elemBatchIndex.
        // For operands with batch warps: runtime filter using batchWarpIdx.
        // In warp-distributed mode (curBatchRound < 0), scatter all elements,
        // each to its own batch region based on elemBatchTGOffset.
        auto stripScatter = [&](Value baseRow, Value baseCol,
                                SmallVector<Value> &elems,
                                SmallVector<SmallVector<unsigned>> &offsets,
                                int64_t stride, int64_t rowStart,
                                bool mixed, int64_t curBatchRound,
                                Value operandBatchWarpIdx,
                                bool opHasBatchWarps) {
            for (size_t i = 0; i < elems.size(); ++i) {
                int64_t eb = (rowDim > 0) ? elemBatchIndex(offsets, i) : 0;

                // In sequential batch mode, skip elements not in current batch.
                if (curBatchRound >= 0 && rowDim > 0 && !opHasBatchWarps) {
                    // No batch warps: compile-time batch index IS the actual batch.
                    if (eb != curBatchRound) continue;
                }
                // For operands WITH batch warps: can't skip at compile time.
                // Runtime check is added below.

                int64_t rowOff = offsets[i][rowDim];
                int64_t colOff = offsets[i][colDim];
                Value actualRow = arith::AddIOp::create(rewriter, loc, baseRow,
                    arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
                Value inStrip = arith::AndIOp::create(rewriter, loc,
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                        actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                        actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart + 8, 32)));

                // For operands with batch warps in sequential mode, add runtime batch check.
                // actual batch = elemBatchIndex + batchWarpIdx
                // match condition: batchWarpIdx == curBatchRound - elemBatchIndex
                if (curBatchRound >= 0 && rowDim > 0 && opHasBatchWarps) {
                    Value targetBatchWarp = arith::ConstantIntOp::create(rewriter, loc,
                        curBatchRound - eb, 32);
                    Value batchMatch = arith::CmpIOp::create(rewriter, loc,
                        arith::CmpIPredicate::eq, operandBatchWarpIdx, targetBatchWarp);
                    inStrip = arith::AndIOp::create(rewriter, loc, inStrip, batchMatch);
                }

                Value idx = stripFlatIdx(baseRow, baseCol, rowOff, colOff, stride, rowStart);
                if (curBatchRound >= 0) {
                    // Sequential mode: all data goes to TG base.
                    Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip, idx, garbageIdx);
                    Value val = toF32(rewriter, loc, elems[i], f32Ty);
                    Value gep = LLVM::GEPOp::create(rewriter, loc,
                        tgPtrTy, f32Ty, ptrTG, ArrayRef<LLVM::GEPArg>{safeIdx});
                    LLVM::StoreOp::create(rewriter, loc, val, gep);
                } else {
                    // Warp-distributed mode: add per-element batch TG offset.
                    Value batchOff = elemBatchTGOffset(offsets, i, mixed);
                    Value batchIdx = arith::AddIOp::create(rewriter, loc, idx, batchOff);
                    Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip, batchIdx, garbageIdx);
                    Value val = toF32(rewriter, loc, elems[i], f32Ty);
                    Value gep = LLVM::GEPOp::create(rewriter, loc,
                        tgPtrTy, f32Ty, ptrTG, ArrayRef<LLVM::GEPArg>{safeIdx});
                    LLVM::StoreOp::create(rewriter, loc, val, gep);
                }
            }
        };

        // ── Initialize result to zero ────────────────────────────────────
        auto outElemTy = cType.getElementType();
        SmallVector<Value> resultElems(elemsC.size());
        for (size_t i = 0; i < elemsC.size(); ++i)
            resultElems[i] = arith::ConstantOp::create(rewriter, loc,
                rewriter.getZeroAttr(outElemTy));

        // ── MMA computation (batch-aware) ─────────────────────────────
        // When numBatchWarps >= batchSize, each warp handles one batch
        // (ptrTGBatch already points to that batch's TG region) -- single pass.
        // When numBatchWarps < batchSize, process one batch per round.
        // Each round: scatter/load/MMA/store/gather for a single batch.
        // All data goes to TG base (single region). Runtime batch filtering
        // ensures only the correct batch's data is scattered.

        bool batchConsistent = true;
        if (batchSize > 1) {
            auto aWpc = aSrcEnc.getWarpsPerCTA();
            auto bWpc = bSrcEnc.getWarpsPerCTA();
            int64_t matWarpsA = aWpc[rowDim] * aWpc[colDim];
            int64_t matWarpsB = bWpc[rowDim] * bWpc[colDim];
            if (matWarpsA != matWarpsC || matWarpsB != matWarpsC)
                batchConsistent = false;
        }

        int64_t batchRounds = 1;
        if (batchSize > 1 && (numBatchWarps < batchSize || !batchConsistent))
            batchRounds = batchSize;

        for (int64_t batchRound = 0; batchRound < batchRounds; ++batchRound) {
            Value curPtrTGBatch = ptrTGBatch;
            int64_t scatterBatchRound = -1; // -1 = warp-distributed (no filter)
            if (batchRounds > 1) {
                curPtrTGBatch = ptrTG; // sequential: single TG region at base
                scatterBatchRound = batchRound;
            }

            // Phase 1: Load A tiles (8-row strips)
            SmallVector<SmallVector<Value>> matA_tiles(tilesM);
            for (int64_t tm = 0; tm < tilesM; ++tm) {
                matA_tiles[tm].resize(tilesK);
                int64_t rowStart = tm * 8;

                stripScatter(aBaseRow, aBaseCol, elemsA, aOffsets, K, rowStart, aMixed, scatterBatchRound, aBatchWarpIdx, aHasBatchWarps);
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

                for (int64_t tk = 0; tk < tilesK; ++tk) {
                    Value aOff = makeI64Vec2(rewriter, loc, tk * 8, 0);
                    Value aStride = makeI64Vec2(rewriter, loc, 1, K);
                    Value aShape  = makeI64Vec2(rewriter, loc, K, 8);
                    matA_tiles[tm][tk] = LLVM::CallOp::create(rewriter, loc, loadFn,
                        ValueRange{curPtrTGBatch, aShape, aStride, aOff}).getResult();
                }
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
            }

            // Phase 2: Load C tiles (8-row strips)
            SmallVector<SmallVector<Value>> matC_tiles(tilesM);
            for (int64_t tm = 0; tm < tilesM; ++tm) {
                matC_tiles[tm].resize(tilesN);
                int64_t rowStart = tm * 8;

                stripScatter(cBaseRow, cBaseCol, elemsC, cOffsets, N, rowStart, cMixed, scatterBatchRound, cBatchWarpIdx, cHasBatchWarps);
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

                Value cStride = makeI64Vec2(rewriter, loc, 1, N);
                Value cShape  = makeI64Vec2(rewriter, loc, N, 8);
                for (int64_t tn = 0; tn < tilesN; ++tn) {
                    Value cOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
                    matC_tiles[tm][tn] = LLVM::CallOp::create(rewriter, loc, loadFn,
                        ValueRange{curPtrTGBatch, cShape, cStride, cOff}).getResult();
                }
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
            }

            // Phase 3: B strips + MMA
            for (int64_t tk = 0; tk < tilesK; ++tk) {
                int64_t rowStart = tk * 8;

                stripScatter(bBaseRow, bBaseCol, elemsB, bOffsets, N, rowStart, bMixed, scatterBatchRound, bBatchWarpIdx, bHasBatchWarps);
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

                Value bStride = makeI64Vec2(rewriter, loc, 1, N);
                Value bShape  = makeI64Vec2(rewriter, loc, N, 8);
                for (int64_t tn = 0; tn < tilesN; ++tn) {
                    Value bOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
                    Value matB = LLVM::CallOp::create(rewriter, loc, loadFn,
                        ValueRange{curPtrTGBatch, bShape, bStride, bOff}).getResult();

                    for (int64_t tm = 0; tm < tilesM; ++tm) {
                        matC_tiles[tm][tn] = LLVM::CallOp::create(rewriter, loc, mmaFn,
                            ValueRange{matA_tiles[tm][tk], matB, matC_tiles[tm][tn]}).getResult();
                    }
                }
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
            }

            // Phase 4: Store C tiles -> TG (8-row strips), gather
            for (int64_t tm = 0; tm < tilesM; ++tm) {
                int64_t rowStart = tm * 8;

                Value cStoreStride = makeI64Vec2(rewriter, loc, 1, N);
                Value cStoreShape  = makeI64Vec2(rewriter, loc, N, 8);
                for (int64_t tn = 0; tn < tilesN; ++tn) {
                    Value cOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
                    LLVM::CallOp::create(rewriter, loc, storeFn,
                        ValueRange{matC_tiles[tm][tn], curPtrTGBatch, cStoreShape, cStoreStride, cOff});
                }
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

                // Gather: each thread reads its C elements from TG.
                for (size_t i = 0; i < elemsC.size(); ++i) {
                    int64_t elemBatch = (rowDim > 0) ? elemBatchIndex(cOffsets, i) : 0;

                    // In sequential batch mode, skip elements not in current batch.
                    if (batchRounds > 1 && rowDim > 0 && !cHasBatchWarps) {
                        // No batch warps: compile-time batch IS actual batch.
                        if (elemBatch != batchRound) continue;
                    }

                    int64_t rowOff = cOffsets[i][rowDim];
                    int64_t colOff = cOffsets[i][colDim];
                    Value actualRow = arith::AddIOp::create(rewriter, loc, cBaseRow,
                        arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
                    Value inStrip = arith::AndIOp::create(rewriter, loc,
                        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                            actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
                        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                            actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart + 8, 32)));

                    // For C with batch warps in sequential mode, add runtime batch check.
                    if (batchRounds > 1 && rowDim > 0 && cHasBatchWarps) {
                        Value targetBatchWarp = arith::ConstantIntOp::create(rewriter, loc,
                            batchRound - elemBatch, 32);
                        Value batchMatch = arith::CmpIOp::create(rewriter, loc,
                            arith::CmpIPredicate::eq, cBatchWarpIdx, targetBatchWarp);
                        inStrip = arith::AndIOp::create(rewriter, loc, inStrip, batchMatch);
                    }

                    Value idx = stripFlatIdx(cBaseRow, cBaseCol, rowOff, colOff, N, rowStart);
                    if (batchRounds > 1) {
                        // Sequential mode: data is at TG base.
                        Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip, idx, garbageIdx);
                        Value val = gather1(ptrTG, safeIdx);
                        if (val.getType() != outElemTy)
                            val = fromF32(rewriter, loc, val, outElemTy);
                        resultElems[i] = arith::SelectOp::create(rewriter, loc, inStrip,
                            val, resultElems[i]);
                    } else {
                        // Warp-distributed mode: add batch TG offset.
                        Value batchOff = elemBatchTGOffset(cOffsets, i, cMixed);
                        Value batchIdx = arith::AddIOp::create(rewriter, loc, idx, batchOff);
                        Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip, batchIdx, garbageIdx);
                        Value val = gather1(ptrTG, safeIdx);
                        if (val.getType() != outElemTy)
                            val = fromF32(rewriter, loc, val, outElemTy);
                        resultElems[i] = arith::SelectOp::create(rewriter, loc, inStrip,
                            val, resultElems[i]);
                    }
                }
                LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
            }
        }  // end batchRound loop

        // ── Pack result ───────────────────────────────────────────────────
        auto outLLVMTy = getTypeConverter()->convertType(cType);
        if (!outLLVMTy) return failure();

        if (auto outStructTy = dyn_cast<LLVMStructType>(outLLVMTy)) {
            Value result = UndefOp::create(rewriter, loc, outStructTy);
            for (size_t i = 0; i < resultElems.size(); ++i)
                result = InsertValueOp::create(rewriter, loc, outStructTy,
                             result, resultElems[i],
                             ArrayRef<int64_t>{(int64_t)i});
            rewriter.replaceOp(op, result);
        } else {
            rewriter.replaceOp(op, resultElems[0]);
        }
        return success();
    }
};

// ============================================================================
// DotOpAppleMmaConversion: AppleMmaEncoding on C, rank-2 only.
// Handles dots where AccelerateAppleMatmul has rewritten C to AppleMmaEncoding.
// Uses the simpler 2D-only path with static strip bucketing.
//
// OPTIMIZATION: When A/B operands come from tt.load (device memory), loads
// MMA tiles directly from device memory via p1f32 intrinsics, eliminating
// the TG scatter/gather bottleneck and ~12 barriers per dot iteration.
// Falls back to TG scatter path when device pointers are unavailable.
// ============================================================================
struct DotOpAppleMmaConversion : public ConvertOpToLLVMPattern<tt::DotOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        tt::DotOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {

        auto loc = op.getLoc();
        auto ctx = op.getContext();
        auto mod = op->getParentOfType<ModuleOp>();

        auto cType = cast<RankedTensorType>(op.getC().getType());
        auto cMmaEnc = dyn_cast<AppleMmaEncodingAttr>(cType.getEncoding());
        // Only handle AppleMmaEncoding on C.
        if (!cMmaEnc)
            return failure();

        // Only support rank-2 tensors for MMA encoding path.
        unsigned rank = cType.getRank();
        if (rank != 2)
            return failure();

        auto aType = cast<RankedTensorType>(op.getA().getType());
        auto bType = cast<RankedTensorType>(op.getB().getType());

        int64_t M = cType.getShape()[0];
        int64_t N = cType.getShape()[1];
        int64_t K = aType.getShape()[1]; // A is [M, K]

        auto f32Ty     = Float32Type::get(ctx);
        auto tgPtrTy   = LLVMPointerType::get(ctx, 3);
        auto devPtrTy  = LLVMPointerType::get(ctx, 1);
        auto matTy     = getSimdgroupMatrixType(ctx);
        auto i32Ty     = IntegerType::get(ctx, 32);
        auto i64Ty     = IntegerType::get(ctx, 64);

        // ── Declare air intrinsics ────────────────────────────────────────

        auto laneIdFn = getOrInsertIntrinsic(rewriter, mod,
            "air.thread_index_in_simdgroup",
            LLVMFunctionType::get(i32Ty, {}, false));

        auto voidTy = LLVMVoidType::get(ctx);
        auto barrTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
        auto tgBarrFn = getOrInsertIntrinsic(rewriter, mod,
            "air.threadgroup.barrier", barrTy);
        (void)getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup.barrier", barrTy);

        auto vec2i64Ty = LLVM::getVectorType(IntegerType::get(ctx, 64), 2);
        auto loadFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_load.v64f32.p3f32",
            LLVMFunctionType::get(matTy, {tgPtrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty}, false));
        auto mmaFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
            LLVMFunctionType::get(matTy, {matTy, matTy, matTy}, false));
        auto storeFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_store.v64f32.p3f32",
            LLVMFunctionType::get(voidTy, {matTy, tgPtrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty}, false));

        // ── Device memory MMA intrinsics (type-specific) ─────────────────
        // For f16/bf16 data: load returns <64 x half/bfloat>, MMA takes
        // half/bfloat inputs with f32 accumulator.
        // For f32 data: load returns <64 x float>, MMA takes f32 inputs.
        auto aElemTy = aType.getElementType();
        auto f16Ty = Float16Type::get(ctx);
        auto bf16Ty = BFloat16Type::get(ctx);
        bool isF16Input = aElemTy.isF16();
        bool isBF16Input = aElemTy.isBF16();

        // Determine device load intrinsic based on element type
        Type devMatElemTy = f32Ty;  // element type for device MMA matrix
        std::string devLoadName = "air.simdgroup_matrix_8x8_load.v64f32.p1f32";
        std::string devMmaName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32";
        Type devGepElemTy = f32Ty;  // GEP element type

        if (isF16Input) {
            devMatElemTy = f16Ty;
            devLoadName = "air.simdgroup_matrix_8x8_load.v64f16.p1f16";
            devMmaName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32";
            devGepElemTy = f16Ty;
        } else if (isBF16Input) {
            devMatElemTy = bf16Ty;
            devLoadName = "air.simdgroup_matrix_8x8_load.v64bf16.p1bf16";
            devMmaName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64bf16.v64bf16.v64f32";
            devGepElemTy = bf16Ty;
        }

        auto devMatTy = LLVM::getVectorType(devMatElemTy, 64);
        auto devLoadFn = getOrInsertIntrinsic(rewriter, mod, devLoadName,
            LLVMFunctionType::get(devMatTy, {devPtrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty}, false));
        auto devMmaFn = getOrInsertIntrinsic(rewriter, mod, devMmaName,
            LLVMFunctionType::get(matTy, {devMatTy, devMatTy, matTy}, false));

        // ── Constants ────────────────────────────────────────────────────

        Value fenceTG  = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
        Value execMod  = arith::ConstantIntOp::create(rewriter, loc, 4, 32);

        // ── Thread identification ─────────────────────────────────────────

        Value laneId = LLVM::CallOp::create(rewriter, loc, laneIdFn,
                                             ValueRange{}).getResult();

        auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
        auto tidFn = getOrInsertIntrinsic(rewriter, mod,
            "air.thread_position_in_threadgroup",
            LLVMFunctionType::get(arrI32x3Ty, {}, false));
        Value tidStruct = LLVM::CallOp::create(rewriter, loc, tidFn,
                                                ValueRange{}).getResult();
        Value tid32 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty,
                          tidStruct, ArrayRef<int64_t>{0});
        Value c32    = arith::ConstantIntOp::create(rewriter, loc, 32, 32);
        Value warpId = arith::DivUIOp::create(rewriter, loc, tid32, c32);

        // ── Get blocked encoding params for A, B ────────────────────────

        auto unpack = [&](Value v) -> SmallVector<Value> {
            SmallVector<Value> elems;
            if (auto sTy = dyn_cast<LLVMStructType>(v.getType())) {
                for (unsigned i = 0; i < sTy.getBody().size(); ++i)
                    elems.push_back(ExtractValueOp::create(rewriter, loc,
                        sTy.getBody()[i], v, ArrayRef<int64_t>{(int64_t)i}));
            } else {
                elems = {v};
            }
            return elems;
        };

        auto resolveOperand = [&](Value tritonVal, Value adaptorVal,
                                  RankedTensorType opTy)
            -> std::tuple<SmallVector<Value>,
                          SmallVector<SmallVector<unsigned>>,
                          ttg::BlockedEncodingAttr> {
            if (auto cvt = tritonVal.getDefiningOp<ttg::ConvertLayoutOp>()) {
                Value mapped = rewriter.getRemappedValue(cvt.getSrc());
                if (mapped) {
                    auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
                    auto srcEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
                    if (srcEnc) {
                        auto offsets = emitOffsetForLayout(srcEnc, srcTy);
                        return {unpack(mapped), offsets, srcEnc};
                    }
                }
            }
            auto enc = opTy.getEncoding();
            if (auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(enc)) {
                auto parentEnc = dyn_cast<ttg::BlockedEncodingAttr>(dotEnc.getParent());
                if (parentEnc) {
                    auto offsets = emitOffsetForLayout(enc, opTy);
                    return {unpack(adaptorVal), offsets, parentEnc};
                }
            }
            if (auto blk = dyn_cast<ttg::BlockedEncodingAttr>(enc)) {
                auto offsets = emitOffsetForLayout(blk, opTy);
                return {unpack(adaptorVal), offsets, blk};
            }
            return {{}, {}, nullptr};
        };

        // ── Try to extract device pointers for direct MMA loads ──────────
        // Trace: dot.A → ConvertLayoutOp → LoadOp → pointer operand
        // Returns per-thread device pointers with the same encoding/offsets as the values.
        auto resolveDevicePointers = [&](Value tritonVal)
            -> SmallVector<Value> {
            // Look through ConvertLayoutOp if present
            Value src = tritonVal;
            if (auto cvt = tritonVal.getDefiningOp<ttg::ConvertLayoutOp>())
                src = cvt.getSrc();
            // Check if source is a LoadOp
            auto loadOp = src.getDefiningOp<tt::LoadOp>();
            if (!loadOp)
                return {};
            // Get the pointer operand (tensor of device pointers)
            Value ptrTensor = loadOp.getPtr();
            Value mappedPtrs = rewriter.getRemappedValue(ptrTensor);
            if (!mappedPtrs)
                return {};
            return unpack(mappedPtrs);
        };

        auto [elemsA, aOffsets, aSrcEnc] = resolveOperand(op.getA(), adaptor.getA(), aType);
        auto [elemsB, bOffsets, bSrcEnc] = resolveOperand(op.getB(), adaptor.getB(), bType);
        auto elemsC = unpack(adaptor.getC());
        auto cOffsets = emitOffsetForLayout(cMmaEnc, cType);

        if (!aSrcEnc || !bSrcEnc)
            return failure();

        if ((int64_t)elemsA.size() != (int64_t)aOffsets.size() ||
            (int64_t)elemsB.size() != (int64_t)bOffsets.size() ||
            (int64_t)elemsC.size() != (int64_t)cOffsets.size())
            return failure();

        // Try to get device pointers for A and B
        auto aPtrs = resolveDevicePointers(op.getA());
        auto bPtrs = resolveDevicePointers(op.getB());
        bool useDeviceA = (aPtrs.size() == elemsA.size() && !aPtrs.empty());
        bool useDeviceB = (bPtrs.size() == elemsB.size() && !bPtrs.empty());

        // Env var TRITON_DEVICE_MMA=0 to force TG path (for debugging)
        {
            const char *envVar = std::getenv("TRITON_DEVICE_MMA");
            if (envVar && std::string(envVar) == "0") {
                useDeviceA = false;
                useDeviceB = false;
            }
        }

        // ── Compute row stride from pointer differences ──────────────────
        // For device MMA loads, we need the row stride (leading dimension)
        // in elements. Strategy:
        //   1. Try to find two elements on THIS thread with same col, different row.
        //   2. If not (common: all elements on same row), use simd_shuffle to
        //      get the same-column element from the thread in the NEXT row.
        //      For encoding spt=[1,8], tpw=[16,2], order=[1,0]:
        //        lR = laneId / tpw[colDim] = laneId / 2
        //        lC = laneId % tpw[colDim] = laneId % 2
        //      Thread laneId and laneId+tpw[colDim] differ by 1 row.
        //      Shuffle offset = tpw[colDim] for row-adjacent lane.
        auto computeRowStride = [&](SmallVector<Value> &ptrs,
                                    SmallVector<SmallVector<unsigned>> &offsets,
                                    Type elemTy,
                                    ttg::BlockedEncodingAttr enc) -> Value {
            unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;

            // Strategy 1: same-thread pair with same col, different row
            for (size_t i = 0; i < offsets.size(); ++i) {
                for (size_t j = i + 1; j < offsets.size(); ++j) {
                    if (offsets[i][1] == offsets[j][1] &&
                        offsets[i][0] != offsets[j][0]) {
                        int64_t rowDiff = (int64_t)offsets[j][0] - (int64_t)offsets[i][0];
                        Value ptrI64_i = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[i]);
                        Value ptrI64_j = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[j]);
                        Value byteDiff = arith::SubIOp::create(rewriter, loc, ptrI64_j, ptrI64_i);
                        Value elemSize = arith::ConstantIntOp::create(rewriter, loc, (int64_t)elemBytes, 64);
                        Value elemDiff = arith::DivSIOp::create(rewriter, loc, byteDiff, elemSize);
                        Value rowDiffVal = arith::ConstantIntOp::create(rewriter, loc, rowDiff, 64);
                        return arith::DivSIOp::create(rewriter, loc, elemDiff, rowDiffVal);
                    }
                }
            }

            // Strategy 2: simd_shuffle_xor to get pointer from adjacent-row thread.
            // For blocked encoding with order=[1,0] (col-fastest):
            //   XOR mask = threadsPerWarp[colDim] swaps row-adjacent threads
            // For order=[0,1] (row-fastest):
            //   XOR mask = 1 swaps row-adjacent threads
            auto order = enc.getOrder();
            auto tpw = enc.getThreadsPerWarp();
            auto spt = enc.getSizePerThread();
            unsigned encRank = spt.size();
            unsigned encColDim = encRank - 1;
            unsigned encRowDim = encRank - 2;
            bool colFastest = (order[0] == (unsigned)encColDim);

            // XOR mask that swaps row-adjacent threads
            int64_t xorMask = colFastest ? tpw[encColDim] : 1;
            int64_t rowDiff = spt[encRowDim];

            // Convert ptr to i64, shuffle_xor, compute stride.
            // Split i64 into two i32 for the shuffle (Metal doesn't support i64 shuffle).
            Value ptrI64 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[0]);
            auto i16Ty = IntegerType::get(ctx, 16);
            Value xorVal = arith::ConstantIntOp::create(rewriter, loc, xorMask, 16);

            // Split i64 into lo/hi i32, shuffle each, recombine
            Value lo = arith::TruncIOp::create(rewriter, loc, i32Ty, ptrI64);
            Value hi = arith::TruncIOp::create(rewriter, loc, i32Ty,
                arith::ShRUIOp::create(rewriter, loc, ptrI64,
                    arith::ConstantIntOp::create(rewriter, loc, 32, 64)));

            auto shuffleFn32 = getOrInsertIntrinsic(rewriter, mod,
                "air.simd_shuffle_xor.s.i32",
                LLVMFunctionType::get(i32Ty, {i32Ty, i16Ty}, false));
            Value shufLo = LLVM::CallOp::create(rewriter, loc, shuffleFn32,
                ValueRange{lo, xorVal}).getResult();
            Value shufHi = LLVM::CallOp::create(rewriter, loc, shuffleFn32,
                ValueRange{hi, xorVal}).getResult();

            // Recombine: result = zext(shuf_lo) | (zext(shuf_hi) << 32)
            Value loExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, shufLo);
            Value hiExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, shufHi);
            Value hiShl = arith::ShLIOp::create(rewriter, loc, hiExt,
                arith::ConstantIntOp::create(rewriter, loc, 32, 64));
            Value otherPtrI64 = arith::OrIOp::create(rewriter, loc, loExt, hiShl);

            // Compute absolute stride: abs(other - this) / rowDiff
            Value byteDiff = arith::SubIOp::create(rewriter, loc, otherPtrI64, ptrI64);
            // Take absolute value: if negative, negate
            Value zero64 = arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64);
            Value isNeg = arith::CmpIOp::create(rewriter, loc,
                arith::CmpIPredicate::slt, byteDiff, zero64);
            Value negDiff = arith::SubIOp::create(rewriter, loc, zero64, byteDiff);
            Value absDiff = arith::SelectOp::create(rewriter, loc, isNeg, negDiff, byteDiff);

            Value elemSize = arith::ConstantIntOp::create(rewriter, loc, (int64_t)elemBytes, 64);
            Value elemDiff = arith::DivUIOp::create(rewriter, loc, absDiff, elemSize);
            if (rowDiff > 1) {
                Value rowDiffVal = arith::ConstantIntOp::create(rewriter, loc, rowDiff, 64);
                elemDiff = arith::DivUIOp::create(rewriter, loc, elemDiff, rowDiffVal);
            }
            return elemDiff;
        };

        // Compute column stride (distance between adjacent columns in same row).
        // Mirrors computeRowStride but looks for same-row-different-col pairs.
        auto computeColStride = [&](SmallVector<Value> &ptrs,
                                    SmallVector<SmallVector<unsigned>> &offsets,
                                    Type elemTy,
                                    ttg::BlockedEncodingAttr enc) -> Value {
            unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;

            // Strategy 1: same-thread pair with same row, different col
            for (size_t i = 0; i < offsets.size(); ++i) {
                for (size_t j = i + 1; j < offsets.size(); ++j) {
                    if (offsets[i][0] == offsets[j][0] &&
                        offsets[i][1] != offsets[j][1]) {
                        int64_t colDiff = (int64_t)offsets[j][1] - (int64_t)offsets[i][1];
                        Value ptrI64_i = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[i]);
                        Value ptrI64_j = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[j]);
                        Value byteDiff = arith::SubIOp::create(rewriter, loc, ptrI64_j, ptrI64_i);
                        Value elemSize = arith::ConstantIntOp::create(rewriter, loc, (int64_t)elemBytes, 64);
                        Value elemDiff = arith::DivSIOp::create(rewriter, loc, byteDiff, elemSize);
                        Value colDiffVal = arith::ConstantIntOp::create(rewriter, loc, colDiff, 64);
                        return arith::DivSIOp::create(rewriter, loc, elemDiff, colDiffVal);
                    }
                }
            }

            // Strategy 2: simd_shuffle_xor to get pointer from adjacent-col thread.
            auto order = enc.getOrder();
            auto tpw = enc.getThreadsPerWarp();
            auto spt = enc.getSizePerThread();
            unsigned encRank = spt.size();
            unsigned encColDim = encRank - 1;
            unsigned encRowDim = encRank - 2;
            bool colFastest = (order[0] == (unsigned)encColDim);

            // XOR mask that swaps col-adjacent threads (opposite of row)
            int64_t xorMask = colFastest ? 1 : tpw[encRowDim];
            int64_t colDiff = spt[encColDim];

            Value ptrI64 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[0]);
            auto i16Ty = IntegerType::get(ctx, 16);
            Value xorVal = arith::ConstantIntOp::create(rewriter, loc, xorMask, 16);

            Value lo = arith::TruncIOp::create(rewriter, loc, i32Ty, ptrI64);
            Value hi = arith::TruncIOp::create(rewriter, loc, i32Ty,
                arith::ShRUIOp::create(rewriter, loc, ptrI64,
                    arith::ConstantIntOp::create(rewriter, loc, 32, 64)));

            auto shuffleFn32 = getOrInsertIntrinsic(rewriter, mod,
                "air.simd_shuffle_xor.s.i32",
                LLVMFunctionType::get(i32Ty, {i32Ty, i16Ty}, false));
            Value shufLo = LLVM::CallOp::create(rewriter, loc, shuffleFn32,
                ValueRange{lo, xorVal}).getResult();
            Value shufHi = LLVM::CallOp::create(rewriter, loc, shuffleFn32,
                ValueRange{hi, xorVal}).getResult();

            Value loExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, shufLo);
            Value hiExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, shufHi);
            Value hiShl = arith::ShLIOp::create(rewriter, loc, hiExt,
                arith::ConstantIntOp::create(rewriter, loc, 32, 64));
            Value otherPtrI64 = arith::OrIOp::create(rewriter, loc, loExt, hiShl);

            Value byteDiff = arith::SubIOp::create(rewriter, loc, otherPtrI64, ptrI64);
            Value zero64 = arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64);
            Value isNeg = arith::CmpIOp::create(rewriter, loc,
                arith::CmpIPredicate::slt, byteDiff, zero64);
            Value negDiff = arith::SubIOp::create(rewriter, loc, zero64, byteDiff);
            Value absDiff = arith::SelectOp::create(rewriter, loc, isNeg, negDiff, byteDiff);

            Value elemSize = arith::ConstantIntOp::create(rewriter, loc, (int64_t)elemBytes, 64);
            Value elemDiff = arith::DivUIOp::create(rewriter, loc, absDiff, elemSize);
            if (colDiff > 1) {
                Value colDiffVal = arith::ConstantIntOp::create(rewriter, loc, colDiff, 64);
                elemDiff = arith::DivUIOp::create(rewriter, loc, elemDiff, colDiffVal);
            }
            return elemDiff;
        };

        Value aRowStride, bRowStride, aColStride, bColStride;
        if (useDeviceA) {
            aRowStride = computeRowStride(aPtrs, aOffsets, aType.getElementType(), aSrcEnc);
            aColStride = computeColStride(aPtrs, aOffsets, aType.getElementType(), aSrcEnc);
        }
        if (useDeviceB) {
            bRowStride = computeRowStride(bPtrs, bOffsets, bType.getElementType(), bSrcEnc);
            bColStride = computeColStride(bPtrs, bOffsets, bType.getElementType(), bSrcEnc);
        }

        // If stride computation failed, fall back to TG path
        if (useDeviceA && (!aRowStride || !aColStride)) useDeviceA = false;
        if (useDeviceB && (!bRowStride || !bColStride)) useDeviceB = false;

        // ── Compute device base pointer for each 8x8 MMA tile ───────────
        // tile_base = ptr[0] - (aOffsets[0][0] * rowStride + aOffsets[0][1]) * elemSize
        // Then offset by (tm*8 * rowStride + tk*8) * elemSize for each tile.
        //
        // For the MMA load, we pass the pointer as float* regardless of the
        // actual element type (the intrinsic always loads as f32). But the
        // stride and offset must be in elements of the SOURCE type, so we
        // convert the base pointer to the proper byte address.
        // Compute tile base pointer for MMA load.
        // ALL threads must compute the SAME base pointer.
        // ptrs[0] points to the thread's own element at (baseRow+offsets[0][0],
        // baseCol+offsets[0][1]). Subtract this position to get the tile origin.
        //
        // tile_ptr = ptrs[0] + (tileRow - baseRow - offsets[0][0]) * rowStride
        //                    + (tileCol - baseCol - offsets[0][1])
        auto computeTileDevPtr = [&](SmallVector<Value> &ptrs,
                                     SmallVector<SmallVector<unsigned>> &offsets,
                                     Value rowStride, Value colStride,
                                     Value baseRow, Value baseCol,
                                     int64_t tileRow, int64_t tileCol) -> Value {
            Value refPtr = ptrs[0];
            int64_t refRowOff = offsets[0][0];
            int64_t refColOff = offsets[0][1];

            // rowDelta = tileRow - baseRow - refRowOff (runtime)
            Value baseRowExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, baseRow);
            Value rowDelta = arith::SubIOp::create(rewriter, loc,
                arith::ConstantIntOp::create(rewriter, loc, (int64_t)tileRow - refRowOff, 64),
                baseRowExt);

            // colDelta = tileCol - baseCol - refColOff (runtime)
            Value baseColExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, baseCol);
            Value colDelta = arith::SubIOp::create(rewriter, loc,
                arith::ConstantIntOp::create(rewriter, loc, (int64_t)tileCol - refColOff, 64),
                baseColExt);

            // elemOff = rowDelta * rowStride + colDelta * colStride
            Value elemOff = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, rowDelta, rowStride),
                arith::MulIOp::create(rewriter, loc, colDelta, colStride));

            Value tilePtr = LLVM::GEPOp::create(rewriter, loc, devPtrTy,
                devGepElemTy, refPtr, ArrayRef<LLVM::GEPArg>{elemOff});
            return tilePtr;
        };

        // ── Make MMA stride vector for device loads ──────────────────────
        // stride = <colStride, rowStride>
        // For row-major data: colStride=1, rowStride=numCols
        // For col-major data: colStride=numRows, rowStride=1
        auto makeDevMmaStride = [&](Value colStride, Value rowStride) -> Value {
            auto ty = LLVM::getVectorType(IntegerType::get(ctx, 64), 2);
            Value vec = UndefOp::create(rewriter, loc, ty);
            Value i0 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
            Value i1 = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
            vec = InsertElementOp::create(rewriter, loc, ty, vec, colStride, i0);
            vec = InsertElementOp::create(rewriter, loc, ty, vec, rowStride, i1);
            return vec;
        };

        // ── Compute runtime thread base position ──────────────────────────
        auto makeBase = [&](ttg::BlockedEncodingAttr enc, int64_t rows, int64_t cols)
            -> std::pair<Value, Value> {
            auto spt = enc.getSizePerThread();
            auto tpw = enc.getThreadsPerWarp();
            auto wpc = enc.getWarpsPerCTA();
            auto order = enc.getOrder();

            unsigned encRank = spt.size();
            unsigned encRowDim = encRank - 2;
            unsigned encColDim = encRank - 1;

            int64_t sM = spt[encRowDim], sN = spt[encColDim];
            int64_t tM = tpw[encRowDim], tN = tpw[encColDim];
            int64_t wM = wpc[encRowDim], wN = wpc[encColDim];
            int64_t tileM = wM * tM * sM;
            int64_t tileN = wN * tN * sN;

            bool colFastest = (order[0] == (unsigned)encColDim);

            Value wN_val  = arith::ConstantIntOp::create(rewriter, loc, wN, 32);
            Value tN_val  = arith::ConstantIntOp::create(rewriter, loc, tN, 32);
            Value tMsM    = arith::ConstantIntOp::create(rewriter, loc, tM * sM, 32);
            Value sM_val  = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
            Value tNsN    = arith::ConstantIntOp::create(rewriter, loc, tN * sN, 32);
            Value sN_val  = arith::ConstantIntOp::create(rewriter, loc, sN, 32);

            Value wR, wC;
            if (colFastest) {
                wR = arith::DivUIOp::create(rewriter, loc, warpId, wN_val);
                wC = arith::RemUIOp::create(rewriter, loc, warpId, wN_val);
            } else {
                Value wM_val = arith::ConstantIntOp::create(rewriter, loc, wM, 32);
                wR = arith::RemUIOp::create(rewriter, loc, warpId, wM_val);
                wC = arith::DivUIOp::create(rewriter, loc, warpId, wM_val);
            }
            Value lR, lC;
            if (colFastest) {
                lR = arith::DivUIOp::create(rewriter, loc, laneId, tN_val);
                lC = arith::RemUIOp::create(rewriter, loc, laneId, tN_val);
            } else {
                Value tM_val = arith::ConstantIntOp::create(rewriter, loc, tM, 32);
                lR = arith::RemUIOp::create(rewriter, loc, laneId, tM_val);
                lC = arith::DivUIOp::create(rewriter, loc, laneId, tM_val);
            }

            Value baseRow = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, wR, tMsM),
                arith::MulIOp::create(rewriter, loc, lR, sM_val));
            Value baseCol = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, wC, tNsN),
                arith::MulIOp::create(rewriter, loc, lC, sN_val));

            if (tileM > rows) {
                Value modRow = arith::ConstantIntOp::create(rewriter, loc, rows, 32);
                baseRow = arith::RemUIOp::create(rewriter, loc, baseRow, modRow);
            }
            if (tileN > cols) {
                Value modCol = arith::ConstantIntOp::create(rewriter, loc, cols, 32);
                baseCol = arith::RemUIOp::create(rewriter, loc, baseCol, modCol);
            }

            return {baseRow, baseCol};
        };

        // MMA base: lane->(row,col) within 8x8 tile, warp->tile position
        auto makeBaseMma = [&](AppleMmaEncodingAttr enc, int64_t rows, int64_t cols)
            -> std::pair<Value, Value> {
            auto wpc = enc.getWarpsPerCTA();
            unsigned wN = wpc[1];

            Value c7    = arith::ConstantIntOp::create(rewriter, loc, 7, 32);
            Value c3    = arith::ConstantIntOp::create(rewriter, loc, 3, 32);
            Value laneCol = arith::AndIOp::create(rewriter, loc, laneId, c7);
            Value laneRow = arith::ShRUIOp::create(rewriter, loc, laneId, c3);

            Value wN_val  = arith::ConstantIntOp::create(rewriter, loc, wN, 32);
            Value c8      = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
            Value warpRow = arith::DivUIOp::create(rewriter, loc, warpId, wN_val);
            Value warpCol = arith::RemUIOp::create(rewriter, loc, warpId, wN_val);

            Value baseRow = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, warpRow, c8), laneRow);
            Value baseCol = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, warpCol, c8), laneCol);

            return {baseRow, baseCol};
        };

        auto [aBaseRow, aBaseCol] = makeBase(aSrcEnc, M, K);
        auto [bBaseRow, bBaseCol] = makeBase(bSrcEnc, K, N);
        auto [cBaseRow, cBaseCol] = makeBaseMma(cMmaEnc, M, N);

        // ── Create threadgroup global ─────────────────────────────────────
        // TG buffer is needed for: C scatter/load (always), and A/B scatter
        // (only when device path is not available).
        unsigned id = getDotCounter(ctx)++;
        int64_t tgCStripSize = 8 * N;  // C always needs N-wide strips
        int64_t tgABStripSize = 8 * std::max(K, N);
        int64_t tgStripSize = (useDeviceA && useDeviceB) ? tgCStripSize : 8 * std::max(K, N);
        int64_t tgSize = tgStripSize + 1;
        auto tgBuf = getOrCreateTGGlobal(rewriter, mod,
            ("__tg_dot_ab_" + llvm::Twine(id)).str(), tgSize);

        Value ptrTG = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgBuf.getName());

        // ── GEP helpers ───────────────────────────────────────────────────

        auto gather1 = [&](Value ptr, Value flatIdx64) -> Value {
            Value gep = LLVM::GEPOp::create(rewriter, loc,
                tgPtrTy, f32Ty, ptr, ArrayRef<LLVM::GEPArg>{flatIdx64});
            return LLVM::LoadOp::create(rewriter, loc, f32Ty, gep).getResult();
        };

        auto stripFlatIdx = [&](Value baseRow, Value baseCol,
                                int64_t rowOff, int64_t colOff,
                                int64_t stride, int64_t stripRowStart) -> Value {
            Value row32 = arith::AddIOp::create(rewriter, loc, baseRow,
                arith::ConstantIntOp::create(rewriter, loc, rowOff - stripRowStart, 32));
            Value col32 = arith::AddIOp::create(rewriter, loc, baseCol,
                arith::ConstantIntOp::create(rewriter, loc, colOff, 32));
            Value flat32 = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, row32,
                    arith::ConstantIntOp::create(rewriter, loc, stride, 32)),
                col32);
            return arith::ExtUIOp::create(rewriter, loc, i64Ty, flat32);
        };

        int64_t tilesM = M / 8;
        int64_t tilesN = N / 8;
        int64_t tilesK = K / 8;

        // ── Static strip filtering ────────────────────────────────────────
        auto maxBaseRow = [](ttg::BlockedEncodingAttr enc) -> int64_t {
            auto spt = enc.getSizePerThread();
            auto tpw = enc.getThreadsPerWarp();
            auto wpc = enc.getWarpsPerCTA();
            unsigned encRank = spt.size();
            unsigned encRowDim = encRank - 2;
            return wpc[encRowDim] * tpw[encRowDim] * spt[encRowDim] - 1;
        };

        // For MMA encoding, max base row = warpsM * 8 - 1
        int64_t cMaxBase = cMmaEnc.getWarpsPerCTA()[0] * 8 - 1;

        auto bucketElements = [](SmallVector<SmallVector<unsigned>> &offsets,
                                 int64_t maxBase, int64_t numStrips,
                                 unsigned rowIdx)
            -> SmallVector<SmallVector<size_t>> {
            SmallVector<SmallVector<size_t>> buckets(numStrips);
            for (size_t i = 0; i < offsets.size(); ++i) {
                int64_t rowOff = offsets[i][rowIdx];
                for (int64_t s = 0; s < numStrips; ++s) {
                    int64_t stripStart = s * 8;
                    int64_t lo = stripStart - rowOff;
                    int64_t hi = lo + 8;
                    if (lo <= maxBase && hi > 0)
                        buckets[s].push_back(i);
                }
            }
            return buckets;
        };

        auto cBuckets = bucketElements(cOffsets, cMaxBase, tilesM, 0);

        Value garbageIdx = arith::ConstantIntOp::create(rewriter, loc, tgStripSize, 64);

        // ── Phase 1: Load C tiles (filtered strip scatter — always via TG) ──
        // C accumulator comes from the previous iteration or zero init.
        // It uses AppleMmaEncoding, so we always go through TG scatter/load.

        // For C, we still need aBaseRow/Col for scatter — but only when NOT
        // using device path for A/B. Actually, C always uses TG regardless.
        // But we need cBaseRow/cBaseCol from makeBaseMma (already computed).

        // filteredScatter needs baseRow/baseCol. For C, use MMA-derived base.
        auto filteredScatter = [&](Value ptr, Value garbIdx,
                                   Value baseRow, Value baseCol,
                                   SmallVector<Value> &elems,
                                   SmallVector<SmallVector<unsigned>> &offsets,
                                   SmallVector<size_t> &bucket,
                                   int64_t stride, int64_t rowStart) {
            for (size_t idx : bucket) {
                int64_t rowOff = offsets[idx][0];
                int64_t colOff = offsets[idx][1];
                Value actualRow = arith::AddIOp::create(rewriter, loc, baseRow,
                    arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
                Value inStrip = arith::AndIOp::create(rewriter, loc,
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                        actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                        actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart + 8, 32)));
                Value sIdx = stripFlatIdx(baseRow, baseCol, rowOff, colOff, stride, rowStart);
                Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip, sIdx, garbIdx);
                Value val = elems[idx];
                if (val.getType() != f32Ty)
                    val = toF32(rewriter, loc, val, f32Ty);
                Value gep = LLVM::GEPOp::create(rewriter, loc,
                    tgPtrTy, f32Ty, ptr, ArrayRef<LLVM::GEPArg>{safeIdx});
                LLVM::StoreOp::create(rewriter, loc, val, gep);
            }
        };

        SmallVector<SmallVector<Value>> matC_tiles(tilesM);
        for (int64_t tm = 0; tm < tilesM; ++tm) {
            matC_tiles[tm].resize(tilesN);
            int64_t rowStart = tm * 8;

            filteredScatter(ptrTG, garbageIdx, cBaseRow, cBaseCol,
                            elemsC, cOffsets, cBuckets[tm], N, rowStart);
            LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

            Value cStride = makeI64Vec2(rewriter, loc, 1, N);
            Value cShape  = makeI64Vec2(rewriter, loc, N, 8);
            for (int64_t tn = 0; tn < tilesN; ++tn) {
                Value cOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
                matC_tiles[tm][tn] = LLVM::CallOp::create(rewriter, loc, loadFn,
                    ValueRange{ptrTG, cShape, cStride, cOff}).getResult();
            }
            LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
        }

        // ── Phase 2: A/B loads + MMA ──────────────────────────────────
        if (useDeviceA && useDeviceB) {
            // ── DEVICE PATH: Direct MMA loads from device memory ──────────
            // No TG scatter, no barriers for A/B. Just compute tile pointers
            // and call the p1f32 MMA load intrinsic directly.
            Value aDevStride = makeDevMmaStride(aColStride, aRowStride);
            Value bDevStride = makeDevMmaStride(bColStride, bRowStride);
            Value mmaShape = makeI64Vec2(rewriter, loc, 8, 8);
            Value zeroOff = makeI64Vec2(rewriter, loc, 0, 0);

            for (int64_t tk = 0; tk < tilesK; ++tk) {
                // Load A tiles for this K strip directly from device memory
                SmallVector<Value> matA_strip(tilesM);
                for (int64_t tm = 0; tm < tilesM; ++tm) {
                    Value aTilePtr = computeTileDevPtr(aPtrs, aOffsets,
                        aRowStride, aColStride, aBaseRow, aBaseCol, tm * 8, tk * 8);
                    matA_strip[tm] = LLVM::CallOp::create(rewriter, loc, devLoadFn,
                        ValueRange{aTilePtr, mmaShape, aDevStride, zeroOff}).getResult();
                }

                // Load B tiles and accumulate
                for (int64_t tn = 0; tn < tilesN; ++tn) {
                    Value bTilePtr = computeTileDevPtr(bPtrs, bOffsets,
                        bRowStride, bColStride, bBaseRow, bBaseCol, tk * 8, tn * 8);
                    Value matB = LLVM::CallOp::create(rewriter, loc, devLoadFn,
                        ValueRange{bTilePtr, mmaShape, bDevStride, zeroOff}).getResult();

                    for (int64_t tm = 0; tm < tilesM; ++tm) {
                        // Use type-specific MMA: devMmaFn handles f16*f16+f32 or f32*f32+f32
                        matC_tiles[tm][tn] = LLVM::CallOp::create(rewriter, loc, devMmaFn,
                            ValueRange{matA_strip[tm], matB, matC_tiles[tm][tn]}).getResult();
                    }
                }
            }
        } else {
            // ── TG PATH: Original scatter/load through threadgroup memory ──
            // Used when device pointers are not available (e.g., operand is
            // not directly from a load, or stride couldn't be computed).
            // aBaseRow/aBaseCol/bBaseRow/bBaseCol already computed above.

            int64_t aMaxBase = maxBaseRow(aSrcEnc);
            int64_t bMaxBase = maxBaseRow(bSrcEnc);
            auto aBuckets = bucketElements(aOffsets, aMaxBase, tilesM, 0);
            auto bBuckets = bucketElements(bOffsets, bMaxBase, tilesK, 0);

            for (int64_t tk = 0; tk < tilesK; ++tk) {
                SmallVector<Value> matA_strip(tilesM);
                {
                    for (int64_t tm = 0; tm < tilesM; ++tm) {
                        int64_t rowStart = tm * 8;
                        filteredScatter(ptrTG, garbageIdx, aBaseRow, aBaseCol,
                                        elemsA, aOffsets, aBuckets[tm], K, rowStart);
                        LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

                        Value aOff = makeI64Vec2(rewriter, loc, tk * 8, 0);
                        Value aStride = makeI64Vec2(rewriter, loc, 1, K);
                        Value aShape  = makeI64Vec2(rewriter, loc, K, 8);
                        matA_strip[tm] = LLVM::CallOp::create(rewriter, loc, loadFn,
                            ValueRange{ptrTG, aShape, aStride, aOff}).getResult();
                        LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
                    }
                }

                {
                    int64_t rowStart = tk * 8;
                    filteredScatter(ptrTG, garbageIdx, bBaseRow, bBaseCol,
                                    elemsB, bOffsets, bBuckets[tk], N, rowStart);
                    LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

                    Value bStride = makeI64Vec2(rewriter, loc, 1, N);
                    Value bShape  = makeI64Vec2(rewriter, loc, N, 8);
                    for (int64_t tn = 0; tn < tilesN; ++tn) {
                        Value bOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
                        Value matB = LLVM::CallOp::create(rewriter, loc, loadFn,
                            ValueRange{ptrTG, bShape, bStride, bOff}).getResult();

                        for (int64_t tm = 0; tm < tilesM; ++tm) {
                            matC_tiles[tm][tn] = LLVM::CallOp::create(rewriter, loc, mmaFn,
                                ValueRange{matA_strip[tm], matB, matC_tiles[tm][tn]}).getResult();
                        }
                    }
                    LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
                }
            }
        }

        // ── Phase 4: Store C tiles -> TG, gather ─────────────────────────
        auto outElemTy = cType.getElementType();
        SmallVector<Value> resultElems(elemsC.size());
        for (size_t i = 0; i < elemsC.size(); ++i)
            resultElems[i] = arith::ConstantOp::create(rewriter, loc,
                rewriter.getZeroAttr(outElemTy));

        for (int64_t tm = 0; tm < tilesM; ++tm) {
            int64_t rowStart = tm * 8;

            Value cStoreStride = makeI64Vec2(rewriter, loc, 1, N);
            Value cStoreShape  = makeI64Vec2(rewriter, loc, N, 8);
            for (int64_t tn = 0; tn < tilesN; ++tn) {
                Value cOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
                LLVM::CallOp::create(rewriter, loc, storeFn,
                    ValueRange{matC_tiles[tm][tn], ptrTG, cStoreShape, cStoreStride, cOff});
            }
            LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

            for (size_t idx : cBuckets[tm]) {
                int64_t rowOff = cOffsets[idx][0];
                int64_t colOff = cOffsets[idx][1];
                Value actualRow = arith::AddIOp::create(rewriter, loc, cBaseRow,
                    arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
                Value inStrip = arith::AndIOp::create(rewriter, loc,
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                        actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                        actualRow, arith::ConstantIntOp::create(rewriter, loc, rowStart + 8, 32)));
                Value sIdx = stripFlatIdx(cBaseRow, cBaseCol, rowOff, colOff, N, rowStart);
                Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip, sIdx, garbageIdx);
                Value val = gather1(ptrTG, safeIdx);
                if (val.getType() != outElemTy)
                    val = fromF32(rewriter, loc, val, outElemTy);
                resultElems[idx] = arith::SelectOp::create(rewriter, loc, inStrip,
                    val, resultElems[idx]);
            }
            LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
        }

        // ── Pack result ───────────────────────────────────────────────────
        auto outLLVMTy = getTypeConverter()->convertType(cType);
        if (!outLLVMTy) return failure();

        if (auto outStructTy = dyn_cast<LLVMStructType>(outLLVMTy)) {
            Value result = UndefOp::create(rewriter, loc, outStructTy);
            for (size_t i = 0; i < resultElems.size(); ++i)
                result = InsertValueOp::create(rewriter, loc, outStructTy,
                             result, resultElems[i],
                             ArrayRef<int64_t>{(int64_t)i});
            rewriter.replaceOp(op, result);
        } else {
            rewriter.replaceOp(op, resultElems[0]);
        }
        return success();
    }
};

} // anonymous namespace

namespace mlir::triton::applegpu {

void populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    PatternBenefit benefit) {
    patterns.add<DotOpBlockedConversion>(typeConverter, benefit);
    patterns.add<DotOpAppleMmaConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::applegpu
