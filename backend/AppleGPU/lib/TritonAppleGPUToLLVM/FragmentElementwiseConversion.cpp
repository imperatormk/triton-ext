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

// Fragment ABI elementwise: apply the op per fragment vector (<64 x f32>).
// Identical struct layout across same-#mma operands means a binary op is
// lane-wise per matching slot, a unary op maps over slots. Fires only when the
// operand/result types are the fragment struct; else defers to flat lowering.
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

// f16/bf16 epilogue truncf on a fragment struct: a no-op forward. The result
// rides the same <64 x f32> accumulator fragment; narrowing happens per-element
// in the #mma->#blocked store convert. Matches only same f32 in/out struct.
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

// Fragment-ABI integer/mask + view lowerings: keep i32/i1/f32 temporaries on
// the simdgroup register path so the accumulator stays in the <64 x ELT>
// fragment struct. expand_dims = local reg-pack; broadcast = scalar
// simd_shuffle replicate; cmpi/andi/select = per-slot lane-wise vector ops.

static LLVMStructType fragStructOf(Type t) {
  auto s = dyn_cast_or_null<LLVMStructType>(t);
  if (s && !s.getBody().empty() && isa<VectorType>(s.getBody()[0]))
    return s;
  return nullptr;
}
// expand_dims: slice<#mma> (flat per-thread scalars) → #mma fragment struct.
// Layout-preserving: pack flat element k into its (row_k,col_k) fragment slot.
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

    // Enumerate the SLICE source's per-thread offsets (1D, kept dim) - these
    // match the flat source element count - and insert the expand axis to find
    // each element's (row,col) home. (The result's own emitOffsetForLayout
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

} // anonymous namespace

void populateFragmentElementwisePatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
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
}

} // namespace mlir::triton::applegpu
