// LoadStoreToLLVM.cpp — Lower tt.addptr to LLVM IR
//
// Handles both scalar and blocked-tensor (struct-of-pointers) paths.

#include "TritonAppleGPUToLLVM/Passes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::triton;

namespace mlir::triton::applegpu {

namespace {

// Unpack an LLVM struct into individual values, or return {v} if scalar.
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

// Pack values into an LLVM struct. Always packs (even 1 element) to match the
// type converter's struct<(T)> expectation for 1-element tensors.
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

// tt.addptr %base, %offset → GEP per element (scalar or struct path)
struct AddPtrOpConversion : public ConvertOpToLLVMPattern<triton::AddPtrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value base = adaptor.getPtr();
    Value offset = adaptor.getOffset();

    // Determine element type from the original Triton pointer type
    auto srcTy = op.getPtr().getType();
    Type elemTy;
    if (auto ptrTy = dyn_cast<triton::PointerType>(srcTy)) {
      elemTy = getTypeConverter()->convertType(ptrTy.getPointeeType());
    } else {
      // tensor<Nx!tt.ptr<T>> — element type is T
      auto tensorTy = cast<RankedTensorType>(srcTy);
      auto tPtrTy = cast<triton::PointerType>(tensorTy.getElementType());
      elemTy = getTypeConverter()->convertType(tPtrTy.getPointeeType());
    }
    if (!elemTy)
      return failure();

    // Scalar addptr: bare pointer + scalar offset → single GEP.
    if (!isa<LLVMStructType>(base.getType()) &&
        !isa<LLVMStructType>(offset.getType())) {
      Value gep = LLVM::GEPOp::create(rewriter, loc, base.getType(), elemTy,
                                      base, ArrayRef<LLVM::GEPArg>{offset});
      rewriter.replaceOp(op, gep);
      return success();
    }

    auto basePtrs = unpackElems(base, rewriter, loc);
    auto offsets = unpackElems(offset, rewriter, loc);

    // Broadcast scalar base to match tensor offsets (e.g. addptr(scalar_ptr,
    // tensor_offsets))
    if (basePtrs.size() == 1 && offsets.size() > 1) {
      Value scalarBase = basePtrs[0];
      basePtrs.clear();
      basePtrs.resize(offsets.size(), scalarBase);
    }
    // Broadcast scalar offset to match tensor base
    if (offsets.size() == 1 && basePtrs.size() > 1) {
      Value scalarOff = offsets[0];
      offsets.clear();
      offsets.resize(basePtrs.size(), scalarOff);
    }

    if (basePtrs.size() != offsets.size())
      return failure();

    SmallVector<Value> results;
    for (size_t i = 0; i < basePtrs.size(); ++i) {
      results.push_back(
          LLVM::GEPOp::create(rewriter, loc, basePtrs[i].getType(), elemTy,
                              basePtrs[i], ArrayRef<LLVM::GEPArg>{offsets[i]}));
    }

    rewriter.replaceOp(op, packElems(results, rewriter, loc));
    return success();
  }
};

} // anonymous namespace

void populateLoadStoreToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::applegpu
