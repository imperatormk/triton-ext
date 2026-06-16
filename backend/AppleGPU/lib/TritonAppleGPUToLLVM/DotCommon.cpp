// Shared helper definitions for the tt.dot -> AIR simdgroup-matrix lowerings.
#include "DotCommon.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdlib>

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;

namespace mlir::triton::applegpu::dotcommon {

bool isPowerOf2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }
unsigned log2(int64_t v) {
  unsigned r = 0;
  while ((1LL << r) < v)
    ++r;
  return r;
}

Value divByConst(OpBuilder &b, Location loc, Value v, int64_t c) {
  if (isPowerOf2(c)) {
    unsigned shift = log2(c);
    if (shift == 0)
      return v;
    Value shAmt = arith::ConstantIntOp::create(b, loc, shift, 32);
    return arith::ShRUIOp::create(b, loc, v, shAmt);
  }
  Value cv = arith::ConstantIntOp::create(b, loc, c, 32);
  return arith::DivUIOp::create(b, loc, v, cv);
}

Value remByConst(OpBuilder &b, Location loc, Value v, int64_t c) {
  if (isPowerOf2(c)) {
    Value mask = arith::ConstantIntOp::create(b, loc, c - 1, 32);
    return arith::AndIOp::create(b, loc, v, mask);
  }
  Value cv = arith::ConstantIntOp::create(b, loc, c, 32);
  return arith::RemUIOp::create(b, loc, v, cv);
}

int64_t tgPadForType(Type elemTy) {
  if (isa<IntegerType>(elemTy))
    return 0; // no padding for int8/int16/int32
  unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
  if (elemBytes == 0)
    return TG_PAD;       // fallback
  return 16 / elemBytes; // 4 for f32, 8 for f16/bf16
}

int64_t paddedStride(int64_t stride, int64_t budget_bytes, int64_t pad) {
  if (pad == 0)
    return stride;
  if (stride % 8 != 0)
    return stride; // odd strides don't bank-conflict
  int64_t padded = stride + pad;
  // 8 rows * padded * 4 bytes/f32 must fit in budget
  if (8 * padded * 4 > budget_bytes)
    return stride;
  return padded;
}

Type getSimdgroupMatrixType(MLIRContext *ctx) {
  return LLVM::getVectorType(Float32Type::get(ctx), 64);
}

Value makeI64Vec2(OpBuilder &b, Location loc, int64_t a, int64_t b_val) {
  auto ty = LLVM::getVectorType(IntegerType::get(b.getContext(), 64), 2);
  Value vec = UndefOp::create(b, loc, ty);
  Value va = arith::ConstantIntOp::create(b, loc, a, 64);
  Value vb = arith::ConstantIntOp::create(b, loc, b_val, 64);
  Value i0 = arith::ConstantIntOp::create(b, loc, 0, 32);
  Value i1 = arith::ConstantIntOp::create(b, loc, 1, 32);
  vec = InsertElementOp::create(b, loc, ty, vec, va, i0);
  vec = InsertElementOp::create(b, loc, ty, vec, vb, i1);
  return vec;
}

Value makeI64(OpBuilder &b, Location loc, int64_t v) {
  return arith::ConstantIntOp::create(b, loc, v, 64);
}
Value makeI1False(OpBuilder &b, Location loc) {
  return arith::ConstantIntOp::create(b, loc, 0, 1);
}

unsigned getTargetOSMajor() {
  if (const char *e = std::getenv("TRITON_MPS_TARGET_OS_MAJOR")) {
    unsigned v = std::atoi(e);
    if (v)
      return v;
  }
  return 16; // default = current shipping target (3-vector signature)
}
bool useCanonicalSimdgroupSig() { return getTargetOSMajor() <= 15; }

LLVMFuncOp getOrInsertIntrinsic(ConversionPatternRewriter &rewriter,
                                ModuleOp mod, StringRef name,
                                LLVMFunctionType fnTy) {
  if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto fn =
      LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy, Linkage::External);
  // The simdgroup-matrix intrinsics need the same attributes Apple's `xcrun
  // metal` emits — the macOS 13/14/15 Metal driver rejects the declarations
  // otherwise ("Compiler encountered an internal error"); macOS 26 is lenient.
  // The attributes are universal (Apple emits them on all OSes), so safe both.
  if (name.contains("simdgroup")) {
    auto *ctx = mod.getContext();
    auto unit = UnitAttr::get(ctx);
    bool isLoad = name.contains("_load");
    bool isStore = name.contains("_store");

    SmallVector<Attribute> pass;
    auto add = [&](StringRef kw) { pass.push_back(StringAttr::get(ctx, kw)); };
    add("convergent");
    add("mustprogress");
    if (isLoad)
      add("nofree");
    add("nounwind");
    add("willreturn");
    fn.setPassthroughAttr(ArrayAttr::get(ctx, pass));

    // readonly/writeonly: not valid as function keyword attrs in modern LLVM,
    // expressed via memory effects instead (semantics identical).
    if (isLoad || isStore) {
      auto mr = isStore ? LLVM::ModRefInfo::Mod : LLVM::ModRefInfo::Ref;
      fn.setMemoryEffectsAttr(
          LLVM::MemoryEffectsAttr::get(ctx, {mr, mr, mr, mr, mr, mr}));
    }

    fn.setUnnamedAddr(LLVM::UnnamedAddr::Local);

    // Matrix pointer arg: nocapture readonly (load) / nocapture writeonly
    // (store). Load ptr is arg 0; store ptr is arg 1.
    if (isLoad || isStore) {
      unsigned ptrArg = isStore ? 1u : 0u;
      unsigned nArgs = fnTy.getNumParams();
      SmallVector<Attribute> argDicts(nArgs, DictionaryAttr::get(ctx, {}));
      SmallVector<NamedAttribute> ptrAttrs;
      ptrAttrs.push_back(NamedAttribute(
          StringAttr::get(ctx, LLVM::LLVMDialect::getNoCaptureAttrName()),
          unit));
      ptrAttrs.push_back(NamedAttribute(
          StringAttr::get(ctx, isStore
                                   ? LLVM::LLVMDialect::getWriteOnlyAttrName()
                                   : LLVM::LLVMDialect::getReadonlyAttrName()),
          unit));
      argDicts[ptrArg] = DictionaryAttr::get(ctx, ptrAttrs);
      fn.setArgAttrsAttr(ArrayAttr::get(ctx, argDicts));
    }
  }
  return fn;
}

Value getOrInsertSimdgroupInitFilled(ConversionPatternRewriter &rewriter,
                                     Location loc, ModuleOp mod, float fill) {
  auto *ctx = mod.getContext();
  auto f32Ty = Float32Type::get(ctx);
  auto matTy = getSimdgroupMatrixType(ctx);
  auto initFn = getOrInsertIntrinsic(
      rewriter, mod, "air.simdgroup_matrix_8x8_init_filled.v64f32.f32",
      LLVMFunctionType::get(matTy, {f32Ty}, false));
  Value fz =
      arith::ConstantOp::create(rewriter, loc, rewriter.getF32FloatAttr(fill));
  return LLVM::CallOp::create(rewriter, loc, initFn, ValueRange{fz})
      .getResult();
}

LLVM::GlobalOp getOrCreateTGGlobal(ConversionPatternRewriter &rewriter,
                                   ModuleOp mod, StringRef name, int64_t size) {
  if (auto g = mod.lookupSymbol<LLVM::GlobalOp>(name))
    return g;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto f32Ty = Float32Type::get(mod.getContext());
  auto arrTy = LLVMArrayType::get(f32Ty, size);
  return LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy,
                                /*isConstant=*/false, LLVM::Linkage::Internal,
                                name,
                                /*value=*/Attribute(),
                                /*alignment=*/4,
                                /*addrspace=*/3u);
}

LLVM::GlobalOp getOrCreateTypedTGGlobal(ConversionPatternRewriter &rewriter,
                                        ModuleOp mod, StringRef name,
                                        int64_t numElements, Type elemTy) {
  if (auto g = mod.lookupSymbol<LLVM::GlobalOp>(name))
    return g;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto arrTy = LLVMArrayType::get(elemTy, numElements);
  unsigned alignment = isa<Float32Type>(elemTy) ? 4 : 2;
  return LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy,
                                /*isConstant=*/false, LLVM::Linkage::Internal,
                                name,
                                /*value=*/Attribute(),
                                /*alignment=*/alignment,
                                /*addrspace=*/3u);
}

MMAIntrinsicInfo getMMAIntrinsicInfo(MLIRContext *ctx, Type elemTy) {
  MMAIntrinsicInfo info;
  if (elemTy.isBF16()) {
    info.tgLoadName = "air.simdgroup_matrix_8x8_load.v64bf16.p3bf16";
    info.mmaName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32."
                   "v64bf16.v64bf16.v64f32";
    info.matVecTy = LLVM::getVectorType(BFloat16Type::get(ctx), 64);
  } else if (elemTy.isF16()) {
    info.tgLoadName = "air.simdgroup_matrix_8x8_load.v64f16.p3f16";
    info.mmaName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16."
                   "v64f16.v64f32";
    info.matVecTy = LLVM::getVectorType(Float16Type::get(ctx), 64);
  } else {
    info.tgLoadName = "air.simdgroup_matrix_8x8_load.v64f32.p3f32";
    info.mmaName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32."
                   "v64f32.v64f32";
    info.matVecTy = LLVM::getVectorType(Float32Type::get(ctx), 64);
  }
  return info;
}

Value toMmaInputType(OpBuilder &rewriter, Location loc, Value val,
                     Type targetTy) {
  auto valTy = val.getType();
  if (valTy == targetTy)
    return val;
  if (isa<FloatType>(valTy) && isa<FloatType>(targetTy)) {
    unsigned srcBits = cast<FloatType>(valTy).getWidth();
    unsigned dstBits = cast<FloatType>(targetTy).getWidth();
    if (srcBits < dstBits)
      return arith::ExtFOp::create(rewriter, loc, targetTy, val);
    else
      return arith::TruncFOp::create(rewriter, loc, targetTy, val);
  }
  // Integer type path: convert to f32 first, then truncate if needed
  auto f32Ty = Float32Type::get(rewriter.getContext());
  Value f32Val = arith::SIToFPOp::create(rewriter, loc, f32Ty, val);
  if (targetTy == f32Ty)
    return f32Val;
  return arith::TruncFOp::create(rewriter, loc, targetTy, f32Val);
}

Value toF32(OpBuilder &rewriter, Location loc, Value val, Type f32Ty) {
  auto valTy = val.getType();
  if (valTy == f32Ty)
    return val;
  if (isa<FloatType>(valTy))
    return arith::ExtFOp::create(rewriter, loc, f32Ty, val);
  // Integer type (e.g. i8, i16, i32) -- use signed conversion
  return arith::SIToFPOp::create(rewriter, loc, f32Ty, val);
}

Value fromF32(OpBuilder &rewriter, Location loc, Value val, Type targetTy) {
  if (val.getType() == targetTy)
    return val;
  if (isa<FloatType>(targetTy))
    return arith::TruncFOp::create(rewriter, loc, targetTy, val);
  // Integer type -- use signed conversion
  return arith::FPToSIOp::create(rewriter, loc, targetTy, val);
}

unsigned &getDotCounter(MLIRContext *ctx) {
  static llvm::DenseMap<MLIRContext *, unsigned> counters;
  return counters[ctx];
}

} // namespace mlir::triton::applegpu::dotcommon
