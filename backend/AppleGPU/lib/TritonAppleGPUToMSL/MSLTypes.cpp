// MSLTypes.cpp - AST-typed forms of the MSL type helpers.
//
// Node-returning counterparts to mslScalarType/mslUnsignedType/mslStorageType
// in MSLEmitter.h: they build msl::Type nodes (f64 -> float, pointee ->
// `device T*`) for the builders to use instead of type strings.

#include "MSLEmitter.h"

namespace mlir::triton::applegpu {

msl::Type *MSLEmitter::astScalarType(Type t) {
  if (t.isF32() || t.isF64())
    return ctx.scalar(msl::Scalar::F32);
  if (t.isF16())
    return ctx.scalar(msl::Scalar::F16);
  if (t.isBF16())
    return ctx.scalar(msl::Scalar::BF16);
  if (isFp8Type(t))
    return ctx.scalar(msl::Scalar::U8);
  if (auto it = dyn_cast<IntegerType>(t)) {
    switch (it.getWidth()) {
    case 1:
      return ctx.scalar(msl::Scalar::I1);
    case 8:
      return ctx.scalar(msl::Scalar::I8);
    case 16:
      return ctx.scalar(msl::Scalar::I16);
    case 32:
      return ctx.scalar(msl::Scalar::I32);
    case 64:
      return ctx.scalar(msl::Scalar::I64);
    default:
      break;
    }
  }
  return nullptr;
}

msl::Type *MSLEmitter::astUnsignedType(Type t) {
  if (auto it = dyn_cast<IntegerType>(t)) {
    switch (it.getWidth()) {
    case 8:
      return ctx.scalar(msl::Scalar::U8);
    case 16:
      return ctx.scalar(msl::Scalar::U16);
    case 32:
      return ctx.scalar(msl::Scalar::U32);
    case 64:
      return ctx.scalar(msl::Scalar::U64);
    default:
      break;
    }
  }
  return nullptr;
}

msl::Type *MSLEmitter::astStorageType(Type t) {
  if (auto rt = dyn_cast<RankedTensorType>(t))
    t = rt.getElementType();
  if (auto pt = dyn_cast<tt::PointerType>(t))
    return ctx.ptr(astScalarType(pt.getPointeeType()), msl::AddrSpace::Device);
  return astScalarType(t);
}

bool MSLEmitter::isDatalessType(Type t) { return isa<ttg::AsyncTokenType>(t); }

int64_t MSLEmitter::bitsOf(Type t) {
  if (isa<tt::PointerType>(t))
    return 64;
  int64_t bits = t.getIntOrFloatBitWidth();
  return bits == 1 ? 8 : bits;
}

bool MSLEmitter::isDotOperandElem(Type t) {
  return t.isF32() || t.isF16() || t.isBF16();
}

std::string MSLEmitter::sgOperandScalar(Type t) {
  if (t.isF16())
    return "half";
  if (t.isBF16())
    return "bfloat";
  return "float";
}

std::string MSLEmitter::init0(const std::string &sc) {
  return sc == "float" || sc == "half" ? "0.0" : "0";
}

msl::Expr *MSLEmitter::astInPlaceBase(const InPlaceOperand &op) {
  if (!op.baseOffset)
    return ctx.var(op.buf);
  return ctx.paren(ctx.binary(msl::BinOp::Add, ctx.var(op.buf), op.baseOffset));
}

} // namespace mlir::triton::applegpu
