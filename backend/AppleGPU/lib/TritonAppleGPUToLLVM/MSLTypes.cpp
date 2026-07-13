// MSLTypes.cpp - AST-typed forms of the MSL type helpers.
//
// Node-returning counterparts to mslScalarType/mslUnsignedType/mslStorageType
// in MSLEmitter.h. Not yet wired into emission (Layer 1); they exist so later
// layers build msl::Type nodes instead of type strings. Mappings mirror the
// string helpers exactly (f64 -> float, pointee -> `device T*`).

#include "MSLEmitter.h"

namespace mlir::triton::applegpu {

msl::Type *MSLEmitter::astScalarType(Type t) {
  if (t.isF32() || t.isF64())
    return ctx.scalar(msl::Scalar::F32);
  if (t.isF16())
    return ctx.scalar(msl::Scalar::F16);
  if (t.isBF16())
    return ctx.scalar(msl::Scalar::BF16);
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

} // namespace mlir::triton::applegpu
