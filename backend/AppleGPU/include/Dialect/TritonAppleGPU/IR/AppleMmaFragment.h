#ifndef TRITON_APPLEGPU_APPLEMMAFRAGMENT_H
#define TRITON_APPLEGPU_APPLEMMAFRAGMENT_H

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::triton::applegpu {

// Per-warp ownership of an AppleMma tensor, expressed in 8x8 simdgroup tiles.
// A warp owns ownM x ownN tiles; each tile is one <64 x f32> simdgroup_matrix
// fragment register. The fragment ABI keeps these vectors intact through the
// LLVM mid-end (no SROA-to-scalar of the accumulator), preserving occupancy.
struct AppleMmaFragmentInfo {
  int64_t warpsM = 1;
  int64_t warpsN = 1;
  int64_t ownM = 1;     // owned tile rows
  int64_t ownN = 1;     // owned tile cols
  int64_t numFrags = 1; // ownM * ownN
};

inline AppleMmaFragmentInfo getAppleMmaFragmentInfo(RankedTensorType ty,
                                                    AppleMmaEncodingAttr enc) {
  AppleMmaFragmentInfo info;
  auto wpc = enc.getWarpsPerCTA();
  unsigned rank = ty.getRank();
  int64_t M = ty.getShape()[rank - 2];
  int64_t N = ty.getShape()[rank - 1];
  info.warpsM = wpc[rank - 2];
  info.warpsN = wpc[rank - 1];
  int64_t tilesM = M / 8;
  int64_t tilesN = N / 8;
  info.ownM = std::max<int64_t>(1, tilesM / info.warpsM);
  info.ownN = std::max<int64_t>(1, tilesN / info.warpsN);
  info.numFrags = info.ownM * info.ownN;
  return info;
}

// The fragment LLVM ABI type: !llvm.struct<(vector<64xELT> x numFrags)>, where
// ELT is the per-element LLVM scalar (f32/i32/i1).
inline Type getAppleMmaFragmentElemType(MLIRContext *ctx, RankedTensorType ty) {
  Type elt = ty.getElementType();
  if (elt.isF32() || elt.isInteger(32) || elt.isInteger(1))
    return elt;
  // f16/bf16 #mma accumulators ride the SAME <64 x f32> fragment as f32;
  // narrowing happens on the extracted scalar in the store convert. A vector
  // <64 x bf16>/<64 x f16> simdgroup fragment crashes the AGX PSO materializer
  // and a vector bf16 round-trip miscompiles to zero (agx-crash-trunk:
  // solve_tril_bf16_merge_pso_crash, bf16_sgmatrix_fptrunc_miscompile).
  return Float32Type::get(ctx);
}

inline Type getAppleMmaFragmentType(MLIRContext *ctx, RankedTensorType ty,
                                    AppleMmaEncodingAttr enc) {
  auto info = getAppleMmaFragmentInfo(ty, enc);
  auto vecTy = LLVM::getVectorType(getAppleMmaFragmentElemType(ctx, ty), 64);
  SmallVector<Type> body(info.numFrags, vecTy);
  return LLVM::LLVMStructType::getLiteral(ctx, body);
}

// Slot mapping for the element at (rowOff, colOff) within the warp's owned
// tiles. fragIdx selects the owned tile (row-major over ownM x ownN); vecIdx
// is the per-lane register bit (col bit 0). Verified vs laneLocalCIn
// (DotOpToLLVM.cpp) and makeBaseMma (ConvertTritonAppleGPUToLLVM.cpp).
inline void appleMmaFragmentSlot(int64_t rowOff, int64_t colOff,
                                 const AppleMmaFragmentInfo &info,
                                 int64_t &fragIdx, int64_t &vecIdx) {
  int64_t localTm = (rowOff / 8) / std::max<int64_t>(1, info.warpsM);
  int64_t localTn = (colOff / 8) / std::max<int64_t>(1, info.warpsN);
  fragIdx = localTm * info.ownN + localTn;
  vecIdx = colOff % 2;
}

} // namespace mlir::triton::applegpu

#endif // TRITON_APPLEGPU_APPLEMMAFRAGMENT_H
