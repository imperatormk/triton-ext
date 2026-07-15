// MSLFusedDot.h - lowering-state structs for the register-resident C GEMM
// fusion and direct-store readback path. Plain data; owned by MSLEmitter.
#ifndef MSL_FUSED_DOT_H
#define MSL_FUSED_DOT_H

#include "MSLAst.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>

namespace mlir::triton::applegpu {

namespace tt = mlir::triton;

// baseOffset is a typed offset node into `buf`; nullptr means no base offset.
struct MemDescInfo {
  std::string buf;
  msl::Expr *baseOffset = nullptr;
  SmallVector<int64_t> bufStrides;
};

struct InPlaceOperand {
  std::string buf;
  msl::Expr *baseOffset = nullptr;
};

// Register-resident C GEMM fusion. When the scf.for handler recognises an
// `acc = tl.dot(a, b, acc)` K-loop it drives the enclosed tt.dot through the
// three-phase path below: PhaseDecl declares persistent simdgroup fragments
// (once, pre-loop), PhaseMMA stages A/B and accumulates into them (each
// iteration, no tgC round-trip), PhaseReadback stores the fragments and
// gathers the #mma->scalar result (once, post-loop).
enum class FusedDotPhase { None, Decl, MMA, Readback };

// When the fused GEMM accumulator flows only into a terminal row-major
// tt.store, the readback stores the accumulator fragments straight to device
// memory (skipping the threadgroup pool round-trip and the swizzled scalar
// gather). Populated when matchDirectStore succeeds; empty otherwise.
struct DirectStore {
  tt::StoreOp store;
  Value basePtr; // C matrix base pointer (scalar kernel arg)
  Value ldc;     // row stride (scalar), col stride is 1 (row-major)
  Value rowBase; // global row of the tile's top-left element (scalar)
  Value colBase; // global col of the tile's top-left element (scalar)
  Value boundM;  // store-mask row bound, or null when unmasked
  Value boundN;  // store-mask col bound, or null when unmasked
  std::string fullTileVar; // runtime "whole tile in bounds" predicate
};

struct FusedDotCtx {
  FusedDotPhase phase = FusedDotPhase::None;
  SmallVector<std::string> accNames;
  SmallVector<std::string> ids;
  SmallVector<std::string> baseNames;
  std::string tgA, tgB, tgC;
  std::optional<DirectStore> direct;
};

} // namespace mlir::triton::applegpu

#endif // MSL_FUSED_DOT_H
