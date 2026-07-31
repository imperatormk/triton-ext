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
  Value basePtr;           // C matrix base pointer (scalar kernel arg)
  Value ldc;               // row stride (scalar), col stride is 1 (row-major)
  Value rowBase;           // global row of the tile's top-left element (scalar)
  Value colBase;           // global col of the tile's top-left element (scalar)
  Value boundM;            // store-mask row bound, or null when unmasked
  Value boundN;            // store-mask col bound, or null when unmasked
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

// The lowering strategy for one tt.dot plus every derived quantity the
// emitters need. Produced by MSLEmitter::planDot as a pure function of the op
// and the emitter's budget state - no emission, no name allocation - so the
// choice is one inspectable value instead of six interacting flags.
// Row padding for the staged A (M x K) and B (K x N) simdgroup-matrix
// operands. Measured on the emitted 64x64x32 fp32 GEMM: an unpadded row
// stride makes consecutive rows collide in the threadgroup banks; +4
// elements recovers ~4%, and the optimum is flat (+2..+8 land within noise).
// Padding only pays when the row stride is bank-aligned to begin with, and it
// is dropped outright when the widened tiles would not fit the 32KB cap.
inline void dotStageRowPads(int64_t M, int64_t N, int64_t K, int64_t aEb,
                            int64_t bEb, int64_t &aPad, int64_t &bPad) {
  aPad = (K % 8 || aEb * K % 64) ? 0 : 4;
  bPad = (N % 8 || bEb * N % 64) ? 0 : 4;
  if (M * (K + aPad) * aEb + K * (N + bPad) * bEb > kTGResidentBudgetBytes)
    aPad = bPad = 0;
}

struct DotPlan {
  enum class Kind {
    Unsupported, // shape/element type outside the simdgroup-matrix path
    Scalar,      // integer operands: per-thread scalar K-loop
    Panel,       // A+B alone overflow the budget: walk (mp x np) panels
    Fused,       // register-resident C across a K-loop (3-phase)
    Direct       // single dot; C either disjoint from A/B or row-banded
  };

  Kind kind = Kind::Unsupported;

  int rank = 0;
  int64_t Bd = 1, M = 0, N = 0, K = 0;
  // Fragment counts: M/8, N/8, K/8 and the mT*nT fragment grid.
  int64_t mT = 0, nT = 0, kT = 0, nFrag = 0;
  int64_t numWarps = 1;

  // Operands already resident in a threadgroup buffer; staging is skipped and
  // the tg pointer aliases that buffer.
  std::optional<InPlaceOperand> aInPlace, bInPlace;
  // The value whose registers get staged - the operand itself, or the source
  // of a convert_layout the staging makes redundant.
  Value aStage, bStage;

  // Pool layout: A at 0, B at stagedA, C at stagedAB (disjoint) or 0 (aliased
  // over A/B, which forces the banded C round-trip).
  int64_t stagedA = 0, stagedB = 0, stagedAB = 0;
  bool disjointC = false;
  int64_t bandRows = 0;

  // Extra columns appended to each staged A/B row so consecutive rows land in
  // different threadgroup banks. Zero when staging is skipped (in-place).
  int64_t aPad = 0, bPad = 0;

  // Fused three-phase state, mirrored from FusedDotCtx::phase.
  FusedDotPhase phase = FusedDotPhase::None;
  bool needAB = true, needC = true;
};

// Everything emitDot's prologue derives once and hands down to whichever
// strategy emitter it dispatches to: the fragment types, the staged operand
// values and their register name lists, the threadgroup pool pointer names,
// the result register ids, and the C row/col layout dims. Every field is
// already-computed state - constructing one mints no names and emits nothing.
struct DotEmitCtx {
  std::string opScalar;
  msl::MatrixType *opFrag = nullptr;
  msl::MatrixType *accFragTy = nullptr;

  Value aStage, bStage;
  ArrayRef<std::string> aNames, bNames, cInit;

  std::string tgA, tgB, tgC;
  SmallVector<std::string> ids;

  StringAttr rowDim, colDim;
};

} // namespace mlir::triton::applegpu

#endif // MSL_FUSED_DOT_H
