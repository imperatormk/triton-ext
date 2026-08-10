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

// A tile-uniform integer operand: either an SSA scalar that was splatted over
// the tile, or a splat constant (which carries no Value). Exactly one of the
// two is set when the value is present.
struct UniformInt {
  Value val;
  std::optional<int64_t> lit;
  explicit operator bool() const { return val || lit.has_value(); }
};

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
  Value basePtr;     // C matrix base pointer (scalar kernel arg)
  UniformInt baseOff; // tile-uniform element offset (bmm's batch term)
  Value tileGuard;    // tile-uniform store predicate (bmm's idx_q < BATCH)
  UniformInt ldc;    // row stride, col stride is 1 (row-major)
  Value rowBase;     // global row of the tile's top-left element (scalar)
  Value colBase;     // global col of the tile's top-left element (scalar)
  UniformInt boundM; // store-mask row bound, unset when unmasked
  UniformInt boundN; // store-mask col bound, unset when unmasked
  Type narrowTo;     // f16/bf16 output element type, null when f32
  Operation *narrowOp = nullptr; // the elided truncf, when narrowing
  Operation *cvt = nullptr;      // the layout convert feeding the store
  // An `acc + bias` epilogue whose bias is one value per column, broadcast down
  // the rows (the addmm bias). Folded into the fragments before the store, so
  // it costs 8 device loads per fragment column instead of a whole tile round
  // trip. Null when the accumulator reaches the store unmodified.
  Value biasPtr;   // bias base pointer (scalar kernel arg)
  Value biasCol;   // global col of the tile's first bias element (scalar)
  Operation *biasAdd = nullptr; // the elided arith.addf
  Operation *biasCvt = nullptr; // the elided layout convert, when present
  // A pure elementwise epilogue between the accumulator and the store. Each op
  // maps one element to one element, so it applies to the fragment's
  // thread_elements() in place and the tile never leaves register layout.
  SmallVector<Operation *> elementwise;
  // The accumulator the region reads. It may be read more than once, so
  // emission resolves it by value rather than by position.
  Value elementwiseAcc;
  std::string fullTileVar;       // runtime "whole tile in bounds" predicate
  // True when fullTileVar is the constant `true` (an unmasked store): the
  // threadgroup fallback arm is then unreachable and must not be emitted, or
  // it forces a C pool reservation that is never indexed.
  bool alwaysFullTile = false;
};

struct FusedDotCtx {
  FusedDotPhase phase = FusedDotPhase::None;
  SmallVector<std::string> accNames;
  // Every dot in a chain shares accNames, so they must agree on the warp
  // tiling; the Decl phase pins the input that selects it.
  std::optional<bool> warpTilingMSplit;
  SmallVector<std::string> ids;
  SmallVector<std::string> baseNames;
  std::string tgA, tgB, tgC;
  std::optional<DirectStore> direct;
  // Device-direct B staging: the in-flight copy's event token and the flag
  // selecting which of the two B tiles it targets. Both are loop-carried.
  std::string dmaHandle, dmaParity;
};

// The lowering strategy for one tt.dot plus every derived quantity the
// emitters need. Produced by MSLEmitter::planDot as a pure function of the op
// and the emitter's budget state - no emission, no name allocation - so the
// choice is one inspectable value instead of six interacting flags.
// Row padding for the staged A (M x K) and B (K x N) simdgroup-matrix
// operands. An unpadded row stride makes consecutive rows collide in the
// threadgroup banks; +4 elements clears it, and the optimum is flat. Padding
// only pays when the row stride is bank-aligned to begin with, and it is
// dropped outright when the widened tiles would not fit the 32KB cap.
//
// It is also dropped when it would cost a threadgroup-residency step, but only
// while residency is still the binding limit. Past kTGResidencyFloor the pool
// no longer decides occupancy - registers and warp slots do - so a nominal step
// costs nothing while the bank conflicts remain real.
// An operand that never reaches the pool - device-direct, or read in place out
// of a caller's buffer - contributes no footprint, so charging its tile against
// the budget makes the widened tiles look unaffordable and silently drops the
// other operand's padding too.
inline void dotStageRowPads(int64_t M, int64_t N, int64_t K, int64_t aEb,
                            int64_t bEb, int64_t &aPad, int64_t &bPad,
                            bool aUnstaged = false, bool bUnstaged = false) {
  aPad = (aUnstaged || K % 8 || aEb * K % 64) ? 0 : 4;
  bPad = (bUnstaged || N % 8 || bEb * N % 64) ? 0 : 4;
  int64_t aBare = aUnstaged ? 0 : M * K * aEb;
  int64_t bBare = bUnstaged ? 0 : K * N * bEb;
  int64_t aPadded = aUnstaged ? 0 : M * (K + aPad) * aEb;
  int64_t bPadded = bUnstaged ? 0 : K * (N + bPad) * bEb;
  int64_t bare = aBare + bBare;
  int64_t padded = aPadded + bPadded;
  if (padded > kTGResidentBudgetBytes ||
      (tgResidency(padded) < tgResidency(bare) &&
       tgResidency(padded) < kTGResidencyFloor))
    aPad = bPad = 0;
}

// A dot operand whose load denotes a contiguous 2D device tile, so it can be
// staged by threadgroup DMA instead of through registers. The tile origin is
// `basePtr + rowShift*rowStride + colShift`, each shift being a uniform scalar
// (null when absent); `ptrDelta` is added once per K-loop trip.
struct DirectStage {
  Value basePtr;
  // The row pitch, either as a bound SSA scalar (a kernel stride argument) or,
  // when the template folded it, as a literal. Exactly one is set.
  Value rowStride;
  std::optional<int64_t> rowStrideLit;
  Value rowShift;
  Value colShift;
  std::optional<int64_t> rowShiftLit;
  std::optional<int64_t> colShiftLit;
  Value ptrDelta;
  // Elements the tile origin advances per K-trip. Either a scalar SSA value
  // (ptrDelta) or, when the recurrence steps by a splat constant, a literal.
  std::optional<int64_t> ptrDeltaLit;
  // K-blocks this copy runs ahead of the loop induction variable.
  int aheadSteps = 0;
  int64_t rows = 0;
  int64_t cols = 0;
  // The device tile is contiguous down its columns rather than across its rows
  // (a B operand handed in with strides [1, N]). The `_tr` shim entry points
  // swap the source strides so such a tile still lands row-major in threadgroup
  // memory, which keeps the transpose out of the MMA entirely.
  bool srcTransposed = false;
  // The source load's mask, when it had one. A direct read ignores per-element
  // predicates, so this is only sound once the caller proves the tile is never
  // ragged along the masked axis.
  Value rowMask;
  // A tt.trans was peeled off the operand: rows and cols below describe the
  // device tile, and the fragment load transposes it.
  bool fragTransposed = false;
};

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
  // B is staged in the pre-transpose order and read back through
  // simdgroup_load's transpose flag, which costs nothing, rather than being
  // permuted element by element on the way into threadgroup memory.
  bool bStageTransposed = false;
  // Read out of its own allocation, so it claims no pool bytes.
  bool aNoStage = false;
  bool bNoStage = false;

  // Pool layout: A at 0, B at stagedA, C at stagedAB (disjoint) or 0 (aliased
  // over A/B, which forces the banded C round-trip).
  int64_t stagedA = 0, stagedB = 0, stagedAB = 0;
  // Device-direct B staging keeps two B tiles, at stagedA and stagedA+stagedB,
  // so the copy feeding the next K-trip is in flight while this trip's MMAs
  // read the other. Set only on the fused path.
  bool dmaB = false;
  // A's fragments are simdgroup_load-ed straight off the device tile instead of
  // being staged: the warp partition splits along M, so each warp reads only
  // the rows it owns and A is never shared. Requires an unmasked row-major
  // operand; `aDirect` carries the tile origin the loads address off.
  std::optional<DirectStage> aDirect;
  // A arrives in #mma registers (a preceding dot's result), so its fragments
  // are assembled from those registers instead of staged and reloaded.
  bool aFrag = false;
  // aFrag AND no other reader of A, so the staging can be dropped outright.
  bool aFragOnly = false;
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
  // With double-buffered DMA staging tgB is the base of the tile pair and
  // tgBCur is the one this trip reads; otherwise they are the same.
  std::string tgBCur;
  // B sits in threadgroup memory in its pre-transpose order, so its fragments
  // are read through simdgroup_load's transpose flag at pitch bStageLd.
  bool bStageTransposed = false;
  int64_t bStageLd = 0;
  // Device pointer to this warp's A rows, when A bypasses threadgroup memory.
  // Empty on the staged path, where the A fragments read tgA instead.
  std::string devA;
  // Element stride between consecutive A rows in device memory.
  msl::Expr *devALda = nullptr;
  SmallVector<std::string> ids;

  StringAttr rowDim, colDim;
};

} // namespace mlir::triton::applegpu

#endif // MSL_FUSED_DOT_H
