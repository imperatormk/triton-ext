// AppleMmaEncodingAttr::toLinearLayout
//
// Converts Apple simdgroup MMA encoding → LinearLayout used throughout
// the Triton compiler for layout propagation, conversion insertion,
// and shared memory access analysis.
//
// Verified layout (verify_simdgroup.metal, M1 hardware):
//   lane T, reg R → row = (T >> 3) + R*4,  col = T & 7

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

#define S(v) StringAttr::get(ctx, (v))

namespace mlir::triton::applegpu {

LinearLayout
AppleMmaEncodingAttr::toLinearLayout(llvm::ArrayRef<int64_t> shape) const {

  MLIRContext *ctx = getContext();
  int rank = shape.size();
  assert(rank == 2 && "AppleMmaEncoding only supports 2D tensors for now");

  auto dimNames = standardOutDimNames(ctx, rank);
  auto dimRow = dimNames[0]; // "dim0"
  auto dimCol = dimNames[1]; // "dim1"

  // ── Single 8×8 simdgroup tile ─────────────────────────────────────────
  // PHYSICAL layout matches the simdgroup_matrix per-lane storage so the
  // #mma<->simdgroup_matrix bridge is lane-local (extract/insert at 0,1):
  //   phys_row = L[1] | (L[2]<<1) | (L[4]<<2)
  //   phys_col = (L[0]<<1) | (L[3]<<2) | R
  // It is UNCONDITIONAL: every C-accumulator bridge site is lane-local against
  // it, so forcing the logical layout (row = (T>>3)+R*4, col = T&7) here would
  // miscompile those bridges. Logical bases kept for reference only.
  bool physLayout = true;
  std::vector<std::vector<int32_t>> registerBases =
      physLayout ? std::vector<std::vector<int32_t>>{{0, 1}}
                 : std::vector<std::vector<int32_t>>{
                       {4, 0} // reg bit 0 → row=4, col=0
                   };
  std::vector<std::vector<int32_t>> laneBases =
      physLayout ? std::vector<std::vector<int32_t>>{
                       {0, 2}, // L0 -> col bit1
                       {1, 0}, // L1 -> row bit0
                       {2, 0}, // L2 -> row bit1
                       {0, 4}, // L3 -> col bit2
                       {4, 0}, // L4 -> row bit2
                   }
                 : std::vector<std::vector<int32_t>>{
                       {0, 1}, // lane bit 0 → row=0, col=1
                       {0, 2}, // lane bit 1 → row=0, col=2
                       {0, 4}, // lane bit 2 → row=0, col=4
                       {1, 0}, // lane bit 3 → row=1, col=0
                       {2, 0}, // lane bit 4 → row=2, col=0
                   };
  LinearLayout ctaLayout(
      SmallVector<std::pair<StringAttr, std::vector<std::vector<int32_t>>>>{
          {S("register"), registerBases}, {S("lane"), laneBases}},
      SmallVector<StringAttr>{dimRow, dimCol});

  // ── Tile simdgroups across M and N ────────────────────────────────────
  auto wpc = getWarpsPerCTA();
  assert(wpc.size() == 2);

  // Column-major warp tiling: warpId % wN → col, warpId / wN → row
  SmallVector<unsigned> warpOrder{1, 0};
  ctaLayout *= identityStandardND(S("warp"), wpc, warpOrder)
                   .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  // ── Broadcast to full tensor shape ────────────────────────────────────
  // Apple has no CGA — use trivial 1-CTA layout.
  return combineCtaCgaWithShape(ctaLayout, getCGALayout(), shape);
}

} // namespace mlir::triton::applegpu
