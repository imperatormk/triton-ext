#ifndef TRITON_APPLE_GPU_ATTRDEFS
#define TRITON_APPLE_GPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "TritonAppleGPUDialect.td"

//===----------------------------------------------------------------------===//
// AppleMmaEncodingAttr
//===----------------------------------------------------------------------===//
//
// Encoding for tensors produced by simdgroup_multiply_accumulate on Apple GPUs.
// Apple simdgroup 8x8 MMA layout (M1, 32 threads, 2 regs each). Lane T and
// reg R scatter into row/col bits (see the bases in
// AppleMmaLayoutConversions.cpp, which must match simdgroup_matrix per-lane
// storage):
//   row = ((T>>1)&1) | ((T>>2)&1)<<1 | ((T>>4)&1)<<2
//   col = R          | ((T>>0)&1)<<1 | ((T>>3)&1)<<2

def AppleMmaEncodingAttr : AttrDef<TritonAppleGPU_Dialect, "AppleMmaEncoding",
    [DeclareAttrInterfaceMethods<DistributedEncodingTrait>,
     DeclareAttrInterfaceMethods<MmaEncodingTrait>,
     LayoutEncodingTrait]> {

  let mnemonic = "apple_mma";

  let summary = "Apple simdgroup matmul encoding";

  let description = [{
    Encoding for tensors distributed across Apple simdgroup_multiply_accumulate
    8x8 tiles. Each simdgroup (32 threads) owns one 8x8 output tile.
    warpsPerCTA controls how many simdgroups tile the full tensor.
  }];

  let parameters = (ins
    // How many simdgroups tile the [M, N] dims of the full tensor.
    // e.g. warpsPerCTA=[2,4] → 8 simdgroups cover 16x32 output.
    ArrayRefParameter<"unsigned">:$warpsPerCTA
  );

  let assemblyFormat = "`<` `{` `warpsPerCTA` `=` `[` $warpsPerCTA `]` `}` `>`";

  let extraClassDeclaration = [{
    // Tile shape for one simdgroup — fixed by hardware
    static constexpr unsigned kTileM = 8;
    static constexpr unsigned kTileN = 8;
    static constexpr unsigned kTileK = 8;
    static constexpr unsigned kLanesPerSimdgroup = 32;
    static constexpr unsigned kRegsPerLane = 2;

    SmallVector<unsigned> getInstrShape() const {
      return {kTileM, kTileN};
    }

    // Full CTA tile = instrShape × warpsPerCTA
    SmallVector<unsigned> getCtaTileShape() const {
      auto wpc = getWarpsPerCTA();
      return {kTileM * wpc[0], kTileN * wpc[1]};
    }

    unsigned getRank() const { return getWarpsPerCTA().size(); }

    // Required by LayoutEncodingTrait — Apple has no CGA (single CTA)
    mlir::triton::gpu::CGAEncodingAttr getCGALayout() const {
      return mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(getContext(), getRank());
    }

    // Required by DistributedEncodingTrait
    SmallVector<unsigned> getThreadsPerWarp() const {
      // 32 threads per simdgroup = 1 warp on Apple
      return {kLanesPerSimdgroup};
    }

    // toLinearLayout and getWarpsPerCTA declared by tablegen from interfaces/parameters
  }];

  let genVerifyDecl = 1;
}

#endif // TRITON_APPLE_GPU_ATTRDEFS
