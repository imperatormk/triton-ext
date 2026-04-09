// Pipeline builder + TG memory analysis.

#include "metal-ir/Pipeline.h"
#include "metal-ir/KernelProfile.h"
#include "metal-ir/MetalConstraints.h"
#include "metal-ir/PassUtil.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace metalir {

// ── MMA Presence Analysis ────────────────────────────────────────────────
//
// Still needed — InferTypedPointers queries it to decide whether to widen
// device loads for MMA compatibility. For vector_add the result is always
// "no MMA".

AnalysisKey MMAPresenceAnalysis::Key;

MMAPresence MMAPresenceAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  MMAPresence result;
  for (auto &F : M) {
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_")) {
      result.hasMMA = true;
      break;
    }
  }
  return result;
}

// ── TG Memory Analysis ──────────────────────────────────────────────────

AnalysisKey TGMemoryAnalysis::Key;

void TGMemoryBudget::addGlobal(StringRef name, unsigned bytes) {
  usedBytes += bytes;
}

bool TGMemoryBudget::fits(unsigned additionalBytes) const {
  return (usedBytes + additionalBytes) <= kMaxBytes;
}

TGMemoryBudget TGMemoryAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  TGMemoryBudget budget;
  for (auto &GV : M.globals())
    if (GV.getAddressSpace() == AS::Threadgroup)
      budget.addGlobal(GV.getName(),
                       M.getDataLayout().getTypeAllocSize(GV.getValueType()));
  return budget;
}

// ── Pipeline Builder ─────────────────────────────────────────────────────

void buildMetalIRPipeline(ModulePassManager &MPM) {
  // Phase 1: Structural transforms
  MPM.addPass(InlineNonKernelFunctionsPass());
  MPM.addPass(DecomposeStructPhisPass());
  MPM.addPass(PtrPhiToI64Pass());
  MPM.addPass(PtrSelectToI64Pass());

  // Phase 3: Instruction lowering (only the bits vector_add needs)
  MPM.addPass(LLVMToAIRIntrinsicsPass());

  // Phase 5: Type system — opaque → typed pointers
  MPM.addPass(InferTypedPointersPass());

  // Phase 6: Kernel ABI
  MPM.addPass(ScalarBufferPackingPass());
  MPM.addPass(AIRSystemValuesPass());

  // Phase 7: Pre-serialization normalization (part 1)
  MPM.addPass(NormalizeI1PointersPass());

  // Phase 8: Device memory fixup
  MPM.addPass(DeviceLoadsVolatilePass());

  // Phase 10: Pre-serialization normalization (part 2)
  MPM.addPass(NormalizeAllocasPass());

  // Phase 11: Final sanitization — replace poison with undef and strip dead
  // LLVM intrinsic declarations. Must run AFTER all lowering passes.
  struct SanitizeForMetalPass : PassInfoMixin<SanitizeForMetalPass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
      bool changed = false;
      for (auto &F : M)
        for (auto &BB : F)
          for (auto &I : BB)
            for (unsigned i = 0; i < I.getNumOperands(); i++)
              if (isa<PoisonValue>(I.getOperand(i))) {
                I.setOperand(i, UndefValue::get(I.getOperand(i)->getType()));
                changed = true;
              }
      SmallVector<Function *, 4> deadDecls;
      for (auto &F : M)
        if (F.isDeclaration() && F.use_empty() && F.isIntrinsic())
          deadDecls.push_back(&F);
      for (auto *F : deadDecls) {
        F->eraseFromParent();
        changed = true;
      }
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
  MPM.addPass(SanitizeForMetalPass());
}

} // namespace metalir
