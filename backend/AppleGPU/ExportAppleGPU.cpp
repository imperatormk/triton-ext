/// Export Apple GPU backend passes and dialect via the Triton plugin API.
/// Registers the MSL emitter + TTGIR transforms and the dialect in a single
/// shared library.

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "TritonAppleGPUToMSL/Passes.h"
#include "TritonAppleGPUTransforms/Passes.h"
#include "triton/Tools/PluginUtils.h"

#include <iterator>

// ---------------------------------------------------------------------------
// Pass "add" callbacks — each appends its pass to the pipeline.
// ---------------------------------------------------------------------------

static void addAccelerateMatmul(mlir::PassManager *pm,
                                const std::vector<std::string> &) {
  pm->addPass(mlir::triton::applegpu::createAccelerateAppleMatmulPass());
}
static void addSimplifyGather(mlir::PassManager *pm,
                              const std::vector<std::string> &) {
  pm->addPass(mlir::triton::applegpu::createSimplifyGatherLayoutPass());
}
static void addStoreShuffleLayout(mlir::PassManager *pm,
                                  const std::vector<std::string> &) {
  pm->addPass(mlir::triton::applegpu::createStoreShuffleLayoutPass());
}
static void addPrefetchDotOperand(mlir::PassManager *pm,
                                  const std::vector<std::string> &) {
  pm->addPass(mlir::triton::applegpu::createPrefetchDotOperandPass());
}
static void addShareDotOperands(mlir::PassManager *pm,
                                const std::vector<std::string> &) {
  pm->addPass(mlir::triton::applegpu::createShareDotOperandsPass());
}
static void addEmitMSL(mlir::PassManager *pm,
                       const std::vector<std::string> &) {
  pm->addPass(mlir::triton::applegpu::createEmitMSLPass());
}

// ---------------------------------------------------------------------------
// Pass "register" callbacks — register each pass with MLIR.
// ---------------------------------------------------------------------------

static void registerAccelerateMatmul() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createAccelerateAppleMatmulPass();
  });
}
static void registerSimplifyGather() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createSimplifyGatherLayoutPass();
  });
}
static void registerStoreShuffleLayout() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createStoreShuffleLayoutPass();
  });
}
static void registerPrefetchDotOperand() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createPrefetchDotOperandPass();
  });
}
static void registerShareDotOperands() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createShareDotOperandsPass();
  });
}
static void registerEmitMSL() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createEmitMSLPass();
  });
}

// ---------------------------------------------------------------------------
// Dialect registration callback.
// ---------------------------------------------------------------------------

static void insertAppleGPUDialect(mlir::DialectRegistry *registry) {
  registry->insert<mlir::triton::applegpu::TritonAppleGPUDialect>();
}

// ---------------------------------------------------------------------------
// Plugin entry point — returns a static PluginInfo struct.
// ---------------------------------------------------------------------------

using namespace mlir::triton;

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::PassInfo passes[] = {
      {"accelerate_matmul", "0.1.0", addAccelerateMatmul,
       registerAccelerateMatmul},
      {"simplify_gather", "0.1.0", addSimplifyGather, registerSimplifyGather},
      {"store_shuffle_layout", "0.1.0", addStoreShuffleLayout,
       registerStoreShuffleLayout},
      {"prefetch_dot_operand", "0.1.0", addPrefetchDotOperand,
       registerPrefetchDotOperand},
      {"share_dot_operands", "0.1.0", addShareDotOperands,
       registerShareDotOperands},
      {"emit_msl", "0.1.0", addEmitMSL, registerEmitMSL},
  };

  static plugin::DialectInfo dialects[] = {
      {"TritonAppleGPU", "0.1.0", insertAppleGPUDialect},
  };

  static plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      "TritonAppleGPUBackend",
      "0.1.0",
      passes,
      // Derived from the arrays above: a hardcoded count silently truncates
      // the table when a pass is added (the dropped entry then resolves to a
      // null function pointer and segfaults at pipeline construction).
      std::size(passes),
      dialects,
      std::size(dialects),
      nullptr,
      0, // numOps
      TRITON_VERSION,
  };
  return &info;
}
