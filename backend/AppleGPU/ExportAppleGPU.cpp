/// Export Apple GPU backend passes and dialect via the Triton plugin API.
/// Registers 5 passes + 1 dialect in a single shared library.

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "TritonAppleGPUToLLVM/Passes.h"
#include "TritonAppleGPUTransforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "triton/Tools/PluginUtils.h"

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
static void addToLLVMIR(mlir::PassManager *pm,
                        const std::vector<std::string> &) {
  pm->addPass(mlir::triton::applegpu::createConvertTritonAppleGPUToLLVMPass());
}
static void addLowerGPUToAIR(mlir::PassManager *pm,
                             const std::vector<std::string> &) {
  pm->addPass(mlir::triton::applegpu::createLowerGPUToAirPass());
}
static void addReconcileUnrealizedCasts(mlir::PassManager *pm,
                                        const std::vector<std::string> &) {
  pm->addPass(mlir::createReconcileUnrealizedCastsPass());
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
static void registerToLLVMIR() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createConvertTritonAppleGPUToLLVMPass();
  });
}
static void registerLowerGPUToAIR() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::applegpu::createLowerGPUToAirPass();
  });
}
static void registerReconcileUnrealizedCasts() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createReconcileUnrealizedCastsPass();
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
      {"add_accelerate_matmul", "0.1.0", addAccelerateMatmul,
       registerAccelerateMatmul},
      {"add_simplify_gather", "0.1.0", addSimplifyGather,
       registerSimplifyGather},
      {"add_to_llvmir", "0.1.0", addToLLVMIR, registerToLLVMIR},
      {"add_lower_gpu_to_air", "0.1.0", addLowerGPUToAIR, registerLowerGPUToAIR},
      {"add_reconcile_unrealized_casts", "0.1.0", addReconcileUnrealizedCasts,
       registerReconcileUnrealizedCasts},
  };

  static plugin::DialectInfo dialects[] = {
      {"TritonAppleGPU", "0.1.0", insertAppleGPUDialect},
  };

  static plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      "TritonAppleGPUBackend",
      "0.1.0",
      passes,
      5, // numPasses
      dialects,
      1, // numDialects
      nullptr,
      0, // numOps
      TRITON_VERSION,
  };
  return &info;
}
