//===- TypeIdReproPass.cpp - Minimal repro: triton op TypeID split --------===//
//
// Minimal reproduction of the op-identity wall that prevents an out-of-tree
// extension pass from operating on Triton ops under `triton-opt`.
//
// The pass walks the module, finds the `tt.func`, and:
//   1. prints the TypeID `triton-opt` registered the op under
//      (op->getName().getTypeID()), and the TypeID this plugin computes
//      (TypeID::get<triton::FuncOp>()) -- they differ;
//   2. calls isa<triton::FuncOp>(op), which returns false and (in a typical
//      build) aborts the subsequent cast<>.
//
// Root cause: `triton-opt` statically links the Triton dialect and registers
// its ops with self-owning TypeID symbols baked into the executable. This
// plugin links `libtriton.so` -- a separate copy with its own TypeID symbols.
// MLIR identity is symbol-address equality, so the two copies never match.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/PluginUtils.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct TypeIdReproPass
    : public PassWrapper<TypeIdReproPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeIdReproPass)

  StringRef getArgument() const override { return "typeid-repro"; }
  StringRef getDescription() const override {
    return "Repro: triton::FuncOp TypeID mismatch between triton-opt and a "
           "libtriton-linked plugin";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // The TypeID this plugin's TU computes for triton::FuncOp (resolves to
    // libtriton.so's copy of the self-owning id symbol).
    const void *pluginTID =
        TypeID::get<triton::FuncOp>().getAsOpaquePointer();

    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "tt.func")
        return;

      // The TypeID triton-opt registered tt.func under (its own static copy).
      const void *opTID = op->getName().getTypeID().getAsOpaquePointer();

      llvm::errs() << "tt.func registered TypeID (triton-opt's copy): " << opTID
                   << "\n";
      llvm::errs() << "TypeID::get<triton::FuncOp>() (this plugin's copy): "
                   << pluginTID << "\n";
      llvm::errs() << "isa<triton::FuncOp>(op): "
                   << isa<triton::FuncOp>(op) << "\n";

      if (opTID != pluginTID)
        llvm::errs() << "MISMATCH: the two copies are different symbols; any "
                        "pattern doing cast<triton::FuncOp> will abort.\n";

      // This is what any real conversion pattern does, and where it aborts:
      //   auto f = cast<triton::FuncOp>(op);
      // Left commented so the repro prints cleanly; uncomment to see the abort.
    });
  }
};

} // namespace

static void addTypeIdReproPass(PassManager *pm,
                               const std::vector<std::string> &) {
  pm->addPass(std::make_unique<TypeIdReproPass>());
}

static void registerTypeIdReproPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> {
        return std::make_unique<TypeIdReproPass>();
      });
}

using namespace mlir::triton;

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::PassInfo pass = {"typeid-repro", "0.1.0", addTypeIdReproPass,
                                  registerTypeIdReproPass};
  static plugin::PassInfo passes[] = {pass};
  static plugin::PluginInfo info = {TRITON_PLUGIN_API_VERSION,
                                    "TypeIdReproPlugin",
                                    "0.1.0",
                                    passes,
                                    1,
                                    nullptr,
                                    0,
                                    nullptr,
                                    0,
                                    TRITON_VERSION};
  return &info;
}
