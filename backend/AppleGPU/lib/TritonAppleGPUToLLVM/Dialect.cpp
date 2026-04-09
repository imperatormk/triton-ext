// TritonAppleGPU dialect registration.

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"

// Pull in tablegen-generated definitions
#include "Dialect/TritonAppleGPU/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/TritonAppleGPU/IR/TritonAppleGPUAttrDefs.cpp.inc"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::applegpu {

void TritonAppleGPUDialect::initialize() { registerTypes(); }

void TritonAppleGPUDialect::registerTypes() {}

} // namespace mlir::triton::applegpu
