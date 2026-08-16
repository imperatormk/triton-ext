// EmitMSLReduce.cpp - map_elementwise lowering.
//
// AST builders for the map_elementwise family, dispatched from emitOp. The
// unstructured region is modelled as a real StateMachineScope node (MSL forbids
// goto).
//
// INVARIANT: the printer inserts no grouping parens; a builder inserts an
// explicit ctx.paren(...) wherever a subexpression needs precedence grouping.

#include "MSLConstants.h"
#include "MSLEmitter.h"

using namespace mlir;

namespace mlir::triton::applegpu {

//===----------------------------------------------------------------------===//
// map* - state-machine dispatch of a CFG region
//===----------------------------------------------------------------------===//

// sc capture;  (predeclared multi-block result slot).
msl::Stmt *MSLEmitter::mapCaptureDecl(StringRef sc, StringRef name) {
  return ctx.declStmt(ctx.named(sc), name);
}

// capture = operand;  the map_elementwise.return spill into a caller slot.
msl::Stmt *MSLEmitter::mapReturnSpill(StringRef capture, StringRef operand) {
  return ctx.assignStmt(ctx.var(capture), ctx.var(operand));
}

// Assemble the full `int state = 0; while (true) { if (state==L){..} ... }`
// dispatch as one StateMachineScope. `cases` supplies each block's label and
// its already-built body block (op emission + spill/break tails live in the
// body).
msl::Stmt *MSLEmitter::mapCFGStateMachine(
    StringRef state, ArrayRef<std::pair<std::string, msl::Block>> cases) {
  llvm::SmallVector<msl::StateMachineScope::Case, 4> smCases;
  for (auto &c : cases)
    smCases.push_back({ctx.save(c.first), c.second});
  return ctx.stateMachineScope(ctx.scalar(msl::Scalar::I32), state,
                               std::move(smCases));
}

} // namespace mlir::triton::applegpu
