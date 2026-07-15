// EmitMSLControlFlow.cpp - region / CFG-walk + scope-builder AST builders.
//
// The structured-control-flow scope builders (astForScope/astTripCountForScope/
// astForNode/astIfScope/astWhileScope), the unstructured-CFG -> state-machine
// dispatch (astEmitBlockCFG/astWalkBlock/astWalkBlock2/
// astBranchEdge/astCondBranch/astTerminatorEdge) and the loop-carried /
// if-result yield resolution (astYieldAssign). astEmitOp drives these.

#include "MSLConstants.h"
#include "MSLEmitter.h"

using namespace mlir;

namespace mlir::triton::applegpu {

using B = msl::BinOp;
using CS = msl::Cast::Style;

//===----------------------------------------------------------------------===//
// Structured control flow: for / if / while
//===----------------------------------------------------------------------===//

msl::ForScope *MSLEmitter::astForScope(scf::ForOp op, msl::Block body,
                                       StringRef iv, StringRef ivTy) {
  std::string lo = names(op.getLowerBound())[0];
  std::string hi = names(op.getUpperBound())[0];
  std::string st = names(op.getStep())[0];
  msl::Type *ty = ctx.named(ivTy);
  msl::Stmt *init = ctx.declStmt(ty, iv, ctx.var(lo));
  msl::Expr *cond = ctx.binary(B::Lt, ctx.var(iv), ctx.var(hi));
  msl::Stmt *step = ctx.addAssignStmt(ctx.var(iv), ctx.var(st));
  return ctx.forScope(init, cond, step, std::move(body));
}

// Wide-IV i64 form. The `iv = lo + tc*st` decl and `if(!(iv<hi)) break;` guard
// are the FIRST body stmts (the i65 dodge); `body` is spliced after them.
msl::TripCountForScope *MSLEmitter::astTripCountForScope(
    scf::ForOp op, msl::Block body, StringRef counter, StringRef iv,
    StringRef ivTy) {
  std::string lo = names(op.getLowerBound())[0];
  std::string hi = names(op.getUpperBound())[0];
  std::string st = names(op.getStep())[0];
  msl::Type *ty = ctx.named(ivTy);
  // iv = lo + tc * st
  msl::Expr *ivInit = ctx.binary(
      B::Add, ctx.var(lo),
      ctx.binary(B::Mul, ctx.var(counter), ctx.var(st)));
  msl::Stmt *ivDecl = ctx.declStmt(ty, iv, ivInit);
  // guard `iv < hi` (printer renders `if (!(iv < hi)) break;`).
  msl::Expr *guard = ctx.binary(B::Lt, ctx.var(iv), ctx.var(hi));
  return ctx.tripCountForScope(ty, counter, ivDecl, guard, std::move(body));
}

msl::Stmt *MSLEmitter::astForNode(scf::ForOp op, msl::Block body, StringRef iv,
                                  StringRef tc, StringRef ivTy, bool wideIv) {
  if (wideIv)
    return astTripCountForScope(op, std::move(body), tc, iv, ivTy);
  return astForScope(op, std::move(body), iv, ivTy);
}

msl::IfScope *MSLEmitter::astIfScope(scf::IfOp op, msl::Block thenB,
                                     msl::Block elseB) {
  msl::Expr *c = ctx.var(names(op.getCondition())[0]);
  if (op.getElseRegion().empty())
    return ctx.ifScope(c, std::move(thenB));
  return ctx.ifElseScope(c, std::move(thenB), std::move(elseB));
}

// while(true){ <before> if(!(c)){<fwd> break;} <after-forwarded body via body> }
// The before-region ops + condition-forward + after-region ops are all in
// `body`; the caller assembles them in order.
msl::WhileScope *MSLEmitter::astWhileScope(scf::WhileOp op, msl::Block body) {
  return ctx.whileScope(/*cond=*/nullptr, std::move(body));
}

//===----------------------------------------------------------------------===//
// Unstructured CFG -> state-machine dispatch
//===----------------------------------------------------------------------===//

// phi-assign the successor's arg vars from the branch operands, set the state
// var, and continue. Returned as a Block the caller splices into the case body.
// Per-register `dst[r] = src[bcast]` with the size==1 splat broadcast.
void MSLEmitter::astCopyRegs(msl::Block &out, ArrayRef<std::string> dst,
                             ArrayRef<std::string> src) {
  for (size_t r = 0; r < dst.size(); ++r)
    out.push_back(ctx.assignStmt(
        ctx.var(dst[r]), ctx.var(src[src.size() == 1 ? 0 : r])));
}

msl::Block MSLEmitter::astBranchEdge(Block *succ, Operation::operand_range args,
                                     StringRef state) {
  msl::Block out;
  for (auto [i, operand] : llvm::enumerate(args))
    astCopyRegs(out, valMap[succ->getArgument(i)], names(operand));
  out.push_back(ctx.assignStmt(ctx.var(state), ctx.lit(blockLabel[succ])));
  out.push_back(ctx.continueStmt());
  return out;
}

// cond_branch -> if(cond){<trueEdge>} else {<falseEdge>}; the two edge Blocks
// are built by astBranchEdge.
msl::Stmt *MSLEmitter::astCondBranch(Value cond, msl::Block thenB,
                                     msl::Block elseB) {
  return ctx.ifElseScope(ctx.var(names(cond)[0]), std::move(thenB),
                         std::move(elseB));
}

//===----------------------------------------------------------------------===//
// Loop-carried / if-result yield resolution
//===----------------------------------------------------------------------===//

msl::Block MSLEmitter::astYieldAssign(
    Operation *term, ArrayRef<SmallVector<std::string>> dsts) {
  msl::Block out;
  for (auto [i, operand] : llvm::enumerate(term->getOperands()))
    astCopyRegs(out, dsts[i], names(operand));
  return out;
}

// Multi-block region -> state-machine dispatch Block. Predeclares block-arg vars
// + cross-block hoisted vars, then a StateMachineScope with per-block cases
// (walked ops + hoist spills + terminator edge).
msl::Block MSLEmitter::astEmitBlockCFG(Region &region) {
  msl::Block out;
  blockLabel.clear();
  int idx = 0;
  for (Block &blk : region)
    blockLabel[&blk] = std::to_string(idx++);

  for (Block &blk : llvm::drop_begin(region))
    for (BlockArgument arg : blk.getArguments()) {
      if (isDatalessType(arg.getType())) {
        valMap[arg] = SmallVector<std::string>{};
        continue;
      }
      valMap[arg] = astDeclResultVars(arg, out);
    }
  llvm::DenseMap<Value, SmallVector<std::string>> hoist;
  for (Block &blk : region)
    for (Operation &op : blk)
      for (Value res : op.getResults()) {
        if (isDatalessType(res.getType()))
          continue;
        bool crosses = llvm::any_of(res.getUsers(), [&](Operation *u) {
          return u->getBlock() != &blk;
        });
        if (crosses)
          hoist[res] = astDeclResultVars(res, out);
      }

  std::string state = fresh();
  cfgState = state;
  llvm::SmallVector<std::pair<std::string, msl::Block>> cases;
  for (Block &blk : region) {
    msl::Block caseBody = astWalkBlock2(blk, hoist);
    // terminator edge
    for (msl::Stmt *s : astTerminatorEdge(blk.getTerminator(), state))
      caseBody.push_back(s);
    cases.push_back({blockLabel[&blk], std::move(caseBody)});
  }
  cfgState.clear();
  out.push_back(astMapCFGStateMachine(state, cases));
  return out;
}

// Walk a CFG block's non-terminator ops, spilling cross-block results into their
// hoisted vars (and rebinding) after each defining op.
msl::Block
MSLEmitter::astWalkBlock2(Block &blk,
                          llvm::DenseMap<Value, SmallVector<std::string>> &hoist) {
  msl::Block body;
  int savedIndent = indent;
  ++indent; // state-machine case body prints one level deeper
  ++indent;
  for (Operation &op : blk.without_terminator()) {
    if (!astEmitOp(&op, body)) {
      op.emitError("EmitMSL: unhandled CFG op '" +
                   op.getName().getStringRef() + "'");
      emitFailed = true;
    }
    for (Value res : op.getResults()) {
      auto it = hoist.find(res);
      if (it == hoist.end())
        continue;
      auto &cur = names(res);
      for (size_t r = 0; r < it->second.size(); ++r)
        body.push_back(ctx.assignStmt(
            ctx.var(it->second[r]), ctx.var(cur[cur.size() == 1 ? 0 : r])));
      valMap[res] = it->second;
    }
  }
  indent = savedIndent;
  return body;
}

// Terminator -> state transition: branch (edge), cond_branch (if/else edges), or
// a normal op (return, walked via astEmitOp).
msl::Block MSLEmitter::astTerminatorEdge(Operation *term, StringRef state) {
  msl::Block out;
  if (auto br = dyn_cast<cf::BranchOp>(term)) {
    for (msl::Stmt *s : astBranchEdge(br.getDest(), br.getDestOperands(), state))
      out.push_back(s);
    return out;
  }
  if (auto cbr = dyn_cast<cf::CondBranchOp>(term)) {
    msl::Block thenB =
        astBranchEdge(cbr.getTrueDest(), cbr.getTrueDestOperands(), state);
    msl::Block elseB =
        astBranchEdge(cbr.getFalseDest(), cbr.getFalseDestOperands(), state);
    out.push_back(astCondBranch(cbr.getCondition(), std::move(thenB),
                                std::move(elseB)));
    return out;
  }
  // return / other terminator through astEmitOp.
  if (!astEmitOp(term, out)) {
    term->emitError("EmitMSL: unhandled CFG terminator '" +
                    term->getName().getStringRef() + "'");
    emitFailed = true;
  }
  return out;
}

// Walk a single-block region's ops into a Block: each op goes through astEmitOp.
// `depth` is the printer nesting the Block prints at. Terminators
// (yield/return/branch) are walked too; astEmitOp handles the dataless ones.
msl::Block MSLEmitter::astWalkBlock(Block &blk, unsigned depth) {
  msl::Block body;
  int savedIndent = indent;
  indent = depth;
  for (Operation &op : blk) {
    if (astEmitOp(&op, body))
      continue;
    // No real-kernel op family falls through: astEmitOp is exhaustive. An
    // unhandled op is a hard error.
    op.emitError("EmitMSL: unhandled op '" + op.getName().getStringRef() + "'");
    emitFailed = true;
    continue;
  }
  indent = savedIndent;
  return body;
}

} // namespace mlir::triton::applegpu
