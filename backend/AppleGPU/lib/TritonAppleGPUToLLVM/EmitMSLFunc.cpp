// EmitMSLFunc.cpp - function / scope / control-flow AST sibling builders.
//
// Out-lined AST siblings of the scope emitters in MSLEmitter.h: emitFunc,
// emitDeviceFunc/Signature/Proto, declRetStruct/deviceRetType, emitReturn,
// emitFor/emitIf/emitWhile, emitBlockCFG/emitBranchEdge/emitTerminator,
// emitRegionBody/emitYieldAssign, and the emitOp dispatch spine (astEmitOp).
//
// Control flow is modelled with real scope nodes (KernelFn/DeviceFn/ForScope/
// TripCountForScope/IfScope/WhileScope/StateMachineScope) - never Raw blocks.
// Emission still runs on the string path this layer; these builders only need
// to compile and be structurally faithful. The flip layer (7b) walks each op's
// body into a Block and wraps it with the scope builders here, at which point
// astEmitOp becomes the sole minter of fresh() names.
//
// INVARIANTS mirrored from the string path:
//  - wide-IV i64 loops carry the induction-var decl as the FIRST body stmt
//    (never floated into the for-header) so the AGX i65 Gauss-sum fold can't
//    fire; the break is an IfScope + BreakStmt.
//  - a barrier is a BarrierStmt node (the printer's peephole collapses adjacent
//    ones); builders here never spell threadgroup_barrier text.
//  - a sibling never mutates nextId/indent - emission owns those - so builders
//    that mint fresh names run over a private LocalGen seeded from nextId.

#include "MSLConstants.h"
#include "MSLEmitter.h"

using namespace mlir;

namespace mlir::triton::applegpu {

namespace {
// Private fresh() over a copy of the emitter's id counter (mirrors fresh()); it
// carries the running id by reference so a single scope's param + prologue names
// stay sequential without touching the emitter's nextId.
struct LocalGen {
  int &id;
  std::string fresh() { return "v" + std::to_string(id++); }
};
} // namespace

using B = msl::BinOp;
using CS = msl::Cast::Style;

static msl::BinOp cmpBinOp(llvm::StringRef o) {
  if (o == "<") return B::Lt;
  if (o == "<=") return B::Le;
  if (o == ">") return B::Gt;
  if (o == ">=") return B::Ge;
  if (o == "==") return B::Eq;
  return B::Ne;
}

//===----------------------------------------------------------------------===//
// Return-struct type + device return type
//===----------------------------------------------------------------------===//

// The `struct fn_<name>_ret { sc f0; ... };` declaration. The field body has no
// dedicated node kind (a struct decl is not a Stmt in the set), so it is a
// RawStmt - the one design-sanctioned escape for a leaf with no node. Emission
// still writes it via declRetStruct; this only exists for the flip's module
// preamble assembly.
msl::Stmt *MSLEmitter::astRetStructDecl(tt::FuncOp func) {
  auto results = func.getFunctionType().getResults();
  std::string name = mslDeviceFuncName(func.getName()) + "_ret";
  std::string body = "struct " + name + " {\n";
  if (isTensorResult(results)) {
    std::string sc =
        mslScalarType(cast<RankedTensorType>(results[0]).getElementType());
    Value res = func.getBody().front().getTerminator()->getOperand(0);
    int rc = regCount(res);
    for (int i = 0; i < rc; ++i)
      body += "  " + sc + " f" + std::to_string(i) + ";\n";
  } else {
    for (auto [i, ty] : llvm::enumerate(results))
      body += "  " + mslScalarType(ty) + " f" + std::to_string(i) + ";\n";
  }
  body += "};";
  return ctx.rawStmt(body);
}

msl::NamedType *MSLEmitter::astRetStructType(tt::FuncOp func) {
  return ctx.named(mslDeviceFuncName(func.getName()) + "_ret");
}

msl::Type *MSLEmitter::astDeviceRetType(tt::FuncOp func) {
  auto results = func.getFunctionType().getResults();
  if (results.empty())
    return ctx.scalar(msl::Scalar::Void);
  if (isTensorResult(results))
    return astRetStructType(func);
  if (results.size() == 1)
    return astStorageType(results[0]);
  return astRetStructType(func);
}

//===----------------------------------------------------------------------===//
// Kernel / device-function signatures
//===----------------------------------------------------------------------===//

// kernel void <name>( device T* v [[buffer(N)]], ..., constant char* args
// [[buffer(K)]], uint3 tgpos [[..]], uint3 tid [[..]], uint3 numtg [[..]] )
// plus the scalar-arg readback DeclStmts and lane/warp DeclStmts prepended to
// `body`. Mirrors emitFunc's param + prologue string emission exactly.
msl::KernelFn *MSLEmitter::astKernelFn(tt::FuncOp func, msl::Block body) {
  int id = nextId;
  LocalGen g{id};
  auto fnTy = func.getFunctionType();

  msl::Attr *maxThreads = nullptr;
  if (auto nw = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
    maxThreads = ctx.maxThreadsAttr(nw.getInt() * 32);

  llvm::SmallVector<msl::Param, 8> params;
  llvm::SmallVector<BlockArgument> scalarArgs;
  unsigned buffer = 0;
  for (auto [i, argTy] : llvm::enumerate(fnTy.getInputs())) {
    BlockArgument arg = func.getArgument(i);
    if (auto pt = dyn_cast<tt::PointerType>(argTy)) {
      msl::Type *sc = ctx.ptr(astScalarType(pt.getPointeeType()),
                              msl::AddrSpace::Device);
      params.push_back(ctx.param(sc, g.fresh(), ctx.bufferAttr(buffer++)));
    } else {
      scalarArgs.push_back(arg);
    }
  }

  std::string argbufId;
  if (!scalarArgs.empty()) {
    argbufId = g.fresh();
    msl::Type *cc = ctx.ptr(ctx.scalar(msl::Scalar::I8), msl::AddrSpace::Constant);
    params.push_back(ctx.param(cc, argbufId, ctx.bufferAttr(buffer++)));
  }

  msl::Type *u3 = ctx.vector(msl::Scalar::U32, 3);
  std::string tgpos = g.fresh(), tid = g.fresh(), numtg = g.fresh();
  params.push_back(ctx.param(u3, tgpos, ctx.tgPosAttr()));
  params.push_back(ctx.param(u3, tid, ctx.threadPosAttr()));
  params.push_back(ctx.param(u3, numtg, ctx.tgsPerGridAttr()));

  // Prologue: scalar readback + lane/warp, prepended ahead of the walked body.
  msl::Block prologue;
  int off = 0;
  for (BlockArgument arg : scalarArgs) {
    Type ty = arg.getType();
    unsigned bits = ty.getIntOrFloatBitWidth();
    int size = bits == 1 ? 1 : (int)(bits / 8);
    off = (off + size - 1) / size * size;
    msl::Type *sc = astScalarType(ty);
    // *(constant sc*)(argbuf + off)
    msl::Expr *addr = ctx.paren(
        ctx.binary(B::Add, ctx.var(argbufId), ctx.lit(std::to_string(off))));
    msl::Expr *rd = ctx.deref(
        ctx.cast(CS::CStyle, ctx.ptr(sc, msl::AddrSpace::Constant), addr));
    prologue.push_back(ctx.declStmt(sc, g.fresh(), rd));
    off += size;
  }
  for (msl::Stmt *s : laneWarpDecls(id, tid))
    prologue.push_back(s);

  for (msl::Stmt *s : body)
    prologue.push_back(s);
  return ctx.kernelFn(maxThreads, mslKernelName(func.getName()), params,
                      std::move(prologue));
}

// `int lane = (int)(tid.x & 31u); int warp = (int)(tid.x >> 5);` - shared by
// kernel + device-fn prologues.
llvm::SmallVector<msl::Stmt *, 2> MSLEmitter::laneWarpDecls(int &id,
                                                            StringRef tid) {
  LocalGen g{id};
  msl::Type *i32 = ctx.scalar(msl::Scalar::I32);
  msl::Expr *tidx = ctx.member(ctx.var(tid), "x");
  msl::Expr *lane = ctx.cast(
      CS::CStyle, i32,
      ctx.paren(ctx.binary(B::And, tidx, ctx.lit("31u"))));
  msl::Expr *warp = ctx.cast(
      CS::CStyle, i32,
      ctx.paren(ctx.binary(B::Shr, ctx.member(ctx.var(tid), "x"), ctx.lit("5"))));
  return {ctx.declStmt(i32, g.fresh(), lane),
          ctx.declStmt(i32, g.fresh(), warp)};
}

// Build the device-fn param list. `bind` mirrors emitDeviceSignature's bindArgs:
// true mints fresh() names (the definition), false uses aN/__tgpos/... (the
// prototype). Trailing thread-context uint3s + optional threadgroup pool ptr.
llvm::SmallVector<msl::Param, 8>
MSLEmitter::deviceParams(tt::FuncOp func, int &id, bool bind) {
  LocalGen g{id};
  auto fnTy = func.getFunctionType();
  llvm::SmallVector<msl::Param, 8> params;
  for (auto [i, argTy] : llvm::enumerate(fnTy.getInputs())) {
    std::string id = bind ? g.fresh() : ("a" + std::to_string(i));
    if (auto pt = dyn_cast<tt::PointerType>(argTy))
      params.push_back(ctx.param(
          ctx.ptr(astScalarType(pt.getPointeeType()), msl::AddrSpace::Device),
          id));
    else
      params.push_back(ctx.param(astScalarType(argTy), id));
  }
  msl::Type *u3 = ctx.vector(msl::Scalar::U32, 3);
  params.push_back(ctx.param(u3, bind ? g.fresh() : "__tgpos"));
  params.push_back(ctx.param(u3, bind ? g.fresh() : "__tid"));
  params.push_back(ctx.param(u3, bind ? g.fresh() : "__numtg"));
  if (globalPoolBytes > 0)
    params.push_back(ctx.param(
        ctx.ptr(ctx.scalar(msl::Scalar::I8), msl::AddrSpace::Threadgroup),
        bind ? g.fresh() : "__poolptr"));
  return params;
}

msl::DeviceFn *MSLEmitter::astDeviceProto(tt::FuncOp func) {
  int id = nextId;
  return ctx.deviceFn(astDeviceRetType(func), mslDeviceFuncName(func.getName()),
                      deviceParams(func, id, /*bind=*/false), msl::Block{});
}

msl::DeviceFn *MSLEmitter::astDeviceFn(tt::FuncOp func, msl::Block body) {
  int id = nextId;
  auto params = deviceParams(func, id, /*bind=*/true);
  // lane/warp read the bound tid param (index = #inputs + 1).
  StringRef tid = params[func.getFunctionType().getInputs().size() + 1].name;
  msl::Block prologue;
  for (msl::Stmt *s : laneWarpDecls(id, tid))
    prologue.push_back(s);
  for (msl::Stmt *s : body)
    prologue.push_back(s);
  return ctx.deviceFn(astDeviceRetType(func), mslDeviceFuncName(func.getName()),
                      params, std::move(prologue));
}

//===----------------------------------------------------------------------===//
// Return
//===----------------------------------------------------------------------===//

msl::Stmt *MSLEmitter::astReturn(tt::ReturnOp op) {
  unsigned n = op.getNumOperands();
  if (n == 0)
    return ctx.returnStmt();
  if (n == 1 && !isa<RankedTensorType>(op.getOperand(0).getType()))
    return ctx.returnStmt(ctx.var(names(op.getOperand(0))[0]));
  llvm::SmallVector<msl::Expr *, 4> fields;
  if (n == 1) {
    for (const std::string &nm : names(op.getOperand(0)))
      fields.push_back(ctx.var(nm));
  } else {
    for (Value v : op.getOperands())
      fields.push_back(ctx.var(names(v)[0]));
  }
  return ctx.returnStmt(nullptr, fields);
}

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

msl::StateMachineScope *MSLEmitter::astBlockCFG(
    Region &region, StringRef state,
    ArrayRef<std::pair<std::string, msl::Block>> cases) {
  llvm::SmallVector<msl::StateMachineScope::Case, 4> smCases;
  for (auto &c : cases)
    smCases.push_back({ctx.save(c.first), c.second});
  return ctx.stateMachineScope(ctx.scalar(msl::Scalar::I32), state,
                               std::move(smCases));
}

// phi-assign the successor's arg vars from the branch operands, set the state
// var, and continue. Returned as a Block the caller splices into the case body.
msl::Block MSLEmitter::astBranchEdge(Block *succ, Operation::operand_range args,
                                     StringRef state) {
  msl::Block out;
  for (auto [i, operand] : llvm::enumerate(args)) {
    auto &src = names(operand);
    auto &dst = valMap[succ->getArgument(i)];
    for (size_t r = 0; r < dst.size(); ++r)
      out.push_back(ctx.assignStmt(
          ctx.var(dst[r]), ctx.var(src[src.size() == 1 ? 0 : r])));
  }
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
  for (auto [i, operand] : llvm::enumerate(term->getOperands())) {
    auto &src = names(operand);
    const SmallVector<std::string> &dst = dsts[i];
    for (size_t r = 0; r < dst.size(); ++r)
      out.push_back(ctx.assignStmt(
          ctx.var(dst[r]), ctx.var(src[src.size() == 1 ? 0 : r])));
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Op dispatch spine
//===----------------------------------------------------------------------===//

// Append the sibling nodes for a single elementwise/expr op's per-register
// DeclStmts, using the expr-sibling `mk(r)` per register. Names come from a
// LocalGen (not emission's nextId); post-flip astEmitOp mints the real names.
bool MSLEmitter::astElemwiseDecls(
    Operation *op, msl::Type *declTy, int &id, msl::Block &body,
    llvm::function_ref<msl::Expr *(int)> mk) {
  LocalGen g{id};
  int rc = regCount(op->getResult(0));
  for (int r = 0; r < rc; ++r)
    body.push_back(ctx.declStmt(declTy, g.fresh(), mk(r)));
  return true;
}

// Predeclare a value's per-register result variables (`sc id;`) with no init,
// mirroring declResultVars; returns the minted names (caller binds valMap).
SmallVector<std::string>
MSLEmitter::astDeclResultVars(Value v, msl::Block &body) {
  Type elem = v.getType();
  if (auto rt = dyn_cast<RankedTensorType>(elem))
    elem = rt.getElementType();
  msl::Type *sc = isa<tt::PointerType>(elem)
                      ? astStorageType(v.getType())
                      : astScalarType(elementScalarType(v.getType()));
  int rc = regCount(v);
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    body.push_back(ctx.declStmt(sc, id, nullptr));
    ids.push_back(id);
  }
  return ids;
}

// Flip-aware per-register decls: mints real fresh() names (advancing nextId so
// downstream captured/flipped ops stay in lockstep) and binds valMap[result].
bool MSLEmitter::astDeclBind(Operation *op, msl::Type *declTy, msl::Block &body,
                             llvm::function_ref<msl::Expr *(int)> mk) {
  int rc = regCount(op->getResult(0));
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    body.push_back(ctx.declStmt(declTy, id, mk(r)));
    ids.push_back(id);
  }
  valMap[op->getResult(0)] = ids;
  return true;
}

// `*p`, or the coherent-cast deref `*(device coherent(device) sc*)p` when a
// scalar spinlock forces a coherent access (mirrors emitLoad/emitStoreBody).
msl::Expr *MSLEmitter::astDerefPtr(Value, StringRef name, StringRef scName) {
  if (scalarSpinlock) {
    // Exact `device coherent(device) sc*` cast form used by emitLoad/StoreBody
    // (the printer's plain coherent-ptr form omits the leading `device`).
    msl::Type *cp = ctx.named("device coherent(device) " + scName.str() + "*");
    return ctx.deref(ctx.cast(CS::CStyle, cp, ctx.var(name)));
  }
  return ctx.deref(ctx.var(name));
}

// Per-register store: `[if (guard)] *p = v;` mirroring emitStoreBody's thread
// predicate + mask guard.
void MSLEmitter::astStoreBody(tt::StoreOp op, msl::Block &body) {
  auto &ptrs = names(op.getPtr());
  auto &vals = names(op.getValue());
  bool hasMask = op.getMask() != nullptr;
  SmallVector<std::string> *mask = hasMask ? &names(op.getMask()) : nullptr;
  bool uniform = !isa<RankedTensorType>(op.getPtr().getType());
  int rc = ptrs.size();

  unsigned laneFree = 0, warpFree = 0;
  if (!uniform) {
    auto ptrTy = cast<RankedTensorType>(op.getPtr().getType());
    tt::LinearLayout ll = ttg::toLinearLayout(ptrTy);
    MLIRContext *c = op.getContext();
    auto masks = ll.getFreeVariableMasks();
    laneFree = masks.lookup(StringAttr::get(c, "lane"));
    warpFree = masks.lookup(StringAttr::get(c, "warp"));
  }
  // Thread predicate as an Expr (built to print identically to emitStoreBody).
  msl::Expr *threadPred = nullptr;
  if (uniform) {
    threadPred = ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0"));
  } else {
    if (laneFree)
      threadPred = ctx.paren(ctx.binary(
          B::Eq,
          ctx.paren(ctx.binary(B::And, ctx.var(laneId),
                               ctx.lit(std::to_string(laneFree)))),
          ctx.lit("0")));
    if (warpFree) {
      msl::Expr *wp = ctx.paren(ctx.binary(
          B::Eq,
          ctx.paren(ctx.binary(B::And, ctx.var(warpId),
                               ctx.lit(std::to_string(warpFree)))),
          ctx.lit("0")));
      threadPred = threadPred ? ctx.binary(B::LAnd, threadPred, wp) : wp;
    }
  }

  std::string scName = mslScalarType(elementScalarType(op.getValue().getType()));
  for (int r = 0; r < rc; ++r) {
    msl::Expr *lhs = astDerefPtr(op.getPtr(), ptrs[r], scName);
    msl::Expr *v = ctx.var(vals[vals.size() == 1 ? 0 : r]);
    msl::Stmt *assign = ctx.assignStmt(lhs, v);
    msl::Expr *guard = threadPred;
    if (hasMask) {
      msl::Expr *m = ctx.var((*mask)[mask->size() == 1 ? 0 : r]);
      guard = guard ? ctx.binary(B::LAnd, guard, m) : m;
    }
    if (guard)
      body.push_back(ctx.compactIf(guard, assign));
    else
      body.push_back(assign);
  }
}

// Walk a single-block region's ops into a Block: each op goes through astEmitOp,
// falling back to a verbatim capture of the string emitOp for not-yet-flipped
// families. `depth` is the printer nesting the Block prints at, so a captured op
// bakes matching indentation. Terminators (yield/return/branch) are walked too;
// astEmitOp handles the dataless ones and captures the rest.
msl::Block MSLEmitter::astWalkBlock(Block &blk, unsigned depth) {
  msl::Block body;
  int savedIndent = indent;
  indent = depth;
  for (Operation &op : blk) {
    if (astEmitOp(&op, body))
      continue;
    Operation *opp = &op;
    msl::Stmt *raw = captureRaw([&] {
      if (failed(emitOp(opp)))
        emitFailed = true;
    });
    // An alias/dataless op (splat/reshape/broadcast/...) rebinds valMap but emits
    // no text; drop its empty RawStmt so no node lingers for it.
    if (!llvm::cast<msl::RawStmt>(raw)->text.empty())
      body.push_back(raw);
    // A captured op may leave a pending string-path barrier (it flushes only at
    // scope boundaries). Translate it to a BarrierStmt so the printer peephole
    // still collapses it with an adjacent barrier uniformly.
    if (barrierPending) {
      body.push_back(ctx.barrier(barrierPendingDevice));
      barrierPending = false;
      barrierPendingDevice = false;
    }
  }
  indent = savedIndent;
  return body;
}

// Route `op` to its sibling-builder(s), appending nodes to `body`. Returns true
// when handled (including alias/dataless ops that append nothing). Ops whose
// full lowering is not yet expressible from existing siblings return false -
// the flip layer (7b) wires those; see the report for the exact list.
bool MSLEmitter::astEmitOp(Operation *op, msl::Block &body) {
  auto opnd = [&](Value v, int r) -> StringRef {
    auto &nm = names(v);
    return nm[nm.size() == 1 ? 0 : r];
  };

  // Barriers: BarrierStmt nodes (printer peephole collapses adjacent ones).
  if (auto b = dyn_cast<ttg::BarrierOp>(op)) {
    uint32_t bits = static_cast<uint32_t>(b.getAddrSpace());
    bool device = bits & (static_cast<uint32_t>(ttg::AddrSpace::GlobalRead) |
                          static_cast<uint32_t>(ttg::AddrSpace::GlobalWrite));
    body.push_back(ctx.barrier(device));
    return true;
  }
  if (isa<mlir::gpu::BarrierOp>(op)) {
    body.push_back(ctx.barrier(/*device=*/true));
    return true;
  }
  if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(op)) {
    body.push_back(ctx.barrier(/*device=*/false));
    for (Value r : op->getResults())
      valMap[r] = SmallVector<std::string>{};
    return true;
  }

  // Structural no-ops that neither emit text nor bind a named value.
  if (isa<ttg::LocalDeallocOp, scf::YieldOp, scf::ConditionOp>(op))
    return true;
  if (op->getName().getStringRef() == "llvm.intr.assume")
    return true;
  if (isa<tt::AssertOp, tt::PrintOp>(op)) {
    for (Value r : op->getResults())
      valMap[r] = SmallVector<std::string>{};
    return true;
  }

  Type resElem = op->getNumResults()
                     ? elementScalarType(op->getResult(0).getType())
                     : Type();

  // Float binaries: `sc id = (a o b);`
  if (isa<arith::AddFOp, arith::MulFOp, arith::SubFOp, arith::DivFOp,
          tt::PreciseDivFOp>(op))
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astElementwiseExpr(
          isa<arith::AddFOp>(op)   ? B::Add
          : isa<arith::SubFOp>(op) ? B::Sub
          : isa<arith::MulFOp>(op) ? B::Mul
                                   : B::Div,
          nullptr, opnd(op->getOperand(0), r), opnd(op->getOperand(1), r));
    });

  // Integer add/sub/mul/div/rem (astIntBinaryExpr handles the i1 and unsigned
  // paths; the decl type must match the string path's unsigned promotion).
  if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp, arith::DivSIOp,
          arith::DivUIOp, arith::RemSIOp, arith::RemUIOp>(op)) {
    msl::Type *declTy = astScalarType(resElem);
    if (auto it = dyn_cast<IntegerType>(resElem); it && it.getWidth() == 1)
      declTy = ctx.scalar(msl::Scalar::I1);
    else if (isa<arith::DivUIOp, arith::RemUIOp>(op))
      declTy = astUnsignedType(resElem);
    return astDeclBind(op, declTy, body, [&](int r) {
      return astIntBinaryExpr(op, opnd(op->getOperand(0), r),
                              opnd(op->getOperand(1), r));
    });
  }

  // Bitwise/logical and/or/xor.
  if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
    B bo = isa<arith::AndIOp>(op) ? B::And
           : isa<arith::OrIOp>(op) ? B::Or
                                   : B::Xor;
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astElementwiseExpr(bo, nullptr, opnd(op->getOperand(0), r),
                                opnd(op->getOperand(1), r));
    });
  }

  // Shifts.
  if (isa<arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp>(op))
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astShiftExpr(op, opnd(op->getOperand(0), r),
                          opnd(op->getOperand(1), r));
    });

  // Program-id / num-programs: `int id = (int)(builtin.comp);`
  if (auto p = dyn_cast<tt::GetProgramIdOp>(op)) {
    const char *comp = p.getAxis() == tt::ProgramIDDim::X   ? "x"
                       : p.getAxis() == tt::ProgramIDDim::Y ? "y"
                                                            : "z";
    msl::Expr *e = ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::I32),
                            ctx.paren(ctx.member(ctx.var(tgposId), comp)));
    std::string id = fresh();
    body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), id, e));
    bindScalar(op->getResult(0), id);
    return true;
  }
  if (auto n = dyn_cast<tt::GetNumProgramsOp>(op)) {
    const char *comp = n.getAxis() == tt::ProgramIDDim::X   ? "x"
                       : n.getAxis() == tt::ProgramIDDim::Y ? "y"
                                                            : "z";
    msl::Expr *e = ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::I32),
                            ctx.paren(ctx.member(ctx.var(numTgId), comp)));
    std::string id = fresh();
    body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), id, e));
    bindScalar(op->getResult(0), id);
    return true;
  }

  // arith.constant: scalar / splat-tensor / dense-table.
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    Value res = cst.getResult();
    if (auto rt = dyn_cast<RankedTensorType>(res.getType())) {
      auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!dense)
        return false; // string path emits the error
      msl::Type *sc = astScalarType(rt.getElementType());
      bool isFloat = isa<FloatType>(rt.getElementType());
      std::string scStr = mslScalarType(rt.getElementType());
      if (dense.isSplat()) {
        std::string lit =
            isFloat ? floatLit(dense.getSplatValue<APFloat>(), scStr)
                    : std::to_string(dense.getSplatValue<APInt>().getSExtValue());
        return astDeclBind(op, sc, body,
                           [&](int) { return ctx.lit(lit); });
      }
      // Dense table: `sc tbl[N] = {..}; sc id = tbl[flatTileOffset];`
      SmallVector<msl::Expr *> init;
      if (isFloat)
        for (const APFloat &v : dense.getValues<APFloat>())
          init.push_back(astFloatLit(v, scStr));
      else
        for (const APInt &v : dense.getValues<APInt>())
          init.push_back(ctx.lit(std::to_string(v.getSExtValue())));
      std::string tbl = fresh();
      body.push_back(ctx.arrayDeclStmt(sc, tbl, dense.getNumElements(), init));
      return astDeclBind(op, sc, body, [&](int r) {
        return ctx.subscript(ctx.var(tbl), astFlatTileOffset(rt, r));
      });
    }
    msl::Type *sc = astScalarType(res.getType());
    std::string scStr = mslScalarType(res.getType());
    msl::Expr *lit;
    if (auto fa = dyn_cast<FloatAttr>(cst.getValue()))
      lit = astFloatLit(fa.getValue(), scStr);
    else if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
      lit = ctx.lit(std::to_string(ia.getInt()));
    else
      return false;
    std::string id = fresh();
    body.push_back(ctx.declStmt(sc, id, lit));
    bindScalar(res, id);
    return true;
  }

  // make_range: `int id = start + off;`
  if (auto mr = dyn_cast<tt::MakeRangeOp>(op)) {
    auto rt = cast<RankedTensorType>(mr.getResult().getType());
    int start = mr.getStart();
    return astDeclBind(op, ctx.scalar(msl::Scalar::I32), body, [&](int r) {
      return astMakeRangeElem(start, layoutOffsetExpr(rt, r));
    });
  }

  // Integer compare: `bool id = (casta o castb);`
  if (auto ci = dyn_cast<arith::CmpIOp>(op)) {
    const char *o;
    bool uns = false;
    switch (ci.getPredicate()) {
    case arith::CmpIPredicate::ult: uns = true; [[fallthrough]];
    case arith::CmpIPredicate::slt: o = "<"; break;
    case arith::CmpIPredicate::ule: uns = true; [[fallthrough]];
    case arith::CmpIPredicate::sle: o = "<="; break;
    case arith::CmpIPredicate::ugt: uns = true; [[fallthrough]];
    case arith::CmpIPredicate::sgt: o = ">"; break;
    case arith::CmpIPredicate::uge: uns = true; [[fallthrough]];
    case arith::CmpIPredicate::sge: o = ">="; break;
    case arith::CmpIPredicate::eq: o = "=="; break;
    case arith::CmpIPredicate::ne: o = "!="; break;
    }
    msl::BinOp bo = cmpBinOp(o);
    msl::Type *opCast =
        uns ? astUnsignedType(elementScalarType(ci.getLhs().getType())) : nullptr;
    return astDeclBind(op, ctx.scalar(msl::Scalar::I1), body, [&](int r) {
      return astElementwiseExpr(bo, opCast, opnd(op->getOperand(0), r),
                                opnd(op->getOperand(1), r));
    });
  }
  if (auto cf = dyn_cast<arith::CmpFOp>(op)) {
    const char *o;
    switch (cf.getPredicate()) {
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT: o = "<"; break;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE: o = "<="; break;
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT: o = ">"; break;
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE: o = ">="; break;
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ: o = "=="; break;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE: o = "!="; break;
    default: return false; // unsupported predicate: let string path error
    }
    msl::BinOp bo = cmpBinOp(o);
    return astDeclBind(op, ctx.scalar(msl::Scalar::I1), body, [&](int r) {
      return astElementwiseExpr(bo, nullptr, opnd(op->getOperand(0), r),
                                opnd(op->getOperand(1), r));
    });
  }

  // Select: `sc id = c ? t : f;`
  if (auto s = dyn_cast<arith::SelectOp>(op)) {
    Type re = op->getResult(0).getType();
    if (auto rt = dyn_cast<RankedTensorType>(re))
      re = rt.getElementType();
    msl::Type *declTy = isa<tt::PointerType>(re)
                            ? astStorageType(op->getResult(0).getType())
                            : astScalarType(elementScalarType(re));
    return astDeclBind(op, declTy, body, [&](int r) {
      return astSelectExpr(opnd(s.getCondition(), r), opnd(s.getTrueValue(), r),
                           opnd(s.getFalseValue(), r));
    });
  }

  // Clamp.
  if (auto c = dyn_cast<tt::ClampFOp>(op))
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astClampExpr(c, opnd(c.getX(), r), opnd(c.getMin(), r),
                          opnd(c.getMax(), r));
    });

  // Casts (non fp-narrowing) / bitcast / ptr<->int.
  if (isa<arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp,
          arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op))
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astCastExpr(op, opnd(op->getOperand(0), r));
    });
  // TruncF / FpToFp: f32->half/bfloat narrowing emits a self-contained multi-line
  // block (RTZ or RTNE); other float casts are a plain static_cast. The narrowing
  // block is captured verbatim (imperative multi-stmt, no expr sibling) but still
  // advances the real nextId + binds valMap via the string helper.
  if (isa<arith::TruncFOp, tt::FpToFpOp>(op)) {
    std::string dst = mslScalarType(elementScalarType(op->getResult(0).getType()));
    Type srcElem = elementScalarType(op->getOperand(0).getType());
    bool toHalf = dst == "half" || dst == "bfloat";
    bool rtz = false, narrow = false;
    if (auto f = dyn_cast<tt::FpToFpOp>(op)) {
      if (auto rnd = f.getRounding()) {
        narrow = srcElem.isF32() && toHalf;
        rtz = *rnd == tt::RoundingMode::RTZ;
      }
    } else if (srcElem.isF32() && toHalf) {
      narrow = true;
    }
    if (narrow) {
      auto &a = names(op->getOperand(0));
      int rc = regCount(op->getResult(0));
      SmallVector<std::string> ids;
      for (int r = 0; r < rc; ++r) {
        const std::string &v = a[a.size() == 1 ? 0 : r];
        std::string out;
        body.push_back(captureRaw([&] {
          out = rtz ? emitTruncatedFloatValue(dst, v)
                    : emitRoundedHalfValueFull(dst, v);
        }));
        ids.push_back(out);
      }
      valMap[op->getResult(0)] = ids;
      return true;
    }
    // Non-narrowing float cast: static_cast<dst>(v).
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astCastExpr(op, opnd(op->getOperand(0), r));
    });
  }
  if (isa<arith::BitcastOp, tt::BitcastOp>(op))
    return astDeclBind(op, astStorageType(op->getResult(0).getType()), body,
                       [&](int r) {
                         return astBitcastExpr(op, opnd(op->getOperand(0), r));
                       });
  if (isa<tt::IntToPtrOp, tt::PtrToIntOp>(op))
    return astDeclBind(op, astStorageType(op->getResult(0).getType()), body,
                       [&](int r) {
                         return astPtrIntCastExpr(op, opnd(op->getOperand(0), r));
                       });

  // negf: `sc id = -a;`
  if (isa<arith::NegFOp>(op))
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return ctx.unary(msl::UnOp::Neg, ctx.var(opnd(op->getOperand(0), r)));
    });

  // min/max family (decl type is always the signed scalar; opCast/propagateNan
  // per the string emitMinMax). Covers arith min/max, mulhi, remf(fmod).
  {
    StringRef fn;
    msl::Type *opCast = nullptr;
    bool propagateNan = false;
    bool isMinMax = true;
    if (isa<arith::MaximumFOp>(op)) { fn = "max"; propagateNan = true; }
    else if (isa<arith::MinimumFOp>(op)) { fn = "min"; propagateNan = true; }
    else if (isa<arith::MaxUIOp>(op)) {
      fn = "max"; opCast = astUnsignedType(resElem);
    } else if (isa<arith::MinUIOp>(op)) {
      fn = "min"; opCast = astUnsignedType(resElem);
    } else if (isa<arith::MaxNumFOp, arith::MaxSIOp>(op)) fn = "max";
    else if (isa<arith::MinNumFOp, arith::MinSIOp>(op)) fn = "min";
    else if (isa<arith::RemFOp>(op)) fn = "metal::fmod";
    else if (isa<tt::MulhiUIOp>(op)) {
      fn = "mulhi"; opCast = astUnsignedType(resElem);
    } else isMinMax = false;
    if (isMinMax)
      return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
        return astMinMaxExpr(fn, opCast, propagateNan,
                             opnd(op->getOperand(0), r),
                             opnd(op->getOperand(1), r));
      });
  }

  // precise_sqrt: `sc id = (sc)metal::precise::sqrt(a);`
  if (isa<tt::PreciseSqrtOp>(op))
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astUnaryExpr(msl::builtin::precise::Sqrt, astScalarType(resElem),
                          opnd(op->getOperand(0), r));
    });

  // math.* dialect: unary/binary transcendentals, fma, exp10.
  if (op->getDialect() ==
      op->getContext()->getLoadedDialect<math::MathDialect>()) {
    msl::Type *sc = astScalarType(resElem);
    StringRef n = op->getName().getStringRef();
    namespace bi = msl::builtin;
    static const llvm::StringMap<StringRef> unary = {
        {"math.exp", bi::precise::Exp},   {"math.exp2", bi::precise::Exp2},
        {"math.log", bi::precise::Log},   {"math.log2", bi::precise::Log2},
        {"math.log10", bi::precise::Log10}, {"math.sin", bi::precise::Sin},
        {"math.cos", bi::precise::Cos},   {"math.tan", bi::precise::Tan},
        {"math.tanh", bi::precise::Tanh}, {"math.sinh", bi::precise::Sinh},
        {"math.cosh", bi::precise::Cosh}, {"math.asin", bi::precise::Asin},
        {"math.acos", bi::precise::Acos}, {"math.atan", bi::precise::Atan},
        {"math.sqrt", bi::precise::Sqrt}, {"math.rsqrt", bi::precise::Rsqrt},
        {"math.cbrt", bi::precise::Cbrt}, {"math.floor", bi::math::Floor},
        {"math.ceil", bi::math::Ceil},    {"math.absf", bi::math::Fabs},
        {"math.absi", bi::math::Abs},     {"math.erf", "tt_erf"},
        {"math.round", bi::math::Round},  {"math.trunc", bi::math::Trunc},
        {"math.roundeven", bi::math::Rint}};
    if (auto it = unary.find(n); it != unary.end()) {
      StringRef fn = it->second;
      return astDeclBind(op, sc, body, [&](int r) {
        return astUnaryExpr(fn, sc, opnd(op->getOperand(0), r));
      });
    }
    static const llvm::StringMap<StringRef> binary = {
        {"math.atan2", bi::precise::Atan2}, {"math.powf", bi::precise::Pow},
        {"math.fpowi", bi::precise::Pow}, {"math.copysign", bi::math::Copysign}};
    if (auto it = binary.find(n); it != binary.end()) {
      StringRef fn = it->second;
      return astDeclBind(op, sc, body, [&](int r) {
        return astMinMaxExpr(fn, nullptr, false, opnd(op->getOperand(0), r),
                             opnd(op->getOperand(1), r));
      });
    }
    if (n == "math.fma")
      return astDeclBind(op, sc, body, [&](int r) {
        return astTernaryCallExpr(bi::math::Fma, opnd(op->getOperand(0), r),
                                  opnd(op->getOperand(1), r),
                                  opnd(op->getOperand(2), r));
      });
    if (n == "math.exp10")
      return astDeclBind(op, sc, body, [&](int r) {
        // pow((sc)10, a)
        msl::Expr *ten = ctx.cast(CS::CStyle, sc, ctx.lit("10"));
        return ctx.call(bi::precise::Pow, {ten, ctx.var(opnd(op->getOperand(0), r))});
      });
    return false; // unhandled math op: let string path emit the error
  }

  bool noCF = getenv("MSL_AST_NO_CF");

  // addptr: `device sc* id = b + o;`
  if (auto ap = dyn_cast<tt::AddPtrOp>(op)) {
    msl::Type *sc = ctx.ptr(astScalarType(elementScalarType(op->getResult(0).getType())),
                            msl::AddrSpace::Device);
    auto &base = names(ap.getPtr());
    auto &offs = names(ap.getOffset());
    return astDeclBind(op, sc, body, [&](int r) {
      return ctx.binary(B::Add, ctx.var(base[base.size() == 1 ? 0 : r]),
                        ctx.var(offs[offs.size() == 1 ? 0 : r]));
    });
  }

  // load: `sc id = init; [if (m)] id = *p;`
  if (auto ld = dyn_cast<tt::LoadOp>(op)) {
    Value res = ld.getResult();
    msl::Type *sc = astScalarType(elementScalarType(res.getType()));
    std::string scName = mslScalarType(elementScalarType(res.getType()));
    auto &ptrs = names(ld.getPtr());
    bool hasMask = ld.getMask() != nullptr;
    SmallVector<std::string> *mask = hasMask ? &names(ld.getMask()) : nullptr;
    SmallVector<std::string> *other = ld.getOther() ? &names(ld.getOther()) : nullptr;
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      msl::Expr *init =
          other ? static_cast<msl::Expr *>(
                      ctx.var((*other)[other->size() == 1 ? 0 : r]))
                : static_cast<msl::Expr *>(ctx.lit("0"));
      body.push_back(ctx.declStmt(sc, id, init));
      msl::Expr *deref = astDerefPtr(ld.getPtr(), ptrs[r], scName);
      msl::Stmt *assign = ctx.assignStmt(ctx.var(id), deref);
      if (hasMask)
        body.push_back(ctx.compactIf(
            ctx.var((*mask)[mask->size() == 1 ? 0 : r]), assign));
      else
        body.push_back(assign);
      ids.push_back(id);
    }
    valMap[res] = ids;
    return true;
  }

  // store: `[if (guard)] *p = v;` (optionally wrapped by a directStore guard).
  if (auto st = dyn_cast<tt::StoreOp>(op)) {
    auto handled = directStoreHandled.find(st.getOperation());
    if (handled != directStoreHandled.end()) {
      msl::Block inner;
      astStoreBody(st, inner);
      // if (!<fullTile>) { <store body> }
      body.push_back(ctx.ifScope(
          ctx.unary(msl::UnOp::LNot, ctx.var(handled->second)),
          std::move(inner)));
      return true;
    }
    astStoreBody(st, body);
    return true;
  }

  // scf.if: predeclare result vars, then IfScope with then/else sub-blocks.
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (noCF) return false;
    SmallVector<SmallVector<std::string>> results;
    for (Value res : ifOp.getResults())
      results.push_back(astDeclResultVars(res, body));

    unsigned d = (unsigned)indent + 1;
    msl::Block thenB = astWalkBlock(ifOp.getThenRegion().front(), d);
    if (!results.empty())
      for (msl::Stmt *s :
           astYieldAssign(ifOp.thenBlock()->getTerminator(), results))
        thenB.push_back(s);
    if (ifOp.getElseRegion().empty()) {
      body.push_back(ctx.ifScope(ctx.var(names(ifOp.getCondition())[0]),
                                 std::move(thenB)));
    } else {
      msl::Block elseB = astWalkBlock(ifOp.getElseRegion().front(), d);
      if (!results.empty())
        for (msl::Stmt *s :
             astYieldAssign(ifOp.elseBlock()->getTerminator(), results))
          elseB.push_back(s);
      body.push_back(ctx.ifElseScope(ctx.var(names(ifOp.getCondition())[0]),
                                     std::move(thenB), std::move(elseB)));
    }
    for (auto [i, res] : llvm::enumerate(ifOp.getResults()))
      valMap[res] = results[i];
    return true;
  }

  // scf.for (non-fused, non-wide-IV). Fused GEMM K-loops and i64-IV loops keep
  // the string path (captured) for now.
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    if (noCF)
      return false;
    if (matchGemmDotLoop(forOp))
      return false;
    Type ivType = forOp.getInductionVar().getType();
    bool wideIv = ivType.isInteger(64);

    SmallVector<SmallVector<std::string>> carried;
    for (auto [i, init, res] :
         llvm::enumerate(forOp.getInitArgs(), forOp.getResults())) {
      if (isDatalessType(res.getType())) {
        valMap[forOp.getRegionIterArg(i)] = SmallVector<std::string>{};
        valMap[res] = SmallVector<std::string>{};
        carried.push_back({});
        continue;
      }
      auto &initNames = names(init);
      SmallVector<std::string> vars = astDeclResultVars(res, body);
      for (size_t r = 0; r < vars.size(); ++r)
        body.push_back(ctx.assignStmt(
            ctx.var(vars[r]),
            ctx.var(initNames[initNames.size() == 1 ? 0 : r])));
      valMap[forOp.getRegionIterArg(i)] = vars;
      valMap[res] = vars;
      carried.push_back(vars);
    }

    // Wide-IV (i64) loops carry `tc` as the header counter and the real IV decl
    // as the first body stmt (the AGX i65 Gauss-sum dodge). The string path mints
    // iv first then tc; match that order.
    std::string iv = fresh();
    std::string tc = wideIv ? fresh() : "";
    bindScalar(forOp.getInductionVar(), iv);
    std::string ivTy = mslScalarType(ivType);
    if (ivTy.empty())
      ivTy = "int";

    unsigned d = (unsigned)indent + 1;
    msl::Block loopBody = astWalkBlock(forOp.getRegion().front(), d);
    for (msl::Stmt *s :
         astYieldAssign(forOp.getBody()->getTerminator(), carried))
      loopBody.push_back(s);
    body.push_back(astForNode(forOp, std::move(loopBody), iv, tc, ivTy, wideIv));
    return true;
  }

  // scf.while: `while (true) { <before> if (!(c)) { <fwd> break; } <after> <yield> }`
  if (auto wh = dyn_cast<scf::WhileOp>(op)) {
    if (noCF) return false;
    SmallVector<SmallVector<std::string>> carried;
    for (auto [i, init] : llvm::enumerate(wh.getInits())) {
      auto &initNames = names(init);
      SmallVector<std::string> vars = astDeclResultVars(init, body);
      for (size_t r = 0; r < vars.size(); ++r)
        body.push_back(ctx.assignStmt(
            ctx.var(vars[r]),
            ctx.var(initNames[initNames.size() == 1 ? 0 : r])));
      valMap[wh.getBeforeArguments()[i]] = vars;
      carried.push_back(vars);
    }
    SmallVector<SmallVector<std::string>> results;
    for (Value res : wh.getResults())
      results.push_back(astDeclResultVars(res, body));

    unsigned d = (unsigned)indent + 1;
    msl::Block loopBody = astWalkBlock(wh.getBefore().front(), d);
    auto cond = cast<scf::ConditionOp>(wh.getBefore().front().getTerminator());
    // if (!(c)) { <forward results> break; }
    msl::Block brk;
    for (auto [i, fwd] : llvm::enumerate(cond.getArgs())) {
      auto &src = names(fwd);
      for (size_t r = 0; r < results[i].size(); ++r)
        brk.push_back(ctx.assignStmt(ctx.var(results[i][r]),
                                     ctx.var(src[src.size() == 1 ? 0 : r])));
    }
    brk.push_back(ctx.breakStmt());
    msl::Expr *guard = ctx.unary(
        msl::UnOp::LNot, ctx.paren(ctx.var(names(cond.getCondition())[0])));
    loopBody.push_back(ctx.ifScope(guard, std::move(brk)));

    for (auto [i, fwd] : llvm::enumerate(cond.getArgs()))
      valMap[wh.getAfterArguments()[i]] = names(fwd);

    for (msl::Stmt *s : astWalkBlock(wh.getAfter().front(), d))
      loopBody.push_back(s);
    for (msl::Stmt *s :
         astYieldAssign(wh.getAfter().front().getTerminator(), carried))
      loopBody.push_back(s);

    body.push_back(ctx.whileScope(nullptr, std::move(loopBody)));
    for (auto [i, res] : llvm::enumerate(wh.getResults()))
      valMap[res] = results[i];
    return true;
  }

  // --- Flipped families slot in above this line; unflipped ops fall through to
  // the string capture in astWalkBlock. ---
  return false;
}

} // namespace mlir::triton::applegpu
