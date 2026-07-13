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
  msl::Stmt *step = ctx.assignStmt(
      ctx.var(iv), ctx.binary(B::Add, ctx.var(iv), ctx.var(st)));
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
  msl::Block full;
  // iv = lo + tc * st
  msl::Expr *ivInit = ctx.binary(
      B::Add, ctx.var(lo),
      ctx.binary(B::Mul, ctx.var(counter), ctx.var(st)));
  full.push_back(ctx.declStmt(ty, iv, ivInit));
  // if (!(iv < hi)) break;
  msl::Expr *guard =
      ctx.unary(msl::UnOp::LNot, ctx.paren(ctx.binary(B::Lt, ctx.var(iv),
                                                      ctx.var(hi))));
  msl::Block brk;
  brk.push_back(ctx.breakStmt());
  full.push_back(ctx.ifScope(guard, std::move(brk)));
  for (msl::Stmt *s : body)
    full.push_back(s);
  return ctx.tripCountForScope(ty, counter, std::move(full));
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
    body.push_back(captureRaw([&] {
      if (failed(emitOp(opp)))
        emitFailed = true;
    }));
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
  if (isa<ttg::LocalDeallocOp, scf::YieldOp>(op))
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

  // --- Flipped families slot in above this line; unflipped ops fall through to
  // the string capture in astWalkBlock. ---
  return false;
}

} // namespace mlir::triton::applegpu
