// EmitMSLFunc.cpp - function / scope / control-flow AST builders.
//
// AST builders for the function scopes (driven from emitFunc / emitDeviceFunc /
// emitDeviceFuncProto / declRetStruct in MSLEmitter.h) plus the per-op dispatch
// spine (astEmitOp): each op's body is walked into a Block and wrapped with a
// scope builder here, and astEmitOp is the sole minter of fresh() names.
//
// Control flow is modelled with real scope nodes (KernelFn/DeviceFn/ForScope/
// TripCountForScope/IfScope/WhileScope/StateMachineScope) - never Raw blocks.
//
// INVARIANTS:
//  - wide-IV i64 loops carry the induction-var decl as the FIRST body stmt
//    (never floated into the for-header) so the AGX i65 Gauss-sum fold can't
//    fire; the break is an IfScope + BreakStmt.
//  - a barrier is a BarrierStmt node (the printer's peephole collapses adjacent
//    ones); builders here never spell threadgroup_barrier text.

#include "MSLConstants.h"
#include "MSLEmitter.h"

using namespace mlir;

namespace mlir::triton::applegpu {

using B = msl::BinOp;
using CS = msl::Cast::Style;

//===----------------------------------------------------------------------===//
// Return-struct type + device return type
//===----------------------------------------------------------------------===//

// The `struct fn_<name>_ret { sc f0; ... };` declaration. The field body has no
// dedicated node kind (a struct decl is not a Stmt in the set), so it is a
// RawStmt - the one design-sanctioned escape for a leaf with no node.
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

// `int lane = (int)(tid.x & 31u); int warp = (int)(tid.x >> 5);` using the
// already-minted laneId/warpId/tidId members (kernel + device-fn prologues).
llvm::SmallVector<msl::Stmt *, 2> MSLEmitter::laneWarpProlog() {
  msl::Type *i32 = ctx.scalar(msl::Scalar::I32);
  msl::Expr *lane =
      ctx.cast(CS::CStyle, i32,
               ctx.paren(ctx.binary(B::And, ctx.member(ctx.var(tidId), "x"),
                                    ctx.lit("31u"))));
  msl::Expr *warp =
      ctx.cast(CS::CStyle, i32,
               ctx.paren(ctx.binary(B::Shr, ctx.member(ctx.var(tidId), "x"),
                                    ctx.lit("5"))));
  return {ctx.declStmt(i32, laneId, lane), ctx.declStmt(i32, warpId, warp)};
}

// Build the device-fn param list. `bind`:
// true mints fresh() names (the definition), false uses aN/__tgpos/... (the
// prototype). Trailing thread-context uint3s + optional threadgroup pool ptr.
llvm::SmallVector<msl::Param, 8> MSLEmitter::deviceParams(tt::FuncOp func,
                                                          int &id, bool bind) {
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
      fields.push_back(ctx.var(scalarName(v)));
  }
  return ctx.returnStmt(nullptr, fields);
}

// Mirrors the conditions under which astEmitMath / the fp_to_fp narrowing path
// / astEmitAtomic emit a call to each preamble helper. Must stay in lockstep
// with those emit sites: a missed helper is a compile error in the MSL.
msl::HelperSet MSLEmitter::scanHelpers() {
  msl::HelperSet h;
  auto elemOf = [](Type t) {
    if (auto rt = dyn_cast<RankedTensorType>(t))
      t = rt.getElementType();
    return t;
  };
  mod.walk([&](Operation *op) {
    for (Type t : op->getOperandTypes())
      h.fp8 |= isFp8Type(elemOf(t));
    for (Type t : op->getResultTypes())
      h.fp8 |= isFp8Type(elemOf(t));

    if (op->getName().getStringRef() == "math.erf")
      h.erf = true;

    if (isa<arith::TruncFOp, tt::FpToFpOp>(op)) {
      Type dstE = elementScalarType(op->getResult(0).getType());
      Type srcE = elementScalarType(op->getOperand(0).getType());
      bool toHalf = dstE.isF16() || dstE.isBF16();
      if (srcE.isF32() && toHalf && !isFp8Type(dstE) && !isFp8Type(srcE)) {
        bool rtz = false, narrow = !isa<tt::FpToFpOp>(op);
        if (auto f = dyn_cast<tt::FpToFpOp>(op))
          if (auto rnd = f.getRounding()) {
            narrow = true;
            rtz = *rnd == tt::RoundingMode::RTZ;
          }
        if (narrow) {
          if (rtz)
            (dstE.isF16() ? h.rtzHalf : h.rtzBfloat) = true;
          else
            (dstE.isF16() ? h.rtneHalf : h.rtneBfloat) = true;
        }
      }
    }

    if (auto ar = dyn_cast<tt::AtomicRMWOp>(op)) {
      Type sTy = elementScalarType(ar.getResult().getType());
      if (isa<FloatType>(sTy)) {
        unsigned bw = sTy.getIntOrFloatBitWidth();
        tt::RMWOp kind = ar.getAtomicRmwOp();
        bool native =
            bw == 32 && (kind == tt::RMWOp::ADD || kind == tt::RMWOp::FADD);
        if (!native) {
          if (bw == 32)
            h.atomicF32 = true;
          else if (bw == 16) {
            // The packed16 CAS loops narrow through tt_rtne_int_*.
            if (sTy.isF16())
              h.atomicPacked16Half = h.rtneIntHalf = true;
            else
              h.atomicPacked16Bfloat = h.rtneIntBfloat = true;
          }
        }
      }
    }
  });
  return h;
}

LogicalResult MSLEmitter::emit() {
  msl::HelperSet helpers = scanHelpers();

  msl::MSLPrinter preamblePrinter(os);
  preamblePrinter.printPreamble(helpers);

  llvm::DenseSet<StringRef> callTargets;
  mod.walk([&](tt::CallOp call) {
    callTargets.insert(call.getCalleeAttr().getValue());
  });

  SmallVector<tt::FuncOp> devFuncs, kernels;
  for (auto func : mod.getOps<tt::FuncOp>()) {
    if (callTargets.contains(func.getSymName()))
      devFuncs.push_back(func);
    else
      kernels.push_back(func);
  }

  // MSL forbids declaring threadgroup memory in a non-kernel function, so a
  // shared pool is declared once in the kernel and passed down to every
  // device function as a threadgroup pointer. Size it for the whole module.
  globalPoolBytes = 0;
  for (auto func : mod.getOps<tt::FuncOp>()) {
    poolBytes = 0;
    liveTgBytes = 0;
    func.walk([&](ttg::LocalAllocOp la) {
      auto mt = cast<ttg::MemDescType>(la.getResult().getType());
      liveTgBytes += memdescFlatSize(mt) * byteWidth(mt.getElementType());
    });
    for (Block &blk : func.getBody())
      for (Operation &op : blk)
        scanPool(&op);
    globalPoolBytes = std::max(globalPoolBytes, poolBytes);
  }
  moduleHasDevFuncs = !devFuncs.empty();

  for (auto func : devFuncs)
    if (failed(emitDeviceFuncProto(func, /*asDecl=*/true)))
      return failure();
  if (!devFuncs.empty())
    os << "\n";
  for (auto func : devFuncs)
    if (failed(emitDeviceFunc(func)))
      return failure();
  for (auto func : kernels)
    if (failed(emitFunc(func)))
      return failure();
  return success();
}

LogicalResult MSLEmitter::emitFunc(tt::FuncOp func) {
  auto fnTy = func.getFunctionType();
  // Pin the threadgroup size the runtime always dispatches (num_warps*32) so
  // the Metal compiler budgets for exactly this size instead of an
  // occupancy-driven ceiling that could reject a valid launch.
  msl::Attr *maxThreads = nullptr;
  if (auto nw = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
    maxThreads = ctx.maxThreadsAttr(nw.getInt() * 32);

  // Params + prologue mint real fresh() names IN ORDER (before the body walk)
  // so body register names stay in lockstep with the string ABI numbering.
  unsigned buffer = 0;
  llvm::SmallVector<msl::Param, 8> params;
  SmallVector<BlockArgument> scalarArgs;
  for (auto [i, argTy] : llvm::enumerate(fnTy.getInputs())) {
    BlockArgument arg = func.getArgument(i);
    if (auto pt = dyn_cast<tt::PointerType>(argTy)) {
      std::string id = fresh();
      params.push_back(ctx.param(
          ctx.ptr(astScalarType(pt.getPointeeType()), msl::AddrSpace::Device),
          id, ctx.bufferAttr(buffer++)));
      bindScalar(arg, id);
    } else if (isa<IntegerType, FloatType>(argTy)) {
      scalarArgs.push_back(arg);
    } else {
      func.emitError("EmitMSL: unsupported kernel argument type");
      return failure();
    }
  }
  std::string argbufId;
  if (!scalarArgs.empty()) {
    argbufId = fresh();
    params.push_back(ctx.param(
        ctx.ptr(ctx.scalar(msl::Scalar::I8), msl::AddrSpace::Constant),
        argbufId, ctx.bufferAttr(buffer++)));
  }
  msl::Type *u3 = ctx.vector(msl::Scalar::U32, 3);
  tgposId = fresh();
  tidId = fresh();
  numTgId = fresh();
  params.push_back(ctx.param(u3, tgposId, ctx.tgPosAttr()));
  params.push_back(ctx.param(u3, tidId, ctx.threadPosAttr()));
  params.push_back(ctx.param(u3, numTgId, ctx.tgsPerGridAttr()));

  msl::Block prologue;
  int off = 0;
  for (BlockArgument arg : scalarArgs) {
    Type ty = arg.getType();
    unsigned bits = ty.getIntOrFloatBitWidth();
    int size = bits == 1 ? 1 : (int)(bits / 8);
    off = (off + size - 1) / size * size;
    msl::Type *sc = astScalarType(ty);
    std::string id = fresh();
    // *(constant sc*)(argbuf + off)
    msl::Expr *addr = ctx.paren(ctx.binary(msl::BinOp::Add, ctx.var(argbufId),
                                           ctx.lit(std::to_string(off))));
    prologue.push_back(ctx.declStmt(
        sc, id,
        ctx.deref(ctx.cast(msl::Cast::Style::CStyle,
                           ctx.ptr(sc, msl::AddrSpace::Constant), addr))));
    bindScalar(arg, id);
    off += size;
  }
  laneId = fresh();
  warpId = fresh();
  for (msl::Stmt *s : laneWarpProlog())
    prologue.push_back(s);

  scalarSpinlock = false;
  func.walk([&](tt::AtomicCASOp cas) {
    if (isa<RankedTensorType>(cas.getPtr().getType()))
      return;
    if (tracesToKernelArg(cas.getPtr()))
      scalarSpinlock = true;
  });

  coherentScalarPtrs.clear();
  func.walk([&](LoopLikeOpInterface loop) {
    DenseSet<Value> stored;
    loop->walk([&](tt::StoreOp st) {
      if (isa<RankedTensorType>(st.getPtr().getType()))
        return;
      if (Value base = traceToKernelArg(st.getPtr()))
        stored.insert(base);
    });
    if (stored.empty())
      return;
    loop->walk([&](tt::LoadOp ld) {
      if (isa<RankedTensorType>(ld.getPtr().getType()))
        return;
      Value base = traceToKernelArg(ld.getPtr());
      if (base && stored.contains(base))
        coherentScalarPtrs.insert(base);
    });
  });

  poolBytes = 0;
  liveTgBytes = 0;
  func.walk([&](ttg::LocalAllocOp la) {
    auto mt = cast<ttg::MemDescType>(la.getResult().getType());
    liveTgBytes += memdescFlatSize(mt) * byteWidth(mt.getElementType());
  });
  for (Block &blk : func.getBody())
    for (Operation &op : blk)
      scanPool(&op);
  int64_t kernelPool = moduleHasDevFuncs ? globalPoolBytes : poolBytes;
  if (kernelPool > 0) {
    poolBuf = "__pool";
    prologue.push_back(
        ctx.arrayDeclStmt(ctx.named("threadgroup char"), poolBuf, kernelPool));
  }

  Region &region = func.getBody();
  msl::Block body = region.hasOneBlock() ? astWalkBlock(region.front(), indent)
                                         : astEmitBlockCFG(region);
  if (emitFailed)
    return failure();
  for (msl::Stmt *s : body)
    prologue.push_back(s);
  msl::KernelFn *fn = ctx.kernelFn(maxThreads, mslKernelName(func.getName()),
                                   params, std::move(prologue));
  msl::MSLPrinter printer(os);
  printer.print(fn);
  return success();
}

LogicalResult MSLEmitter::declRetStruct(tt::FuncOp func) {
  auto results = func.getFunctionType().getResults();
  if (!isTensorResult(results) && results.size() <= 1)
    return success();
  for (Type ty : results)
    if (!isTensorResult(results) && !isa<IntegerType, FloatType>(ty)) {
      func.emitError("EmitMSL: unsupported device function result type");
      return failure();
    }
  devRetStruct[func] = mslDeviceFuncName(func.getName()) + "_ret";
  msl::MSLPrinter printer(os);
  printer.print(astRetStructDecl(func));
  os << "\n";
  return success();
}

LogicalResult MSLEmitter::emitDeviceFuncProto(tt::FuncOp func, bool asDecl) {
  if (failed(declRetStruct(func)))
    return failure();
  msl::MSLPrinter printer(os);
  printer.printProto(astDeviceProto(func));
  return success();
}

LogicalResult MSLEmitter::emitDeviceFunc(tt::FuncOp func) {
  auto fnTy = func.getFunctionType();
  llvm::SmallVector<msl::Param, 8> params;
  for (auto [i, argTy] : llvm::enumerate(fnTy.getInputs())) {
    std::string id = fresh();
    if (auto pt = dyn_cast<tt::PointerType>(argTy))
      params.push_back(ctx.param(
          ctx.ptr(astScalarType(pt.getPointeeType()), msl::AddrSpace::Device),
          id));
    else if (isa<IntegerType, FloatType>(argTy))
      params.push_back(ctx.param(astScalarType(argTy), id));
    else {
      func.emitError("EmitMSL: unsupported device function argument type");
      return failure();
    }
    bindScalar(func.getArgument(i), id);
  }
  msl::Type *u3 = ctx.vector(msl::Scalar::U32, 3);
  tgposId = fresh();
  tidId = fresh();
  numTgId = fresh();
  params.push_back(ctx.param(u3, tgposId));
  params.push_back(ctx.param(u3, tidId));
  params.push_back(ctx.param(u3, numTgId));
  devPoolPtr.clear();
  if (globalPoolBytes > 0) {
    devPoolPtr = fresh();
    params.push_back(ctx.param(
        ctx.ptr(ctx.scalar(msl::Scalar::I8), msl::AddrSpace::Threadgroup),
        devPoolPtr));
  }

  laneId = fresh();
  warpId = fresh();
  msl::Block prologue = laneWarpProlog();

  poolBuf = devPoolPtr;
  curDevFunc = func;
  Region &region = func.getBody();
  msl::Block body = region.hasOneBlock() ? astWalkBlock(region.front(), indent)
                                         : astEmitBlockCFG(region);
  if (emitFailed)
    return failure();
  curDevFunc = nullptr;
  for (msl::Stmt *s : body)
    prologue.push_back(s);
  msl::DeviceFn *fn =
      ctx.deviceFn(astDeviceRetType(func), mslDeviceFuncName(func.getName()),
                   params, std::move(prologue));
  msl::MSLPrinter printer(os);
  printer.print(fn);
  return success();
}

} // namespace mlir::triton::applegpu
