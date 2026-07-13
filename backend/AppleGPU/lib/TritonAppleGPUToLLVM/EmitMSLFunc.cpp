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


// `int lane = (int)(tid.x & 31u); int warp = (int)(tid.x >> 5);` using the
// already-minted laneId/warpId/tidId members (kernel + device-fn prologues).
llvm::SmallVector<msl::Stmt *, 2> MSLEmitter::laneWarpProlog() {
  msl::Type *i32 = ctx.scalar(msl::Scalar::I32);
  msl::Expr *lane = ctx.cast(
      CS::CStyle, i32,
      ctx.paren(ctx.binary(B::And, ctx.member(ctx.var(tidId), "x"),
                           ctx.lit("31u"))));
  msl::Expr *warp = ctx.cast(
      CS::CStyle, i32,
      ctx.paren(ctx.binary(B::Shr, ctx.member(ctx.var(tidId), "x"),
                           ctx.lit("5"))));
  return {ctx.declStmt(i32, laneId, lane), ctx.declStmt(i32, warpId, warp)};
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

} // namespace mlir::triton::applegpu
