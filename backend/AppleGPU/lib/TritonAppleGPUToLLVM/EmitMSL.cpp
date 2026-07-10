// EmitMSL.cpp - direct TritonGPU IR to MSL source printer.
//
// Consumes TTGIR (tt.*, arith.*, scf.*) plus TritonGPU blocked layouts and
// emits Metal Shading Language source. Cross-lane ops, thread indices and
// address spaces come from the MLIR types and TritonGPU LinearLayout, never
// from air.* intrinsics.

#include "TritonAppleGPUToLLVM/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

namespace mlir::triton::applegpu {

namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static std::string mslScalarType(Type t) {
  if (t.isF32())
    return "float";
  if (t.isF16())
    return "half";
  if (auto it = dyn_cast<IntegerType>(t)) {
    unsigned w = it.getWidth();
    if (w == 1)
      return "bool";
    if (w == 8)
      return "char";
    if (w == 16)
      return "short";
    if (w == 32)
      return "int";
    if (w == 64)
      return "long";
  }
  return "";
}

static Type elementScalarType(Type t) {
  if (auto rt = dyn_cast<RankedTensorType>(t))
    t = rt.getElementType();
  if (auto pt = dyn_cast<tt::PointerType>(t))
    t = pt.getPointeeType();
  return t;
}

// Per-value symbol table: maps an SSA Value to its MSL identifier. For tensor
// values we store one identifier per register (per-thread element).
class MSLEmitter {
public:
  MSLEmitter(ModuleOp mod, raw_ostream &os) : mod(mod), os(os) {}

  LogicalResult emit() {
    os << "#include <metal_stdlib>\n";
    os << "using namespace metal;\n\n";
    for (auto func : mod.getOps<tt::FuncOp>()) {
      if (failed(emitFunc(func)))
        return failure();
    }
    return success();
  }

private:
  ModuleOp mod;
  raw_ostream &os;
  int nextId = 0;
  llvm::DenseMap<Value, SmallVector<std::string>> valMap;

  std::string fresh() { return "v" + std::to_string(nextId++); }

  // Number of per-thread registers (unrolled elements) for a value.
  int regCount(Value v) {
    auto rt = dyn_cast<RankedTensorType>(v.getType());
    if (!rt)
      return 1;
    tt::LinearLayout ll = ttg::toLinearLayout(rt);
    auto kReg = StringAttr::get(v.getContext(), "register");
    return ll.getInDimSize(kReg);
  }

  SmallVector<std::string> &names(Value v) { return valMap[v]; }

  void bindScalar(Value v, std::string name) {
    valMap[v] = SmallVector<std::string>{std::move(name)};
  }

  LogicalResult emitFunc(tt::FuncOp func) {
    auto fnTy = func.getFunctionType();
    os << "kernel void " << func.getName() << "(\n";
    unsigned buffer = 0;
    SmallVector<std::string> argLines;
    for (auto [i, argTy] : llvm::enumerate(fnTy.getInputs())) {
      BlockArgument arg = func.getArgument(i);
      std::string id = fresh();
      std::string line;
      if (auto pt = dyn_cast<tt::PointerType>(argTy)) {
        std::string sc = mslScalarType(pt.getPointeeType());
        line = "    device " + sc + "* " + id + " [[buffer(" +
               std::to_string(buffer++) + ")]]";
        bindScalar(arg, id);
      } else if (auto st = dyn_cast<IntegerType>(argTy)) {
        std::string sc = mslScalarType(argTy);
        line = "    constant " + sc + "& " + id + " [[buffer(" +
               std::to_string(buffer++) + ")]]";
        bindScalar(arg, id);
      } else if (isa<FloatType>(argTy)) {
        std::string sc = mslScalarType(argTy);
        line = "    constant " + sc + "& " + id + " [[buffer(" +
               std::to_string(buffer++) + ")]]";
        bindScalar(arg, id);
      } else {
        func.emitError("EmitMSL: unsupported kernel argument type");
        return failure();
      }
      argLines.push_back(line);
    }
    tgposId = fresh();
    tidId = fresh();
    argLines.push_back("    uint3 " + tgposId +
                       " [[threadgroup_position_in_grid]]");
    argLines.push_back("    uint3 " + tidId +
                       " [[thread_position_in_threadgroup]]");
    os << llvm::join(argLines, ",\n") << ") {\n";

    // lane = tid.x & (warpSize-1); warp = tid.x >> log2(warpSize)
    laneId = fresh();
    warpId = fresh();
    os << "    int " << laneId << " = (int)(" << tidId << ".x & 31u);\n";
    os << "    int " << warpId << " = (int)(" << tidId << ".x >> 5);\n";

    Block &body = func.getBody().front();
    for (Operation &op : body) {
      if (failed(emitOp(&op)))
        return failure();
    }
    os << "}\n";
    return success();
  }

  std::string tgposId, tidId, laneId, warpId;

  // Build the MSL offset expression for register `reg` of a distributed 1-D
  // tensor value, using its TritonGPU LinearLayout. block/lane/warp are runtime
  // ids; register bits are compile-time constant. Contributions XOR together
  // (general linear-layout semantics).
  std::string layoutOffsetExpr(RankedTensorType rt, int reg) {
    MLIRContext *ctx = rt.getContext();
    tt::LinearLayout ll = ttg::toLinearLayout(rt);
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    auto outDim = *ll.getOutDimNames().begin();

    SmallVector<std::string> terms;

    int32_t constPart = 0;
    for (int b = 0, n = ll.getInDimSizeLog2(kReg); b < n; ++b)
      if (reg & (1 << b))
        constPart ^= ll.getBasis(kReg, b, outDim);
    if (constPart != 0)
      terms.push_back(std::to_string(constPart));

    auto runtimeDim = [&](StringAttr in, StringRef idExpr) {
      if (!ll.hasInDim(in))
        return;
      for (int b = 0, n = ll.getInDimSizeLog2(in); b < n; ++b) {
        int32_t basis = ll.getBasis(in, b, outDim);
        if (basis == 0)
          continue;
        std::string bitExpr =
            "(((" + idExpr.str() + " >> " + std::to_string(b) + ") & 1) * " +
            std::to_string(basis) + ")";
        terms.push_back(bitExpr);
      }
    };
    runtimeDim(kLane, laneId);
    runtimeDim(kWarp, warpId);
    runtimeDim(kBlock, tgposId + ".x");

    if (terms.empty())
      return "0";
    std::string expr = terms[0];
    for (size_t i = 1; i < terms.size(); ++i)
      expr = "(" + expr + " ^ " + terms[i] + ")";
    return expr;
  }

  LogicalResult emitOp(Operation *op) {
    if (auto c = dyn_cast<arith::ConstantOp>(op))
      return emitConstant(c);
    if (auto p = dyn_cast<tt::GetProgramIdOp>(op))
      return emitProgramId(p);
    if (auto r = dyn_cast<tt::MakeRangeOp>(op))
      return emitMakeRange(r);
    if (auto s = dyn_cast<tt::SplatOp>(op))
      return emitSplat(s);
    if (auto a = dyn_cast<tt::AddPtrOp>(op))
      return emitAddPtr(a);
    if (auto l = dyn_cast<tt::LoadOp>(op))
      return emitLoad(l);
    if (auto s = dyn_cast<tt::StoreOp>(op))
      return emitStore(s);
    if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp>(op))
      return emitIntBinary(op);
    if (isa<arith::AddFOp, arith::MulFOp, arith::SubFOp, arith::DivFOp>(op))
      return emitFloatBinary(op);
    if (auto c = dyn_cast<arith::CmpIOp>(op))
      return emitCmpI(c);
    if (isa<tt::ReturnOp>(op)) {
      os << "    return;\n";
      return success();
    }
    op->emitError("EmitMSL: unhandled op '" + op->getName().getStringRef() +
                  "'");
    return failure();
  }

  LogicalResult emitConstant(arith::ConstantOp op) {
    Value res = op.getResult();
    if (auto rt = dyn_cast<RankedTensorType>(res.getType())) {
      auto dense = dyn_cast<DenseElementsAttr>(op.getValue());
      if (!dense || !dense.isSplat()) {
        op.emitError("EmitMSL: only splat tensor constants supported");
        return failure();
      }
      std::string sc = mslScalarType(rt.getElementType());
      int rc = regCount(res);
      SmallVector<std::string> ids;
      for (int r = 0; r < rc; ++r) {
        std::string id = fresh();
        std::string lit;
        if (isa<FloatType>(rt.getElementType()))
          lit = std::to_string(dense.getSplatValue<APFloat>().convertToDouble());
        else
          lit = std::to_string(
              dense.getSplatValue<APInt>().getSExtValue());
        os << "    " << sc << " " << id << " = " << lit << ";\n";
        ids.push_back(id);
      }
      valMap[res] = ids;
      return success();
    }
    std::string sc = mslScalarType(res.getType());
    std::string id = fresh();
    std::string lit;
    if (auto fa = dyn_cast<FloatAttr>(op.getValue()))
      lit = std::to_string(fa.getValueAsDouble());
    else if (auto ia = dyn_cast<IntegerAttr>(op.getValue()))
      lit = std::to_string(ia.getInt());
    else {
      op.emitError("EmitMSL: unsupported scalar constant");
      return failure();
    }
    os << "    " << sc << " " << id << " = " << lit << ";\n";
    bindScalar(res, id);
    return success();
  }

  LogicalResult emitProgramId(tt::GetProgramIdOp op) {
    std::string id = fresh();
    const char *comp = op.getAxis() == tt::ProgramIDDim::X   ? "x"
                       : op.getAxis() == tt::ProgramIDDim::Y ? "y"
                                                             : "z";
    os << "    int " << id << " = (int)(" << tgposId << "." << comp << ");\n";
    bindScalar(op.getResult(), id);
    return success();
  }

  LogicalResult emitMakeRange(tt::MakeRangeOp op) {
    auto rt = cast<RankedTensorType>(op.getResult().getType());
    int rc = regCount(op.getResult());
    int start = op.getStart();
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      std::string off = layoutOffsetExpr(rt, r);
      os << "    int " << id << " = " << start << " + " << off << ";\n";
      ids.push_back(id);
    }
    valMap[op.getResult()] = ids;
    return success();
  }

  LogicalResult emitSplat(tt::SplatOp op) {
    std::string src = names(op.getSrc())[0];
    int rc = regCount(op.getResult());
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r)
      ids.push_back(src);
    valMap[op.getResult()] = ids;
    return success();
  }

  LogicalResult emitIntBinary(Operation *op) {
    const char *o = isa<arith::AddIOp>(op)   ? "+"
                    : isa<arith::SubIOp>(op) ? "-"
                                             : "*";
    return emitElementwise(op, o, "int");
  }

  LogicalResult emitFloatBinary(Operation *op) {
    const char *o = isa<arith::AddFOp>(op)   ? "+"
                    : isa<arith::SubFOp>(op) ? "-"
                    : isa<arith::MulFOp>(op) ? "*"
                                             : "/";
    std::string sc =
        mslScalarType(elementScalarType(op->getResult(0).getType()));
    return emitElementwise(op, o, sc);
  }

  LogicalResult emitElementwise(Operation *op, StringRef binop, StringRef sc) {
    Value res = op->getResult(0);
    auto &lhs = names(op->getOperand(0));
    auto &rhs = names(op->getOperand(1));
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      const std::string &a = lhs[lhs.size() == 1 ? 0 : r];
      const std::string &b = rhs[rhs.size() == 1 ? 0 : r];
      os << "    " << sc.str() << " " << id << " = (" << a << " " << binop.str()
         << " " << b << ");\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  LogicalResult emitCmpI(arith::CmpIOp op) {
    const char *o;
    switch (op.getPredicate()) {
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      o = "<";
      break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      o = "<=";
      break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      o = ">";
      break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      o = ">=";
      break;
    case arith::CmpIPredicate::eq:
      o = "==";
      break;
    case arith::CmpIPredicate::ne:
      o = "!=";
      break;
    }
    return emitElementwise(op, o, "bool");
  }

  LogicalResult emitAddPtr(tt::AddPtrOp op) {
    Value res = op.getResult();
    auto &base = names(op.getPtr());
    auto &offs = names(op.getOffset());
    Type scalarTy = elementScalarType(res.getType());
    std::string sc = mslScalarType(scalarTy);
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      const std::string &b = base[base.size() == 1 ? 0 : r];
      const std::string &o = offs[offs.size() == 1 ? 0 : r];
      os << "    device " << sc << "* " << id << " = " << b << " + " << o
         << ";\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  LogicalResult emitLoad(tt::LoadOp op) {
    Value res = op.getResult();
    auto &ptrs = names(op.getPtr());
    Type scalarTy = elementScalarType(res.getType());
    std::string sc = mslScalarType(scalarTy);
    bool hasMask = op.getMask() != nullptr;
    SmallVector<std::string> *mask = hasMask ? &names(op.getMask()) : nullptr;
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      const std::string &p = ptrs[r];
      os << "    " << sc << " " << id << " = 0;\n";
      if (hasMask) {
        const std::string &m = (*mask)[mask->size() == 1 ? 0 : r];
        os << "    if (" << m << ") " << id << " = *" << p << ";\n";
      } else {
        os << "    " << id << " = *" << p << ";\n";
      }
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  LogicalResult emitStore(tt::StoreOp op) {
    auto &ptrs = names(op.getPtr());
    auto &vals = names(op.getValue());
    bool hasMask = op.getMask() != nullptr;
    SmallVector<std::string> *mask = hasMask ? &names(op.getMask()) : nullptr;
    int rc = ptrs.size();
    for (int r = 0; r < rc; ++r) {
      const std::string &p = ptrs[r];
      const std::string &v = vals[vals.size() == 1 ? 0 : r];
      if (hasMask) {
        const std::string &m = (*mask)[mask->size() == 1 ? 0 : r];
        os << "    if (" << m << ") *" << p << " = " << v << ";\n";
      } else {
        os << "    *" << p << " = " << v << ";\n";
      }
    }
    return success();
  }
};

class EmitMSLPass
    : public PassWrapper<EmitMSLPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitMSLPass)

  StringRef getArgument() const final { return "emit-msl"; }
  StringRef getDescription() const final {
    return "Emit MSL source from TritonGPU IR";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    std::string msl;
    llvm::raw_string_ostream ss(msl);
    MSLEmitter emitter(mod, ss);
    if (failed(emitter.emit())) {
      signalPassFailure();
      return;
    }
    ss.flush();

    if (const char *path = getenv("TRITON_MSL_OUT")) {
      std::error_code ec;
      llvm::raw_fd_ostream out(path, ec);
      if (ec) {
        mod.emitError("EmitMSL: cannot open TRITON_MSL_OUT '" +
                      std::string(path) + "': " + ec.message());
        signalPassFailure();
        return;
      }
      out << msl;
    } else {
      llvm::errs() << msl;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createEmitMSLPass() {
  return std::make_unique<EmitMSLPass>();
}

} // namespace mlir::triton::applegpu
