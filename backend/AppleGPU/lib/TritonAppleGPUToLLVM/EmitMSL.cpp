// EmitMSL.cpp - direct TritonGPU IR to MSL source printer.
//
// Consumes TTGIR (tt.*, arith.*, scf.*) plus TritonGPU blocked layouts and
// emits Metal Shading Language source. Cross-lane ops, thread indices and
// address spaces come from the MLIR types and TritonGPU LinearLayout, never
// from air.* intrinsics.

#include "TritonAppleGPUToLLVM/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  if (t.isBF16())
    return "bfloat";
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

static std::string mslStorageType(Type t) {
  if (auto rt = dyn_cast<RankedTensorType>(t))
    t = rt.getElementType();
  if (auto pt = dyn_cast<tt::PointerType>(t))
    return "device " + mslScalarType(pt.getPointeeType()) + "*";
  return mslScalarType(t);
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
    os << "#include <metal_simdgroup_matrix>\n";
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
  int indent = 1;
  llvm::DenseMap<Value, SmallVector<std::string>> valMap;

  std::string fresh() { return "v" + std::to_string(nextId++); }

  std::string ind() const { return std::string(indent * 4, ' '); }

  // Number of per-thread registers (unrolled elements) for a value.
  int regCount(Value v) {
    auto rt = dyn_cast<RankedTensorType>(v.getType());
    if (!rt)
      return 1;
    tt::LinearLayout ll = ttg::toLinearLayout(rt);
    auto kReg = StringAttr::get(v.getContext(), "register");
    return ll.getInDimSize(kReg);
  }

  // Per-register out-dim coordinate of a distributed tensor value, evaluated
  // at lane=warp=block=0 (the compile-time register component). Returns the
  // coordinate along each out-dim in tensor-dim order (dim0, dim1, ...).
  SmallVector<int32_t> registerCoords(RankedTensorType rt, int reg) {
    MLIRContext *ctx = rt.getContext();
    tt::LinearLayout ll = ttg::toLinearLayout(rt);
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    SmallVector<std::pair<StringAttr, int32_t>> ins;
    ins.push_back({kReg, reg});
    if (ll.hasInDim(kLane))
      ins.push_back({kLane, 0});
    if (ll.hasInDim(kWarp))
      ins.push_back({kWarp, 0});
    if (ll.hasInDim(kBlock))
      ins.push_back({kBlock, 0});
    auto outs = ll.apply(ins);
    SmallVector<int32_t> coords;
    for (auto &p : outs)
      coords.push_back(p.second);
    return coords;
  }

  SmallVector<std::string> &names(Value v) { return valMap[v]; }

  void bindScalar(Value v, std::string name) {
    valMap[v] = SmallVector<std::string>{std::move(name)};
  }

  LogicalResult emitFunc(tt::FuncOp func) {
    auto fnTy = func.getFunctionType();
    os << "kernel void " << func.getName() << "(\n";

    // Match the runtime ABI (driver.py): pointer args first, each in its own
    // buffer; then all scalar args packed with natural alignment into a single
    // trailing buffer. Scalars are read back by reinterpreting a byte offset.
    unsigned buffer = 0;
    SmallVector<std::string> argLines;
    SmallVector<BlockArgument> scalarArgs;
    for (auto [i, argTy] : llvm::enumerate(fnTy.getInputs())) {
      BlockArgument arg = func.getArgument(i);
      if (auto pt = dyn_cast<tt::PointerType>(argTy)) {
        std::string id = fresh();
        std::string sc = mslScalarType(pt.getPointeeType());
        argLines.push_back("    device " + sc + "* " + id + " [[buffer(" +
                           std::to_string(buffer++) + ")]]");
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
      argLines.push_back("    constant char* " + argbufId + " [[buffer(" +
                         std::to_string(buffer++) + ")]]");
    }

    tgposId = fresh();
    tidId = fresh();
    argLines.push_back("    uint3 " + tgposId +
                       " [[threadgroup_position_in_grid]]");
    argLines.push_back("    uint3 " + tidId +
                       " [[thread_position_in_threadgroup]]");
    os << llvm::join(argLines, ",\n") << ") {\n";

    int off = 0;
    for (BlockArgument arg : scalarArgs) {
      Type ty = arg.getType();
      int size = ty.getIntOrFloatBitWidth() / 8;
      off = (off + size - 1) / size * size;
      std::string sc = mslScalarType(ty);
      std::string id = fresh();
      os << ind() << sc << " " << id << " = *(constant " << sc << "*)("
         << argbufId << " + " << off << ");\n";
      bindScalar(arg, id);
      off += size;
    }

    // lane = tid.x & (warpSize-1); warp = tid.x >> log2(warpSize)
    laneId = fresh();
    warpId = fresh();
    os << ind() << "int " << laneId << " = (int)(" << tidId << ".x & 31u);\n";
    os << ind() << "int " << warpId << " = (int)(" << tidId << ".x >> 5);\n";

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
    tt::LinearLayout ll = ttg::toLinearLayout(rt);
    auto outDim = *ll.getOutDimNames().begin();
    return layoutCoordExpr(rt, reg, outDim);
  }

  // MSL expression for the coordinate of register `reg` along a single tensor
  // out-dim (dim0, dim1, ...). block/lane/warp are runtime ids; register bits
  // are compile-time constant. Contributions XOR together.
  std::string layoutCoordExpr(RankedTensorType rt, int reg, StringAttr outDim) {
    MLIRContext *ctx = rt.getContext();
    tt::LinearLayout ll = ttg::toLinearLayout(rt);
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");

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
    if (auto e = dyn_cast<tt::ExpandDimsOp>(op))
      return emitReshapeLike(e.getResult(), e.getSrc(), e.getAxis(), true);
    if (auto b = dyn_cast<tt::BroadcastOp>(op))
      return emitReshapeLike(b.getResult(), b.getSrc(), -1, false);
    if (auto a = dyn_cast<tt::AddPtrOp>(op))
      return emitAddPtr(a);
    if (auto l = dyn_cast<tt::LoadOp>(op))
      return emitLoad(l);
    if (auto s = dyn_cast<tt::StoreOp>(op))
      return emitStore(s);
    if (auto a = dyn_cast<tt::AtomicRMWOp>(op))
      return emitAtomicRMW(a);
    if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp>(op))
      return emitIntBinary(op);
    if (isa<arith::AddFOp, arith::MulFOp, arith::SubFOp, arith::DivFOp>(op))
      return emitFloatBinary(op);
    if (auto c = dyn_cast<arith::CmpIOp>(op))
      return emitCmpI(c);
    if (auto c = dyn_cast<arith::CmpFOp>(op))
      return emitCmpF(c);
    if (auto s = dyn_cast<arith::SelectOp>(op))
      return emitSelect(s);
    if (auto f = dyn_cast<scf::ForOp>(op))
      return emitFor(f);
    if (auto i = dyn_cast<scf::IfOp>(op))
      return emitIf(i);
    if (auto r = dyn_cast<tt::ReduceOp>(op))
      return emitReduce(r);
    if (auto d = dyn_cast<tt::DotOp>(op))
      return emitDot(d);
    if (auto c = dyn_cast<ttg::ConvertLayoutOp>(op))
      return emitConvertLayout(c);
    if (isa<scf::YieldOp>(op))
      return success();
    if (isa<tt::ReturnOp>(op)) {
      os << ind() << "return;\n";
      return success();
    }
    op->emitError("EmitMSL: unhandled op '" + op->getName().getStringRef() +
                  "'");
    return failure();
  }

  std::string floatLit(const APFloat &v) {
    if (v.isInfinity())
      return v.isNegative() ? "(-INFINITY)" : "INFINITY";
    if (v.isNaN())
      return "NAN";
    return std::to_string(v.convertToDouble());
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
          lit = floatLit(dense.getSplatValue<APFloat>());
        else
          lit = std::to_string(
              dense.getSplatValue<APInt>().getSExtValue());
        os << ind() << "" << sc << " " << id << " = " << lit << ";\n";
        ids.push_back(id);
      }
      valMap[res] = ids;
      return success();
    }
    std::string sc = mslScalarType(res.getType());
    std::string id = fresh();
    std::string lit;
    if (auto fa = dyn_cast<FloatAttr>(op.getValue()))
      lit = floatLit(fa.getValue());
    else if (auto ia = dyn_cast<IntegerAttr>(op.getValue()))
      lit = std::to_string(ia.getInt());
    else {
      op.emitError("EmitMSL: unsupported scalar constant");
      return failure();
    }
    os << ind() << "" << sc << " " << id << " = " << lit << ";\n";
    bindScalar(res, id);
    return success();
  }

  LogicalResult emitProgramId(tt::GetProgramIdOp op) {
    std::string id = fresh();
    const char *comp = op.getAxis() == tt::ProgramIDDim::X   ? "x"
                       : op.getAxis() == tt::ProgramIDDim::Y ? "y"
                                                             : "z";
    os << ind() << "int " << id << " = (int)(" << tgposId << "." << comp << ");\n";
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
      os << ind() << "int " << id << " = " << start << " + " << off << ";\n";
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

  // Map result registers to source registers by matching per-register
  // coordinates. Handles tt.expand_dims (insert a size-1 dim at `axis`) and
  // tt.broadcast (replicate size-1 source dims). Value carries through; only
  // the register->register permutation/replication changes.
  LogicalResult emitReshapeLike(Value res, Value src, int axis, bool isExpand) {
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    auto &srcNames = names(src);
    int srcRc = regCount(src);
    int resRc = regCount(res);

    llvm::DenseMap<uint64_t, int> srcByCoord;
    auto keyOf = [](ArrayRef<int32_t> c) -> uint64_t {
      uint64_t k = 0;
      for (int32_t v : c)
        k = k * 100003u + (uint32_t)v + 1;
      return k;
    };
    for (int r = 0; r < srcRc; ++r)
      srcByCoord[keyOf(registerCoords(srcTy, r))] = r;

    auto srcShape = srcTy.getShape();
    SmallVector<std::string> ids;
    for (int r = 0; r < resRc; ++r) {
      SmallVector<int32_t> rc = registerCoords(resTy, r);
      SmallVector<int32_t> sc;
      if (isExpand) {
        for (int d = 0; d < (int)rc.size(); ++d)
          if (d != axis)
            sc.push_back(rc[d]);
      } else {
        for (int d = 0; d < (int)rc.size(); ++d)
          sc.push_back(srcShape[d] == 1 ? 0 : rc[d]);
      }
      auto it = srcByCoord.find(keyOf(sc));
      if (it == srcByCoord.end()) {
        res.getDefiningOp()->emitError(
            "EmitMSL: reshape register coordinate has no source");
        return failure();
      }
      ids.push_back(srcNames[srcNames.size() == 1 ? 0 : it->second]);
    }
    valMap[res] = ids;
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
      os << ind() << "" << sc.str() << " " << id << " = (" << a << " " << binop.str()
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

  LogicalResult emitCmpF(arith::CmpFOp op) {
    const char *o;
    switch (op.getPredicate()) {
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT:
      o = "<";
      break;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE:
      o = "<=";
      break;
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT:
      o = ">";
      break;
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE:
      o = ">=";
      break;
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ:
      o = "==";
      break;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE:
      o = "!=";
      break;
    default:
      op.emitError("EmitMSL: unsupported cmpf predicate");
      return failure();
    }
    return emitElementwise(op, o, "bool");
  }

  LogicalResult emitSelect(arith::SelectOp op) {
    Value res = op.getResult();
    auto &cond = names(op.getCondition());
    auto &tval = names(op.getTrueValue());
    auto &fval = names(op.getFalseValue());
    std::string sc = mslScalarType(elementScalarType(res.getType()));
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      const std::string &c = cond[cond.size() == 1 ? 0 : r];
      const std::string &t = tval[tval.size() == 1 ? 0 : r];
      const std::string &f = fval[fval.size() == 1 ? 0 : r];
      os << ind() << sc << " " << id << " = " << c << " ? " << t << " : " << f
         << ";\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  LogicalResult emitRegionBody(Region &region) {
    Block &blk = region.front();
    for (Operation &op : blk.without_terminator())
      if (failed(emitOp(&op)))
        return failure();
    return success();
  }

  // Reassign the destination variable names from a yield's operands. Used to
  // resolve scf loop-carried and if-result values MSL-style.
  void emitYieldAssign(Operation *term,
                       const SmallVector<SmallVector<std::string>> &dsts) {
    for (auto [i, operand] : llvm::enumerate(term->getOperands())) {
      auto &src = names(operand);
      const SmallVector<std::string> &dst = dsts[i];
      for (size_t r = 0; r < dst.size(); ++r)
        os << ind() << dst[r] << " = " << src[src.size() == 1 ? 0 : r]
           << ";\n";
    }
  }

  SmallVector<std::string> declResultVars(Value v, StringRef init) {
    std::string sc = mslScalarType(elementScalarType(v.getType()));
    int rc = regCount(v);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      os << ind() << sc << " " << id;
      if (!init.empty())
        os << " = " << init.str();
      os << ";\n";
      ids.push_back(id);
    }
    return ids;
  }

  LogicalResult emitFor(scf::ForOp op) {
    SmallVector<SmallVector<std::string>> carried;
    for (auto [init, res] :
         llvm::zip(op.getInitArgs(), op.getResults())) {
      auto &initNames = names(init);
      SmallVector<std::string> vars =
          declResultVars(res, StringRef());
      for (size_t r = 0; r < vars.size(); ++r)
        os << ind() << vars[r] << " = "
           << initNames[initNames.size() == 1 ? 0 : r] << ";\n";
      valMap[op.getRegionIterArg(carried.size())] = vars;
      valMap[res] = vars;
      carried.push_back(vars);
    }

    std::string iv = fresh();
    const std::string &lo = names(op.getLowerBound())[0];
    const std::string &hi = names(op.getUpperBound())[0];
    const std::string &st = names(op.getStep())[0];
    bindScalar(op.getInductionVar(), iv);
    os << ind() << "for (int " << iv << " = " << lo << "; " << iv << " < " << hi
       << "; " << iv << " += " << st << ") {\n";
    ++indent;
    if (failed(emitRegionBody(op.getRegion())))
      return failure();
    emitYieldAssign(op.getBody()->getTerminator(), carried);
    --indent;
    os << ind() << "}\n";
    return success();
  }

  LogicalResult emitIf(scf::IfOp op) {
    SmallVector<SmallVector<std::string>> results;
    for (Value res : op.getResults())
      results.push_back(declResultVars(res, StringRef()));

    const std::string &c = names(op.getCondition())[0];
    os << ind() << "if (" << c << ") {\n";
    ++indent;
    if (failed(emitRegionBody(op.getThenRegion())))
      return failure();
    if (!results.empty())
      emitYieldAssign(op.thenBlock()->getTerminator(), results);
    --indent;
    os << ind() << "}";
    if (!op.getElseRegion().empty()) {
      os << " else {\n";
      ++indent;
      if (failed(emitRegionBody(op.getElseRegion())))
        return failure();
      if (!results.empty())
        emitYieldAssign(op.elseBlock()->getTerminator(), results);
      --indent;
      os << ind() << "}";
    }
    os << "\n";
    for (auto [i, res] : llvm::enumerate(op.getResults()))
      valMap[res] = results[i];
    return success();
  }

  // Emit the combiner region as an MSL statement writing `dst` from operands
  // `a` and `b`, translating its single arith op. Reused for register, lane
  // and warp combine steps.
  LogicalResult emitCombine(Region &region, StringRef dst, StringRef a,
                            StringRef b, StringRef sc) {
    Block &blk = region.front();
    Value lhs = blk.getArgument(0);
    Value rhs = blk.getArgument(1);
    bindScalar(lhs, a.str());
    bindScalar(rhs, b.str());
    std::string expr;
    for (Operation &o : blk.without_terminator()) {
      std::string id = fresh();
      if (isa<arith::AddFOp>(o))
        os << ind() << sc.str() << " " << id << " = (" << names(o.getOperand(0))[0]
           << " + " << names(o.getOperand(1))[0] << ");\n";
      else if (isa<arith::MulFOp>(o))
        os << ind() << sc.str() << " " << id << " = (" << names(o.getOperand(0))[0]
           << " * " << names(o.getOperand(1))[0] << ");\n";
      else if (isa<arith::AddIOp>(o))
        os << ind() << sc.str() << " " << id << " = (" << names(o.getOperand(0))[0]
           << " + " << names(o.getOperand(1))[0] << ");\n";
      else if (isa<arith::MulIOp>(o))
        os << ind() << sc.str() << " " << id << " = (" << names(o.getOperand(0))[0]
           << " * " << names(o.getOperand(1))[0] << ");\n";
      else if (isa<arith::MaxNumFOp, arith::MaximumFOp>(o))
        os << ind() << sc.str() << " " << id << " = max("
           << names(o.getOperand(0))[0] << ", " << names(o.getOperand(1))[0]
           << ");\n";
      else if (isa<arith::MinNumFOp, arith::MinimumFOp>(o))
        os << ind() << sc.str() << " " << id << " = min("
           << names(o.getOperand(0))[0] << ", " << names(o.getOperand(1))[0]
           << ");\n";
      else if (isa<arith::MaxSIOp, arith::MaxUIOp>(o))
        os << ind() << sc.str() << " " << id << " = max("
           << names(o.getOperand(0))[0] << ", " << names(o.getOperand(1))[0]
           << ");\n";
      else if (isa<arith::MinSIOp, arith::MinUIOp>(o))
        os << ind() << sc.str() << " " << id << " = min("
           << names(o.getOperand(0))[0] << ", " << names(o.getOperand(1))[0]
           << ");\n";
      else {
        o.emitError("EmitMSL: unsupported reduce combiner op");
        return failure();
      }
      bindScalar(o.getResult(0), id);
      expr = id;
    }
    os << ind() << dst.str() << " = " << expr << ";\n";
    return success();
  }

  // Bitmask over lane (or warp) bits that reduce the given axis: a bit reduces
  // if its LinearLayout basis maps to a nonzero coordinate on the reduced
  // out-dim, i.e. distinct lanes/warps hold distinct axis elements.
  unsigned reduceMask(const tt::LinearLayout &ll, StringAttr inDim,
                      StringAttr outDim) {
    unsigned mask = 0;
    if (!ll.hasInDim(inDim))
      return 0;
    for (int b = 0, n = ll.getInDimSizeLog2(inDim); b < n; ++b)
      if (ll.getBasis(inDim, b, outDim) != 0)
        mask |= (1u << b);
    return mask;
  }

  LogicalResult emitReduce(tt::ReduceOp op) {
    if (op.getNumOperands() != 1) {
      op.emitError("EmitMSL: only single-operand reduce supported");
      return failure();
    }
    Value src = op.getOperand(0);
    auto srcTy = cast<RankedTensorType>(src.getType());
    Value res = op.getResult()[0];
    if (isa<RankedTensorType>(res.getType())) {
      op.emitError("EmitMSL: only scalar-result reduce supported");
      return failure();
    }
    std::string sc = mslScalarType(elementScalarType(res.getType()));
    Region &region = op.getCombineOp();
    auto &srcNames = names(src);

    std::string acc = fresh();
    os << ind() << sc << " " << acc << " = " << srcNames[0] << ";\n";
    for (size_t r = 1; r < srcNames.size(); ++r)
      if (failed(emitCombine(region, acc, acc, srcNames[r], sc)))
        return failure();

    MLIRContext *ctx = op.getContext();
    tt::LinearLayout ll = ttg::toLinearLayout(srcTy);
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto outDim = *ll.getOutDimNames().begin();

    unsigned laneMask = reduceMask(ll, kLane, outDim);
    for (int bit = 31; bit >= 0; --bit) {
      unsigned m = 1u << bit;
      if ((laneMask & m) == 0)
        continue;
      std::string other = fresh();
      os << ind() << sc << " " << other << " = simd_shuffle_xor(" << acc << ", "
         << m << "u);\n";
      if (failed(emitCombine(region, acc, acc, other, sc)))
        return failure();
    }

    unsigned warpMask = reduceMask(ll, kWarp, outDim);
    if (warpMask != 0) {
      auto kWarpDim = StringAttr::get(ctx, "warp");
      int numWarps = ll.getInDimSize(kWarpDim);
      std::string scratch = "__red_scratch_" + std::to_string(nextId++);
      os << ind() << "threadgroup " << sc << " " << scratch << "[" << numWarps
         << "];\n";
      os << ind() << "if (" << laneId << " == 0) " << scratch << "[" << warpId
         << "] = " << acc << ";\n";
      os << ind()
         << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      std::string wacc = fresh();
      os << ind() << sc << " " << wacc << " = " << scratch << "[0];\n";
      for (int w = 1; w < numWarps; ++w) {
        std::string wv = fresh();
        os << ind() << sc << " " << wv << " = " << scratch << "[" << w << "];\n";
        if (failed(emitCombine(region, wacc, wacc, wv, sc)))
          return failure();
      }
      acc = wacc;
    }

    bindScalar(res, acc);
    return success();
  }

  int tgScratchId = 0;

  // Full flat size (product of shape) of a tensor tile.
  int64_t tileSize(RankedTensorType rt) {
    int64_t n = 1;
    for (int64_t d : rt.getShape())
      n *= d;
    return n;
  }

  // Row-major flat offset expression (into a full tile buffer) for register r.
  std::string flatTileOffset(RankedTensorType rt, int reg) {
    MLIRContext *ctx = rt.getContext();
    tt::LinearLayout ll = ttg::toLinearLayout(rt);
    auto outNames = llvm::to_vector(ll.getOutDimNames());
    auto shape = rt.getShape();
    std::string expr;
    int64_t stride = 1;
    for (int d = (int)outNames.size() - 1; d >= 0; --d) {
      std::string c = layoutCoordExpr(rt, reg, outNames[d]);
      std::string term = stride == 1 ? c : ("(" + c + " * " +
                                            std::to_string(stride) + ")");
      expr = expr.empty() ? term : ("(" + expr + " + " + term + ")");
      stride *= shape[d];
    }
    (void)ctx;
    return expr.empty() ? "0" : expr;
  }

  // General layout conversion via a full-tile threadgroup round-trip: every
  // thread writes its source registers at their (row,col) tile offset, barrier,
  // then reads its destination registers back from the same offsets. Correct
  // for arbitrary distributed src/dst layouts (moves data across lanes/warps).
  LogicalResult emitConvertLayout(ttg::ConvertLayoutOp op) {
    Value src = op.getSrc();
    Value res = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    Type elemTy = resTy.getElementType();
    bool isPtr = isa<tt::PointerType>(elemTy);
    std::string ptrTy = mslStorageType(resTy);
    std::string sc = isPtr ? "ulong" : ptrTy;
    auto &srcNames = names(src);

    std::string buf = "__cvt_scratch_" + std::to_string(tgScratchId++);
    os << ind() << "threadgroup " << sc << " " << buf << "["
       << tileSize(srcTy) << "];\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    for (int r = 0, n = regCount(src); r < n; ++r) {
      std::string sv =
          isPtr ? "(ulong)" + srcNames[r] : srcNames[r];
      os << ind() << buf << "[" << flatTileOffset(srcTy, r) << "] = " << sv
         << ";\n";
    }
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    SmallVector<std::string> ids;
    for (int r = 0, n = regCount(res); r < n; ++r) {
      std::string id = fresh();
      std::string rd = buf + "[" + flatTileOffset(resTy, r) + "]";
      if (isPtr)
        rd = "(" + ptrTy + ")" + rd;
      os << ind() << ptrTy << " " << id << " = " << rd << ";\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  // Lower tt.dot to MSL simdgroup_matrix 8x8 fragment MMA. A (MxK) and B (KxN)
  // per-thread registers are staged row-major into threadgroup memory; one
  // simdgroup cooperatively runs the 8x8 fragment MMA loop over K into an MxN
  // accumulator, stores it to threadgroup, and every thread reads its C result
  // registers back. Emits simdgroup_load / simdgroup_multiply_accumulate /
  // simdgroup_store only, never air.*.
  static bool isDotOperandElem(Type t) {
    return t.isF32() || t.isF16() || t.isBF16();
  }
  static std::string sgFragType(Type t) {
    if (t.isF16())
      return "simdgroup_half8x8";
    if (t.isBF16())
      return "simdgroup_bfloat8x8";
    return "simdgroup_float8x8";
  }
  static std::string sgOperandScalar(Type t) {
    if (t.isF16())
      return "half";
    if (t.isBF16())
      return "bfloat";
    return "float";
  }

  LogicalResult emitDot(tt::DotOp op) {
    auto aTy = cast<RankedTensorType>(op.getA().getType());
    auto bTy = cast<RankedTensorType>(op.getB().getType());
    auto cTy = cast<RankedTensorType>(op.getResult().getType());
    Type aElem = aTy.getElementType();
    Type bElem = bTy.getElementType();
    Type cElem = cTy.getElementType();
    if (aTy.getRank() != 2 || !isDotOperandElem(aElem) ||
        !isDotOperandElem(bElem) || aElem != bElem ||
        !(cElem.isF32() || cElem.isF16())) {
      op.emitError("EmitMSL: unsupported tt.dot operand/accumulator types");
      return failure();
    }
    int64_t M = cTy.getShape()[0];
    int64_t N = cTy.getShape()[1];
    int64_t K = aTy.getShape()[1];
    if (M % 8 || N % 8 || K % 8) {
      op.emitError("EmitMSL: tt.dot tile dims must be multiples of 8");
      return failure();
    }

    auto &aNames = names(op.getA());
    auto &bNames = names(op.getB());
    auto &cInit = names(op.getC());

    std::string opScalar = sgOperandScalar(aElem);
    std::string accScalar = mslScalarType(cElem);
    std::string opFrag = sgFragType(aElem);
    std::string accFrag = sgFragType(cElem);

    std::string tgA = "__dotA_" + std::to_string(tgScratchId++);
    std::string tgB = "__dotB_" + std::to_string(tgScratchId++);
    std::string tgC = "__dotC_" + std::to_string(tgScratchId++);
    os << ind() << "threadgroup " << opScalar << " " << tgA << "[" << M * K
       << "];\n";
    os << ind() << "threadgroup " << opScalar << " " << tgB << "[" << K * N
       << "];\n";
    os << ind() << "threadgroup " << accScalar << " " << tgC << "[" << M * N
       << "];\n";

    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    for (int r = 0, n = regCount(op.getA()); r < n; ++r)
      os << ind() << tgA << "[" << flatTileOffset(aTy, r) << "] = " << aNames[r]
         << ";\n";
    for (int r = 0, n = regCount(op.getB()); r < n; ++r)
      os << ind() << tgB << "[" << flatTileOffset(bTy, r) << "] = " << bNames[r]
         << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

    int64_t mT = M / 8, nT = N / 8, kT = K / 8;
    os << ind() << "if (" << warpId << " == 0) {\n";
    ++indent;
    for (int64_t mi = 0; mi < mT; ++mi)
      for (int64_t ni = 0; ni < nT; ++ni) {
        std::string acc = "acc_" + std::to_string(mi) + "_" + std::to_string(ni);
        os << ind() << accFrag << " " << acc << " = " << accFrag << "(0.0f);\n";
        for (int64_t ki = 0; ki < kT; ++ki) {
          std::string fa = fresh(), fb = fresh();
          os << ind() << opFrag << " " << fa << ";\n";
          os << ind() << "simdgroup_load(" << fa << ", " << tgA << " + "
             << (mi * 8 * K + ki * 8) << ", " << K << ");\n";
          os << ind() << opFrag << " " << fb << ";\n";
          os << ind() << "simdgroup_load(" << fb << ", " << tgB << " + "
             << (ki * 8 * N + ni * 8) << ", " << N << ");\n";
          os << ind() << "simdgroup_multiply_accumulate(" << acc << ", " << fa
             << ", " << fb << ", " << acc << ");\n";
        }
        os << ind() << "simdgroup_store(" << acc << ", " << tgC << " + "
           << (mi * 8 * N + ni * 8) << ", " << N << ");\n";
      }
    --indent;
    os << ind() << "}\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

    SmallVector<std::string> ids;
    for (int r = 0, n = regCount(op.getResult()); r < n; ++r) {
      std::string id = fresh();
      std::string base = cInit[cInit.size() == 1 ? 0 : r];
      os << ind() << accScalar << " " << id << " = " << tgC << "["
         << flatTileOffset(cTy, r) << "] + " << base << ";\n";
      ids.push_back(id);
    }
    valMap[op.getResult()] = ids;
    return success();
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
      os << ind() << "device " << sc << "* " << id << " = " << b << " + " << o
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
    SmallVector<std::string> *other =
        op.getOther() ? &names(op.getOther()) : nullptr;
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      const std::string &p = ptrs[r];
      std::string init =
          other ? (*other)[other->size() == 1 ? 0 : r] : std::string("0");
      os << ind() << sc << " " << id << " = " << init << ";\n";
      if (hasMask) {
        const std::string &m = (*mask)[mask->size() == 1 ? 0 : r];
        os << ind() << "if (" << m << ") " << id << " = *" << p << ";\n";
      } else {
        os << ind() << id << " = *" << p << ";\n";
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
        os << ind() << "if (" << m << ") *" << p << " = " << v << ";\n";
      } else {
        os << ind() << "*" << p << " = " << v << ";\n";
      }
    }
    return success();
  }

  LogicalResult emitAtomicRMW(tt::AtomicRMWOp op) {
    Value res = op.getResult();
    Type scalarTy = elementScalarType(res.getType());
    std::string sc = mslScalarType(scalarTy);
    bool isFloat = isa<FloatType>(scalarTy);
    tt::RMWOp kind = op.getAtomicRmwOp();

    std::string atomicTy;
    if (isFloat)
      atomicTy = "atomic_float";
    else if (kind == tt::RMWOp::UMAX || kind == tt::RMWOp::UMIN)
      atomicTy = "atomic_uint";
    else
      atomicTy = scalarTy.getIntOrFloatBitWidth() == 64 ? "atomic_long"
                                                        : "atomic_int";

    const char *fn = nullptr;
    switch (kind) {
    case tt::RMWOp::ADD:
    case tt::RMWOp::FADD:
      fn = "atomic_fetch_add_explicit";
      break;
    case tt::RMWOp::MAX:
    case tt::RMWOp::UMAX:
      fn = "atomic_fetch_max_explicit";
      break;
    case tt::RMWOp::MIN:
    case tt::RMWOp::UMIN:
      fn = "atomic_fetch_min_explicit";
      break;
    case tt::RMWOp::AND:
      fn = "atomic_fetch_and_explicit";
      break;
    case tt::RMWOp::OR:
      fn = "atomic_fetch_or_explicit";
      break;
    case tt::RMWOp::XOR:
      fn = "atomic_fetch_xor_explicit";
      break;
    case tt::RMWOp::XCHG:
      fn = "atomic_exchange_explicit";
      break;
    default:
      op.emitError("EmitMSL: unsupported atomic rmw kind");
      return failure();
    }

    auto &ptrs = names(op.getPtr());
    auto &vals = names(op.getVal());
    bool hasMask = op.getMask() != nullptr;
    SmallVector<std::string> *mask = hasMask ? &names(op.getMask()) : nullptr;
    bool uniform = !isa<RankedTensorType>(op.getPtr().getType());
    int rc = ptrs.size();
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      const std::string &p = ptrs[r];
      const std::string &v = vals[vals.size() == 1 ? 0 : r];
      std::string id = fresh();
      std::string call = std::string(fn) + "((device " + atomicTy + "*)" + p +
                         ", " + v + ", memory_order_relaxed)";
      os << ind() << sc << " " << id << " = " << init0(sc) << ";\n";
      std::string guard;
      if (uniform)
        guard = tidId + ".x == 0";
      if (hasMask) {
        const std::string &m = (*mask)[mask->size() == 1 ? 0 : r];
        guard = guard.empty() ? m : guard + " && " + m;
      }
      if (guard.empty())
        os << ind() << id << " = " << call << ";\n";
      else
        os << ind() << "if (" << guard << ") " << id << " = " << call << ";\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  static std::string init0(const std::string &sc) {
    return sc == "float" || sc == "half" ? "0.0" : "0";
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
