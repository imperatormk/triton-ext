// EmitMSL.cpp - direct TritonGPU IR to MSL source printer.
//
// Consumes TTGIR (tt.*, arith.*, scf.*) plus TritonGPU blocked layouts and
// emits Metal Shading Language source. Cross-lane ops, thread indices and
// address spaces come from the MLIR types and TritonGPU LinearLayout, never
// from air.* intrinsics.

#include "TritonAppleGPUToLLVM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseMap.h"
#include <map>
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

static std::string mslKernelName(StringRef name) {
  if (name.starts_with("triton_"))
    return name.str();
  return "triton_" + name.str();
}

static std::string mslStorageType(Type t) {
  if (auto rt = dyn_cast<RankedTensorType>(t))
    t = rt.getElementType();
  if (auto pt = dyn_cast<tt::PointerType>(t))
    return "device " + mslScalarType(pt.getPointeeType()) + "*";
  return mslScalarType(t);
}

static std::string barrierMemFlags(ttg::AddrSpace addrSpace) {
  uint32_t bits = static_cast<uint32_t>(addrSpace);
  bool device = bits & (static_cast<uint32_t>(ttg::AddrSpace::GlobalRead) |
                        static_cast<uint32_t>(ttg::AddrSpace::GlobalWrite));
  bool tg = bits & static_cast<uint32_t>(ttg::AddrSpace::Local);
  if (device && tg)
    return "mem_flags::mem_threadgroup | mem_flags::mem_device";
  if (device)
    return "mem_flags::mem_device";
  return "mem_flags::mem_threadgroup";
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

  struct MemDescInfo {
    std::string buf;
    std::string baseOffset;
  };
  llvm::DenseMap<Value, MemDescInfo> memdescMap;

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
    os << "kernel void " << mslKernelName(func.getName()) << "(\n";

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

    poolBytes = 0;
    Block &body = func.getBody().front();
    for (Operation &op : body)
      scanPool(&op);
    if (poolBytes > 0) {
      poolBuf = "__pool";
      os << ind() << "threadgroup char " << poolBuf << "[" << poolBytes
         << "];\n";
    }


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
    if (auto a = dyn_cast<tt::AtomicCASOp>(op))
      return emitAtomicCAS(a);
    if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp, arith::DivSIOp,
            arith::DivUIOp, arith::RemSIOp, arith::RemUIOp>(op))
      return emitIntBinary(op);
    if (isa<arith::AndIOp>(op))
      return emitElementwise(
          op, "&", mslScalarType(elementScalarType(op->getResult(0).getType())));
    if (isa<arith::OrIOp>(op))
      return emitElementwise(
          op, "|", mslScalarType(elementScalarType(op->getResult(0).getType())));
    if (isa<arith::XOrIOp>(op))
      return emitElementwise(
          op, "^", mslScalarType(elementScalarType(op->getResult(0).getType())));
    if (isa<arith::AddFOp, arith::MulFOp, arith::SubFOp, arith::DivFOp>(op))
      return emitFloatBinary(op);
    if (isa<arith::RemFOp>(op))
      return emitMinMax(op, "metal::fmod");
    if (isa<arith::NegFOp>(op)) {
      Value res = op->getResult(0);
      std::string sc = mslScalarType(elementScalarType(res.getType()));
      auto &a = names(op->getOperand(0));
      int rc = regCount(res);
      SmallVector<std::string> ids;
      for (int r = 0; r < rc; ++r) {
        std::string id = fresh();
        os << ind() << sc << " " << id << " = -" << a[a.size() == 1 ? 0 : r]
           << ";\n";
        ids.push_back(id);
      }
      valMap[res] = ids;
      return success();
    }
    if (isa<arith::MaxNumFOp, arith::MaximumFOp, arith::MaxSIOp, arith::MaxUIOp>(
            op))
      return emitMinMax(op, "max");
    if (isa<arith::MinNumFOp, arith::MinimumFOp, arith::MinSIOp, arith::MinUIOp>(
            op))
      return emitMinMax(op, "min");
    if (auto c = dyn_cast<arith::CmpIOp>(op))
      return emitCmpI(c);
    if (auto c = dyn_cast<arith::CmpFOp>(op))
      return emitCmpF(c);
    if (auto s = dyn_cast<arith::SelectOp>(op))
      return emitSelect(s);
    if (op->getDialect() ==
        op->getContext()->getLoadedDialect<math::MathDialect>())
      return emitMathUnary(op);
    if (isa<arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp,
            arith::ExtFOp, arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp,
            arith::TruncIOp>(op))
      return emitCast(op);
    if (isa<arith::BitcastOp, tt::BitcastOp>(op))
      return emitBitcast(op);
    if (auto f = dyn_cast<tt::FpToFpOp>(op))
      return emitCast(op);
    if (auto c = dyn_cast<tt::ClampFOp>(op))
      return emitClamp(c);
    if (isa<tt::MulhiUIOp>(op))
      return emitMinMax(op, "mulhi");
    if (isa<tt::PreciseSqrtOp>(op))
      return emitUnary(
          op, "metal::sqrt",
          mslScalarType(elementScalarType(op->getResult(0).getType())));
    if (isa<tt::PreciseDivFOp>(op))
      return emitFloatBinary(op);
    if (isa<tt::AssertOp>(op)) {
      for (Value r : op->getResults())
        valMap[r] = SmallVector<std::string>{};
      return success();
    }
    if (isa<tt::PrintOp>(op)) {
      for (Value r : op->getResults())
        valMap[r] = SmallVector<std::string>{};
      return success();
    }
    if (auto f = dyn_cast<scf::ForOp>(op))
      return emitFor(f);
    if (auto i = dyn_cast<scf::IfOp>(op))
      return emitIf(i);
    if (auto w = dyn_cast<scf::WhileOp>(op))
      return emitWhile(w);
    if (auto r = dyn_cast<tt::ReduceOp>(op))
      return emitReduce(r);
    if (auto s = dyn_cast<tt::ScanOp>(op))
      return emitScan(s);
    if (auto d = dyn_cast<tt::DotOp>(op))
      return emitDot(d);
    if (auto c = dyn_cast<ttg::ConvertLayoutOp>(op))
      return emitConvertLayout(c);
    if (auto t = dyn_cast<tt::TransOp>(op))
      return emitTrans(t);
    if (auto j = dyn_cast<tt::JoinOp>(op))
      return emitJoin(j);
    if (auto s = dyn_cast<tt::SplitOp>(op))
      return emitSplit(s);
    if (auto c = dyn_cast<tt::CatOp>(op))
      return emitCat(c);
    if (auto r = dyn_cast<tt::ReshapeOp>(op))
      return emitReshape(r);
    if (auto a = dyn_cast<ttg::LocalAllocOp>(op))
      return emitLocalAlloc(a);
    if (auto i = dyn_cast<ttg::MemDescIndexOp>(op))
      return emitMemDescIndex(i);
    if (auto c = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op))
      return emitAsyncCopy(c);
    if (auto l = dyn_cast<ttg::LocalStoreOp>(op))
      return emitLocalStore(l);
    if (auto l = dyn_cast<ttg::LocalLoadOp>(op))
      return emitLocalLoad(l);
    if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(op)) {
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup | "
                     "mem_flags::mem_device);\n";
      for (Value r : op->getResults())
        valMap[r] = SmallVector<std::string>{};
      return success();
    }
    if (auto b = dyn_cast<ttg::BarrierOp>(op)) {
      os << ind() << "threadgroup_barrier(" << barrierMemFlags(b.getAddrSpace())
         << ");\n";
      return success();
    }
    if (isa<mlir::gpu::BarrierOp>(op)) {
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup | "
                     "mem_flags::mem_device);\n";
      return success();
    }
    if (isa<ttg::LocalDeallocOp>(op))
      return success();
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
      if (!dense) {
        op.emitError("EmitMSL: unsupported tensor constant");
        return failure();
      }
      std::string sc = mslScalarType(rt.getElementType());
      bool isFloat = isa<FloatType>(rt.getElementType());
      int rc = regCount(res);
      SmallVector<std::string> ids;
      if (dense.isSplat()) {
        std::string lit = isFloat
                              ? floatLit(dense.getSplatValue<APFloat>())
                              : std::to_string(
                                    dense.getSplatValue<APInt>().getSExtValue());
        for (int r = 0; r < rc; ++r) {
          std::string id = fresh();
          os << ind() << sc << " " << id << " = " << lit << ";\n";
          ids.push_back(id);
        }
        valMap[res] = ids;
        return success();
      }

      std::string tbl = fresh();
      os << ind() << sc << " " << tbl << "[" << dense.getNumElements()
         << "] = {";
      int64_t n = 0;
      if (isFloat) {
        for (const APFloat &v : dense.getValues<APFloat>())
          os << (n++ ? ", " : "") << floatLit(v);
      } else {
        for (const APInt &v : dense.getValues<APInt>())
          os << (n++ ? ", " : "") << std::to_string(v.getSExtValue());
      }
      os << "};\n";
      for (int r = 0; r < rc; ++r) {
        std::string id = fresh();
        os << ind() << sc << " " << id << " = " << tbl << "["
           << flatTileOffset(rt, r) << "];\n";
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

  static uint64_t coordKey(ArrayRef<int32_t> c) {
    uint64_t k = 0;
    for (int32_t v : c)
      k = k * 100003u + (uint32_t)v + 1;
    return k;
  }

  // tt.join(a, b): both operands share a layout; result adds a trailing size-2
  // dim whose two entries live in the same thread (distinct registers). Result
  // register r with trailing coord t sources from operand t at the result
  // coords minus the trailing dim.
  LogicalResult emitJoin(tt::JoinOp op) {
    Value res = op.getResult();
    auto resTy = cast<RankedTensorType>(res.getType());
    int trailing = resTy.getRank() - 1;
    SmallVector<SmallVector<std::string> *> srcNames = {&names(op.getLhs()),
                                                        &names(op.getRhs())};
    auto srcTy = cast<RankedTensorType>(op.getLhs().getType());
    int srcRc = regCount(op.getLhs());

    llvm::DenseMap<uint64_t, int> srcByCoord;
    for (int r = 0; r < srcRc; ++r)
      srcByCoord[coordKey(registerCoords(srcTy, r))] = r;

    int resRc = regCount(res);
    SmallVector<std::string> ids(resRc);
    for (int r = 0; r < resRc; ++r) {
      SmallVector<int32_t> rc = registerCoords(resTy, r);
      int t = rc[trailing];
      rc.pop_back();
      auto it = srcByCoord.find(coordKey(rc));
      if (it == srcByCoord.end() || t < 0 || t > 1) {
        op.emitError("EmitMSL: join register coordinate has no source");
        return failure();
      }
      auto &sn = *srcNames[t];
      ids[r] = sn[sn.size() == 1 ? 0 : it->second];
    }
    valMap[res] = ids;
    return success();
  }

  // tt.split(x): inverse of join. Two results share a layout; result k register
  // r sources from x at the result coords with trailing coord k appended.
  LogicalResult emitSplit(tt::SplitOp op) {
    Value src = op.getOperand();
    auto srcTy = cast<RankedTensorType>(src.getType());
    int trailing = srcTy.getRank() - 1;
    auto &srcNames = names(src);
    int srcRc = regCount(src);

    llvm::DenseMap<uint64_t, int> srcByCoord;
    for (int r = 0; r < srcRc; ++r)
      srcByCoord[coordKey(registerCoords(srcTy, r))] = r;

    for (int k = 0; k < 2; ++k) {
      Value res = op.getResult(k);
      auto resTy = cast<RankedTensorType>(res.getType());
      int resRc = regCount(res);
      SmallVector<std::string> ids(resRc);
      for (int r = 0; r < resRc; ++r) {
        SmallVector<int32_t> rc = registerCoords(resTy, r);
        rc.push_back(k);
        auto it = srcByCoord.find(coordKey(rc));
        if (it == srcByCoord.end()) {
          op.emitError("EmitMSL: split register coordinate has no source");
          return failure();
        }
        (void)trailing;
        ids[r] = srcNames[srcNames.size() == 1 ? 0 : it->second];
      }
      valMap[res] = ids;
    }
    return success();
  }

  LogicalResult emitIntBinary(Operation *op) {
    const char *o = isa<arith::AddIOp>(op)   ? "+"
                    : isa<arith::SubIOp>(op) ? "-"
                    : isa<arith::MulIOp>(op) ? "*"
                    : isa<arith::DivSIOp, arith::DivUIOp>(op) ? "/"
                                                              : "%";
    std::string sc =
        mslScalarType(elementScalarType(op->getResult(0).getType()));
    if (isa<arith::DivUIOp, arith::RemUIOp>(op) && sc.front() != 'u')
      sc = "u" + sc;
    return emitElementwise(op, o, sc);
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

  LogicalResult emitMinMax(Operation *op, StringRef fn) {
    Value res = op->getResult(0);
    std::string sc = mslScalarType(elementScalarType(res.getType()));
    auto &lhs = names(op->getOperand(0));
    auto &rhs = names(op->getOperand(1));
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      const std::string &a = lhs[lhs.size() == 1 ? 0 : r];
      const std::string &b = rhs[rhs.size() == 1 ? 0 : r];
      os << ind() << sc << " " << id << " = " << fn.str() << "(" << a << ", "
         << b << ");\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  LogicalResult emitUnary(Operation *op, StringRef fn, StringRef sc) {
    Value res = op->getResult(0);
    auto &a = names(op->getOperand(0));
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      const std::string &v = a[a.size() == 1 ? 0 : r];
      os << ind() << sc.str() << " " << id << " = " << fn.str() << "(" << v
         << ");\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  LogicalResult emitMathUnary(Operation *op) {
    std::string sc = mslScalarType(elementScalarType(op->getResult(0).getType()));
    StringRef n = op->getName().getStringRef();
    static const llvm::StringMap<const char *> unary = {
        {"math.exp", "metal::exp"},       {"math.exp2", "metal::exp2"},
        {"math.log", "metal::log"},       {"math.log2", "metal::log2"},
        {"math.log10", "metal::log10"},   {"math.sin", "metal::sin"},
        {"math.cos", "metal::cos"},       {"math.tan", "metal::tan"},
        {"math.tanh", "metal::tanh"},     {"math.sinh", "metal::sinh"},
        {"math.cosh", "metal::cosh"},     {"math.asin", "metal::asin"},
        {"math.acos", "metal::acos"},     {"math.atan", "metal::atan"},
        {"math.sqrt", "metal::sqrt"},     {"math.rsqrt", "metal::rsqrt"},
        {"math.cbrt", "metal::cbrt"},     {"math.floor", "metal::floor"},
        {"math.ceil", "metal::ceil"},     {"math.absf", "metal::abs"},
        {"math.absi", "metal::abs"},      {"math.erf", "metal::erf"},
        {"math.round", "metal::round"},   {"math.trunc", "metal::trunc"},
        {"math.roundeven", "metal::rint"}};
    if (auto it = unary.find(n); it != unary.end())
      return emitUnary(op, it->second, sc);
    static const llvm::StringMap<const char *> binary = {
        {"math.atan2", "metal::atan2"},
        {"math.powf", "metal::pow"},
        {"math.fpowi", "metal::pow"},
        {"math.copysign", "metal::copysign"}};
    if (auto it = binary.find(n); it != binary.end())
      return emitMinMax(op, it->second);
    if (n == "math.fma")
      return emitTernary(op, "metal::fma", sc);
    if (n == "math.exp10") {
      Value res = op->getResult(0);
      auto &a = names(op->getOperand(0));
      int rc = regCount(res);
      SmallVector<std::string> ids;
      for (int r = 0; r < rc; ++r) {
        std::string id = fresh();
        os << ind() << sc << " " << id << " = metal::pow((" << sc << ")10, "
           << a[a.size() == 1 ? 0 : r] << ");\n";
        ids.push_back(id);
      }
      valMap[res] = ids;
      return success();
    }
    op->emitError("EmitMSL: unhandled math op '" + n + "'");
    return failure();
  }

  LogicalResult emitTernary(Operation *op, StringRef fn, StringRef sc) {
    Value res = op->getResult(0);
    auto &a = names(op->getOperand(0));
    auto &b = names(op->getOperand(1));
    auto &c = names(op->getOperand(2));
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      os << ind() << sc.str() << " " << id << " = " << fn.str() << "("
         << a[a.size() == 1 ? 0 : r] << ", " << b[b.size() == 1 ? 0 : r] << ", "
         << c[c.size() == 1 ? 0 : r] << ");\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  LogicalResult emitCast(Operation *op) {
    Value res = op->getResult(0);
    std::string dst = mslScalarType(elementScalarType(res.getType()));
    if (dst.empty()) {
      op->emitError("EmitMSL: unhandled cast target type");
      return failure();
    }
    auto &a = names(op->getOperand(0));
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      const std::string &v = a[a.size() == 1 ? 0 : r];
      os << ind() << dst << " " << id << " = static_cast<" << dst << ">(" << v
         << ");\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  LogicalResult emitBitcast(Operation *op) {
    Value res = op->getResult(0);
    std::string dst = mslScalarType(elementScalarType(res.getType()));
    if (dst.empty()) {
      op->emitError("EmitMSL: unhandled bitcast target type");
      return failure();
    }
    auto &a = names(op->getOperand(0));
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      const std::string &v = a[a.size() == 1 ? 0 : r];
      os << ind() << dst << " " << id << " = as_type<" << dst << ">(" << v
         << ");\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  LogicalResult emitClamp(tt::ClampFOp op) {
    Value res = op.getResult();
    std::string sc = mslScalarType(elementScalarType(res.getType()));
    auto &x = names(op.getX());
    auto &lo = names(op.getMin());
    auto &hi = names(op.getMax());
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      const std::string &xv = x[x.size() == 1 ? 0 : r];
      const std::string &lv = lo[lo.size() == 1 ? 0 : r];
      const std::string &hv = hi[hi.size() == 1 ? 0 : r];
      os << ind() << sc << " " << id << " = metal::clamp(" << xv << ", " << lv
         << ", " << hv << ");\n";
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

  static bool isDatalessType(Type t) {
    return isa<ttg::AsyncTokenType>(t);
  }

  LogicalResult emitFor(scf::ForOp op) {
    SmallVector<SmallVector<std::string>> carried;
    for (auto [i, init, res] :
         llvm::enumerate(op.getInitArgs(), op.getResults())) {
      if (isDatalessType(res.getType())) {
        valMap[op.getRegionIterArg(i)] = SmallVector<std::string>{};
        valMap[res] = SmallVector<std::string>{};
        carried.push_back({});
        continue;
      }
      auto &initNames = names(init);
      SmallVector<std::string> vars =
          declResultVars(res, StringRef());
      for (size_t r = 0; r < vars.size(); ++r)
        os << ind() << vars[r] << " = "
           << initNames[initNames.size() == 1 ? 0 : r] << ";\n";
      valMap[op.getRegionIterArg(i)] = vars;
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

  LogicalResult emitWhile(scf::WhileOp op) {
    SmallVector<SmallVector<std::string>> carried;
    for (auto [i, init] : llvm::enumerate(op.getInits())) {
      auto &initNames = names(init);
      SmallVector<std::string> vars = declResultVars(init, StringRef());
      for (size_t r = 0; r < vars.size(); ++r)
        os << ind() << vars[r] << " = "
           << initNames[initNames.size() == 1 ? 0 : r] << ";\n";
      valMap[op.getBeforeArguments()[i]] = vars;
      carried.push_back(vars);
    }

    SmallVector<SmallVector<std::string>> results;
    for (Value res : op.getResults())
      results.push_back(declResultVars(res, StringRef()));

    os << ind() << "while (true) {\n";
    ++indent;
    if (failed(emitRegionBody(op.getBefore())))
      return failure();
    auto cond = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    const std::string &c = names(cond.getCondition())[0];
    os << ind() << "if (!(" << c << ")) {\n";
    ++indent;
    for (auto [i, fwd] : llvm::enumerate(cond.getArgs())) {
      auto &src = names(fwd);
      for (size_t r = 0; r < results[i].size(); ++r)
        os << ind() << results[i][r] << " = " << src[src.size() == 1 ? 0 : r]
           << ";\n";
    }
    os << ind() << "break;\n";
    --indent;
    os << ind() << "}\n";

    for (auto [i, fwd] : llvm::enumerate(cond.getArgs())) {
      SmallVector<std::string> fwdNames = names(fwd);
      valMap[op.getAfterArguments()[i]] = fwdNames;
    }

    if (failed(emitRegionBody(op.getAfter())))
      return failure();
    emitYieldAssign(op.getAfter().front().getTerminator(), carried);
    --indent;
    os << ind() << "}\n";

    for (auto [i, res] : llvm::enumerate(op.getResults()))
      valMap[res] = results[i];
    return success();
  }

  // Single-result, single-op wrapper over emitCombineN. `dst` must be a
  // predeclared MSL variable; assigns the combined value into it.
  LogicalResult emitCombine(Region &region, StringRef dst, StringRef a,
                            StringRef b, StringRef sc) {
    SmallVector<std::string> res;
    if (failed(emitCombineN(region, {a.str()}, {b.str()}, res)))
      return failure();
    os << ind() << dst.str() << " = " << res[0] << ";\n";
    return success();
  }

  // Evaluate a reduce/scan combiner region: bind its 2N block args to the
  // operand names, emit each region op via emitOp, and return the N terminator
  // result names. Region args are ordered a0,a1,...,aN-1,b0,b1,...,bN-1 per the
  // combiner ABI (all left operands, then all right operands).
  LogicalResult emitCombineN(Region &region, ArrayRef<std::string> aVals,
                             ArrayRef<std::string> bVals,
                             SmallVectorImpl<std::string> &results) {
    Block &blk = region.front();
    int n = aVals.size();
    for (int i = 0; i < n; ++i) {
      bindScalar(blk.getArgument(i), aVals[i]);
      bindScalar(blk.getArgument(n + i), bVals[i]);
    }
    for (Operation &o : blk.without_terminator())
      if (failed(emitOp(&o)))
        return failure();
    Operation *term = blk.getTerminator();
    for (Value r : term->getOperands())
      results.push_back(names(r)[0]);
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

  // All distinct values reachable by ORing subsets of the mask's set bits.
  // Used to enumerate the reduced-axis warp partition offsets while keeping the
  // surviving-axis warp bits fixed, so a cross-warp combine never mixes warps
  // that hold different output elements.
  SmallVector<int> subsetsOf(unsigned mask, int numWarps) {
    SmallVector<int> bits;
    for (int b = 0; b < 16; ++b)
      if (mask & (1u << b))
        bits.push_back(b);
    SmallVector<int> vals;
    for (int s = 0; s < (1 << bits.size()); ++s) {
      int v = 0;
      for (int i = 0; i < (int)bits.size(); ++i)
        if (s & (1 << i))
          v |= (1 << bits[i]);
      if (v < numWarps)
        vals.push_back(v);
    }
    return vals;
  }

  LogicalResult emitReduce(tt::ReduceOp op) {
    int nOp = op.getNumOperands();
    auto srcTy = cast<RankedTensorType>(op.getOperand(0).getType());
    int axis = op.getAxis();
    bool tensorResult = isa<RankedTensorType>(op.getResult()[0].getType());

    SmallVector<std::string> scTys(nOp);
    for (int k = 0; k < nOp; ++k)
      scTys[k] = mslScalarType(elementScalarType(op.getResult()[k].getType()));
    Region &region = op.getCombineOp();

    SmallVector<SmallVector<std::string> *> srcNames(nOp);
    for (int k = 0; k < nOp; ++k)
      srcNames[k] = &names(op.getOperand(k));
    int nReg = srcNames[0]->size();

    MLIRContext *ctx = op.getContext();
    tt::LinearLayout ll = ttg::toLinearLayout(srcTy);
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto outDims = llvm::to_vector(ll.getOutDimNames());
    auto redDim = outDims[axis];

    // Group source registers by their coordinate on every non-reduced out-dim
    // (the surviving key). Registers in a group differ only along the reduced
    // axis and reduce together into one output element.
    auto survKey = [&](int reg) {
      SmallVector<int32_t> coords = registerCoords(srcTy, reg);
      std::string key;
      for (int d = 0; d < (int)coords.size(); ++d)
        if (d != axis)
          key += std::to_string(coords[d]) + ",";
      return key;
    };
    std::map<std::string, SmallVector<int>> groups;
    for (int r = 0; r < nReg; ++r)
      groups[survKey(r)].push_back(r);

    unsigned laneMask = reduceMask(ll, kLane, redDim);
    unsigned warpMask = reduceMask(ll, kWarp, redDim);
    int numWarps = ll.hasInDim(kWarp) ? ll.getInDimSize(kWarp) : 1;

    int64_t warpByteStride = 0;
    for (int k = 0; k < nOp; ++k)
      warpByteStride +=
          bitsOf(elementScalarType(op.getResult()[k].getType())) / 8;

    std::map<std::string, SmallVector<std::string>> groupResult;

    int64_t slot = 0;
    for (auto &g : groups) {
      SmallVector<int> &regs = g.second;
      SmallVector<std::string> accs(nOp);
      for (int k = 0; k < nOp; ++k) {
        accs[k] = fresh();
        os << ind() << scTys[k] << " " << accs[k] << " = "
           << (*srcNames[k])[regs[0]] << ";\n";
      }
      for (size_t i = 1; i < regs.size(); ++i) {
        SmallVector<std::string> bVals(nOp);
        for (int k = 0; k < nOp; ++k)
          bVals[k] = (*srcNames[k])[regs[i]];
        SmallVector<std::string> out;
        if (failed(emitCombineN(region, accs, bVals, out)))
          return failure();
        for (int k = 0; k < nOp; ++k)
          os << ind() << accs[k] << " = " << out[k] << ";\n";
      }

      for (int bit = 31; bit >= 0; --bit) {
        unsigned m = 1u << bit;
        if ((laneMask & m) == 0)
          continue;
        SmallVector<std::string> others(nOp);
        for (int k = 0; k < nOp; ++k) {
          others[k] = fresh();
          os << ind() << scTys[k] << " " << others[k] << " = simd_shuffle_xor("
             << accs[k] << ", " << m << "u);\n";
        }
        SmallVector<std::string> out;
        if (failed(emitCombineN(region, accs, others, out)))
          return failure();
        for (int k = 0; k < nOp; ++k)
          os << ind() << accs[k] << " = " << out[k] << ";\n";
      }

      if (warpMask != 0) {
        // Scratch is indexed by (warp * 32 + lane): the surviving axis may live
        // partly on lanes, so each lane must recover ITS own output element.
        // Every lane writes; each lane reads across the reduced-axis warps at
        // its own lane offset. Warps that only vary over reduced-axis bits are
        // summed; surviving-axis warp bits stay fixed to this warp.
        SmallVector<std::string> scratch(nOp);
        int64_t byteOff = slot * numWarps * 32 * warpByteStride;
        for (int k = 0; k < nOp; ++k) {
          scratch[k] = fresh();
          os << ind() << "threadgroup " << scTys[k] << "* " << scratch[k]
             << " = " << poolRegion(byteOff, scTys[k]) << ";\n";
          byteOff += numWarps * 32 *
                     (bitsOf(elementScalarType(op.getResult()[k].getType())) /
                      8);
        }
        os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        for (int k = 0; k < nOp; ++k)
          os << ind() << scratch[k] << "[" << warpId << " * 32 + " << laneId
             << "] = " << accs[k] << ";\n";
        os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        SmallVector<int> redVals = subsetsOf(warpMask, numWarps);
        std::string base = "((" + warpId + " & " + std::to_string(~warpMask) +
                           ") * 32 + " + laneId + ")";
        SmallVector<std::string> wacc(nOp);
        for (int k = 0; k < nOp; ++k) {
          wacc[k] = fresh();
          os << ind() << scTys[k] << " " << wacc[k] << " = " << scratch[k] << "["
             << base << "];\n";
        }
        for (size_t i = 1; i < redVals.size(); ++i) {
          SmallVector<std::string> wv(nOp);
          for (int k = 0; k < nOp; ++k) {
            wv[k] = fresh();
            os << ind() << scTys[k] << " " << wv[k] << " = " << scratch[k] << "["
               << base << " + " << (redVals[i] * 32) << "];\n";
          }
          SmallVector<std::string> out;
          if (failed(emitCombineN(region, wacc, wv, out)))
            return failure();
          for (int k = 0; k < nOp; ++k)
            os << ind() << wacc[k] << " = " << out[k] << ";\n";
        }
        accs = wacc;
      }

      groupResult[g.first] = accs;
      ++slot;
    }

    if (!tensorResult) {
      for (int k = 0; k < nOp; ++k)
        bindScalar(op.getResult()[k], groupResult.begin()->second[k]);
      return success();
    }

    auto resTy = cast<RankedTensorType>(op.getResult()[0].getType());
    int nResReg = regCount(op.getResult()[0]);
    SmallVector<SmallVector<std::string>> resIds(nOp);
    for (int r = 0; r < nResReg; ++r) {
      SmallVector<int32_t> rc = registerCoords(resTy, r);
      std::string key;
      for (int32_t c : rc)
        key += std::to_string(c) + ",";
      auto it = groupResult.find(key);
      if (it == groupResult.end()) {
        op.emitError("EmitMSL: reduce result register has no source group");
        return failure();
      }
      for (int k = 0; k < nOp; ++k)
        resIds[k].push_back(it->second[k]);
    }
    for (int k = 0; k < nOp; ++k)
      valMap[op.getResult()[k]] = resIds[k];
    return success();
  }

  // Ordered (bit, axisStride) pairs for an inDim (lane/warp), only for bits that
  // move along the scanned axis. Sorted by ascending axis stride.
  SmallVector<std::pair<int, int32_t>> axisBits(const tt::LinearLayout &ll,
                                                StringAttr inDim,
                                                StringAttr outDim) {
    SmallVector<std::pair<int, int32_t>> bits;
    if (!ll.hasInDim(inDim))
      return bits;
    for (int b = 0, n = ll.getInDimSizeLog2(inDim); b < n; ++b) {
      int32_t basis = ll.getBasis(inDim, b, outDim);
      if (basis != 0)
        bits.push_back({b, basis});
    }
    llvm::sort(bits, [](auto &a, auto &c) { return a.second < c.second; });
    return bits;
  }

  // Cross-warp inclusive carry for one register run. Applies the prefix of all
  // lower warp partitions to the run's registers, and writes the run's grand
  // total (inclusive across every lane and warp) to runTotalOut. laneScan holds
  // each warp's lane-inclusive total. Self-brackets with threadgroup barriers.
  LogicalResult emitScanWarpCarry(Region &region,
                                  ArrayRef<std::pair<int, int32_t>> warpBits,
                                  ArrayRef<int> regs,
                                  SmallVectorImpl<std::string> &accs,
                                  StringRef laneScan, StringRef sc, bool rev,
                                  StringRef runTotalOut) {
    if (warpBits.empty()) {
      unsigned topLane = rev ? 0u : 31u;
      os << ind() << runTotalOut << " = simd_shuffle(" << laneScan << ", "
         << topLane << "u);\n";
      return success();
    }

    int nParts = 1 << warpBits.size();
    SmallVector<std::string> posTerms;
    for (size_t r = 0; r < warpBits.size(); ++r) {
      int bitIdx = warpBits[r].first;
      posTerms.push_back("((((" + warpId + " >> " + std::to_string(bitIdx) +
                         ") & 1) << " + std::to_string(r) + "))");
    }
    std::string warpPos = posTerms[0];
    for (size_t i = 1; i < posTerms.size(); ++i)
      warpPos = "(" + warpPos + " | " + posTerms[i] + ")";
    std::string wpos = fresh();
    os << ind() << "int " << wpos << " = " << warpPos << ";\n";

    std::string scratch = fresh();
    os << ind() << "threadgroup " << sc << "* " << scratch << " = "
       << poolRegion(0, sc) << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    unsigned totLane = rev ? 0u : 31u;
    os << ind() << "if (" << laneId << " == " << totLane << ") " << scratch
       << "[" << wpos << "] = " << laneScan << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

    std::string grand = fresh();
    os << ind() << sc << " " << grand << " = " << scratch << "[0];\n";
    for (int p = 1; p < nParts; ++p) {
      std::string comb = fresh();
      os << ind() << sc << " " << comb << ";\n";
      if (failed(emitCombine(region, comb,
                             grand, scratch + "[" + std::to_string(p) + "]",
                             sc)))
        return failure();
      os << ind() << grand << " = " << comb << ";\n";
    }
    os << ind() << runTotalOut << " = " << grand << ";\n";

    std::string carry = fresh();
    std::string loop;
    if (rev) {
      os << ind() << sc << " " << carry << " = " << scratch << "[" << wpos
         << " + 1];\n";
      loop = "for (int wp = " + wpos + " + 2; wp < " + std::to_string(nParts) +
             "; ++wp) {";
    } else {
      os << ind() << sc << " " << carry << " = " << scratch << "[0];\n";
      loop = "for (int wp = 1; wp < " + wpos + "; ++wp) {";
    }
    os << ind() << loop << "\n";
    indent++;
    std::string wv = fresh();
    os << ind() << sc << " " << wv << " = " << scratch << "[wp];\n";
    std::string comb = fresh();
    os << ind() << sc << " " << comb << ";\n";
    if (failed(emitCombine(region, comb, carry, wv, sc)))
      return failure();
    os << ind() << carry << " = " << comb << ";\n";
    indent--;
    os << ind() << "}\n";
    std::string guard =
        rev ? (wpos + " < " + std::to_string(nParts - 1)) : (wpos + " > 0");
    for (int r : regs) {
      std::string comb = fresh();
      os << ind() << sc << " " << comb << ";\n";
      if (failed(rev ? emitCombine(region, comb, accs[r], carry, sc)
                     : emitCombine(region, comb, carry, accs[r], sc)))
        return failure();
      os << ind() << accs[r] << " = (" << guard << " ? " << comb << " : "
         << accs[r] << ");\n";
    }
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    return success();
  }

  LogicalResult emitScan(tt::ScanOp op) {
    if (op.getNumOperands() != 1) {
      op.emitError("EmitMSL: only single-operand scan supported");
      return failure();
    }
    bool rev = op.getReverse();
    Value src = op.getOperand(0);
    auto srcTy = cast<RankedTensorType>(src.getType());
    Value res = op.getResult()[0];
    std::string sc = mslScalarType(elementScalarType(res.getType()));
    Region &region = op.getRegion();
    auto &srcNames = names(src);
    int nReg = srcNames.size();

    MLIRContext *ctx = op.getContext();
    tt::LinearLayout ll = ttg::toLinearLayout(srcTy);
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto outDim = *ll.getOutDimNames().begin();

    auto laneBits = axisBits(ll, kLane, outDim);
    auto warpBits = axisBits(ll, kWarp, outDim);

    // Largest lane/warp axis stride: register bits above it are OUTER scan
    // dimensions that straddle the lane/warp span, so registers group into
    // coord-contiguous runs and the cross-run carry runs after lane/warp.
    int32_t laneWarpReach = 0;
    for (auto &pr : laneBits)
      laneWarpReach = std::max(laneWarpReach, pr.second);
    for (auto &pr : warpBits)
      laneWarpReach = std::max(laneWarpReach, pr.second);

    SmallVector<int> runId(nReg, 0);
    for (int r = 0; r < nReg; ++r) {
      int32_t c = registerCoords(srcTy, r)[0];
      runId[r] = laneWarpReach ? (c / (2 * laneWarpReach)) : 0;
    }
    SmallVector<int> runOrder;
    for (int r = 0; r < nReg; ++r)
      if (llvm::find(runOrder, runId[r]) == runOrder.end())
        runOrder.push_back(runId[r]);
    llvm::sort(runOrder, [&](int a, int b) { return rev ? a > b : a < b; });

    SmallVector<std::string> accs(nReg);
    for (int r = 0; r < nReg; ++r) {
      accs[r] = fresh();
      os << ind() << sc << " " << accs[r] << " = " << srcNames[r] << ";\n";
    }

    const char *shuf = rev ? "simd_shuffle_down" : "simd_shuffle_up";
    SmallVector<std::string> runTotals(runOrder.size());

    for (size_t ri = 0; ri < runOrder.size(); ++ri) {
      int run = runOrder[ri];
      SmallVector<int> regs;
      for (int r = 0; r < nReg; ++r)
        if (runId[r] == run)
          regs.push_back(r);
      llvm::sort(regs, [&](int a, int b) {
        int32_t ca = registerCoords(srcTy, a)[0];
        int32_t cb = registerCoords(srcTy, b)[0];
        return rev ? ca > cb : ca < cb;
      });

      for (size_t i = 1; i < regs.size(); ++i) {
        std::string comb = fresh();
        os << ind() << sc << " " << comb << ";\n";
        if (failed(emitCombine(region, comb, accs[regs[i - 1]], accs[regs[i]],
                               sc)))
          return failure();
        os << ind() << accs[regs[i]] << " = " << comb << ";\n";
      }

      std::string laneScan = fresh();
      os << ind() << sc << " " << laneScan << " = " << accs[regs.back()]
         << ";\n";
      for (auto &pr : laneBits) {
        unsigned delta = 1u << pr.first;
        std::string nb = fresh();
        os << ind() << sc << " " << nb << " = " << shuf << "(" << laneScan
           << ", " << delta << "u);\n";
        std::string comb = fresh();
        os << ind() << sc << " " << comb << ";\n";
        if (failed(rev ? emitCombine(region, comb, laneScan, nb, sc)
                       : emitCombine(region, comb, nb, laneScan, sc)))
          return failure();
        std::string guard = rev ? (laneId + " <= " + std::to_string(31u - delta))
                                : (laneId + " >= " + std::to_string(delta));
        os << ind() << laneScan << " = (" << guard << " ? " << comb << " : "
           << laneScan << ");\n";
      }
      if (!laneBits.empty()) {
        std::string lanePrefix = fresh();
        os << ind() << sc << " " << lanePrefix << " = " << shuf << "("
           << laneScan << ", 1u);\n";
        std::string guard = rev ? (laneId + " <= 30") : (laneId + " >= 1");
        for (int r : regs) {
          std::string comb = fresh();
          os << ind() << sc << " " << comb << ";\n";
          if (failed(rev ? emitCombine(region, comb, accs[r], lanePrefix, sc)
                         : emitCombine(region, comb, lanePrefix, accs[r], sc)))
            return failure();
          os << ind() << accs[r] << " = (" << guard << " ? " << comb << " : "
             << accs[r] << ");\n";
        }
      }

      runTotals[ri] = fresh();
      os << ind() << sc << " " << runTotals[ri] << ";\n";
      if (failed(emitScanWarpCarry(region, warpBits, regs, accs, laneScan, sc,
                                   rev, runTotals[ri])))
        return failure();
    }

    for (size_t ri = 1; ri < runOrder.size(); ++ri) {
      std::string carry = runTotals[0];
      for (size_t j = 1; j < ri; ++j) {
        std::string comb = fresh();
        os << ind() << sc << " " << comb << ";\n";
        if (failed(emitCombine(region, comb, carry, runTotals[j], sc)))
          return failure();
        carry = comb;
      }
      int run = runOrder[ri];
      for (int r = 0; r < nReg; ++r) {
        if (runId[r] != run)
          continue;
        std::string comb = fresh();
        os << ind() << sc << " " << comb << ";\n";
        if (failed(rev ? emitCombine(region, comb, accs[r], carry, sc)
                       : emitCombine(region, comb, carry, accs[r], sc)))
          return failure();
        os << ind() << accs[r] << " = " << comb << ";\n";
      }
    }

    SmallVector<std::string> outNames(nReg);
    for (int r = 0; r < nReg; ++r)
      outNames[r] = accs[r];
    valMap[res] = outNames;
    return success();
  }

  int tgScratchId = 0;

  // A single per-kernel threadgroup pool shared by every barrier-separated
  // transient scratch site (convert-layout, dot A/B/C staging, cross-warp
  // reduce). Each site is separated from the next by a threadgroup_barrier, so
  // one region is reused rather than summed into the PSO budget. Buffers that
  // stay live across a loop (ttg.local_alloc rotating stages) do NOT draw from
  // the pool. Sized in bytes to the max single-site footprint; regions are
  // reinterpreted to the site's element type.
  std::string poolBuf;
  int64_t poolBytes = 0;

  static int64_t bitsOf(Type t) {
    return isa<tt::PointerType>(t) ? 64 : t.getIntOrFloatBitWidth();
  }

  std::string poolRegion(int64_t byteOffset, StringRef sc) {
    std::string base = byteOffset == 0
                           ? poolBuf
                           : "(" + poolBuf + " + " + std::to_string(byteOffset) +
                                 ")";
    return "((threadgroup " + sc.str() + "*)" + base + ")";
  }

  // Full flat size (product of shape) of a tensor tile.
  int64_t tileSize(RankedTensorType rt) {
    int64_t n = 1;
    for (int64_t d : rt.getShape())
      n *= d;
    return n;
  }

  // Peak byte footprint of a single transient scratch site.
  void scanPool(Operation *op) {
    if (auto c = dyn_cast<ttg::ConvertLayoutOp>(op)) {
      auto st = cast<RankedTensorType>(c.getSrc().getType());
      Type e = st.getElementType();
      int64_t bytes = tileSize(st) * (bitsOf(e) / 8);
      poolBytes = std::max(poolBytes, bytes);
    } else if (auto t = dyn_cast<tt::TransOp>(op)) {
      auto st = cast<RankedTensorType>(t.getSrc().getType());
      Type e = st.getElementType();
      poolBytes = std::max(poolBytes, tileSize(st) * (bitsOf(e) / 8));
    } else if (auto c = dyn_cast<tt::CatOp>(op)) {
      auto rt = cast<RankedTensorType>(c.getResult().getType());
      Type e = rt.getElementType();
      poolBytes = std::max(poolBytes, tileSize(rt) * (bitsOf(e) / 8));
    } else if (auto rs = dyn_cast<tt::ReshapeOp>(op)) {
      auto rt = cast<RankedTensorType>(rs.getResult().getType());
      Type e = rt.getElementType();
      poolBytes = std::max(poolBytes, tileSize(rt) * (bitsOf(e) / 8));
    } else if (auto d = dyn_cast<tt::DotOp>(op)) {
      auto aTy = cast<RankedTensorType>(d.getA().getType());
      auto bTy = cast<RankedTensorType>(d.getB().getType());
      auto cTy = cast<RankedTensorType>(d.getResult().getType());
      int64_t aB = tileSize(aTy) * (bitsOf(aTy.getElementType()) / 8);
      int64_t ab = aB + tileSize(bTy) * (bitsOf(bTy.getElementType()) / 8);
      int64_t cBytes = tileSize(cTy) * (bitsOf(cTy.getElementType()) / 8);
      poolBytes = std::max(poolBytes, std::max(ab, cBytes));
    } else if (auto r = dyn_cast<tt::ReduceOp>(op)) {
      auto st = cast<RankedTensorType>(r.getOperand(0).getType());
      tt::LinearLayout ll = ttg::toLinearLayout(st);
      auto kWarp = StringAttr::get(op->getContext(), "warp");
      if (ll.hasInDim(kWarp)) {
        int64_t nw = ll.getInDimSize(kWarp);
        int64_t nGroups = 1;
        if (auto resT = dyn_cast<RankedTensorType>(r.getResult()[0].getType()))
          nGroups = ttg::toLinearLayout(resT).getInDimSize(
              StringAttr::get(op->getContext(), "register"));
        int64_t bytes = 0;
        for (Value res : r.getResult())
          bytes += nw * 32 * (bitsOf(elementScalarType(res.getType())) / 8);
        poolBytes = std::max(poolBytes, nGroups * bytes);
      }
    } else if (auto s = dyn_cast<tt::ScanOp>(op)) {
      auto st = cast<RankedTensorType>(s.getOperand(0).getType());
      tt::LinearLayout ll = ttg::toLinearLayout(st);
      auto kWarp = StringAttr::get(op->getContext(), "warp");
      auto outDim = *ll.getOutDimNames().begin();
      if (!axisBits(ll, kWarp, outDim).empty()) {
        int64_t nw = ll.getInDimSize(kWarp);
        Type e = st.getElementType();
        poolBytes = std::max(poolBytes, nw * (bitsOf(e) / 8));
      }
    }
    for (Region &reg : op->getRegions())
      for (Block &blk : reg)
        for (Operation &o : blk)
          scanPool(&o);
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

  // Row-major flat offset of source register r, with its out-dim coordinates
  // permuted by `perm` and strides taken from `resShape` (the transposed
  // shape). Places the source element at its transposed logical position.
  std::string transFlatOffset(RankedTensorType srcTy, ArrayRef<int32_t> perm,
                              ArrayRef<int64_t> resShape, int reg) {
    tt::LinearLayout ll = ttg::toLinearLayout(srcTy);
    auto outNames = llvm::to_vector(ll.getOutDimNames());
    int rank = outNames.size();
    std::string expr;
    int64_t stride = 1;
    for (int d = rank - 1; d >= 0; --d) {
      std::string c = layoutCoordExpr(srcTy, reg, outNames[perm[d]]);
      std::string term = stride == 1 ? c : ("(" + c + " * " +
                                            std::to_string(stride) + ")");
      expr = expr.empty() ? term : ("(" + expr + " + " + term + ")");
      stride *= resShape[d];
    }
    return expr.empty() ? "0" : expr;
  }

  LogicalResult emitTrans(tt::TransOp op) {
    Value src = op.getSrc();
    Value res = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    auto perm = op.getOrder();
    std::string sc = mslScalarType(resTy.getElementType());
    auto &srcNames = names(src);

    std::string buf = fresh();
    os << ind() << "threadgroup " << sc << "* " << buf << " = "
       << poolRegion(0, sc) << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    for (int r = 0, n = regCount(src); r < n; ++r)
      os << ind() << buf << "["
         << transFlatOffset(srcTy, perm, resTy.getShape(), r)
         << "] = " << srcNames[r] << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    SmallVector<std::string> ids;
    for (int r = 0, n = regCount(res); r < n; ++r) {
      std::string id = fresh();
      os << ind() << sc << " " << id << " = " << buf << "["
         << flatTileOffset(resTy, r) << "];\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  // tt.reshape preserves row-major linear order. Elements can move across
  // threads, so we round-trip through a full-tile threadgroup buffer keyed by
  // the row-major flat offset both layouts agree on.
  LogicalResult emitReshape(tt::ReshapeOp op) {
    Value src = op.getSrc();
    Value res = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    std::string sc = mslScalarType(resTy.getElementType());
    auto &srcNames = names(src);
    int srcRc = regCount(src);
    int resRc = regCount(res);

    std::string buf = fresh();
    os << ind() << "threadgroup " << sc << "* " << buf << " = "
       << poolRegion(0, sc) << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    for (int r = 0; r < srcRc; ++r)
      os << ind() << buf << "[" << flatTileOffset(srcTy, r)
         << "] = " << srcNames[r] << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    SmallVector<std::string> outs;
    for (int r = 0; r < resRc; ++r) {
      std::string id = fresh();
      os << ind() << sc << " " << id << " = " << buf << "["
         << flatTileOffset(resTy, r) << "];\n";
      outs.push_back(id);
    }
    valMap[res] = outs;
    return success();
  }

  // tt.cat concatenates two same-layout operands along the leading dim. The
  // result layout redistributes elements across threads, so it needs a full
  // threadgroup round-trip: both halves write to their concatenated logical
  // offset (the second half shifted past the first's flat size), then every
  // thread reads back its result registers.
  LogicalResult emitCat(tt::CatOp op) {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value res = op.getResult();
    auto lhsTy = cast<RankedTensorType>(lhs.getType());
    auto rhsTy = cast<RankedTensorType>(rhs.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    std::string sc = mslScalarType(resTy.getElementType());
    auto &lhsNames = names(lhs);
    auto &rhsNames = names(rhs);
    int64_t lhsFlat = tileSize(lhsTy);

    std::string buf = fresh();
    os << ind() << "threadgroup " << sc << "* " << buf << " = "
       << poolRegion(0, sc) << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    for (int r = 0, n = regCount(lhs); r < n; ++r)
      os << ind() << buf << "[" << flatTileOffset(lhsTy, r)
         << "] = " << lhsNames[r] << ";\n";
    for (int r = 0, n = regCount(rhs); r < n; ++r)
      os << ind() << buf << "[" << flatTileOffset(rhsTy, r) << " + " << lhsFlat
         << "] = " << rhsNames[r] << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    SmallVector<std::string> ids;
    for (int r = 0, n = regCount(res); r < n; ++r) {
      std::string id = fresh();
      os << ind() << sc << " " << id << " = " << buf << "["
         << flatTileOffset(resTy, r) << "];\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
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

    std::string bufptr = fresh();
    os << ind() << "threadgroup " << sc << "* " << bufptr << " = "
       << poolRegion(0, sc) << ";\n";
    std::string buf = bufptr;
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

  // Pipelined (software-pipeliner) ops lower to SYNCHRONOUS threadgroup
  // staging: MSL has no async copy and M3+ dropped the DMA hardware, so the
  // rotating multi-buffer collapses to a plain threadgroup buffer written by a
  // masked per-thread copy and read back with a barrier between.
  static int64_t memdescFlatSize(ttg::MemDescType mt) {
    int64_t n = 1;
    for (int64_t d : mt.getShape())
      n *= d;
    return n;
  }

  LogicalResult emitLocalAlloc(ttg::LocalAllocOp op) {
    auto mt = cast<ttg::MemDescType>(op.getResult().getType());
    std::string sc = mslScalarType(mt.getElementType());
    std::string buf = "__tg_buf_" + std::to_string(tgScratchId++);
    os << ind() << "threadgroup " << sc << " " << buf << "["
       << memdescFlatSize(mt) << "];\n";
    memdescMap[op.getResult()] = {buf, "0"};
    return success();
  }

  LogicalResult emitMemDescIndex(ttg::MemDescIndexOp op) {
    auto srcMt = cast<ttg::MemDescType>(op.getSrc().getType());
    auto resMt = cast<ttg::MemDescType>(op.getResult().getType());
    MemDescInfo parent = memdescMap[op.getSrc()];
    int64_t sliceSize = memdescFlatSize(resMt);
    (void)srcMt;
    const std::string &idx = names(op.getIndex())[0];
    std::string base = parent.baseOffset == "0"
                           ? ("(" + idx + " * " + std::to_string(sliceSize) +
                              ")")
                           : ("(" + parent.baseOffset + " + " + idx + " * " +
                              std::to_string(sliceSize) + ")");
    memdescMap[op.getResult()] = {parent.buf, base};
    return success();
  }

  std::string memdescElemAddr(const MemDescInfo &info, RankedTensorType tileTy,
                              int reg) {
    std::string off = flatTileOffset(tileTy, reg);
    if (info.baseOffset == "0")
      return off;
    return "(" + info.baseOffset + " + " + off + ")";
  }

  LogicalResult emitAsyncCopy(ttg::AsyncCopyGlobalToLocalOp op) {
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    MemDescInfo dst = memdescMap[op.getResult()];
    auto &ptrs = names(op.getSrc());
    bool hasMask = op.getMask() != nullptr;
    SmallVector<std::string> *mask =
        hasMask ? &names(op.getMask()) : nullptr;
    for (int r = 0, n = regCount(op.getSrc()); r < n; ++r) {
      std::string addr = dst.buf + "[" + memdescElemAddr(dst, srcTy, r) + "]";
      std::string load = "*" + ptrs[r];
      if (hasMask) {
        const std::string &m = (*mask)[mask->size() == 1 ? 0 : r];
        os << ind() << "if (" << m << ") " << addr << " = " << load << ";\n";
      } else {
        os << ind() << addr << " = " << load << ";\n";
      }
    }
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    valMap[op.getResult()] = SmallVector<std::string>{};
    return success();
  }

  LogicalResult emitLocalStore(ttg::LocalStoreOp op) {
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    MemDescInfo dst = memdescMap[op.getDst()];
    auto &vals = names(op.getSrc());
    for (int r = 0, n = regCount(op.getSrc()); r < n; ++r)
      os << ind() << dst.buf << "[" << memdescElemAddr(dst, srcTy, r)
         << "] = " << vals[r] << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    return success();
  }

  LogicalResult emitLocalLoad(ttg::LocalLoadOp op) {
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    MemDescInfo src = memdescMap[op.getSrc()];
    std::string sc = mslScalarType(resTy.getElementType());
    SmallVector<std::string> ids;
    for (int r = 0, n = regCount(op.getResult()); r < n; ++r) {
      std::string id = fresh();
      os << ind() << sc << " " << id << " = " << src.buf << "["
         << memdescElemAddr(src, resTy, r) << "];\n";
      ids.push_back(id);
    }
    valMap[op.getResult()] = ids;
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
    if (isa<IntegerType>(aElem) || isa<IntegerType>(bElem) ||
        isa<IntegerType>(cElem))
      return emitDotScalar(op);
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

    int64_t aBytes = M * K * (bitsOf(aElem) / 8);
    std::string tgA = fresh(), tgB = fresh(), tgC = fresh();
    os << ind() << "threadgroup " << opScalar << "* " << tgA << " = "
       << poolRegion(0, opScalar) << ";\n";
    os << ind() << "threadgroup " << opScalar << "* " << tgB << " = "
       << poolRegion(aBytes, opScalar) << ";\n";
    os << ind() << "threadgroup " << accScalar << "* " << tgC << " = "
       << poolRegion(0, accScalar) << ";\n";

    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    for (int r = 0, n = regCount(op.getA()); r < n; ++r)
      os << ind() << tgA << "[" << flatTileOffset(aTy, r) << "] = " << aNames[r]
         << ";\n";
    for (int r = 0, n = regCount(op.getB()); r < n; ++r)
      os << ind() << tgB << "[" << flatTileOffset(bTy, r) << "] = " << bNames[r]
         << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

    int64_t mT = M / 8, nT = N / 8, kT = K / 8;
    tt::LinearLayout cLL = ttg::toLinearLayout(cTy);
    auto kWarpDim = StringAttr::get(op.getContext(), "warp");
    int64_t numWarps = cLL.hasInDim(kWarpDim) ? cLL.getInDimSize(kWarpDim) : 1;
    int64_t nFrag = mT * nT;
    if (numWarps > nFrag)
      numWarps = nFrag;

    // Distribute the mTxnT output fragments round-robin across simdgroups: warp
    // w owns fragments f with f % numWarps == w. Accumulators are declared for
    // every fragment so a warp's owned registers stay live across the uniform
    // barrier that separates the A/B-consuming compute phase from the C store
    // phase (the store aliases the A region in the pool, so all A reads must
    // finish first). The C readback below is layout-driven and warp-agnostic.
    for (int64_t f = 0; f < nFrag; ++f) {
      int64_t mi = f / nT, ni = f % nT;
      std::string acc = "acc_" + std::to_string(mi) + "_" + std::to_string(ni);
      os << ind() << accFrag << " " << acc << " = " << accFrag << "(0.0f);\n";
    }
    for (int64_t w = 0; w < numWarps; ++w) {
      os << ind() << "if (" << warpId << " == " << w << ") {\n";
      ++indent;
      for (int64_t f = w; f < nFrag; f += numWarps) {
        int64_t mi = f / nT, ni = f % nT;
        std::string acc = "acc_" + std::to_string(mi) + "_" + std::to_string(ni);
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
      }
      --indent;
      os << ind() << "}\n";
    }
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    for (int64_t w = 0; w < numWarps; ++w) {
      os << ind() << "if (" << warpId << " == " << w << ") {\n";
      ++indent;
      for (int64_t f = w; f < nFrag; f += numWarps) {
        int64_t mi = f / nT, ni = f % nT;
        std::string acc = "acc_" + std::to_string(mi) + "_" + std::to_string(ni);
        os << ind() << "simdgroup_store(" << acc << ", " << tgC << " + "
           << (mi * 8 * N + ni * 8) << ", " << N << ");\n";
      }
      --indent;
      os << ind() << "}\n";
    }
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

  // Non-simdgroup tt.dot for integer operands: simdgroup_matrix has no integer
  // path. A/B tiles are staged row-major into threadgroup memory, then each
  // thread computes its owned C output registers with a scalar K-loop
  // accumulating in the (wider) result element type. Correctness over speed.
  LogicalResult emitDotScalar(tt::DotOp op) {
    auto aTy = cast<RankedTensorType>(op.getA().getType());
    auto bTy = cast<RankedTensorType>(op.getB().getType());
    auto cTy = cast<RankedTensorType>(op.getResult().getType());
    Type aElem = aTy.getElementType();
    Type bElem = bTy.getElementType();
    Type cElem = cTy.getElementType();
    if (aTy.getRank() != 2) {
      op.emitError("EmitMSL: scalar tt.dot requires 2-D operands");
      return failure();
    }
    int64_t K = aTy.getShape()[1];
    int64_t M = cTy.getShape()[0];
    int64_t N = cTy.getShape()[1];

    std::string aScalar = mslScalarType(aElem);
    std::string bScalar = mslScalarType(bElem);
    std::string accScalar = mslScalarType(cElem);

    auto &aNames = names(op.getA());
    auto &bNames = names(op.getB());
    auto &cInit = names(op.getC());

    int64_t aBytes = M * K * (bitsOf(aElem) / 8);
    std::string tgA = fresh(), tgB = fresh();
    os << ind() << "threadgroup " << aScalar << "* " << tgA << " = "
       << poolRegion(0, aScalar) << ";\n";
    os << ind() << "threadgroup " << bScalar << "* " << tgB << " = "
       << poolRegion(aBytes, bScalar) << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    for (int r = 0, n = regCount(op.getA()); r < n; ++r)
      os << ind() << tgA << "[" << flatTileOffset(aTy, r) << "] = " << aNames[r]
         << ";\n";
    for (int r = 0, n = regCount(op.getB()); r < n; ++r)
      os << ind() << tgB << "[" << flatTileOffset(bTy, r) << "] = " << bNames[r]
         << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

    tt::LinearLayout cLL = ttg::toLinearLayout(cTy);
    auto outNames = llvm::to_vector(cLL.getOutDimNames());
    StringAttr d0 = outNames[0], d1 = outNames[1];
    SmallVector<std::string> ids;
    for (int r = 0, n = regCount(op.getResult()); r < n; ++r) {
      std::string mExpr = layoutCoordExpr(cTy, r, d0);
      std::string nExpr = layoutCoordExpr(cTy, r, d1);
      std::string mrow = fresh(), ncol = fresh(), acc = fresh();
      os << ind() << "int " << mrow << " = " << mExpr << ";\n";
      os << ind() << "int " << ncol << " = " << nExpr << ";\n";
      std::string base = cInit[cInit.size() == 1 ? 0 : r];
      os << ind() << accScalar << " " << acc << " = " << base << ";\n";
      std::string kv = fresh();
      os << ind() << "for (int " << kv << " = 0; " << kv << " < " << K << "; ++"
         << kv << ") {\n";
      ++indent;
      os << ind() << acc << " += (" << accScalar << ")" << tgA << "[" << mrow
         << " * " << K << " + " << kv << "] * (" << accScalar << ")" << tgB
         << "[" << kv << " * " << N << " + " << ncol << "];\n";
      --indent;
      os << ind() << "}\n";
      ids.push_back(acc);
    }
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
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

  LogicalResult emitAtomicCAS(tt::AtomicCASOp op) {
    Value res = op.getResult();
    Type scalarTy = elementScalarType(res.getType());
    std::string sc = mslScalarType(scalarTy);
    if (isa<FloatType>(scalarTy) || scalarTy.getIntOrFloatBitWidth() != 32) {
      op.emitError("EmitMSL: only 32-bit integer atomic_cas supported");
      return failure();
    }
    std::string atomicTy = "atomic_int";

    auto &ptrs = names(op.getPtr());
    auto &cmps = names(op.getCmp());
    auto &vals = names(op.getVal());
    bool uniform = !isa<RankedTensorType>(op.getPtr().getType());
    int rc = ptrs.size();
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      const std::string &p = ptrs[r];
      const std::string &c = cmps[cmps.size() == 1 ? 0 : r];
      const std::string &v = vals[vals.size() == 1 ? 0 : r];
      std::string exp = fresh();
      std::string id = fresh();
      os << ind() << sc << " " << exp << " = " << c << ";\n";
      std::string call = "atomic_compare_exchange_weak_explicit((device " +
                         atomicTy + "*)" + p + ", &" + exp + ", " + v +
                         ", memory_order_relaxed, memory_order_relaxed)";
      std::string body = "while (" + exp + " == " + c + " && !(" + call +
                         ")) {}";
      if (uniform)
        os << ind() << "if (" << tidId << ".x == 0) " << body << "\n";
      else
        os << ind() << body << "\n";
      os << ind() << sc << " " << id << " = " << exp << ";\n";
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
