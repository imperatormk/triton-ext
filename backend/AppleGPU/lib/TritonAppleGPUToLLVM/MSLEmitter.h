// MSLEmitter.h - TritonGPU IR -> MSL lowering state (the emitter class).
//
// Extracted from EmitMSL.cpp. Method definitions live in EmitMSL.cpp; the
// AST type helpers (astScalarType/...) live in MSLTypes.cpp. See
// MSL_AST_DESIGN.md for the frozen AST contract.
#ifndef MSL_EMITTER_H
#define MSL_EMITTER_H

#include "MSLAst.h"
#include "MSLConstants.h"
#include "MSLPrinter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <string>

namespace mlir::triton::applegpu {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

inline std::string mslScalarType(Type t) {
  if (t.isF32() || t.isF64())
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

inline std::string mslUnsignedType(Type t) {
  if (auto it = dyn_cast<IntegerType>(t)) {
    switch (it.getWidth()) {
    case 8:
      return "uchar";
    case 16:
      return "ushort";
    case 32:
      return "uint";
    case 64:
      return "ulong";
    default:
      break;
    }
  }
  return "";
}

inline std::string mslKernelName(StringRef name) {
  if (name.starts_with("triton_"))
    return name.str();
  return "triton_" + name.str();
}

inline std::string mslDeviceFuncName(StringRef name) {
  std::string out = "fn_";
  for (char c : name)
    out += (isalnum((unsigned char)c) || c == '_') ? c : '_';
  return out;
}

inline std::string mslStorageType(Type t) {
  if (auto rt = dyn_cast<RankedTensorType>(t))
    t = rt.getElementType();
  if (auto pt = dyn_cast<tt::PointerType>(t))
    return "device " + mslScalarType(pt.getPointeeType()) + "*";
  return mslScalarType(t);
}

inline std::string barrierMemFlags(ttg::AddrSpace addrSpace) {
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

inline Type elementScalarType(Type t) {
  if (auto rt = dyn_cast<RankedTensorType>(t))
    t = rt.getElementType();
  if (auto pt = dyn_cast<tt::PointerType>(t))
    t = pt.getPointeeType();
  return t;
}

// Forwarding stream: `os` writes here, and the sink can be retargeted so a
// still-string op family can be captured into a buffer (hybrid RawStmt path).
class ForwardOStream : public llvm::raw_ostream {
public:
  explicit ForwardOStream(llvm::raw_ostream *sink) : sink(sink) {
    SetUnbuffered();
  }
  ~ForwardOStream() override { flush(); }
  llvm::raw_ostream *swap(llvm::raw_ostream *s) {
    flush();
    llvm::raw_ostream *prev = sink;
    sink = s;
    return prev;
  }

private:
  void write_impl(const char *ptr, size_t size) override {
    sink->write(ptr, size);
  }
  uint64_t current_pos() const override { return sink->tell(); }
  llvm::raw_ostream *sink;
};

// Per-value symbol table: maps an SSA Value to its MSL identifier. For tensor
// values we store one identifier per register (per-thread element).
class MSLEmitter {
public:
  MSLEmitter(ModuleOp mod, raw_ostream &realOut)
      : mod(mod), fwd(&realOut), os(fwd) {}

  LogicalResult emit() {
    os << "#include <metal_stdlib>\n";
    os << "#include <metal_simdgroup_matrix>\n";
    os << "using namespace metal;\n\n";
    os << "static inline float tt_erf(float x){\n"
          "  float t = 1.0f/(1.0f+0.5f*metal::fabs(x));\n"
          "  float y = t*metal::exp(-x*x-1.26551223f+t*(1.00002368f+t*(0.37409196f"
          "+t*(0.09678418f+t*(-0.18628806f+t*(0.27886807f+t*(-1.13520398f"
          "+t*(1.48851587f+t*(-0.82215223f+t*0.17087277f)))))))));\n"
          "  float r = 1.0f - y;\n"
          "  return x >= 0.0f ? r : -r;\n"
          "}\n\n";

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
        liveTgBytes += memdescFlatSize(mt) *
                       (bitsOf(mt.getElementType()) / 8);
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

private:
  ModuleOp mod;
  ForwardOStream fwd;
  raw_ostream &os;

  // Capture the string output of `fn` (which writes to `os`) into a verbatim
  // RawStmt. Transitional: lets a not-yet-flipped op family print byte-identical
  // while the surrounding function is AST-driven. Removed once all families flip.
  template <typename Fn> msl::Stmt *captureRaw(Fn &&fn) {
    std::string buf;
    llvm::raw_string_ostream ss(buf);
    llvm::raw_ostream *prev = fwd.swap(&ss);
    fn();
    ss.flush();
    fwd.swap(prev);
    return ctx.rawVerbatim(buf);
  }
  // AST arena. Unused by emission yet; later layers build nodes into it and the
  // printer renders them, replacing the string emission below.
  msl::MSLContext ctx;

  // AST-typed forms of the string type helpers (mslScalarType/mslUnsignedType/
  // mslStorageType). Definitions in MSLTypes.cpp. Not yet used by emission.
  msl::Type *astScalarType(Type t);
  msl::Type *astUnsignedType(Type t);
  msl::Type *astStorageType(Type t);

  // Per-element value builders for the reshape/aliasing ops. makeRange yields
  // `start + off`; splat/reshape/join/split just alias a source register name.
  msl::Expr *astMakeRangeElem(int start, StringRef off);
  msl::Expr *astAliasElem(StringRef name);

  int nextId = 0;
  int indent = 1;
  llvm::DenseMap<Value, SmallVector<std::string>> valMap;

  struct MemDescInfo {
    std::string buf;
    std::string baseOffset;
    SmallVector<int64_t> bufStrides;
  };
  llvm::DenseMap<Value, MemDescInfo> memdescMap;

  // AST siblings of the offset/address string helpers (defs in
  // EmitMSLMemory.cpp). Not yet driving emission; mirror the string parens.
  msl::Expr *astLayoutCoordExpr(RankedTensorType rt, int reg, StringAttr outDim);
  msl::Expr *astLayoutOffsetExpr(RankedTensorType rt, int reg);
  msl::Expr *astPoolRegion(int64_t byteOffset, StringRef sc);
  msl::Expr *astFlatTileOffset(RankedTensorType rt, int reg);
  msl::Expr *astSliceFlatOffset(RankedTensorType rt, int reg);
  msl::Expr *astBatchCoordExpr(RankedTensorType rt, int reg);
  msl::Expr *astTransFlatOffset(RankedTensorType srcTy, ArrayRef<int32_t> perm,
                                ArrayRef<int64_t> resShape, int reg);
  msl::Expr *astMemdescElemAddr(const MemDescInfo &info, RankedTensorType tileTy,
                                int reg);

  // AST siblings of the dot/GEMM simdgroup-matrix emission (defs in
  // EmitMSLDot.cpp). Not driving emission this layer; mirror the string parens.
  msl::MatrixType *astSgFragType(Type t);
  msl::Expr *astFragAddr(StringRef base, int64_t off);
  msl::Stmt *astFragDecl(msl::MatrixType *frag, StringRef name);
  msl::Stmt *astAccFragDecl(msl::MatrixType *frag, StringRef name);
  msl::Stmt *astSgLoad(StringRef frag, StringRef base, int64_t off, int64_t ld);
  msl::Stmt *astSgStore(StringRef acc, StringRef base, int64_t off, int64_t ld);
  msl::Stmt *astSgMultiplyAccumulate(StringRef acc, StringRef a, StringRef b);
  msl::Expr *astReadbackValue(StringRef buf, msl::Expr *off, StringRef base);

  // fp-narrowing siblings. The bit-twiddling body is the design-sanctioned Raw
  // escape hatch; only the outer `sc h = as_type<sc>(bits);` decl is real nodes.
  msl::Stmt *astRoundedHalfValueFull(const std::string &sc, const std::string &v,
                                     std::string &outName);
  msl::Stmt *astTruncatedFloatValue(const std::string &sc, const std::string &v,
                                    std::string &outName);
  msl::Stmt *astRoundedHalfValue(const std::string &sc, const std::string &v,
                                 std::string &outName);

  // atomic siblings (EmitMSLAtomic.cpp). Multi-statement builders return a
  // msl::Block so the caller splices them at its own indent (no stray braces);
  // single-node builders return Stmt*/Expr*.
  msl::Expr *astInit0(StringRef sc);
  msl::Expr *astDeviceAtomicPtr(StringRef atomicTy, StringRef p);
  msl::Block astPacked16Base(StringRef p, std::string &wordPtrOut,
                             std::string &isHighOut);
  msl::Expr *astAtomicRmwCall(StringRef fn, StringRef atomicTy, StringRef p,
                              StringRef v, StringRef order, bool memFlags);
  msl::Block astFloat32CASLoop(StringRef p, StringRef curId,
                               msl::Expr *newFloatExpr, StringRef id);
  msl::Block astPacked16CASLoop(StringRef wordPtr, StringRef isHigh,
                                StringRef sc, StringRef curId,
                                msl::Expr *newHalfExpr, StringRef id);
  msl::Block astInt32CAS(StringRef p, StringRef c, StringRef v, StringRef sc,
                         StringRef id, bool declare);
  msl::Block astFloat32CAS(StringRef p, StringRef c, StringRef v, StringRef id,
                           bool declare);
  msl::Block astPacked16CAS(StringRef wordPtr, StringRef isHigh, StringRef c,
                            StringRef v, StringRef sc, StringRef id,
                            bool declare);
  msl::Stmt *astAcquireFence();
  msl::Stmt *astPollSpin(msl::Expr *loadExpr, StringRef want);
  msl::Expr *astPoll64Load(StringRef p, StringRef wordPtr, msl::Stmt *&out);
  msl::Stmt *astHistBinsDecl(StringRef bins);
  msl::Stmt *astHistZeroInit(StringRef bins, StringRef zi, int64_t nBins,
                             int64_t threads);
  msl::Stmt *astHistFetchAdd(msl::Expr *guard, StringRef bins, StringRef v);

  // reduce / scan / map siblings (EmitMSLReduce.cpp). Multi-statement builders
  // return a msl::Block spliced at caller indent; single-node builders return
  // Stmt*/Expr*. The cross-lane shuffles go through astShuffleExpr with the
  // builtin::simd names; control flow is real If/State-machine node trees.
  msl::Stmt *astCombineAssign(StringRef dst, StringRef res);
  msl::Stmt *astReduceAccInit(StringRef sc, StringRef acc, StringRef src);
  msl::Stmt *astReduceAccStep(StringRef acc, StringRef out);
  msl::Expr *astReduceLaneShuffle(StringRef sc, StringRef acc, unsigned m);
  msl::Stmt *astReduceScratchDecl(StringRef sc, StringRef scratch,
                                  int64_t byteOff);
  msl::Stmt *astReduceScratchStore(StringRef scratch, StringRef acc);
  msl::Expr *astReduceWarpBase(unsigned warpMask);
  msl::Stmt *astReduceWarpLoad(StringRef sc, StringRef dst, StringRef scratch,
                               msl::Expr *base, int off);
  msl::Stmt *astScanAccInit(StringRef sc, StringRef acc, StringRef src);
  msl::Stmt *astScanLaneSeed(StringRef sc, StringRef laneScan, StringRef acc);
  msl::Expr *astScanLaneShuffle(bool rev, StringRef sc, StringRef val,
                                unsigned delta);
  msl::Expr *astScanLaneGuard(bool rev, unsigned axisLaneMask, unsigned delta);
  msl::Stmt *astScanGuardedUpdate(StringRef dst, msl::Expr *guard,
                                  StringRef out);
  msl::Expr *astScanAxisTopLane(bool rev, unsigned axisLaneMask);
  msl::Stmt *astWarpCarryLaneOnly(StringRef sc, StringRef runTotal,
                                  StringRef laneScan, StringRef axisTopLane);
  msl::Expr *astWarpCarryTopGuard(StringRef axisTopLane);
  msl::Stmt *astWarpCarryTopStore(msl::Expr *topGuard, StringRef scratch,
                                  StringRef laneScan);
  msl::Expr *astWarpCarryBase(unsigned axisWarpMask, StringRef axisTopLane);
  msl::Stmt *astWarpCarryMyPart(StringRef myPart, ArrayRef<int> maskBits);
  msl::Expr *astWarpCarryPartCond(bool rev, StringRef myPart, int p);
  msl::Stmt *astWarpCarryInitFlag(StringRef init);
  msl::Stmt *astWarpCarryApply(StringRef acc, StringRef init, StringRef out);
  msl::Stmt *astMapCaptureDecl(StringRef sc, StringRef name);
  msl::Stmt *astMapReturnSpill(StringRef capture, StringRef operand);
  msl::Stmt *astMapHoistCopy(StringRef dst, StringRef src);
  msl::Stmt *astMapCFGStateMachine(
      StringRef state,
      ArrayRef<std::pair<std::string, msl::Block>> cases);

  // function / scope / control-flow sibling builders (EmitMSLFunc.cpp). Each
  // takes a pre-built body Block and returns the scope node; they never emit,
  // so the flip layer (7b) walks the ops into the body then wraps it here.
  msl::KernelFn *astKernelFn(tt::FuncOp func, msl::Block body);
  msl::DeviceFn *astDeviceFn(tt::FuncOp func, msl::Block body);
  msl::DeviceFn *astDeviceProto(tt::FuncOp func);
  msl::NamedType *astRetStructType(tt::FuncOp func);
  msl::Stmt *astRetStructDecl(tt::FuncOp func); // ArrayDecl-less; RawStmt struct
  msl::Type *astDeviceRetType(tt::FuncOp func);
  msl::Stmt *astReturn(tt::ReturnOp op);
  msl::ForScope *astForScope(scf::ForOp op, msl::Block body, StringRef iv,
                             StringRef ivTy);
  msl::TripCountForScope *astTripCountForScope(scf::ForOp op, msl::Block body,
                                               StringRef counter, StringRef iv,
                                               StringRef ivTy);
  msl::Stmt *astForNode(scf::ForOp op, msl::Block body, StringRef iv,
                        StringRef tc, StringRef ivTy, bool wideIv);
  msl::IfScope *astIfScope(scf::IfOp op, msl::Block thenB, msl::Block elseB);
  msl::WhileScope *astWhileScope(scf::WhileOp op, msl::Block body);
  msl::StateMachineScope *astBlockCFG(
      Region &region, StringRef state,
      ArrayRef<std::pair<std::string, msl::Block>> cases);
  msl::Block astBranchEdge(Block *succ, Operation::operand_range args,
                           StringRef state);
  msl::Stmt *astCondBranch(Value cond, msl::Block thenB, msl::Block elseB);
  msl::Block astYieldAssign(Operation *term,
                            ArrayRef<SmallVector<std::string>> dsts);
  // Helpers below mint fresh names over a raw id counter passed by reference
  // (seeded from nextId by the caller) so they never mutate nextId directly.
  llvm::SmallVector<msl::Stmt *, 2> laneWarpDecls(int &id, StringRef tid);
  llvm::SmallVector<msl::Param, 8> deviceParams(tt::FuncOp func, int &id,
                                                bool bind);
  bool astElemwiseDecls(Operation *op, msl::Type *declTy, int &id,
                        msl::Block &body,
                        llvm::function_ref<msl::Expr *(int)> mk);
  bool astDeclBind(Operation *op, msl::Type *declTy, msl::Block &body,
                   llvm::function_ref<msl::Expr *(int)> mk);
  // Dispatch spine: appends the sibling nodes for `op` to `body`. Returns true
  // when the op is wired (nodes appended, or nothing for alias/dataless ops);
  // false when the op has no whole-op sibling yet (flip layer 7b fills these).
  bool astEmitOp(Operation *op, msl::Block &body);
  msl::Block astWalkBlock(Block &blk, unsigned depth);
  SmallVector<std::string> astDeclResultVars(Value v, msl::Block &body);
  msl::Expr *astDerefPtr(Value ptr, StringRef name, StringRef scName);
  void astStoreBody(tt::StoreOp op, msl::Block &body);
  bool emitFailed = false;

  std::string fresh() { return "v" + std::to_string(nextId++); }

  std::string ind() const { return std::string(indent * 4, ' '); }

  // Barrier peephole: a barrier is held pending rather than written immediately,
  // so two barriers emitted back-to-back with no intervening statement collapse
  // into one (keeping the stronger memory scope). flushBarrier() must run before
  // any non-barrier output and at every scope boundary so a pending barrier is
  // never lost or reordered past a real memory operation.
  bool barrierPending = false;
  bool barrierPendingDevice = false;

  void emitBarrier(bool device) {
    barrierPending = true;
    barrierPendingDevice = barrierPendingDevice || device;
  }

  void flushBarrier() {
    if (!barrierPending)
      return;
    barrierPending = false;
    bool device = barrierPendingDevice;
    barrierPendingDevice = false;
    if (device)
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup | "
                     "mem_flags::mem_device);\n";
    else
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  }

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
    // Pin the threadgroup size the runtime always dispatches (num_warps*32).
    // Without it the Metal compiler picks an occupancy-driven ceiling that can
    // fall below the requested size (register/threadgroup pressure), so a valid
    // launch is rejected as OutOfResources; the attribute makes the compiler
    // budget for exactly this size (spilling if needed) instead.
    if (auto nw = mod->getAttrOfType<IntegerAttr>("ttg.num-warps")) {
      int64_t threads = nw.getInt() * 32;
      os << "[[max_total_threads_per_threadgroup(" << threads << ")]]\n";
    }
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
    numTgId = fresh();
    argLines.push_back("    uint3 " + tgposId +
                       " [[threadgroup_position_in_grid]]");
    argLines.push_back("    uint3 " + tidId +
                       " [[thread_position_in_threadgroup]]");
    argLines.push_back("    uint3 " + numTgId +
                       " [[threadgroups_per_grid]]");
    os << llvm::join(argLines, ",\n") << ") {\n";

    int off = 0;
    for (BlockArgument arg : scalarArgs) {
      Type ty = arg.getType();
      unsigned bits = ty.getIntOrFloatBitWidth();
      int size = bits == 1 ? 1 : (int)(bits / 8);
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

    scalarSpinlock = false;
    func.walk([&](tt::AtomicCASOp cas) {
      if (isa<RankedTensorType>(cas.getPtr().getType()))
        return;
      if (tracesToKernelArg(cas.getPtr()))
        scalarSpinlock = true;
    });

    poolBytes = 0;
    liveTgBytes = 0;
    func.walk([&](ttg::LocalAllocOp la) {
      auto mt = cast<ttg::MemDescType>(la.getResult().getType());
      liveTgBytes += memdescFlatSize(mt) * (bitsOf(mt.getElementType()) / 8);
    });
    for (Block &blk : func.getBody())
      for (Operation &op : blk)
        scanPool(&op);
    int64_t kernelPool = moduleHasDevFuncs ? globalPoolBytes : poolBytes;
    if (kernelPool > 0) {
      poolBuf = "__pool";
      os << ind() << "threadgroup char " << poolBuf << "[" << kernelPool
         << "];\n";
    }

    Region &region = func.getBody();
    if (region.hasOneBlock()) {
      msl::Block body = astWalkBlock(region.front(), indent);
      if (emitFailed)
        return failure();
      msl::MSLPrinter printer(os);
      printer.printBlockAt(body, indent);
    } else {
      if (failed(emitBlockCFG(region)))
        return failure();
      flushBarrier();
    }
    os << "}\n";
    return success();
  }

  llvm::DenseMap<Operation *, std::string> devRetStruct;

  // Multi-result callees return a small struct; declare it and remember the
  // name so callers can bind each field.
  // A single tensor result is returned as a struct of its per-thread registers,
  // reusing the same struct mechanism as a multi-scalar result. Callee and
  // caller share the tensor's layout (Triton inserts convert_layout at the
  // boundary), so register index i maps directly to caller register i.
  static bool isTensorResult(ArrayRef<Type> results) {
    return results.size() == 1 && isa<RankedTensorType>(results[0]);
  }

  LogicalResult declRetStruct(tt::FuncOp func) {
    auto results = func.getFunctionType().getResults();
    if (isTensorResult(results)) {
      std::string name = mslDeviceFuncName(func.getName()) + "_ret";
      devRetStruct[func] = name;
      std::string sc =
          mslScalarType(cast<RankedTensorType>(results[0]).getElementType());
      Value res = func.getBody().front().getTerminator()->getOperand(0);
      int rc = regCount(res);
      os << "struct " << name << " {\n";
      for (int i = 0; i < rc; ++i)
        os << "  " << sc << " f" << i << ";\n";
      os << "};\n";
      return success();
    }
    if (results.size() <= 1)
      return success();
    std::string name = mslDeviceFuncName(func.getName()) + "_ret";
    devRetStruct[func] = name;
    os << "struct " << name << " {\n";
    for (auto [i, ty] : llvm::enumerate(results)) {
      if (!isa<IntegerType, FloatType>(ty)) {
        func.emitError("EmitMSL: unsupported device function result type");
        return failure();
      }
      os << "  " << mslScalarType(ty) << " f" << i << ";\n";
    }
    os << "};\n";
    return success();
  }

  std::string deviceRetType(tt::FuncOp func) {
    auto results = func.getFunctionType().getResults();
    if (results.empty())
      return "void";
    if (isTensorResult(results))
      return devRetStruct[func];
    if (results.size() == 1)
      return mslScalarType(results[0]);
    return devRetStruct[func];
  }

  // Signature `RetType fn_name(args..., thread-context)`. Callee bodies may use
  // program-id / thread builtins, so the kernel's thread context is threaded
  // through every device call as trailing params.
  LogicalResult emitDeviceSignature(tt::FuncOp func, bool bindArgs) {
    auto fnTy = func.getFunctionType();
    os << deviceRetType(func) << " " << mslDeviceFuncName(func.getName())
       << "(";
    SmallVector<std::string> params;
    for (auto [i, argTy] : llvm::enumerate(fnTy.getInputs())) {
      std::string id = bindArgs ? fresh() : ("a" + std::to_string(i));
      if (auto pt = dyn_cast<tt::PointerType>(argTy)) {
        params.push_back("device " + mslScalarType(pt.getPointeeType()) + "* " +
                         id);
      } else if (isa<IntegerType, FloatType>(argTy)) {
        params.push_back(mslScalarType(argTy) + " " + id);
      } else {
        func.emitError("EmitMSL: unsupported device function argument type");
        return failure();
      }
      if (bindArgs)
        bindScalar(func.getArgument(i), id);
    }
    std::string tg = bindArgs ? fresh() : "__tgpos";
    std::string ti = bindArgs ? fresh() : "__tid";
    std::string nt = bindArgs ? fresh() : "__numtg";
    params.push_back("uint3 " + tg);
    params.push_back("uint3 " + ti);
    params.push_back("uint3 " + nt);
    std::string pp = bindArgs ? fresh() : "__poolptr";
    if (globalPoolBytes > 0)
      params.push_back("threadgroup char* " + pp);
    if (bindArgs) {
      tgposId = tg;
      tidId = ti;
      numTgId = nt;
      devPoolPtr = pp;
    }
    os << llvm::join(params, ", ") << ")";
    return success();
  }

  LogicalResult emitDeviceFuncProto(tt::FuncOp func, bool asDecl) {
    if (failed(declRetStruct(func)))
      return failure();
    if (failed(emitDeviceSignature(func, /*bindArgs=*/false)))
      return failure();
    os << ";\n";
    return success();
  }

  LogicalResult emitDeviceFunc(tt::FuncOp func) {
    if (failed(emitDeviceSignature(func, /*bindArgs=*/true)))
      return failure();
    os << " {\n";

    laneId = fresh();
    warpId = fresh();
    os << ind() << "int " << laneId << " = (int)(" << tidId << ".x & 31u);\n";
    os << ind() << "int " << warpId << " = (int)(" << tidId << ".x >> 5);\n";

    poolBuf = devPoolPtr;
    curDevFunc = func;
    Region &region = func.getBody();
    if (region.hasOneBlock()) {
      msl::Block body = astWalkBlock(region.front(), indent);
      if (emitFailed)
        return failure();
      msl::MSLPrinter printer(os);
      printer.printBlockAt(body, indent);
    } else {
      if (failed(emitBlockCFG(region)))
        return failure();
      flushBarrier();
    }
    curDevFunc = nullptr;
    os << "}\n";
    return success();
  }

  tt::FuncOp curDevFunc;
  std::string devPoolPtr;

  LogicalResult emitReturn(tt::ReturnOp op) {
    unsigned n = op.getNumOperands();
    if (n == 0) {
      os << ind() << "return;\n";
      return success();
    }
    if (n == 1 && !isa<RankedTensorType>(op.getOperand(0).getType())) {
      auto &nm = names(op.getOperand(0));
      os << ind() << "return " << nm[0] << ";\n";
      return success();
    }
    if (n == 1) {
      auto &nm = names(op.getOperand(0));
      os << ind() << "return { " << llvm::join(nm, ", ") << " };\n";
      return success();
    }
    std::string st = curDevFunc ? devRetStruct.lookup(curDevFunc) : "";
    os << ind() << "return { ";
    SmallVector<std::string> fields;
    for (Value v : op.getOperands())
      fields.push_back(names(v)[0]);
    os << llvm::join(fields, ", ") << " };\n";
    (void)st;
    return success();
  }

  LogicalResult emitCall(tt::CallOp op) {
    auto callee =
        mod.lookupSymbol<tt::FuncOp>(op.getCalleeAttr().getValue());
    if (!callee) {
      op.emitError("EmitMSL: call to unknown callee");
      return failure();
    }
    SmallVector<std::string> argExprs;
    for (Value operand : op.getOperands()) {
      auto &nm = names(operand);
      if (nm.size() != 1) {
        op.emitError("EmitMSL: tensor-valued call argument unsupported");
        return failure();
      }
      argExprs.push_back(nm[0]);
    }
    argExprs.push_back(tgposId);
    argExprs.push_back(tidId);
    argExprs.push_back(numTgId);
    if (globalPoolBytes > 0)
      argExprs.push_back(poolBuf.empty() ? "__pool" : poolBuf);
    std::string call = mslDeviceFuncName(callee.getName()) + "(" +
                       llvm::join(argExprs, ", ") + ")";

    unsigned nRes = op.getNumResults();
    if (nRes == 1 && isa<RankedTensorType>(op.getResult(0).getType())) {
      Value res = op.getResult(0);
      std::string sc =
          mslScalarType(cast<RankedTensorType>(res.getType()).getElementType());
      std::string tmp = fresh();
      os << ind() << deviceRetType(callee) << " " << tmp << " = " << call
         << ";\n";
      int rc = regCount(res);
      SmallVector<std::string> ids;
      for (int i = 0; i < rc; ++i) {
        std::string id = fresh();
        os << ind() << sc << " " << id << " = " << tmp << ".f" << i << ";\n";
        ids.push_back(id);
      }
      valMap[res] = ids;
      return success();
    }
    if (nRes == 0) {
      os << ind() << call << ";\n";
    } else if (nRes == 1) {
      if (!isa<IntegerType, FloatType>(op.getResult(0).getType())) {
        op.emitError("EmitMSL: tensor-valued call result unsupported");
        return failure();
      }
      std::string id = fresh();
      os << ind() << mslScalarType(op.getResult(0).getType()) << " " << id
         << " = " << call << ";\n";
      bindScalar(op.getResult(0), id);
    } else {
      std::string tmp = fresh();
      os << ind() << deviceRetType(callee) << " " << tmp << " = " << call
         << ";\n";
      for (auto [i, res] : llvm::enumerate(op.getResults())) {
        std::string id = fresh();
        os << ind() << mslScalarType(res.getType()) << " " << id << " = " << tmp
           << ".f" << i << ";\n";
        bindScalar(res, id);
      }
    }
    return success();
  }

  // Emit a multi-block region as a state-machine dispatch loop. MSL forbids
  // goto/labels, so unstructured CFG is lowered to `while (true) { if (state ==
  // N) { ... } }` with an integer state var per block. Block arguments become
  // predeclared variables assigned on each branch edge (phi resolution) before
  // the state transition.
  LogicalResult emitBlockCFG(Region &region) {
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
        valMap[arg] = declResultVars(arg, StringRef());
      }

    // Each per-block `if` opens a C scope, so any SSA value used outside its
    // defining block must live in a function-scope variable. Predeclare a
    // hoisted var per crossing value; after the defining op emits, spill into
    // it and rebind so cross-block uses read the hoisted name.
    llvm::DenseMap<Value, SmallVector<std::string>> hoist;
    for (Block &blk : region)
      for (Operation &op : blk)
        for (Value res : op.getResults()) {
          if (isDatalessType(res.getType()))
            continue;
          bool crosses = llvm::any_of(res.getUsers(), [&](Operation *u) {
            return u->getBlock() != &blk;
          });
          if (!crosses)
            continue;
          hoist[res] = declResultVars(res, StringRef());
        }

    std::string state = fresh();
    os << ind() << "int " << state << " = 0;\n";
    cfgState = state;
    os << ind() << "while (true) {\n";
    ++indent;
    bool first = true;
    for (Block &blk : region) {
      os << ind() << (first ? "if" : "else if") << " (" << state << " == "
         << blockLabel[&blk] << ") {\n";
      first = false;
      ++indent;
      for (Operation &op : blk.without_terminator()) {
        if (failed(emitOp(&op)))
          return failure();
        for (Value res : op.getResults()) {
          auto it = hoist.find(res);
          if (it == hoist.end())
            continue;
          auto &cur = names(res);
          for (size_t r = 0; r < it->second.size(); ++r)
            os << ind() << it->second[r] << " = "
               << cur[cur.size() == 1 ? 0 : r] << ";\n";
          valMap[res] = it->second;
        }
      }
      flushBarrier();
      if (failed(emitTerminator(blk.getTerminator())))
        return failure();
      --indent;
      os << ind() << "}\n";
    }
    --indent;
    os << ind() << "}\n";
    cfgState.clear();
    return success();
  }

  // Assign a successor block's argument variables from the branch's operands
  // (phi resolution), transition the dispatch state, and re-enter the loop.
  void emitBranchEdge(Block *succ, Operation::operand_range args) {
    for (auto [i, operand] : llvm::enumerate(args)) {
      auto &src = names(operand);
      auto &dst = valMap[succ->getArgument(i)];
      for (size_t r = 0; r < dst.size(); ++r)
        os << ind() << dst[r] << " = " << src[src.size() == 1 ? 0 : r] << ";\n";
    }
    os << ind() << cfgState << " = " << blockLabel[succ] << ";\n";
    os << ind() << "continue;\n";
  }

  LogicalResult emitTerminator(Operation *term) {
    if (auto br = dyn_cast<cf::BranchOp>(term)) {
      emitBranchEdge(br.getDest(), br.getDestOperands());
      return success();
    }
    if (auto cbr = dyn_cast<cf::CondBranchOp>(term)) {
      const std::string &c = names(cbr.getCondition())[0];
      os << ind() << "if (" << c << ") {\n";
      ++indent;
      emitBranchEdge(cbr.getTrueDest(), cbr.getTrueDestOperands());
      --indent;
      os << ind() << "} else {\n";
      ++indent;
      emitBranchEdge(cbr.getFalseDest(), cbr.getFalseDestOperands());
      --indent;
      os << ind() << "}\n";
      return success();
    }
    return emitOp(term);
  }

  std::string tgposId, tidId, numTgId, laneId, warpId;
  bool scalarSpinlock = false;

  // Register-resident C GEMM fusion. When emitFor recognises an
  // `acc = tl.dot(a, b, acc)` K-loop it drives the enclosed tt.dot through the
  // three-phase path below: PhaseDecl declares persistent simdgroup fragments
  // (once, pre-loop), PhaseMMA stages A/B and accumulates into them (each
  // iteration, no tgC round-trip), PhaseReadback stores the fragments and
  // gathers the #mma->scalar result (once, post-loop).
  enum class FusedDotPhase { None, Decl, MMA, Readback };
  // When the fused GEMM accumulator flows only into a terminal row-major
  // tt.store, the readback stores the accumulator fragments straight to device
  // memory (skipping the threadgroup pool round-trip and the swizzled scalar
  // gather). Populated when matchDirectStore succeeds; empty otherwise.
  struct DirectStore {
    tt::StoreOp store;
    Value basePtr; // C matrix base pointer (scalar kernel arg)
    Value ldc;     // row stride (scalar), col stride is 1 (row-major)
    Value rowBase; // global row of the tile's top-left element (scalar)
    Value colBase; // global col of the tile's top-left element (scalar)
    Value boundM;  // store-mask row bound, or null when unmasked
    Value boundN;  // store-mask col bound, or null when unmasked
    std::string fullTileVar; // runtime "whole tile in bounds" predicate
  };
  struct FusedDotCtx {
    FusedDotPhase phase = FusedDotPhase::None;
    SmallVector<std::string> accNames;
    SmallVector<std::string> ids;
    SmallVector<std::string> baseNames;
    std::string tgA, tgB, tgC;
    std::optional<DirectStore> direct;
  };
  FusedDotCtx fusedDot;
  // Terminal stores the fused readback already wrote directly to device (keyed
  // by op), each with the runtime predicate under which it did so. emitStore
  // guards the store on the negation to avoid a double write.
  DenseMap<Operation *, std::string> directStoreHandled;

  static bool tracesToKernelArg(Value v) {
    while (v) {
      if (isa<BlockArgument>(v))
        return true;
      Operation *def = v.getDefiningOp();
      if (auto ap = dyn_cast_or_null<tt::AddPtrOp>(def)) {
        v = ap.getPtr();
        continue;
      }
      if (auto sp = dyn_cast_or_null<tt::SplatOp>(def)) {
        v = sp.getSrc();
        continue;
      }
      if (auto bc = dyn_cast_or_null<tt::BitcastOp>(def)) {
        v = bc.getSrc();
        continue;
      }
      return false;
    }
    return false;
  }

  static Value peelBroadcastExpand(Value v) {
    while (v) {
      Operation *def = v.getDefiningOp();
      if (auto b = dyn_cast_or_null<tt::BroadcastOp>(def)) {
        v = b.getSrc();
        continue;
      }
      if (auto e = dyn_cast_or_null<tt::ExpandDimsOp>(def)) {
        v = e.getSrc();
        continue;
      }
      return v;
    }
    return v;
  }

  // A per-tile index `pidBase*TILE + iota` where iota is a 0-based make_range.
  // Returns the scalar tile-base value (pidBase*TILE) or null.
  static Value matchTileIndex(Value v) {
    v = peelBroadcastExpand(v);
    auto add = dyn_cast_or_null<arith::AddIOp>(v.getDefiningOp());
    if (!add)
      return Value();
    Value a = peelBroadcastExpand(add.getLhs());
    Value b = peelBroadcastExpand(add.getRhs());
    auto isIota = [](Value x) {
      auto sp = dyn_cast_or_null<tt::SplatOp>(x.getDefiningOp());
      Value base = sp ? sp.getSrc() : x;
      auto mr = dyn_cast_or_null<tt::MakeRangeOp>(
          peelBroadcastExpand(base).getDefiningOp());
      return mr && mr.getStart() == 0;
    };
    auto splatScalar = [](Value x) -> Value {
      auto sp = dyn_cast_or_null<tt::SplatOp>(x.getDefiningOp());
      return sp ? sp.getSrc() : Value();
    };
    if (isIota(b))
      if (Value s = splatScalar(a))
        return s;
    if (isIota(a))
      if (Value s = splatScalar(b))
        return s;
    return Value();
  }

  // Split an addptr offset `add(rowTerm, colTerm)` into (rowBase, ldc, colBase)
  // where rowTerm = broadcast(rowIdx * splat(ldc)) and colTerm =
  // broadcast(colIdx), each index a per-tile `base + iota`. ldc must be a
  // scalar (loop-invariant); col stride is the implicit 1 of row-major.
  bool matchRowMajorOffset(Value off, Value &rowBase, Value &ldc,
                           Value &colBase) {
    auto add = dyn_cast_or_null<arith::AddIOp>(off.getDefiningOp());
    if (!add)
      return false;
    auto tryTerm = [&](Value rowT, Value colT) {
      Value cb = matchTileIndex(colT);
      if (!cb)
        return false;
      Value rowScaled = peelBroadcastExpand(rowT);
      auto mul = dyn_cast_or_null<arith::MulIOp>(rowScaled.getDefiningOp());
      if (!mul)
        return false;
      auto scalarSrc = [](Value x) -> Value {
        auto sp = dyn_cast_or_null<tt::SplatOp>(
            peelBroadcastExpand(x).getDefiningOp());
        return sp ? sp.getSrc() : Value();
      };
      Value rIdxA = mul.getLhs(), rIdxB = mul.getRhs();
      Value rb = matchTileIndex(rIdxA);
      Value stride = scalarSrc(rIdxB);
      if (!rb || !stride) {
        rb = matchTileIndex(rIdxB);
        stride = scalarSrc(rIdxA);
      }
      if (!rb || !stride)
        return false;
      rowBase = rb;
      ldc = stride;
      colBase = cb;
      return true;
    };
    return tryTerm(add.getLhs(), add.getRhs()) ||
           tryTerm(add.getRhs(), add.getLhs());
  }

  // The store boundary mask `(row < boundM) && (col < boundN)`, with row/col the
  // same per-tile indices as the address. Extracts the two scalar bounds.
  bool matchBoundaryMask(Value m, Value &boundM, Value &boundN) {
    auto conj = dyn_cast_or_null<arith::AndIOp>(m.getDefiningOp());
    if (!conj)
      return false;
    auto cmpBound = [&](Value side, bool wantRow) -> Value {
      Value c = peelBroadcastExpand(side);
      auto cmp = dyn_cast_or_null<arith::CmpIOp>(c.getDefiningOp());
      if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
        return Value();
      if (!matchTileIndex(cmp.getLhs()))
        return Value();
      auto sp = dyn_cast_or_null<tt::SplatOp>(
          peelBroadcastExpand(cmp.getRhs()).getDefiningOp());
      return sp ? sp.getSrc() : Value();
    };
    if (Value bm = cmpBound(conj.getLhs(), true))
      if (Value bn = cmpBound(conj.getRhs(), false)) {
        boundM = bm;
        boundN = bn;
        return true;
      }
    if (Value bm = cmpBound(conj.getRhs(), true))
      if (Value bn = cmpBound(conj.getLhs(), false)) {
        boundM = bm;
        boundN = bn;
        return true;
      }
    return false;
  }

  // Recognise `acc(#mma) -> convert_layout -> tt.store(rowmajor)` as the sole
  // consumer of the fused accumulator. Only an f32->f32 store through a single
  // convert_layout is taken; anything else (dtype cast, reduction, second dot,
  // extra users) falls back to the pool readback. Fills `ds` on success.
  bool matchDirectStore(Value forResult, DirectStore &ds) {
    if (getenv("MSL_NO_DIRECT_STORE"))
      return false;
    if (!forResult.hasOneUse())
      return false;
    auto cvt = dyn_cast<ttg::ConvertLayoutOp>(*forResult.user_begin());
    if (!cvt || !cvt.getResult().hasOneUse())
      return false;
    auto store = dyn_cast<tt::StoreOp>(*cvt.getResult().user_begin());
    if (!store || store.getValue() != cvt.getResult())
      return false;
    auto cTy = dyn_cast<RankedTensorType>(cvt.getResult().getType());
    if (!cTy || !cTy.getElementType().isF32())
      return false;
    auto ptr = dyn_cast_or_null<tt::AddPtrOp>(store.getPtr().getDefiningOp());
    if (!ptr)
      return false;
    auto splat = dyn_cast_or_null<tt::SplatOp>(ptr.getPtr().getDefiningOp());
    if (!splat || !isa<BlockArgument>(splat.getSrc()))
      return false;
    Value rowBase, ldc, colBase;
    if (!matchRowMajorOffset(ptr.getOffset(), rowBase, ldc, colBase))
      return false;
    Value boundM, boundN;
    if (store.getMask()) {
      if (!matchBoundaryMask(store.getMask(), boundM, boundN))
        return false;
    }
    ds.store = store;
    ds.basePtr = splat.getSrc();
    ds.ldc = ldc;
    ds.rowBase = rowBase;
    ds.colBase = colBase;
    ds.boundM = boundM;
    ds.boundN = boundN;
    return true;
  }

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

  static bool isPureBarrierOp(Operation *op) {
    return isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp, ttg::BarrierOp,
               mlir::gpu::BarrierOp>(op);
  }

  LogicalResult emitOp(Operation *op) {
    if (!isPureBarrierOp(op))
      flushBarrier();
    if (auto c = dyn_cast<arith::ConstantOp>(op))
      return emitConstant(c);
    if (auto p = dyn_cast<tt::GetProgramIdOp>(op))
      return emitProgramId(p);
    if (auto n = dyn_cast<tt::GetNumProgramsOp>(op))
      return emitNumPrograms(n);
    if (auto r = dyn_cast<tt::MakeRangeOp>(op))
      return emitMakeRange(r);
    if (auto s = dyn_cast<tt::SplatOp>(op))
      return emitSplat(s);
    if (auto u = dyn_cast<tt::UnsplatOp>(op)) {
      bindScalar(u.getResult(), names(u.getSrc())[0]);
      return success();
    }
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
    if (auto a = dyn_cast<tt::AtomicPollOp>(op))
      return emitAtomicPoll(a);
    if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp, arith::DivSIOp,
            arith::DivUIOp, arith::RemSIOp, arith::RemUIOp>(op))
      return emitIntBinary(op);
    if (isa<arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp>(op))
      return emitShift(op);
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
    if (isa<arith::MaximumFOp>(op))
      return emitMinMax(op, "max", "", /*propagateNan=*/true);
    if (isa<arith::MinimumFOp>(op))
      return emitMinMax(op, "min", "", /*propagateNan=*/true);
    if (isa<arith::MaxUIOp>(op))
      return emitMinMax(
          op, "max",
          mslUnsignedType(elementScalarType(op->getResult(0).getType())));
    if (isa<arith::MinUIOp>(op))
      return emitMinMax(
          op, "min",
          mslUnsignedType(elementScalarType(op->getResult(0).getType())));
    if (isa<arith::MaxNumFOp, arith::MaxSIOp>(op))
      return emitMinMax(op, "max");
    if (isa<arith::MinNumFOp, arith::MinSIOp>(op))
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
    if (isa<tt::IntToPtrOp, tt::PtrToIntOp>(op))
      return emitPtrIntCast(op);
    if (auto f = dyn_cast<tt::FpToFpOp>(op))
      return emitCast(op);
    if (auto c = dyn_cast<tt::ClampFOp>(op))
      return emitClamp(c);
    if (isa<tt::MulhiUIOp>(op))
      return emitMinMax(
          op, "mulhi",
          mslUnsignedType(elementScalarType(op->getResult(0).getType())));
    if (isa<tt::PreciseSqrtOp>(op))
      return emitUnary(
          op, msl::builtin::precise::Sqrt,
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
    if (auto h = dyn_cast<tt::HistogramOp>(op))
      return emitHistogram(h);
    if (auto m = dyn_cast<tt::MapElementwiseOp>(op))
      return emitMapElementwise(m);
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
    if (auto g = dyn_cast<tt::GatherOp>(op))
      return emitGather(g);
    if (auto r = dyn_cast<tt::ReshapeOp>(op))
      return emitReshape(r);
    if (auto a = dyn_cast<ttg::LocalAllocOp>(op))
      return emitLocalAlloc(a);
    if (auto i = dyn_cast<ttg::MemDescIndexOp>(op))
      return emitMemDescIndex(i);
    if (auto s = dyn_cast<ttg::MemDescSubsliceOp>(op))
      return emitMemDescSubslice(s);
    if (auto c = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op))
      return emitAsyncCopy(c);
    if (auto l = dyn_cast<ttg::LocalStoreOp>(op))
      return emitLocalStore(l);
    if (auto l = dyn_cast<ttg::LocalLoadOp>(op))
      return emitLocalLoad(l);
    if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(op)) {
      emitBarrier(/*device=*/false);
      for (Value r : op->getResults())
        valMap[r] = SmallVector<std::string>{};
      return success();
    }
    if (auto b = dyn_cast<ttg::BarrierOp>(op)) {
      uint32_t bits = static_cast<uint32_t>(b.getAddrSpace());
      bool device = bits & (static_cast<uint32_t>(ttg::AddrSpace::GlobalRead) |
                            static_cast<uint32_t>(ttg::AddrSpace::GlobalWrite));
      emitBarrier(device);
      return success();
    }
    if (isa<mlir::gpu::BarrierOp>(op)) {
      emitBarrier(/*device=*/true);
      return success();
    }
    if (isa<ttg::LocalDeallocOp>(op))
      return success();
    if (op->getName().getStringRef() == "llvm.intr.assume")
      return success();
    if (isa<scf::YieldOp>(op))
      return success();
    if (auto c = dyn_cast<tt::CallOp>(op))
      return emitCall(c);
    if (auto r = dyn_cast<tt::ReturnOp>(op))
      return emitReturn(r);
    if (op->getName().getStringRef() == "ub.poison") {
      Value res = op->getResult(0);
      Type elem = res.getType();
      if (auto rt = dyn_cast<RankedTensorType>(elem))
        elem = rt.getElementType();
      bool isPtr = isa<tt::PointerType>(elem);
      std::string sc =
          isPtr ? mslStorageType(res.getType())
                : mslScalarType(elementScalarType(res.getType()));
      if (sc.empty()) {
        op->emitError("EmitMSL: unhandled poison type");
        return failure();
      }
      int rc = regCount(res);
      SmallVector<std::string> ids;
      for (int r = 0; r < rc; ++r) {
        std::string id = fresh();
        if (isPtr)
          os << ind() << sc << " " << id << " = nullptr;\n";
        else
          os << ind() << sc << " " << id << " = (" << sc << ")0;\n";
        ids.push_back(id);
      }
      valMap[res] = ids;
      return success();
    }
    op->emitError("EmitMSL: unhandled op '" + op->getName().getStringRef() +
                  "'");
    return failure();
  }

  std::string floatLit(const APFloat &v);
  std::string floatLit(const APFloat &v, StringRef sc);
  msl::Expr *astFloatLit(const APFloat &v, StringRef sc);

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
                              ? floatLit(dense.getSplatValue<APFloat>(), sc)
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
          os << (n++ ? ", " : "") << floatLit(v, sc);
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
      lit = floatLit(fa.getValue(), sc);
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

  LogicalResult emitNumPrograms(tt::GetNumProgramsOp op) {
    std::string id = fresh();
    const char *comp = op.getAxis() == tt::ProgramIDDim::X   ? "x"
                       : op.getAxis() == tt::ProgramIDDim::Y ? "y"
                                                             : "z";
    os << ind() << "int " << id << " = (int)(" << numTgId << "." << comp
       << ");\n";
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

  LogicalResult emitIntBinary(Operation *op);
  LogicalResult emitFloatBinary(Operation *op);
  LogicalResult emitShift(Operation *op);
  LogicalResult emitElementwise(Operation *op, StringRef binop, StringRef sc,
                                StringRef opCast = "");
  LogicalResult emitMinMax(Operation *op, StringRef fn, StringRef opCast = "",
                           bool propagateNan = false);
  LogicalResult emitUnary(Operation *op, StringRef fn, StringRef sc);

  // AST sub-expression builders: yield the RHS Expr* the string paths above
  // print, resolving operands from already-looked-up register names. Not yet
  // driving output (Layer 2); later layers assign these into DeclStmt inits.
  msl::Expr *astIntBinaryExpr(Operation *op, StringRef a, StringRef b);
  msl::Expr *astShiftExpr(Operation *op, StringRef a, StringRef b);
  msl::Expr *astElementwiseExpr(msl::BinOp op, msl::Type *opCast, StringRef a,
                                StringRef b);
  msl::Expr *astMinMaxExpr(StringRef fn, msl::Type *opCast, bool propagateNan,
                           StringRef a, StringRef b);
  msl::Expr *astUnaryExpr(StringRef fn, msl::Type *sc, StringRef v);
  msl::Expr *astTernaryCallExpr(StringRef fn, StringRef a, StringRef b,
                                StringRef c);

  LogicalResult emitMathUnary(Operation *op) {
    std::string sc = mslScalarType(elementScalarType(op->getResult(0).getType()));
    StringRef n = op->getName().getStringRef();
    // metal:: trig/exp/log/sqrt lower to air.fast_* (approximate) even under
    // -fmetal-math-mode=safe; the math mode never controls transcendental
    // accuracy, only the namespace does. Inductor checks against aten's accurate
    // transcendentals at fp32 tolerance (rel ~1e-6), which the fast_* variants
    // miss (e.g. tan via sin/cos, pow via exp2/log2). Use metal::precise:: for
    // the accuracy-sensitive functions; exact ops (floor/ceil/round/abs) keep
    // the plain form.
    namespace bi = msl::builtin;
    static const llvm::StringMap<StringRef> unary = {
        {"math.exp", bi::precise::Exp},
        {"math.exp2", bi::precise::Exp2},
        {"math.log", bi::precise::Log},
        {"math.log2", bi::precise::Log2},
        {"math.log10", bi::precise::Log10},
        {"math.sin", bi::precise::Sin},
        {"math.cos", bi::precise::Cos},
        {"math.tan", bi::precise::Tan},
        {"math.tanh", bi::precise::Tanh},
        {"math.sinh", bi::precise::Sinh},
        {"math.cosh", bi::precise::Cosh},
        {"math.asin", bi::precise::Asin},
        {"math.acos", bi::precise::Acos},
        {"math.atan", bi::precise::Atan},
        {"math.sqrt", bi::precise::Sqrt},
        {"math.rsqrt", bi::precise::Rsqrt},
        {"math.cbrt", bi::precise::Cbrt},
        {"math.floor", bi::math::Floor}, {"math.ceil", bi::math::Ceil},
        {"math.absf", bi::math::Fabs},   {"math.absi", bi::math::Abs},
        {"math.erf", "tt_erf"},          {"math.round", bi::math::Round},
        {"math.trunc", bi::math::Trunc}, {"math.roundeven", bi::math::Rint}};
    if (auto it = unary.find(n); it != unary.end())
      return emitUnary(op, it->second, sc);
    static const llvm::StringMap<StringRef> binary = {
        {"math.atan2", bi::precise::Atan2},
        {"math.powf", bi::precise::Pow},
        {"math.fpowi", bi::precise::Pow},
        {"math.copysign", bi::math::Copysign}};
    if (auto it = binary.find(n); it != binary.end())
      return emitMinMax(op, it->second);
    if (n == "math.fma")
      return emitTernary(op, msl::builtin::math::Fma, sc);
    if (n == "math.exp10") {
      Value res = op->getResult(0);
      auto &a = names(op->getOperand(0));
      int rc = regCount(res);
      SmallVector<std::string> ids;
      for (int r = 0; r < rc; ++r) {
        std::string id = fresh();
        os << ind() << sc << " " << id << " = " << msl::builtin::precise::Pow
           << "((" << sc << ")10, " << a[a.size() == 1 ? 0 : r] << ");\n";
        ids.push_back(id);
      }
      valMap[res] = ids;
      return success();
    }
    op->emitError("EmitMSL: unhandled math op '" + n + "'");
    return failure();
  }

  LogicalResult emitTernary(Operation *op, StringRef fn, StringRef sc);
  LogicalResult emitCast(Operation *op);
  LogicalResult emitPtrIntCast(Operation *op);
  LogicalResult emitBitcast(Operation *op);
  LogicalResult emitClamp(tt::ClampFOp op);
  LogicalResult emitCmpI(arith::CmpIOp op);
  LogicalResult emitCmpF(arith::CmpFOp op);
  LogicalResult emitSelect(arith::SelectOp op);

  msl::Expr *astCastExpr(Operation *op, StringRef v);
  msl::Expr *astPtrIntCastExpr(Operation *op, StringRef v);
  msl::Expr *astBitcastExpr(Operation *op, StringRef v);
  msl::Expr *astClampExpr(tt::ClampFOp op, StringRef x, StringRef lo,
                          StringRef hi);
  msl::Expr *astSelectExpr(StringRef c, StringRef t, StringRef f);

  LogicalResult emitRegionBody(Region &region) {
    Block &blk = region.front();
    for (Operation &op : blk.without_terminator())
      if (failed(emitOp(&op)))
        return failure();
    flushBarrier();
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
    Type elem = v.getType();
    if (auto rt = dyn_cast<RankedTensorType>(elem))
      elem = rt.getElementType();
    std::string sc = isa<tt::PointerType>(elem)
                         ? mslStorageType(v.getType())
                         : mslScalarType(elementScalarType(v.getType()));
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

  // Recognise the register-resident GEMM shape: a loop-carried #mma iter-arg
  // that is the C operand of exactly one tt.dot in the body and whose only
  // other use is being yielded back as that same iter-arg (the standard
  // accumulating `acc = tl.dot(a, b, acc)` K-loop). Returns the dot and the
  // iter-arg index, or nullopt.
  std::optional<std::pair<tt::DotOp, unsigned>>
  matchGemmDotLoop(scf::ForOp op) {
    if (getenv("MSL_NO_FUSE"))
      return std::nullopt;
    Block *body = op.getBody();
    auto yield = cast<scf::YieldOp>(body->getTerminator());
    tt::DotOp found;
    int nDots = 0;
    for (Operation &o : body->without_terminator())
      if (auto d = dyn_cast<tt::DotOp>(&o)) {
        found = d;
        ++nDots;
      }
    if (nDots != 1 || !found)
      return std::nullopt;

    auto cArg = dyn_cast<BlockArgument>(found.getC());
    if (!cArg || cArg.getOwner() != body)
      return std::nullopt;
    unsigned idx = cArg.getArgNumber();
    if (idx == 0)
      return std::nullopt; // arg 0 is the induction var
    unsigned iterIdx = idx - 1;
    if (yield.getOperand(iterIdx) != found.getResult())
      return std::nullopt;
    // The iter-arg feeds the dot's C and nothing else; the dot result feeds the
    // yield and nothing else. This keeps the accumulator purely register-carried.
    for (Operation *u : cArg.getUsers())
      if (u != found.getOperation())
        return std::nullopt;
    for (Operation *u : found.getResult().getUsers())
      if (u != yield.getOperation())
        return std::nullopt;

    auto cTy = dyn_cast<RankedTensorType>(found.getResult().getType());
    if (!cTy || cTy.getRank() != 2)
      return std::nullopt;
    Type aElem = cast<RankedTensorType>(found.getA().getType()).getElementType();
    Type cElem = cTy.getElementType();
    if (isa<IntegerType>(aElem) || !(cElem.isF32() || cElem.isF16()))
      return std::nullopt;
    int64_t M = cTy.getShape()[0], N = cTy.getShape()[1];
    int64_t K = cast<RankedTensorType>(found.getA().getType()).getShape()[1];
    if (M % 8 || N % 8 || K % 8)
      return std::nullopt;

    // Gate: the fused path only wins with a small warp-tile (<= 8
    // simdgroup_float8x8 accumulators per warp) AND the disjoint staging path
    // where staged A+B+C fits the pool (band == M, one readback). Anything
    // larger falls back to the per-dot path. Staging bytes mirror emitDot: an
    // operand already resident in a threadgroup buffer (in-place) stages 0.
    int64_t aBytes = M * K * (bitsOf(aElem) / 8);
    int64_t bBytes = N * K * (bitsOf(aElem) / 8);
    int64_t cFull = M * N * 4;
    bool wholeTileFits = aBytes + bBytes <= 32768;
    // A/B that structurally resolve to a local_alloc buffer are loaded in place
    // by emitDot (stage 0). The precise in-place base lives in memdescMap, which
    // is only populated once the enclosing memdesc_index is emitted inside the
    // loop; here (pre-loop) the structural walk is the reliable signal.
    int64_t stagedA = aBytes, stagedB = bBytes;
    if (wholeTileFits) {
      if (dotOperandLocalLoad(found.getA(), M, K))
        stagedA = 0;
      if (dotOperandLocalLoad(found.getB(), K, N))
        stagedB = 0;
    }
    if (stagedA + stagedB + cFull > poolBudget())
      return std::nullopt;
    tt::LinearLayout cLL = ttg::toLinearLayout(cTy);
    auto kWarpDim = StringAttr::get(op.getContext(), "warp");
    int64_t numWarps = cLL.hasInDim(kWarpDim) ? cLL.getInDimSize(kWarpDim) : 1;
    int64_t nFrag = (M / 8) * (N / 8);
    if (numWarps > nFrag)
      numWarps = nFrag;
    int64_t fragsPerWarp = (nFrag + numWarps - 1) / numWarps;
    if (fragsPerWarp > 8)
      return std::nullopt;

    return std::make_pair(found, iterIdx);
  }

  LogicalResult emitFor(scf::ForOp op) {
    if (auto m = matchGemmDotLoop(op))
      return emitFusedGemm(op, m->first, m->second);
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
    std::string lo = names(op.getLowerBound())[0];
    std::string hi = names(op.getUpperBound())[0];
    std::string st = names(op.getStep())[0];
    bindScalar(op.getInductionVar(), iv);
    Type ivType = op.getInductionVar().getType();
    std::string ivTy = mslScalarType(ivType);
    if (ivTy.empty())
      ivTy = "int";
    bool wideIv = ivType.isInteger(64);
    if (wideIv) {
      std::string tc = fresh();
      os << ind() << "for (" << ivTy << " " << tc << " = 0; ; " << tc
         << " += 1) {\n";
      ++indent;
      os << ind() << ivTy << " " << iv << " = " << lo << " + " << tc << " * "
         << st << ";\n";
      os << ind() << "if (!(" << iv << " < " << hi << ")) break;\n";
    } else {
      os << ind() << "for (" << ivTy << " " << iv << " = " << lo << "; " << iv
         << " < " << hi << "; " << iv << " += " << st << ") {\n";
      ++indent;
    }
    if (failed(emitRegionBody(op.getRegion())))
      return failure();
    emitYieldAssign(op.getBody()->getTerminator(), carried);
    --indent;
    os << ind() << "}\n";
    return success();
  }

  // Lower a recognised GEMM K-loop with the dot accumulator kept resident in
  // simdgroup_matrix registers across the whole loop. The enclosed tt.dot is
  // driven in three phases (see FusedDotCtx); every other carried value uses
  // the normal scalar carry.
  LogicalResult emitFusedGemm(scf::ForOp op, tt::DotOp dot, unsigned iterIdx) {
    SmallVector<SmallVector<std::string>> carried;
    SmallVector<std::string> initBase;
    for (auto [i, init, res] :
         llvm::enumerate(op.getInitArgs(), op.getResults())) {
      if (isDatalessType(res.getType())) {
        valMap[op.getRegionIterArg(i)] = SmallVector<std::string>{};
        valMap[res] = SmallVector<std::string>{};
        carried.push_back({});
        continue;
      }
      auto &initNames = names(init);
      if (i == iterIdx) {
        // The accumulator carry is the persistent simdgroup fragments; its
        // scalar registers are only produced once after the loop. Capture the
        // init value as the readback base and give the iter-arg placeholder
        // names (never read: its only use is the dot's C operand).
        SmallVector<std::string> ids = declResultVars(res, StringRef());
        initBase.assign(initNames.begin(), initNames.end());
        fusedDot.ids = ids;
        valMap[op.getRegionIterArg(i)] = ids;
        valMap[res] = ids;
        carried.push_back({});
        continue;
      }
      SmallVector<std::string> vars = declResultVars(res, StringRef());
      for (size_t r = 0; r < vars.size(); ++r)
        os << ind() << vars[r] << " = "
           << initNames[initNames.size() == 1 ? 0 : r] << ";\n";
      valMap[op.getRegionIterArg(i)] = vars;
      valMap[res] = vars;
      carried.push_back(vars);
    }

    fusedDot.baseNames = initBase;
    fusedDot.phase = FusedDotPhase::Decl;
    if (failed(emitDot(dot)))
      return failure();

    std::string iv = fresh();
    std::string lo = names(op.getLowerBound())[0];
    std::string hi = names(op.getUpperBound())[0];
    std::string st = names(op.getStep())[0];
    bindScalar(op.getInductionVar(), iv);
    std::string ivTy = mslScalarType(op.getInductionVar().getType());
    if (ivTy.empty())
      ivTy = "int";
    os << ind() << "for (" << ivTy << " " << iv << " = " << lo << "; " << iv
       << " < " << hi << "; " << iv << " += " << st << ") {\n";
    ++indent;
    fusedDot.phase = FusedDotPhase::MMA;
    if (failed(emitRegionBody(op.getRegion())))
      return failure();
    fusedDot.phase = FusedDotPhase::None;
    // Carry every value except the accumulator (its state lives in the frags).
    auto *term = op.getBody()->getTerminator();
    for (auto [i, operand] : llvm::enumerate(term->getOperands())) {
      if (i == iterIdx || carried[i].empty())
        continue;
      auto &src = names(operand);
      for (size_t r = 0; r < carried[i].size(); ++r)
        os << ind() << carried[i][r] << " = "
           << src[src.size() == 1 ? 0 : r] << ";\n";
    }
    --indent;
    os << ind() << "}\n";

    DirectStore ds;
    if (matchDirectStore(op.getResult(iterIdx), ds)) {
      int64_t M = cast<RankedTensorType>(dot.getResult().getType()).getShape()[0];
      int64_t N = cast<RankedTensorType>(dot.getResult().getType()).getShape()[1];
      std::string ft = fresh();
      ds.fullTileVar = ft;
      std::string cond = "true";
      if (ds.boundM) {
        cond = "(" + names(ds.rowBase)[0] + " + " + std::to_string(M) +
               " <= " + names(ds.boundM)[0] + " && " + names(ds.colBase)[0] +
               " + " + std::to_string(N) + " <= " + names(ds.boundN)[0] + ")";
      }
      os << ind() << "bool " << ft << " = " << cond << ";\n";
      fusedDot.direct = ds;
      directStoreHandled[ds.store.getOperation()] = ft;
    }

    fusedDot.phase = FusedDotPhase::Readback;
    if (failed(emitDot(dot)))
      return failure();
    fusedDot = FusedDotCtx{};
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
  LogicalResult emitMapElementwise(tt::MapElementwiseOp op) {
    Region &region = op.getScalarOp();
    Block &blk = region.front();
    int nSrc = op.getNumOperands();
    int nRes = op.getNumResults();
    int pack = op.getPack();

    // Snapshot by value: bindScalar/emitOp below insert into valMap, which can
    // rehash the DenseMap and dangle any reference held into it.
    SmallVector<SmallVector<std::string>> srcNames(nSrc);
    for (int s = 0; s < nSrc; ++s)
      srcNames[s] = names(op.getOperand(s));
    int nReg = srcNames[0].size();
    int nGroup = nReg / pack;
    bool multiBlock = !region.hasOneBlock();

    SmallVector<SmallVector<std::string>> resIds(nRes);
    for (int g = 0; g < nGroup; ++g) {
      for (int s = 0; s < nSrc; ++s)
        for (int p = 0; p < pack; ++p)
          bindScalar(blk.getArgument(s * pack + p),
                     srcNames[s][g * pack + p]);
      if (multiBlock) {
        SmallVector<std::string> capture(nRes * pack);
        for (int i = 0; i < nRes * pack; ++i) {
          capture[i] = fresh();
          Value r = op->getResult(i / pack);
          os << ind()
             << mslScalarType(elementScalarType(r.getType())) << " "
             << capture[i] << ";\n";
        }
        if (failed(emitMapCFG(region, capture)))
          return failure();
        for (int k = 0; k < nRes; ++k)
          for (int p = 0; p < pack; ++p)
            resIds[k].push_back(capture[k * pack + p]);
        continue;
      }
      for (Operation &o : blk.without_terminator())
        if (failed(emitOp(&o)))
          return failure();
      Operation *term = blk.getTerminator();
      for (int k = 0; k < nRes; ++k)
        for (int p = 0; p < pack; ++p)
          resIds[k].push_back(names(term->getOperand(k * pack + p))[0]);
    }
    for (int k = 0; k < nRes; ++k)
      valMap[op->getResult(k)] = resIds[k];
    return success();
  }

  // Run one group of a multi-block map_elementwise region as a state-machine
  // dispatch loop (MSL forbids goto), spilling the map_elementwise.return
  // operands into caller-provided capture vars and exiting the loop.
  LogicalResult emitMapCFG(Region &region, ArrayRef<std::string> capture) {
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
        valMap[arg] = declResultVars(arg, StringRef());
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
          if (!crosses)
            continue;
          hoist[res] = declResultVars(res, StringRef());
        }

    std::string state = fresh();
    os << ind() << "int " << state << " = 0;\n";
    cfgState = state;
    os << ind() << "while (true) {\n";
    ++indent;
    bool first = true;
    for (Block &blk : region) {
      os << ind() << (first ? "if" : "else if") << " (" << state << " == "
         << blockLabel[&blk] << ") {\n";
      first = false;
      ++indent;
      for (Operation &op : blk.without_terminator()) {
        if (failed(emitOp(&op)))
          return failure();
        for (Value res : op.getResults()) {
          auto it = hoist.find(res);
          if (it == hoist.end())
            continue;
          auto &cur = names(res);
          for (size_t r = 0; r < it->second.size(); ++r)
            os << ind() << it->second[r] << " = "
               << cur[cur.size() == 1 ? 0 : r] << ";\n";
          valMap[res] = it->second;
        }
      }
      Operation *term = blk.getTerminator();
      if (term->getName().getStringRef() == "tt.map_elementwise.return") {
        for (auto [i, operand] : llvm::enumerate(term->getOperands()))
          os << ind() << capture[i] << " = " << names(operand)[0] << ";\n";
        os << ind() << "break;\n";
      } else if (failed(emitTerminator(term))) {
        return failure();
      }
      --indent;
      os << ind() << "}\n";
    }
    --indent;
    os << ind() << "}\n";
    cfgState.clear();
    return success();
  }

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

    // Snapshot by value: emitCombineN below inserts into valMap, which can
    // rehash the DenseMap and dangle any reference held into it.
    SmallVector<SmallVector<std::string>> srcNames(nOp);
    for (int k = 0; k < nOp; ++k)
      srcNames[k] = names(op.getOperand(k));
    int nReg = srcNames[0].size();

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
    auto fullKey = [&](int reg) {
      SmallVector<int32_t> coords = registerCoords(srcTy, reg);
      std::string key;
      for (int32_t c : coords)
        key += std::to_string(c) + ",";
      return key;
    };
    std::map<std::string, SmallVector<int>> groups;
    std::set<std::string> seenFull;
    for (int r = 0; r < nReg; ++r)
      if (seenFull.insert(fullKey(r)).second)
        groups[survKey(r)].push_back(r);

    unsigned laneMask = reduceMask(ll, kLane, redDim);
    unsigned warpMask = reduceMask(ll, kWarp, redDim);
    int numWarps = ll.hasInDim(kWarp) ? ll.getInDimSize(kWarp) : 1;

    std::map<std::string, SmallVector<std::string>> groupResult;

    for (auto &g : groups) {
      SmallVector<int> &regs = g.second;
      SmallVector<std::string> accs(nOp);
      for (int k = 0; k < nOp; ++k) {
        accs[k] = fresh();
        os << ind() << scTys[k] << " " << accs[k] << " = "
           << srcNames[k][regs[0]] << ";\n";
      }
      for (size_t i = 1; i < regs.size(); ++i) {
        SmallVector<std::string> bVals(nOp);
        for (int k = 0; k < nOp; ++k)
          bVals[k] = srcNames[k][regs[i]];
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
        for (int k = 0; k < nOp; ++k)
          others[k] = emitShuffle("simd_shuffle_xor", scTys[k], accs[k],
                                  std::to_string(m) + "u");
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
        int64_t byteOff = 0;
        for (int k = 0; k < nOp; ++k) {
          scratch[k] = fresh();
          os << ind() << "threadgroup " << scTys[k] << "* " << scratch[k]
             << " = " << poolRegion(byteOff, scTys[k]) << ";\n";
          byteOff += numWarps * 32 *
                     std::max<int64_t>(
                         1, bitsOf(elementScalarType(
                                op.getResult()[k].getType())) /
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

  LogicalResult emitHistogram(tt::HistogramOp op) {
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    int64_t nBins = tileSize(resTy);

    int64_t threads = 32;
    if (auto nw = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
      threads = nw.getInt() * 32;

    std::string bins = fresh();
    os << ind() << "threadgroup atomic_uint* " << bins << " = "
       << poolRegion(0, "atomic_uint") << ";\n";
    std::string zi = fresh();
    os << ind() << "for (uint " << zi << " = " << tidId << ".x; " << zi << " < "
       << nBins << "u; " << zi << " += " << threads << "u) "
       << "atomic_store_explicit(&" << bins << "[" << zi
       << "], 0u, memory_order_relaxed);\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

    auto &srcVals = names(op.getSrc());
    SmallVector<std::string> *maskVals = nullptr;
    if (op.getMask())
      maskVals = &names(op.getMask());
    std::string srcU =
        mslUnsignedType(elementScalarType(op.getSrc().getType()));

    MLIRContext *ctx = op.getContext();
    tt::LinearLayout srcLL = ttg::toLinearLayout(srcTy);
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto srcOut = llvm::to_vector(srcLL.getOutDimNames());
    uint32_t freeMask = 0;
    auto scanFree = [&](StringAttr in, int shift) {
      if (!srcLL.hasInDim(in))
        return;
      for (int b = 0, n = srcLL.getInDimSizeLog2(in); b < n; ++b) {
        bool moves = false;
        for (auto od : srcOut)
          if (srcLL.getBasis(in, b, od) != 0)
            moves = true;
        if (!moves)
          freeMask |= 1u << (shift + b);
      }
    };
    scanFree(kLane, 0);
    scanFree(kWarp, 5);
    std::string ownerGuard =
        freeMask == 0
            ? ""
            : "(" + tidId + ".x & " + std::to_string(freeMask) + "u) == 0u";

    for (int r = 0; r < (int)srcVals.size(); ++r) {
      const std::string &v = srcVals[r];
      std::string guard = "(" + srcU + ")" + v + " < " + std::to_string(nBins) +
                          "u";
      if (!ownerGuard.empty())
        guard = ownerGuard + " && (" + guard + ")";
      if (maskVals)
        guard = "(" + (*maskVals)[maskVals->size() == 1 ? 0 : r] + ") && (" +
                guard + ")";
      os << ind() << "if (" << guard << ") atomic_fetch_add_explicit(&" << bins
         << "[" << v << "], 1u, memory_order_relaxed);\n";
    }
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

    auto outDims = llvm::to_vector(
        ttg::toLinearLayout(resTy).getOutDimNames());
    std::string resSc = mslScalarType(resTy.getElementType());
    int nResReg = regCount(op.getResult());
    SmallVector<std::string> resIds;
    for (int r = 0; r < nResReg; ++r) {
      std::string idx = layoutCoordExpr(resTy, r, outDims[0]);
      std::string id = fresh();
      os << ind() << resSc << " " << id << " = (" << resSc
         << ")atomic_load_explicit(&" << bins << "[" << idx
         << "], memory_order_relaxed);\n";
      resIds.push_back(id);
    }
    valMap[op.getResult()] = resIds;
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

  // Emit a warp shuffle (simd_shuffle / _up / _down) of `val` (type `sc`),
  // returning the fresh result name. Metal's shuffle intrinsics reject 64-bit
  // and bfloat scalars, so those are bitcast to an integer type of the same
  // width, shuffled, and reassembled.
  std::string emitShuffle(StringRef op, StringRef sc, StringRef val,
                          StringRef arg);

  // Single-expression form of the shuffle (no temporaries): the bitcast-through
  // wrapper collapsed into one nested Expr. `sc` picks the reinterpret path.
  msl::Expr *astShuffleExpr(StringRef op, StringRef sc, StringRef val,
                            StringRef arg);

  // Cross-warp inclusive carry for one register group (one independent scan).
  // Every lane writes its lane-inclusive total to threadgroup scratch keyed by
  // its full (warp*32 + lane) slot; each lane then combines the totals of the
  // lower warp partitions that share its non-axis coordinate and applies that
  // exclusive prefix to the group's registers. runTotalOut receives the group's
  // grand total (inclusive across every axis lane and warp). Self-brackets with
  // threadgroup barriers.
  LogicalResult
  emitScanWarpCarry(Region &region, int nOp, ArrayRef<std::string> scTys,
                    ArrayRef<int64_t> byteWidths,
                    ArrayRef<std::pair<int, int32_t>> warpBits,
                    ArrayRef<int> regs, SmallVector<SmallVector<std::string>> &accs,
                    ArrayRef<std::string> laneScan, StringRef axisTopLane,
                    unsigned axisWarpMask, int numWarps, bool rev,
                    SmallVectorImpl<std::string> &runTotalOut) {
    if (warpBits.empty()) {
      for (int k = 0; k < nOp; ++k) {
        std::string s =
            emitShuffle("simd_shuffle", scTys[k], laneScan[k], axisTopLane);
        os << ind() << runTotalOut[k] << " = " << s << ";\n";
      }
      return success();
    }

    SmallVector<std::string> scratch(nOp);
    int64_t byteOff = 0;
    for (int k = 0; k < nOp; ++k) {
      scratch[k] = fresh();
      os << ind() << "threadgroup " << scTys[k] << "* " << scratch[k] << " = "
         << poolRegion(byteOff, scTys[k]) << ";\n";
      byteOff += (int64_t)numWarps * 32 * byteWidths[k];
    }
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    std::string topGuard = axisTopLane == (laneId)
                               ? std::string("true")
                               : (laneId + " == " + axisTopLane.str());
    for (int k = 0; k < nOp; ++k)
      os << ind() << "if (" << topGuard << ") " << scratch[k] << "[" << warpId
         << " * 32 + " << laneId << "] = " << laneScan[k] << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

    std::string base = "((" + warpId + " & " + std::to_string(~axisWarpMask) +
                       ") * 32 + " + axisTopLane.str() + ")";

    // Warp offset (into scratch) of axis-partition index `part`, spreading its
    // bits back onto the axis-warp mask positions in ascending order.
    SmallVector<int> maskBits;
    for (size_t r = 0; r < warpBits.size(); ++r)
      maskBits.push_back(warpBits[r].first);
    int nParts = 1 << warpBits.size();
    auto partWarp = [&](int part) {
      int w = 0;
      for (size_t b = 0; b < maskBits.size(); ++b)
        if (part & (1 << b))
          w |= (1 << maskBits[b]);
      return w;
    };

    // This warp's axis-partition index (scan position among the axis warps).
    std::string myPart = fresh();
    {
      SmallVector<std::string> posTerms;
      for (size_t r = 0; r < maskBits.size(); ++r)
        posTerms.push_back("((((" + warpId + " >> " +
                           std::to_string(maskBits[r]) + ") & 1) << " +
                           std::to_string(r) + "))");
      std::string warpPos = posTerms[0];
      for (size_t i = 1; i < posTerms.size(); ++i)
        warpPos = "(" + warpPos + " | " + posTerms[i] + ")";
      os << ind() << "int " << myPart << " = " << warpPos << ";\n";
    }

    // Scan-order visit of the axis partitions (ascending, or descending for a
    // reverse scan) keeps every non-commutative combine in the right order.
    SmallVector<int> order;
    for (int p = 0; p < nParts; ++p)
      order.push_back(rev ? nParts - 1 - p : p);

    // Grand total (inclusive over all axis partitions) in scan order.
    SmallVector<std::string> grand(nOp);
    for (int k = 0; k < nOp; ++k) {
      grand[k] = fresh();
      os << ind() << scTys[k] << " " << grand[k] << " = " << scratch[k] << "["
         << base << " + " << (partWarp(order[0]) * 32) << "];\n";
    }
    for (int idx = 1; idx < nParts; ++idx) {
      SmallVector<std::string> pv(nOp);
      for (int k = 0; k < nOp; ++k) {
        pv[k] = fresh();
        os << ind() << scTys[k] << " " << pv[k] << " = " << scratch[k] << "["
           << base << " + " << (partWarp(order[idx]) * 32) << "];\n";
      }
      SmallVector<std::string> out;
      if (failed(emitCombineN(region, grand, pv, out)))
        return failure();
      for (int k = 0; k < nOp; ++k)
        os << ind() << grand[k] << " = " << out[k] << ";\n";
    }
    for (int k = 0; k < nOp; ++k)
      os << ind() << runTotalOut[k] << " = " << grand[k] << ";\n";

    // Exclusive prefix over the partitions that precede this warp in scan order.
    SmallVector<std::string> carry(nOp);
    for (int k = 0; k < nOp; ++k) {
      carry[k] = fresh();
      os << ind() << scTys[k] << " " << carry[k] << " = " << grand[k] << ";\n";
    }
    std::string init = fresh();
    os << ind() << "bool " << init << " = false;\n";
    for (int idx = 0; idx < nParts; ++idx) {
      int p = order[idx];
      std::string cond = rev ? (myPart + " < " + std::to_string(p))
                             : (myPart + " > " + std::to_string(p));
      os << ind() << "if (" << cond << ") {\n";
      indent++;
      SmallVector<std::string> pv(nOp);
      for (int k = 0; k < nOp; ++k) {
        pv[k] = fresh();
        os << ind() << scTys[k] << " " << pv[k] << " = " << scratch[k] << "["
           << base << " + " << (partWarp(p) * 32) << "];\n";
      }
      os << ind() << "if (" << init << ") {\n";
      indent++;
      SmallVector<std::string> out;
      if (failed(emitCombineN(region, carry, pv, out)))
        return failure();
      for (int k = 0; k < nOp; ++k)
        os << ind() << carry[k] << " = " << out[k] << ";\n";
      indent--;
      os << ind() << "} else {\n";
      indent++;
      for (int k = 0; k < nOp; ++k)
        os << ind() << carry[k] << " = " << pv[k] << ";\n";
      os << ind() << init << " = true;\n";
      indent--;
      os << ind() << "}\n";
      indent--;
      os << ind() << "}\n";
    }
    for (int r : regs) {
      SmallVector<std::string> ar(nOp);
      for (int k = 0; k < nOp; ++k)
        ar[k] = accs[k][r];
      SmallVector<std::string> out;
      if (failed(emitCombineN(region, carry, ar, out)))
        return failure();
      for (int k = 0; k < nOp; ++k)
        os << ind() << accs[k][r] << " = (" << init << " ? " << out[k] << " : "
           << accs[k][r] << ");\n";
    }
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    return success();
  }

  LogicalResult emitScan(tt::ScanOp op) {
    bool rev = op.getReverse();
    int nOp = op.getNumOperands();
    auto srcTy = cast<RankedTensorType>(op.getOperand(0).getType());
    int axis = op.getAxis();
    Region &region = op.getRegion();

    SmallVector<std::string> scTys(nOp);
    SmallVector<int64_t> byteWidths(nOp);
    for (int k = 0; k < nOp; ++k) {
      scTys[k] = mslScalarType(elementScalarType(op.getResult()[k].getType()));
      byteWidths[k] = bitsOf(elementScalarType(op.getResult()[k].getType())) / 8;
    }
    SmallVector<SmallVector<std::string>> srcNames(nOp);
    for (int k = 0; k < nOp; ++k)
      srcNames[k] = names(op.getOperand(k));
    int nReg = srcNames[0].size();

    MLIRContext *ctx = op.getContext();
    tt::LinearLayout ll = ttg::toLinearLayout(srcTy);
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto outDims = llvm::to_vector(ll.getOutDimNames());
    auto outDim = outDims[axis];

    auto laneBits = axisBits(ll, kLane, outDim);
    auto warpBits = axisBits(ll, kWarp, outDim);

    // The lane shuffle steps use raw-delta simd_shuffle_up/down and an
    // axis-local guard; that keeps every shuffle within one independent scan
    // only when the axis-carrying lane bits form a contiguous run (so
    // laneId & axisLaneMask is the axis lane field and no borrow crosses into a
    // non-axis lane bit). Bail loud on any other lane layout.
    unsigned axisLaneMask = 0;
    for (auto &pr : laneBits)
      axisLaneMask |= (1u << pr.first);
    unsigned axisLaneLow = axisLaneMask & (~axisLaneMask + 1);
    unsigned normMask = axisLaneMask / (axisLaneLow ? axisLaneLow : 1);
    if (axisLaneMask && (normMask & (normMask + 1))) {
      op.emitError("EmitMSL: unsupported scan lane layout");
      return failure();
    }
    unsigned axisWarpMask = 0;
    for (auto &pr : warpBits)
      axisWarpMask |= (1u << pr.first);
    int numWarps = ll.hasInDim(kWarp) ? ll.getInDimSize(kWarp) : 1;

    // Largest lane/warp axis stride: register bits above it are OUTER scan
    // dimensions that straddle the lane/warp span, so registers group into
    // coord-contiguous runs and the cross-run carry runs after lane/warp.
    int32_t laneWarpReach = 0;
    for (auto &pr : laneBits)
      laneWarpReach = std::max(laneWarpReach, pr.second);
    for (auto &pr : warpBits)
      laneWarpReach = std::max(laneWarpReach, pr.second);

    // Independent-scan key: register coordinates on every non-axis out-dim.
    // Registers with different keys are distinct scans and must never combine.
    auto keyOf = [&](int reg) {
      SmallVector<int32_t> coords = registerCoords(srcTy, reg);
      std::string key;
      for (int d = 0; d < (int)coords.size(); ++d)
        if (d != axis)
          key += std::to_string(coords[d]) + ",";
      return key;
    };
    SmallVector<int> runId(nReg, 0);
    for (int r = 0; r < nReg; ++r) {
      int32_t c = registerCoords(srcTy, r)[axis];
      runId[r] = laneWarpReach ? (c / (2 * laneWarpReach)) : 0;
    }

    SmallVector<SmallVector<std::string>> accs(nOp,
                                              SmallVector<std::string>(nReg));
    for (int k = 0; k < nOp; ++k)
      for (int r = 0; r < nReg; ++r) {
        accs[k][r] = fresh();
        os << ind() << scTys[k] << " " << accs[k][r] << " = "
           << srcNames[k][r] << ";\n";
      }

    const char *shuf = rev ? "simd_shuffle_down" : "simd_shuffle_up";
    std::string axisTopLane =
        axisLaneMask == 0
            ? laneId
            : ("((" + laneId + " & " + std::to_string(~axisLaneMask) + ") | " +
               (rev ? "0" : std::to_string(axisLaneMask)) + ")");

    std::map<std::string, SmallVector<int>> keys;
    SmallVector<std::string> keyOrder;
    for (int r = 0; r < nReg; ++r) {
      std::string k = keyOf(r);
      if (keys.find(k) == keys.end())
        keyOrder.push_back(k);
      keys[k].push_back(r);
    }

    for (std::string &key : keyOrder) {
      SmallVector<int> &keyRegs = keys[key];
      SmallVector<int> runOrder;
      for (int r : keyRegs)
        if (llvm::find(runOrder, runId[r]) == runOrder.end())
          runOrder.push_back(runId[r]);
      llvm::sort(runOrder, [&](int a, int b) { return rev ? a > b : a < b; });

      SmallVector<SmallVector<std::string>> runTotals(
          runOrder.size(), SmallVector<std::string>(nOp));

      for (size_t ri = 0; ri < runOrder.size(); ++ri) {
        int run = runOrder[ri];
        SmallVector<int> regs;
        for (int r : keyRegs)
          if (runId[r] == run)
            regs.push_back(r);
        llvm::sort(regs, [&](int a, int b) {
          int32_t ca = registerCoords(srcTy, a)[axis];
          int32_t cb = registerCoords(srcTy, b)[axis];
          return rev ? ca > cb : ca < cb;
        });

        for (size_t i = 1; i < regs.size(); ++i) {
          SmallVector<std::string> a(nOp), b(nOp);
          for (int k = 0; k < nOp; ++k) {
            a[k] = accs[k][regs[i - 1]];
            b[k] = accs[k][regs[i]];
          }
          SmallVector<std::string> out;
          if (failed(emitCombineN(region, a, b, out)))
            return failure();
          for (int k = 0; k < nOp; ++k)
            os << ind() << accs[k][regs[i]] << " = " << out[k] << ";\n";
        }

        SmallVector<std::string> laneScan(nOp);
        for (int k = 0; k < nOp; ++k) {
          laneScan[k] = fresh();
          os << ind() << scTys[k] << " " << laneScan[k] << " = "
             << accs[k][regs.back()] << ";\n";
        }
        for (auto &pr : laneBits) {
          unsigned delta = 1u << pr.first;
          SmallVector<std::string> nb(nOp);
          for (int k = 0; k < nOp; ++k)
            nb[k] = emitShuffle(shuf, scTys[k], laneScan[k],
                                std::to_string(delta) + "u");
          SmallVector<std::string> out;
          if (failed(emitCombineN(region, nb, laneScan, out)))
            return failure();
          std::string local = "(" + laneId + " & " +
                              std::to_string(axisLaneMask) + ")";
          std::string guard =
              rev ? (local + " <= " + std::to_string(axisLaneMask - delta))
                  : (local + " >= " + std::to_string(delta));
          for (int k = 0; k < nOp; ++k)
            os << ind() << laneScan[k] << " = (" << guard << " ? " << out[k]
               << " : " << laneScan[k] << ");\n";
        }
        if (!laneBits.empty()) {
          SmallVector<std::string> lanePrefix(nOp);
          for (int k = 0; k < nOp; ++k)
            lanePrefix[k] = emitShuffle(shuf, scTys[k], laneScan[k],
                                        std::to_string(axisLaneLow) + "u");
          std::string local =
              "(" + laneId + " & " + std::to_string(axisLaneMask) + ")";
          std::string guard =
              rev ? (local + " <= " + std::to_string(axisLaneMask - axisLaneLow))
                  : (local + " >= " + std::to_string(axisLaneLow));
          for (int r : regs) {
            SmallVector<std::string> out;
            SmallVector<std::string> ar(nOp);
            for (int k = 0; k < nOp; ++k)
              ar[k] = accs[k][r];
            if (failed(emitCombineN(region, lanePrefix, ar, out)))
              return failure();
            for (int k = 0; k < nOp; ++k)
              os << ind() << accs[k][r] << " = (" << guard << " ? " << out[k]
                 << " : " << accs[k][r] << ");\n";
          }
        }

        for (int k = 0; k < nOp; ++k) {
          runTotals[ri][k] = fresh();
          os << ind() << scTys[k] << " " << runTotals[ri][k] << ";\n";
        }
        if (failed(emitScanWarpCarry(region, nOp, scTys, byteWidths, warpBits,
                                     regs, accs, laneScan, axisTopLane,
                                     axisWarpMask, numWarps, rev,
                                     runTotals[ri])))
          return failure();
      }

      for (size_t ri = 1; ri < runOrder.size(); ++ri) {
        SmallVector<std::string> carry = runTotals[0];
        for (size_t j = 1; j < ri; ++j) {
          SmallVector<std::string> out;
          if (failed(emitCombineN(region, carry, runTotals[j], out)))
            return failure();
          carry = out;
        }
        int run = runOrder[ri];
        for (int r : keyRegs) {
          if (runId[r] != run)
            continue;
          SmallVector<std::string> ar(nOp);
          for (int k = 0; k < nOp; ++k)
            ar[k] = accs[k][r];
          SmallVector<std::string> out;
          if (failed(emitCombineN(region, carry, ar, out)))
            return failure();
          for (int k = 0; k < nOp; ++k)
            os << ind() << accs[k][r] << " = " << out[k] << ";\n";
        }
      }
    }

    for (int k = 0; k < nOp; ++k)
      valMap[op.getResult()[k]] = accs[k];
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
  int64_t globalPoolBytes = 0;
  int64_t liveTgBytes = 0;
  bool moduleHasDevFuncs = false;

  // Threadgroup budget left for the reused pool after the always-live
  // local_alloc buffers, which coexist with the pool in kernel scope.
  int64_t poolBudget() const {
    int64_t b = 32768 - liveTgBytes;
    return b < 0 ? 0 : b;
  }

  llvm::DenseMap<Block *, std::string> blockLabel;
  std::string cfgState;

  static int64_t bitsOf(Type t) {
    if (isa<tt::PointerType>(t))
      return 64;
    int64_t bits = t.getIntOrFloatBitWidth();
    return bits == 1 ? 8 : bits;
  }

  std::string poolRegion(int64_t byteOffset, StringRef sc) {
    std::string base = byteOffset == 0
                           ? poolBuf
                           : "(" + poolBuf + " + " + std::to_string(byteOffset) +
                                 ")";
    return "((threadgroup " + sc.str() + "*)" + base + ")";
  }

  // Elements per band for a threadgroup-staged reshape whose full tile exceeds
  // the 32KB budget: the largest chunk of flat offsets that fits.
  static int64_t reshapeBandElems(int64_t totalElems, int64_t elemBytes,
                                  int64_t budget = 32768) {
    int64_t cap = budget / elemBytes;
    if (cap < 1)
      cap = 1;
    int64_t nBands = (totalElems + cap - 1) / cap;
    return (totalElems + nBands - 1) / nBands;
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
      if (convertLayoutIsDeadDotStage(c) || convertLayoutIsDeadDotStageSource(c))
        return;
      auto st = cast<RankedTensorType>(c.getSrc().getType());
      Type e = st.getElementType();
      int64_t elemBytes = bitsOf(e) / 8;
      int64_t bytes = tileSize(st) * elemBytes;
      int rk = st.getRank();
      int64_t cap = poolBudget();
      if (bytes > cap && rk >= 2) {
        int64_t N = st.getShape()[rk - 1];
        int64_t bandRows = cap / (N * elemBytes);
        if (bandRows < 1)
          bandRows = 1;
        bytes = bandRows * N * elemBytes;
      } else if (bytes > cap) {
        bytes = reshapeBandElems(tileSize(st), elemBytes, cap) * elemBytes;
      }
      poolBytes = std::max(poolBytes, bytes);
    } else if (auto t = dyn_cast<tt::TransOp>(op)) {
      auto rt = cast<RankedTensorType>(t.getResult().getType());
      Type e = rt.getElementType();
      int64_t elemBytes = bitsOf(e) / 8;
      int64_t bytes = tileSize(rt) * elemBytes;
      if (bytes > 32768)
        bytes = reshapeBandElems(tileSize(rt), elemBytes) * elemBytes;
      poolBytes = std::max(poolBytes, bytes);
    } else if (auto c = dyn_cast<tt::CatOp>(op)) {
      auto rt = cast<RankedTensorType>(c.getResult().getType());
      Type e = rt.getElementType();
      poolBytes = std::max(poolBytes, tileSize(rt) * (bitsOf(e) / 8));
    } else if (auto rs = dyn_cast<tt::ReshapeOp>(op)) {
      auto rt = cast<RankedTensorType>(rs.getResult().getType());
      Type e = rt.getElementType();
      int64_t elemBytes = bitsOf(e) / 8;
      int64_t bytes = tileSize(rt) * elemBytes;
      if (bytes > 32768)
        bytes = reshapeBandElems(tileSize(rt), elemBytes) * elemBytes;
      poolBytes = std::max(poolBytes, bytes);
    } else if (auto g = dyn_cast<tt::GatherOp>(op)) {
      auto st = cast<RankedTensorType>(g.getSrc().getType());
      Type e = st.getElementType();
      poolBytes = std::max(poolBytes, tileSize(st) * (bitsOf(e) / 8));
    } else if (auto d = dyn_cast<tt::DotOp>(op)) {
      auto aTy = cast<RankedTensorType>(d.getA().getType());
      auto bTy = cast<RankedTensorType>(d.getB().getType());
      auto cTy = cast<RankedTensorType>(d.getResult().getType());
      int rk = cTy.getRank();
      int64_t M = cTy.getShape()[rk - 2];
      int64_t N = cTy.getShape()[rk - 1];
      int64_t Kd = aTy.getShape()[rk - 1];
      int64_t aBy = M * Kd * (bitsOf(aTy.getElementType()) / 8);
      int64_t bBy = Kd * N * (bitsOf(bTy.getElementType()) / 8);
      Type cE = cTy.getElementType();
      int64_t need;
      if (isa<IntegerType>(cE)) {
        need = aBy + bBy;
      } else {
        int64_t accBytes = 4;
        int64_t elemBytes = bitsOf(aTy.getElementType()) / 8;
        int64_t stagedA = aBy, stagedB = bBy;
        if (rk == 2 && aBy + bBy <= 32768) {
          if (dotOperandLocalLoad(d.getA(), M, Kd))
            stagedA = 0;
          if (dotOperandLocalLoad(d.getB(), Kd, N))
            stagedB = 0;
        }
        int64_t stagedAB = stagedA + stagedB;
        int64_t cFull = M * N * accBytes;
        if (stagedAB == aBy + bBy && dotNeedsPanel(M, N, Kd, elemBytes, accBytes)) {
          int64_t mp, np;
          dotPanelDims(M, N, Kd, elemBytes, accBytes, mp, np);
          need = mp * Kd * elemBytes + Kd * np * elemBytes + mp * np * accBytes;
        } else if (stagedAB + cFull <= poolBudget()) {
          need = stagedAB + cFull;
        } else {
          int64_t band = dotCBandRows(M, N, poolBudget(), accBytes);
          need = std::max(stagedAB, band * N * accBytes);
        }
      }
      poolBytes = std::max(poolBytes, need);
    } else if (auto r = dyn_cast<tt::ReduceOp>(op)) {
      auto st = cast<RankedTensorType>(r.getOperand(0).getType());
      tt::LinearLayout ll = ttg::toLinearLayout(st);
      auto kWarp = StringAttr::get(op->getContext(), "warp");
      if (ll.hasInDim(kWarp)) {
        int64_t nw = ll.getInDimSize(kWarp);
        int64_t bytes = 0;
        for (Value res : r.getResult())
          bytes += nw * 32 *
                   std::max<int64_t>(
                       1, bitsOf(elementScalarType(res.getType())) / 8);
        poolBytes = std::max(poolBytes, bytes);
      }
    } else if (auto h = dyn_cast<tt::HistogramOp>(op)) {
      auto rt = cast<RankedTensorType>(h.getResult().getType());
      poolBytes = std::max(poolBytes, tileSize(rt) * 4);
    } else if (auto ca = dyn_cast<tt::AtomicCASOp>(op)) {
      if (!isa<RankedTensorType>(ca.getPtr().getType()))
        poolBytes = std::max<int64_t>(poolBytes, 8);
    } else if (auto ar = dyn_cast<tt::AtomicRMWOp>(op)) {
      if (auto ptrTy = dyn_cast<RankedTensorType>(ar.getPtr().getType())) {
        tt::LinearLayout ll = ttg::toLinearLayout(ptrTy);
        MLIRContext *c = op->getContext();
        unsigned warpFree =
            ll.getFreeVariableMasks().lookup(StringAttr::get(c, "warp"));
        if (warpFree) {
          int64_t eb = std::max<int64_t>(
              1, bitsOf(elementScalarType(ar.getResult().getType())) / 8);
          int64_t rc = ll.getInDimSize(StringAttr::get(c, "register"));
          int64_t nw = ll.hasInDim(StringAttr::get(c, "warp"))
                           ? ll.getInDimSize(StringAttr::get(c, "warp"))
                           : 1;
          poolBytes = std::max<int64_t>(poolBytes, rc * 32 * nw * eb);
        }
      }
    } else if (auto s = dyn_cast<tt::ScanOp>(op)) {
      auto st = cast<RankedTensorType>(s.getOperand(0).getType());
      tt::LinearLayout ll = ttg::toLinearLayout(st);
      auto kWarp = StringAttr::get(op->getContext(), "warp");
      auto outDims = llvm::to_vector(ll.getOutDimNames());
      auto outDim = outDims[s.getAxis()];
      if (!axisBits(ll, kWarp, outDim).empty()) {
        int64_t nw = ll.getInDimSize(kWarp);
        int64_t bytes = 0;
        for (Value res : s.getResult())
          bytes += nw * 32 * (bitsOf(elementScalarType(res.getType())) / 8);
        poolBytes = std::max(poolBytes, bytes);
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

  // Row-major flat offset of register r within its batch slice: the row-major
  // offset over the trailing (rank-1) out-dims only, dropping the leading batch
  // dim. Used to index a single per-batch staging region reused across slices.
  std::string sliceFlatOffset(RankedTensorType rt, int reg) {
    tt::LinearLayout ll = ttg::toLinearLayout(rt);
    auto outNames = llvm::to_vector(ll.getOutDimNames());
    auto shape = rt.getShape();
    int lo = std::max<int>(0, (int)outNames.size() - 2);
    std::string expr;
    int64_t stride = 1;
    for (int d = (int)outNames.size() - 1; d >= lo; --d) {
      std::string c = layoutCoordExpr(rt, reg, outNames[d]);
      std::string term = stride == 1 ? c : ("(" + c + " * " +
                                            std::to_string(stride) + ")");
      expr = expr.empty() ? term : ("(" + expr + " + " + term + ")");
      stride *= shape[d];
    }
    return expr.empty() ? "0" : expr;
  }

  std::string batchCoordExpr(RankedTensorType rt, int reg) {
    tt::LinearLayout ll = ttg::toLinearLayout(rt);
    auto outNames = llvm::to_vector(ll.getOutDimNames());
    return layoutCoordExpr(rt, reg, outNames[0]);
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

    int srcRc = regCount(src);
    int resRc = regCount(res);
    int64_t elemBytes = bitsOf(resTy.getElementType()) / 8;
    int64_t total = tileSize(resTy);

    std::string buf = fresh();
    os << ind() << "threadgroup " << sc << "* " << buf << " = "
       << poolRegion(0, sc) << ";\n";

    int64_t band = total * elemBytes > 32768
                       ? reshapeBandElems(total, elemBytes)
                       : total;

    SmallVector<std::string> ids = declResultVars(res, StringRef());

    for (int64_t lo = 0; lo < total; lo += band) {
      int64_t hi = std::min(lo + band, total);
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      for (int r = 0; r < srcRc; ++r) {
        std::string off = transFlatOffset(srcTy, perm, resTy.getShape(), r);
        const std::string &sv = srcNames[srcNames.size() == 1 ? 0 : r];
        if (band == total)
          os << ind() << buf << "[" << off << "] = " << sv << ";\n";
        else
          os << ind() << "{ int __f = " << off << "; if (__f >= " << lo
             << " && __f < " << hi << ") " << buf << "[__f - " << lo
             << "] = " << sv << "; }\n";
      }
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      for (int r = 0; r < resRc; ++r) {
        std::string off = flatTileOffset(resTy, r);
        if (band == total)
          os << ind() << ids[r] << " = " << buf << "[" << off << "];\n";
        else
          os << ind() << "{ int __f = " << off << "; if (__f >= " << lo
             << " && __f < " << hi << ") " << ids[r] << " = " << buf
             << "[__f - " << lo << "]; }\n";
      }
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
    int64_t elemBytes = bitsOf(resTy.getElementType()) / 8;
    int64_t total = tileSize(resTy);

    std::string buf = fresh();
    os << ind() << "threadgroup " << sc << "* " << buf << " = "
       << poolRegion(0, sc) << ";\n";

    // A reshape is a flat-offset identity: result flat offset f reads the src
    // element written at flat offset f. When the full fp tile exceeds the 32KB
    // budget, stage it in flat-offset bands; each register belongs to exactly
    // one band, so guarding write/read by band keeps the round-trip correct.
    int64_t band = total * elemBytes > 32768
                       ? reshapeBandElems(total, elemBytes)
                       : total;

    SmallVector<std::string> outs = declResultVars(res, StringRef());

    for (int64_t lo = 0; lo < total; lo += band) {
      int64_t hi = std::min(lo + band, total);
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      for (int r = 0; r < srcRc; ++r) {
        std::string off = flatTileOffset(srcTy, r);
        const std::string &sv = srcNames[srcNames.size() == 1 ? 0 : r];
        if (band == total)
          os << ind() << buf << "[" << off << "] = " << sv << ";\n";
        else
          os << ind() << "{ int __f = " << off << "; if (__f >= " << lo
             << " && __f < " << hi << ") " << buf << "[__f - " << lo
             << "] = " << sv << "; }\n";
      }
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      for (int r = 0; r < resRc; ++r) {
        std::string off = flatTileOffset(resTy, r);
        if (band == total)
          os << ind() << outs[r] << " = " << buf << "[" << off << "];\n";
        else
          os << ind() << "{ int __f = " << off << "; if (__f >= " << lo
             << " && __f < " << hi << ") " << outs[r] << " = " << buf
             << "[__f - " << lo << "]; }\n";
      }
    }
    valMap[res] = outs;
    return success();
  }

  // tt.gather %src[%idx] {axis}: out has idx's shape/layout;
  // out[coords] = src[coords with dim `axis` replaced by idx[coords]].
  // Stage the full src tile into threadgroup memory, then each thread reads its
  // owned output element at the index-selected source offset.
  LogicalResult emitGather(tt::GatherOp op) {
    Value src = op.getSrc();
    Value idx = op.getIndices();
    Value res = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    int axis = op.getAxis();
    std::string sc = mslScalarType(elementScalarType(resTy));
    auto &srcNames = names(src);
    auto &idxNames = names(idx);
    int srcRc = regCount(src);
    int resRc = regCount(res);

    auto srcShape = srcTy.getShape();
    tt::LinearLayout resLL = ttg::toLinearLayout(resTy);
    auto resOut = llvm::to_vector(resLL.getOutDimNames());

    std::string buf = fresh();
    os << ind() << "threadgroup " << sc << "* " << buf << " = "
       << poolRegion(0, sc) << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    for (int r = 0; r < srcRc; ++r)
      os << ind() << buf << "[" << flatTileOffset(srcTy, r)
         << "] = " << srcNames[srcNames.size() == 1 ? 0 : r] << ";\n";
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

    SmallVector<std::string> outs;
    for (int r = 0; r < resRc; ++r) {
      std::string off;
      int64_t stride = 1;
      for (int d = (int)srcShape.size() - 1; d >= 0; --d) {
        std::string c = (d == axis)
                            ? ("(int)(" + idxNames[idxNames.size() == 1 ? 0 : r] +
                               ")")
                            : layoutCoordExpr(resTy, r, resOut[d]);
        std::string term =
            stride == 1 ? c : ("(" + c + " * " + std::to_string(stride) + ")");
        off = off.empty() ? term : ("(" + off + " + " + term + ")");
        stride *= srcShape[d];
      }
      if (off.empty())
        off = "0";
      std::string id = fresh();
      os << ind() << sc << " " << id << " = " << buf << "[" << off << "];\n";
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
    if (convertLayoutIsDeadDotStage(op) ||
        convertLayoutIsDeadDotStageSource(op)) {
      valMap[op.getResult()] = SmallVector<std::string>{};
      return success();
    }
    Value src = op.getSrc();
    Value res = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    Type elemTy = resTy.getElementType();
    bool isPtr = isa<tt::PointerType>(elemTy);
    std::string ptrTy = mslStorageType(resTy);
    std::string sc = isPtr ? "ulong" : ptrTy;
    auto &srcNames = names(src);

    int64_t elemBytes = bitsOf(elemTy) / 8;
    int64_t tileBytes = tileSize(resTy) * elemBytes;
    int rank = resTy.getRank();
    ArrayRef<int64_t> shape = resTy.getShape();

    std::string bufptr = fresh();
    os << ind() << "threadgroup " << sc << "* " << bufptr << " = "
       << poolRegion(0, sc) << ";\n";
    std::string buf = bufptr;

    int64_t convCap = poolBudget();
    if (tileBytes > convCap && rank >= 2) {
      int64_t N = shape[rank - 1];
      int64_t bandRows = convCap / (N * elemBytes);
      if (bandRows < 1)
        bandRows = 1;
      int64_t rowsTotal = shape[rank - 2];
      tt::LinearLayout srcLL = ttg::toLinearLayout(srcTy);
      auto srcOut = llvm::to_vector(srcLL.getOutDimNames());
      StringAttr srcRowDim = srcOut[rank - 2];
      tt::LinearLayout resLL = ttg::toLinearLayout(resTy);
      auto resOut = llvm::to_vector(resLL.getOutDimNames());
      StringAttr resRowDim = resOut[rank - 2];
      SmallVector<std::string> ids(regCount(res));

      auto bandOffset = [&](RankedTensorType rt, int reg, int64_t r0) {
        auto outN = llvm::to_vector(ttg::toLinearLayout(rt).getOutDimNames());
        std::string expr;
        int64_t stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
          std::string c = layoutCoordExpr(rt, reg, outN[d]);
          if (d == rank - 2)
            c = "(" + c + " - " + std::to_string(r0) + ")";
          std::string term = stride == 1
                                 ? c
                                 : ("(" + c + " * " + std::to_string(stride) +
                                    ")");
          expr = expr.empty() ? term : ("(" + expr + " + " + term + ")");
          stride *= (d == rank - 2) ? bandRows : shape[d];
        }
        return expr.empty() ? std::string("0") : expr;
      };

      for (int64_t r0 = 0; r0 < rowsTotal; r0 += bandRows) {
        int64_t r1 = std::min<int64_t>(r0 + bandRows, rowsTotal);
        os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        for (int r = 0, n = regCount(src); r < n; ++r) {
          std::string rowc = layoutCoordExpr(srcTy, r, srcRowDim);
          std::string sv = isPtr ? "(ulong)" + srcNames[r] : srcNames[r];
          os << ind() << "if (" << rowc << " >= " << r0 << " && " << rowc
             << " < " << r1 << ") " << buf << "[" << bandOffset(srcTy, r, r0)
             << "] = " << sv << ";\n";
        }
        os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        for (int r = 0, n = regCount(res); r < n; ++r) {
          std::string rowc = layoutCoordExpr(resTy, r, resRowDim);
          std::string rd = buf + "[" + bandOffset(resTy, r, r0) + "]";
          if (isPtr)
            rd = "(" + ptrTy + ")" + rd;
          if (ids[r].empty()) {
            ids[r] = fresh();
            os << ind() << ptrTy << " " << ids[r] << ";\n";
          }
          os << ind() << "if (" << rowc << " >= " << r0 << " && " << rowc
             << " < " << r1 << ") " << ids[r] << " = " << rd << ";\n";
        }
      }
      valMap[res] = ids;
      return success();
    }

    // Over-budget tile with no bandable row dim (e.g. rank-1): band by flat
    // offset. A convert_layout permutes the same logical tensor, so result
    // register at flat offset f reads the src element written at flat offset f;
    // guarding write/read by band keeps the round-trip correct at any rank.
    if (tileBytes > convCap) {
      int64_t total = tileSize(resTy);
      int64_t band = reshapeBandElems(total, elemBytes, convCap);
      SmallVector<std::string> ids(regCount(res));
      for (int r = 0, n = regCount(res); r < n; ++r) {
        ids[r] = fresh();
        os << ind() << ptrTy << " " << ids[r] << ";\n";
      }
      for (int64_t lo = 0; lo < total; lo += band) {
        int64_t hi = std::min(lo + band, total);
        os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        for (int r = 0, n = regCount(src); r < n; ++r) {
          std::string sv = isPtr ? "(ulong)" + srcNames[r] : srcNames[r];
          os << ind() << "{ int __f = " << flatTileOffset(srcTy, r)
             << "; if (__f >= " << lo << " && __f < " << hi << ") " << buf
             << "[__f - " << lo << "] = " << sv << "; }\n";
        }
        os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        for (int r = 0, n = regCount(res); r < n; ++r) {
          std::string rd = buf + std::string("[__f - ") + std::to_string(lo) +
                           "]";
          if (isPtr)
            rd = "(" + ptrTy + ")" + rd;
          os << ind() << "{ int __f = " << flatTileOffset(resTy, r)
             << "; if (__f >= " << lo << " && __f < " << hi << ") " << ids[r]
             << " = " << rd << "; }\n";
        }
      }
      valMap[res] = ids;
      return success();
    }

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
    if (Value init = op.getSrc()) {
      auto srcTy = cast<RankedTensorType>(init.getType());
      auto &vals = names(init);
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      for (int r = 0, n = regCount(init); r < n; ++r)
        os << ind() << buf << "[" << flatTileOffset(srcTy, r)
           << "] = " << vals[r] << ";\n";
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    }
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

  LogicalResult emitMemDescSubslice(ttg::MemDescSubsliceOp op) {
    auto srcMt = cast<ttg::MemDescType>(op.getSrc().getType());
    MemDescInfo parent = memdescMap[op.getSrc()];
    ArrayRef<int64_t> srcShape = srcMt.getShape();
    ArrayRef<int32_t> offsets = op.getOffsets();
    if (offsets.size() != srcShape.size())
      return op.emitError("EmitMSL: memdesc_subslice rank mismatch");

    SmallVector<int64_t> strides(srcShape.size());
    if (!parent.bufStrides.empty()) {
      strides.assign(parent.bufStrides.begin(), parent.bufStrides.end());
    } else {
      int64_t s = 1;
      for (int d = (int)srcShape.size() - 1; d >= 0; --d) {
        strides[d] = s;
        s *= srcShape[d];
      }
    }

    int64_t constOff = 0;
    for (int d = 0; d < (int)offsets.size(); ++d)
      constOff += (int64_t)offsets[d] * strides[d];

    std::string base;
    if (parent.baseOffset == "0")
      base = std::to_string(constOff);
    else
      base = constOff == 0 ? parent.baseOffset
                           : ("(" + parent.baseOffset + " + " +
                              std::to_string(constOff) + ")");

    memdescMap[op.getResult()] = {parent.buf, base, strides};
    return success();
  }

  std::string memdescElemAddr(const MemDescInfo &info, RankedTensorType tileTy,
                              int reg) {
    std::string off;
    if (info.bufStrides.empty()) {
      off = flatTileOffset(tileTy, reg);
    } else {
      tt::LinearLayout ll = ttg::toLinearLayout(tileTy);
      auto outNames = llvm::to_vector(ll.getOutDimNames());
      for (int d = 0; d < (int)outNames.size(); ++d) {
        std::string c = layoutCoordExpr(tileTy, reg, outNames[d]);
        int64_t s = info.bufStrides[d];
        std::string term =
            s == 1 ? c : ("(" + c + " * " + std::to_string(s) + ")");
        off = off.empty() ? term : ("(" + off + " + " + term + ")");
      }
      if (off.empty())
        off = "0";
    }
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
    emitBarrier(/*device=*/false);
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
    emitBarrier(/*device=*/false);
    return success();
  }

  LogicalResult emitLocalLoad(ttg::LocalLoadOp op) {
    if (localLoadIsDeadDotStage(op)) {
      valMap[op.getResult()] = SmallVector<std::string>{};
      return success();
    }
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

  struct InPlaceOperand {
    std::string buf;
    std::string baseOffset;
  };

  // A local_load of a contiguous row-major [rows][cols] threadgroup buffer
  // (local_alloc, optionally memdesc_index'd, never subsliced) reached through
  // convert_layouts, or null.
  static ttg::LocalLoadOp dotOperandLocalLoad(Value operand, int64_t rows,
                                              int64_t cols) {
    Value v = operand;
    while (auto cvt = dyn_cast_or_null<ttg::ConvertLayoutOp>(v.getDefiningOp()))
      v = cvt.getSrc();
    auto ll = dyn_cast_or_null<ttg::LocalLoadOp>(v.getDefiningOp());
    if (!ll)
      return nullptr;
    auto mt = cast<ttg::MemDescType>(ll.getSrc().getType());
    if (mt.getRank() != 2 || mt.getShape()[0] != rows ||
        mt.getShape()[1] != cols)
      return nullptr;
    Value src = ll.getSrc();
    while (Operation *def = src.getDefiningOp()) {
      if (auto mi = dyn_cast<ttg::MemDescIndexOp>(def)) {
        src = mi.getSrc();
        continue;
      }
      if (isa<ttg::LocalAllocOp>(def))
        return ll;
      return nullptr;
    }
    return nullptr;
  }

  static bool dotReadsOperandInPlace(tt::DotOp d, Value operand) {
    auto cTy = cast<RankedTensorType>(d.getResult().getType());
    if (cTy.getRank() != 2)
      return false;
    int64_t M = cTy.getShape()[0], N = cTy.getShape()[1];
    int64_t Kd = cast<RankedTensorType>(d.getA().getType()).getShape()[1];
    int64_t aBy = M * Kd * (bitsOf(cast<RankedTensorType>(d.getA().getType())
                                       .getElementType()) /
                            8);
    int64_t bBy = Kd * N * (bitsOf(cast<RankedTensorType>(d.getB().getType())
                                       .getElementType()) /
                            8);
    if (aBy + bBy > 32768)
      return false;
    if (operand == d.getA())
      return dotOperandLocalLoad(operand, M, Kd);
    if (operand == d.getB())
      return dotOperandLocalLoad(operand, Kd, N);
    return false;
  }

  static bool convertLayoutIsDeadDotStage(ttg::ConvertLayoutOp c) {
    if (c.getResult().use_empty())
      return false;
    for (OpOperand &use : c.getResult().getUses()) {
      auto d = dyn_cast<tt::DotOp>(use.getOwner());
      if (!d || !dotReadsOperandInPlace(d, c.getResult()))
        return false;
    }
    return true;
  }

  static bool localLoadIsDeadDotStage(ttg::LocalLoadOp ll) {
    if (ll.getResult().use_empty())
      return false;
    for (OpOperand &use : ll.getResult().getUses()) {
      Operation *owner = use.getOwner();
      if (auto c = dyn_cast<ttg::ConvertLayoutOp>(owner)) {
        if (!convertLayoutIsDeadDotStage(c))
          return false;
        continue;
      }
      auto d = dyn_cast<tt::DotOp>(owner);
      if (!d || !dotReadsOperandInPlace(d, ll.getResult()))
        return false;
    }
    return true;
  }

  std::optional<InPlaceOperand>
  dotOperandInPlaceBuf(Value operand, int64_t rows, int64_t cols) {
    ttg::LocalLoadOp ll = dotOperandLocalLoad(operand, rows, cols);
    if (!ll)
      return std::nullopt;
    auto it = memdescMap.find(ll.getSrc());
    if (it == memdescMap.end() || !it->second.bufStrides.empty())
      return std::nullopt;
    return InPlaceOperand{it->second.buf, it->second.baseOffset};
  }

  static std::string inPlaceBase(const InPlaceOperand &op) {
    if (op.baseOffset == "0")
      return op.buf;
    return "(" + op.buf + " + " + op.baseOffset + ")";
  }

  // A dot operand reached through a convert_layout of a rank-2 distributed
  // tensor. The dot stages its operand into threadgroup memory by row-major
  // (m,k) offset regardless of the operand's distributed layout, so it can read
  // the CONVERT SOURCE registers at the source layout's row-major offsets and
  // produce a bit-identical tile, making the convert_layout's own threadgroup
  // round-trip dead weight. Returns the source value to stage, or null.
  static Value dotOperandConvertSource(tt::DotOp d, Value operand) {
    auto cvt = dyn_cast_or_null<ttg::ConvertLayoutOp>(operand.getDefiningOp());
    if (!cvt)
      return nullptr;
    Value src = cvt.getSrc();
    auto st = dyn_cast<RankedTensorType>(src.getType());
    if (!st || st.getRank() != 2)
      return nullptr;
    if (dotOperandLocalLoad(operand, st.getShape()[0], st.getShape()[1]))
      return nullptr;
    auto cTy = cast<RankedTensorType>(d.getResult().getType());
    if (cTy.getRank() != 2)
      return nullptr;
    return src;
  }

  // The convert_layout is dead when every use is a dot that will stage the
  // convert source directly (dotOperandConvertSource matches). Emitting its
  // round-trip is then pure waste.
  static bool convertLayoutIsDeadDotStageSource(ttg::ConvertLayoutOp c) {
    if (c.getResult().use_empty())
      return false;
    for (OpOperand &use : c.getResult().getUses()) {
      auto d = dyn_cast<tt::DotOp>(use.getOwner());
      if (!d)
        return false;
      if (use.get() != d.getA() && use.get() != d.getB())
        return false;
      auto cTy = cast<RankedTensorType>(d.getResult().getType());
      if (cTy.getRank() != 2)
        return false;
      if (!dotOperandConvertSource(d, use.get()))
        return false;
    }
    return true;
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

  // Row band (multiple of 8) for the C store/readback so its threadgroup
  // footprint never exceeds the A+B staging footprint that already sizes the
  // pool. The full MxN accumulator at acc width can dwarf A+B (e.g. float32
  // 128x128 C is 64KB vs 32KB A+B); banding the C round-trip over row groups
  // keeps a single reused region small enough to fit alongside A/B in 32KB.
  // When full A+B+C staging exceeds the 32KB threadgroup budget, walk the
  // output tile in (mp x np) panels (both multiples of 8) staging only
  // A[mp x K] + B[K x np] + C[mp x np] at a time, so peak live TG stays under
  // 32768. Picks the largest square-ish panel that fits.
  static void dotPanelDims(int64_t M, int64_t N, int64_t K, int64_t elemBytes,
                           int64_t accBytes, int64_t &mp, int64_t &np) {
    mp = M;
    np = N;
    auto fits = [&](int64_t m, int64_t n) {
      return m * K * elemBytes + K * n * elemBytes + m * n * accBytes <= 32768;
    };
    while (!fits(mp, np)) {
      if (mp >= np && mp > 8)
        mp -= 8;
      else if (np > 8)
        np -= 8;
      else if (mp > 8)
        mp -= 8;
      else
        break;
    }
  }

  static bool dotNeedsPanel(int64_t M, int64_t N, int64_t K, int64_t elemBytes,
                            int64_t accBytes) {
    return M * K * elemBytes + K * N * elemBytes > 32768;
  }

  static int64_t dotCBandRows(int64_t M, int64_t N, int64_t cBudget,
                              int64_t accBytes) {
    int64_t rowBytes = N * accBytes;
    int64_t band = cBudget / rowBytes;
    band -= band % 8;
    if (band < 8)
      band = 8;
    if (band > M)
      band = M;
    return band;
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
    int rank = aTy.getRank();
    if ((rank != 2 && rank != 3) || !isDotOperandElem(aElem) ||
        !isDotOperandElem(bElem) || aElem != bElem ||
        !(cElem.isF32() || cElem.isF16())) {
      op.emitError("EmitMSL: unsupported tt.dot operand/accumulator types");
      return failure();
    }
    int64_t B = rank == 3 ? cTy.getShape()[0] : 1;
    int64_t M = cTy.getShape()[rank - 2];
    int64_t N = cTy.getShape()[rank - 1];
    int64_t K = aTy.getShape()[rank - 1];
    if (M % 8 || N % 8 || K % 8) {
      op.emitError("EmitMSL: tt.dot tile dims must be multiples of 8");
      return failure();
    }

    auto &cInit = names(op.getC());

    std::string opScalar = sgOperandScalar(aElem);
    std::string accScalar = mslScalarType(cElem);
    std::string opFrag = sgFragType(aElem);
    std::string accFrag = "simdgroup_float8x8";

    int64_t aBytes = M * K * (bitsOf(aElem) / 8);
    int64_t bBytes = N * K * (bitsOf(bElem) / 8);
    int64_t accBytes = 4;
    int64_t cFull = M * N * accBytes;

    // When an operand already sits in a row-major threadgroup buffer (the
    // pipeliner's local_alloc), load fragments straight from it. Only the
    // whole-tile rank-2 path indexes a buffer simdgroup_load can address.
    std::optional<InPlaceOperand> aInPlace, bInPlace;
    bool wholeTileFits = M * K * (bitsOf(aElem) / 8) + bBytes <= 32768;
    if (rank == 2 && wholeTileFits) {
      aInPlace = dotOperandInPlaceBuf(op.getA(), M, K);
      bInPlace = dotOperandInPlaceBuf(op.getB(), K, N);
    }
    // When an operand is a folded convert (its round-trip was skipped), stage
    // the convert SOURCE registers at the source layout's row-major offsets:
    // the tile content is layout-independent, so this reproduces the exact
    // staged tile without the convert's threadgroup bounce.
    Value aStage = op.getA(), bStage = op.getB();
    if (!aInPlace)
      if (Value s = dotOperandConvertSource(op, op.getA()))
        aStage = s;
    if (!bInPlace)
      if (Value s = dotOperandConvertSource(op, op.getB()))
        bStage = s;
    auto aStageTy = cast<RankedTensorType>(aStage.getType());
    auto bStageTy = cast<RankedTensorType>(bStage.getType());
    auto &aNames = names(aStage);
    auto &bNames = names(bStage);

    int64_t stagedA = aInPlace ? 0 : aBytes;
    int64_t stagedB = bInPlace ? 0 : bBytes;
    int64_t stagedAB = stagedA + stagedB;

    // DISJOINT: C gets its own pool region past staged A/B; one accumulator is
    // live at a time (low register pressure). ALIASED: C reuses the pool and
    // the M×N store is banded to fit. DISJOINT when staged A+B+C fits the pool.
    bool disjointC = stagedAB + cFull <= poolBudget();
    int64_t bandRows = disjointC ? M : dotCBandRows(M, N, poolBudget(), accBytes);

    // In the fused GEMM path the A/B in-place base is loop-variant, so its
    // pointer decl must sit inside the loop (MMA phase); tgC is the static pool
    // region. Decl declares only the persistent frags; Readback touches only tgC.
    FusedDotPhase phase = fusedDot.phase;
    bool needAB = phase == FusedDotPhase::None || phase == FusedDotPhase::MMA;
    bool needC = phase != FusedDotPhase::Decl;
    std::string tgA = fresh(), tgB = fresh(), tgC = fresh();
    if (needAB) {
      if (aInPlace)
        os << ind() << "threadgroup " << opScalar << "* " << tgA << " = "
           << inPlaceBase(*aInPlace) << ";\n";
      else
        os << ind() << "threadgroup " << opScalar << "* " << tgA << " = "
           << poolRegion(0, opScalar) << ";\n";
      if (bInPlace)
        os << ind() << "threadgroup " << opScalar << "* " << tgB << " = "
           << inPlaceBase(*bInPlace) << ";\n";
      else
        os << ind() << "threadgroup " << opScalar << "* " << tgB << " = "
           << poolRegion(stagedA, opScalar) << ";\n";
    }
    if (needC)
      os << ind() << "threadgroup float* " << tgC << " = "
         << poolRegion(disjointC ? stagedAB : 0, "float") << ";\n";

    int64_t mT = M / 8, nT = N / 8, kT = K / 8;
    tt::LinearLayout cLL = ttg::toLinearLayout(cTy);
    auto kWarpDim = StringAttr::get(op.getContext(), "warp");
    int64_t numWarps = cLL.hasInDim(kWarpDim) ? cLL.getInDimSize(kWarpDim) : 1;
    int64_t nFrag = mT * nT;
    if (numWarps > nFrag)
      numWarps = nFrag;

    auto batchGuard = [&](RankedTensorType rt, int reg, int64_t bi)
        -> std::string {
      if (rank != 3)
        return "";
      return "if (" + batchCoordExpr(rt, reg) + " == " + std::to_string(bi) +
             ") ";
    };

    int nRes = regCount(op.getResult());
    bool fused = phase != FusedDotPhase::None;
    SmallVector<std::string> ids(nRes);
    if (fused) {
      ids = fusedDot.ids;
    } else {
      for (int r = 0; r < nRes; ++r) {
        ids[r] = fresh();
        os << ind() << accScalar << " " << ids[r] << " = ("
           << accScalar << ")0;\n";
      }
    }

    auto outNames = llvm::to_vector(cLL.getOutDimNames());
    StringAttr rowDim = outNames[rank - 2], colDim = outNames[rank - 1];

    if (dotNeedsPanel(M, N, K, bitsOf(aElem) / 8, accBytes)) {
      int64_t elemBytes = bitsOf(aElem) / 8;
      int64_t mp, np;
      dotPanelDims(M, N, K, elemBytes, accBytes, mp, np);
      int64_t aPanelBytes = mp * K * elemBytes;
      int64_t bPanelBytes = K * np * elemBytes;
      tt::LinearLayout aLL = ttg::toLinearLayout(aStageTy);
      auto aOut = llvm::to_vector(aLL.getOutDimNames());
      StringAttr aRowDim = aOut[rank - 2], aColDim = aOut[rank - 1];
      tt::LinearLayout bLL = ttg::toLinearLayout(bStageTy);
      auto bOut = llvm::to_vector(bLL.getOutDimNames());
      StringAttr bColDim = bOut[rank - 1], bRowDim = bOut[rank - 2];

      std::string pA = fresh(), pB = fresh(), pC = fresh();
      os << ind() << "threadgroup " << opScalar << "* " << pA << " = "
         << poolRegion(0, opScalar) << ";\n";
      os << ind() << "threadgroup " << opScalar << "* " << pB << " = "
         << poolRegion(aPanelBytes, opScalar) << ";\n";
      os << ind() << "threadgroup float* " << pC << " = "
         << poolRegion(aPanelBytes + bPanelBytes, "float") << ";\n";

      int nARegs = regCount(aStage), nBRegs = regCount(bStage);
      for (int64_t bi = 0; bi < B; ++bi) {
        for (int64_t m0 = 0; m0 < M; m0 += mp) {
          int64_t m1 = std::min<int64_t>(m0 + mp, M);
          int64_t mpCur = m1 - m0;

          os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
          for (int r = 0; r < nARegs; ++r) {
            std::string row = layoutCoordExpr(aStageTy, r, aRowDim);
            std::string col = layoutCoordExpr(aStageTy, r, aColDim);
            std::string guard = "(" + row + " >= " + std::to_string(m0) +
                                " && " + row + " < " + std::to_string(m1) + ")";
            if (rank == 3)
              guard = "(" + batchCoordExpr(aStageTy, r) + " == " +
                      std::to_string(bi) + " && " + guard + ")";
            std::string off = "((" + row + " - " + std::to_string(m0) +
                              ") * " + std::to_string(K) + " + " + col + ")";
            os << ind() << "if " << guard << " " << pA << "[" << off
               << "] = " << aNames[r] << ";\n";
          }

          for (int64_t n0 = 0; n0 < N; n0 += np) {
            int64_t n1 = std::min<int64_t>(n0 + np, N);
            int64_t npCur = n1 - n0;
            int64_t pmT = mpCur / 8, pnT = npCur / 8;

            os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
            for (int r = 0; r < nBRegs; ++r) {
              std::string col = layoutCoordExpr(bStageTy, r, bColDim);
              std::string row = layoutCoordExpr(bStageTy, r, bRowDim);
              std::string guard = "(" + col + " >= " + std::to_string(n0) +
                                  " && " + col + " < " + std::to_string(n1) +
                                  ")";
              if (rank == 3)
                guard = "(" + batchCoordExpr(bStageTy, r) + " == " +
                        std::to_string(bi) + " && " + guard + ")";
              std::string off = "(" + row + " * " + std::to_string(npCur) +
                                " + (" + col + " - " + std::to_string(n0) +
                                "))";
              os << ind() << "if " << guard << " " << pB << "[" << off
                 << "] = " << bNames[r] << ";\n";
            }
            os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

            int64_t pnFrag = pmT * pnT;
            int64_t pWarps = numWarps > pnFrag ? pnFrag : numWarps;
            for (int64_t w = 0; w < pWarps; ++w) {
              os << ind() << "if (" << warpId << " == " << w << ") {\n";
              ++indent;
              for (int64_t f = w; f < pnFrag; f += pWarps) {
                int64_t mi = f / pnT, ni = f % pnT;
                std::string acc = fresh();
                os << ind() << accFrag << " " << acc << " = " << accFrag
                   << "(0.0f);\n";
                for (int64_t ki = 0; ki < kT; ++ki) {
                  std::string fa = fresh(), fb = fresh();
                  os << ind() << opFrag << " " << fa << ";\n";
                  os << ind() << "simdgroup_load(" << fa << ", " << pA << " + "
                     << (mi * 8 * K + ki * 8) << ", " << K << ");\n";
                  os << ind() << opFrag << " " << fb << ";\n";
                  os << ind() << "simdgroup_load(" << fb << ", " << pB << " + "
                     << (ki * 8 * npCur + ni * 8) << ", " << npCur << ");\n";
                  os << ind() << "simdgroup_multiply_accumulate(" << acc << ", "
                     << fa << ", " << fb << ", " << acc << ");\n";
                }
                os << ind() << "simdgroup_store(" << acc << ", " << pC << " + "
                   << (mi * 8 * npCur + ni * 8) << ", " << npCur << ");\n";
              }
              --indent;
              os << ind() << "}\n";
            }
            os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

            for (int r = 0; r < nRes; ++r) {
              std::string base = cInit[cInit.size() == 1 ? 0 : r];
              std::string rowExpr = layoutCoordExpr(cTy, r, rowDim);
              std::string colExpr = layoutCoordExpr(cTy, r, colDim);
              std::string off = "((" + rowExpr + " - " + std::to_string(m0) +
                                ") * " + std::to_string(npCur) + " + (" +
                                colExpr + " - " + std::to_string(n0) + "))";
              std::string guard = "(" + rowExpr + " >= " + std::to_string(m0) +
                                  " && " + rowExpr + " < " + std::to_string(m1) +
                                  " && " + colExpr +
                                  " >= " + std::to_string(n0) + " && " +
                                  colExpr + " < " + std::to_string(n1) + ")";
              if (rank == 3)
                guard = "(" + batchCoordExpr(cTy, r) + " == " +
                        std::to_string(bi) + " && " + guard + ")";
              os << ind() << "if " << guard << " " << ids[r] << " = " << pC
                 << "[" << off << "] + " << base << ";\n";
            }
          }
        }
      }
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      valMap[op.getResult()] = ids;
      return success();
    }

    ArrayRef<std::string> rbBase = fused ? ArrayRef<std::string>(fusedDot.baseNames)
                                         : ArrayRef<std::string>(cInit);
    auto emitReadback = [&](int64_t bi, int64_t r0, int64_t r1) {
      for (int r = 0; r < nRes; ++r) {
        std::string base = rbBase[rbBase.size() == 1 ? 0 : r];
        std::string rowExpr = layoutCoordExpr(cTy, r, rowDim);
        std::string colExpr = layoutCoordExpr(cTy, r, colDim);
        std::string bandOff = "((" + rowExpr + " - " + std::to_string(r0) +
                              ") * " + std::to_string(N) + " + " + colExpr +
                              ")";
        std::string guard = "(" + rowExpr + " >= " + std::to_string(r0) +
                            " && " + rowExpr + " < " + std::to_string(r1) + ")";
        if (rank == 3)
          guard = "(" + batchCoordExpr(cTy, r) + " == " + std::to_string(bi) +
                  " && " + guard + ")";
        os << ind() << "if " << guard << " " << ids[r] << " = " << tgC << "["
           << bandOff << "] + " << base << ";\n";
      }
    };

    auto emitFragMMAInto = [&](StringRef tgAn, StringRef tgBn, int64_t mi,
                               int64_t ni, StringRef acc) {
      for (int64_t ki = 0; ki < kT; ++ki) {
        std::string fa = fresh(), fb = fresh();
        os << ind() << opFrag << " " << fa << ";\n";
        os << ind() << "simdgroup_load(" << fa << ", " << tgAn << " + "
           << (mi * 8 * K + ki * 8) << ", " << K << ");\n";
        os << ind() << opFrag << " " << fb << ";\n";
        os << ind() << "simdgroup_load(" << fb << ", " << tgBn << " + "
           << (ki * 8 * N + ni * 8) << ", " << N << ");\n";
        os << ind() << "simdgroup_multiply_accumulate(" << acc << ", " << fa
           << ", " << fb << ", " << acc << ");\n";
      }
    };

    // Register-resident C fusion. The disjoint staging geometry is identical to
    // the per-dot path below; only the accumulator lifetime changes: the frags
    // are declared/zeroed once (Decl), accumulated each K-slab (MMA, no tgC
    // round-trip), then stored + gathered once (Readback, the same emitReadback
    // as the per-dot path, so the #mma->scalar extraction is bit-identical).
    if (fused) {
      // Frags are indexed by per-warp slot, not global tile: every warp shares
      // the same `fragsPerWarp` accumulator vars (the winner's layout), so a
      // thread holds only its own tiles' registers instead of all nFrag. Warp
      // w's slot j is global tile f = w + j*numWarps, which the store maps back
      // to the exact tgC offset the per-dot disjoint path uses (bit-exact
      // readback). Warps run their slots inside `if (warp == w)`.
      int64_t fragsPerWarp = (nFrag + numWarps - 1) / numWarps;
      if (fusedDot.phase == FusedDotPhase::Decl) {
        fusedDot.accNames.assign(fragsPerWarp, "");
        for (int64_t j = 0; j < fragsPerWarp; ++j) {
          std::string acc = fresh();
          fusedDot.accNames[j] = acc;
          os << ind() << accFrag << " " << acc << " = " << accFrag
             << "(0.0f);\n";
        }
        return success();
      }
      if (fusedDot.phase == FusedDotPhase::MMA) {
        bool stagesHere = !aInPlace || !bInPlace;
        if (stagesHere)
          os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        if (!aInPlace)
          for (int r = 0, n = regCount(aStage); r < n; ++r)
            os << ind() << tgA << "[" << sliceFlatOffset(aStageTy, r) << "] = "
               << aNames[r] << ";\n";
        if (!bInPlace)
          for (int r = 0, n = regCount(bStage); r < n; ++r)
            os << ind() << tgB << "[" << sliceFlatOffset(bStageTy, r) << "] = "
               << bNames[r] << ";\n";
        os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        // Slot j of warp w owns global fragment f = w + j*numWarps; its tile
        // coords are mi=f/nT, ni=f%nT. When numWarps == nT (one warp per N
        // column strip), mi=j is a per-slot compile-time constant and ni=warpId,
        // so every warp runs identical branchless code with warpId-derived
        // offsets. Otherwise fall back to the per-warp `if` ladder.
        bool branchless = (numWarps == nT);
        auto aOff = [&](int64_t mi, int64_t ki) {
          return std::to_string(mi * 8 * K + ki * 8);
        };
        auto bOff = [&](const std::string &niExpr, int64_t ki) {
          return "(" + std::to_string(ki * 8 * N) + " + " + niExpr + " * 8)";
        };
        auto emitSlots =
            [&](ArrayRef<std::pair<int64_t, std::string>> slots) {
              for (int64_t ki = 0; ki < kT; ++ki) {
                DenseMap<int64_t, std::string> aFrag;
                DenseMap<StringRef, std::string> bFrag;
                for (auto &[mi, niExpr] : slots) {
                  if (!aFrag.count(mi)) {
                    std::string fa = fresh();
                    aFrag[mi] = fa;
                    os << ind() << opFrag << " " << fa << ";\n";
                    os << ind() << "simdgroup_load(" << fa << ", " << tgA << " + "
                       << aOff(mi, ki) << ", " << K << ");\n";
                  }
                  if (!bFrag.count(niExpr)) {
                    std::string fb = fresh();
                    bFrag[niExpr] = fb;
                    os << ind() << opFrag << " " << fb << ";\n";
                    os << ind() << "simdgroup_load(" << fb << ", " << tgB << " + "
                       << bOff(niExpr, ki) << ", " << N << ");\n";
                  }
                }
                for (auto [j, mn] : llvm::enumerate(slots)) {
                  const std::string &acc = fusedDot.accNames[j];
                  os << ind() << "simdgroup_multiply_accumulate(" << acc << ", "
                     << aFrag[mn.first] << ", " << bFrag[mn.second] << ", " << acc
                     << ");\n";
                }
              }
            };
        if (branchless) {
          std::string niExpr = "(" + warpId + " % " + std::to_string(nT) + ")";
          SmallVector<std::pair<int64_t, std::string>> slots;
          for (int64_t j = 0; j * numWarps < nFrag; ++j) {
            int64_t base = j * numWarps;
            slots.push_back({base / nT, niExpr});
          }
          emitSlots(slots);
        } else {
          for (int64_t w = 0; w < numWarps; ++w) {
            os << ind() << "if (" << warpId << " == " << w << ") {\n";
            ++indent;
            SmallVector<std::pair<int64_t, std::string>> slots;
            for (int64_t f = w; f < nFrag; f += numWarps)
              slots.push_back({f / nT, std::to_string(f % nT)});
            emitSlots(slots);
            --indent;
            os << ind() << "}\n";
          }
        }
        // When this phase stages A/B into the pool, the next slab's leading
        // stage barrier (and the post-loop readback's leading barrier) already
        // fence these simdgroup_loads before any thread overwrites the pool, so
        // no trailing fence is needed. Only the fully in-place path (no staging
        // barrier next slab) needs an explicit fence here.
        if (!stagesHere)
          os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        valMap[op.getResult()] = ids;
        return success();
      }
      // Readback. When the accumulator flows only into a terminal row-major
      // store and the whole tile is in bounds, scatter each fragment straight
      // to device C (no pool, no swizzled gather). Otherwise (ragged tile, or
      // no direct-store match) fall back to the pool store + gather.
      if (fusedDot.direct) {
        const DirectStore &d = *fusedDot.direct;
        std::string base = names(d.basePtr)[0], ldc = names(d.ldc)[0];
        std::string rowB = names(d.rowBase)[0], colB = names(d.colBase)[0];
        os << ind() << "if (" << d.fullTileVar << ") {\n";
        ++indent;
        for (int64_t w = 0; w < numWarps; ++w) {
          os << ind() << "if (" << warpId << " == " << w << ") {\n";
          ++indent;
          for (int64_t f = w, j = 0; f < nFrag; f += numWarps, ++j) {
            int64_t mi = f / nT, ni = f % nT;
            os << ind() << "simdgroup_store(" << fusedDot.accNames[j] << ", "
               << base << " + (" << rowB << " + " << (mi * 8) << ") * " << ldc
               << " + (" << colB << " + " << (ni * 8) << "), " << ldc << ");\n";
          }
          --indent;
          os << ind() << "}\n";
        }
        --indent;
        os << ind() << "} else {\n";
        ++indent;
      }
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      for (int64_t w = 0; w < numWarps; ++w) {
        os << ind() << "if (" << warpId << " == " << w << ") {\n";
        ++indent;
        for (int64_t f = w, j = 0; f < nFrag; f += numWarps, ++j) {
          int64_t mi = f / nT, ni = f % nT;
          os << ind() << "simdgroup_store(" << fusedDot.accNames[j] << ", "
             << tgC << " + " << (mi * 8 * N + ni * 8) << ", " << N << ");\n";
        }
        --indent;
        os << ind() << "}\n";
      }
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      emitReadback(0, 0, M);
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      if (fusedDot.direct) {
        --indent;
        os << ind() << "}\n";
      }
      valMap[op.getResult()] = ids;
      return success();
    }

    // Per batch slice: stage the M×K / K×N operand slice into one reused pool
    // region, run the 8×8 fragment MMA over K, then read the M×N accumulator
    // back. Slices are barrier-separated and reuse the same pool.
    for (int64_t bi = 0; bi < B; ++bi) {
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      if (!aInPlace)
        for (int r = 0, n = regCount(aStage); r < n; ++r)
          os << ind() << batchGuard(aStageTy, r, bi) << tgA << "["
             << sliceFlatOffset(aStageTy, r) << "] = " << aNames[r] << ";\n";
      if (!bInPlace)
        for (int r = 0, n = regCount(bStage); r < n; ++r)
          os << ind() << batchGuard(bStageTy, r, bi) << tgB << "["
             << sliceFlatOffset(bStageTy, r) << "] = " << bNames[r] << ";\n";
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

      auto emitFragMMA = [&](int64_t mi, int64_t ni, StringRef acc) {
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
      };

      if (disjointC) {
        for (int64_t w = 0; w < numWarps; ++w) {
          os << ind() << "if (" << warpId << " == " << w << ") {\n";
          ++indent;
          for (int64_t f = w; f < nFrag; f += numWarps) {
            int64_t mi = f / nT, ni = f % nT;
            std::string acc = fresh();
            os << ind() << accFrag << " " << acc << " = " << accFrag
               << "(0.0f);\n";
            emitFragMMA(mi, ni, acc);
            os << ind() << "simdgroup_store(" << acc << ", " << tgC << " + "
               << (mi * 8 * N + ni * 8) << ", " << N << ");\n";
          }
          --indent;
          os << ind() << "}\n";
        }
        os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        emitReadback(bi, 0, M);
        continue;
      }

      std::string accBase = fresh() + "_" + std::to_string(bi) + "_";
      for (int64_t f = 0; f < nFrag; ++f) {
        int64_t mi = f / nT, ni = f % nT;
        std::string acc =
            accBase + std::to_string(mi) + "_" + std::to_string(ni);
        os << ind() << accFrag << " " << acc << " = " << accFrag << "(0.0f);\n";
      }
      for (int64_t w = 0; w < numWarps; ++w) {
        os << ind() << "if (" << warpId << " == " << w << ") {\n";
        ++indent;
        for (int64_t f = w; f < nFrag; f += numWarps) {
          int64_t mi = f / nT, ni = f % nT;
          std::string acc =
              accBase + std::to_string(mi) + "_" + std::to_string(ni);
          emitFragMMA(mi, ni, acc);
        }
        --indent;
        os << ind() << "}\n";
      }

      for (int64_t r0 = 0; r0 < M; r0 += bandRows) {
        int64_t r1 = std::min<int64_t>(r0 + bandRows, M);
        os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        for (int64_t w = 0; w < numWarps; ++w) {
          os << ind() << "if (" << warpId << " == " << w << ") {\n";
          ++indent;
          for (int64_t f = w; f < nFrag; f += numWarps) {
            int64_t mi = f / nT, ni = f % nT;
            if (mi * 8 < r0 || mi * 8 >= r1)
              continue;
            std::string acc =
                accBase + std::to_string(mi) + "_" + std::to_string(ni);
            os << ind() << "simdgroup_store(" << acc << ", " << tgC << " + "
               << ((mi * 8 - r0) * N + ni * 8) << ", " << N << ");\n";
          }
          --indent;
          os << ind() << "}\n";
        }
        os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        emitReadback(bi, r0, r1);
      }
    }
    os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

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
    int rank = aTy.getRank();
    if (rank != 2 && rank != 3) {
      op.emitError("EmitMSL: scalar tt.dot requires 2-D or 3-D operands");
      return failure();
    }
    int64_t B = rank == 3 ? cTy.getShape()[0] : 1;
    int64_t K = aTy.getShape()[rank - 1];
    int64_t M = cTy.getShape()[rank - 2];
    int64_t N = cTy.getShape()[rank - 1];

    std::string aScalar = mslScalarType(aElem);
    std::string bScalar = mslScalarType(bElem);
    std::string accScalar = mslScalarType(cElem);

    Value aStage = op.getA(), bStage = op.getB();
    if (rank == 2) {
      if (Value s = dotOperandConvertSource(op, op.getA()))
        aStage = s;
      if (Value s = dotOperandConvertSource(op, op.getB()))
        bStage = s;
    }
    auto aStageTy = cast<RankedTensorType>(aStage.getType());
    auto bStageTy = cast<RankedTensorType>(bStage.getType());
    auto &aNames = names(aStage);
    auto &bNames = names(bStage);
    auto &cInit = names(op.getC());

    int64_t aBytes = M * K * (bitsOf(aElem) / 8);
    std::string tgA = fresh(), tgB = fresh();
    os << ind() << "threadgroup " << aScalar << "* " << tgA << " = "
       << poolRegion(0, aScalar) << ";\n";
    os << ind() << "threadgroup " << bScalar << "* " << tgB << " = "
       << poolRegion(aBytes, bScalar) << ";\n";

    tt::LinearLayout cLL = ttg::toLinearLayout(cTy);
    auto outNames = llvm::to_vector(cLL.getOutDimNames());
    StringAttr dRow = outNames[rank - 2], dCol = outNames[rank - 1];

    auto batchGuard = [&](RankedTensorType rt, int reg, int64_t bi)
        -> std::string {
      if (rank != 3)
        return "";
      return "if (" + batchCoordExpr(rt, reg) + " == " + std::to_string(bi) +
             ") ";
    };

    int nRes = regCount(op.getResult());
    SmallVector<std::string> ids(nRes);
    for (int r = 0; r < nRes; ++r) {
      ids[r] = fresh();
      std::string base = cInit[cInit.size() == 1 ? 0 : r];
      os << ind() << accScalar << " " << ids[r] << " = " << base << ";\n";
    }

    for (int64_t bi = 0; bi < B; ++bi) {
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      for (int r = 0, n = regCount(aStage); r < n; ++r)
        os << ind() << batchGuard(aStageTy, r, bi) << tgA << "["
           << sliceFlatOffset(aStageTy, r) << "] = " << aNames[r] << ";\n";
      for (int r = 0, n = regCount(bStage); r < n; ++r)
        os << ind() << batchGuard(bStageTy, r, bi) << tgB << "["
           << sliceFlatOffset(bStageTy, r) << "] = " << bNames[r] << ";\n";
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

      for (int r = 0; r < nRes; ++r) {
        std::string mrow = fresh(), ncol = fresh(), acc = fresh();
        os << ind() << "int " << mrow << " = " << layoutCoordExpr(cTy, r, dRow)
           << ";\n";
        os << ind() << "int " << ncol << " = " << layoutCoordExpr(cTy, r, dCol)
           << ";\n";
        os << ind() << accScalar << " " << acc << " = (" << accScalar << ")0;\n";
        std::string kv = fresh();
        os << ind() << "for (int " << kv << " = 0; " << kv << " < " << K
           << "; ++" << kv << ") {\n";
        ++indent;
        os << ind() << acc << " += (" << accScalar << ")" << tgA << "[" << mrow
           << " * " << K << " + " << kv << "] * (" << accScalar << ")" << tgB
           << "[" << kv << " * " << N << " + " << ncol << "];\n";
        --indent;
        os << ind() << "}\n";
        std::string guard =
            rank == 3 ? ("if (" + batchCoordExpr(cTy, r) + " == " +
                         std::to_string(bi) + ") ")
                      : "";
        os << ind() << guard << ids[r] << " += " << acc << ";\n";
      }
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
      std::string deref = scalarSpinlock ? ("*(device coherent(device) " + sc +
                                            "*)" + p)
                                         : ("*" + p);
      if (hasMask) {
        const std::string &m = (*mask)[mask->size() == 1 ? 0 : r];
        os << ind() << "if (" << m << ") " << id << " = " << deref << ";\n";
      } else {
        os << ind() << id << " = " << deref << ";\n";
      }
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  LogicalResult emitStore(tt::StoreOp op) {
    auto handled = directStoreHandled.find(op.getOperation());
    if (handled != directStoreHandled.end()) {
      os << ind() << "if (!" << handled->second << ") {\n";
      ++indent;
      LogicalResult r = emitStoreBody(op);
      --indent;
      os << ind() << "}\n";
      return r;
    }
    return emitStoreBody(op);
  }

  LogicalResult emitStoreBody(tt::StoreOp op) {
    auto &ptrs = names(op.getPtr());
    auto &vals = names(op.getValue());
    bool hasMask = op.getMask() != nullptr;
    SmallVector<std::string> *mask = hasMask ? &names(op.getMask()) : nullptr;
    bool uniform = !isa<RankedTensorType>(op.getPtr().getType());
    int rc = ptrs.size();

    // Redundant-thread predicate: when the pointer layout replicates an element
    // across lanes/warps, only the canonical thread may write, else racing
    // threads clobber each other on a read-modify-write to the same address.
    unsigned laneFree = 0, warpFree = 0;
    if (!uniform) {
      auto ptrTy = cast<RankedTensorType>(op.getPtr().getType());
      tt::LinearLayout ll = ttg::toLinearLayout(ptrTy);
      MLIRContext *c = op.getContext();
      auto masks = ll.getFreeVariableMasks();
      laneFree = masks.lookup(StringAttr::get(c, "lane"));
      warpFree = masks.lookup(StringAttr::get(c, "warp"));
    }
    std::string threadPred;
    if (uniform) {
      threadPred = tidId + ".x == 0";
    } else {
      if (laneFree)
        threadPred = "((" + laneId + " & " + std::to_string(laneFree) +
                     ") == 0)";
      if (warpFree) {
        std::string wp =
            "((" + warpId + " & " + std::to_string(warpFree) + ") == 0)";
        threadPred = threadPred.empty() ? wp : threadPred + " && " + wp;
      }
    }

    std::string sc = mslScalarType(elementScalarType(op.getValue().getType()));
    for (int r = 0; r < rc; ++r) {
      const std::string &p = ptrs[r];
      const std::string &v = vals[vals.size() == 1 ? 0 : r];
      std::string lhs = scalarSpinlock
                            ? ("*(device coherent(device) " + sc + "*)" + p)
                            : ("*" + p);
      std::string guard = threadPred;
      if (hasMask) {
        const std::string &m = (*mask)[mask->size() == 1 ? 0 : r];
        guard = guard.empty() ? m : guard + " && " + m;
      }
      if (guard.empty())
        os << ind() << lhs << " = " << v << ";\n";
      else
        os << ind() << "if (" << guard << ") " << lhs << " = " << v << ";\n";
    }
    return success();
  }

  // IEEE round-to-nearest-even narrowing of an f32 `v` to half/bfloat, with
  // correct NaN/Inf/overflow/subnormal handling (used by fp_to_fp rtne).
  std::string emitRoundedHalfValueFull(const std::string &sc,
                                       const std::string &v) {
    std::string f = fresh(), u = fresh(), h = fresh(), bits = fresh();
    os << ind() << "float " << f << " = (float)(" << v << ");\n";
    os << ind() << "uint " << u << " = as_type<uint>(" << f << ");\n";
    os << ind() << "ushort " << bits << ";\n";
    std::string sgn = fresh(), e32 = fresh(), mant = fresh();
    os << ind() << "uint " << sgn << " = (" << u << " >> 16) & 0x8000u;\n";
    os << ind() << "int " << e32 << " = (int)((" << u << " >> 23) & 0xffu);\n";
    os << ind() << "uint " << mant << " = " << u << " & 0x7fffffu;\n";
    if (sc == "bfloat") {
      std::string r = fresh();
      os << ind() << "if (" << e32 << " == 0xff) {\n";
      ++indent;
      os << ind() << bits << " = (ushort)(((" << u << " >> 16) & 0xffffu) | ("
         << mant << " ? 0x40u : 0u));\n";
      --indent;
      os << ind() << "} else {\n";
      ++indent;
      os << ind() << "uint " << r << " = (" << u << " >> 16) & 1u;\n";
      os << ind() << "uint __t = (" << u << " + 0x7fffu + " << r << ");\n";
      os << ind() << bits << " = (ushort)((__t >> 16) & 0xffffu);\n";
      --indent;
      os << ind() << "}\n";
    } else {
      std::string ex = fresh();
      os << ind() << "int " << ex << " = " << e32 << " - 112;\n";
      os << ind() << "if (" << e32 << " == 0xff) {\n";
      ++indent;
      os << ind() << bits << " = (ushort)(" << sgn << " | 0x7c00u | ("
         << mant << " ? 0x200u : 0u));\n";
      --indent;
      os << ind() << "} else if (" << ex << " >= 31) {\n";
      ++indent;
      os << ind() << bits << " = (ushort)(" << sgn << " | 0x7c00u);\n";
      --indent;
      os << ind() << "} else if (" << ex << " <= 0) {\n";
      ++indent;
      os << ind() << "if (" << ex << " < -10) { " << bits << " = (ushort)"
         << sgn << "; }\n";
      os << ind() << "else {\n";
      ++indent;
      os << ind() << "uint __fm = " << mant << " | 0x800000u;\n";
      os << ind() << "int __sh = 14 - " << ex << ";\n";
      os << ind() << "uint __m = __fm >> __sh;\n";
      os << ind() << "uint __rem = __fm & ((1u << __sh) - 1u);\n";
      os << ind() << "uint __half = 1u << (__sh - 1);\n";
      os << ind() << "if (__rem > __half || (__rem == __half && (__m & 1u))) "
         << "__m += 1;\n";
      os << ind() << bits << " = (ushort)(" << sgn << " | __m);\n";
      --indent;
      os << ind() << "}\n";
      --indent;
      os << ind() << "} else {\n";
      ++indent;
      std::string m = fresh(), rem = fresh();
      os << ind() << "uint " << m << " = " << mant << " >> 13;\n";
      os << ind() << "uint " << rem << " = " << mant << " & 0x1fffu;\n";
      os << ind() << bits << " = (ushort)(" << sgn << " | ((uint)" << ex
         << " << 10) | " << m << ");\n";
      os << ind() << "if (" << rem << " > 0x1000u || (" << rem
         << " == 0x1000u && (" << m << " & 1u))) " << bits << " += 1;\n";
      --indent;
      os << ind() << "}\n";
    }
    os << ind() << sc << " " << h << " = as_type<" << sc << ">(" << bits
       << ");\n";
    return h;
  }

  // Round-toward-zero (truncating) narrowing of an f32 `v` to half/bfloat.
  // RTZ drops the low mantissa bits with no rounding increment.
  std::string emitTruncatedFloatValue(const std::string &sc,
                                      const std::string &v) {
    std::string f = fresh(), u = fresh(), h = fresh(), bits = fresh();
    os << ind() << "float " << f << " = (float)(" << v << ");\n";
    os << ind() << "uint " << u << " = as_type<uint>(" << f << ");\n";
    os << ind() << "ushort " << bits << ";\n";
    if (sc == "bfloat") {
      os << ind() << bits << " = (ushort)((" << u << " >> 16) & 0xffffu);\n";
    } else {
      std::string sgn = fresh(), ex = fresh(), mant = fresh();
      os << ind() << "uint " << sgn << " = (" << u << " >> 16) & 0x8000u;\n";
      os << ind() << "int " << ex << " = (int)((" << u
         << " >> 23) & 0xffu) - 112;\n";
      os << ind() << "uint " << mant << " = " << u << " & 0x7fffffu;\n";
      os << ind() << "if (((" << u << " >> 23) & 0xffu) == 0xffu) {\n";
      ++indent;
      os << ind() << bits << " = (ushort)(" << sgn << " | 0x7c00u | ("
         << mant << " ? 0x200u : 0u));\n";
      --indent;
      os << ind() << "} else if (" << ex << " >= 31) {\n";
      ++indent;
      os << ind() << bits << " = (ushort)(" << sgn << " | 0x7bffu);\n";
      --indent;
      os << ind() << "} else if (" << ex << " <= 0) {\n";
      ++indent;
      os << ind() << "if (" << ex << " < -10) { " << bits << " = (ushort)"
         << sgn << "; }\n";
      os << ind() << "else { uint __m = (" << mant
         << " | 0x800000u) >> (14 - " << ex << "); " << bits
         << " = (ushort)(" << sgn << " | __m); }\n";
      --indent;
      os << ind() << "} else {\n";
      ++indent;
      os << ind() << bits << " = (ushort)(" << sgn << " | ((uint)" << ex
         << " << 10) | (" << mant << " >> 13));\n";
      --indent;
      os << ind() << "}\n";
    }
    os << ind() << sc << " " << h << " = as_type<" << sc << ">(" << bits
       << ");\n";
    return h;
  }

  // Integer RTNE narrowing of `v` to `sc`. Metal fast-math elides a plain
  // (half)/(bfloat) cast whose only consumer re-widens to float (the CAS add),
  // dropping the round; the integer path forces one the optimizer cannot cancel.
  std::string emitRoundedHalfValue(const std::string &sc, const std::string &v) {
    std::string f = fresh(), u = fresh(), h = fresh();
    os << ind() << "float " << f << " = (float)(" << v << ");\n";
    os << ind() << "uint " << u << " = as_type<uint>(" << f << ");\n";
    std::string bits = fresh();
    os << ind() << "ushort " << bits << ";\n";
    if (sc == "bfloat") {
      std::string r = fresh();
      os << ind() << "uint " << r << " = (" << u << " >> 16) & 1u;\n";
      os << ind() << bits << " = (ushort)(((" << u << " + 0x7fffu + " << r
         << ") >> 16) & 0xffffu);\n";
    } else {
      std::string sgn = fresh(), ex = fresh(), mant = fresh(), m = fresh(),
                  rem = fresh();
      os << ind() << "uint " << sgn << " = (" << u << " >> 16) & 0x8000u;\n";
      os << ind() << "int " << ex << " = (int)((" << u
         << " >> 23) & 0xffu) - 112;\n";
      os << ind() << "uint " << mant << " = " << u << " & 0x7fffffu;\n";
      os << ind() << "if (" << ex << " <= 0) {\n";
      ++indent;
      os << ind() << bits << " = (ushort)" << sgn << ";\n";
      --indent;
      os << ind() << "} else if (" << ex << " >= 31) {\n";
      ++indent;
      os << ind() << bits << " = (ushort)(" << sgn << " | 0x7c00u);\n";
      --indent;
      os << ind() << "} else {\n";
      ++indent;
      os << ind() << "uint " << m << " = " << mant << " >> 13;\n";
      os << ind() << "uint " << rem << " = " << mant << " & 0x1fffu;\n";
      os << ind() << bits << " = (ushort)(" << sgn << " | ((uint)" << ex
         << " << 10) | " << m << ");\n";
      os << ind() << "if (" << rem << " > 0x1000u || (" << rem
         << " == 0x1000u && (" << m << " & 1u))) " << bits << " += 1;\n";
      --indent;
      os << ind() << "}\n";
    }
    os << ind() << sc << " " << h << " = as_type<" << sc << ">(" << bits
       << ");\n";
    return h;
  }

  // For a 16-bit-float element pointer p, emit statements binding a uint* to the
  // containing aligned 32-bit word and a bool selecting the high half-word.
  // Returns {wordPtr, isHigh} identifiers.
  std::pair<std::string, std::string> emitPacked16Base(const std::string &p,
                                                       const std::string &sc) {
    std::string bytePtr = fresh(), wordAddr = fresh(), isHigh = fresh(),
                wordPtr = fresh();
    os << ind() << "device uchar *" << bytePtr
       << " = (device uchar *)(" << p << ");\n";
    os << ind() << "size_t " << wordAddr << " = (size_t)" << bytePtr
       << " & ~(size_t)3;\n";
    os << ind() << "bool " << isHigh << " = ((size_t)" << bytePtr
       << " & 2u) != 0u;\n";
    os << ind() << "device atomic_uint *" << wordPtr
       << " = (device atomic_uint *)" << wordAddr << ";\n";
    (void)sc;
    return {wordPtr, isHigh};
  }

  // Extract the selected 16-bit-float lane from a loaded 32-bit word into a
  // float, apply `newFloatExpr` (which may reference `curId`), repack into the
  // word, and CAS-loop until it lands. Binds `id` to the pre-op value.
  void emitPacked16CASLoop(const std::string &wordPtr, const std::string &isHigh,
                           const std::string &sc, const std::string &curId,
                           const std::string &newHalfExpr,
                           const std::string &id) {
    std::string word = fresh(), lane = fresh(), newLane = fresh(),
                newWord = fresh();
    os << ind() << "uint " << word
       << " = atomic_load_explicit(" << wordPtr << ", memory_order_relaxed);\n";
    os << ind() << "while (true) {\n";
    ++indent;
    os << ind() << "ushort " << lane << " = (ushort)((" << isHigh << ") ? ("
       << word << " >> 16) : (" << word << " & 0xffffu));\n";
    os << ind() << id << " = as_type<" << sc << ">(" << lane << ");\n";
    os << ind() << sc << " " << curId << " = as_type<" << sc << ">(" << lane
       << ");\n";
    os << ind() << sc << " " << newLane << " = " << newHalfExpr << ";\n";
    os << ind() << "uint " << newWord << " = (" << isHigh
       << ") ? ((" << word << " & 0x0000ffffu) | ((uint)as_type<ushort>("
       << newLane << ") << 16)) : ((" << word
       << " & 0xffff0000u) | (uint)as_type<ushort>(" << newLane << "));\n";
    os << ind() << "if (atomic_compare_exchange_weak_explicit(" << wordPtr
       << ", &" << word << ", " << newWord
       << ", memory_order_relaxed, memory_order_relaxed)) break;\n";
    --indent;
    os << ind() << "}\n";
  }

  // f32 atomic RMW via a 32-bit CAS loop (Metal's atomic_float only supports
  // fetch_add; max/min/xchg need emulation). `newFloatExpr` references `curId`.
  void emitFloat32CASLoop(const std::string &p, const std::string &curId,
                          const std::string &newFloatExpr,
                          const std::string &id) {
    std::string wordPtr = fresh(), word = fresh(), newWord = fresh();
    os << ind() << "device atomic_uint *" << wordPtr
       << " = (device atomic_uint *)(" << p << ");\n";
    os << ind() << "uint " << word
       << " = atomic_load_explicit(" << wordPtr << ", memory_order_relaxed);\n";
    os << ind() << "while (true) {\n";
    ++indent;
    os << ind() << id << " = as_type<float>(" << word << ");\n";
    os << ind() << "float " << curId << " = as_type<float>(" << word << ");\n";
    os << ind() << "uint " << newWord << " = as_type<uint>((float)("
       << newFloatExpr << "));\n";
    os << ind() << "if (atomic_compare_exchange_weak_explicit(" << wordPtr
       << ", &" << word << ", " << newWord
       << ", memory_order_relaxed, memory_order_relaxed)) break;\n";
    --indent;
    os << ind() << "}\n";
  }

  LogicalResult emitAtomicRMW(tt::AtomicRMWOp op) {
    Value res = op.getResult();
    Type scalarTy = elementScalarType(res.getType());
    std::string sc = mslScalarType(scalarTy);
    bool isFloat = isa<FloatType>(scalarTy);
    unsigned bw = scalarTy.getIntOrFloatBitWidth();
    tt::RMWOp kind = op.getAtomicRmwOp();

    if (!isFloat && bw == 64) {
      op.emitError("EmitMSL: 64-bit integer atomics are not supported on Metal");
      return failure();
    }

    bool floatNative = isFloat && bw == 32 &&
                       (kind == tt::RMWOp::ADD || kind == tt::RMWOp::FADD);
    bool floatEmulated = isFloat && !floatNative;

    if (floatEmulated && bw != 16 && bw != 32) {
      op.emitError("EmitMSL: unsupported floating-point atomic rmw width");
      return failure();
    }

    std::string atomicTy;
    if (isFloat)
      atomicTy = "atomic_float";
    else if (kind == tt::RMWOp::UMAX || kind == tt::RMWOp::UMIN)
      atomicTy = "atomic_uint";
    else
      atomicTy = bw == 64 ? "atomic_long" : "atomic_int";

    const char *fn = nullptr;
    if (!floatEmulated) {
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
    }

    auto &ptrs = names(op.getPtr());
    auto &vals = names(op.getVal());
    bool hasMask = op.getMask() != nullptr;
    SmallVector<std::string> *mask = hasMask ? &names(op.getMask()) : nullptr;
    bool uniform = !isa<RankedTensorType>(op.getPtr().getType());
    int rc = ptrs.size();

    // Redundant-thread predicate: when the pointer layout replicates an element
    // across lanes/warps/registers, only the canonical thread/register may
    // perform the atomic, else the update lands multiple times.
    unsigned laneFree = 0, warpFree = 0, regFree = 0;
    if (!uniform) {
      auto ptrTy = cast<RankedTensorType>(op.getPtr().getType());
      tt::LinearLayout ll = ttg::toLinearLayout(ptrTy);
      MLIRContext *c = op.getContext();
      auto masks = ll.getFreeVariableMasks();
      laneFree = masks.lookup(StringAttr::get(c, "lane"));
      warpFree = masks.lookup(StringAttr::get(c, "warp"));
      regFree = masks.lookup(StringAttr::get(c, "register"));
    }
    std::string threadPred;
    if (laneFree)
      threadPred = "((" + laneId + " & " + std::to_string(laneFree) + ") == 0)";
    if (warpFree) {
      std::string wp =
          "((" + warpId + " & " + std::to_string(warpFree) + ") == 0)";
      threadPred = threadPred.empty() ? wp : threadPred + " && " + wp;
    }

    tt::MemSemantic sem = op.getSem();

    SmallVector<std::string> ids(rc);
    for (int r = 0; r < rc; ++r) {
      if (regFree && (r & regFree) != 0) {
        ids[r] = ids[r & ~regFree];
        continue;
      }
      const std::string &p = ptrs[r];
      const std::string &v = vals[vals.size() == 1 ? 0 : r];
      std::string id = fresh();
      os << ind() << sc << " " << id << " = " << init0(sc) << ";\n";
      std::string guard;
      if (uniform)
        guard = tidId + ".x == 0";
      else if (!threadPred.empty())
        guard = threadPred;
      if (hasMask) {
        const std::string &m = (*mask)[mask->size() == 1 ? 0 : r];
        guard = guard.empty() ? m : guard + " && " + m;
      }
      bool guarded = !guard.empty();
      if (guarded) {
        os << ind() << "if (" << guard << ") {\n";
        ++indent;
      }
      if (floatEmulated) {
        std::string cur = fresh();
        if (bw == 16) {
          std::string vh = emitRoundedHalfValue(sc, v);
          std::string newExpr = floatRmwExpr(kind, cur, vh);
          auto base = emitPacked16Base(p, sc);
          emitPacked16CASLoop(base.first, base.second, sc, cur, newExpr, id);
        } else {
          std::string newExpr = floatRmwExpr(kind, cur, "(float)(" + v + ")");
          emitFloat32CASLoop(p, cur, newExpr, id);
        }
      } else {
        std::string order = "memory_order_relaxed";
        std::string tail;
        switch (sem) {
        case tt::MemSemantic::ACQUIRE:
          order = "memory_order_acquire";
          break;
        case tt::MemSemantic::RELEASE:
          order = "memory_order_release";
          break;
        case tt::MemSemantic::ACQUIRE_RELEASE:
          order = "memory_order_acq_rel";
          break;
        default:
          break;
        }
        if (order != "memory_order_relaxed")
          tail = ", mem_flags::mem_device";
        std::string call = std::string(fn) + "((device " + atomicTy + "*)" + p +
                           ", " + v + ", " + order + tail + ")";
        os << ind() << id << " = " << call << ";\n";
      }
      if (guarded) {
        --indent;
        os << ind() << "}\n";
      }
      // The atomic ran only on the canonical lane; replica lanes hold init0.
      // Broadcast the returned pre-op value across the lane-replicas so a
      // downstream use of the result (use_result kernels) reads it on every
      // lane that logically owns this element.
      if (!uniform && laneFree) {
        std::string src = "(uint)(" + laneId + " & " +
                          std::to_string(~laneFree & 31) + ")";
        id = emitShuffle("simd_shuffle", sc, id, src);
      }
      ids[r] = id;
    }

    // Cross-warp replicas: when the pointer layout replicates an element across
    // warps, only the canonical warp ran the atomic, so replica warps hold
    // init0. Stage the canonical warp's per-lane results through threadgroup
    // memory keyed by (register, canonical-lane) so every warp reads the true
    // pre-op value for its logical element.
    if (!uniform && warpFree) {
      auto ptrTy = cast<RankedTensorType>(op.getPtr().getType());
      tt::LinearLayout ll = ttg::toLinearLayout(ptrTy);
      MLIRContext *c = op.getContext();
      int64_t numWarps = ll.hasInDim(StringAttr::get(c, "warp"))
                             ? ll.getInDimSize(StringAttr::get(c, "warp"))
                             : 1;
      std::string bcbuf = fresh();
      os << ind() << "threadgroup " << sc << "* " << bcbuf << " = "
         << poolRegion(0, sc) << ";\n";
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      std::string wcanon =
          "((" + warpId + " & " + std::to_string(warpFree) + ") == 0)";
      // A replica warp shares its canonical source's non-free warp bits, so key
      // the staging slot on those bits to keep distinct canonical warps apart.
      std::string warpKey =
          "(" + warpId + " & " + std::to_string(~warpFree & (numWarps - 1)) +
          ")";
      auto slotFor = [&](int reg) {
        return "((" + warpKey + " * " + std::to_string(rc * 32) + ") + " +
               std::to_string(reg) + " * 32 + (" + laneId + " & " +
               std::to_string(~laneFree & 31) + "))";
      };
      for (int r = 0; r < rc; ++r) {
        if (regFree && (r & regFree) != 0)
          continue;
        os << ind() << "if (" << wcanon << ") " << bcbuf << "[" << slotFor(r)
           << "] = " << ids[r] << ";\n";
      }
      os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      for (int r = 0; r < rc; ++r) {
        int src = regFree ? (r & ~regFree) : r;
        std::string bc = fresh();
        os << ind() << sc << " " << bc << " = " << bcbuf << "[" << slotFor(src)
           << "];\n";
        ids[r] = bc;
      }
    }
    valMap[res] = ids;
    return success();
  }

  static std::string floatRmwExpr(tt::RMWOp kind, const std::string &cur,
                                  const std::string &v) {
    switch (kind) {
    case tt::RMWOp::ADD:
    case tt::RMWOp::FADD:
      return cur + " + " + v;
    case tt::RMWOp::MAX:
      return "fmax(" + cur + ", " + v + ")";
    case tt::RMWOp::MIN:
      return "fmin(" + cur + ", " + v + ")";
    case tt::RMWOp::XCHG:
      return v;
    default:
      return v;
    }
  }

  // A bare Metal device atomic accepts only memory_order_relaxed, but the
  // mem_flags overload (…, order, order, mem_flags::mem_device) does take
  // acquire/release. A scalar spinlock therefore CAS-acquires and xchg-releases
  // the lock word directly; the lock-guarded foreign buffer is accessed through
  // `device coherent(device)` so the acquire/release pair actually flushes it to
  // the next holder (else the AGX per-threadgroup cache silently drops updates).
  LogicalResult emitAtomicCAS(tt::AtomicCASOp op) {
    Value res = op.getResult();
    Type scalarTy = elementScalarType(res.getType());
    std::string sc = mslScalarType(scalarTy);
    bool isFloat = isa<FloatType>(scalarTy);
    unsigned bw = scalarTy.getIntOrFloatBitWidth();

    bool packed16 = isFloat && bw == 16;
    bool word32 = bw == 32;
    if (!packed16 && !word32) {
      op.emitError("EmitMSL: unsupported atomic_cas element width");
      return failure();
    }

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
      std::string id = fresh();
      bool uni = uniform;
      if (uni) {
        os << ind() << sc << " " << id << " = " << c << ";\n";
        os << ind() << "if (" << tidId << ".x == 0) {\n";
        ++indent;
      }
      if (packed16)
        emitPacked16CAS(p, c, v, sc, id, !uni);
      else if (isFloat)
        emitFloat32CAS(p, c, v, id, !uni);
      else
        emitInt32CAS(p, c, v, sc, id, !uni);
      if (uni) {
        --indent;
        os << ind() << "}\n";
        // Cross-threadgroup spinlock: lane 0 owns the CAS, so its result must
        // be broadcast to every lane (else non-acquiring lanes see the stale
        // compare value and enter the critical section early). The acquire
        // device fence between the store and the reload publishes post-lock
        // memory to the whole threadgroup.
        std::string bcast = fresh();
        os << ind() << "threadgroup " << sc << "* " << bcast << " = "
           << poolRegion(0, sc) << ";\n";
        os << ind() << "if (" << tidId << ".x == 0) " << bcast << "[0] = " << id
           << ";\n";
        os << ind()
           << "atomic_thread_fence(mem_flags::mem_device, "
              "memory_order_acquire);\n";
        os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        os << ind() << id << " = " << bcast << "[0];\n";
        os << ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      }
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }

  void emitInt32CAS(const std::string &p, const std::string &c,
                    const std::string &v, const std::string &sc,
                    const std::string &id, bool declare) {
    std::string exp = fresh();
    os << ind() << sc << " " << exp << " = " << c << ";\n";
    std::string call = "atomic_compare_exchange_weak_explicit((device "
                       "atomic_int *)" +
                       p + ", &" + exp + ", " + v +
                       ", memory_order_acquire, memory_order_relaxed, "
                       "mem_flags::mem_device)";
    os << ind() << "while (" << exp << " == " << c << " && !(" << call
       << ")) {}\n";
    os << ind() << (declare ? sc + " " : "") << id << " = " << exp << ";\n";
  }

  void emitFloat32CAS(const std::string &p, const std::string &c,
                      const std::string &v, const std::string &id,
                      bool declare) {
    std::string exp = fresh(), cbits = fresh();
    os << ind() << "uint " << exp << " = as_type<uint>(" << c << ");\n";
    os << ind() << "uint " << cbits << " = as_type<uint>(" << c << ");\n";
    std::string call = "atomic_compare_exchange_weak_explicit((device "
                       "atomic_uint *)" +
                       p + ", &" + exp + ", as_type<uint>(" + v +
                       "), memory_order_relaxed, memory_order_relaxed)";
    os << ind() << "while (" << exp << " == " << cbits << " && !(" << call
       << ")) {}\n";
    os << ind() << (declare ? std::string("float ") : "") << id
       << " = as_type<float>(" << exp << ");\n";
  }

  void emitPacked16CAS(const std::string &p, const std::string &c,
                       const std::string &v, const std::string &sc,
                       const std::string &id, bool declare) {
    auto base = emitPacked16Base(p, sc);
    const std::string &wordPtr = base.first, &isHigh = base.second;
    std::string word = fresh(), cur = fresh(), lane = fresh(), newWord = fresh(),
                matched = fresh();
    os << ind() << "ushort " << cur
       << " = as_type<ushort>((" << sc << ")(" << c << "));\n";
    os << ind() << "ushort " << lane
       << " = as_type<ushort>((" << sc << ")(" << v << "));\n";
    os << ind() << "uint " << word
       << " = atomic_load_explicit(" << wordPtr << ", memory_order_relaxed);\n";
    os << ind() << (declare ? std::string(sc) + " " : "") << id << " = "
       << init0(sc) << ";\n";
    os << ind() << "while (true) {\n";
    ++indent;
    std::string got = fresh();
    os << ind() << "ushort " << got << " = (ushort)((" << isHigh << ") ? ("
       << word << " >> 16) : (" << word << " & 0xffffu));\n";
    os << ind() << id << " = as_type<" << sc << ">(" << got << ");\n";
    os << ind() << "if (" << got << " != " << cur << ") break;\n";
    os << ind() << "uint " << newWord << " = (" << isHigh
       << ") ? ((" << word << " & 0x0000ffffu) | ((uint)" << lane
       << " << 16)) : ((" << word << " & 0xffff0000u) | (uint)" << lane
       << ");\n";
    os << ind() << "if (atomic_compare_exchange_weak_explicit(" << wordPtr
       << ", &" << word << ", " << newWord
       << ", memory_order_relaxed, memory_order_relaxed)) break;\n";
    --indent;
    os << ind() << "}\n";
  }

  LogicalResult emitAtomicPoll(tt::AtomicPollOp op) {
    Type expTy = op.getExpected().getType();
    unsigned bw = expTy.getIntOrFloatBitWidth();
    if (bw != 16 && bw != 32 && bw != 64) {
      op.emitError("EmitMSL: atomic_poll supports only 16/32/64-bit values");
      return failure();
    }
    bool acquire = op.getSem() == tt::MemSemantic::ACQUIRE;

    const std::string &p = names(op.getPtr())[0];
    const std::string &exp = names(op.getExpected())[0];

    std::string barrierFlags =
        acquire ? "mem_flags::mem_device | mem_flags::mem_threadgroup"
                : "mem_flags::mem_threadgroup";
    std::string result = fresh();

    // Binds `wordPtr` to a device atomic_uint* on the aligned containing word
    // and returns an expression evaluating to the polled value: the full word
    // for 32-bit, or the masked 16-bit lane for 16-bit (Metal has no 16-bit
    // device atomic).
    auto emitProbe = [&](std::string &loadExpr) -> std::string {
      std::string wordPtr = fresh();
      if (bw == 16) {
        std::string isHigh = fresh();
        os << ind() << "bool " << isHigh << " = ((size_t)(" << p
           << ") & 2u) != 0u;\n";
        os << ind() << "device atomic_uint *" << wordPtr
           << " = (device atomic_uint *)((size_t)(" << p
           << ") & ~(size_t)3);\n";
        loadExpr = "(ushort)((" + isHigh + ") ? (atomic_load_explicit(" +
                   wordPtr + ", memory_order_relaxed) >> 16) : " +
                   "(atomic_load_explicit(" + wordPtr +
                   ", memory_order_relaxed) & 0xffffu))";
      } else if (bw == 64) {
        os << ind() << "volatile device ulong *" << wordPtr
           << " = (volatile device ulong *)(" << p << ");\n";
        loadExpr = "(*" + wordPtr + ")";
      } else {
        os << ind() << "device atomic_uint *" << wordPtr
           << " = (device atomic_uint *)(" << p << ");\n";
        loadExpr =
            "atomic_load_explicit(" + wordPtr + ", memory_order_relaxed)";
      }
      return wordPtr;
    };
    std::string wordTy = bw == 16 ? "ushort" : (bw == 64 ? "ulong" : "uint");

    if (!op.getTimeout()) {
      // Without a timeout the poll can only complete on a match, so every
      // thread rendezvouses and the result is a compile-time true (no shared
      // memory needed to broadcast).
      os << ind() << "if (" << tidId << ".x == 0) {\n";
      ++indent;
      std::string want = fresh(), loadExpr;
      emitProbe(loadExpr);
      os << ind() << wordTy << " " << want << " = (" << wordTy << ")" << exp
         << ";\n";
      os << ind() << "while (" << loadExpr << " != " << want << ") {}\n";
      --indent;
      os << ind() << "}\n";
      os << ind() << "threadgroup_barrier(" << barrierFlags << ");\n";
      os << ind() << "bool " << result << " = true;\n";
      valMap[op.getResult()] = {result};
      return success();
    }

    // MSL has no in-kernel global timer; only timeout_ns==0 (single probe) is
    // expressible. The elected thread's match is broadcast through a scratch
    // threadgroup flag so every thread returns the same value.
    std::string flag = fresh();
    os << ind() << "threadgroup bool " << flag << ";\n";
    os << ind() << "if (" << tidId << ".x == 0) {\n";
    ++indent;
    std::string want = fresh(), loaded = fresh(), loadExpr;
    emitProbe(loadExpr);
    os << ind() << wordTy << " " << want << " = (" << wordTy << ")" << exp
       << ";\n";
    os << ind() << wordTy << " " << loaded << " = " << loadExpr << ";\n";
    os << ind() << flag << " = (" << loaded << " == " << want << ");\n";
    --indent;
    os << ind() << "}\n";
    os << ind() << "threadgroup_barrier(" << barrierFlags << ");\n";
    os << ind() << "bool " << result << " = " << flag << ";\n";
    valMap[op.getResult()] = {result};
    return success();
  }

  static std::string init0(const std::string &sc) {
    return sc == "float" || sc == "half" ? "0.0" : "0";
  }
};

} // namespace mlir::triton::applegpu

#endif // MSL_EMITTER_H
