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

// fresh() over the emitter's live id counter (by reference): builders mint real
// names that advance the caller's id in lockstep.
struct LocalGen {
  int &id;
  std::string fresh() { return "v" + std::to_string(id++); }
};

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

// arith add/sub/mul/div/rem/and/or/xor op -> msl::BinOp.
msl::BinOp arithBinOp(Operation *op);

// Forwarding stream: `os` writes here, and the sink can be retargeted so a
// design-sanctioned raw leaf's text can be captured into a buffer (RawStmt).
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
    msl::MSLPrinter preamblePrinter(os);
    preamblePrinter.printPreamble();

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

  // Capture the text `fn` writes to `os` into a verbatim RawStmt. Used for the
  // design-sanctioned raw leaves (fp-emulated CAS, fp-narrowing) that have no
  // dedicated node kind.
  template <typename Fn> msl::Stmt *captureRaw(Fn &&fn) {
    std::string buf;
    llvm::raw_string_ostream ss(buf);
    llvm::raw_ostream *prev = fwd.swap(&ss);
    fn();
    ss.flush();
    fwd.swap(prev);
    return ctx.rawVerbatim(buf);
  }
  // AST arena. Builders allocate nodes into it and the printer renders them.
  msl::MSLContext ctx;

  // AST-typed forms of the string type helpers (mslScalarType/mslUnsignedType/
  // mslStorageType). Definitions in MSLTypes.cpp.
  msl::Type *astScalarType(Type t);
  msl::Type *astUnsignedType(Type t);
  msl::Type *astStorageType(Type t);

  // Per-element value builders for the reshape/aliasing ops. makeRange yields
  // `start + off`; splat/reshape/join/split just alias a source register name.
  msl::Expr *astMakeRangeElem(int start, msl::Expr *off);
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

  // AST forms of the offset/address helpers (defs in EmitMSLMemory.cpp);
  // insert explicit parens where a subexpression needs precedence grouping.
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

  // AST builders for the dot/GEMM simdgroup-matrix lowering (defs in
  // EmitMSLDot.cpp).
  msl::MatrixType *astSgFragType(Type t);
  msl::Expr *astFragAddr(StringRef base, int64_t off);
  msl::Stmt *astFragDecl(msl::MatrixType *frag, StringRef name);
  msl::Stmt *astAccFragDecl(msl::MatrixType *frag, StringRef name);
  msl::Stmt *astSgLoad(StringRef frag, StringRef base, int64_t off, int64_t ld);
  msl::Stmt *astSgStore(StringRef acc, StringRef base, int64_t off, int64_t ld);
  msl::Stmt *astSgMultiplyAccumulate(StringRef acc, StringRef a, StringRef b);
  msl::Expr *astReadbackValue(StringRef buf, msl::Expr *off, StringRef base);

  // atomic builders (EmitMSLAtomic.cpp). Multi-statement builders return a
  // msl::Block so the caller splices them at its own indent (no stray braces);
  // single-node builders return Stmt*/Expr*.
  msl::Expr *astInit0(StringRef sc);
  msl::Expr *astDeviceAtomicPtr(StringRef atomicTy, StringRef p);
  msl::Expr *astCasWeak(msl::Expr *ptr, StringRef expVar, msl::Expr *newVal);
  msl::Expr *astPacked16Extract(StringRef word, StringRef isHigh);
  msl::Expr *astPacked16Merge(StringRef word, StringRef isHigh,
                              msl::Expr *newBitsU32);
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

  // reduce / scan / map builders (EmitMSLReduce.cpp). Multi-statement builders
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

  // function / scope / control-flow builders (EmitMSLFunc.cpp). Each takes a
  // pre-built body Block and returns the scope node: astEmitOp walks the ops
  // into the body, then wraps it here.
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
  void astCopyRegs(msl::Block &out, ArrayRef<std::string> dst,
                   ArrayRef<std::string> src);
  msl::Block astBranchEdge(Block *succ, Operation::operand_range args,
                           StringRef state);
  msl::Stmt *astCondBranch(Value cond, msl::Block thenB, msl::Block elseB);
  msl::Block astYieldAssign(Operation *term,
                            ArrayRef<SmallVector<std::string>> dsts);
  // Helpers below mint fresh names over a raw id counter passed by reference
  // (seeded from nextId by the caller) so they never mutate nextId directly.
  llvm::SmallVector<msl::Stmt *, 2> laneWarpProlog();
  llvm::SmallVector<msl::Param, 8> deviceParams(tt::FuncOp func, int &id,
                                                bool bind);
  bool astElemwiseDecls(Operation *op, msl::Type *declTy, int &id,
                        msl::Block &body,
                        llvm::function_ref<msl::Expr *(int)> mk);
  bool astDeclBind(Operation *op, msl::Type *declTy, msl::Block &body,
                   llvm::function_ref<msl::Expr *(int)> mk);
  // Dispatch spine: appends the nodes for `op` to `body`. Returns true when the
  // op is handled (nodes appended, or nothing for alias/dataless ops); false for
  // an unsupported op, which astWalkBlock turns into a hard error.
  bool astEmitOp(Operation *op, msl::Block &body);
  std::optional<bool> astEmitArithBinop(Operation *op, msl::Block &body);
  std::optional<bool> astEmitConstGrid(Operation *op, msl::Block &body);
  std::optional<bool> astEmitArithMisc(Operation *op, msl::Block &body);
  std::optional<bool> astEmitMath(Operation *op, msl::Block &body);
  std::optional<bool> astEmitReshape(Operation *op, msl::Block &body);
  std::optional<bool> astEmitMemDesc(Operation *op, msl::Block &body);
  std::optional<bool> astEmitDotMap(Operation *op, msl::Block &body);
  std::optional<bool> astEmitAtomic(Operation *op, msl::Block &body);
  std::optional<bool> astEmitScanReduce(Operation *op, msl::Block &body);
  std::optional<bool> astEmitTensorMove(Operation *op, msl::Block &body);
  std::optional<bool> astEmitCallReturn(Operation *op, msl::Block &body);
  std::optional<bool> astEmitControlFlow(Operation *op, msl::Block &body);
  void astProgramDim(Operation *op, StringRef builtinVar, tt::ProgramIDDim axis,
                     msl::Block &body);
  msl::Block astWalkBlock(Block &blk, unsigned depth);
  msl::Block astEmitBlockCFG(Region &region);
  void astEmitMapCFG(Region &region, ArrayRef<std::string> capture,
                     msl::Block &body);
  msl::Block astWalkBlock2(Block &blk,
                           llvm::DenseMap<Value, SmallVector<std::string>> &hoist);
  msl::Block astTerminatorEdge(Operation *term, StringRef state);
  SmallVector<std::string> astDeclResultVars(Value v, msl::Block &body);
  msl::Expr *astDerefPtr(Value ptr, StringRef name, StringRef scName);
  void astStoreBody(tt::StoreOp op, msl::Block &body);
  bool astCombineN(Region &region, ArrayRef<std::string> aVals,
                   ArrayRef<std::string> bVals, msl::Block &body,
                   SmallVectorImpl<std::string> &results);
  struct InPlaceOperand {
    std::string buf;
    std::string baseOffset;
  };
  bool astEmitDot(tt::DotOp op, msl::Block &body);
  bool astEmitFusedGemm(scf::ForOp op, tt::DotOp dot, unsigned iterIdx,
                        msl::Block &body);
  bool astEmitDotScalar(tt::DotOp op, msl::Block &body);
  void astStageOperand(msl::Block &body, StringRef tgName, Value stage,
                       RankedTensorType stageTy, ArrayRef<std::string> names,
                       bool skip, llvm::function_ref<msl::Expr *(int)> guard);
  bool astEmitDotPanel(tt::DotOp op, msl::Block &body, Value aStage,
                       Value bStage, ArrayRef<std::string> aNames,
                       ArrayRef<std::string> bNames, ArrayRef<std::string> cInit,
                       ArrayRef<std::string> ids, int64_t M, int64_t N, int64_t K,
                       int64_t Bd, int rank, msl::MatrixType *opFrag,
                       StringRef opScalar, int64_t numWarps, StringAttr rowDim,
                       StringAttr colDim);
  bool astEmitDotFused(
      tt::DotOp op, msl::Block &body, Value aStage, Value bStage,
      ArrayRef<std::string> aNames, ArrayRef<std::string> bNames, StringRef tgA,
      StringRef tgB, StringRef tgC, ArrayRef<std::string> ids, int64_t M,
      int64_t N, int64_t K, int64_t mT, int64_t nT, int64_t kT, int64_t nFrag,
      int64_t numWarps, std::optional<InPlaceOperand> aInPlace,
      std::optional<InPlaceOperand> bInPlace, msl::MatrixType *opFrag,
      msl::MatrixType *accFragTy,
      llvm::function_ref<void(msl::Block &, int64_t, int64_t, int64_t)>
          readbackInto);
  std::string astShuffle(StringRef op, StringRef sc, StringRef val,
                         StringRef arg, msl::Block &body);
  bool astScanWarpCarry(Region &region, int nOp, ArrayRef<std::string> scTys,
                        ArrayRef<int64_t> byteWidths,
                        ArrayRef<std::pair<int, int32_t>> warpBits,
                        ArrayRef<int> regs,
                        SmallVector<SmallVector<std::string>> &accs,
                        ArrayRef<std::string> laneScan, StringRef axisTopLane,
                        unsigned axisWarpMask, int numWarps, bool rev,
                        SmallVectorImpl<std::string> &runTotalOut,
                        msl::Block &body);
  void astBandRoundTrip(msl::Block &body, StringRef buf, int64_t total,
                        int64_t band, int srcRc, int resRc,
                        ArrayRef<std::string> outs,
                        llvm::function_ref<msl::Expr *(int)> srcOff,
                        llvm::function_ref<msl::Expr *(int)> srcVal,
                        llvm::function_ref<msl::Expr *(int)> resOff);
  bool emitFailed = false;

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
      msl::Expr *addr = ctx.paren(
          ctx.binary(msl::BinOp::Add, ctx.var(argbufId),
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
      prologue.push_back(ctx.arrayDeclStmt(ctx.named("threadgroup char"),
                                           poolBuf, kernelPool));
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



  LogicalResult emitDeviceFuncProto(tt::FuncOp func, bool asDecl) {
    if (failed(declRetStruct(func)))
      return failure();
    msl::MSLPrinter printer(os);
    printer.printProto(astDeviceProto(func));
    return success();
  }

  LogicalResult emitDeviceFunc(tt::FuncOp func) {
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
    msl::DeviceFn *fn = ctx.deviceFn(astDeviceRetType(func),
                                     mslDeviceFuncName(func.getName()), params,
                                     std::move(prologue));
    msl::MSLPrinter printer(os);
    printer.print(fn);
    return success();
  }

  tt::FuncOp curDevFunc;
  std::string devPoolPtr;






  std::string tgposId, tidId, numTgId, laneId, warpId;
  bool scalarSpinlock = false;

  // Register-resident C GEMM fusion. When the scf.for handler recognises an
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
  // by op), each with the runtime predicate under which it did so. The tt.store
  // handler guards the store on the negation to avoid a double write.
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
    if (getenv("MSL_NO_DIRECT_STORE")) // escape hatch: fall back to pool readback
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


  std::string floatLit(const APFloat &v);
  std::string floatLit(const APFloat &v, Type ty);
  msl::Expr *astFloatLit(const APFloat &v, Type ty);





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

  // AST sub-expression builders: yield the RHS Expr* the decl init uses,
  // resolving operands from already-looked-up register names.
  msl::Expr *astIntBinaryExpr(Operation *op, StringRef a, StringRef b);
  msl::Expr *astShiftExpr(Operation *op, StringRef a, StringRef b);
  msl::Expr *recast(msl::Type *opCast, StringRef n);
  msl::Expr *astElementwiseExpr(msl::BinOp op, msl::Type *opCast, StringRef a,
                                StringRef b);
  msl::Expr *astMinMaxExpr(StringRef fn, msl::Type *opCast, bool propagateNan,
                           StringRef a, StringRef b);
  msl::Expr *astUnaryExpr(StringRef fn, msl::Type *sc, StringRef v);
  msl::Expr *astTernaryCallExpr(StringRef fn, StringRef a, StringRef b,
                                StringRef c);

  msl::Expr *astCastExpr(Operation *op, StringRef v);
  msl::Expr *astPtrIntCastExpr(Operation *op, StringRef v);
  msl::Expr *astBitcastExpr(Operation *op, StringRef v);
  msl::Expr *astClampExpr(tt::ClampFOp op, StringRef x, StringRef lo,
                          StringRef hi);
  msl::Expr *astSelectExpr(StringRef c, StringRef t, StringRef f);




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
    if (getenv("MSL_NO_FUSE")) // escape hatch: fall back to the per-dot path
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
    // larger falls back to the per-dot path. Staging bytes mirror the dot path: an
    // operand already resident in a threadgroup buffer (in-place) stages 0.
    int64_t aBytes = M * K * (bitsOf(aElem) / 8);
    int64_t bBytes = N * K * (bitsOf(aElem) / 8);
    int64_t cFull = M * N * 4;
    bool wholeTileFits = aBytes + bBytes <= 32768;
    // A/B that structurally resolve to a local_alloc buffer are loaded in place
    // by astEmitDot (stage 0). The precise in-place base lives in memdescMap, which
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

  // Single-expression form of the warp shuffle (no temporaries): the
  // bitcast-through wrapper collapsed into one nested Expr. `sc` picks the
  // reinterpret path (64-bit/bfloat scalars reject the intrinsic directly).
  msl::Expr *astShuffleExpr(StringRef op, StringRef sc, StringRef val,
                            StringRef arg);



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







  // Design-sanctioned Raw escape-hatch leaves (see EmitMSLRawLeaves.cpp): the
  // fp32->half/bfloat narrowing (rtne full / rtz / rtne integer) and the
  // emulated fp32 / packed-fp16 CAS loops.
  std::string emitRoundedHalfValueFull(const std::string &sc,
                                       const std::string &v);
  std::string emitTruncatedFloatValue(const std::string &sc,
                                      const std::string &v);
  std::string emitRoundedHalfValue(const std::string &sc, const std::string &v);
  std::pair<std::string, std::string> emitPacked16Base(const std::string &p,
                                                       const std::string &sc);
  void emitPacked16CASLoop(const std::string &wordPtr, const std::string &isHigh,
                           const std::string &sc, const std::string &curId,
                           const std::string &newHalfExpr,
                           const std::string &id);
  void emitFloat32CASLoop(const std::string &p, const std::string &curId,
                          const std::string &newFloatExpr,
                          const std::string &id);

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






  static std::string init0(const std::string &sc) {
    return sc == "float" || sc == "half" ? "0.0" : "0";
  }
};

} // namespace mlir::triton::applegpu

#endif // MSL_EMITTER_H
