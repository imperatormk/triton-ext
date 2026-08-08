// MSLEmitter.h - TritonGPU IR -> MSL lowering state (the emitter class).
//
// Extracted from EmitMSL.cpp. Method definitions live in EmitMSL.cpp; the
// AST type helpers (scalarType/...) live in MSLTypes.cpp.
#ifndef MSL_EMITTER_H
#define MSL_EMITTER_H

#include "MSLAst.h"
#include "MSLAtomic.h"
#include "MSLConstants.h"
#include "MSLFusedDot.h"
#include "MSLLayoutExpr.h"
#include "MSLPrinter.h"
#include "MSLShufflePlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <string>

namespace mlir::triton::applegpu {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

bool mslLogReject();
void mslReject(Operation *op, StringRef gate, StringRef reason);

// fresh() over the emitter's live id counter (by reference): builders mint real
// names that advance the caller's id in lockstep.
struct LocalGen {
  int &id;
  std::string fresh() { return "v" + std::to_string(id++); }
};

inline bool isFp8Type(Type t) {
  return isa<Float8E4M3FNType, Float8E5M2Type>(t);
}

inline std::string mslScalarType(Type t) {
  if (t.isF32() || t.isF64())
    return "float";
  if (t.isF16())
    return "half";
  if (t.isBF16())
    return "bfloat";
  if (isFp8Type(t))
    return "uchar";
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

  LogicalResult emit();

private:
  msl::HelperSet scanHelpers();

  ModuleOp mod;
  ForwardOStream fwd;
  raw_ostream &os;

  // Pointer contiguity/alignment for load vectorization. Built lazily on the
  // first load, since most kernels reach emit() without needing it.
  std::unique_ptr<tt::ModuleAxisInfoAnalysis> axisInfo;
  tt::ModuleAxisInfoAnalysis &getAxisInfo();

  // AST arena. Builders allocate nodes into it and the printer renders them.
  msl::MSLContext ctx;

  // AST-typed forms of the string type helpers (mslScalarType/mslUnsignedType/
  // mslStorageType). Definitions in MSLTypes.cpp.
  msl::Type *scalarType(Type t);
  msl::Type *unsignedType(Type t);
  msl::Type *storageType(Type t);

  // Per-element value builders for the reshape/aliasing ops. makeRange yields
  // `start + off`; splat/reshape/join/split just alias a source register name.
  msl::Expr *makeRangeElem(int start, msl::Expr *off);
  msl::Expr *aliasElem(StringRef name);

  int nextId = 0;
  int indent = 1;
  llvm::DenseMap<Value, SmallVector<std::string>> valMap;

  llvm::DenseMap<Value, MemDescInfo> memdescMap;

  // Layout coordinate / flat-offset exprs come from `layout` directly.
  // poolRegion/memdescElemAddr stay here (they touch pool/memdesc state, not
  // just layout coords).
  msl::Expr *poolRegion(int64_t byteOffset, StringRef sc);
  static SmallVector<int64_t> memdescStrides(ttg::MemDescType mt);
  static bool barrierCoversTail(const msl::Block &body);
  // Set once a batch of async copies has been fenced, cleared at the wait that
  // closes the batch, so its members share a single fence.
  bool asyncCopyFenced = false;
  // Event tokens of DMA copies issued since the last wait.
  SmallVector<std::string> pendingDmaHandles;
  // Tokens needing a prologue declaration (issue and wait can be in different
  // scopes).
  SmallVector<std::string> dmaHandleDecls;
  // One event token per async-copy site, reused across trips.
  llvm::DenseMap<Operation *, std::string> dmaHandleFor;
  msl::Expr *memdescElemAddr(const MemDescInfo &info, RankedTensorType tileTy,
                             int reg);

  // AST builders for the dot/GEMM simdgroup-matrix lowering (defs in
  // EmitMSLDot.cpp).
  msl::MatrixType *sgFragType(Type t);
  msl::Expr *fragAddr(StringRef base, int64_t off);
  msl::Stmt *fragDecl(msl::MatrixType *frag, StringRef name);
  msl::Stmt *accFragDecl(msl::MatrixType *frag, StringRef name);
  msl::Stmt *sgLoad(StringRef frag, StringRef base, int64_t off, int64_t ld,
                    bool transpose = false);
  msl::Expr *uniformSplatScalar(Value v);
  msl::Expr *scalarEpilogueExpr(Operation *op, msl::Expr *cur, msl::Expr *rhs);
  msl::Stmt *sgStore(StringRef acc, StringRef base, int64_t off, int64_t ld);
  msl::Stmt *sgMultiplyAccumulate(StringRef acc, StringRef a, StringRef b);
  msl::Expr *readbackValue(StringRef buf, msl::Expr *off, StringRef base);

  // Atomic RMW / CAS / poll / histogram node trees are built by the
  // AtomicEmitter; emitAtomic calls through the `atomic` member.
  AtomicEmitter atomic{ctx, nextId, tidId, poolBuf};

  // reduce / scan / map builders (EmitMSLReduce.cpp). Multi-statement builders
  // return a msl::Block spliced at caller indent; single-node builders return
  // Stmt*/Expr*. The cross-lane shuffles go through shuffleExpr with the
  // builtin::simd names; control flow is real If/State-machine node trees.
  msl::Stmt *mapCaptureDecl(StringRef sc, StringRef name);
  msl::Stmt *mapReturnSpill(StringRef capture, StringRef operand);
  msl::Stmt *
  mapCFGStateMachine(StringRef state,
                     ArrayRef<std::pair<std::string, msl::Block>> cases);

  // function / scope / control-flow builders (EmitMSLFunc.cpp). Each takes a
  // pre-built body Block and returns the scope node: emitOp walks the ops
  // into the body, then wraps it here.
  msl::DeviceFn *deviceProto(tt::FuncOp func);
  msl::NamedType *retStructType(tt::FuncOp func);
  msl::Stmt *retStructDecl(tt::FuncOp func); // ArrayDecl-less; RawStmt struct
  msl::Type *deviceRetType(tt::FuncOp func);
  msl::Stmt *emitReturn(tt::ReturnOp op);
  msl::ForScope *forScope(scf::ForOp op, msl::Block body, StringRef iv,
                          StringRef ivTy);
  msl::TripCountForScope *tripCountForScope(scf::ForOp op, msl::Block body,
                                            StringRef counter, StringRef iv,
                                            StringRef ivTy);
  msl::Stmt *forNode(scf::ForOp op, msl::Block body, StringRef iv, StringRef tc,
                     StringRef ivTy, bool wideIv);
  msl::IfScope *ifScope(scf::IfOp op, msl::Block thenB, msl::Block elseB);
  msl::WhileScope *whileScope(scf::WhileOp op, msl::Block body);
  void copyRegs(msl::Block &out, ArrayRef<std::string> dst,
                ArrayRef<std::string> src);
  msl::Block branchEdge(Block *succ, Operation::operand_range args,
                        StringRef state);
  msl::Stmt *condBranch(Value cond, msl::Block thenB, msl::Block elseB);
  msl::Block yieldAssign(Operation *term,
                         ArrayRef<SmallVector<std::string>> dsts);
  // Helpers below mint fresh names over a raw id counter passed by reference
  // (seeded from nextId by the caller) so they never mutate nextId directly.
  llvm::SmallVector<msl::Stmt *, 2> laneWarpProlog();
  llvm::SmallVector<msl::Param, 8> deviceParams(tt::FuncOp func, int &id,
                                                bool bind);
  bool elemwiseDecls(Operation *op, msl::Type *declTy, int &id,
                     msl::Block &body, llvm::function_ref<msl::Expr *(int)> mk);
  bool declBind(Operation *op, msl::Type *declTy, msl::Block &body,
                llvm::function_ref<msl::Expr *(int)> mk);
  // Dispatch spine: appends the nodes for `op` to `body`. Returns true when the
  // op is handled (nodes appended, or nothing for alias/dataless ops); false
  // for an unsupported op, which walkBlock turns into a hard error.
  bool emitOp(Operation *op, msl::Block &body);
  std::optional<bool> emitArithBinop(Operation *op, msl::Block &body);
  std::optional<bool> emitConstGrid(Operation *op, msl::Block &body);
  std::optional<bool> emitArithMisc(Operation *op, msl::Block &body);
  std::optional<bool> emitMath(Operation *op, msl::Block &body);
  std::optional<bool> emitReshape(Operation *op, msl::Block &body);
  std::optional<bool> emitMemDesc(Operation *op, msl::Block &body);
  std::optional<bool> emitDotMap(Operation *op, msl::Block &body);
  std::optional<bool> emitAtomicRMW(Operation *op, msl::Block &body);
  std::optional<bool> emitAtomicPoll(Operation *op, msl::Block &body);
  std::optional<bool> emitAtomicCAS(Operation *op, msl::Block &body);
  std::optional<bool> emitScan(Operation *op, msl::Block &body);
  std::optional<bool> emitReduce(Operation *op, msl::Block &body);
  std::optional<bool> emitTensorMove(Operation *op, msl::Block &body);
  std::optional<bool> emitCallReturn(Operation *op, msl::Block &body);
  std::optional<bool> emitControlFlow(Operation *op, msl::Block &body);
  void programDim(Operation *op, StringRef builtinVar, tt::ProgramIDDim axis,
                  msl::Block &body);
  msl::Block walkBlock(Block &blk, unsigned depth);
  msl::Block emitBlockCFG(Region &region);
  void emitMapCFG(Region &region, ArrayRef<std::string> capture,
                  msl::Block &body);
  msl::Block walkBlock2(Block &blk,
                        llvm::DenseMap<Value, SmallVector<std::string>> &hoist);
  msl::Block terminatorEdge(Operation *term, StringRef state);
  SmallVector<std::string> declResultVars(Value v, msl::Block &body);
  msl::Expr *derefPtr(Value ptr, StringRef name, StringRef scName);
  // Largest power-of-two run of consecutive registers of a tensor of type
  // `valueTy` accessed through `ptr` that is contiguous in memory, hence
  // transferable as one vecN. 1 = no vectorization.
  int accessVectorWidth(Type valueTy, Value ptr, bool *packed = nullptr);
  int loadVectorWidth(tt::LoadOp ld, bool *packed = nullptr);
  int storeVectorWidth(tt::StoreOp st);

  // `x % c` with x already known below c: the op is dropped and its result
  // aliases the input registers.
  std::optional<bool> foldBoundedRem(arith::RemSIOp op);
  std::optional<bool> clampTileOrigin(arith::MulIOp mul, msl::Type *declTy,
                                      msl::Block &body);
  bool lhsHasClampedOrigin(Value v);
  bool lhsHasGridBound(Value v, int64_t c);
  // The extent a `pid * BLOCK` tile origin is indexed against, found by walking
  // forward to the remainder that consumes it, or 0 if it reaches none.
  int64_t tileOriginExtent(Value origin, int64_t &block);
  // Tile origins whose grid is sized by the extent they index into, mapped to
  // that extent. A remainder by exactly this extent cannot wrap.
  DenseMap<Value, int64_t> gridBoundedOrigins;
  // Tile origins pulled back so a boundary tile lands inside the dimension;
  // a remainder on one of these can no longer wrap.
  DenseSet<Value> clampedOrigins;
  void storeBody(tt::StoreOp op, msl::Block &body);
  bool combineN(Region &region, ArrayRef<std::string> aVals,
                ArrayRef<std::string> bVals, msl::Block &body,
                SmallVectorImpl<std::string> &results);
  // All strategy selection and budget arithmetic for one tt.dot, with no
  // emission and no fresh() consumption. emitDot is a dispatch over the
  // returned plan.
  DotPlan planDot(tt::DotOp op);
  bool emitDot(tt::DotOp op, msl::Block &body);
  bool emitFusedGemm(scf::ForOp op, tt::DotOp dot, unsigned iterIdx,
                     msl::Block &body);
  bool emitDotScalar(tt::DotOp op, msl::Block &body);
  void stageOperand(msl::Block &body, StringRef tgName, Value stage,
                    RankedTensorType stageTy, ArrayRef<std::string> names,
                    bool skip, llvm::function_ref<msl::Expr *(int)> guard,
                    int64_t rowPad = 0);
  bool emitDotPanel(tt::DotOp op, msl::Block &body, const DotPlan &plan,
                    const DotEmitCtx &dc);
  bool
  emitDotFused(tt::DotOp op, msl::Block &body, const DotPlan &plan,
               const DotEmitCtx &dc,
               llvm::function_ref<void(msl::Block &, int64_t, int64_t, int64_t)>
                   readbackInto);
  bool emitDotDirect(tt::DotOp op, msl::Block &body, const DotPlan &plan,
                     const DotEmitCtx &dc);
  void dotPoolPtrs(msl::Block &body, tt::DotOp op, const DotPlan &plan,
                   DotEmitCtx &dc);
  static bool dmaStagingEnabled();
  std::optional<DirectStage> bDmaCandidate(tt::DotOp op,
                                           bool requireBound = true);
  static bool aDirectEnabled();
  std::optional<DirectStage> aDirectCandidate(tt::DotOp op, int64_t M,
                                              int64_t K, int64_t numWarps,
                                              bool requireBound = true);
  // The one place the device-direct A decision is made. planDot and scanPool
  // both route through it, so the pool reservation cannot disagree with the
  // emission about whether A claims a region.
  std::optional<DirectStage> dotADirect(tt::DotOp op,
                                        bool knownFusedAcc = false);
  std::optional<DirectStage> dotDmaStage(tt::DotOp op);
  msl::Stmt *dmaBeginInto(StringRef handle, const DotPlan &plan,
                          const DotEmitCtx &dc, const DirectStage &ds,
                          RankedTensorType bStageTy, int64_t ldb,
                          StringRef tripVar, bool nextTrip);
  StringRef dotDmaTripVar(Operation *op);
  msl::Expr *dmaTileOrigin(const DirectStage &ds, StringRef tripVar);
  msl::Stmt *dmaBegin(StringRef handle, StringRef tgBuf, int64_t pitch,
                      msl::Expr *src, const DirectStage &ds, int64_t elemBytes);
  static std::string dmaCallee(int64_t elemBytes, bool transposed);
  msl::Expr *dmaRowStride(const DirectStage &ds);
  msl::Stmt *dmaWait(StringRef handle);
  std::optional<DirectStage>
  asyncCopyDma(ttg::AsyncCopyGlobalToLocalOp ac);
  bool dmaCopyEligible(ttg::AsyncCopyGlobalToLocalOp ac);
  SmallVector<std::string> dotResultIds(msl::Block &body, tt::DotOp op,
                                        const DotPlan &plan);
  void dotReadback(msl::Block &tgt, RankedTensorType cTy, StringRef tgC,
                   ArrayRef<std::string> ids, ArrayRef<std::string> base,
                   int rank, int64_t N, int64_t bi, int64_t r0, int64_t r1,
                   StringAttr rowDim, StringAttr colDim);
  std::string shuffle(StringRef op, StringRef sc, StringRef val, StringRef arg,
                      msl::Block &body);
  bool scanWarpCarry(Region &region, int nOp, ArrayRef<std::string> scTys,
                     ArrayRef<int64_t> byteWidths,
                     ArrayRef<std::pair<int, int32_t>> warpBits,
                     ArrayRef<int> regs,
                     SmallVector<SmallVector<std::string>> &accs,
                     ArrayRef<std::string> laneScan, StringRef axisTopLane,
                     unsigned axisWarpMask, int numWarps, bool rev,
                     SmallVectorImpl<std::string> &runTotalOut,
                     msl::Block &body);
  void bandRoundTrip(msl::Block &body, StringRef buf, int64_t total,
                     int64_t band, int srcRc, int resRc,
                     ArrayRef<std::string> outs,
                     llvm::function_ref<msl::Expr *(int)> srcOff,
                     llvm::function_ref<msl::Expr *(int)> srcVal,
                     llvm::function_ref<msl::Expr *(int)> resOff);

  // Shared tt.trans / tt.reshape lowering: stage src regs into a pool buffer at
  // srcOff(r), then read them back at the result's row-major flat offset. Only
  // the source-offset mapping differs between the two ops.
  void emitTileRoundTrip(Value res, Value src, RankedTensorType srcTy,
                         RankedTensorType resTy, msl::Block &body,
                         llvm::function_ref<msl::Expr *(int)> srcOff);
  // Assigning true prints the stack that set it when TRITON_MSL_TRACE_FAIL is
  // set: emission failure otherwise unwinds through several `return false`
  // hops and surfaces only as "PassManager::run failed".
  struct FailFlag {
    bool v = false;
    FailFlag &operator=(bool b) {
      if (b && !v && getenv("TRITON_MSL_TRACE_FAIL")) {
        llvm::errs() << "[msl] emitFailed set here:\n";
        llvm::sys::PrintStackTrace(llvm::errs());
      }
      v = b;
      return *this;
    }
    operator bool() const { return v; }
  };
  FailFlag emitFailed;

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

  SmallVector<std::string> &names(Value v) { return valMap[v]; }

  // Rotated dot-operand loads held back by walkBlock so the fused MMA phase
  // can emit them after its publish barrier (see deferPrefetchLoad).
  SmallVector<Operation *> pendingPrefetch;
  bool deferPrefetchLoad(Operation *op);
  void emitPendingPrefetchLoads(msl::Block &body);

  // The single register name of a scalar / single-register value.
  StringRef scalarName(Value v) { return valMap[v][0]; }

  // Broadcast-aware per-register operand pick: a splat has 1 name, a tensor
  // has regCount names.
  StringRef reg(Value v, int r) {
    auto &nm = valMap[v];
    return nm[nm.size() == 1 ? 0 : r];
  }

  void bindScalar(Value v, std::string name) {
    valMap[v] = SmallVector<std::string>{std::move(name)};
  }

  // Bind a value to its per-register name vector.
  void bindRegs(Value v, SmallVector<std::string> regs) {
    valMap[v] = std::move(regs);
  }

  // Bind a value carrying no materialized register (memdesc / dataless).
  void bindDataless(Value v) { valMap[v] = SmallVector<std::string>{}; }

  // Bind dst to the same register names as src.
  void bindAlias(Value dst, Value src) { valMap[dst] = valMap[src]; }

  LogicalResult emitFunc(tt::FuncOp func);

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

  LogicalResult declRetStruct(tt::FuncOp func);

  LogicalResult emitDeviceFuncProto(tt::FuncOp func, bool asDecl);

  LogicalResult emitDeviceFunc(tt::FuncOp func);

  tt::FuncOp curDevFunc;
  std::string devPoolPtr;

  std::string tgposId, tidId, numTgId, laneId, warpId;
  bool scalarSpinlock = false;
  // Scalar kernel-arg pointers a loop both loads and thread-predicates a store
  // to. The macOS 26 Metal compiler forwards a stale value into such a load
  // across the guarded store, dropping an iteration's update; a coherent deref
  // blocks the forwarding. Keyed by the traced-to base kernel argument.
  DenseSet<Value> coherentScalarPtrs;
  bool needsCoherentDeref(Value ptr) const;

  LayoutExprBuilder layout{ctx, laneId, warpId, tgposId};

  // Register-resident C GEMM fusion state (FusedDotCtx / FusedDotPhase /
  // DirectStore in MSLFusedDot.h). The scf.for handler drives an
  // `acc = tl.dot(a, b, acc)` K-loop through the three-phase path: Decl
  // (persistent simdgroup fragments, pre-loop), MMA (stage A/B + accumulate,
  // per iteration, no tgC round-trip), Readback (store fragments + gather the
  // #mma->scalar result, post-loop).
  FusedDotCtx fusedDot;
  // Terminal stores the fused readback already wrote directly to device (keyed
  // by op), each with the runtime predicate under which it did so. The tt.store
  // handler guards the store on the negation to avoid a double write.
  DenseMap<Operation *, std::string> directStoreHandled;
  DenseSet<Operation *> directStoreRaggedHandled;
  // Scalar defs hoisted ahead of the walk so the fused epilogue can name them;
  // the walk must not emit them a second time.
  DenseSet<Operation *> preEmitted;

  bool preEmitScalarChain(Value v, msl::Block &body);

  static bool tracesToKernelArg(Value v);

  static Value traceToKernelArg(Value v);

  static Value peelBroadcastExpand(Value v);

  // A tile-uniform integer: a splat of an SSA scalar, or a splat constant.
  static UniformInt matchUniformInt(Value v);

  // A per-tile index `pidBase*TILE + iota` where iota is a 0-based make_range.
  // Returns the scalar tile-base value (pidBase*TILE) or null.
  static Value matchTileIndex(Value v);

  // Split an addptr offset `add(rowTerm, colTerm)` into (rowBase, ldc, colBase)
  // where rowTerm = broadcast(rowIdx * splat(ldc)) and colTerm =
  // broadcast(colIdx), each index a per-tile `base + iota`. ldc must be a
  // scalar (loop-invariant); col stride is the implicit 1 of row-major.
  bool matchRowMajorOffset(Value off, Value &rowBase, UniformInt &ldc,
                           Value &colBase);

  // The store boundary mask `(row < boundM) && (col < boundN)`, with row/col
  // the same per-tile indices as the address. Extracts the two bounds.
  bool matchBoundaryMask(Value m, Value rowBase, Value colBase,
                         UniformInt &boundM, UniformInt &boundN,
                         Value &tileGuard);

  // True when a boundary-masked store can never actually be ragged: both bounds
  // are compile-time literals, and each tile origin is provably a multiple of
  // its block size (`pid * BLOCK`), which the bound divides exactly. Both halves
  // are required -- a literal bound alone says nothing about where the tile
  // starts. Anything dynamic or indivisible keeps the fallback arm and its pool
  // reservation.
  struct PoolRegion {
    const char *name;
    int64_t begin;
    int64_t bytes;
  };
  msl::Stmt *bFragLoad(const DotEmitCtx &dc, int64_t ki, msl::Expr *niExpr,
                       int64_t niLit, StringRef fb, int64_t ldb);
  void checkPoolRegions(Operation *op, ArrayRef<PoolRegion> live);
  void checkDotPoolRegions(tt::DotOp op, const DotPlan &plan);
  bool rowBoundNeverRagged(const DirectStage &ds, int64_t M);
  bool dotRaggedDrainAffine(tt::DotOp d);
  bool directStoreRowNeverRagged(const DirectStore &ds, int64_t M);
  bool directStoreColNeverRagged(const DirectStore &ds, int64_t N);
  bool directStoreNeverRagged(const DirectStore &ds, int64_t M, int64_t N);

  int stageVectorWidth(RankedTensorType stageTy, int regs, int64_t rowPad);

  // Recognise `acc(#mma) -> [narrowing] -> convert_layout ->
  // tt.store(rowmajor)` as the sole consumer of the fused accumulator. An
  // optional float narrowing (f32 -> f16/bf16) is threaded through and applied
  // per fragment element on the way out; anything else (reduction, second dot,
  // extra users) falls back to the pool readback.
  std::optional<DirectStore> matchDirectStore(Value forResult);

  // Recognise `acc + <row-broadcast bias>`: a bias vector with one value per
  // column, replicated down every row. Fills ds.bias* and returns the addf
  // result to continue matching from. Returns null when `v`'s user is not such
  // an add, leaving `ds` untouched.
  Value matchRowBroadcastBias(Value v, DirectStore &ds);

  static bool isPureBarrierOp(Operation *op);

  std::string floatLit(const APFloat &v);
  std::string floatLit(const APFloat &v, Type ty);
  msl::Expr *floatLitExpr(const APFloat &v, Type ty);

  LogicalResult emitSplat(tt::SplatOp op);

  // Map result registers to source registers by matching per-register
  // coordinates. Handles tt.expand_dims (insert a size-1 dim at `axis`) and
  // tt.broadcast (replicate size-1 source dims). Value carries through; only
  // the register->register permutation/replication changes.
  LogicalResult emitReshapeLike(Value res, Value src, int axis, bool isExpand);

  // tt.join(a, b): both operands share a layout; result adds a trailing size-2
  // dim whose two entries live in the same thread (distinct registers). Result
  // register r with trailing coord t sources from operand t at the result
  // coords minus the trailing dim.
  LogicalResult emitJoin(tt::JoinOp op);

  // tt.split(x): inverse of join. Two results share a layout; result k register
  // r sources from x at the result coords with trailing coord k appended.
  LogicalResult emitSplit(tt::SplitOp op);

  // AST sub-expression builders: yield the RHS Expr* the decl init uses,
  // resolving operands from already-looked-up register names.
  msl::Expr *intBinaryExpr(Operation *op, StringRef a, StringRef b);
  msl::Expr *shiftExpr(Operation *op, StringRef a, StringRef b);
  msl::Expr *recast(msl::Type *opCast, StringRef n);
  msl::Expr *elementwiseExpr(msl::BinOp op, msl::Type *opCast, StringRef a,
                             StringRef b);
  msl::Expr *minMaxExpr(StringRef fn, msl::Type *opCast, bool propagateNan,
                        StringRef a, StringRef b);
  msl::Expr *cmpFNaNGuard(msl::Expr *cmp, StringRef a, StringRef b,
                          bool ordered);
  msl::Expr *unaryExpr(StringRef fn, msl::Type *sc, StringRef v);
  msl::Expr *ternaryCallExpr(StringRef fn, StringRef a, StringRef b,
                             StringRef c);

  msl::Expr *castExpr(Operation *op, StringRef v);
  msl::Expr *ptrIntCastExpr(Operation *op, StringRef v);
  msl::Expr *bitcastExpr(Operation *op, StringRef v);
  msl::Expr *clampExpr(tt::ClampFOp op, StringRef x, StringRef lo,
                       StringRef hi);
  msl::Expr *selectExpr(StringRef c, StringRef t, StringRef f);

  static bool isDatalessType(Type t);

  // Recognise the register-resident GEMM shape: a loop-carried #mma iter-arg
  // that is the C operand of the first tt.dot in the body, where each further
  // dot accumulates into the previous one's result and the last result is
  // yielded back as that same iter-arg. Returns the last dot of the chain
  // (the one the epilogue reads) and the iter-arg index, or nullopt.
  std::optional<std::pair<tt::DotOp, unsigned>> matchGemmDotLoop(scf::ForOp op);

  // True when this dot is the accumulator of a fused GEMM K-loop, i.e. the
  // one emitFusedGemm will route through its post-loop epilogue.
  bool dotIsFusedGemmAcc(tt::DotOp d);
  // True when every convert_layout the fused accumulator flows into lowers to
  // a lane permutation, so the C tile never reaches threadgroup memory and the
  // pool does not have to reserve it.
  bool fusedGemmCIsShuffled(tt::DotOp d);
  bool fusedGemmCHasFallback(tt::DotOp d);
  bool convertLayoutIsDirectStoreC(ttg::ConvertLayoutOp c);
  // Elements of threadgroup scratch the ragged arm needs. It drains one 8x8
  // fragment per warp at a time, so this is numWarps*64 rather than the whole
  // MxN tile; 0 when the arm does not take that path.
  int64_t fusedGemmCCompactScratch(tt::DotOp d);
  bool cStoresDirect(tt::DotOp d);

  // Bitmask over lane (or warp) bits that reduce the given axis: a bit reduces
  // if its LinearLayout basis maps to a nonzero coordinate on the reduced
  // out-dim, i.e. distinct lanes/warps hold distinct axis elements.
  unsigned reduceMask(const tt::LinearLayout &ll, StringAttr inDim,
                      StringAttr outDim);

  // All distinct values reachable by ORing subsets of the mask's set bits.
  // Used to enumerate the reduced-axis warp partition offsets while keeping the
  // surviving-axis warp bits fixed, so a cross-warp combine never mixes warps
  // that hold different output elements.
  SmallVector<int> subsetsOf(unsigned mask, int numWarps);

  // Ordered (bit, axisStride) pairs for an inDim (lane/warp), only for bits
  // that move along the scanned axis. Sorted by ascending axis stride.
  SmallVector<std::pair<int, int32_t>>
  axisBits(const tt::LinearLayout &ll, StringAttr inDim, StringAttr outDim);

  // Single-expression form of the warp shuffle (no temporaries): the
  // bitcast-through wrapper collapsed into one nested Expr. `sc` picks the
  // reinterpret path (64-bit/bfloat scalars reject the intrinsic directly).
  msl::Expr *shuffleExpr(StringRef op, StringRef sc, StringRef val,
                         StringRef arg);

  // Lowers a convert_layout as a lane permutation when planIntraWarpShuffle
  // accepts it. Returns false (emitting nothing) otherwise, so the caller falls
  // through to the threadgroup round-trip.
  bool emitIntraWarpShuffleConvert(ttg::ConvertLayoutOp cl, msl::Block &body);

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
  int64_t poolBudget() const;

  llvm::DenseMap<Block *, std::string> blockLabel;
  std::string cfgState;

  static int64_t bitsOf(Type t);
  static int64_t byteWidth(Type t) { return bitsOf(t) / 8; }

  // Elements per band for a threadgroup-staged reshape whose full tile exceeds
  // the 32KB budget: the largest chunk of flat offsets that fits.
  static int64_t reshapeBandElems(int64_t totalElems, int64_t elemBytes,
                                  int64_t budget = kTGResidentBudgetBytes);

  // Flat element band for a threadgroup-staged reshape/trans round-trip,
  // clamped to the live pool budget so emission bands identically to how
  // scanPool/tgScratchBytes sized the pool (no sizing/lowering drift).
  int64_t reshapeBand(int64_t totalElems, int64_t elemBytes) const {
    return totalElems * elemBytes > poolBudget()
               ? reshapeBandElems(totalElems, elemBytes, poolBudget())
               : totalElems;
  }

  // Full flat size (product of shape) of a tensor tile.
  int64_t tileSize(RankedTensorType rt);

  // Threadgroup scratch bytes for a full tile of `ty`, clamped to the live pool
  // budget (poolBudget()) when it overflows. band2D uses the rk>=2 row-band
  // clamp (last dim intact); otherwise the flat reshape-band clamp.
  int64_t tgScratchBytes(RankedTensorType ty, bool band2D);

  // Peak byte footprint of a single transient scratch site.
  void scanPool(Operation *op);

  // Pipelined (software-pipeliner) ops lower to SYNCHRONOUS threadgroup
  // staging: MSL has no async copy and M3+ dropped the DMA hardware, so the
  // rotating multi-buffer collapses to a plain threadgroup buffer written by a
  // masked per-thread copy and read back with a barrier between.
  static int64_t memdescFlatSize(ttg::MemDescType mt);

  LogicalResult emitMemDescIndex(ttg::MemDescIndexOp op);

  LogicalResult emitMemDescSubslice(ttg::MemDescSubsliceOp op);

  // A local_load of a contiguous row-major [rows][cols] threadgroup buffer
  // (local_alloc, optionally memdesc_index'd, never subsliced) reached through
  // convert_layouts, or null.
  static ttg::LocalLoadOp dotOperandLocalLoad(Value operand, int64_t rows,
                                              int64_t cols);

  static bool dotReadsOperandInPlace(tt::DotOp d, Value operand);

  static bool convertLayoutIsDeadDotStage(ttg::ConvertLayoutOp c);

  static bool localLoadIsDeadDotStage(ttg::LocalLoadOp ll);

  std::optional<InPlaceOperand>
  dotOperandInPlaceBuf(Value operand, int64_t rows, int64_t cols);

  msl::Expr *inPlaceBase(const InPlaceOperand &op);

  // A dot operand reached through a convert_layout of a rank-2 distributed
  // tensor. The dot stages its operand into threadgroup memory by row-major
  // (m,k) offset regardless of the operand's distributed layout, so it can read
  // the CONVERT SOURCE registers at the source layout's row-major offsets and
  // produce a bit-identical tile, making the convert_layout's own threadgroup
  // round-trip dead weight. Returns the source value to stage, or null.
  static Value dotOperandConvertSource(tt::DotOp d, Value operand);

  // The convert_layout is dead when every use is a dot that will stage the
  // convert source directly (dotOperandConvertSource matches). Emitting its
  // round-trip is then pure waste.
  static bool convertLayoutIsDeadDotStageSource(ttg::ConvertLayoutOp c);

  // Lower tt.dot to MSL simdgroup_matrix 8x8 fragment MMA. A (MxK) and B (KxN)
  // per-thread registers are staged row-major into threadgroup memory; one
  // simdgroup cooperatively runs the 8x8 fragment MMA loop over K into an MxN
  // accumulator, stores it to threadgroup, and every thread reads its C result
  // registers back. Emits simdgroup_load / simdgroup_multiply_accumulate /
  // simdgroup_store only.
  static bool isDotOperandElem(Type t);
  static std::string sgOperandScalar(Type t);

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
                           int64_t accBytes, int64_t &mp, int64_t &np);

  static bool dotNeedsPanel(int64_t M, int64_t N, int64_t K, int64_t elemBytes,
                            int64_t accBytes);

  static int64_t dotCBandRows(int64_t M, int64_t N, int64_t cBudget,
                              int64_t accBytes);

  // Simdgroup-matrix registers a warp block may keep live before reload
  // traffic becomes the cheaper trade.
  static constexpr int64_t kTargetLiveFrags = 16;

  // Columns per fragment-cache scope: the widest divisor of nT whose chunk*kT
  // B fragments stay under kTargetLiveFrags.
  static int64_t dotColChunk(int64_t nT, int64_t kT);
};

} // namespace mlir::triton::applegpu

#endif // MSL_EMITTER_H
