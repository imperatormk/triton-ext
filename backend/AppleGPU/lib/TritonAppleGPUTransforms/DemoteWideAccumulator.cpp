// Demote a wide loop-carried literal-struct accumulator (an LLVM-dialect block
// argument of type !llvm.struct<(T x N)> for large N) to scalarized per-element
// allocas that the loop load/modify/stores in place.
//
// WHY THIS EXISTS
// ---------------
// A rolled `scf.for` whose iter_arg is a large per-thread tensor slice (e.g.
// the depthwise-conv1d accumulator, a tensor<16x64x64xf32> that lowers to a
// 1024-element per-thread struct) becomes, after scf->cf + TritonGPU->LLVM
// lowering, a loop with a single literal-struct block argument:
//
//   ^bb1(%iv: i32, %acc: !llvm.struct<(f32 x1024)>):     ; the phi
//     ... %e = llvm.extractvalue %acc[i] ...             ; 1024 reads
//     ... %acc.next = llvm.insertvalue ..., i ...        ; 1024 writes (undef
//     base) llvm.br ^bb1(%iv.next, %acc.next)
//
// LLVM's interprocedural SCCP (part of optimize_module(O3), the same call the
// NVIDIA backend makes) NON-TERMINATES on this shape: getStructValueState /
// visitInsertValueInst spin on the 1024-field aggregate lattice. Confirmed:
// `opt -O3` on the pre-opt module hangs at 100% CPU forever.
//
// THE FIX (robust by construction)
// --------------------------------
// We replace the wide struct phi with N scalar allocas in the entry block:
//   * the preheader edge stores the init value's N fields into the slots,
//   * every `extractvalue %acc[i]` becomes `load slot_i`,
//   * the latch edge stores the next value's N fields into the slots, with the
//     field value pulled directly out of the producing insertvalue chain so no
//     wide aggregate value is ever materialized, and
//   * the struct is dropped from the block-argument and branch-operand lists.
//
// After this there is no struct-typed SSA value crossing the loop at all: only
// scalar allocas with scalar load/stores, which mem2reg/SROA handle trivially
// and which present IPSCCP with N independent scalar lattices instead of one
// N-field aggregate lattice. The wide-aggregate phi cannot be reintroduced
// because the carried value's type is now plain scalars in memory.
//
// The pass only fires above kMinStructFields, so small loop-carried tensors
// (the common case) are left byte-for-byte unchanged.

#include "TritonAppleGPUTransforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::applegpu {

#define GEN_PASS_DEF_DEMOTEWIDEACCUMULATOR
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace {

namespace LLVM = mlir::LLVM;

// IPSCCP's aggregate lattice blows up well before 1024; 64 fields is already
// far past anything a normal scalar accumulator produces, and small enough to
// leave ordinary loop-carried tensors untouched.
constexpr int64_t kMinStructFields = 64;

// A literal struct of all-identical scalar element type, or nullopt.
static std::optional<std::pair<Type, int64_t>> wideScalarStruct(Type t) {
  auto st = dyn_cast<LLVM::LLVMStructType>(t);
  if (!st || st.isOpaque() || st.isPacked())
    return std::nullopt;
  auto body = st.getBody();
  if (static_cast<int64_t>(body.size()) < kMinStructFields)
    return std::nullopt;
  Type elem = body.front();
  if (!elem.isIntOrFloat())
    return std::nullopt;
  for (Type f : body)
    if (f != elem)
      return std::nullopt;
  return std::make_pair(elem, static_cast<int64_t>(body.size()));
}

// Resolve field `idx` of a value known to be a literal struct: walk back
// through the insertvalue chain so we never have to read it with extractvalue
// (which would keep the aggregate alive). Returns null if the field can't be
// resolved structurally (caller then falls back to extractvalue).
static Value resolveField(Value structVal, int64_t idx) {
  Value cur = structVal;
  while (auto ins = cur.getDefiningOp<LLVM::InsertValueOp>()) {
    auto pos = ins.getPosition();
    if (pos.size() == 1 && pos[0] == idx)
      return ins.getValue();
    cur = ins.getContainer();
  }
  return nullptr; // hit a phi / undef / call result we can't see through
}

// Per-edge: store every field of `structVal` into the slots, just before
// `insertPt` (the terminator of the predecessor block).
static void storeFields(OpBuilder &b, Location loc, Value structVal,
                        ArrayRef<Value> slots, Operation *insertPt) {
  b.setInsertionPoint(insertPt);
  for (int64_t i = 0, n = slots.size(); i < n; ++i) {
    Value v = resolveField(structVal, i);
    if (!v)
      v = LLVM::ExtractValueOp::create(b, loc, structVal, ArrayRef<int64_t>{i});
    LLVM::StoreOp::create(b, loc, v, slots[i]);
  }
}

struct DemoteWideAccumulator
    : public impl::DemoteWideAccumulatorBase<DemoteWideAccumulator> {
  void runOnOperation() override {
    getOperation().walk([&](LLVM::LLVMFuncOp func) { processFunc(func); });
  }

  void processFunc(LLVM::LLVMFuncOp func) {
    if (func.isExternal())
      return;
    // Collect wide struct block-args first; we mutate the CFG as we go.
    SmallVector<std::pair<Block *, unsigned>> targets;
    for (Block &blk : func.getBody()) {
      if (blk.isEntryBlock())
        continue;
      for (unsigned i = 0, e = blk.getNumArguments(); i < e; ++i)
        if (wideScalarStruct(blk.getArgument(i).getType()))
          targets.push_back({&blk, i});
    }
    if (targets.empty())
      return;
    for (auto [blk, argIdx] : targets)
      demote(func, blk, argIdx);

    // After demotion the latch's wide-struct construction (the insertvalue
    // chain that fed the old phi, plus its undef/zero base) is dead — its only
    // consumer was the branch operand we dropped. It must be ERASED, not just
    // left dead: IPSCCP still walks the lattice of a 1024-field aggregate even
    // when the result is unused, which is exactly the non-termination we are
    // eliminating.
    //
    // Collect every wide-aggregate op once, then erase in REVERSE program order
    // (uses before defs): an insertvalue chain is built defs-then-uses, so the
    // chain head (last built, the dead root) comes last and is erased first,
    // dropping the use on the next link, and so on — a single linear sweep, no
    // re-walk. A use-less op stays use-less, so this can't erase a live value.
    SmallVector<Operation *> wideOps;
    func.walk([&](Operation *op) {
      if (!isa<LLVM::InsertValueOp, LLVM::ExtractValueOp, LLVM::UndefOp,
               LLVM::ZeroOp>(op))
        return;
      bool wide = false;
      for (Type t : op->getResultTypes())
        if (wideScalarStruct(t))
          wide = true;
      for (Value o : op->getOperands())
        if (wideScalarStruct(o.getType()))
          wide = true;
      if (wide)
        wideOps.push_back(op);
    });
    for (Operation *op : llvm::reverse(wideOps))
      if (op->use_empty())
        op->erase();
  }

  void demote(LLVM::LLVMFuncOp func, Block *blk, unsigned argIdx) {
    BlockArgument arg = blk->getArgument(argIdx);
    auto [elemTy, n] = *wideScalarStruct(arg.getType());

    // Every predecessor must reach `blk` via an unconditional llvm.br (the
    // canonical loop shape this targets). Bail otherwise — correctness over
    // coverage; the only consumer is the rolled-accumulator loop.
    SmallVector<LLVM::BrOp> preds;
    for (Block *pred : blk->getPredecessors()) {
      auto br = dyn_cast<LLVM::BrOp>(pred->getTerminator());
      if (!br || br.getDest() != blk)
        return;
      preds.push_back(br);
    }
    if (preds.empty())
      return;

    OpBuilder b(func.getContext());
    Location loc = arg.getLoc();

    // N scalar slots in the entry block.
    Block &entry = func.getBody().front();
    b.setInsertionPointToStart(&entry);
    Value one = LLVM::ConstantOp::create(b, loc, b.getI32Type(),
                                         b.getI32IntegerAttr(1));
    auto ptrTy = LLVM::LLVMPointerType::get(func.getContext());
    SmallVector<Value> slots;
    slots.reserve(n);
    // build(res, arraySize, alignment(IntegerAttr, optional), elem_type)
    for (int64_t i = 0; i < n; ++i)
      slots.push_back(LLVM::AllocaOp::create(b, loc, ptrTy, /*arraySize=*/one,
                                             /*alignment=*/IntegerAttr(),
                                             /*elem_type=*/elemTy));

    // Each predecessor stores the carried struct's fields, then drops the
    // struct operand from the branch.
    for (LLVM::BrOp br : preds) {
      Value carried = br.getDestOperands()[argIdx];
      storeFields(b, loc, carried, slots, br);
      SmallVector<Value> newOps(br.getDestOperands().begin(),
                                br.getDestOperands().end());
      newOps.erase(newOps.begin() + argIdx);
      b.setInsertionPoint(br);
      LLVM::BrOp::create(b, br.getLoc(), newOps, blk);
      br.erase();
    }

    // Replace extractvalue(arg, i) consumers with load slot_i; any residual
    // direct use gets a freshly reloaded struct (rebuilt from the slots) so the
    // rewrite stays correct even if some user isn't a single-index extract.
    b.setInsertionPointToStart(blk);
    SmallVector<Operation *> extractUsers;
    bool hasOtherUse = false;
    for (Operation *user : arg.getUsers()) {
      auto ev = dyn_cast<LLVM::ExtractValueOp>(user);
      if (ev && ev.getPosition().size() == 1)
        extractUsers.push_back(ev);
      else
        hasOtherUse = true;
    }
    for (Operation *u : extractUsers) {
      auto ev = cast<LLVM::ExtractValueOp>(u);
      b.setInsertionPoint(ev);
      Value ld = LLVM::LoadOp::create(b, ev.getLoc(), elemTy,
                                      slots[ev.getPosition()[0]]);
      ev.getResult().replaceAllUsesWith(ld);
      ev.erase();
    }
    if (hasOtherUse) {
      b.setInsertionPointToStart(blk);
      Value rebuilt = LLVM::UndefOp::create(b, loc, arg.getType());
      for (int64_t i = 0; i < n; ++i) {
        Value ld = LLVM::LoadOp::create(b, loc, elemTy, slots[i]);
        rebuilt = LLVM::InsertValueOp::create(b, loc, rebuilt, ld,
                                              ArrayRef<int64_t>{i});
      }
      arg.replaceAllUsesWith(rebuilt);
    }

    // Drop the now-dead struct block argument.
    blk->eraseArgument(argIdx);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createDemoteWideAccumulatorPass() {
  return std::make_unique<DemoteWideAccumulator>();
}

} // namespace mlir::triton::applegpu
