// Widen num_stages=2 pipelined-dot SMEM staging from 1 slot to 2 rotating
// slots so the DotOp fast path can elide the per-dot post-load barrier.
//
// The upstream pipeliner allocates (num_stages - 1) staging slots, so at
// num_stages=2 the local_alloc carries a single slot and every K-step's
// async copy overwrites the strip the in-flight dot is still reading,
// forcing a TG barrier after the SG loads. With 2 rotating slots the next
// copy targets the other slot and the barrier is redundant (same shape as
// num_stages=3, where the DotOp lowering already drops it).
//
// The pipeliner already emits rotation counters for both the insert and
// extract index; with one slot they wrap at bound 1 and fold to 0. Widening
// is therefore: alloc shape[0] 1 -> 2 and the wrap bound 1 -> 2.
//
// Widening both operands doubles the staging footprint, and on big tiles
// (e.g. 128x128) that caps the core at one resident threadgroup, too few
// simds to hide SG-load latency. Above kAsymPreferBytes only one operand is
// widened (the smaller slot) and the other keeps one slot; if even that
// overflows kAsymPreferBytes the loop is left at single-slot staging, which
// outruns the 1-TG/core both-widened shape. Both operands share the
// pipeliner's rotation counters, so the widened operand gets private slot
// indices derived from the induction variable: i = (iv - lb) / step, read
// slot = i & 1, write slot = (i & 1) ^ 1, matching the counters' phase
// (prologue fills slot 0, first in-loop write rotates to slot 1). The other
// operand keeps the wrap-at-1 counters, which still fold to 0, and the
// lowering keeps its post-load barrier.

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "TritonAppleGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::applegpu {

#define GEN_PASS_DEF_WIDENPIPELINEDSTAGING
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = mlir::triton::gpu;

constexpr int64_t kTGBudgetBytes = 32 * 1024;
constexpr int64_t kAsymPreferBytes = 16 * 1024;

static std::optional<int64_t> constValue(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue();
  return std::nullopt;
}

// Match the pipeliner rotation idiom
//   %n = arith.addi %cnt, %c1
//   %w = arith.cmpi sge, %n, %bound
//   %i = arith.select %w, %c0, %n
// and return the cmp whose bound must be bumped to the new slot count.
static arith::CmpIOp matchRotation(Value idx) {
  auto sel = idx.getDefiningOp<arith::SelectOp>();
  if (!sel)
    return nullptr;
  auto cmp = sel.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::sge)
    return nullptr;
  auto zero = constValue(sel.getTrueValue());
  if (!zero || *zero != 0)
    return nullptr;
  auto add = sel.getFalseValue().getDefiningOp<arith::AddIOp>();
  if (!add || cmp.getLhs() != add.getResult())
    return nullptr;
  auto one = constValue(add.getRhs());
  if (!one || *one != 1)
    return nullptr;
  return cmp;
}

// The rotation counter must be loop-carried with the expected phase: the
// prologue fills slot 0, so the first in-loop write must rotate to slot 1
// (counter init 0) and the first read must hit slot 0 (counter init -1).
static bool counterInitIs(arith::CmpIOp cmp, scf::ForOp loop, int64_t want) {
  auto add = cmp.getLhs().getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;
  auto arg = dyn_cast<BlockArgument>(add.getLhs());
  if (!arg || arg.getOwner() != loop.getBody() || arg.getArgNumber() == 0)
    return false;
  auto init = constValue(loop.getInitArgs()[arg.getArgNumber() - 1]);
  return init && *init == want;
}

// The wrap bound must currently be 1: either a constant or a loop-carried
// value that is initialized to 1 and yields 1 every iteration.
static bool boundIsOne(Value bound, scf::ForOp loop) {
  if (auto c = constValue(bound))
    return *c == 1;
  auto arg = dyn_cast<BlockArgument>(bound);
  if (!arg || arg.getOwner() != loop.getBody() || arg.getArgNumber() == 0)
    return false;
  unsigned idx = arg.getArgNumber() - 1;
  auto init = constValue(loop.getInitArgs()[idx]);
  auto next = constValue(loop.getBody()->getTerminator()->getOperand(idx));
  return init && next && *init == 1 && *next == 1;
}

// Uniform mask: every leaf of the and/or tree is a scalar splat.
static bool maskIsUniform(Value mask) {
  if (mask.getDefiningOp<triton::SplatOp>())
    return true;
  if (auto andOp = mask.getDefiningOp<arith::AndIOp>())
    return maskIsUniform(andOp.getLhs()) && maskIsUniform(andOp.getRhs());
  if (auto orOp = mask.getDefiningOp<arith::OrIOp>())
    return maskIsUniform(orOp.getLhs()) && maskIsUniform(orOp.getRhs());
  return false;
}

static int64_t slotBytes(ttg::MemDescType ty) {
  int64_t elems = 1;
  for (int64_t d : ty.getShape().drop_front())
    elems *= d;
  return elems * ty.getElementType().getIntOrFloatBitWidth() / 8;
}

static int64_t totalBytes(ttg::MemDescType ty) {
  int64_t elems = 1;
  for (int64_t d : ty.getShape())
    elems *= d;
  return elems * ty.getElementType().getIntOrFloatBitWidth() / 8;
}

struct WidenPipelinedStaging
    : public impl::WidenPipelinedStagingBase<WidenPipelinedStaging> {
  void runOnOperation() override {
    getOperation().walk([&](scf::ForOp loop) {
      if (!dotsTakeSmemFastPath(loop))
        return;
      widenLoopStaging(loop);
      foldStagingLoadConverts(loop);
    });
  }

  // local_load -> convert_layout(blocked -> blocked) with the load's only
  // user being the convert: retype the load to the converted layout and drop
  // the convert. The convert otherwise reserves cross-warp scratch in
  // allocate-shared-memory (8KB at 128x16xf32) even though the dot lowering
  // peels it and reads the staging buffer directly.
  void foldStagingLoadConverts(scf::ForOp loop) {
    SmallVector<ttg::ConvertLayoutOp> deadCvts;
    loop.getBody()->walk([&](ttg::ConvertLayoutOp cvt) {
      auto load = cvt.getSrc().getDefiningOp<ttg::LocalLoadOp>();
      if (!load || !load.getResult().hasOneUse())
        return;
      auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
      auto dstTy = cast<RankedTensorType>(cvt.getType());
      if (!isa<ttg::BlockedEncodingAttr>(srcTy.getEncoding()) ||
          !isa<ttg::BlockedEncodingAttr>(dstTy.getEncoding()))
        return;
      for (OpOperand &use : cvt->getUses()) {
        auto dot = dyn_cast<DotOp>(use.getOwner());
        if (!dot || use.getOperandNumber() >= 2 ||
            !isa<AppleMmaEncodingAttr>(
                cast<RankedTensorType>(dot.getType()).getEncoding()))
          return;
      }
      load.getResult().setType(dstTy);
      cvt.getResult().replaceAllUsesWith(load.getResult());
      deadCvts.push_back(cvt);
    });
    for (auto cvt : deadCvts)
      cvt.erase();
  }

  // Mirror DotOpToLLVM's SMEM fast-path gates: widening only pays when the
  // dot SG-loads operands straight from the staging buffer. When the padded
  // resident grid fits the TG budget the lowering prefers the batched TG
  // strip path, where 2-slot staging buys nothing and only eats the strip
  // budget. The operand must also be structurally SG-loadable from staging
  // (resolveSmemOperand's checks).
  static bool smemResolvable(Value operand) {
    Value src = operand;
    if (auto cvt = src.getDefiningOp<ttg::ConvertLayoutOp>())
      src = cvt.getSrc();
    auto load = src.getDefiningOp<ttg::LocalLoadOp>();
    if (!load)
      return false;
    auto opTy = cast<RankedTensorType>(operand.getType());
    auto memTy = dyn_cast<ttg::MemDescType>(load.getSrc().getType());
    if (!memTy || memTy.getRank() != 2 || memTy.getShape() != opTy.getShape() ||
        memTy.getElementType() != opTy.getElementType() ||
        isa<IntegerType>(memTy.getElementType()))
      return false;
    auto shEnc = dyn_cast<ttg::SwizzledSharedEncodingAttr>(memTy.getEncoding());
    return shEnc && shEnc.getMaxPhase() == 1 && shEnc.getOrder().size() == 2 &&
           shEnc.getOrder()[0] == 1;
  }

  static bool dotsTakeSmemFastPath(scf::ForOp loop) {
    bool any = false, allFast = true;
    loop.getBody()->walk([&](DotOp dot) {
      if (!isa<AppleMmaEncodingAttr>(
              cast<RankedTensorType>(dot.getType()).getEncoding()))
        return;
      any = true;
      auto aTy = cast<RankedTensorType>(dot.getA().getType());
      auto bTy = cast<RankedTensorType>(dot.getB().getType());
      int64_t M = aTy.getDimSize(0), K = aTy.getDimSize(1);
      int64_t N = bTy.getDimSize(1);
      Type elemTy = aTy.getElementType();
      int64_t padEst = isa<IntegerType>(elemTy)
                           ? 0
                           : 16 / (elemTy.getIntOrFloatBitWidth() / 8);
      int64_t maxStrideEst = std::max(K, N);
      bool canPadEst =
          (padEst > 0) && ((8 * maxStrideEst + 8 * padEst) * 4 <= 16384);
      int64_t KpadEst = canPadEst ? K + padEst : K;
      int64_t NpadEst = canPadEst ? N + padEst : N;
      int64_t maxStripsEst = std::max(M / 8, K / 8);
      int64_t residentEst =
          std::max(maxStripsEst * 8 * std::max(KpadEst, NpadEst),
                   (M / 8) * 8 * NpadEst) *
          4;
      if (residentEst <= kTGResidentBudgetBytes ||
          !smemResolvable(dot.getA()) || !smemResolvable(dot.getB()))
        allFast = false;
    });
    return any && allFast;
  }

  struct StagedAlloc {
    ttg::LocalAllocOp alloc;
    llvm::SetVector<arith::CmpIOp> cmps;
    SmallVector<ttg::MemDescIndexOp> inserts;
    SmallVector<ttg::MemDescIndexOp> extracts;
  };

  // The staged alloc feeding every AppleMma dot's A operand in the loop (via
  // local_load <- memdesc_index <- alloc, optionally through a layout
  // convert). Null when the dots disagree or A is not staged.
  static ttg::LocalAllocOp dotAStagingAlloc(scf::ForOp loop) {
    ttg::LocalAllocOp aAlloc;
    bool ok = true;
    loop.getBody()->walk([&](DotOp dot) {
      if (!isa<AppleMmaEncodingAttr>(
              cast<RankedTensorType>(dot.getType()).getEncoding()))
        return;
      Value src = dot.getA();
      if (auto cvt = src.getDefiningOp<ttg::ConvertLayoutOp>())
        src = cvt.getSrc();
      auto load = src.getDefiningOp<ttg::LocalLoadOp>();
      auto idxOp = load ? load.getSrc().getDefiningOp<ttg::MemDescIndexOp>()
                        : ttg::MemDescIndexOp();
      auto alloc = idxOp ? idxOp.getSrc().getDefiningOp<ttg::LocalAllocOp>()
                         : ttg::LocalAllocOp();
      if (!alloc || (aAlloc && aAlloc != alloc))
        ok = false;
      else
        aAlloc = alloc;
    });
    return ok ? aAlloc : ttg::LocalAllocOp();
  }

  static void widenAlloc(ttg::LocalAllocOp alloc) {
    auto ty = cast<ttg::MemDescType>(alloc.getType());
    SmallVector<int64_t> shape(ty.getShape());
    shape[0] = 2;
    SmallVector<int64_t> allocShape(ty.getAllocShape());
    if (allocShape.size() == shape.size())
      allocShape[0] = 2;
    alloc.getResult().setType(ttg::MemDescType::get(
        shape, ty.getElementType(), ty.getEncoding(), ty.getMemorySpace(),
        ty.getMutableMemory(), allocShape));
  }

  void widenLoopStaging(scf::ForOp loop) {
    SmallVector<StagedAlloc> staged;
    int64_t extraBytes = 0, otherBytes = 0;

    auto func = loop->getParentOfType<FunctionOpInterface>();
    if (!func)
      return;

    func.walk([&](ttg::LocalAllocOp alloc) {
      auto ty = cast<ttg::MemDescType>(alloc.getType());
      bool candidate = !alloc.getSrc() && ty.getRank() == 3 &&
                       ty.getShape()[0] == 1 && ty.getMutableMemory();
      if (!candidate || alloc->getParentRegion() != loop->getParentRegion()) {
        otherBytes += totalBytes(ty);
        return;
      }

      bool touchesLoop = false, ok = true;
      StagedAlloc info{alloc, {}, {}, {}};
      for (Operation *user : alloc->getUsers()) {
        if (isa<ttg::LocalDeallocOp>(user))
          continue;
        auto idxOp = dyn_cast<ttg::MemDescIndexOp>(user);
        if (!idxOp) {
          ok = false;
          break;
        }
        if (!loop->isAncestor(idxOp)) {
          auto c = constValue(idxOp.getIndex());
          ok = c && *c == 0 && idxOp->isBeforeInBlock(loop);
          if (!ok)
            break;
          continue;
        }
        // Only widen DMA-fed staging. A per-element-masked copy lowers to the
        // sync scatter; rotating its slot makes the store offsets dynamic,
        // breaks vectorization, and runs slower than the barrier it would
        // elide. Uniform splat masks (the pipeliner's K>0 / last-iter guards)
        // keep the DMA.
        for (Operation *use : idxOp->getUsers())
          if (auto copy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(use))
            if (copy.getMask() && !maskIsUniform(copy.getMask())) {
              ok = false;
              break;
            }
        if (!ok)
          break;
        touchesLoop = true;
        auto cmp = matchRotation(idxOp.getIndex());
        if (!cmp || !boundIsOne(cmp.getRhs(), loop)) {
          ok = false;
          break;
        }
        for (Operation *use : idxOp->getUsers()) {
          int64_t wantInit;
          if (isa<ttg::AsyncCopyGlobalToLocalOp>(use)) {
            wantInit = 0;
            info.inserts.push_back(idxOp);
          } else if (isa<ttg::LocalLoadOp>(use)) {
            wantInit = -1;
            info.extracts.push_back(idxOp);
          } else {
            ok = false;
            break;
          }
          if (!counterInitIs(cmp, loop, wantInit)) {
            ok = false;
            break;
          }
        }
        if (!ok)
          break;
        info.cmps.insert(cmp);
      }
      if (ok && touchesLoop) {
        staged.push_back(std::move(info));
        extraBytes += 2 * slotBytes(ty);
      } else {
        otherBytes += slotBytes(ty);
      }
    });

    if (staged.empty())
      return;

    if (extraBytes > kAsymPreferBytes && staged.size() == 2) {
      switch (widenAsymmetric(loop, staged, otherBytes)) {
      case AsymResult::Applied:
      case AsymResult::KeepSingleSlot:
        return;
      case AsymResult::Fallback:
        break;
      }
    }

    if (otherBytes + extraBytes > kTGBudgetBytes)
      return;

    OpBuilder b(loop.getBody(), loop.getBody()->begin());
    Value two = arith::ConstantIntOp::create(b, loop.getLoc(), 2, 32);
    for (StagedAlloc &info : staged)
      for (arith::CmpIOp cmp : info.cmps)
        cmp.getRhsMutable().assign(two);
    for (StagedAlloc &info : staged)
      widenAlloc(info.alloc);
  }

  enum class AsymResult { Applied, KeepSingleSlot, Fallback };

  // Widen only one operand to 2 slots; the other keeps 1 slot. The rotation
  // counters are shared between the operands, so the widened operand gets
  // private slot indices derived from the induction variable (i = (iv - lb) /
  // step, exact for scf.for): the extract reads slot i & 1 and the insert
  // writes (i & 1) ^ 1, the same phase as the counters (prologue fills slot
  // 0). The shared counters keep their wrap-at-1 bound and fold to 0 for the
  // single-slot operand, whose post-load barrier the lowering keeps.
  //
  // The smaller slot is widened (the dot's A operand on a tie) to keep the
  // total under kAsymPreferBytes; above that the asymmetric copy buys no
  // occupancy and single-slot staging outruns it, so leave the loop alone.
  AsymResult widenAsymmetric(scf::ForOp loop, SmallVector<StagedAlloc> &staged,
                             int64_t otherBytes) {
    auto aAlloc = dotAStagingAlloc(loop);
    if (!aAlloc)
      return AsymResult::Fallback;
    auto lb = constValue(loop.getLowerBound());
    auto step = constValue(loop.getStep());
    if (!lb || *lb < 0 || !step || *step <= 0)
      return AsymResult::Fallback;
    Type ivTy = loop.getInductionVar().getType();
    for (StagedAlloc &info : staged)
      if (info.inserts.empty() || info.extracts.empty() ||
          ivTy != info.extracts.front().getIndex().getType())
        return AsymResult::Fallback;

    auto slot = [](StagedAlloc &info) {
      return slotBytes(cast<ttg::MemDescType>(info.alloc.getType()));
    };
    StagedAlloc *wide = &staged[0];
    if (slot(staged[1]) < slot(*wide) ||
        (slot(staged[1]) == slot(*wide) && staged[1].alloc == aAlloc))
      wide = &staged[1];
    int64_t total =
        otherBytes + slot(staged[0]) + slot(staged[1]) + slot(*wide);
    if (total > kAsymPreferBytes)
      return AsymResult::KeepSingleSlot;

    Location loc = loop.getLoc();
    OpBuilder b(loop.getBody(), loop.getBody()->begin());
    auto cst = [&](int64_t v) {
      return arith::ConstantOp::create(b, loc, ivTy, IntegerAttr::get(ivTy, v));
    };
    Value i = loop.getInductionVar();
    if (*lb != 0)
      i = arith::SubIOp::create(b, loc, i, cst(*lb));
    if (*step != 1)
      i = arith::DivUIOp::create(b, loc, i, cst(*step));
    Value readSlot = arith::AndIOp::create(b, loc, i, cst(1));
    Value writeSlot = arith::XOrIOp::create(b, loc, readSlot, cst(1));
    for (ttg::MemDescIndexOp idxOp : wide->extracts)
      idxOp.getIndexMutable().assign(readSlot);
    for (ttg::MemDescIndexOp idxOp : wide->inserts)
      idxOp.getIndexMutable().assign(writeSlot);
    widenAlloc(wide->alloc);
    return AsymResult::Applied;
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createWidenPipelinedStagingPass() {
  return std::make_unique<WidenPipelinedStaging>();
}

} // namespace mlir::triton::applegpu
