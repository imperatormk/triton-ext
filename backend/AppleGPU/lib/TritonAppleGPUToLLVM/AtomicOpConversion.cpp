#include "ConvertCommon.h"
#include "Dialect/TritonAppleGPU/IR/AppleMmaFragment.h"
#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "TritonAppleGPUToLLVM/Passes.h"
#include "TritonAppleGPUToLLVM/TargetInfo.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::applegpu {

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
namespace ttg = mlir::triton::gpu;

namespace {

// Lower triton::AtomicRMWOp → air.atomic.global.{op}.{type}
//
// Metal uses explicit AIR intrinsics for atomics:
//   float @air.atomic.global.add.f32(float addrspace(1)*, float, i32 order, i32
//   scope, i1 volatile) i32   @air.atomic.global.add.s.i32(i32 addrspace(1)*,
//   i32, i32 order, i32 scope, i1 volatile) i32
//   @air.atomic.global.max.s.i32(...) i32   @air.atomic.global.min.s.i32(...)
//   i32   @air.atomic.global.xchg.s.i32(...)
//
// For unsupported native atomics (f32 max/min, f16/bf16 add), we emit a CAS
// loop:
//   air.atomic.global.cmpxchg.weak.i32(ptr, expected_ptr, desired, succ_order,
//   fail_order, scope, vol) returns old i32. Expected is passed by pointer and
//   updated on failure.
struct AtomicRMWOpAppleConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LLVMFuncOp declareAIR(ConversionPatternRewriter &rewriter, ModuleOp mod,
                        StringRef name, Type retTy,
                        ArrayRef<Type> argTys) const {
    if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
      return fn;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto fnTy = LLVMFunctionType::get(retTy, argTys, false);
    return LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                              Linkage::External);
  }

  Value emitDirectAtomic(ConversionPatternRewriter &rewriter, Location loc,
                         ModuleOp mod, StringRef airName, Type valueTy,
                         Value ptr) const {
    auto *ctx = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto fn = declareAIR(rewriter, mod, airName, valueTy,
                         {ptrTy, valueTy, i32Ty, i32Ty, i1Ty});
    (void)fn;
    return {};
  }

  // Emit CAS loop for f32 max/min:
  //   alloca expected
  //   load old from *ptr (via xchg 0 trick or just initial load)
  //   loop:
  //     store old → expected
  //     new_f = max/min(old_f, val_f)
  //     new_i = bitcast new_f → i32
  //     old_i = bitcast old_f → i32
  //     store old_i → expected
  //     old_ret = cmpxchg(ptr_i32, &expected, new_i, ...)
  //     expected_after = load expected
  //     cmp = icmp eq old_ret, old_i (success if unchanged)
  //     br cmp → done, loop
  //   done:
  //     result = bitcast old_ret → float
  Value emitF32CASLoop(ConversionPatternRewriter &rewriter, Location loc,
                       ModuleOp mod, Value ptr, Value val, RMWOp rmwOp) const {
    auto *ctx = rewriter.getContext();
    auto f32Ty = Float32Type::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);  // device
    auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0); // private (alloca)

    auto cmpxchgFn =
        declareAIR(rewriter, mod, "air.atomic.global.cmpxchg.weak.i32", i32Ty,
                   {ptrTy, ptrTy0, i32Ty, i32Ty, i32Ty, i32Ty, i1Ty});

    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    Value expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, i32Ty,
                                                  one, /*alignment=*/4);

    // Non-atomic initial guess is fine — the CAS loop retries if it's stale.
    Value oldI32 = LLVM::LoadOp::create(rewriter, loc, i32Ty, ptr);
    Value oldF32 = LLVM::BitcastOp::create(rewriter, loc, f32Ty, oldI32);

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *loopBlock = rewriter.createBlock(afterBlock);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{oldF32, oldI32}, loopBlock);

    // Loop block: phi for old_f32, old_i32
    loopBlock->addArgument(f32Ty, loc);
    loopBlock->addArgument(i32Ty, loc);
    Value phiOldF32 = loopBlock->getArgument(0);
    Value phiOldI32 = loopBlock->getArgument(1);

    rewriter.setInsertionPointToStart(loopBlock);

    Value newF32;
    if (rmwOp == RMWOp::MAX)
      newF32 = LLVM::MaximumOp::create(rewriter, loc, phiOldF32, val);
    else
      newF32 = LLVM::MinimumOp::create(rewriter, loc, phiOldF32, val);

    Value newI32 = LLVM::BitcastOp::create(rewriter, loc, i32Ty, newF32);

    // Store expected (old) into the by-pointer alloca for cmpxchg.
    LLVM::StoreOp::create(rewriter, loc, phiOldI32, expectedAlloca);

    // cmpxchg(ptr, &expected, desired, succ_order=0, fail_order=0, scope=2,
    // vol=true)
    Value order0 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope2 = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value volT = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    Value oldRet =
        LLVM::CallOp::create(rewriter, loc, cmpxchgFn,
                             ValueRange{ptr, expectedAlloca, newI32, order0,
                                        order0, scope2, volT})
            .getResult();

    // Success iff old_ret == expected, i.e. no other thread changed the word.
    Value success = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                         oldRet, phiOldI32);

    // On failure, old_ret carries the current value to retry against.
    Value failedOldF32 = LLVM::BitcastOp::create(rewriter, loc, f32Ty, oldRet);

    LLVM::CondBrOp::create(rewriter, loc, success, afterBlock, ValueRange{},
                           loopBlock, ValueRange{failedOldF32, oldRet});

    // afterBlock carries the matched old value out via a block arg.
    afterBlock->addArgument(f32Ty, loc);
    rewriter.eraseOp(success.getDefiningOp()->getBlock()->getTerminator());
    LLVM::CondBrOp::create(rewriter, loc, success, afterBlock,
                           ValueRange{phiOldF32}, loopBlock,
                           ValueRange{failedOldF32, oldRet});

    rewriter.setInsertionPointToStart(afterBlock);
    return afterBlock->getArgument(0);
  }

  // Emit CAS loop for f16/bf16 atomic add. Metal has NO i16 cmpxchg, so we
  // widen to i32: load the aligned i32 word, extract the target half, compute,
  // pack back, cmpxchg i32. The f16/bf16 element may sit at an odd offset, so:
  //   - aligned_ptr = ptr & ~3   (align down)
  //   - byte_offset = ptr & 3    → 0 or 2
  //   - shift       = byte_offset * 8  → 0 or 16
  //   - mask        = 0xFFFF << shift
  Value emitF16BF16CASLoop(ConversionPatternRewriter &rewriter, Location loc,
                           ModuleOp mod, Value ptr, Value val, Type elemTy,
                           RMWOp rmwOp) const {
    auto *ctx = rewriter.getContext();
    auto i16Ty = IntegerType::get(ctx, 16);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0);

    auto cmpxchgFn =
        declareAIR(rewriter, mod, "air.atomic.global.cmpxchg.weak.i32", i32Ty,
                   {ptrTy, ptrTy0, i32Ty, i32Ty, i32Ty, i32Ty, i1Ty});

    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    Value expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, i32Ty,
                                                  one, /*alignment=*/4);

    Value ptrInt = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptr);
    Value three64 = arith::ConstantIntOp::create(rewriter, loc, 3, 64);
    Value byteOff64 = LLVM::AndOp::create(rewriter, loc, ptrInt, three64);
    Value notThree64 = arith::ConstantIntOp::create(rewriter, loc, ~3LL, 64);
    Value alignedInt = LLVM::AndOp::create(rewriter, loc, ptrInt, notThree64);
    Value alignedPtr =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, alignedInt);
    Value byteOff32 = LLVM::TruncOp::create(rewriter, loc, i32Ty, byteOff64);
    Value eight = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
    Value shift = LLVM::MulOp::create(rewriter, loc, byteOff32, eight);
    Value mask16 = arith::ConstantIntOp::create(rewriter, loc, 0xFFFF, 32);
    Value mask = LLVM::ShlOp::create(rewriter, loc, mask16, shift);
    Value notMask = LLVM::XOrOp::create(
        rewriter, loc, mask,
        arith::ConstantIntOp::create(rewriter, loc, -1, 32));

    // Compute the add in f32, then narrow back to the element type.
    Value valF32;
    if (elemTy.isF16())
      valF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, val);
    else // bf16
      valF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, val);

    Value oldI32 = LLVM::LoadOp::create(rewriter, loc, i32Ty, alignedPtr);

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *loopBlock = rewriter.createBlock(afterBlock);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{oldI32}, loopBlock);

    loopBlock->addArgument(i32Ty, loc);
    Value phiOldI32 = loopBlock->getArgument(0);

    rewriter.setInsertionPointToStart(loopBlock);

    // Extract the target i16 half from the i32 word, widen to f32.
    Value shifted = LLVM::LShrOp::create(rewriter, loc, phiOldI32, shift);
    Value oldI16 = LLVM::TruncOp::create(rewriter, loc, i16Ty, shifted);

    Value oldF32;
    if (elemTy.isF16()) {
      Value oldF16 =
          LLVM::BitcastOp::create(rewriter, loc, Float16Type::get(ctx), oldI16);
      oldF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, oldF16);
    } else {
      Value oldBF16 = LLVM::BitcastOp::create(rewriter, loc,
                                              BFloat16Type::get(ctx), oldI16);
      oldF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, oldBF16);
    }

    Value newF32 = arith::AddFOp::create(rewriter, loc, oldF32, valF32);

    Value newI16;
    if (elemTy.isF16()) {
      Value newF16 =
          arith::TruncFOp::create(rewriter, loc, Float16Type::get(ctx), newF32);
      newI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, newF16);
    } else {
      Value newBF16 = arith::TruncFOp::create(rewriter, loc,
                                              BFloat16Type::get(ctx), newF32);
      newI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, newBF16);
    }

    // Pack back into i32: (old & ~mask) | (new_i16_zext << shift)
    Value newI32Ext = LLVM::ZExtOp::create(rewriter, loc, i32Ty, newI16);
    Value newShifted = LLVM::ShlOp::create(rewriter, loc, newI32Ext, shift);
    Value cleared = LLVM::AndOp::create(rewriter, loc, phiOldI32, notMask);
    Value newI32 = LLVM::OrOp::create(rewriter, loc, cleared, newShifted);

    LLVM::StoreOp::create(rewriter, loc, phiOldI32, expectedAlloca);
    Value order0 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope2 = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value volT = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    Value oldRet =
        LLVM::CallOp::create(rewriter, loc, cmpxchgFn,
                             ValueRange{alignedPtr, expectedAlloca, newI32,
                                        order0, order0, scope2, volT})
            .getResult();

    Value success = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                         oldRet, phiOldI32);

    LLVM::CondBrOp::create(rewriter, loc, success, afterBlock, ValueRange{},
                           loopBlock, ValueRange{oldRet});

    rewriter.setInsertionPointToStart(afterBlock);

    // Return the pre-update element value: on success oldI16 (extracted in the
    // loop block) was the matched half. Carry it out via an afterBlock arg.
    afterBlock->addArgument(elemTy, loc);

    auto *term = loopBlock->getTerminator();
    rewriter.setInsertionPoint(term);

    Value oldElem;
    if (elemTy.isF16())
      oldElem =
          LLVM::BitcastOp::create(rewriter, loc, Float16Type::get(ctx), oldI16);
    else
      oldElem = LLVM::BitcastOp::create(rewriter, loc, BFloat16Type::get(ctx),
                                        oldI16);

    rewriter.eraseOp(term);
    LLVM::CondBrOp::create(rewriter, loc, success, afterBlock,
                           ValueRange{oldElem}, loopBlock, ValueRange{oldRet});

    rewriter.setInsertionPointToStart(afterBlock);
    return afterBlock->getArgument(0);
  }

  // Emit a single scalar atomic (direct AIR intrinsic or CAS loop). Returns the
  // old value.
  Value emitOneAtomic(ConversionPatternRewriter &rewriter, Location loc,
                      ModuleOp mod, Value ptr, Value val, Value mask,
                      Type valueElemTy, RMWOp rmwOp, const std::string &airName,
                      bool needsCAS) const {
    auto *ctx = rewriter.getContext();

    if (needsCAS) {
      auto emitCAS = [&]() -> Value {
        if (valueElemTy.isF32())
          return emitF32CASLoop(rewriter, loc, mod, ptr, val, rmwOp);
        else
          return emitF16BF16CASLoop(rewriter, loc, mod, ptr, val, valueElemTy,
                                    rmwOp);
      };

      if (mask) {
        // Wrap CAS in conditional to skip non-canonical threads
        auto *currentBlock = rewriter.getInsertionBlock();
        auto *afterBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        auto *casBlock = rewriter.createBlock(afterBlock);

        afterBlock->addArgument(valueElemTy, loc);

        rewriter.setInsertionPointToEnd(currentBlock);
        Value zeroVal = LLVM::ConstantOp::create(
            rewriter, loc, valueElemTy, rewriter.getZeroAttr(valueElemTy));
        LLVM::CondBrOp::create(rewriter, loc, mask, casBlock, ValueRange{},
                               afterBlock, ValueRange{zeroVal});

        rewriter.setInsertionPointToStart(casBlock);
        Value casResult = emitCAS();
        LLVM::BrOp::create(rewriter, loc, ValueRange{casResult}, afterBlock);

        rewriter.setInsertionPointToStart(afterBlock);
        return afterBlock->getArgument(0);
      }
      return emitCAS();
    }

    // Direct AIR intrinsic path
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto fnTy = LLVMFunctionType::get(
        valueElemTy, {ptrTy, valueElemTy, i32Ty, i32Ty, i1Ty}, false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (!mod.lookupSymbol<LLVMFuncOp>(airName))
        LLVMFuncOp::create(rewriter, mod.getLoc(), airName, fnTy,
                           Linkage::External);
    }
    auto atomicFn = mod.lookupSymbol<LLVMFuncOp>(airName);

    Value order = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value vol = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    if (mask) {
      auto *currentBlock = rewriter.getInsertionBlock();
      auto *afterBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      auto *atomicBlock = rewriter.createBlock(afterBlock);

      afterBlock->addArgument(valueElemTy, loc);

      rewriter.setInsertionPointToEnd(currentBlock);
      Value zeroVal = LLVM::ConstantOp::create(
          rewriter, loc, valueElemTy, rewriter.getZeroAttr(valueElemTy));
      LLVM::CondBrOp::create(rewriter, loc, mask, atomicBlock, ValueRange{},
                             afterBlock, ValueRange{zeroVal});

      rewriter.setInsertionPointToStart(atomicBlock);
      Value atomicResult =
          LLVM::CallOp::create(rewriter, loc, atomicFn,
                               ValueRange{ptr, val, order, scope, vol})
              .getResult();
      LLVM::BrOp::create(rewriter, loc, ValueRange{atomicResult}, afterBlock);

      rewriter.setInsertionPointToStart(afterBlock);
      return afterBlock->getArgument(0);
    } else {
      return LLVM::CallOp::create(rewriter, loc, atomicFn,
                                  ValueRange{ptr, val, order, scope, vol})
          .getResult();
    }
  }

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto rmwOp = op.getAtomicRmwOp();
    auto tensorTy = dyn_cast<RankedTensorType>(op.getType());

    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : getTypeConverter()->convertType(op.getType());

    // Determine AIR intrinsic name or CAS loop
    std::string airName;
    bool needsCAS = false;
    if (valueElemTy.isF32()) {
      switch (rmwOp) {
      case RMWOp::FADD:
        airName = "air.atomic.global.add.f32";
        break;
      case RMWOp::XCHG:
        airName = "air.atomic.global.xchg.f32";
        break;
      default:
        return failure();
      }
    } else if (valueElemTy.isF16() || valueElemTy.isBF16()) {
      switch (rmwOp) {
      case RMWOp::FADD:
        needsCAS = true;
        break;
      default:
        return failure();
      }
    } else if (valueElemTy.isInteger(32)) {
      switch (rmwOp) {
      case RMWOp::ADD:
        airName = "air.atomic.global.add.s.i32";
        break;
      case RMWOp::MAX:
        airName = "air.atomic.global.max.s.i32";
        break;
      case RMWOp::MIN:
        airName = "air.atomic.global.min.s.i32";
        break;
      case RMWOp::UMAX:
        airName = "air.atomic.global.max.u.i32";
        break;
      case RMWOp::UMIN:
        airName = "air.atomic.global.min.u.i32";
        break;
      case RMWOp::AND:
        airName = "air.atomic.global.and.s.i32";
        break;
      case RMWOp::OR:
        airName = "air.atomic.global.or.s.i32";
        break;
      case RMWOp::XOR:
        airName = "air.atomic.global.xor.s.i32";
        break;
      case RMWOp::XCHG:
        airName = "air.atomic.global.xchg.i32";
        break;
      default:
        return failure();
      }
    } else {
      return rewriter.notifyMatchFailure(
          op, "Apple GPU has only 32-bit atomics (i64/f64 atomic_rmw "
              "unsupported)");
    }

    if (!tensorTy) {
      // Scalar atomic: only thread 0 executes, broadcast result via TG.
      // Without this, all threads execute the atomic independently,
      // which corrupts spin-lock patterns (e.g. all threads doing
      // xchg(Lock, 0) allows another group to acquire between threads).
      auto i32Ty = IntegerType::get(ctx, 32);

      auto arrI32x3Ty = LLVMArrayType::get(i32Ty, 3);
      auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
          LLVMFuncOp::create(rewriter, mod.getLoc(),
                             "air.thread_position_in_threadgroup", tidFnTy,
                             Linkage::External);
      }
      auto tidFn =
          mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");

      Value tidStruct =
          LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
      Value tid0 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                                ArrayRef<int64_t>{0});
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      Value isThread0 = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, tid0, zero);

      // Combine thread-0 predicate with the op's own mask (e.g. sign-based
      // masks from float atomic_max/min decomposition).
      Value combinedMask = isThread0;
      if (llMask) {
        combinedMask = LLVM::AndOp::create(rewriter, loc, isThread0, llMask);
      }

      // Thread 0 (with mask) executes the atomic; others get a default value
      Value result =
          emitOneAtomic(rewriter, loc, mod, llPtr, llVal, combinedMask,
                        valueElemTy, rmwOp, airName, needsCAS);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Tensor atomic: unpack → per-element atomic → pack
    // Compute redundant thread predicate to mask out threads that
    // don't own unique elements (e.g. 128 threads, 4 elements → 124 idle)
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitAppleRedundantThreadPredicate(freeVarMasks, rewriter, loc, mod);
    uint32_t regMask = freeVarMasks[StringAttr::get(ctx, "reg")];

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    SmallVector<Value> resultVals(ptrElements.size());
    for (size_t i = 0; i < ptrElements.size(); ++i) {
      // Redundant register elements reuse the canonical result.
      if (!isCanonicalIndex(i, regMask)) {
        resultVals[i] = resultVals[i & ~regMask];
        continue;
      }

      Value mask = llMask ? maybeAnd(rewriter, loc, threadPred, maskElements[i])
                          : threadPred;
      Value res =
          emitOneAtomic(rewriter, loc, mod, ptrElements[i], valElements[i],
                        mask, valueElemTy, rmwOp, airName, needsCAS);
      resultVals[i] = res;
    }

    Value packed = packLLElements(loc, getTypeConverter(), resultVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, packed);
    return success();
  }
};

// Lower triton::AtomicCASOp → air.atomic.global.cmpxchg.weak.{i32,i64}
//
// Metal CAS signature:
//   i32 @air.atomic.global.cmpxchg.weak.i32(
//       ptr addrspace(1) ptr, ptr addrspace(0) expected,
//       i32 desired, i32 succ_order, i32 fail_order, i32 scope, i1 volatile)
// expected is passed by pointer and updated on failure.
// Returns old value.
struct AtomicCASOpAppleConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Emit f16/bf16 CAS via i32 CAS on aligned word.
  // Strategy: align ptr to i32 boundary, read i32 word, replace the target
  // half with cmp/val, do i32 CAS. No loop needed — CAS semantics guarantee
  // atomicity. If the CAS fails (other half changed), return the old f16 value.
  Value emitF16BF16CAS(ConversionPatternRewriter &rewriter, Location loc,
                       ModuleOp mod, Value ptr, Value cmp, Value val,
                       Type elemTy) const {
    auto *ctx = rewriter.getContext();
    auto i16Ty = IntegerType::get(ctx, 16);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0);

    auto airName = StringRef("air.atomic.global.cmpxchg.weak.i32");
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (!mod.lookupSymbol<LLVMFuncOp>(airName)) {
        auto fnTy = LLVMFunctionType::get(
            i32Ty, {ptrTy, ptrTy0, i32Ty, i32Ty, i32Ty, i32Ty, i1Ty}, false);
        LLVMFuncOp::create(rewriter, mod.getLoc(), airName, fnTy,
                           Linkage::External);
      }
    }
    auto casFn = mod.lookupSymbol<LLVMFuncOp>(airName);

    // Align the f16/bf16 pointer down to its i32 word: aligned = ptr & ~3.
    Value ptrInt = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptr);
    Value one64 = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    Value offset = LLVM::AndOp::create(rewriter, loc, ptrInt, one64);
    Value three64 = arith::ConstantIntOp::create(rewriter, loc, 3, 64);
    Value notThree64 = arith::ConstantIntOp::create(rewriter, loc, ~3LL, 64);
    Value alignedInt = LLVM::AndOp::create(rewriter, loc, ptrInt, notThree64);
    Value alignedPtr =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, alignedInt);

    // Bit shift of the target half within the i32 word: (ptr & 3) * 8.
    Value byteOffset = LLVM::AndOp::create(rewriter, loc, ptrInt, three64);
    Value byteOffset32 =
        LLVM::TruncOp::create(rewriter, loc, i32Ty, byteOffset);
    Value shift =
        LLVM::ShlOp::create(rewriter, loc, byteOffset32,
                            arith::ConstantIntOp::create(rewriter, loc, 3, 32));

    Value mask16 = arith::ConstantIntOp::create(rewriter, loc, 0xFFFF, 32);
    Value mask = LLVM::ShlOp::create(rewriter, loc, mask16, shift);
    Value notMask = LLVM::XOrOp::create(
        rewriter, loc, mask,
        arith::ConstantIntOp::create(rewriter, loc, -1, 32));

    // Truncate f32 cmp/val to f16/bf16 if needed (Triton CAS passes f32 cmp/val
    // for f16 ptrs)
    Value cmpElem = cmp, valElem = val;
    if (cmp.getType().isF32()) {
      cmpElem = arith::TruncFOp::create(rewriter, loc, elemTy, cmp);
      valElem = arith::TruncFOp::create(rewriter, loc, elemTy, val);
    }
    Value cmpI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, cmpElem);
    Value valI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, valElem);
    Value cmpI32 = LLVM::ZExtOp::create(rewriter, loc, i32Ty, cmpI16);
    Value valI32 = LLVM::ZExtOp::create(rewriter, loc, i32Ty, valI16);

    Value cmpShifted = LLVM::ShlOp::create(rewriter, loc, cmpI32, shift);
    Value valShifted = LLVM::ShlOp::create(rewriter, loc, valI32, shift);

    // Non-atomic read of the current word for the initial guess.
    Value curI32 = LLVM::LoadOp::create(rewriter, loc, i32Ty, alignedPtr);

    // expected/desired splice cmp/val into the target half: keep other bits,
    // replace the masked half. expected = (cur & ~mask) | cmpShifted.
    Value otherBits = LLVM::AndOp::create(rewriter, loc, curI32, notMask);
    Value expectedI32 =
        LLVM::OrOp::create(rewriter, loc, otherBits, cmpShifted);
    Value desiredI32 = LLVM::OrOp::create(rewriter, loc, otherBits, valShifted);

    // Alloca for expected — must be in entry block.
    Value expectedAlloca;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto *funcOp = rewriter.getInsertionBlock()->getParent()->getParentOp();
      auto &entryBlock = cast<LLVM::LLVMFuncOp>(funcOp).getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      Value oneI64 = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, i32Ty,
                                              oneI64, /*alignment=*/4);
    }

    LLVM::StoreOp::create(rewriter, loc, expectedI32, expectedAlloca);

    Value order = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value vol = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    Value oldI32 =
        LLVM::CallOp::create(rewriter, loc, casFn,
                             ValueRange{alignedPtr, expectedAlloca, desiredI32,
                                        order, order, scope, vol})
            .getResult();

    // Extract the old f16/bf16 half from the returned i32 word.
    Value oldShifted = LLVM::LShrOp::create(rewriter, loc, oldI32, shift);
    Value oldI16 = LLVM::TruncOp::create(rewriter, loc, i16Ty, oldShifted);
    return LLVM::BitcastOp::create(rewriter, loc, elemTy, oldI16);
  }

  // Emit a single scalar CAS operation. Returns the old value.
  Value emitOneCAS(ConversionPatternRewriter &rewriter, Location loc,
                   ModuleOp mod, Value ptr, Value cmp, Value val,
                   Type valueTy) const {
    auto *ctx = rewriter.getContext();

    if (valueTy.isF16() || valueTy.isBF16())
      return emitF16BF16CAS(rewriter, loc, mod, ptr, cmp, val, valueTy);

    // Only 32-bit CAS reaches here — i64/f64 are rejected at the pattern entry
    // (no 64-bit atomics on Apple GPU). f16/bf16 take the emitF16BF16CAS path
    // above.
    std::string airName;
    Type casTy;
    if (valueTy.isInteger(32) || valueTy.isF32()) {
      airName = "air.atomic.global.cmpxchg.weak.i32";
      casTy = IntegerType::get(ctx, 32);
    } else {
      llvm_unreachable("unsupported CAS type");
    }

    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (!mod.lookupSymbol<LLVMFuncOp>(airName)) {
        auto fnTy = LLVMFunctionType::get(
            casTy, {ptrTy, ptrTy0, casTy, i32Ty, i32Ty, i32Ty, i1Ty}, false);
        LLVMFuncOp::create(rewriter, mod.getLoc(), airName, fnTy,
                           Linkage::External);
      }
    }
    auto casFn = mod.lookupSymbol<LLVMFuncOp>(airName);

    Value cmpI = cmp, valI = val;
    bool needBitcast = (valueTy != casTy);
    if (needBitcast) {
      cmpI = LLVM::BitcastOp::create(rewriter, loc, casTy, cmp);
      valI = LLVM::BitcastOp::create(rewriter, loc, casTy, val);
    }

    // Alloca for expected — must be in entry block.
    Value one, expectedAlloca;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto *funcOp = rewriter.getInsertionBlock()->getParent()->getParentOp();
      auto &entryBlock = cast<LLVM::LLVMFuncOp>(funcOp).getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, casTy, one,
                                              /*alignment=*/4);
    }

    LLVM::StoreOp::create(rewriter, loc, cmpI, expectedAlloca);

    Value order = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value vol = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    Value oldI = LLVM::CallOp::create(rewriter, loc, casFn,
                                      ValueRange{ptr, expectedAlloca, valI,
                                                 order, order, scope, vol})
                     .getResult();

    if (needBitcast)
      return LLVM::BitcastOp::create(rewriter, loc, valueTy, oldI);
    return oldI;
  }

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    auto tensorTy = dyn_cast<RankedTensorType>(op.getType());
    Type valueTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : getTypeConverter()->convertType(op.getType());

    // Check supported types. Apple GPUs have NO 64-bit atomics — there is no
    // `air.atomic.global.cmpxchg.weak.i64` intrinsic, and emitting one crashes
    // the Metal compiler service (XPC_ERROR_CONNECTION_INTERRUPTED). Fail the
    // lowering cleanly (the op stays illegal -> compile error) rather than
    // producing AIR that brings down the GPU compiler.
    if (!(valueTy.isInteger(32) || valueTy.isF32() || valueTy.isF16() ||
          valueTy.isBF16()))
      return rewriter.notifyMatchFailure(
          op, "Apple GPU has no 64-bit atomics (i64/f64 CAS unsupported)");

    if (!tensorTy) {
      // Scalar CAS: only thread 0 executes, broadcast result to all threads.
      // Without this, a spin-lock pattern (while CAS == 1) deadlocks:
      // all threads spin independently but only one succeeds, and subsequent
      // barriers can never be reached by the blocked threads.
      auto *ctx = rewriter.getContext();
      auto i32Ty = IntegerType::get(ctx, 32);

      auto arrI32x3Ty = LLVMArrayType::get(i32Ty, 3);
      auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
          LLVMFuncOp::create(rewriter, mod.getLoc(),
                             "air.thread_position_in_threadgroup", tidFnTy,
                             Linkage::External);
      }
      auto tidFn =
          mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");

      auto voidTy = LLVMVoidType::get(ctx);
      auto barrFnTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        if (!mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier"))
          LLVMFuncOp::create(rewriter, mod.getLoc(), "air.wg.barrier", barrFnTy,
                             Linkage::External);
      }
      auto barrFn = mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier");

      // Create TG global to broadcast the CAS result
      Type tgElemTy = valueTy.isF32()   ? (Type)i32Ty
                      : valueTy.isF64() ? (Type)IntegerType::get(ctx, 64)
                                        : valueTy;
      std::string tgName = "__tg_cas_bcast";
      auto tgPtrTy = LLVMPointerType::get(ctx, 3);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        if (!mod.lookupSymbol<LLVM::GlobalOp>(tgName)) {
          auto arrTy = LLVMArrayType::get(tgElemTy, 1);
          LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy, false,
                                 Linkage::Internal, tgName, Attribute(), 4, 3u);
        }
      }
      auto tgGlobal = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
      Value tgPtr =
          LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgGlobal.getName());

      Value tidStruct =
          LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
      Value tid0 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                                ArrayRef<int64_t>{0});
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      Value isThread0 = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, tid0, zero);

      // Thread 0 does the CAS and stores to TG; others skip straight to the
      // barrier in mergeBlock.
      auto *currentBlock = rewriter.getInsertionBlock();
      auto *afterBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      auto *casBlock = rewriter.createBlock(afterBlock);
      auto *mergeBlock = rewriter.createBlock(afterBlock);

      rewriter.setInsertionPointToEnd(currentBlock);
      LLVM::CondBrOp::create(rewriter, loc, isThread0, casBlock, mergeBlock);

      rewriter.setInsertionPointToStart(casBlock);
      Value casResult =
          emitOneCAS(rewriter, loc, mod, llPtr, llCmp, llVal, valueTy);
      Value resultToStore = casResult;
      if (valueTy.isF32())
        resultToStore =
            LLVM::BitcastOp::create(rewriter, loc, i32Ty, casResult);
      else if (valueTy.isF64())
        resultToStore = LLVM::BitcastOp::create(
            rewriter, loc, IntegerType::get(ctx, 64), casResult);
      LLVM::StoreOp::create(rewriter, loc, resultToStore, tgPtr);
      LLVM::BrOp::create(rewriter, loc, mergeBlock);

      rewriter.setInsertionPointToStart(mergeBlock);
      // device memory fence (flag=1) to ensure the CAS side-effects are visible
      Value flagDev = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
      Value scope1 = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
      LLVM::CallOp::create(rewriter, loc, barrFn, ValueRange{flagDev, scope1});
      // TG memory fence (flag=2) to ensure the TG store is visible
      Value flagTG = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
      LLVM::CallOp::create(rewriter, loc, barrFn, ValueRange{flagTG, scope1});
      Value loaded =
          LLVM::LoadOp::create(rewriter, loc, tgElemTy, tgPtr).getResult();
      if (valueTy.isF32())
        loaded = LLVM::BitcastOp::create(rewriter, loc, valueTy, loaded);
      else if (valueTy.isF64())
        loaded = LLVM::BitcastOp::create(rewriter, loc, valueTy, loaded);

      rewriter.setInsertionPointAfter(loaded.getDefiningOp());
      // Splice afterBlock's remaining ops into mergeBlock so the loaded result
      // dominates them.
      mergeBlock->getOperations().splice(mergeBlock->end(),
                                         afterBlock->getOperations());
      afterBlock->erase();

      rewriter.replaceOp(op, loaded);
      return success();
    }

    // Tensor CAS: unpack → per-element CAS → pack, masking out threads that
    // don't own unique elements via the redundant thread predicate.
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitAppleRedundantThreadPredicate(freeVarMasks, rewriter, loc, mod);
    uint32_t regMask =
        freeVarMasks[StringAttr::get(rewriter.getContext(), "reg")];

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    SmallVector<Value> resultVals(ptrElements.size());
    for (size_t i = 0; i < ptrElements.size(); ++i) {
      // Redundant register elements reuse the canonical result.
      if (!isCanonicalIndex(i, regMask)) {
        resultVals[i] = resultVals[i & ~regMask];
        continue;
      }

      // Predicate the CAS so only canonical threads execute it.
      if (threadPred) {
        auto *currentBlock = rewriter.getInsertionBlock();
        auto *afterBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        auto *casBlock = rewriter.createBlock(afterBlock);
        afterBlock->addArgument(valueTy, loc);

        rewriter.setInsertionPointToEnd(currentBlock);
        Value zeroVal = LLVM::ConstantOp::create(rewriter, loc, valueTy,
                                                 rewriter.getZeroAttr(valueTy));
        LLVM::CondBrOp::create(rewriter, loc, threadPred, casBlock,
                               ValueRange{}, afterBlock, ValueRange{zeroVal});

        rewriter.setInsertionPointToStart(casBlock);
        Value casResult = emitOneCAS(rewriter, loc, mod, ptrElements[i],
                                     cmpElements[i], valElements[i], valueTy);
        LLVM::BrOp::create(rewriter, loc, ValueRange{casResult}, afterBlock);

        rewriter.setInsertionPointToStart(afterBlock);
        resultVals[i] = afterBlock->getArgument(0);
      } else {
        resultVals[i] = emitOneCAS(rewriter, loc, mod, ptrElements[i],
                                   cmpElements[i], valElements[i], valueTy);
      }
    }

    Value packed = packLLElements(loc, getTypeConverter(), resultVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, packed);
    return success();
  }
};

} // anonymous namespace

void populateAtomicOpPatterns(LLVMTypeConverter &typeConverter,
                              RewritePatternSet &patterns) {
  patterns.add<AtomicRMWOpAppleConversion>(typeConverter,
                                           patternBenefitDefault + 10);
  patterns.add<AtomicCASOpAppleConversion>(typeConverter,
                                           patternBenefitDefault + 10);
}

} // namespace mlir::triton::applegpu
