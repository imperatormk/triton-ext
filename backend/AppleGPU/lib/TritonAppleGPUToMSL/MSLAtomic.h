// MSLAtomic.h - atomic RMW / CAS / poll + histogram AST builders.
//
// Builds the node trees for the atomic families the emitter's emitAtomic
// dispatches over. Pure over its arguments: no emitter state beyond ctx, the
// shared fresh-id counter, the thread-id name, and a pool-region callback for
// the threadgroup histogram bins.
#ifndef MSL_ATOMIC_H
#define MSL_ATOMIC_H

#include "MSLAst.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::triton::applegpu {

class AtomicEmitter {
public:
  // `tidId` / `poolBuf` are the emitter's live names, bound by reference: both
  // are assigned per-kernel after this is constructed.
  AtomicEmitter(msl::MSLContext &ctx, int &nextId, const std::string &tidId,
                const std::string &poolBuf)
      : ctx(ctx), nextId(nextId), tidId(tidId), poolBuf(poolBuf) {}

  msl::Expr *init0(llvm::StringRef sc);
  // as_type<to>(x) - the raw bit-reinterpret used to punt values through the
  // integer atomic word. Single site for every atomic reinterpret escape.
  msl::Expr *asType(msl::Type *to, msl::Expr *x) {
    return ctx.cast(msl::Cast::Style::AsType, to, x);
  }
  msl::Expr *deviceAtomicPtr(llvm::StringRef atomicTy, llvm::StringRef p);
  msl::Expr *casWeak(msl::Expr *ptr, llvm::StringRef expVar, msl::Expr *newVal);
  msl::Expr *packed16Extract(llvm::StringRef word, llvm::StringRef isHigh);
  msl::Expr *packed16Merge(llvm::StringRef word, llvm::StringRef isHigh,
                           msl::Expr *newBitsU32);
  msl::Block packed16Base(llvm::StringRef p, std::string &wordPtrOut,
                          std::string &isHighOut);
  msl::Expr *rmwCall(llvm::StringRef fn, llvm::StringRef atomicTy,
                     llvm::StringRef p, llvm::StringRef v,
                     llvm::StringRef order, bool memFlags);
  msl::Block int32CAS(llvm::StringRef p, llvm::StringRef c, llvm::StringRef v,
                      llvm::StringRef sc, llvm::StringRef id, bool declare);
  msl::Block float32CAS(llvm::StringRef p, llvm::StringRef c, llvm::StringRef v,
                        llvm::StringRef id, bool declare);
  msl::Block packed16CAS(llvm::StringRef wordPtr, llvm::StringRef isHigh,
                         llvm::StringRef c, llvm::StringRef v,
                         llvm::StringRef sc, llvm::StringRef id, bool declare);
  msl::Stmt *deviceFence();
  msl::Stmt *histBinsDecl(llvm::StringRef bins);
  msl::Stmt *histZeroInit(llvm::StringRef bins, llvm::StringRef zi,
                          int64_t nBins, int64_t threads);
  msl::Stmt *histFetchAdd(msl::Expr *guard, llvm::StringRef bins,
                          llvm::StringRef v);

private:
  // Mints names over the emitter's live id counter, so builder-minted names
  // advance the caller's id in lockstep.
  std::string fresh() { return "v" + std::to_string(nextId++); }

  msl::MSLContext &ctx;
  int &nextId;
  const std::string &tidId;
  const std::string &poolBuf;
};

} // namespace mlir::triton::applegpu

#endif // MSL_ATOMIC_H
