// MSLConstants.h - named MSL builtin identifiers.
//
// The exhaustive set of Metal Shading Language builtin names the emitter spells
// (math, simd, simdgroup-matrix, atomics, memory ordering). Call sites use
// these constants instead of bare string literals so a typo is a compile error
// and the full builtin surface is greppable in one place. Flat named constants
// by design
// - no registry framework.
#ifndef MSL_CONSTANTS_H
#define MSL_CONSTANTS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::applegpu {

// Threadgroup-resident scratch budget (32KB Metal cap). The single source of
// truth for every pool-sizing and lowering gate; sizing and emission read this
// through poolBudget(), never a bare literal, so they cannot drift.
inline constexpr int64_t kTGResidentBudgetBytes = 32768;

} // namespace mlir::triton::applegpu

namespace mlir::triton::applegpu::msl {

namespace builtin {

// metal:: math (exact ops keep the plain namespace).
namespace math {
inline constexpr llvm::StringRef Abs = "metal::abs";
inline constexpr llvm::StringRef Ceil = "metal::ceil";
inline constexpr llvm::StringRef Clamp = "metal::clamp";
inline constexpr llvm::StringRef Copysign = "metal::copysign";
inline constexpr llvm::StringRef Exp = "metal::exp";
inline constexpr llvm::StringRef Fabs = "metal::fabs";
inline constexpr llvm::StringRef Floor = "metal::floor";
inline constexpr llvm::StringRef Fma = "metal::fma";
inline constexpr llvm::StringRef Fmod = "metal::fmod";
inline constexpr llvm::StringRef Isnan = "metal::isnan";
inline constexpr llvm::StringRef Rint = "metal::rint";
inline constexpr llvm::StringRef Round = "metal::round";
inline constexpr llvm::StringRef Trunc = "metal::trunc";
} // namespace math

// metal::precise:: transcendentals (accuracy-sensitive; safe-math never
// controls transcendental accuracy, only the namespace does).
namespace precise {
inline constexpr llvm::StringRef Acos = "metal::precise::acos";
inline constexpr llvm::StringRef Asin = "metal::precise::asin";
inline constexpr llvm::StringRef Atan = "metal::precise::atan";
inline constexpr llvm::StringRef Atan2 = "metal::precise::atan2";
inline constexpr llvm::StringRef Cbrt = "metal::precise::cbrt";
inline constexpr llvm::StringRef Cos = "metal::precise::cos";
inline constexpr llvm::StringRef Cosh = "metal::precise::cosh";
inline constexpr llvm::StringRef Exp = "metal::precise::exp";
inline constexpr llvm::StringRef Exp2 = "metal::precise::exp2";
inline constexpr llvm::StringRef Log = "metal::precise::log";
inline constexpr llvm::StringRef Log2 = "metal::precise::log2";
inline constexpr llvm::StringRef Log10 = "metal::precise::log10";
inline constexpr llvm::StringRef Pow = "metal::precise::pow";
inline constexpr llvm::StringRef Rsqrt = "metal::precise::rsqrt";
inline constexpr llvm::StringRef Sin = "metal::precise::sin";
inline constexpr llvm::StringRef Sinh = "metal::precise::sinh";
inline constexpr llvm::StringRef Sqrt = "metal::precise::sqrt";
inline constexpr llvm::StringRef Tan = "metal::precise::tan";
inline constexpr llvm::StringRef Tanh = "metal::precise::tanh";
} // namespace precise

// Cross-lane simd shuffles.
namespace simd {
inline constexpr llvm::StringRef Shuffle = "simd_shuffle";
inline constexpr llvm::StringRef ShuffleUp = "simd_shuffle_up";
inline constexpr llvm::StringRef ShuffleDown = "simd_shuffle_down";
inline constexpr llvm::StringRef ShuffleXor = "simd_shuffle_xor";
} // namespace simd

// simdgroup_matrix element types and MMA intrinsics.
namespace sg {
inline constexpr llvm::StringRef Half8x8 = "simdgroup_half8x8";
inline constexpr llvm::StringRef Bfloat8x8 = "simdgroup_bfloat8x8";
inline constexpr llvm::StringRef Float8x8 = "simdgroup_float8x8";
inline constexpr llvm::StringRef Load = "simdgroup_load";
inline constexpr llvm::StringRef Store = "simdgroup_store";
inline constexpr llvm::StringRef MultiplyAccumulate =
    "simdgroup_multiply_accumulate";
} // namespace sg

// atomic_* element types and explicit-ordering operations.
namespace atomic {
inline constexpr llvm::StringRef Int = "atomic_int";
inline constexpr llvm::StringRef Uint = "atomic_uint";
inline constexpr llvm::StringRef Long = "atomic_long";
inline constexpr llvm::StringRef Float = "atomic_float";
inline constexpr llvm::StringRef Load = "atomic_load_explicit";
inline constexpr llvm::StringRef Store = "atomic_store_explicit";
inline constexpr llvm::StringRef Exchange = "atomic_exchange_explicit";
inline constexpr llvm::StringRef ThreadFence = "atomic_thread_fence";
inline constexpr llvm::StringRef FetchAdd = "atomic_fetch_add_explicit";
inline constexpr llvm::StringRef FetchAnd = "atomic_fetch_and_explicit";
inline constexpr llvm::StringRef FetchMax = "atomic_fetch_max_explicit";
inline constexpr llvm::StringRef FetchMin = "atomic_fetch_min_explicit";
inline constexpr llvm::StringRef FetchOr = "atomic_fetch_or_explicit";
inline constexpr llvm::StringRef FetchXor = "atomic_fetch_xor_explicit";
inline constexpr llvm::StringRef CompareExchangeWeak =
    "atomic_compare_exchange_weak_explicit";
} // namespace atomic

// mem_flags::* memory scope bits.
namespace memflags {
inline constexpr llvm::StringRef Device = "mem_flags::mem_device";
inline constexpr llvm::StringRef Threadgroup = "mem_flags::mem_threadgroup";
} // namespace memflags

// memory_order_* orderings.
namespace order {
inline constexpr llvm::StringRef Relaxed = "memory_order_relaxed";
inline constexpr llvm::StringRef SeqCst = "memory_order_seq_cst";
} // namespace order

} // namespace builtin

} // namespace mlir::triton::applegpu::msl

#endif // MSL_CONSTANTS_H
