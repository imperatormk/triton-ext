// CodeSize.h - what the emitter spends code on: how large a function may grow
// before Metal's compiler fails, when K steps roll into a loop, how far a
// rolled loop unrolls again, and when a masked load earns a peeled fast path.
// Every code-size knob lives here.
#ifndef AGPU_COST_CODE_SIZE_H
#define AGPU_COST_CODE_SIZE_H

#include <cstdint>

namespace agpu::cost {

// Metal's compiler dies with std::bad_alloc under PromoteMemToReg/SROA on an
// oversized function. Not derived from anything; a guess biased toward rolling
// too early.
inline constexpr int64_t kDeclBudget = 10000;

// Below this many fragment declarations, rolling K frees too little of the
// budget to be worth a re-walk. Unmeasured.
inline constexpr int64_t kRollFragFloor = 1024;

// Past this many MMAs a function rolls its K steps whatever its size. On flex
// attention's backward (M1 Pro, 2026-09-24) rolling keeps 94 registers, two
// resident 256-thread threadgroups, where unrolled needs 120 and keeps one:
// 2.43 against 2.84-2.94 ms. The residency is what it protects; the count is
// how it is spelled until the roll is judged in Occupancy.h.
inline constexpr int64_t kRollMmaFloor = 128;

// A rolled panel K loop unrolls until a trip holds about this many MMAs.
// Unmeasured.
inline constexpr int64_t kUnrollCount = 8;

// The fewest registers a masked load must fill to earn a peeled unconditional
// fast path beside its guarded one. Unmeasured.
inline constexpr int64_t kMaskFastPathMinRegs = 4;

} // namespace agpu::cost

#endif // AGPU_COST_CODE_SIZE_H
