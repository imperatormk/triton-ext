// Units.h - byte counts as a distinct type, so pool arithmetic cannot mix
// bytes of threadgroup memory with element counts of a tile.
#ifndef AGPU_UNITS_H
#define AGPU_UNITS_H

#include <algorithm>
#include <cstdint>

namespace agpu {

class Bytes {
public:
  Bytes() = default;
  explicit Bytes(int64_t n) : n_(n) {}
  int64_t count() const { return n_; }

  Bytes operator+(Bytes o) const { return Bytes(n_ + o.n_); }
  Bytes operator-(Bytes o) const { return Bytes(n_ - o.n_); }
  Bytes &operator+=(Bytes o) {
    n_ += o.n_;
    return *this;
  }
  bool operator==(Bytes o) const { return n_ == o.n_; }
  bool operator!=(Bytes o) const { return n_ != o.n_; }
  bool operator<(Bytes o) const { return n_ < o.n_; }
  bool operator<=(Bytes o) const { return n_ <= o.n_; }
  bool operator>(Bytes o) const { return n_ > o.n_; }
  bool operator>=(Bytes o) const { return n_ >= o.n_; }

private:
  int64_t n_ = 0;
};

inline Bytes minBytes(Bytes a, Bytes b) { return std::min(a, b); }
inline Bytes maxBytes(Bytes a, Bytes b) { return std::max(a, b); }

// ── the hardware budget ───────────────────────────────────────────────────

// Metal's maxThreadgroupMemoryLength: the most one threadgroup may declare.
inline constexpr int64_t kTGResidentBudgetBytes = 32768;

// What a core hands out across concurrently resident threadgroups. Measured
// on M1 Pro: a 20480-byte pool keeps three resident, 20736 bytes two.
inline constexpr int64_t kTGCoreBudgetBytes = 61440;

// Threadgroups that stay resident when a pool of `bytes` is declared. A step
// function: 15 KB gives four, 22 KB gives two.
inline constexpr int64_t tgResidency(int64_t bytes) {
  return bytes > 0 ? kTGCoreBudgetBytes / bytes : kTGCoreBudgetBytes;
}

// The register file, in 32-bit registers per core, and the most one thread may
// hold. A pipeline's maxTotalThreadsPerThreadgroup is the threads the file
// holds at the kernel's register count, allocated four at a time, in steps of
// 64 threads. `test_occupancy` holds the M1 Pro kernels this was fitted to.
inline constexpr int64_t kRegisterFileWords = 53248;
inline constexpr int64_t kMaxRegsPerThread = 128;
inline constexpr int64_t kRegisterGranule = 4;
inline constexpr int64_t kThreadStep = 64;

inline constexpr int64_t threadsForRegs(int64_t regs) {
  const int64_t alloc =
      (regs + kRegisterGranule - 1) / kRegisterGranule * kRegisterGranule;
  return kRegisterFileWords / alloc / kThreadStep * kThreadStep;
}

// Resident threadgroups past which threadgroup memory stops binding occupancy.
// Registers and warp slots bind first above this line.
inline constexpr int64_t kTGResidencyFloor = 6;

// Widest launch Metal admits regardless of register appetite: a kernel holding
// every register it may compiles to this, and a wider launch is then rejected
// at dispatch as OutOfResources.
inline constexpr int64_t kAlwaysAdmittedThreads =
    threadsForRegs(kMaxRegsPerThread);

// Threadgroups of `threadsPerTG` a core keeps resident with a pool of `bytes`
// however many registers the kernel turns out to hold. A launch wider than a
// full-register kernel admits is pinned, so the compiler fits one.
inline constexpr int64_t certainResidency(int64_t bytes, int64_t threadsPerTG) {
  return std::min(tgResidency(bytes),
                  std::max<int64_t>(1, kAlwaysAdmittedThreads / threadsPerTG));
}

// Alignment of a threadgroup pool's base address (Metal's widest vector
// access). `kPadBytes` is a whole multiple of it.
inline constexpr int64_t kTGPoolAlignBytes = 16;

inline constexpr int64_t kWarpSize = 32;

inline constexpr int64_t threadsFor(int64_t numWarps) {
  return numWarps * kWarpSize;
}

// Side of the simdgroup MMA fragment. Metal offers exactly one shape.
inline constexpr int64_t kSgFragDim = 8;

// Rounds up: an extent of 60 needs eight fragments. There is no smaller MMA,
// and the surplus is discarded on the way out.
inline constexpr int64_t fragsFor(int64_t extent) {
  return (extent + kSgFragDim - 1) / kSgFragDim;
}

inline constexpr int64_t fragAlignedExtent(int64_t extent) {
  return fragsFor(extent) * kSgFragDim;
}

// Accumulators are fp32 regardless of operand type.
inline constexpr int64_t kAccBytes = 4;

// `planTileActions` takes this as its bit-width argument.
inline constexpr int64_t kAccBits = kAccBytes * 8;

} // namespace agpu

#endif // AGPU_UNITS_H
