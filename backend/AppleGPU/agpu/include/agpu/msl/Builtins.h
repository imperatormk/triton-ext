// Builtins.h - the names Metal spells for us, in one place. Strings only: a
// planner may name a builtin without being able to build a call.
#ifndef AGPU_MSL_BUILTINS_H
#define AGPU_MSL_BUILTINS_H

namespace agpu::msl::builtin {

namespace memflags {
inline constexpr const char *Threadgroup = "mem_flags::mem_threadgroup";
inline constexpr const char *DeviceAndThreadgroup =
    "mem_flags::mem_threadgroup | mem_flags::mem_device";
} // namespace memflags

namespace barrier {
inline constexpr const char *Threadgroup = "threadgroup_barrier";
inline constexpr const char *Simdgroup = "simdgroup_barrier";
} // namespace barrier

namespace comp {
inline constexpr const char *X = "x";
inline constexpr const char *Y = "y";
inline constexpr const char *Z = "z";

// Null for an axis Metal has no component for.
inline const char *of(int axis) {
  switch (axis) {
  case 0:
    return X;
  case 1:
    return Y;
  case 2:
    return Z;
  }
  return nullptr;
}
} // namespace comp

namespace simd {
inline constexpr const char *Shuffle = "simd_shuffle";
} // namespace simd

} // namespace agpu::msl::builtin

#endif // AGPU_MSL_BUILTINS_H
