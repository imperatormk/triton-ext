// Block terminators the walk carries no statement for.
//
// By name, because MLIR's Terminator trait compares a TypeID the plugin and
// the host generate separately, so it answers false wherever the two do not
// share one.
#ifndef AGPU_TERMINATORS_H
#define AGPU_TERMINATORS_H

#include <cstddef>
#include <string_view>

namespace agpu {

inline constexpr std::string_view kTerminators[] = {
    "tt.return",
};

inline bool isTerminator(std::string_view op) {
  for (const std::string_view t : kTerminators)
    if (t == op)
      return true;
  return false;
}

} // namespace agpu

#endif // AGPU_TERMINATORS_H
