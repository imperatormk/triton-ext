#include "agpu/core/Units.h"
#include "harness.h"

using namespace agpu;

int main() {
  CASE("the register file reproduces every measured pipeline width");
  {
    // Kernels compiled on M1 Pro whose width Metal chose (none pinned): the
    // highest register their disassembly touches, plus one, and
    // maxTotalThreadsPerThreadgroup.
    const struct {
      int64_t regs, maxThreads;
    } measured[] = {
        {66, 768},  {81, 576},  {82, 576},  {83, 576},  {109, 448}, {113, 448},
        {114, 448}, {120, 384}, {121, 384}, {125, 384}, {128, 384},
    };
    for (const auto &m : measured)
      CHECK_EQ(threadsForRegs(m.regs), m.maxThreads);
  }

  CASE("a full-register kernel is the widest launch always admitted");
  {
    CHECK_EQ(kAlwaysAdmittedThreads, 384);
    CHECK_EQ(threadsForRegs(kMaxRegsPerThread), kAlwaysAdmittedThreads);
  }

  CASE("registers cap residency below what threadgroup memory allows");
  {
    CHECK_EQ(certainResidency(4096, 64), 6);
    CHECK_EQ(certainResidency(4096, 128), 3);
    CHECK_EQ(certainResidency(4096, 256), 1);
    CHECK_EQ(certainResidency(4096, 512), 1);
  }

  CASE("threadgroup memory binds below the register cap");
  {
    CHECK_EQ(certainResidency(32768, 128), 1);
    CHECK_EQ(certainResidency(24576, 128), 2);
    CHECK_EQ(certainResidency(20480, 128), 3);
  }

  return ::agpu_test::report("Occupancy");
}
