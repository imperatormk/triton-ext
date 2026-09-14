// planKernelAbi decides where every kernel argument lives. driver.py packs the
// scalar buffer to match, so these placements are an ABI both sides rely on.
#include "agpu/emit/KernelAbi.h"
#include "harness.h"

using namespace agpu;

namespace {

KernelArg ptr(msl::Str name) {
  KernelArg a;
  a.name = std::move(name);
  a.isPointer = true;
  return a;
}

KernelArg scalar(msl::Str name, ElemType e) {
  KernelArg a;
  a.name = std::move(name);
  a.elem = e;
  return a;
}

} // namespace

int main() {
  CASE("pointers take buffer bindings in argument order");
  {
    const KernelAbi abi = planKernelAbi({ptr("a"), ptr("b"), ptr("c")}, 4);
    CHECK(abi.bufferCount == 3);
    CHECK(!abi.hasArgBuffer);
    for (int i = 0; i < 3; ++i) {
      CHECK(abi.placements[i].slot == ArgSlot::Buffer);
      CHECK(abi.placements[i].index == i);
    }
  }

  CASE("scalars pack into one buffer after the pointers");
  {
    const KernelAbi abi =
        planKernelAbi({ptr("p"), scalar("n", i32()), scalar("m", i32())}, 4);
    CHECK(abi.hasArgBuffer);
    CHECK(abi.argBufferIndex == 1);
    CHECK(abi.bufferCount == 2);
    CHECK(abi.placements[1].slot == ArgSlot::ArgBuffer);
    CHECK(abi.placements[1].offset == 0);
    CHECK(abi.placements[2].offset == 4);
    CHECK(abi.argBufferBytes == 8);
  }

  CASE("a scalar sits at a multiple of its own size");
  {
    // i8 at 0, i16 rounds 1 up to 2, i32 rounds 4 up to 4, i64 rounds 8 to 8.
    const KernelAbi abi =
        planKernelAbi({scalar("a", ElemType{ElemType::Kind::Int, 8, false}),
                       scalar("b", ElemType{ElemType::Kind::Int, 16, false}),
                       scalar("c", i32()),
                       scalar("d", ElemType{ElemType::Kind::Int, 64, false})},
                      4);
    CHECK(abi.placements[0].offset == 0);
    CHECK(abi.placements[1].offset == 2);
    CHECK(abi.placements[2].offset == 4);
    CHECK(abi.placements[3].offset == 8);
    CHECK(abi.argBufferBytes == 16);
  }

  CASE("a sub-byte scalar still occupies a byte");
  {
    const KernelAbi abi =
        planKernelAbi({scalar("f", i1()), scalar("g", i1())}, 4);
    CHECK(abi.placements[0].offset == 0);
    CHECK(abi.placements[1].offset == 1);
    CHECK(abi.argBufferBytes == 2);
  }

  CASE("more buffers than Metal binds is not usable");
  {
    std::vector<KernelArg> args;
    for (int i = 0; i < kMaxBuffers; ++i)
      args.push_back(ptr("p"));
    CHECK(planKernelAbi(args, 4).usable());

    args.push_back(ptr("one-too-many"));
    CHECK(!planKernelAbi(args, 4).usable());
  }

  return ::agpu_test::report("KernelAbi");
}
