#!/bin/bash
# Fragment-ABI hardware-layout oracle. Self-contained: builds the metallib +
# host harness and runs the bit-exact per-lane check on real Apple GPU hardware.
set -euo pipefail
cd "$(dirname "$0")"

xcrun -sdk macosx metal -O2 -std=metal3.0 -c oracle.metal -o oracle.air
xcrun -sdk macosx metallib oracle.air -o oracle.metallib
clang -fobjc-arc -framework Metal -framework Foundation harness.m -o harness

./harness "$PWD/oracle.metallib"
