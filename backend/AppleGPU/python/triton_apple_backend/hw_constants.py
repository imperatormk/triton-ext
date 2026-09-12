"""Constants compiler.py and driver.py share, in a separate module because
neither can import the other.

kWarpSize in agpu/include/agpu/core/Units.h owns the warp size; WARP_SIZE here
must not drift from it. The other two have no C++ owner in this stage: Triton
requires the option fields and the device property regardless of what the
emitter uses.
"""

WARP_SIZE = 32
SG_FRAG_DIM = 8
TG_BUDGET_BYTES = 32768

# Torch calls the Apple GPU device "mps", so Triton's target must agree.
TARGET = "mps"


def target_arch(arch):
    """The `backend:arch` string Triton's target parser expects."""
    return f"{TARGET}:{arch}"
