"""Hardware constants, mirroring kWarpSize, kSgFragDim and
kTGResidentBudgetBytes in agpu/include/agpu/core/Units.h, which owns them.
test_argbuf_abi.py reads that header and fails if these drift.

Separate module because compiler.py and driver.py both need them and neither
can import the other.
"""

WARP_SIZE = 32
SG_FRAG_DIM = 8
TG_BUDGET_BYTES = 32768
