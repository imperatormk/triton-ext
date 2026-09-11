import sys

assert "triton_apple_backend" not in sys.modules
import triton  # noqa: E402,F401
from triton._C.libtriton import passes  # noqa: E402

for mod in ("triton_apple_backend", "triton_apple_backend.driver",
            "triton_apple_backend.compiler"):
    assert mod in sys.modules, f"{mod} was not imported by discovery"

assert hasattr(passes.plugin, "add_emit_msl"), "plugin pass missing"

import triton_apple_backend.driver as driver  # noqa: E402

assert hasattr(driver, "MetalDriver")
print("OK")
