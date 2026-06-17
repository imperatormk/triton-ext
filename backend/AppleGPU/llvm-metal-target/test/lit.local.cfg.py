# mypy: disable-error-code="name-defined"
import os

# Locate metal-llc. Built top-level via add_subdirectory(llvm-metal-target), the
# binary lands at <triton-ext>/build/bin/metal-llc. Built standalone via
# cmake -S <this-dir>/.. -B <this-dir>/../build, it lands under
# <this-dir>/../build/bin/metal-llc. METAL_LLC_PATH overrides for ad-hoc dev.
metal_llc = os.environ.get("METAL_LLC_PATH")
if not metal_llc:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(config.triton_ext_binary_dir, "bin", "metal-llc"),
        os.path.abspath(os.path.join(here, "..", "build", "bin", "metal-llc")),
    ]
    for c in candidates:
        if os.path.exists(c):
            metal_llc = c
            break

if metal_llc and os.path.exists(metal_llc):
    config.substitutions.append(("%metal-llc", metal_llc))
else:
    config.unsupported = True
