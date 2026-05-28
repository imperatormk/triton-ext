# mypy: disable-error-code="name-defined"
import os

# Locate metal-llc. When built top-level via add_subdirectory(llvm-metal-target)
# from backend/AppleGPU/CMakeLists.txt, the binary lands under
# <triton-ext>/build/backend/AppleGPU/llvm-metal-target/bin/metal-llc.
# When built standalone via cmake -S <this-dir>/.. -B <this-dir>/../build, it
# lands under <this-dir>/../build/bin/metal-llc. We try both; METAL_LLC_PATH
# overrides for ad-hoc dev.
metal_llc = os.environ.get("METAL_LLC_PATH")
if not metal_llc:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # Nested build (default): triton-ext/build/bin/metal-llc
        os.path.join(config.triton_ext_binary_dir, "bin", "metal-llc"),
        # Standalone build: backend/AppleGPU/llvm-metal-target/build/bin/
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
