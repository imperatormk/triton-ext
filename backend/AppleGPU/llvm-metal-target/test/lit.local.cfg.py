# mypy: disable-error-code="name-defined"
import os

# Locate metal-llc. Built top-level via add_subdirectory(llvm-metal-target)
# from backend/AppleGPU/CMakeLists.txt, the binary lands under
# <triton-ext>/build/bin/metal-llc. METAL_LLC_PATH overrides for ad-hoc dev.
metal_llc = os.environ.get("METAL_LLC_PATH") or os.path.join(
    config.triton_ext_binary_dir, "bin", "metal-llc")

if os.path.exists(metal_llc):
    config.substitutions.append(("%metal-llc", metal_llc))
else:
    config.unsupported = True
