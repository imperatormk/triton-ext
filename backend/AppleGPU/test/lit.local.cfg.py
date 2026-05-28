# mypy: disable-error-code="name-defined"
import os
import sys

config.environment["TRITON_PLUGIN_PATHS"] = os.path.join(
    config.triton_ext_binary_dir, "lib", "libapplegpu_backend.so")
print(
    f"ENV: "
    f"LD_LIBRARY_PATH={config.environment['LD_LIBRARY_PATH']} "
    f"TRITON_PLUGIN_PATHS={config.environment['TRITON_PLUGIN_PATHS']}",
    file=sys.stderr)
