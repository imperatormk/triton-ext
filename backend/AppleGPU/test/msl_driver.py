"""Lower a TTGIR file to MSL and print it, for piping into FileCheck.

``add_emit_msl`` writes its output to a path and leaves the module alone, so
this cannot use ``testing.mlir_runner.run_passes`` -- that helper prints the
transformed module, which for this pass is unchanged.

Usage: msl_driver.py <input.mlir>
       msl_driver.py --check
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

from triton._C.libtriton import ir, passes

_SPLIT_RE = re.compile(r"^//\s*-----\s*$", re.MULTILINE)
_SPLIT_OUT = "\n// -----\n"
EXIT_NO_PLUGIN = 3


def _plugin():
    plugin = getattr(passes, "plugin", None)
    if plugin is None or not hasattr(plugin, "add_emit_msl"):
        print(
            "AppleGPU plugin not loaded: build the backend so the dylib "
            "sits beside triton_apple_backend.",
            file=sys.stderr)
        raise SystemExit(EXIT_NO_PLUGIN)
    return plugin


def _emit(mlir: str) -> str:
    plugin = _plugin()
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(mlir)
        src = f.name
    with tempfile.NamedTemporaryFile(suffix=".metal", delete=False) as f:
        out = f.name
    try:
        ctx = ir.context()
        ir.load_dialects(ctx)
        module = ir.parse_mlir_module(src, ctx)
        pm = ir.pass_manager(ctx)
        plugin.add_emit_msl(pm, [out])
        pm.run(module, "msl_driver")
        return Path(out).read_text()
    finally:
        Path(src).unlink(missing_ok=True)
        Path(out).unlink(missing_ok=True)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(f"usage: {Path(argv[0]).name} <input.mlir> | --check")
    if argv[1] == "--check":
        _plugin()
        return
    text = Path(argv[1]).read_text()
    chunks = [c for c in _SPLIT_RE.split(text) if c.strip()]
    sys.stdout.write(_SPLIT_OUT.join(_emit(c) for c in chunks))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv)
