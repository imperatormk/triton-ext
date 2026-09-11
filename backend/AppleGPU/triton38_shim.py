"""Workarounds that let the Apple GPU backend run on stock Triton + torch.

Import this before ``triton``. Nothing here is specific to Apple GPU codegen;
it is all glue around two upstream gaps, kept out of the example scripts so
those can stay byte-identical to Triton's own tutorials.

The two gaps have very different lifetimes, so they are documented apart:
part 1 disappears when Triton is bumped, part 2 does not.


Part 1 - Triton 3.8.x has no plugin ``extend_with`` (always applied)
--------------------------------------------------------------------
Both gaps are in Triton's ``python/src/passes.cc``:

1. ``passes.plugin.extend_with(path)`` does not exist yet. 3.8.x loads plugins
   eagerly from ``TRITON_PLUGIN_PATHS`` at libtriton import instead, so the env
   var is set here and ``extend_with`` is stubbed to a no-op --
   ``triton_apple_backend/__init__.py`` calls it unconditionally, and the
   resulting ``AttributeError`` otherwise takes ``import triton`` down with it.
2. 3.8.x binds each plugin pass under its bare name (``emit_msl``), where
   ``extend_with`` binds it as ``add_<name>``. ``compiler.py`` calls the
   latter, so it is aliased.

Both patches must land between libtriton's C extension loading and
``triton.backends`` importing the backend -- both of which happen inside
``import triton`` -- hence a meta-path hook rather than a plain attribute
assignment. Every patch is guarded by ``hasattr``, so this is inert on a Triton
new enough not to need it. It all goes away once Triton is at or past the
commit ``ci/triton-hash.txt`` pins.


Part 2 - ``do_bench`` cannot time small kernels on MPS (on by default)
------------------------------------------------------------------------
``triton.testing.do_bench`` is replaced with a wall-clock implementation.
Set ``MSL_WALLCLOCK_BENCH=0`` to keep Triton's own. With Triton's own,
benchmarking a small kernel on MPS dies with::

    RuntimeError: End event 1 was not recorded after start event 3

``torch.mps.Event.elapsed_time()`` raises whenever no GPU work is still in
flight when the end event is recorded. Reproduced on torch 2.14 with **no
Triton imported at all**: an empty body fails, one small ``x + y`` fails, and
100 adds or an already-busy queue succeed. For a fast kernel the CPU reaches
``end_event.record()`` after the GPU has drained, so MPS never records the end
event and ``_mps_elapsedTimeOfEvents`` reports it as unrecorded. The event ids
in the message are not even ordered -- the end event simply never got a record.

That is exactly the shape ``do_bench`` uses (``testing.py``, ``start.record();
fn(); end.record()``), so it is fragile on MPS by construction. It works on
CUDA because ``torch.cuda.Event`` records into the stream whether or not work
is pending.

This is a torch/MPS limitation, not a Triton or Apple-backend one, so it will
not age out with a Triton bump -- hence a switch separate from part 1, which
does. It is on by default because the failure is not a corner case: on MPS
every kernel short enough to matter hits it, so leaving Triton's version in
place just means a guaranteed crash the moment anyone benchmarks.

The replacement brackets each call with ``torch.mps.synchronize()`` and reads
``time.perf_counter()``. That measures CPU-observed round trip, not pure GPU
time, so it overstates cost at small sizes where launch overhead dominates.
It is honest about wall time and it always produces a number.
"""

import importlib.abc
import importlib.machinery
import os
import sys

#: The plugin to load. Defaults to this checkout's build output; override to
#: point at an installed copy.
DYLIB = os.environ.get(
    "APPLEGPU_DYLIB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "lib",
                 "libapplegpu_backend.dylib"))


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Runs `patch(module)` right after `target` finishes executing."""

    def __init__(self, target, patch):
        self._target = target
        self._patch = patch
        self._busy = False

    def find_spec(self, name, path=None, target=None):
        if name != self._target or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.machinery.PathFinder.find_spec(name, path)
        finally:
            self._busy = False
        if spec is None:
            return None

        exec_module = spec.loader.exec_module
        patch = self._patch

        def patched(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = patched
        return spec


def _patch_plugin_api(libtriton):
    plugin = libtriton.passes.plugin
    if not hasattr(plugin, "extend_with"):
        plugin.extend_with = lambda _path: None
    if hasattr(plugin, "emit_msl") and not hasattr(plugin, "add_emit_msl"):
        plugin.add_emit_msl = plugin.emit_msl


def _patch_do_bench(testing):
    import time

    import torch

    def do_bench_wall(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None,
                      return_mode="mean"):
        """Wall-clock stand-in for do_bench; see part 2 of the module docstring."""
        fn()
        torch.mps.synchronize()

        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        estimate_ms = (time.perf_counter() - t0) * 1e3 or 1e-3

        for _ in range(max(1, int(warmup / estimate_ms))):
            fn()
        torch.mps.synchronize()

        times = []
        for _ in range(max(1, int(rep / estimate_ms))):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            t0 = time.perf_counter()
            fn()
            torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)

        # A plain list, not a tensor: _quantile only needs len() and sorted(),
        # and a tensor would print as `tensor(0.28)` in perf_report tables.
        return testing._summarize_statistics(times, quantiles, return_mode)

    testing.do_bench = do_bench_wall


if "triton" not in sys.modules:
    os.environ.setdefault("TRITON_PLUGIN_PATHS", DYLIB)
    sys.meta_path.insert(0, _PatchOnImport("triton._C.libtriton",
                                           _patch_plugin_api))
    if os.environ.get("MSL_WALLCLOCK_BENCH", "1").lower() not in ("0", "false",
                                                                  "off", "no",
                                                                  "n"):
        sys.meta_path.insert(0, _PatchOnImport("triton.testing",
                                               _patch_do_bench))
