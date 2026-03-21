# triton-apple: Apple GPU backend for Triton
# Discovered by Triton via entry_points (pyproject.toml)

# Patch libdevice on import (before user code caches references).
import sys as _sys
import importlib.abc as _importlib_abc
import importlib.util as _importlib_util


class _LibdevicePatchFinder(_importlib_abc.MetaPathFinder):

    def find_spec(self, fullname, path, target=None):
        if fullname != 'triton.language.extra.libdevice':
            return None
        _sys.meta_path.remove(self)
        spec = _importlib_util.find_spec(fullname)
        if spec is None:
            return None
        orig_loader = spec.loader

        class _PatchingLoader:
            @staticmethod
            def create_module(spec):
                return (orig_loader.create_module(spec)
                        if hasattr(orig_loader, 'create_module') else None)

            @staticmethod
            def exec_module(module):
                orig_loader.exec_module(module)
                import triton.language as tl
                from triton_apple_backend.libdevice_stubs import ALL_STUBS
                for name, fn in ALL_STUBS.items():
                    if hasattr(module, name):
                        setattr(module, name, fn)
                    if not hasattr(tl.math, name):
                        setattr(tl.math, name, fn)

        spec.loader = _PatchingLoader()
        return spec


_sys.meta_path.insert(0, _LibdevicePatchFinder())
