# mypy: disable-error-code="name-defined"
import os
import platform
import sysconfig

# These tests run the AppleGPU conversion pass inside `triton-opt`, a pure C++
# host (no Python interpreter). Loading the plugin there has two requirements
# that a normal `import triton` satisfies for free but `triton-opt` does not:
#
#   1. libtriton.so must be found on disk (the plugin lists it as NEEDED for
#      Triton's C++ dialect symbols).
#   2. libtriton.so has *undefined* CPython C-API symbols (e.g. _Py_NoneStruct)
#      — it is itself a Python extension module. Under `import triton` those
#      resolve against the running interpreter; under `triton-opt` there is no
#      interpreter, so we preload libpython to supply them. triton-opt never
#      calls the Python C-API (only C++ dialect registration runs), so an
#      un-initialised libpython is enough — the symbols just need to exist.
#
# If either can't be located the suite is marked unsupported rather than
# hard-failing, so the rest of the lit suite still runs.

_is_mac = platform.system() == "Darwin"
_dylib_ext = ".dylib" if _is_mac else ".so"

plugin = os.path.join(config.triton_ext_binary_dir, "lib",
                      f"libapplegpu_backend{_dylib_ext}")


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _find_libtriton_dir():
    # libtriton lives in Triton's `python/triton/_C` (source tree) or the
    # install tree's lib dir. Consult the values the build was configured with:
    # the CMake-baked config.* (reliable on CI, which uses TRITON_INSTALL_DIR)
    # and the corresponding env vars (reliable for local dev).
    bases = [
        getattr(config, "triton_source_dir", "") or "",
        getattr(config, "triton_install_dir", "") or "",
        os.environ.get("TRITON_SOURCE_DIR", ""),
        os.environ.get("TRITON_INSTALL_DIR", ""),
    ]
    repo = config.triton_ext_source_dir
    for base in bases:
        if not base:
            continue
        if not os.path.isabs(base):
            base = os.path.join(repo, base)
        for cand in (os.path.join(base, "python", "triton",
                                  "_C"), os.path.join(base, "lib")):
            if os.path.isfile(os.path.join(cand, "libtriton.so")):
                return cand
    return None


def _find_libpython():
    libdir = sysconfig.get_config_var("LIBDIR")
    candidates = []
    for var in ("LDLIBRARY", "INSTSONAME"):
        name = sysconfig.get_config_var(var)
        if name and libdir:
            candidates.append(os.path.join(libdir, name))
    if _is_mac:
        # Framework build: LIBDIR/../Python is the dylib.
        if libdir:
            candidates.append(
                os.path.abspath(os.path.join(libdir, "..", "Python")))
    return _first_existing(candidates)


libtriton_dir = _find_libtriton_dir()
libpython = _find_libpython()

if not os.path.isfile(plugin) or not libtriton_dir or not libpython:
    config.unsupported = True
else:
    config.environment["TRITON_PLUGIN_PATHS"] = plugin
    # Make libtriton findable by the dynamic loader.
    pathvar = "DYLD_LIBRARY_PATH" if _is_mac else "LD_LIBRARY_PATH"
    existing = config.environment.get(pathvar, "")
    config.environment[pathvar] = (f"{libtriton_dir}:{existing}"
                                   if existing else libtriton_dir)
    # Preload libpython so libtriton's CPython symbols resolve.
    preloadvar = "DYLD_INSERT_LIBRARIES" if _is_mac else "LD_PRELOAD"
    existing_pre = config.environment.get(preloadvar, "")
    config.environment[preloadvar] = (f"{libpython}:{existing_pre}"
                                      if existing_pre else libpython)
