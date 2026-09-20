"""Point Triton at the bundled uTLX plugin before anything imports Triton.

Run at interpreter startup from ``utlx_plugin.pth``, because a Triton older
than ``extend_with`` reads ``TRITON_PLUGIN_PATHS`` only while its own
``libtriton`` is imported -- by the time ``import utlx_plugin`` runs, it is
already too late to register the plugin's ops. Newer Tritons ignore the
variable and register explicitly instead (see ``utlx_plugin/_compat.py``).

Set ``UTLX_NO_AUTOREGISTER=1`` to skip, and import ``utlx_plugin`` before
``triton`` instead.

This runs in every interpreter that has uTLX installed, including ones that
never touch Triton, so it imports nothing but ``os``.
"""

import os

PLUGIN_PATHS_ENV = "TRITON_PLUGIN_PATHS"
OPT_OUT_ENV = "UTLX_NO_AUTOREGISTER"


def register():
    """Append the installed ``libutlx.so`` to ``TRITON_PLUGIN_PATHS``."""
    library = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "utlx_plugin", "libutlx.so")
    if not os.path.isfile(library):
        return

    paths = [
        p for p in os.environ.get(PLUGIN_PATHS_ENV, "").split(os.pathsep) if p
    ]
    if library not in paths:
        paths.append(library)
        os.environ[PLUGIN_PATHS_ENV] = os.pathsep.join(paths)


if not os.environ.get(OPT_OUT_ENV):
    register()
