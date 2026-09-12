#!/usr/bin/env python3
"""
List all available Triton wheel versions.

- **nightly** wheels come from the Triton's Azure `feed`_ published to by CI
  (default)
- **release** wheels come from `PyPI`_

This module exports the :func:`run` function, which fetches a list of wheels for
a given Triton channel (a `PEP 503`_ index read by :mod:`wheel_index`). When run
as a script it prints the retrieved wheel names to stdout.

Usage:
    python ci/list_triton_wheels.py [nightly|release] ['wheel-pattern']

.. _feed: https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/triton/
.. _PEP 503: https://peps.python.org/pep-0503/
.. _PyPI: https://pypi.org/simple/triton/
"""

import doctest
import logging
import os
import sys

import wheel_index

LOG = logging.getLogger(os.path.basename(__file__))
CHANNELS = wheel_index.CHANNELS
run = wheel_index.run
USAGE = ("Usage: python ci/list_triton_wheels.py [nightly|release] "
         "['wheel-pattern']")

if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    import common
    if common.env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    if common.env2bool("DOCTEST"):
        results = doctest.testmod()  # type: ignore[attr-defined]
        sys.exit(int(results.failed > 0))

    channel = sys.argv[1] if len(sys.argv) > 1 else "nightly"
    if channel not in CHANNELS:
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    pattern = sys.argv[2] if len(sys.argv) > 2 else None
    for wheel in run(channel, pattern):
        print(wheel)
