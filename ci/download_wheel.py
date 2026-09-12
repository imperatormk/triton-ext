#!/usr/bin/env python3
"""
Download a wheel from a channel, one of the `PEP 503`_ indexes
:mod:`wheel_index` knows.

Only wheels the running interpreter and platform can install are considered,
and the newest of those is downloaded:

    pip install $(python ci/download_wheel.py release)

A pattern narrows that set; quote it against shell expansion:

    python ci/download_wheel.py release 'triton-3.8.*'

Usage:
    python ci/download_wheel.py <channel> ['wheel-pattern']

.. _PEP 503: https://peps.python.org/pep-0503/
"""

import doctest
import logging
import os
import sys

import common
import wheel_index

LOG = logging.getLogger(os.path.basename(__file__))
USAGE = wheel_index.usage("download_wheel.py")


def main(channel: str,
         pattern: str | None,
         dry_run: bool,
         usage: str = USAGE) -> None:
    if channel not in wheel_index.CHANNELS:
        LOG.error(f"Invalid channel: {channel}")
        print(usage, file=sys.stderr)
        sys.exit(1)

    candidates = wheel_index.run(channel, wheel_index.this_machine())
    if pattern:
        candidates = wheel_index.filter_wheels(candidates, pattern)
    if not candidates:
        LOG.error(f"No wheel for this machine matching: {pattern or 'any'}")
        sys.exit(1)
    wheel = max(candidates, key=lambda w: wheel_index.version_key(w.version()))
    if not dry_run:
        wheel_index.download(wheel, channel)
    print(wheel.filename)


if __name__ == "__main__":
    if common.env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    if common.env2bool("DOCTEST"):
        results = doctest.testmod()  # type: ignore[attr-defined]
        sys.exit(int(results.failed > 0))

    if len(sys.argv) < 2:
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
         common.env2bool("DRY_RUN"))
