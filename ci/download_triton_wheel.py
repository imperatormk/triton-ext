#!/usr/bin/env python3
"""
Download a pre-built Triton wheel to the current directory.

Only wheels this interpreter and platform can install are considered, and the
newest of those is downloaded. The wheel is selected based on:

- `channel`: "nightly" or "release" (default: nightly)
- `wheel-pattern`: narrows that set, with glob wildcards allowed (default: on
  nightly, the Triton commit `ci/triton-hash.txt` pins)

Wheel filenames are long, so a pattern is usually a fragment:

- `triton-*+git0d7dc8626*`: a nightly built from that commit
- `triton-3.8.0-*`: pins release 3.8.0

Usage (NOTE: quote the pattern to avoid shell expansion):
    python ci/download_triton_wheel.py [channel] ['wheel-pattern']
"""

import doctest
import logging
import os
import sys

import common
import download_wheel

USAGE = "Usage: python ci/download_triton_wheel.py [channel] ['wheel-pattern']"
LOG = logging.getLogger(os.path.basename(__file__))


def read_triton_hash():
    """Read the pinned Triton commit hash from ci/triton-hash.txt."""
    dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dir, "triton-hash.txt")
    return open(file).read().strip()


def main(
    channel: str,
    pattern: str | None,
    dry_run: bool,
):
    """
    Download a Triton wheel from the given `channel`, optionally matching a
    wheel `pattern`.
    """
    # A nightly is published per commit, so pin the one this repo tracks.
    if channel == "nightly" and pattern is None:
        pattern = f"triton-*+git{read_triton_hash()[:8]}-*"

    download_wheel.main(channel, pattern, dry_run, USAGE)


if __name__ == "__main__":
    if common.env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    if common.env2bool("DOCTEST"):
        results = doctest.testmod()  # type: ignore[attr-defined]
        sys.exit(int(results.failed > 0))

    dry_run = common.env2bool("DRY_RUN")
    channel = sys.argv[1] if len(sys.argv) > 1 else "nightly"
    wheel = sys.argv[2] if len(sys.argv) > 2 else None
    main(channel, wheel, dry_run)
