#!/usr/bin/env python3
"""
Read wheels from a `PEP 503`_ simple index.

A *channel* names an index to read from.

Usage:
    python ci/wheel_index.py <channel> ['wheel-pattern']

.. _PEP 503: https://peps.python.org/pep-0503/
"""

import doctest
import logging
import os
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from html.parser import HTMLParser

import requests

import common
import probe_sysinfo

LOG = logging.getLogger(os.path.basename(__file__))

#: Requests that do not look like pip are asked to authenticate by the Azure
#: DevOps feed, which then serves a partial index.
PIP_HEADERS = {"User-Agent": "pip/24.0"}


@dataclass
class Channel:
    url: str
    as_pip: bool = False

    def headers(self) -> dict[str, str]:
        return PIP_HEADERS if self.as_pip else {}


NIGHTLY_URL = ("https://aiinfra.pkgs.visualstudio.com/PublicPackages"
               "/_packaging/Triton-Nightly/pypi/simple/triton/")
RELEASE_URL = "https://pypi.org/simple/triton/"

CHANNELS = {
    "nightly": Channel(NIGHTLY_URL, as_pip=True),
    "release": Channel(RELEASE_URL),
}


@dataclass
class Wheel:
    """A wheel retrieved from a PEP 503 index."""
    filename: str
    url: str
    sha256: str | None

    def __init__(self, filename: str, url: str):
        self.filename = filename
        if "#sha256=" in url:
            self.sha256 = url.split("#sha256=", 1)[1]
            self.url = url.split("#", 1)[0]
        else:
            self.sha256 = None
            self.url = url

    def __str__(self):
        return f"{self.filename}"

    def __repr__(self):
        return f"{self.filename}"

    def version(self) -> str:
        """
        Return the base version string from a wheel filename.

        >>> Wheel("triton-3.8.0-cp314-cp314-linux_x86_64.whl", "https://...").version()
        '3.8.0'
        >>> Wheel("triton-3.8.0+gitf6ef5434-cp314-cp314-linux_x86_64.whl", "https://...").version()
        '3.8.0+gitf6ef5434'
        >>> Wheel("triton-3.7.1+gitf6ef5434-cp314-cp314-linux_x86_64.whl", "https://...").version()
        '3.7.1+gitf6ef5434'
        """
        return self.filename.split("-")[1]


class _WheelIndexParser(HTMLParser):
    """Parse a PEP 503 simple index; see `parse_index`."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[Wheel] = []
        self._href: str | None = None
        self._in_anchor = False

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        self._in_anchor = True
        self._href = dict(attrs).get("href", "")

    def handle_data(self, data: str) -> None:
        if self._in_anchor and self._href and data.strip().endswith(".whl"):
            self.results.append(Wheel(data.strip(), self._href))

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._in_anchor = False
            self._href = None


def parse_index(html: str) -> list[Wheel]:
    parser = _WheelIndexParser()
    parser.feed(html)
    return parser.results


def fetch_index(channel: str) -> str:
    entry = CHANNELS[channel]
    LOG.debug(f"Fetching index: {entry.url}")
    response = requests.get(entry.url, timeout=60, headers=entry.headers())
    response.raise_for_status()
    return response.text


def filter_wheels(candidates: list[Wheel], pattern: str) -> list[Wheel]:
    """
    Return the subset of `candidates` whose filenames match `pattern`.

    >>> filter_wheels([], "triton-*")
    []
    >>> filter_wheels([Wheel("triton-3.8.0-cp314-...whl", "https://...")], "triton-*")[0].version()
    '3.8.0'
    >>> wheels = [
    ...   Wheel("triton-3.8.0+gitf6ef5434-...x86_64.whl", "https://..."),
    ...   Wheel("triton-3.8.0+gitf6ef5434-...aarch64.whl", "https://..."),
    ... ]
    >>> filter_wheels(wheels, "triton-*f6ef5434*aarch64*")[0].filename
    'triton-3.8.0+gitf6ef5434-...aarch64.whl'
    """
    return [w for w in candidates if fnmatch(w.filename, pattern)]


def run(channel: str, pattern: str | None = None) -> list[Wheel]:
    """Return the wheels on `channel`, optionally filtered by `pattern`."""
    wheels = parse_index(fetch_index(channel))
    LOG.debug(f"Parsed {len(wheels)} wheel anchors from {channel} index")
    if pattern:
        wheels = filter_wheels(wheels, pattern)
        LOG.debug(f"Filtered to {len(wheels)} wheels matching: {pattern}")
    return wheels


#: A wheel's platform tag per system, as a glob over `{arch}`, keyed on the
#: names `probe_sysinfo` reports. Linux wheels spell the arch `x86_64` and
#: `aarch64`, macOS spells it `arm64`.
PLATFORM_TAGS = {
    "linux": ("*linux*_{arch}*", {
        "x64": "x86_64",
        "arm64": "aarch64"
    }),
    "macos": ("*macosx*_{arch}", {
        "x64": "x86_64",
        "arm64": "arm64"
    }),
}


def platform_pattern(system: str, arch: str) -> str:
    """
    Return a glob matching the wheel platform tag for `system` and `arch`.

    >>> platform_pattern("linux", "x64")
    '*linux*_x86_64*'
    >>> platform_pattern("linux", "arm64")
    '*linux*_aarch64*'
    >>> platform_pattern("macos", "arm64")
    '*macosx*_arm64'
    """
    if system not in PLATFORM_TAGS:
        LOG.error(f"Unrecognised system: {system!r}; "
                  f"expected one of {sorted(PLATFORM_TAGS)}")
        sys.exit(1)
    tag, arches = PLATFORM_TAGS[system]
    if arch not in arches:
        LOG.error(f"Unrecognised arch: {arch!r}; "
                  f"expected one of {sorted(arches)}")
        sys.exit(1)
    return tag.format(arch=arches[arch])


def python_tag() -> str:
    """Return the CPython tag for the running interpreter, e.g. `cp311`."""
    major, minor = sys.version_info[:2]
    return f"cp{major}{minor}"


def this_machine() -> str:
    """Return a glob matching any wheel this interpreter can install."""
    tag = python_tag()
    return f"*-{tag}-{tag}-{platform_pattern(*probe_sysinfo.run())}.whl"


def download(wheel: Wheel, channel: str) -> None:
    """Download `wheel` into the current directory and check its digest."""
    common.download_file(wheel.url,
                         wheel.filename,
                         headers=CHANNELS[channel].headers())
    if wheel.sha256:
        common.verify_checksum(wheel.filename, wheel.sha256)


def version_key(version: str) -> tuple:
    """
    Return a sort key ordering release versions before their local variants.

    >>> version_key("2.9.0") < version_key("2.10.0")
    True
    >>> version_key("2.14.0") < version_key("2.14.0+git1234abcd")
    True
    """
    release, _, local = version.partition("+")
    numbers = tuple(int(p) if p.isdigit() else 0 for p in release.split("."))
    return (numbers, local)


def usage(script: str) -> str:
    return (f"Usage: python ci/{script} [{'|'.join(CHANNELS)}] "
            "['wheel-pattern']")


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    if common.env2bool("VERBOSE"):
        logging.basicConfig(level=logging.DEBUG)

    if common.env2bool("DOCTEST"):
        results = doctest.testmod()  # type: ignore[attr-defined]
        sys.exit(int(results.failed > 0))

    channel = sys.argv[1] if len(sys.argv) > 1 else "nightly"
    if channel not in CHANNELS:
        print(usage("wheel_index.py"), file=sys.stderr)
        sys.exit(1)

    pattern = sys.argv[2] if len(sys.argv) > 2 else None
    for wheel in run(channel, pattern):
        print(wheel)
