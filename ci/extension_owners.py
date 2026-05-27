#!/usr/bin/env python3
"""Resolve the owners of a path from `.github/CODEOWNERS`.

Implements the subset of CODEOWNERS matching we rely on (last-matching-pattern
wins, leading-slash anchoring, trailing-slash directory patterns); not a full
glob engine.

Usage:
    python ci/extension_owners.py <path> [--codeowners <file>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CODEOWNERS = REPO_ROOT / ".github" / "CODEOWNERS"


def _parse(codeowners_path: Path) -> list[tuple[str, list[str]]]:
    rules: list[tuple[str, list[str]]] = []
    for raw in codeowners_path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        pattern, owners = parts[0], parts[1:]
        rules.append((pattern, owners))
    return rules


def _matches(pattern: str, path: str) -> bool:
    """Match a CODEOWNERS pattern against a repo-relative POSIX path."""
    if pattern == "*":
        return True
    anchored = pattern.startswith("/")
    pat = pattern.lstrip("/")
    # A trailing slash matches a directory and everything beneath it.
    if pat.endswith("/"):
        pat = pat.rstrip("/")
        return path == pat or path.startswith(pat + "/")
    if anchored:
        return path == pat or path.startswith(pat + "/")
    # Unanchored: match the segment anywhere in the path.
    segments = path.split("/")
    return pat in segments or path == pat or path.startswith(pat + "/")


def owners_for(path: str,
               codeowners_path: Path = DEFAULT_CODEOWNERS) -> list[str]:
    """Return the owners for a repo-relative path (last matching rule wins)."""
    path = path.strip("/")
    result: list[str] = []
    for pattern, owners in _parse(codeowners_path):
        if _matches(pattern, path):
            result = owners
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="repo-relative path to resolve")
    parser.add_argument("--codeowners", type=Path, default=DEFAULT_CODEOWNERS)
    args = parser.parse_args()

    owners = owners_for(args.path, args.codeowners)
    if not owners:
        print(f"no owners found for {args.path}", file=sys.stderr)
        return 1
    print(" ".join(owners))
    return 0


if __name__ == "__main__":
    sys.exit(main())
