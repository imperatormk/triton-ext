#!/usr/bin/env python3
"""Report every extension's build status and owners.

Reads each `triton-ext.toml` manifest and resolves owners from CODEOWNERS.

Usage:
    python ci/check_extensions.py            # human-readable table
    python ci/check_extensions.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import extension_config
import extension_owners

REPO_ROOT = Path(__file__).resolve().parent.parent
# Directories that hold downloaded artifacts / build trees, not extensions.
_SKIP_TOP = ("build", "ci")


def discover() -> list[dict]:
    extensions: list[dict] = []
    for manifest in sorted(REPO_ROOT.rglob("triton-ext.toml")):
        rel = manifest.relative_to(REPO_ROOT)
        if rel.parts[0].startswith(
            ("triton-", "llvm-")) or rel.parts[0] in _SKIP_TOP:
            continue
        cfg = extension_config.load(manifest)
        ext_dir = manifest.parent.relative_to(REPO_ROOT).as_posix()
        extensions.append({
            "name": cfg.name,
            "dir": ext_dir,
            "status": cfg.status,
            "enabled": cfg.enabled,
            "owners": extension_owners.owners_for(ext_dir),
        })
    return extensions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    extensions = discover()
    if args.json:
        print(json.dumps(extensions, indent=2))
        return 0

    for ext in extensions:
        flag = "enabled" if ext["enabled"] else "DISABLED"
        owners = " ".join(ext["owners"]) or "(no owners)"
        print(f"{ext['name']:<14} {flag:<9} {ext['dir']:<24} owners: {owners}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
