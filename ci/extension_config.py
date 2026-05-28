#!/usr/bin/env python3
"""Parse a per-extension `triton-ext.toml` manifest.

The one place that understands the manifest format: CMake reads the `--cmake`
output (a list of `set(KEY "value")` commands) via `include()`, other code
imports `load()`. Owners are not in the schema -- CODEOWNERS owns that (see
ci/extension_owners.py).

Schema (all keys optional except `name`):
    name     str        extension name (the built lib is lib<name>.so)
    status   str        "experimental" | "stable" (default "experimental")
    enabled  0 | 1       whether to build this extension (default 1)
    version  str         extension version, plumbed to C++ as TRITON_EXT_VERSION
                         (default "0.0.0")

Usage:
    python ci/extension_config.py <path-to-triton-ext.toml> [--cmake]
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

_VALID_STATUS = {"experimental", "stable"}


@dataclass(frozen=True)
class ExtensionConfig:
    name: str
    status: str = "experimental"
    enabled: bool = True
    version: str = "0.0.0"


def load(manifest_path: Path) -> ExtensionConfig:
    """Read and validate a triton-ext.toml manifest."""
    with open(manifest_path, "rb") as f:
        data = tomllib.load(f)

    name = data.get("name")
    if not name or not isinstance(name, str):
        raise ValueError(
            f"{manifest_path}: missing required string field 'name'")

    status = data.get("status", "experimental")
    if status not in _VALID_STATUS:
        raise ValueError(
            f"{manifest_path}: status '{status}' must be one of {sorted(_VALID_STATUS)}"
        )

    enabled = data.get("enabled", 1)
    if enabled not in (0, 1):
        raise ValueError(f"{manifest_path}: 'enabled' must be 0 or 1")

    version = data.get("version", "0.0.0")
    if not isinstance(version, str):
        raise ValueError(f"{manifest_path}: 'version' must be a string")

    return ExtensionConfig(name=name,
                           status=status,
                           enabled=bool(enabled),
                           version=version)


def _as_cmake(cfg: ExtensionConfig) -> str:
    """Emit `set(KEY "value")` commands for CMake to include()."""
    lines = [
        f'set(TRITON_EXT_NAME "{cfg.name}")',
        f'set(TRITON_EXT_STATUS "{cfg.status}")',
        f'set(TRITON_EXT_ENABLED "{"ON" if cfg.enabled else "OFF"}")',
        f'set(TRITON_EXT_VERSION "{cfg.version}")',
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="path to triton-ext.toml")
    parser.add_argument("--cmake",
                        action="store_true",
                        help="emit set(KEY \"value\") commands for CMake")
    args = parser.parse_args()

    try:
        cfg = load(args.manifest)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.cmake:
        print(_as_cmake(cfg))
    else:
        print(f"name={cfg.name} status={cfg.status} enabled={cfg.enabled} "
              f"version={cfg.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
