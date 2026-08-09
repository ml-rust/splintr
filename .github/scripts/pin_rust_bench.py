#!/usr/bin/env python3
"""Point `.github/rust-bench` at a published splintr instead of this checkout.

# Why

`rust-bench` depends on splintr by path, which is what lets the perf workflow
measure a branch before it ships. But the `splintr` input also accepts a
released version, and then the path dependency is wrong: the Python suite would
install that release from PyPI while the Rust suite went on building HEAD, and
the run summary would carry two reports of two different builds under one
heading.

This rewrites the dependency so both halves measure the same thing. It runs in
CI against a checkout that is thrown away afterwards — it is not a way to edit
the manifest in a working tree.

# Usage

    python .github/scripts/pin_rust_bench.py 0.16.1   # exact release
    python .github/scripts/pin_rust_bench.py latest   # whatever cargo resolves

`local` is not a valid argument: keeping the path dependency means not running
this at all.
"""

import re
import sys
from pathlib import Path

MANIFEST = Path(".github/rust-bench/Cargo.toml")

# The committed line, matched exactly. A loose pattern here would silently do
# nothing after an unrelated edit to the manifest, and the run would report a
# pinned version while measuring HEAD — the failure this script exists to
# prevent, made invisible.
PATH_DEP = re.compile(r'^splintr = \{ path = "\.\./\.\." \}$', re.M)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <version|latest>")
    requested = argv[1]
    if requested == "local":
        raise SystemExit("'local' keeps the path dependency; do not run this")

    # `*` rather than a resolved number: cargo picks the newest release, which
    # is the same rule `pip install splintr-rs` follows for `latest`.
    req = "*" if requested == "latest" else f"={requested}"

    text = MANIFEST.read_text()
    patched, count = PATH_DEP.subn(f'splintr = "{req}"', text, count=1)
    if count != 1:
        raise SystemExit(
            f"{MANIFEST}: expected exactly one path dependency on splintr, "
            f"replaced {count}. The manifest changed shape — update PATH_DEP."
        )
    MANIFEST.write_text(patched)
    print(f'splintr = "{req}"')
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
