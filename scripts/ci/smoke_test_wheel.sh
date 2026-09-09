#!/usr/bin/env bash
# Install the wheel a Release Prepare job just built and exercise it.
#
#   scripts/ci/smoke_test_wheel.sh <dist-dir> [container-image]
#
# An importable wheel is the minimum bar for shipping one, and a wheel that
# fails to load is a failure the Linux-only `python` CI job cannot see — it is
# the platform-specific builds that break this way.
#
# With a container image, the wheel is installed and run inside it instead of
# on the runner. That is what makes the musllinux jobs testable at all: those
# wheels are built in a musl cross container while the runner itself is glibc,
# so `pip install` on the host rejects the platform tag and the only way to
# execute the wheel is an Alpine container of the runner's own architecture.
# Without this the two musl platforms would ship with their smoke test quietly
# skipped — the exact hole this check exists to close.

set -euo pipefail

usage='usage: smoke_test_wheel.sh <dist-dir> [container-image]'

die() {
  printf '::error::wheel smoke test: %s\n' "$*" >&2
  exit 1
}

dist_dir="${1:?$usage}"
image="${2:-}"

test -d "$dist_dir" || die "missing dist directory: $dist_dir"

shopt -s nullglob
wheels=("$dist_dir"/*.whl)
shopt -u nullglob

test "${#wheels[@]}" -eq 1 ||
  die "expected one wheel, built ${#wheels[@]}: ${wheels[*]}"

# The abi3 assertion is the load-bearing half: a version-specific wheel still
# installs on the interpreter that built it, so without this check the smoke
# test passes while every other CPython version is left to compile the sdist.
# That is exactly what shipped before — one cp39 wheel from the manylinux
# container, one cp312 wheel from each other runner.
case "${wheels[0]}" in
  *-abi3-*) ;;
  *) die "not an abi3 wheel: ${wheels[0]##*/}" ;;
esac

# Written beside the dist directory rather than under /tmp: on Windows this
# runs in Git Bash, where a relative path survives the handoff to python.exe
# and an MSYS-style absolute path does not.
work="$(mktemp -d ./smoke.XXXXXX)"
trap 'rm -rf "$work"' EXIT

cat > "$work/smoke.py" <<'PY'
import splintr

tok = splintr.Tokenizer.from_pretrained("cl100k_base")
ids = tok.encode("Hello, world!")
assert tok.decode(ids) == "Hello, world!", "round-trip failed"

# `Tokenizer.pcre2` is public API, and it raises rather than switching backend
# when the optional feature was left out of the build. Asserting it here is
# what keeps the feature set identical across platforms: a wheel whose C
# dependency quietly failed to cross-compile would otherwise ship with a
# public method that works everywhere except one platform.
pcre2_ids = splintr.Tokenizer.from_pretrained("cl100k_base").pcre2(True).encode(
    "Hello, world!"
)
assert pcre2_ids == ids, f"pcre2 backend disagrees: {pcre2_ids} != {ids}"

print("wheel ok:", splintr.__version__, ids)
PY

if test -z "$image"; then
  python -m pip install --upgrade pip
  python -m pip install --no-index --find-links "$dist_dir" splintr-rs
  python "$work/smoke.py"
else
  dist_abs="$(cd "$dist_dir" && pwd)"
  work_abs="$(cd "$work" && pwd)"
  # `--platform` is deliberately absent: the container must be the runner's own
  # architecture, so that a wheel built for the wrong one fails here instead of
  # being emulated into passing.
  docker run --rm \
    -v "$dist_abs:/dist:ro" \
    -v "$work_abs:/smoke:ro" \
    "$image" \
    sh -c '
      set -eu
      python -m pip install --upgrade pip
      python -m pip install --no-index --find-links /dist splintr-rs
      python /smoke/smoke.py
    '
fi
